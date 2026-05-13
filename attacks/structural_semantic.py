"""
结构化 mislead 攻击模块（structural_semantic.py）
===============================================
本文件只保存 AoI-only、H-only 和 Joint 三类结构化误导攻击。公共 risk、H 映射、
链路 priority 和 H 下标转换逻辑统一复用 structural_common.py，避免重复维护。
攻击函数只返回 attacked_state 副本，不写回 env.aoi、env.channel_state 或 reward。
"""

from __future__ import annotations

from math import ceil

import numpy as np

from attacks.random_semantic import _build_aoi_info, _build_h_info, _can_attack, _empty_perturb_info, _get_config_value, _merge_info
from attacks.state_ops import merge_state, project_aoi_part, project_h_part, split_state
from attacks.structural_common import (
    compute_link_priority,
    compute_sensor_mse_values,
    compute_sensor_risk,
    flatten_h_index,
    h_to_success_prob,
    unflatten_h_index,
)


def _split_suppress_attract_budget(total_budget: int) -> tuple[int, int]:
    """
    作用:
        将结构化攻击预算拆成 suppression 和 attraction 两部分。
    参数:
        total_budget: 当前 AoI 或 H 分段可用扰动维度数。
    输入输出:
        输入整数预算；输出 (suppression_budget, attraction_budget)。
    核心步骤:
        1. B<=0 时返回 0。
        2. B=1 时只做 suppression。
        3. B>=2 时至少保留 1 个 attraction，其余给 suppression。
    """
    # 将预算转为非负整数。
    budget = max(0, int(total_budget))
    # 没有预算时两类都为 0。
    if budget <= 0:
        return 0, 0
    # 单维预算无法同时隐藏和诱饵，因此只做 suppression。
    if budget == 1:
        return 1, 0
    # B>=2 时至少保留一个诱饵维度。
    attraction_budget = max(1, int(budget - ceil(0.6 * budget)))
    # 剩余预算用于隐藏高风险对象。
    suppression_budget = budget - attraction_budget
    # 返回预算拆分。
    return suppression_budget, attraction_budget


def _select_aoi_indices(aoi_part: np.ndarray, risk: np.ndarray, env, budget: int) -> tuple[list[int], list[int]]:
    """
    作用:
        为 AoI mislead 选择需要降低和抬高的 sensor 下标。
    参数:
        aoi_part: AoI 状态段，shape=(N,)。
        risk: sensor 风险数组，shape=(N,)。
        env: 环境实例，需要 max_aoi。
        budget: AoI 可用扰动维度数。
    输入输出:
        输入 AoI/risk/预算；输出 (suppress_indices, attract_indices)。
    核心步骤:
        1. 高风险且 AoI>1 的 sensor 优先降低。
        2. 低风险且 AoI<max_aoi 的 sensor 优先抬高。
        3. 因 clip 无法变化的预算继续补齐。
    """
    # AoI 预算不能超过 sensor 数。
    budget = min(max(0, int(budget)), int(env.n))
    # 拆成 suppression / attraction 预算。
    suppress_budget, attract_budget = _split_suppress_attract_budget(budget)
    # 标准化 AoI 数组，只读使用。
    aoi_arr = np.asarray(aoi_part, dtype=np.float32).reshape(-1)
    # 高风险 sensor 降序。
    high_order = np.argsort(-np.asarray(risk, dtype=np.float64), kind="mergesort").astype(int).tolist()
    # 低风险 sensor 升序。
    low_order = np.argsort(np.asarray(risk, dtype=np.float64), kind="mergesort").astype(int).tolist()
    # 记录降低 AoI 的 sensor。
    suppress_indices: list[int] = []
    # 记录已使用 sensor，避免同一维同时升降。
    used: set[int] = set()
    # 按高风险顺序选择可降低 sensor。
    for idx in high_order:
        # suppression 预算用完则停止。
        if len(suppress_indices) >= suppress_budget:
            break
        # AoI=1 再降低会被 clip，因此跳过。
        if float(aoi_arr[idx]) > 1.0:
            suppress_indices.append(int(idx))
            used.add(int(idx))
    # 未用完的 suppression 预算转给 attraction。
    target_attract_budget = attract_budget + (suppress_budget - len(suppress_indices))
    # 记录抬高 AoI 的 sensor。
    attract_indices: list[int] = []
    # 按低风险顺序选择可抬高 sensor。
    for idx in low_order:
        # attraction 预算用完则停止。
        if len(attract_indices) >= target_attract_budget:
            break
        # 已用或已到 max_aoi 的 sensor 跳过。
        if int(idx) not in used and float(aoi_arr[idx]) < float(env.max_aoi):
            attract_indices.append(int(idx))
            used.add(int(idx))
    # 若仍有剩余预算，再回到高风险列表补充可降低 sensor。
    remaining_budget = budget - len(suppress_indices) - len(attract_indices)
    # 补齐剩余预算。
    for idx in high_order:
        # 预算用完则停止。
        if remaining_budget <= 0:
            break
        # 未使用且可降低才加入。
        if int(idx) not in used and float(aoi_arr[idx]) > 1.0:
            suppress_indices.append(int(idx))
            used.add(int(idx))
            remaining_budget -= 1
    # 返回两组下标。
    return suppress_indices, attract_indices


def _select_h_indices(h_part: np.ndarray, priority: np.ndarray, risk: np.ndarray, env, budget: int) -> tuple[list[int], list[int]]:
    """
    作用:
        为 H mislead 选择需要降低和抬高的链路展平下标。
    参数:
        h_part: H 状态段，shape=(N*M,)。
        priority: 链路优先级矩阵，shape=(N,M)。
        risk: sensor 风险数组，shape=(N,)。
        env: 环境实例，需要 n 和 m。
        budget: H 可用扰动维度数。
    输入输出:
        输入 H/priority/risk/预算；输出 (suppressed, attracted)。
    核心步骤:
        1. 高 priority 且 H>0 的链路优先降低。
        2. 低风险且 H<4 的链路优先抬高为诱饵。
        3. 因 clip 无法变化的预算继续补齐。
    """
    # 标准化 H 展平段。
    h_arr = np.asarray(h_part, dtype=np.float32).reshape(-1)
    # H 预算不能超过 H 维度数。
    budget = min(max(0, int(budget)), h_arr.size)
    # 拆成 suppression / attraction 预算。
    suppress_budget, attract_budget = _split_suppress_attract_budget(budget)
    # 展平 priority，顺序与 H state 一致。
    priority_flat = np.asarray(priority, dtype=np.float64).reshape(-1)
    # 高 priority 链路降序。
    high_order = np.argsort(-priority_flat, kind="mergesort").astype(int).tolist()
    # 记录降低 H 的链路。
    suppressed: list[int] = []
    # 记录已使用链路。
    used: set[int] = set()
    # 按 priority 选择可降低链路。
    for flat_idx in high_order:
        # suppression 预算用完则停止。
        if len(suppressed) >= suppress_budget:
            break
        # H=0 再降低会被 clip，因此跳过。
        if float(h_arr[flat_idx]) > 0.0:
            suppressed.append(int(flat_idx))
            used.add(int(flat_idx))
    # H 恢复为矩阵，用于枚举诱饵链路。
    h_matrix = h_arr.reshape(int(env.n), int(env.m))
    # 诱饵候选格式为 (risk, H, flat_idx)。
    decoys: list[tuple[float, float, int]] = []
    # 枚举所有 sensor-channel。
    for sensor_idx in range(int(env.n)):
        for channel_idx in range(int(env.m)):
            # 计算展平下标。
            flat_idx = flatten_h_index(sensor_idx, channel_idx, env)
            # 已用或 H=4 时无法抬高，跳过。
            if flat_idx in used or float(h_matrix[sensor_idx, channel_idx]) >= 4.0:
                continue
            # 低风险优先，同风险下低 H 优先。
            decoys.append((float(risk[sensor_idx]), float(h_matrix[sensor_idx, channel_idx]), int(flat_idx)))
    # 稳定排序诱饵候选。
    decoys.sort()
    # 未用完的 suppression 预算转给 attraction。
    target_attract_budget = attract_budget + (suppress_budget - len(suppressed))
    # 记录抬高 H 的链路。
    attracted: list[int] = []
    # 选择诱饵链路。
    for _, _, flat_idx in decoys:
        # attraction 预算用完则停止。
        if len(attracted) >= target_attract_budget:
            break
        attracted.append(int(flat_idx))
        used.add(int(flat_idx))
    # 若仍有剩余预算，再回到高 priority 链路补齐。
    remaining_budget = budget - len(suppressed) - len(attracted)
    # 补齐剩余预算。
    for flat_idx in high_order:
        # 预算用完则停止。
        if remaining_budget <= 0:
            break
        # 未使用且可降低才加入。
        if int(flat_idx) not in used and float(h_arr[flat_idx]) > 0.0:
            suppressed.append(int(flat_idx))
            used.add(int(flat_idx))
            remaining_budget -= 1
    # 返回两组 H 展平下标。
    return suppressed, attracted


def _apply_aoi_mislead(aoi_part: np.ndarray, env, config, budget: int) -> tuple[np.ndarray, dict]:
    """
    作用:
        生成 AoI 结构化误导扰动。
    参数:
        aoi_part: 原始 AoI 状态段。
        env: 环境实例。
        config: 攻击配置，提供 aoi_delta。
        budget: AoI 可用扰动维度数。
    输入输出:
        输入原始 AoI；输出 (attacked_aoi, aoi_info)。
    核心步骤:
        1. 计算 reset_gain risk。
        2. 高风险 sensor AoI 降低，低风险 sensor AoI 抬高。
        3. 投影并统计实际变化。
    """
    # 复制 AoI，避免原地修改。
    original_aoi = np.asarray(aoi_part, dtype=np.float32).reshape(-1).copy()
    # attacked_aoi 是观测副本。
    attacked_aoi = original_aoi.copy()
    # 计算 sensor risk。
    risk = compute_sensor_risk(original_aoi, env, risk_mode="reset_gain")
    # 选择降低/抬高对象。
    suppress_indices, attract_indices = _select_aoi_indices(original_aoi, risk, env, budget)
    # 读取 AoI 扰动幅值。
    delta = float(int(_get_config_value(config, "aoi_delta", 1)))
    # 降低高风险 AoI。
    for idx in suppress_indices:
        attacked_aoi[int(idx)] -= delta
    # 抬高低风险诱饵 AoI。
    for idx in attract_indices:
        attacked_aoi[int(idx)] += delta
    # 投影到合法范围。
    attacked_aoi = project_aoi_part(attacked_aoi, env)
    # 统计实际扰动。
    info = _build_aoi_info(attacked_aoi, original_aoi)
    # 记录解释字段。
    info["high_risk_sensors"] = [int(x) for x in suppress_indices]
    info["low_risk_sensors"] = [int(x) for x in attract_indices]
    # 返回副本和统计。
    return attacked_aoi.astype(np.float32, copy=True), info


def _apply_h_mislead(h_part: np.ndarray, aoi_part: np.ndarray, env, config, budget: int) -> tuple[np.ndarray, dict]:
    """
    作用:
        生成 H 结构化误导扰动。
    参数:
        h_part: 原始 H 状态段。
        aoi_part: 原始 AoI 状态段，用于计算 sensor risk。
        env: 环境实例。
        config: 攻击配置，提供 h_delta。
        budget: H 可用扰动维度数。
    输入输出:
        输入原始 H/AoI；输出 (attacked_h, h_info)。
    核心步骤:
        1. 计算 priority = risk * success_prob。
        2. 高 priority 链路 H 降低，低风险诱饵链路 H 抬高。
        3. 投影并统计实际变化。
    """
    # 复制 H，避免原地修改。
    original_h = np.asarray(h_part, dtype=np.float32).reshape(-1).copy()
    # attacked_h 是观测副本。
    attacked_h = original_h.copy()
    # 计算 priority 和 risk。
    priority, risk, _ = compute_link_priority(aoi_part, original_h, env, risk_mode="reset_gain")
    # 选择降低/抬高链路。
    suppressed, attracted = _select_h_indices(original_h, priority, risk, env, budget)
    # 读取 H 扰动幅值。
    delta = float(int(_get_config_value(config, "h_delta", 1)))
    # 降低高 priority 链路 H。
    for flat_idx in suppressed:
        attacked_h[int(flat_idx)] -= delta
    # 抬高低风险诱饵链路 H。
    for flat_idx in attracted:
        attacked_h[int(flat_idx)] += delta
    # 投影到合法范围。
    attacked_h = project_h_part(attacked_h)
    # 统计实际扰动。
    info = _build_h_info(attacked_h, original_h)
    # 记录展平下标解释字段。
    info["suppressed_h_indices"] = [int(x) for x in suppressed]
    info["attracted_h_indices"] = [int(x) for x in attracted]
    # 记录二维链路解释字段。
    info["high_priority_links"] = [tuple(int(v) for v in unflatten_h_index(x, env)) for x in suppressed]
    info["decoy_links"] = [tuple(int(v) for v in unflatten_h_index(x, env)) for x in attracted]
    # 返回副本和统计。
    return attacked_h.astype(np.float32, copy=True), info


def _finalize(attacked_aoi: np.ndarray, attacked_h: np.ndarray, original_aoi: np.ndarray, original_h: np.ndarray, attack_state: dict, attack_type: str, extra: dict):
    """
    作用:
        合并 attacked_state、扰动统计和 attack_state 计数。
    参数:
        attacked_aoi: 被攻击后的 AoI 观测。
        attacked_h: 被攻击后的 H 观测。
        original_aoi: 原始 AoI 观测。
        original_h: 原始 H 观测。
        attack_state: episode 内攻击状态字典。
        attack_type: 结构化攻击模式名。
        extra: 结构化解释字段。
    输入输出:
        输入攻击前后状态段；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 重算 AoI/H 实际扰动统计。
        2. 合并标准扰动字段和结构化解释字段。
        3. 只在实际攻击发生时更新 attack_state。
    """
    # 重算 AoI 统计。
    aoi_info = _build_aoi_info(attacked_aoi, original_aoi)
    # 重算 H 统计。
    h_info = _build_h_info(attacked_h, original_h)
    # 合并标准字段。
    info = _merge_info(aoi_info, h_info)
    # 写入攻击类型。
    info["structural_attack_type"] = attack_type
    # 合并解释字段。
    info.update(extra)
    # 只有实际变化才消耗攻击步预算。
    if info["is_attacked"]:
        attack_state["attack_steps_used"] = int(attack_state.get("attack_steps_used", 0)) + 1
        attack_state["consecutive_attack_steps"] = int(attack_state.get("consecutive_attack_steps", 0)) + 1
    # 没有实际变化时重置连续攻击计数。
    else:
        attack_state["consecutive_attack_steps"] = 0
    # 返回完整 state。
    return merge_state(attacked_aoi, attacked_h), info


def semantic_aoi_mislead_attack(state, env, config, rng, attack_state):
    """
    作用:
        AoI-only 结构化误导攻击，只改变 agent 看到的 AoI。
    参数:
        state: 当前真实 state。
        env: 环境实例。
        config: 攻击配置。
        rng: 攻击随机数发生器。
        attack_state: episode 内攻击状态字典。
    输入输出:
        输入真实 state；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 判断是否触发攻击。
        2. 生成 AoI 误导扰动。
        3. H 保持不变。
    """
    # 复制 state，避免原地修改。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 不满足攻击触发条件时返回空攻击。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()
    # 拆分 AoI/H。
    aoi_part, h_part = split_state(state_copy, env)
    # 计算 AoI 预算。
    budget = min(int(_get_config_value(config, "max_aoi_features", int(env.n))), int(_get_config_value(config, "max_total_features", int(env.n))), int(env.n))
    # 生成 AoI 扰动。
    attacked_aoi, extra = _apply_aoi_mislead(aoi_part, env, config, budget)
    # 合并结果。
    return _finalize(attacked_aoi, h_part, aoi_part, h_part, attack_state, "semantic_aoi_mislead", extra)


def semantic_h_mislead_attack(state, env, config, rng, attack_state):
    """
    作用:
        H-only 结构化误导攻击，只改变 agent 看到的 H。
    参数:
        state: 当前真实 state。
        env: 环境实例。
        config: 攻击配置。
        rng: 攻击随机数发生器。
        attack_state: episode 内攻击状态字典。
    输入输出:
        输入真实 state；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 判断是否触发攻击。
        2. 生成 H 误导扰动。
        3. AoI 保持不变。
    """
    # 复制 state，避免原地修改。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 不满足攻击触发条件时返回空攻击。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()
    # 拆分 AoI/H。
    aoi_part, h_part = split_state(state_copy, env)
    # 计算 H 预算。
    budget = min(int(_get_config_value(config, "max_h_features", h_part.size)), int(_get_config_value(config, "max_total_features", h_part.size)), h_part.size)
    # 生成 H 扰动。
    attacked_h, extra = _apply_h_mislead(h_part, aoi_part, env, config, budget)
    # 合并结果。
    return _finalize(aoi_part, attacked_h, aoi_part, h_part, attack_state, "semantic_h_mislead", extra)


def semantic_joint_mislead_attack(state, env, config, rng, attack_state):
    """
    作用:
        Joint 结构化误导攻击，同时改变 agent 看到的 AoI 和 H。
    参数:
        state: 当前真实 state。
        env: 环境实例。
        config: 攻击配置。
        rng: 攻击随机数发生器。
        attack_state: episode 内攻击状态字典。
    输入输出:
        输入真实 state；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 判断是否触发攻击。
        2. 按总预算分配 AoI/H 预算。
        3. 分别生成 AoI 和 H 误导扰动。
    """
    # 复制 state，避免原地修改。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 不满足攻击触发条件时返回空攻击。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()
    # 拆分 AoI/H。
    aoi_part, h_part = split_state(state_copy, env)
    # 总预算受完整 state 维度限制。
    total_budget = min(int(_get_config_value(config, "max_total_features", aoi_part.size + h_part.size)), aoi_part.size + h_part.size)
    # AoI 预算取总预算 40%，并受 max_aoi_features 限制。
    aoi_budget = min(int(_get_config_value(config, "max_aoi_features", aoi_part.size)), int(ceil(0.4 * float(total_budget))), aoi_part.size)
    # 生成 AoI 扰动。
    attacked_aoi, aoi_extra = _apply_aoi_mislead(aoi_part, env, config, aoi_budget)
    # 剩余总预算用于 H，按 AoI 实际 L0 回填。
    remaining_total = max(0, total_budget - int(aoi_extra["aoi_l0"]))
    # H 预算受 max_h_features 和剩余总预算限制。
    h_budget = min(int(_get_config_value(config, "max_h_features", h_part.size)), remaining_total, h_part.size)
    # 生成 H 扰动。
    attacked_h, h_extra = _apply_h_mislead(h_part, aoi_part, env, config, h_budget)
    # 合并解释字段。
    extra = {}
    extra.update(aoi_extra)
    extra.update(h_extra)
    # 合并结果。
    return _finalize(attacked_aoi, attacked_h, aoi_part, h_part, attack_state, "semantic_joint_mislead", extra)
