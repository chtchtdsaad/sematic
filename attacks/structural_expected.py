"""
结构化 expected-cost 攻击模块（structural_expected.py）
=====================================================
本文件实现 AoI-only、H-only 和 Joint 的 expected-cost 强基线攻击。攻击者先
生成有限结构化候选 attacked_state，再调用 victim policy 选择动作，并用真实
state 估计下一步期望代价，最终选择代价最大的候选。
"""

from __future__ import annotations

from math import ceil

import numpy as np

from attacks.random_semantic import _build_aoi_info, _build_h_info, _can_attack, _empty_perturb_info, _get_config_value, _merge_info
from attacks.state_ops import merge_state, project_aoi_part, project_h_part, split_state
from attacks.structural_common import (
    compute_link_priority,
    compute_sensor_risk,
    decode_eval_action_to_assignment,
    flatten_h_index,
    score_candidate_state_by_expected_cost,
)
from attacks.action_utils import select_action_for_eval


def _jsonable_action(action):
    """
    作用:
        将 DQN 标量动作或 DDPG assignment 转成 JSON 友好对象。
    参数:
        action: DQN 为 int/np.integer；DDPG 为 list/tuple/np.ndarray assignment。
    输入输出:
        输入任意 eval 动作；输出 int 或 list[int]。
    核心步骤:
        1. 标量动作直接转 int。
        2. 序列动作展平后转 list[int]。
    """
    # DQN 常见返回 int 或 numpy 整数。
    if isinstance(action, (int, np.integer)):
        return int(action)
    # DDPG assignment 转为 list[int]。
    return [int(x) for x in np.asarray(action, dtype=np.int64).reshape(-1).tolist()]


def _make_candidate(label: str, aoi_part: np.ndarray, h_part: np.ndarray, attacked_aoi: np.ndarray, attacked_h: np.ndarray, env, config):
    """
    作用:
        投影、统计并过滤一个 expected-cost 候选状态。
    参数:
        label: 候选来源标签。
        aoi_part: 原始 AoI 状态段。
        h_part: 原始 H 状态段。
        attacked_aoi: 候选 AoI 状态段。
        attacked_h: 候选 H 状态段。
        env: 环境实例，用于合法范围投影。
        config: 攻击配置，用于预算检查。
    输入输出:
        输入原始状态段和候选状态段；输出 None 或 (candidate_state, perturb_info)。
    核心步骤:
        1. 将 AoI/H 投影到合法范围。
        2. 计算实际扰动统计。
        3. 过滤无变化或超预算候选。
    """
    # AoI 候选投影到合法范围。
    projected_aoi = project_aoi_part(attacked_aoi, env)
    # H 候选投影到合法范围。
    projected_h = project_h_part(attacked_h)
    # 计算 AoI 扰动统计。
    aoi_info = _build_aoi_info(projected_aoi, aoi_part)
    # 计算 H 扰动统计。
    h_info = _build_h_info(projected_h, h_part)
    # 合并为统一 perturb_info。
    info = _merge_info(aoi_info, h_info)
    # 无实际变化的候选没有攻击意义。
    if not info["is_attacked"]:
        return None
    # 检查 AoI L0 预算。
    if int(info["aoi_l0"]) > int(_get_config_value(config, "max_aoi_features", int(env.n))):
        return None
    # 检查 H L0 预算。
    if int(info["h_l0"]) > int(_get_config_value(config, "max_h_features", int(env.n) * int(env.m))):
        return None
    # 检查总 L0 预算。
    if int(info["total_l0"]) > int(_get_config_value(config, "max_total_features", int(env.n) + int(env.n) * int(env.m))):
        return None
    # 检查 AoI 单维幅值预算。
    if float(info["aoi_linf"]) > float(_get_config_value(config, "aoi_delta", 0)):
        return None
    # 检查 H 单维幅值预算。
    if float(info["h_linf"]) > float(_get_config_value(config, "h_delta", 0)):
        return None
    # 记录候选来源，便于报告解释。
    info["candidate_label"] = str(label)
    # 返回完整候选 state 和统计。
    return merge_state(projected_aoi, projected_h), info


def _aoi_expected_candidates(state: np.ndarray, env, config) -> list[tuple[np.ndarray, dict]]:
    """
    作用:
        生成 AoI-only expected-cost 候选集合。
    参数:
        state: 当前真实 state，shape=(N+N*M,)。
        env: 环境实例。
        config: 攻击配置，提供 AoI 预算和幅值。
    输入输出:
        输入真实 state；输出候选列表，每个元素为 (candidate_state, perturb_info)。
    核心步骤:
        1. 计算 sensor risk。
        2. 构造 top-risk 降低、bottom-risk 抬高和组合候选。
        3. 用 _make_candidate 过滤非法候选。
    """
    # 拆分真实 state。
    aoi_part, h_part = split_state(state, env)
    # 计算 sensor risk。
    risk = compute_sensor_risk(aoi_part, env, risk_mode="reset_gain")
    # AoI 预算受 max_aoi_features、max_total_features、N 三者限制。
    budget = min(int(_get_config_value(config, "max_aoi_features", int(env.n))), int(_get_config_value(config, "max_total_features", int(env.n))), int(env.n))
    # 读取扰动幅值。
    delta = float(int(_get_config_value(config, "aoi_delta", 1)))
    # 高风险 sensor 顺序。
    high_order = np.argsort(-risk, kind="mergesort").astype(int).tolist()
    # 低风险 sensor 顺序。
    low_order = np.argsort(risk, kind="mergesort").astype(int).tolist()
    # 候选容器。
    candidates: list[tuple[np.ndarray, dict]] = []

    # 内部函数：按降低/抬高 sensor 组合构造候选。
    def add(label: str, decrease_indices: list[int], increase_indices: list[int]) -> None:
        """
        作用:
            根据给定 sensor 升降集合追加一个 AoI 候选。
        参数:
            label: 候选标签。
            decrease_indices: 需要降低 AoI 的 sensor 下标。
            increase_indices: 需要提高 AoI 的 sensor 下标。
        输入输出:
            输入候选规则；无显式返回，可能向 candidates 追加元素。
        核心步骤:
            1. 拷贝 AoI。
            2. 按预算施加降低/提高扰动。
            3. 调用 _make_candidate 过滤并追加。
        """
        # 拷贝 AoI，避免修改真实 state。
        attacked_aoi = aoi_part.copy()
        # 记录已使用 sensor，防止同一维同时升降。
        used: set[int] = set()
        # 对高风险 sensor 降低 AoI。
        for idx in decrease_indices[:budget]:
            attacked_aoi[int(idx)] -= delta
            used.add(int(idx))
        # 剩余预算用于提高低风险诱饵 AoI。
        for idx in increase_indices:
            if len(used) >= budget:
                break
            if int(idx) not in used:
                attacked_aoi[int(idx)] += delta
                used.add(int(idx))
        # 生成候选并过滤。
        candidate = _make_candidate(label, aoi_part, h_part, attacked_aoi, h_part, env, config)
        # 有效候选加入集合。
        if candidate is not None:
            candidates.append(candidate)

    # 候选 1：只压低 top-1 高风险 sensor。
    add("aoi_top1_decrease", high_order[:1], [])
    # 候选 2：压低 top-k 高风险 sensor。
    add("aoi_topk_decrease", high_order[:budget], [])
    # 候选 3：只抬高 bottom-1 低风险诱饵。
    add("aoi_bottom1_increase", [], low_order[:1])
    # 候选 4：高风险隐藏 + 低风险诱饵组合。
    add("aoi_top_bottom_combo", high_order[: max(1, budget - 1)], low_order[:1])
    # 返回候选集合。
    return candidates


def _top_priority_indices(priority: np.ndarray, k: int, h_part: np.ndarray | None = None, require_decrease: bool = False) -> list[int]:
    """
    作用:
        按链路 priority 从高到低选择 H 展平下标。
    参数:
        priority: 链路优先级矩阵，shape=(N,M)。
        k: 需要选择的链路数量。
        h_part: 可选 H 展平段，用于过滤不可降低链路。
        require_decrease: 是否要求被选链路 H>0。
    输入输出:
        输入 priority 和过滤条件；输出 H 展平下标 list[int]。
    核心步骤:
        1. 展平 priority 并降序排序。
        2. 可选过滤 H<=0 的链路。
        3. 返回前 k 个链路下标。
    """
    # k<=0 时没有候选。
    if int(k) <= 0:
        return []
    # 展平 priority。
    priority_flat = np.asarray(priority, dtype=np.float64).reshape(-1)
    # 按 priority 降序排序。
    order = np.argsort(-priority_flat, kind="mergesort").astype(int).tolist()
    # 可选读取 H，用于过滤 H=0 的不可降低链路。
    h_arr = None if h_part is None else np.asarray(h_part, dtype=np.float32).reshape(-1)
    # 结果下标。
    selected: list[int] = []
    # 逐个检查排序后的链路。
    for flat_idx in order:
        # 要求可降低时，H<=0 的链路跳过。
        if require_decrease and h_arr is not None and float(h_arr[flat_idx]) <= 0.0:
            continue
        # 加入候选。
        selected.append(int(flat_idx))
        # 达到数量后停止。
        if len(selected) >= int(k):
            break
    # 返回下标列表。
    return selected


def _decoy_h_indices(h_part: np.ndarray, risk: np.ndarray, env, k: int, used: set[int] | None = None) -> list[int]:
    """
    作用:
        选择低风险 sensor 上低 H 的诱饵链路。
    参数:
        h_part: H 展平状态段，shape=(N*M,)。
        risk: sensor 风险数组，shape=(N,)。
        env: 环境实例，需要提供 n 和 m。
        k: 需要选择的诱饵链路数量。
        used: 已使用链路集合，避免重复扰动。
    输入输出:
        输入 H、risk 和排除集合；输出 H 展平下标 list[int]。
    核心步骤:
        1. 枚举所有 H<4 且未使用的链路。
        2. 按 sensor risk 升序、H 升序排序。
        3. 返回前 k 个链路下标。
    """
    # k<=0 时没有候选。
    if int(k) <= 0:
        return []
    # 已使用集合。
    used_set = set() if used is None else set(used)
    # H 恢复成矩阵。
    h_matrix = np.asarray(h_part, dtype=np.float32).reshape(int(env.n), int(env.m))
    # 候选元组容器。
    candidates: list[tuple[float, float, int]] = []
    # 枚举 sensor。
    for sensor_idx in range(int(env.n)):
        # 枚举 channel。
        for channel_idx in range(int(env.m)):
            # 计算展平下标。
            flat_idx = flatten_h_index(sensor_idx, channel_idx, env)
            # 已使用或 H=4 不可继续抬高时跳过。
            if flat_idx in used_set or float(h_matrix[sensor_idx, channel_idx]) >= 4.0:
                continue
            # 低风险优先，同风险下低 H 优先。
            candidates.append((float(risk[sensor_idx]), float(h_matrix[sensor_idx, channel_idx]), int(flat_idx)))
    # 按 risk、H、下标稳定排序。
    candidates.sort()
    # 返回前 k 个展平下标。
    return [int(item[-1]) for item in candidates[: int(k)]]


def _h_expected_candidates(state: np.ndarray, env, config) -> list[tuple[np.ndarray, dict]]:
    """
    作用:
        生成 H-only expected-cost 候选集合。
    参数:
        state: 当前真实 state，shape=(N+N*M,)。
        env: 环境实例。
        config: 攻击配置，提供 H 预算和幅值。
    输入输出:
        输入真实 state；输出候选列表，每个元素为 (candidate_state, perturb_info)。
    核心步骤:
        1. 计算链路 priority 和 sensor risk。
        2. 构造 top-priority 降低、decoy 抬高和组合候选。
        3. 用 _make_candidate 过滤非法候选。
    """
    # 拆分真实 state。
    aoi_part, h_part = split_state(state, env)
    # 计算链路 priority 和 sensor risk。
    priority, risk, _ = compute_link_priority(aoi_part, h_part, env, risk_mode="reset_gain")
    # H 预算受 max_h_features、max_total_features、N*M 限制。
    budget = min(int(_get_config_value(config, "max_h_features", h_part.size)), int(_get_config_value(config, "max_total_features", h_part.size)), h_part.size)
    # 读取 H 扰动幅值。
    delta = float(int(_get_config_value(config, "h_delta", 1)))
    # 高 priority 链路。
    top_links = _top_priority_indices(priority, budget, h_part=h_part, require_decrease=True)
    # 低风险诱饵链路。
    decoy_links = _decoy_h_indices(h_part, risk, env, budget)
    # 候选容器。
    candidates: list[tuple[np.ndarray, dict]] = []

    # 内部函数：按降低/抬高链路组合构造候选。
    def add(label: str, decrease_indices: list[int], increase_indices: list[int]) -> None:
        """
        作用:
            根据给定链路升降集合追加一个 H 候选。
        参数:
            label: 候选标签。
            decrease_indices: 需要降低 H 的链路展平下标。
            increase_indices: 需要提高 H 的链路展平下标。
        输入输出:
            输入候选规则；无显式返回，可能向 candidates 追加元素。
        核心步骤:
            1. 拷贝 H。
            2. 按预算施加降低/提高扰动。
            3. 调用 _make_candidate 过滤并追加。
        """
        # 拷贝 H，避免修改真实 state。
        attacked_h = h_part.copy()
        # 记录已使用链路。
        used: set[int] = set()
        # 降低高 priority 链路 H。
        for flat_idx in decrease_indices[:budget]:
            attacked_h[int(flat_idx)] -= delta
            used.add(int(flat_idx))
        # 剩余预算提高诱饵链路 H。
        for flat_idx in increase_indices:
            if len(used) >= budget:
                break
            if int(flat_idx) not in used:
                attacked_h[int(flat_idx)] += delta
                used.add(int(flat_idx))
        # 生成候选并过滤。
        candidate = _make_candidate(label, aoi_part, h_part, aoi_part, attacked_h, env, config)
        # 有效候选加入集合。
        if candidate is not None:
            candidates.append(candidate)

    # 候选 1：只压低 top-1 链路。
    add("h_top1_decrease", top_links[:1], [])
    # 候选 2：压低 top-k 链路。
    add("h_topk_decrease", top_links[:budget], [])
    # 候选 3：只抬高诱饵链路。
    add("h_decoy1_increase", [], decoy_links[:1])
    # 候选 4：高 priority 压制 + 诱饵组合。
    add("h_top_decoy_combo", top_links[: max(1, budget - 1)], decoy_links[:1])
    # 返回候选集合。
    return candidates


def _split_joint_budget(config, env) -> tuple[int, int, int]:
    """
    作用:
        为 Joint expected-cost 拆分总预算、AoI 预算和 H 预算。
    参数:
        config: 攻击配置，提供 max_total_features、max_aoi_features、max_h_features。
        env: 环境实例，需要提供 n 和 m。
    输入输出:
        输入配置和环境规模；输出 (total_budget, aoi_budget, h_budget)。
    核心步骤:
        1. 总预算受完整 state 维度限制。
        2. AoI 预算默认取总预算 40% 并受 max_aoi_features 限制。
        3. H 预算使用剩余预算并受 max_h_features 限制。
    """
    # 总预算受完整 state 维度限制。
    total_budget = min(int(_get_config_value(config, "max_total_features", int(env.n) + int(env.n) * int(env.m))), int(env.n) + int(env.n) * int(env.m))
    # AoI 预算取总预算 40%，并受 max_aoi_features 和 N 限制。
    aoi_budget = min(int(_get_config_value(config, "max_aoi_features", int(env.n))), int(ceil(0.4 * float(total_budget))), int(env.n))
    # H 预算使用剩余总预算，并受 max_h_features 和 N*M 限制。
    h_budget = min(int(_get_config_value(config, "max_h_features", int(env.n) * int(env.m))), max(0, total_budget - aoi_budget), int(env.n) * int(env.m))
    # 返回三类预算。
    return total_budget, aoi_budget, h_budget


def _joint_expected_candidates(state: np.ndarray, env, config, algo: str, agent) -> list[tuple[np.ndarray, dict]]:
    """
    作用:
        生成 Joint expected-cost 候选集合。
    参数:
        state: 当前真实 state，shape=(N+N*M,)。
        env: 环境实例。
        config: 攻击配置。
        algo: victim 算法名，支持 DQN 或 DDPG。
        agent: victim agent，用于生成 clean action 附近候选。
    输入输出:
        输入真实 state 和 victim policy；输出候选列表。
    核心步骤:
        1. 合并 AoI-only 与 H-only 候选。
        2. 构造 joint suppression 和 joint suppression decoy 候选。
        3. 基于 clean assignment 构造邻域候选。
    """
    # 拆分真实 state。
    aoi_part, h_part = split_state(state, env)
    # 拆分预算。
    _, aoi_budget, h_budget = _split_joint_budget(config, env)
    # 计算 priority 和 risk。
    priority, risk, _ = compute_link_priority(aoi_part, h_part, env, risk_mode="reset_gain")
    # 读取扰动幅值。
    aoi_delta = float(int(_get_config_value(config, "aoi_delta", 1)))
    h_delta = float(int(_get_config_value(config, "h_delta", 1)))
    # 高风险 sensor 顺序。
    high_sensors = np.argsort(-risk, kind="mergesort").astype(int).tolist()
    # 低风险 sensor 顺序。
    low_sensors = np.argsort(risk, kind="mergesort").astype(int).tolist()
    # 先放入 AoI-only 与 H-only 候选。
    candidates = _aoi_expected_candidates(state, env, config) + _h_expected_candidates(state, env, config)

    # 内部函数：构造 Joint 候选。
    def add(label: str, dec_sensors: list[int], inc_sensors: list[int], dec_h: list[int], inc_h: list[int]) -> None:
        """
        作用:
            根据 AoI/H 升降集合追加一个 Joint 候选。
        参数:
            label: 候选标签。
            dec_sensors: 需要降低 AoI 的 sensor 下标。
            inc_sensors: 需要提高 AoI 的 sensor 下标。
            dec_h: 需要降低 H 的链路展平下标。
            inc_h: 需要提高 H 的链路展平下标。
        输入输出:
            输入候选规则；无显式返回，可能向 candidates 追加元素。
        核心步骤:
            1. 拷贝 AoI/H。
            2. 按 AoI/H 预算分别施加扰动。
            3. 调用 _make_candidate 过滤并追加。
        """
        # 拷贝 AoI。
        attacked_aoi = aoi_part.copy()
        # 拷贝 H。
        attacked_h = h_part.copy()
        # AoI 已用 sensor 集合。
        used_sensors: set[int] = set()
        # 降低高风险 sensor AoI。
        for idx in dec_sensors[:aoi_budget]:
            attacked_aoi[int(idx)] -= aoi_delta
            used_sensors.add(int(idx))
        # 抬高低风险诱饵 sensor AoI。
        for idx in inc_sensors:
            if len(used_sensors) >= aoi_budget:
                break
            if int(idx) not in used_sensors:
                attacked_aoi[int(idx)] += aoi_delta
                used_sensors.add(int(idx))
        # H 已用链路集合。
        used_h: set[int] = set()
        # 降低高 priority 链路 H。
        for flat_idx in dec_h[:h_budget]:
            attacked_h[int(flat_idx)] -= h_delta
            used_h.add(int(flat_idx))
        # 抬高诱饵链路 H。
        for flat_idx in inc_h:
            if len(used_h) >= h_budget:
                break
            if int(flat_idx) not in used_h:
                attacked_h[int(flat_idx)] += h_delta
                used_h.add(int(flat_idx))
        # 生成候选并过滤。
        candidate = _make_candidate(label, aoi_part, h_part, attacked_aoi, attacked_h, env, config)
        # 有效候选加入集合。
        if candidate is not None:
            candidates.append(candidate)

    # 高 priority H 链路。
    top_h = _top_priority_indices(priority, h_budget, h_part=h_part, require_decrease=True)
    # 候选：联合 suppression。
    add("joint_suppression", high_sensors[:aoi_budget], [], top_h[:h_budget], [])
    # 候选：联合 suppression + decoy。
    decoy_h = _decoy_h_indices(h_part, risk, env, h_budget, used=set(top_h[: max(1, h_budget - 1)]))
    add("joint_suppression_decoy", high_sensors[: max(1, aoi_budget - 1)], low_sensors[:1], top_h[: max(1, h_budget - 1)], decoy_h[:1])
    # clean action 附近候选：先得到真实 state 下 victim action。
    clean_action = select_action_for_eval(algo, state, agent, env)
    # 解码为 assignment。
    clean_assignment = decode_eval_action_to_assignment(algo, clean_action, env)
    # 当前 clean action 调度的 sensor。
    selected_sensors = [sensor_idx for sensor_idx, channel_id in enumerate(clean_assignment) if int(channel_id) > 0]
    # 当前 clean action 使用的 H 链路。
    selected_h = [flatten_h_index(sensor_idx, int(channel_id) - 1, env) for sensor_idx, channel_id in enumerate(clean_assignment) if int(channel_id) > 0]
    # clean 附近诱饵链路。
    clean_decoy_h = _decoy_h_indices(h_part, risk, env, h_budget, used=set(selected_h))
    # 加入 clean action 附近候选。
    add("joint_clean_assignment_neighborhood", selected_sensors, low_sensors[:1], selected_h, clean_decoy_h[:1])
    # 返回候选集合。
    return candidates


def _select_best_candidate(candidates: list[tuple[np.ndarray, dict]], real_state: np.ndarray, env, algo: str, agent, cost_mode: str):
    """
    作用:
        从候选集合中选择一步 expected cost 最大的候选。
    参数:
        candidates: 候选列表，每个元素为 (candidate_state, perturb_info)。
        real_state: 真实环境 state，用于 expected-cost 估计。
        env: 环境实例。
        algo: victim 算法名，支持 DQN 或 DDPG。
        agent: victim agent。
        cost_mode: 代价模式，支持 mse 或 sum_aoi。
    输入输出:
        输入候选集合；输出 None 或最优 (attacked_state, perturb_info)。
    核心步骤:
        1. 逐个候选调用 victim policy。
        2. 用真实 state 估计候选动作的 expected cost。
        3. 选择分数最高的候选并记录解释字段。
    """
    # 没有候选直接返回 None。
    if not candidates:
        return None
    # 最优候选初始化。
    best = None
    # 最优分数初始化为负无穷。
    best_score = float("-inf")
    # 遍历所有候选。
    for candidate_state, info in candidates:
        # 计算候选诱导动作和真实 expected cost。
        score, candidate_action = score_candidate_state_by_expected_cost(candidate_state, real_state, env, algo, agent, cost_mode)
        # 分数更高则更新最优。
        if float(score) > best_score:
            best_info = dict(info)
            best_info["expected_cost_score"] = float(score)
            best_info["expected_cost_action"] = _jsonable_action(candidate_action)
            best = (candidate_state.astype(np.float32, copy=True), best_info)
            best_score = float(score)
    # 返回最优候选。
    return best


def _finalize(best, attack_state: dict, attack_type: str):
    """
    作用:
        补充攻击类型并更新 episode 内攻击预算状态。
    参数:
        best: None 或最优候选 (attacked_state, perturb_info)。
        attack_state: episode 内攻击状态字典。
        attack_type: 结构化攻击模式名。
    输入输出:
        输入最优候选；输出 None 或最终 (attacked_state, perturb_info)。
    核心步骤:
        1. 没有候选时返回 None。
        2. 写入 structural_attack_type。
        3. 若实际攻击发生则更新 attack_state。
    """
    # 没有候选时返回 None。
    if best is None:
        return None
    # 拆出 attacked_state 和统计。
    attacked_state, info = best
    # 标记结构化攻击类型。
    info["structural_attack_type"] = str(attack_type)
    # 只有实际攻击才消耗预算。
    if info.get("is_attacked", False):
        attack_state["attack_steps_used"] = int(attack_state.get("attack_steps_used", 0)) + 1
        attack_state["consecutive_attack_steps"] = int(attack_state.get("consecutive_attack_steps", 0)) + 1
    # 返回副本和统计。
    return attacked_state.astype(np.float32, copy=True), info


def semantic_aoi_expected_cost_attack(state, env, config, rng, attack_state, algo, agent):
    """
    作用:
        AoI-only expected-cost 攻击入口。
    参数:
        state: 当前真实 state，shape=(N+N*M,)。
        env: 环境实例。
        config: 攻击配置。
        rng: 攻击随机数发生器，用于触发判断。
        attack_state: episode 内攻击状态字典。
        algo: victim 算法名，支持 DQN 或 DDPG。
        agent: victim agent。
    输入输出:
        输入真实 state；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 判断本 step 是否允许攻击。
        2. 生成 AoI 候选。
        3. 选择 expected cost 最大的候选。
    """
    # 复制 state，避免原地修改真实状态。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 不满足触发条件时返回空攻击。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()
    # 读取代价模式。
    cost_mode = str(_get_config_value(config, "expected_cost_mode", "mse"))
    # 生成 AoI 候选。
    candidates = _aoi_expected_candidates(state_copy, env, config)
    # 选择最优候选。
    best = _select_best_candidate(candidates, state_copy, env, algo, agent, cost_mode)
    # 返回最终攻击结果。
    return _finalize(best, attack_state, "semantic_aoi_expected_cost") or (state_copy, _empty_perturb_info())


def semantic_h_expected_cost_attack(state, env, config, rng, attack_state, algo, agent):
    """
    作用:
        H-only expected-cost 攻击入口。
    参数:
        state: 当前真实 state，shape=(N+N*M,)。
        env: 环境实例。
        config: 攻击配置。
        rng: 攻击随机数发生器，用于触发判断。
        attack_state: episode 内攻击状态字典。
        algo: victim 算法名，支持 DQN 或 DDPG。
        agent: victim agent。
    输入输出:
        输入真实 state；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 判断本 step 是否允许攻击。
        2. 生成 H 候选。
        3. 选择 expected cost 最大的候选。
    """
    # 复制 state，避免原地修改真实状态。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 不满足触发条件时返回空攻击。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()
    # 读取代价模式。
    cost_mode = str(_get_config_value(config, "expected_cost_mode", "mse"))
    # 生成 H 候选。
    candidates = _h_expected_candidates(state_copy, env, config)
    # 选择最优候选。
    best = _select_best_candidate(candidates, state_copy, env, algo, agent, cost_mode)
    # 返回最终攻击结果。
    return _finalize(best, attack_state, "semantic_h_expected_cost") or (state_copy, _empty_perturb_info())


def semantic_joint_expected_cost_attack(state, env, config, rng, attack_state, algo, agent):
    """
    作用:
        Joint expected-cost 攻击入口。
    参数:
        state: 当前真实 state，shape=(N+N*M,)。
        env: 环境实例。
        config: 攻击配置。
        rng: 攻击随机数发生器，用于触发判断。
        attack_state: episode 内攻击状态字典。
        algo: victim 算法名，支持 DQN 或 DDPG。
        agent: victim agent。
    输入输出:
        输入真实 state；输出 (attacked_state, perturb_info)。
    核心步骤:
        1. 判断本 step 是否允许攻击。
        2. 生成 AoI/H/Joint 候选。
        3. 选择 expected cost 最大的候选。
    """
    # 复制 state，避免原地修改真实状态。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 不满足触发条件时返回空攻击。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()
    # 读取代价模式。
    cost_mode = str(_get_config_value(config, "expected_cost_mode", "mse"))
    # 生成 Joint 候选。
    candidates = _joint_expected_candidates(state_copy, env, config, algo, agent)
    # 选择最优候选。
    best = _select_best_candidate(candidates, state_copy, env, algo, agent, cost_mode)
    # 返回最终攻击结果。
    return _finalize(best, attack_state, "semantic_joint_expected_cost") or (state_copy, _empty_perturb_info())
