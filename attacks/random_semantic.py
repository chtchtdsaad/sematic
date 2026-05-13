"""
随机语义攻击模块（random_semantic.py）
====================================
本文件只负责 clean / random_aoi / random_h / random_joint 四类状态观测扰动逻辑。
攻击函数只返回被扰动的 state 副本，不写回 env.aoi、env.channel_state 或 env.channel_loss。
"""

from __future__ import annotations

import numpy as np

from attacks.state_ops import merge_state, project_aoi_part, project_h_part, split_state


def _get_config_value(config, name: str, default):
    """
    作用:
        读取 AttackConfig 或测试配置对象中的字段，并在旧配置缺少字段时使用默认值。
    输入格式:
        config: 任意带属性的配置对象。
        name: str，需要读取的属性名。
        default: 属性不存在时使用的默认值。
    输出格式:
        配置值或默认值。
    核心步骤:
        1. 使用 getattr 读取字段。
        2. 字段不存在时返回默认值，兼容阶段 2 的精简 AttackConfig。
    """
    return getattr(config, name, default)


def _empty_perturb_info() -> dict:
    """
    作用:
        构造未发生攻击时的统一扰动统计字典。
    输入格式:
        无。
    输出格式:
        dict，包含 AoI/H 的 L0、L1、Linf 和变化下标。
    核心步骤:
        1. 将 is_attacked 置为 False。
        2. 将所有扰动强度字段置零。
        3. 将变化下标列表置空。
    """
    return {
        "is_attacked": False,
        "aoi_l0": 0,
        "h_l0": 0,
        "total_l0": 0,
        "aoi_l1": 0.0,
        "h_l1": 0.0,
        "aoi_linf": 0.0,
        "h_linf": 0.0,
        "aoi_changed_indices": [],
        "h_changed_indices": [],
    }


def _sample_integer_bias(delta: int, direction: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    作用:
        根据方向约束生成非零整数扰动，供 AoI 和 H 复用。
    输入格式:
        delta: int，单维最大扰动幅度。
        direction: str，random / increase / decrease / mixed。
        size: int，需要生成的扰动个数。
        rng: np.random.Generator，随机数发生器。
    输出格式:
        np.ndarray[int32], shape=(size,)。
    核心步骤:
        1. delta 小于等于 0 或 size 为 0 时返回空扰动。
        2. 按方向生成正向、负向或混合的非零整数。
        3. random 模式在 [-delta, -1] 和 [1, delta] 中均匀抽样。
    """
    if delta <= 0 or size <= 0:
        return np.zeros(size, dtype=np.int32)

    if direction == "increase":
        return rng.integers(1, delta + 1, size=size, dtype=np.int32)
    if direction == "decrease":
        return -rng.integers(1, delta + 1, size=size, dtype=np.int32)
    if direction == "mixed":
        signs = rng.choice(np.array([-1, 1], dtype=np.int32), size=size)
        magnitudes = rng.integers(1, delta + 1, size=size, dtype=np.int32)
        return signs * magnitudes
    if direction == "random":
        candidates = np.concatenate(
            [
                np.arange(-delta, 0, dtype=np.int32),
                np.arange(1, delta + 1, dtype=np.int32),
            ]
        )
        return rng.choice(candidates, size=size, replace=True).astype(np.int32, copy=False)

    raise ValueError(f"unsupported perturbation direction: {direction}")


def _build_aoi_info(attacked_aoi: np.ndarray, original_aoi: np.ndarray) -> dict:
    """
    作用:
        根据扰动前后的 AoI 段计算统计信息。
    输入格式:
        attacked_aoi: np.ndarray[float32], shape=(N,)。
        original_aoi: np.ndarray[float32], shape=(N,)。
    输出格式:
        dict，包含 aoi_l0 / aoi_l1 / aoi_linf / aoi_changed_indices。
    核心步骤:
        1. 计算逐维绝对差。
        2. 使用非零差值定位实际变化维度。
        3. 汇总 L0、L1 和 Linf。
    """
    diff = np.abs(attacked_aoi.astype(np.float32) - original_aoi.astype(np.float32))
    changed_indices = np.flatnonzero(diff > 0).astype(int).tolist()
    return {
        "aoi_l0": int(len(changed_indices)),
        "aoi_l1": float(np.sum(diff)),
        "aoi_linf": float(np.max(diff)) if diff.size else 0.0,
        "aoi_changed_indices": changed_indices,
    }


def _build_h_info(attacked_h: np.ndarray, original_h: np.ndarray) -> dict:
    """
    作用:
        根据扰动前后的 H 段计算统计信息。
    输入格式:
        attacked_h: np.ndarray[float32], shape=(N*M,)。
        original_h: np.ndarray[float32], shape=(N*M,)。
    输出格式:
        dict，包含 h_l0 / h_l1 / h_linf / h_changed_indices。
    核心步骤:
        1. 计算逐维绝对差。
        2. 使用非零差值定位实际变化维度。
        3. 汇总 L0、L1 和 Linf。
    """
    diff = np.abs(attacked_h.astype(np.float32) - original_h.astype(np.float32))
    changed_indices = np.flatnonzero(diff > 0).astype(int).tolist()
    return {
        "h_l0": int(len(changed_indices)),
        "h_l1": float(np.sum(diff)),
        "h_linf": float(np.max(diff)) if diff.size else 0.0,
        "h_changed_indices": changed_indices,
    }


def random_bias_aoi(aoi_part, env, config, rng, max_features_override=None):
    """
    作用:
        对 AoI 段施加满足幅值、稀疏度和合法性约束的随机扰动。
    输入格式:
        aoi_part: np.ndarray[float32], shape=(N,)。
        env: 环境对象，需包含 env.max_aoi。
        config: AttackConfig 或等价配置对象。
        rng: np.random.Generator。
        max_features_override: int | None，random_joint 下的本步 AoI 预算覆盖值。
    输出格式:
        attacked_aoi: np.ndarray[float32], shape=(N,)。
        info: dict，包含 AoI 扰动统计。
    核心步骤:
        1. 复制 AoI 输入，避免原地修改。
        2. 按 max_aoi_features 和 max_total_features 计算候选扰动数量。
        3. 随机选择 AoI 维度并施加非零整数扰动。
        4. 调用 project_aoi_part 执行 round + clip 到 [1, env.max_aoi]。
        5. 返回实际变化后的统计信息。
    """
    original_aoi = np.asarray(aoi_part, dtype=np.float32).reshape(-1).copy()
    attacked_aoi = original_aoi.copy()

    aoi_delta = int(round(float(_get_config_value(config, "aoi_delta", 1))))
    max_aoi_features = int(_get_config_value(config, "max_aoi_features", attacked_aoi.shape[0]))
    max_total_features = int(_get_config_value(config, "max_total_features", attacked_aoi.shape[0]))
    if max_features_override is not None:
        max_aoi_features = int(max_features_override)

    num_candidates = min(max_aoi_features, max_total_features, attacked_aoi.shape[0])
    if num_candidates <= 0 or aoi_delta <= 0:
        return attacked_aoi.astype(np.float32, copy=True), _build_aoi_info(attacked_aoi, original_aoi)

    selected_indices = rng.choice(attacked_aoi.shape[0], size=num_candidates, replace=False)
    direction = str(_get_config_value(config, "aoi_direction", _get_config_value(config, "direction", "random")))
    bias = _sample_integer_bias(aoi_delta, direction, num_candidates, rng).astype(np.float32)

    attacked_aoi[selected_indices] = attacked_aoi[selected_indices] + bias
    attacked_aoi = project_aoi_part(attacked_aoi, env)
    return attacked_aoi.astype(np.float32, copy=True), _build_aoi_info(attacked_aoi, original_aoi)


def random_bias_h(h_part, config, rng, max_features_override=None):
    """
    作用:
        对 H 段施加满足幅值、稀疏度和合法性约束的随机扰动。
    输入格式:
        h_part: np.ndarray[float32], shape=(N*M,)。
        config: AttackConfig 或等价配置对象。
        rng: np.random.Generator。
        max_features_override: int | None，random_joint 下的本步 H 预算覆盖值。
    输出格式:
        attacked_h: np.ndarray[float32], shape=(N*M,)。
        info: dict，包含 H 扰动统计。
    核心步骤:
        1. 复制 H 输入，避免原地修改。
        2. 按 max_h_features 和 max_total_features 计算候选扰动数量。
        3. 随机选择 H 维度并施加非零整数扰动。
        4. 调用 project_h_part 执行 round + clip 到 [0, 4]。
        5. 返回实际变化后的统计信息。
    """
    original_h = np.asarray(h_part, dtype=np.float32).reshape(-1).copy()
    attacked_h = original_h.copy()

    h_delta = int(round(float(_get_config_value(config, "h_delta", 1))))
    max_h_features = int(_get_config_value(config, "max_h_features", attacked_h.shape[0]))
    max_total_features = int(_get_config_value(config, "max_total_features", attacked_h.shape[0]))
    if max_features_override is not None:
        max_h_features = int(max_features_override)

    num_candidates = min(max_h_features, max_total_features, attacked_h.shape[0])
    if num_candidates <= 0 or h_delta <= 0:
        return attacked_h.astype(np.float32, copy=True), _build_h_info(attacked_h, original_h)

    selected_indices = rng.choice(attacked_h.shape[0], size=num_candidates, replace=False)
    direction = str(_get_config_value(config, "h_direction", _get_config_value(config, "direction", "random")))
    bias = _sample_integer_bias(h_delta, direction, num_candidates, rng).astype(np.float32)

    attacked_h[selected_indices] = attacked_h[selected_indices] + bias
    attacked_h = project_h_part(attacked_h)
    return attacked_h.astype(np.float32, copy=True), _build_h_info(attacked_h, original_h)


def _can_attack(config, rng: np.random.Generator, attack_state: dict) -> bool:
    """
    作用:
        根据概率、episode 预算、连续攻击限制和 cooldown 判断本步是否允许攻击。
    输入格式:
        config: AttackConfig 或等价配置对象。
        rng: np.random.Generator。
        attack_state: dict，记录 episode 内攻击预算使用情况。
    输出格式:
        bool，True 表示本步可以尝试攻击。
    核心步骤:
        1. 若仍处于 cooldown，则消耗一个冷却步并跳过攻击。
        2. 按 attack_prob 抽样判断是否触发。
        3. strict_budget 为 True 时检查 max_attack_steps。
        4. 检查 max_consecutive_steps，触顶时进入 cooldown 并跳过本步。
    """
    # cooldown_remaining 表示还需要强制跳过多少个 step；跳过时不消耗随机数。
    cooldown_remaining = int(attack_state.get("cooldown_remaining", 0))
    if cooldown_remaining > 0:
        # 当前 step 被 cooldown 占用，因此先递减剩余冷却步数。
        attack_state["cooldown_remaining"] = max(0, cooldown_remaining - 1)
        # cooldown 期间连续攻击计数保持为 0。
        attack_state["consecutive_attack_steps"] = 0
        return False

    # 按概率判断本步是否尝试攻击。
    attack_prob = float(_get_config_value(config, "attack_prob", 0.0))
    if rng.random() >= attack_prob:
        # 概率未命中时，连续攻击链条自然断开。
        attack_state["consecutive_attack_steps"] = 0
        return False

    # strict_budget 打开时，episode 内攻击步数不能超过 max_attack_steps。
    strict_budget = bool(_get_config_value(config, "strict_budget", True))
    if strict_budget and int(attack_state.get("attack_steps_used", 0)) >= int(attack_state.get("max_attack_steps", 0)):
        # 总预算耗尽时同样打断连续攻击链条。
        attack_state["consecutive_attack_steps"] = 0
        return False

    # 读取连续攻击上限；小于 1 时视为不允许连续攻击，至少保留 1 的有效上限。
    max_consecutive_steps = int(_get_config_value(config, "max_consecutive_steps", 5))
    max_consecutive_steps = max(1, max_consecutive_steps)
    if int(attack_state.get("consecutive_attack_steps", 0)) >= max_consecutive_steps:
        # 达到连续攻击上限后，本 step 必须跳过，并按配置进入后续 cooldown。
        attack_state["consecutive_attack_steps"] = 0
        # cooldown_steps 包含当前被强制跳过的 step；因此后续剩余冷却步数为 cooldown_steps-1。
        cooldown_steps = max(0, int(_get_config_value(config, "cooldown_steps", 0)))
        attack_state["cooldown_remaining"] = max(0, cooldown_steps - 1)
        return False

    # 所有约束均通过，本步允许攻击。
    return True


def _merge_info(aoi_info: dict, h_info: dict) -> dict:
    """
    作用:
        合并 AoI 和 H 的扰动统计，生成 semantic_random_attack 的统一返回信息。
    输入格式:
        aoi_info: dict，AoI 统计。
        h_info: dict，H 统计。
    输出格式:
        dict，完整扰动统计。
    核心步骤:
        1. 复制默认空统计。
        2. 写入 AoI/H 统计字段。
        3. 根据 total_l0 判断本步是否实际攻击成功。
    """
    info = _empty_perturb_info()
    info.update(aoi_info)
    info.update(h_info)
    info["total_l0"] = int(info["aoi_l0"] + info["h_l0"])
    info["is_attacked"] = bool(info["total_l0"] > 0)
    return info


def semantic_random_attack(state, env, config, rng, attack_state):
    """
    作用:
        根据攻击配置生成 attacked_state，并返回扰动统计。
    输入格式:
        state: np.ndarray[float32], shape=(N + N*M,)。
        env: 环境对象，需包含 n、m、max_aoi 等属性。
        config: AttackConfig 或等价配置对象。
        rng: np.random.Generator。
        attack_state: dict，记录 episode 内攻击预算和连续攻击计数。
    输出格式:
        attacked_state: np.ndarray[float32], shape=(N + N*M,)。
        perturb_info: dict，包含本步扰动统计。
    核心步骤:
        1. clean 模式直接返回 state 副本和空统计。
        2. 按概率、时间预算和连续攻击限制判断是否攻击。
        3. 拆分 state 为 AoI 和 H 段。
        4. 根据 random_aoi / random_h / random_joint 调用对应扰动函数。
        5. 合并被扰动的 state 副本，并只在实际变化时更新 attack_state。
    """
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    mode = str(_get_config_value(config, "mode", "clean"))
    if mode == "clean":
        return state_copy, _empty_perturb_info()

    if mode not in {"random_aoi", "random_h", "random_joint"}:
        raise ValueError(f"unsupported attack mode: {mode}")

    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()

    aoi_part, h_part = split_state(state_copy, env)
    attacked_aoi = aoi_part.copy()
    attacked_h = h_part.copy()
    aoi_info = _build_aoi_info(attacked_aoi, aoi_part)
    h_info = _build_h_info(attacked_h, h_part)

    if mode == "random_aoi":
        attacked_aoi, aoi_info = random_bias_aoi(aoi_part, env, config, rng)
    elif mode == "random_h":
        attacked_h, h_info = random_bias_h(h_part, config, rng)
    elif mode == "random_joint":
        total_budget = int(_get_config_value(config, "max_total_features", aoi_part.size + h_part.size))
        aoi_budget = min(int(_get_config_value(config, "max_aoi_features", aoi_part.size)), total_budget)
        attacked_aoi, aoi_info = random_bias_aoi(aoi_part, env, config, rng, max_features_override=aoi_budget)

        remaining_budget = max(0, total_budget - int(aoi_info["aoi_l0"]))
        h_budget = min(int(_get_config_value(config, "max_h_features", h_part.size)), remaining_budget)
        attacked_h, h_info = random_bias_h(h_part, config, rng, max_features_override=h_budget)

    attacked_state = merge_state(attacked_aoi, attacked_h)
    perturb_info = _merge_info(aoi_info, h_info)

    if perturb_info["is_attacked"]:
        attack_state["attack_steps_used"] = int(attack_state.get("attack_steps_used", 0)) + 1
        attack_state["consecutive_attack_steps"] = int(attack_state.get("consecutive_attack_steps", 0)) + 1
    else:
        attack_state["consecutive_attack_steps"] = 0

    return attacked_state.astype(np.float32, copy=True), perturb_info
