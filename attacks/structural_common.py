"""
结构化攻击公共工具模块（structural_common.py）
============================================
本文件集中保存结构化攻击可复用的工具函数，包括 sensor risk、链路优先级、
H 下标转换、DQN/DDPG 动作解码和一步 expected-cost 估计。这里的函数只读
state/env，不调用 env.step，也不写回 env.aoi、env.channel_state 或 reward。
"""

# 启用前向类型注解，避免运行时解析复杂类型。
from __future__ import annotations

# 导入 numpy，所有状态切片、排序、代价计算都依赖 numpy。
import numpy as np

# 复用 eval 阶段统一动作选择入口，用于 expected-cost 候选评分。
from attacks.action_utils import select_action_for_eval
# 复用状态拆分逻辑，保证 AoI/H 切片格式一致。
from attacks.state_ops import split_state
# 读取 H 离散状态到丢包率的映射表。
from config import PACKET_LOSS_LEVELS
# 复用核心 MSE 计算函数，不在攻击模块重复实现矩阵递推。
from core import compute_mse_from_aoi


def compute_sensor_mse_values(aoi_part: np.ndarray, env) -> np.ndarray:
    """
    作用:
        根据 AoI 向量逐 sensor 计算当前 MSE。
    参数:
        aoi_part: AoI 状态段，shape=(N,)。
        env: 环境实例，需要提供 n、max_aoi、p_bars、a_mats、w_mats 和 MSE 缓存。
    输入输出:
        输入 AoI 可为浮点扰动值；输出为 np.ndarray[float64]，shape=(N,)。
    核心步骤:
        1. 将 AoI round 并 clip 到合法整数范围。
        2. 对每个 sensor 调用 compute_mse_from_aoi。
        3. 返回逐 sensor MSE。
    """
    # 将 AoI 转为合法整数语义，避免浮点扰动残留影响 MSE 计算。
    aoi_arr = np.clip(np.rint(np.asarray(aoi_part, dtype=np.float64).reshape(-1)), 1, int(env.max_aoi)).astype(int)
    # AoI 长度必须等于 sensor 数，否则说明 state 切片错误。
    if aoi_arr.size != int(env.n):
        raise ValueError(f"aoi_part length must be n={env.n}, got {aoi_arr.size}.")
    # 预分配逐 sensor MSE 结果。
    values = np.zeros(int(env.n), dtype=np.float64)
    # 遍历每个 sensor，使用该 sensor 自己的系统矩阵和缓存。
    for sensor_idx in range(int(env.n)):
        # 调用核心函数计算 Tr(P_n(tau_n))。
        values[sensor_idx] = compute_mse_from_aoi(
            p_bar=env.p_bars[sensor_idx],
            a=env.a_mats[sensor_idx],
            w=env.w_mats[sensor_idx],
            aoi=int(aoi_arr[sensor_idx]),
            a_powers=env.a_powers_cache[sensor_idx],
            noise_cov_sums=env.noise_cov_sums_cache[sensor_idx],
        )
    # 返回 shape=(N,) 的逐 sensor MSE。
    return values


def compute_sensor_risk(aoi_part: np.ndarray, env, risk_mode: str = "reset_gain") -> np.ndarray:
    """
    作用:
        根据 AoI 计算 sensor 级风险，供结构化攻击排序使用。
    参数:
        aoi_part: AoI 状态段，shape=(N,)。
        env: 环境实例，需要提供 MSE 计算所需字段。
        risk_mode: 风险模式，支持 current_mse、marginal_growth、reset_gain。
    输入输出:
        输入 AoI 状态段；输出 np.ndarray[float64]，shape=(N,)。
    核心步骤:
        1. 先计算当前 AoI 下的 sensor MSE。
        2. 根据 risk_mode 转换为当前误差、边际增长或重置收益。
        3. 返回用于排序的 risk 数组。
    """
    # 标准化 AoI，所有 risk 模式都基于合法整数 AoI。
    aoi_arr = np.clip(np.rint(np.asarray(aoi_part, dtype=np.float64).reshape(-1)), 1, int(env.max_aoi)).astype(int)
    # 当前 AoI 对应的 MSE。
    current_mse = compute_sensor_mse_values(aoi_arr, env)
    # 统一 risk mode 字符串。
    mode = str(risk_mode)
    # current_mse 模式直接把当前误差视为风险。
    if mode == "current_mse":
        return current_mse
    # marginal_growth 模式衡量下一步不调度带来的 MSE 增量。
    if mode == "marginal_growth":
        next_aoi = np.minimum(aoi_arr + 1, int(env.max_aoi))
        return compute_sensor_mse_values(next_aoi, env) - current_mse
    # reset_gain 模式衡量调度成功重置到 AoI=1 能减少多少 MSE。
    if mode == "reset_gain":
        reset_mse = compute_sensor_mse_values(np.ones(int(env.n), dtype=np.float64), env)
        return current_mse - reset_mse
    # 未知模式直接报错，避免静默使用错误指标。
    raise ValueError(f"unsupported risk_mode: {risk_mode}")


def h_to_success_prob(h_part: np.ndarray) -> np.ndarray:
    """
    作用:
        将离散 H 状态索引映射为链路成功率。
    参数:
        h_part: H 状态段，可为 shape=(N*M,) 或 shape=(N,M)。
    输入输出:
        输入 H 离散状态；输出与输入同形状的 success_prob。
    核心步骤:
        1. 将 H round 并 clip 到 PACKET_LOSS_LEVELS 合法索引。
        2. 查表得到 packet loss。
        3. 返回 1 - packet loss。
    """
    # H 是离散索引，先 round 再裁剪到 PACKET_LOSS_LEVELS 合法范围。
    h_index = np.clip(np.rint(np.asarray(h_part, dtype=np.float64)), 0, len(PACKET_LOSS_LEVELS) - 1).astype(int)
    # 成功率等于 1 - 丢包率。
    return (1.0 - PACKET_LOSS_LEVELS[h_index]).astype(np.float64, copy=False)


def compute_link_priority(aoi_part: np.ndarray, h_part: np.ndarray, env, risk_mode: str = "reset_gain") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    作用:
        计算每条 sensor-channel 链路的结构优先级 priority。
    参数:
        aoi_part: AoI 状态段，shape=(N,)。
        h_part: H 状态段，shape=(N*M,)。
        env: 环境实例，需要提供 n 和 m。
        risk_mode: sensor risk 计算模式。
    输入输出:
        输入 AoI/H 状态段；输出 priority(N,M)、risk(N,) 和 success_prob(N,M)。
    核心步骤:
        1. 根据 AoI 计算 sensor risk。
        2. 将 H 映射为链路成功率。
        3. 用 risk * success_prob 得到链路优先级。
    """
    # 先计算 sensor 级风险。
    risk = compute_sensor_risk(aoi_part, env, risk_mode=risk_mode)
    # 将 H 展平段恢复为 (N,M)。
    h_matrix = np.asarray(h_part, dtype=np.float64).reshape(int(env.n), int(env.m))
    # 每条链路的成功率。
    success_prob = h_to_success_prob(h_matrix)
    # 广播 risk 到每条 sensor-channel 链路。
    priority = risk[:, None] * success_prob
    # 返回链路优先级、sensor risk 和成功率矩阵。
    return priority.astype(np.float64, copy=False), risk, success_prob


def select_top_risk_sensors(risk: np.ndarray, k: int) -> list[int]:
    """
    作用:
        选择 risk 最大的 k 个 sensor 下标。
    参数:
        risk: sensor 风险数组，shape=(N,)。
        k: 需要选择的 sensor 数。
    输入输出:
        输入 risk 数组；输出按 risk 降序排列的 list[int]。
    核心步骤:
        1. 对 -risk 稳定排序。
        2. 截取前 k 个下标。
    """
    # k<=0 时没有选择。
    if int(k) <= 0:
        return []
    # 稳定排序，risk 大的排在前面。
    return np.argsort(-np.asarray(risk, dtype=np.float64).reshape(-1), kind="mergesort")[: int(k)].astype(int).tolist()


def select_bottom_risk_sensors(risk: np.ndarray, k: int) -> list[int]:
    """
    作用:
        选择 risk 最小的 k 个 sensor 下标。
    参数:
        risk: sensor 风险数组，shape=(N,)。
        k: 需要选择的 sensor 数。
    输入输出:
        输入 risk 数组；输出按 risk 升序排列的 list[int]。
    核心步骤:
        1. 对 risk 稳定排序。
        2. 截取前 k 个下标。
    """
    # k<=0 时没有选择。
    if int(k) <= 0:
        return []
    # 稳定排序，risk 小的排在前面。
    return np.argsort(np.asarray(risk, dtype=np.float64).reshape(-1), kind="mergesort")[: int(k)].astype(int).tolist()


def flatten_h_index(sensor_idx: int, channel_idx: int, env) -> int:
    """
    作用:
        将 H 矩阵下标转换为 state 中 H 段的展平下标。
    参数:
        sensor_idx: sensor 下标，范围 [0,N)。
        channel_idx: channel 下标，范围 [0,M)。
        env: 环境实例，需要提供 n 和 m。
    输入输出:
        输入二维链路下标；输出 row-major 展平下标 int。
    核心步骤:
        1. 校验 sensor 和 channel 下标范围。
        2. 计算 sensor_idx * M + channel_idx。
    """
    # sensor 下标转为 Python int。
    sensor_i = int(sensor_idx)
    # channel 下标转为 Python int。
    channel_i = int(channel_idx)
    # sensor 下标必须合法。
    if sensor_i < 0 or sensor_i >= int(env.n):
        raise IndexError(f"sensor_idx out of range: {sensor_idx}")
    # channel 下标必须合法。
    if channel_i < 0 or channel_i >= int(env.m):
        raise IndexError(f"channel_idx out of range: {channel_idx}")
    # row-major 展平规则。
    return sensor_i * int(env.m) + channel_i


def unflatten_h_index(flat_idx: int, env) -> tuple[int, int]:
    """
    作用:
        将 H 展平下标还原为二维链路下标。
    参数:
        flat_idx: H 段展平下标，范围 [0,N*M)。
        env: 环境实例，需要提供 n 和 m。
    输入输出:
        输入展平下标；输出 (sensor_idx, channel_idx)。
    核心步骤:
        1. 校验 flat_idx 范围。
        2. 使用 divmod(flat_idx, M) 还原二维下标。
    """
    # 展平下标转为 Python int。
    flat_i = int(flat_idx)
    # H 段总长度。
    total = int(env.n) * int(env.m)
    # 展平下标必须合法。
    if flat_i < 0 or flat_i >= total:
        raise IndexError(f"flat_idx out of range: {flat_idx}")
    # divmod 按 row-major 规则还原二维下标。
    return divmod(flat_i, int(env.m))


def decode_eval_action_to_assignment(algo: str, action, env) -> tuple[int, ...]:
    """
    作用:
        将 eval 阶段动作统一转换为 assignment tuple。
    参数:
        algo: 算法名，支持 DQN 或 DDPG。
        action: DQN 为 action_id；DDPG 为 assignment 序列。
        env: 环境实例；DQN 需要 action_space，DDPG 需要 n。
    输入输出:
        输入算法动作；输出 tuple[int,...]，长度为 N，0 表示未调度，1..M 表示信道编号。
    核心步骤:
        1. DQN 通过 env.action_space[action_id] 解码。
        2. DDPG 直接规范化为 tuple[int,...]。
        3. 校验 assignment 长度。
    """
    # 算法名统一大写。
    algo_upper = str(algo).upper()
    # DQN 需要通过 action_space 从 action_id 解码。
    if algo_upper == "DQN":
        if not getattr(env, "action_space", None):
            raise RuntimeError("DQN action_space is not built; cannot decode action_id.")
        action_id = int(action)
        if action_id < 0 or action_id >= len(env.action_space):
            raise IndexError(f"DQN action_id out of range: {action_id}")
        assignment = tuple(int(x) for x in env.action_space[action_id])
    # DDPG 已经返回 assignment，不做 normalize，不查 action_space。
    elif algo_upper == "DDPG":
        assignment = tuple(int(x) for x in np.asarray(action, dtype=np.int64).reshape(-1).tolist())
    # 其他算法不是当前评估脚本支持范围。
    else:
        raise ValueError("algo must be 'DQN' or 'DDPG'")
    # assignment 长度必须等于 sensor 数。
    if len(assignment) != int(env.n):
        raise ValueError(f"assignment length must be n={env.n}, got {len(assignment)}.")
    # 返回统一格式。
    return assignment


def _sensor_mse_at(sensor_idx: int, aoi_value: int, env) -> float:
    """
    作用:
        查询单个 sensor 在指定 AoI 下的 MSE。
    参数:
        sensor_idx: sensor 下标。
        aoi_value: 需要查询的整数 AoI。
        env: 环境实例，需要提供 MSE 计算字段。
    输入输出:
        输入 sensor 下标和 AoI；输出该 sensor 的 MSE float。
    核心步骤:
        1. 构造只用于查询的 AoI 向量。
        2. 调用 compute_sensor_mse_values。
        3. 取出目标 sensor 的 MSE。
    """
    # 构造查询向量，其他 sensor 用 AoI=1 作为占位。
    query = np.ones(int(env.n), dtype=np.float64)
    # 设置目标 sensor 的 AoI。
    query[int(sensor_idx)] = int(aoi_value)
    # 返回目标 sensor 对应的 MSE。
    return float(compute_sensor_mse_values(query, env)[int(sensor_idx)])


def estimate_one_step_expected_cost(state: np.ndarray, env, algo: str, action, cost_mode: str = "mse") -> float:
    """
    作用:
        基于真实 state 和某个 action 估计下一步期望代价。
    参数:
        state: 真实环境状态，shape=(N+N*M,)。
        env: 环境实例，需要提供 n、m、max_aoi 和 MSE 计算字段。
        algo: 算法名，支持 DQN 或 DDPG。
        action: DQN action_id 或 DDPG assignment。
        cost_mode: 代价模式，支持 mse 或 sum_aoi。
    输入输出:
        输入真实 state 与动作；输出下一步期望总代价 float。
    核心步骤:
        1. 用真实 state 拆分 AoI/H。
        2. 将动作解码为 assignment。
        3. 对每个 sensor 按调度成功率累加期望 MSE 或 SumAoI。
    """
    # 读取并校验代价模式。
    mode = str(cost_mode)
    if mode not in {"mse", "sum_aoi"}:
        raise ValueError("cost_mode must be 'mse' or 'sum_aoi'")
    # expected-cost 必须基于真实 state，而不是被攻击候选 state。
    aoi_part, h_part = split_state(np.asarray(state, dtype=np.float32).reshape(-1), env)
    # 将 DQN/DDPG 动作统一转成 assignment。
    assignment = decode_eval_action_to_assignment(algo, action, env)
    # H 恢复为矩阵，便于读取被分配信道的 H。
    h_matrix = np.asarray(h_part, dtype=np.float64).reshape(int(env.n), int(env.m))
    # 初始化总代价。
    expected_cost = 0.0
    # 遍历每个 sensor。
    for sensor_idx, channel_id in enumerate(assignment):
        # 当前 AoI 裁剪到合法范围。
        current_aoi = int(np.clip(np.rint(float(aoi_part[sensor_idx])), 1, int(env.max_aoi)))
        # 失败或未调度时 AoI 增加 1。
        fail_aoi = min(current_aoi + 1, int(env.max_aoi))
        # 未调度时下一步确定为失败 AoI。
        if int(channel_id) == 0:
            expected_cost += float(fail_aoi) if mode == "sum_aoi" else _sensor_mse_at(sensor_idx, fail_aoi, env)
            continue
        # assignment 的信道编号为 1..M，矩阵下标为 0..M-1。
        channel_idx = int(channel_id) - 1
        # 信道编号必须合法。
        if channel_idx < 0 or channel_idx >= int(env.m):
            raise ValueError(f"assignment channel id out of range: {channel_id}")
        # 根据真实 H 计算成功率。
        success_prob = float(h_to_success_prob(h_matrix[sensor_idx, channel_idx]))
        # sum_aoi 模式按成功重置为 1、失败变为 fail_aoi 加权。
        if mode == "sum_aoi":
            expected_cost += success_prob * 1.0 + (1.0 - success_prob) * float(fail_aoi)
        # mse 模式按成功 MSE(AoI=1) 和失败 MSE(fail_aoi) 加权。
        else:
            success_mse = _sensor_mse_at(sensor_idx, 1, env)
            fail_mse = _sensor_mse_at(sensor_idx, fail_aoi, env)
            expected_cost += success_prob * success_mse + (1.0 - success_prob) * fail_mse
    # 返回 Python float，方便排序和 JSON 序列化。
    return float(expected_cost)


def score_candidate_state_by_expected_cost(candidate_state: np.ndarray, real_state: np.ndarray, env, algo: str, agent, cost_mode: str) -> tuple[float, object]:
    """
    作用:
        为一个 attacked_state 候选计算 victim policy 的 expected-cost 分数。
    参数:
        candidate_state: 攻击者构造的观测候选 state。
        real_state: 真实环境 state，用于代价估计。
        env: 环境实例。
        algo: 算法名，支持 DQN 或 DDPG。
        agent: 被攻击的 victim agent。
        cost_mode: 代价模式，支持 mse 或 sum_aoi。
    输入输出:
        输入候选 state 与真实 state；输出 (score, candidate_action)。
    核心步骤:
        1. 让 victim agent 在 candidate_state 上选择动作。
        2. 用 real_state 和该动作估计下一步 expected cost。
        3. 返回分数和候选诱导动作。
    """
    # victim policy 只观察候选 attacked_state。
    candidate_action = select_action_for_eval(algo, candidate_state, agent, env)
    # 代价估计必须用真实 state 和候选诱导出的 action。
    score = estimate_one_step_expected_cost(real_state, env, algo, candidate_action, cost_mode=cost_mode)
    # 返回分数和候选动作，便于记录解释字段。
    return float(score), candidate_action
