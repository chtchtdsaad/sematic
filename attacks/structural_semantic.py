"""
结构化语义攻击模块（structural_semantic.py）
========================================
本文件保存 AoI / H / Joint 结构化状态观测误导攻击。
攻击函数只返回被扰动的 state 副本，不写回 env.aoi、env.channel_state 或 env.channel_loss。
"""

# 启用前向类型注解，避免运行时解析尚未定义的类型。
from __future__ import annotations

# 导入 ceil，用于把结构化预算拆成 suppression / attraction 两部分。
from math import ceil

# 导入 numpy，所有状态向量、AoI、H 和风险计算都使用 numpy 数组。
import numpy as np

# 复用 random 攻击模块里已经稳定的预算判断、统计构造和配置读取逻辑。
from attacks.random_semantic import (
    _build_aoi_info,      # 根据扰动前后 AoI 计算 L0/L1/Linf 和变化下标。
    _build_h_info,        # 根据扰动前后 H 计算 L0/L1/Linf 和变化下标。
    _can_attack,          # 判断当前 step 是否满足攻击概率、总预算和连续攻击约束。
    _empty_perturb_info,  # 构造未攻击时的标准空统计字典。
    _get_config_value,    # 兼容 AttackConfig 和测试 SimpleNamespace 的配置读取。
    _merge_info,          # 合并 AoI/H 扰动统计，得到统一 perturb_info。
)
# 复用状态拆分、拼接和合法值投影函数，保证格式与 random 攻击一致。
from attacks.state_ops import merge_state, project_aoi_part, project_h_part, split_state
# 读取 H 离散状态到丢包率的映射表。
from config import PACKET_LOSS_LEVELS
# 复用核心 MSE 计算函数，不在攻击模块里重复实现 Riccati/MSE 递推。
from core import compute_mse_from_aoi


def compute_sensor_mse_values(aoi_part: np.ndarray, env) -> np.ndarray:
    """
    作用:
        根据当前 AoI 逐 sensor 计算 Tr(P_n(tau_n))。
    输入格式:
        aoi_part: np.ndarray, shape=(N,)。
        env: SemanticSchedulingEnv，需包含系统矩阵、P_bar 和 MSE 缓存。
    输出格式:
        mse_values: np.ndarray[float64], shape=(N,)。
    核心步骤:
        1. 将 AoI round + clip 到 [1, env.max_aoi]。
        2. 对每个 sensor 复用 core.compute_mse_from_aoi 和 env 缓存。
        3. 返回逐 sensor MSE 数组。
    """
    # 将输入 AoI 转为一维 float64，再按整数 AoI 语义 round，并裁剪到合法范围。
    aoi_arr = np.clip(np.rint(np.asarray(aoi_part, dtype=np.float64).reshape(-1)), 1, int(env.max_aoi)).astype(int)
    # 防止状态切片长度和环境 sensor 数不一致，避免后续按错 sensor 计算风险。
    if aoi_arr.size != int(env.n):
        raise ValueError(f"aoi_part length must be n={env.n}, got {aoi_arr.size}.")

    # 预分配逐 sensor MSE 数组，dtype 用 float64 保留数值精度。
    mse_values = np.zeros(int(env.n), dtype=np.float64)
    # 遍历每个 sensor，分别用该 sensor 自己的系统矩阵和缓存计算 Tr(P)。
    for sensor_idx in range(int(env.n)):
        # 复用 env 中为当前 sensor 预计算好的 A^k 和噪声累计缓存，避免重复矩阵幂运算。
        mse_values[sensor_idx] = compute_mse_from_aoi(
            p_bar=env.p_bars[sensor_idx],                       # 当前 sensor 的稳态 Kalman 协方差。
            a=env.a_mats[sensor_idx],                            # 当前 sensor 的系统矩阵 A。
            w=env.w_mats[sensor_idx],                            # 当前 sensor 的过程噪声协方差 W。
            aoi=int(aoi_arr[sensor_idx]),                        # 当前 sensor 的整数 AoI。
            a_powers=env.a_powers_cache[sensor_idx],             # 当前 sensor 的 A^k 缓存。
            noise_cov_sums=env.noise_cov_sums_cache[sensor_idx], # 当前 sensor 的噪声累计项缓存。
        )
    # 返回 shape=(N,) 的逐 sensor MSE，用于后续 risk 计算。
    return mse_values


def compute_sensor_risk(aoi_part: np.ndarray, env, risk_mode: str = "reset_gain") -> np.ndarray:
    """
    作用:
        根据 AoI 计算 sensor risk，用于选择高风险 sensor 和低风险诱饵 sensor。
    输入格式:
        aoi_part: np.ndarray, shape=(N,)。
        env: SemanticSchedulingEnv。
        risk_mode: str，支持 current_mse / marginal_growth / reset_gain。
    输出格式:
        risk: np.ndarray[float64], shape=(N,)。
    核心步骤:
        1. current_mse 直接使用当前 MSE。
        2. marginal_growth 使用 AoI+1 的 MSE 增量。
        3. reset_gain 使用当前 MSE 与 AoI=1 MSE 的差值。
    """
    # 标准化 AoI，保证 risk 计算只基于合法整数 AoI。
    aoi_arr = np.clip(np.rint(np.asarray(aoi_part, dtype=np.float64).reshape(-1)), 1, int(env.max_aoi)).astype(int)
    # 先计算当前 AoI 下每个 sensor 的 MSE，这是三种 risk 模式的共同基础。
    current_mse = compute_sensor_mse_values(aoi_arr, env)
    # 将 risk_mode 转成字符串，兼容枚举或其他可字符串化对象。
    mode = str(risk_mode)

    # current_mse 模式：直接把当前估计误差当成风险。
    if mode == "current_mse":
        return current_mse
    # marginal_growth 模式：风险等于下一步不调度时 MSE 的边际增长。
    if mode == "marginal_growth":
        # AoI+1 不能超过 env.max_aoi，因此先做上界截断。
        next_aoi = np.minimum(aoi_arr + 1, int(env.max_aoi))
        # 返回每个 sensor 在 AoI 增长一步后的 MSE 增量。
        return compute_sensor_mse_values(next_aoi, env) - current_mse
    # reset_gain 模式：风险等于如果成功重置到 AoI=1 可减少的 MSE。
    if mode == "reset_gain":
        # 计算所有 sensor 在 AoI=1 时的基准 MSE。
        reset_mse = compute_sensor_mse_values(np.ones(int(env.n), dtype=np.float64), env)
        # 当前 MSE 减去重置后 MSE，表示调度成功的潜在收益。
        return current_mse - reset_mse

    # 未知 risk 模式直接报错，避免静默使用错误指标。
    raise ValueError(f"unsupported risk_mode: {risk_mode}")


def h_to_success_prob(h_part: np.ndarray) -> np.ndarray:
    """
    作用:
        将离散 H 状态索引映射为链路成功率。
    输入格式:
        h_part: np.ndarray, shape=(N*M,) 或 (N,M)。
    输出格式:
        success_prob: np.ndarray[float64]，shape 与输入一致。
    核心步骤:
        1. 将 H round + clip 到 [0,4] 后转 int。
        2. 用 config.PACKET_LOSS_LEVELS 查表得到丢包率。
        3. 返回 success_prob = 1 - packet_loss。
    """
    # H 是离散状态索引，先 round 再裁剪到 PACKET_LOSS_LEVELS 的合法索引范围。
    h_index = np.clip(np.rint(np.asarray(h_part, dtype=np.float64)), 0, len(PACKET_LOSS_LEVELS) - 1).astype(int)
    # 成功率 = 1 - 丢包率，输出 shape 与输入 H 保持一致。
    return (1.0 - PACKET_LOSS_LEVELS[h_index]).astype(np.float64, copy=False)


def compute_link_priority(
    aoi_part: np.ndarray,
    h_part: np.ndarray,
    env,
    risk_mode: str = "reset_gain",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    作用:
        计算 priority_{n,m}=risk_n * p_success(H_{n,m})。
    输入格式:
        aoi_part: np.ndarray, shape=(N,)。
        h_part: np.ndarray, shape=(N*M,)。
        env: SemanticSchedulingEnv。
        risk_mode: str，risk 计算模式。
    输出格式:
        priority: np.ndarray[float64], shape=(N,M)。
        risk: np.ndarray[float64], shape=(N,)。
        success_prob: np.ndarray[float64], shape=(N,M)。
    核心步骤:
        1. 计算 sensor risk。
        2. 将 H 展平段 reshape 为 (N,M) 并映射为成功率。
        3. 逐 sensor 广播相乘得到链路优先级。
    """
    # 先按 AoI 计算 sensor 级风险，shape=(N,)。
    risk = compute_sensor_risk(aoi_part, env, risk_mode=risk_mode)
    # 将展平 H 段恢复成链路矩阵，shape=(N,M)。
    h_matrix = np.asarray(h_part, dtype=np.float64).reshape(int(env.n), int(env.m))
    # 将每条 sensor-channel 链路的 H 状态映射成成功率，shape=(N,M)。
    success_prob = h_to_success_prob(h_matrix)
    # 广播 risk[:,None]，得到每条链路的结构优先级 risk_n * p_success(n,m)。
    priority = risk[:, None] * success_prob
    # 返回 priority、sensor risk 和 success_prob，供 H 攻击和 Joint 攻击复用。
    return priority.astype(np.float64, copy=False), risk, success_prob


def select_top_risk_sensors(risk: np.ndarray, k: int) -> list[int]:
    """
    作用:
        返回 risk 最大的 sensor 下标列表。
    输入格式:
        risk: np.ndarray, shape=(N,)。
        k: int，需要选择的数量。
    输出格式:
        list[int]，按 risk 从大到小排列。
    核心步骤:
        1. 使用稳定排序。
        2. 截取前 k 个下标。
    """
    # k<=0 时不选择任何 sensor，直接返回空列表。
    if int(k) <= 0:
        return []
    # 将 risk 标准化为一维 float64 数组，便于稳定排序。
    risk_arr = np.asarray(risk, dtype=np.float64).reshape(-1)
    # 对 -risk 升序排序等价于按 risk 降序排序，mergesort 保证同分时顺序稳定。
    return np.argsort(-risk_arr, kind="mergesort")[: int(k)].astype(int).tolist()


def select_bottom_risk_sensors(risk: np.ndarray, k: int) -> list[int]:
    """
    作用:
        返回 risk 最小的 sensor 下标列表。
    输入格式:
        risk: np.ndarray, shape=(N,)。
        k: int，需要选择的数量。
    输出格式:
        list[int]，按 risk 从小到大排列。
    核心步骤:
        1. 使用稳定排序。
        2. 截取前 k 个下标。
    """
    # k<=0 时不选择任何 sensor，直接返回空列表。
    if int(k) <= 0:
        return []
    # 将 risk 标准化为一维 float64 数组，便于稳定排序。
    risk_arr = np.asarray(risk, dtype=np.float64).reshape(-1)
    # 按 risk 从小到大排序，取前 k 个作为低风险诱饵 sensor。
    return np.argsort(risk_arr, kind="mergesort")[: int(k)].astype(int).tolist()


def flatten_h_index(sensor_idx: int, channel_idx: int, env) -> int:
    """
    作用:
        将 H 矩阵下标 (sensor_idx, channel_idx) 转换为展平下标。
    输入格式:
        sensor_idx: int，范围 [0,N)。
        channel_idx: int，范围 [0,M)。
        env: SemanticSchedulingEnv，需包含 n / m。
    输出格式:
        int，范围 [0,N*M)。
    核心步骤:
        1. 校验 sensor 与 channel 下标。
        2. 使用 row-major 顺序计算 sensor_idx * M + channel_idx。
    """
    # 将输入下标转成 Python int，避免 numpy 标量影响比较和返回类型。
    sensor_i = int(sensor_idx)
    # 将 channel 下标也转成 Python int。
    channel_i = int(channel_idx)
    # sensor 下标必须落在 [0,N)。
    if sensor_i < 0 or sensor_i >= int(env.n):
        raise IndexError(f"sensor_idx out of range: {sensor_idx}")
    # channel 下标必须落在 [0,M)。
    if channel_i < 0 or channel_i >= int(env.m):
        raise IndexError(f"channel_idx out of range: {channel_idx}")
    # row-major 展平规则：第 sensor_i 行偏移 sensor_i*M，再加 channel_i。
    return sensor_i * int(env.m) + channel_i


def unflatten_h_index(flat_idx: int, env) -> tuple[int, int]:
    """
    作用:
        将 H 展平下标转换为矩阵下标 (sensor_idx, channel_idx)。
    输入格式:
        flat_idx: int，范围 [0,N*M)。
        env: SemanticSchedulingEnv，需包含 n / m。
    输出格式:
        tuple[int,int]。
    核心步骤:
        1. 校验展平下标。
        2. 使用 divmod 按 row-major 顺序还原下标。
    """
    # 将展平下标转成 Python int，避免 numpy 标量影响 divmod 返回。
    flat_i = int(flat_idx)
    # H 展平段总长度为 N*M。
    total = int(env.n) * int(env.m)
    # 展平下标必须落在 [0,N*M)。
    if flat_i < 0 or flat_i >= total:
        raise IndexError(f"flat_idx out of range: {flat_idx}")
    # divmod(flat_i, M) 还原出 (sensor_idx, channel_idx)。
    return divmod(flat_i, int(env.m))


def _split_suppress_attract_budget(total_budget: int) -> tuple[int, int]:
    """
    作用:
        将结构化攻击预算拆成 suppression 和 attraction 两部分。
    输入格式:
        total_budget: int，本段最多可扰动维度数。
    输出格式:
        tuple[int,int]，分别为 suppression_budget 和 attraction_budget。
    核心步骤:
        1. B<=0 时返回 0。
        2. B=1 时只做 suppression。
        3. B>=2 时至少保留一个 attraction，其余按 60/40 近似拆分。
    """
    # 先把预算转成非负整数，负数预算按 0 处理。
    budget = max(0, int(total_budget))
    # 没有预算时 suppression 和 attraction 都为 0。
    if budget <= 0:
        return 0, 0
    # 单维预算无法同时做 suppression 和 attraction，因此只做 suppression。
    if budget == 1:
        return 1, 0
    # B>=2 时至少给 attraction 一个维度，保证存在诱饵扰动。
    attraction_budget = max(1, int(budget - ceil(0.6 * budget)))
    # 剩余预算全部给 suppression，保持高风险隐藏为主。
    suppression_budget = budget - attraction_budget
    # 返回两个互斥预算。
    return suppression_budget, attraction_budget


def _select_aoi_indices(
    aoi_part: np.ndarray,
    risk: np.ndarray,
    env,
    budget: int,
) -> tuple[list[int], list[int]]:
    """
    作用:
        选择 AoI suppression 和 attraction 的实际可改变 sensor 下标。
    输入格式:
        aoi_part: np.ndarray[float32], shape=(N,)。
        risk: np.ndarray[float64], shape=(N,)。
        env: SemanticSchedulingEnv。
        budget: int，AoI 扰动预算。
    输出格式:
        tuple[list[int], list[int]]，高风险降低下标与低风险抬高下标。
    核心步骤:
        1. suppression 优先选择高 risk 且 AoI>1 的 sensor。
        2. attraction 优先选择低 risk 且 AoI<max_aoi 的 sensor。
        3. 若 suppression 因边界不足，将剩余预算补给 attraction，反之亦然。
    """
    # AoI 最多只能扰动 N 个 sensor，因此把预算裁剪到 [0,N]。
    budget = min(max(0, int(budget)), int(env.n))
    # 将预算拆成高风险隐藏和低风险诱饵两部分。
    suppress_budget, attract_budget = _split_suppress_attract_budget(budget)
    # 将 AoI 标准化为一维 float32，后续只读不写。
    aoi_arr = np.asarray(aoi_part, dtype=np.float32).reshape(-1)

    # 高风险顺序：risk 越大越优先被降低 AoI。
    high_order = np.argsort(-np.asarray(risk, dtype=np.float64), kind="mergesort").astype(int).tolist()
    # 低风险顺序：risk 越小越优先被抬高 AoI 作为诱饵。
    low_order = np.argsort(np.asarray(risk, dtype=np.float64), kind="mergesort").astype(int).tolist()

    # 记录最终用于 AoI 降低的 sensor 下标。
    suppress_indices: list[int] = []
    # used 防止同一个 sensor 同时被降低和抬高。
    used: set[int] = set()
    # 按高风险顺序尝试填满 suppression 预算。
    for idx in high_order:
        # suppression 已满则停止。
        if len(suppress_indices) >= suppress_budget:
            break
        # AoI 已经是 1 时再降低会被 clip 掉，因此只选 AoI>1 的 sensor。
        if float(aoi_arr[idx]) > 1.0:
            # 记录该 sensor 作为高风险隐藏对象。
            suppress_indices.append(int(idx))
            # 标记该 sensor 已使用，避免进入诱饵集合。
            used.add(int(idx))

    # suppression 未用满的预算可以转给 attraction，避免边界 clip 浪费预算。
    unused_suppress_budget = suppress_budget - len(suppress_indices)
    # attraction 目标预算 = 原 attraction 预算 + suppression 剩余预算。
    target_attract_budget = attract_budget + unused_suppress_budget
    # 记录最终用于 AoI 抬高的低风险诱饵 sensor 下标。
    attract_indices: list[int] = []
    # 按低风险顺序尝试填满 attraction 预算。
    for idx in low_order:
        # attraction 已满则停止。
        if len(attract_indices) >= target_attract_budget:
            break
        # 已使用的 sensor 跳过；AoI 已到 max_aoi 时再抬高也会被 clip 掉。
        if int(idx) not in used and float(aoi_arr[idx]) < float(env.max_aoi):
            # 记录该 sensor 作为低风险诱饵。
            attract_indices.append(int(idx))
            # 标记该 sensor 已使用。
            used.add(int(idx))

    # 如果 attraction 也没用满预算，再回到高风险列表补充可降低的 sensor。
    remaining_budget = budget - len(suppress_indices) - len(attract_indices)
    # 按高风险顺序继续找未使用且可降低的 sensor。
    for idx in high_order:
        # 没有剩余预算时停止。
        if remaining_budget <= 0:
            break
        # 只选择未使用且 AoI>1 的 sensor，保证扰动后能实际变化。
        if int(idx) not in used and float(aoi_arr[idx]) > 1.0:
            # 加入 suppression 集合。
            suppress_indices.append(int(idx))
            # 标记已使用。
            used.add(int(idx))
            # 消耗一个剩余预算。
            remaining_budget -= 1

    # 返回两组互斥下标，调用方负责真正施加 +/- delta。
    return suppress_indices, attract_indices


def _select_h_indices(
    h_part: np.ndarray,
    priority: np.ndarray,
    risk: np.ndarray,
    env,
    budget: int,
) -> tuple[list[int], list[int]]:
    """
    作用:
        选择 H suppression 和 attraction 的实际可改变链路展平下标。
    输入格式:
        h_part: np.ndarray[float32], shape=(N*M,)。
        priority: np.ndarray[float64], shape=(N,M)。
        risk: np.ndarray[float64], shape=(N,)。
        env: SemanticSchedulingEnv。
        budget: int，H 扰动预算。
    输出格式:
        tuple[list[int], list[int]]，高 priority 降低下标与低风险诱饵抬高下标。
    核心步骤:
        1. suppression 按 priority 从高到低选择 H>0 的链路。
        2. attraction 按 sensor risk 从低到高、H 从低到高选择 H<4 的链路。
        3. 若 suppression 因 clip 不足，将剩余预算补给 attraction。
    """
    # 将 H 展平段标准化为一维 float32，后续只读不写。
    h_arr = np.asarray(h_part, dtype=np.float32).reshape(-1)
    # H 最多只能扰动 N*M 个链路维度，因此把预算裁剪到 [0,N*M]。
    budget = min(max(0, int(budget)), h_arr.size)
    # 将 H 预算拆成高 priority 压制和低风险诱饵抬高两部分。
    suppress_budget, attract_budget = _split_suppress_attract_budget(budget)

    # 将 priority 矩阵展平成一维，展平顺序与 state 中 H 段一致。
    priority_flat = np.asarray(priority, dtype=np.float64).reshape(-1)
    # 高 priority 链路优先被降低 H。
    high_order = np.argsort(-priority_flat, kind="mergesort").astype(int).tolist()

    # 记录最终用于 H 降低的展平链路下标。
    suppressed: list[int] = []
    # used 防止同一条链路同时被降低和抬高。
    used: set[int] = set()
    # 按高 priority 顺序尝试填满 suppression 预算。
    for flat_idx in high_order:
        # suppression 已满则停止。
        if len(suppressed) >= suppress_budget:
            break
        # H 已经为 0 时再降低会被 clip 掉，因此只选 H>0 的链路。
        if float(h_arr[flat_idx]) > 0.0:
            # 记录该链路作为高 priority 压制对象。
            suppressed.append(int(flat_idx))
            # 标记该链路已使用。
            used.add(int(flat_idx))

    # 候选格式为 (risk, H, sensor_idx, channel_idx, flat_idx)，便于按低风险和低 H 排序。
    decoy_candidates: list[tuple[float, float, int, int, int]] = []
    # 将 H 展平段还原成 (N,M)，便于按 sensor-channel 枚举。
    h_matrix = h_arr.reshape(int(env.n), int(env.m))
    # 枚举所有 sensor。
    for sensor_idx in range(int(env.n)):
        # 枚举当前 sensor 的所有 channel。
        for channel_idx in range(int(env.m)):
            # 将二维链路下标转为 state 中 H 段的展平下标。
            flat_idx = flatten_h_index(sensor_idx, channel_idx, env)
            # 已使用链路跳过；H 已经为 4 时再抬高会被 clip 掉，也跳过。
            if flat_idx in used or float(h_matrix[sensor_idx, channel_idx]) >= 4.0:
                continue
            # 低风险优先；同风险时优先当前 H 更低的诱饵链路。
            decoy_candidates.append(
                (
                    float(risk[sensor_idx]),                   # 第一排序键：sensor risk 越低越优先。
                    float(h_matrix[sensor_idx, channel_idx]),  # 第二排序键：当前 H 越低越适合伪装为好链路。
                    int(sensor_idx),                           # 保留 sensor 下标，方便解释字段。
                    int(channel_idx),                          # 保留 channel 下标，方便解释字段。
                    int(flat_idx),                             # 保留展平下标，方便实际修改 H 段。
                )
            )
    # Python tuple 排序会依次使用 risk、H、sensor、channel、flat_idx，结果稳定可复现。
    decoy_candidates.sort()

    # suppression 未用满的预算转给 attraction，避免 H=0 边界浪费预算。
    unused_suppress_budget = suppress_budget - len(suppressed)
    # attraction 目标预算 = 原 attraction 预算 + suppression 剩余预算。
    target_attract_budget = attract_budget + unused_suppress_budget
    # 记录最终用于 H 抬高的诱饵链路下标。
    attracted: list[int] = []
    # 按低风险、低 H 顺序选择诱饵链路。
    for _, _, _, _, flat_idx in decoy_candidates:
        # attraction 已满则停止。
        if len(attracted) >= target_attract_budget:
            break
        # 加入 H 抬高集合。
        attracted.append(int(flat_idx))
        # 标记该链路已使用。
        used.add(int(flat_idx))

    # 如果 attraction 也没用满预算，再回到高 priority 列表补充可降低链路。
    remaining_budget = budget - len(suppressed) - len(attracted)
    # 按高 priority 顺序继续寻找未使用且可降低的链路。
    for flat_idx in high_order:
        # 没有剩余预算时停止。
        if remaining_budget <= 0:
            break
        # 只选择未使用且 H>0 的链路，保证扰动后能实际变化。
        if int(flat_idx) not in used and float(h_arr[flat_idx]) > 0.0:
            # 加入 suppression 集合。
            suppressed.append(int(flat_idx))
            # 标记已使用。
            used.add(int(flat_idx))
            # 消耗一个剩余预算。
            remaining_budget -= 1

    # 返回两组互斥 H 展平下标，调用方负责真正施加 +/- delta。
    return suppressed, attracted


def _apply_aoi_mislead(aoi_part: np.ndarray, env, config, budget: int) -> tuple[np.ndarray, dict]:
    """
    作用:
        按结构化规则生成 AoI-only 误导扰动。
    输入格式:
        aoi_part: np.ndarray[float32], shape=(N,)。
        env: SemanticSchedulingEnv。
        config: AttackConfig。
        budget: int，AoI 扰动预算。
    输出格式:
        attacked_aoi: np.ndarray[float32], shape=(N,)。
        info: dict，AoI 扰动统计和结构化下标信息。
    核心步骤:
        1. 计算 reset_gain risk。
        2. 高风险 sensor AoI 降低。
        3. 低风险诱饵 sensor AoI 提高。
        4. 投影后按实际差异统计。
    """
    # 复制原始 AoI，后续所有修改都发生在副本上。
    original_aoi = np.asarray(aoi_part, dtype=np.float32).reshape(-1).copy()
    # attacked_aoi 是本函数要返回的观测扰动结果，不会写回 env。
    attacked_aoi = original_aoi.copy()
    # 第一轮固定使用 reset_gain 作为 sensor risk。
    risk = compute_sensor_risk(original_aoi, env, risk_mode="reset_gain")
    # 根据 risk 和边界可变性选择高风险降低、低风险抬高的 sensor。
    suppress_indices, attract_indices = _select_aoi_indices(original_aoi, risk, env, budget)
    # 读取 AoI 单维扰动幅值，兼容 AttackConfig 和测试配置对象。
    aoi_delta = int(round(float(_get_config_value(config, "aoi_delta", 1))))

    # delta<=0 时不施加任何扰动，后续统计会自然得到 total_l0=0。
    if aoi_delta > 0:
        # 高风险 sensor 的观测 AoI 降低，用于隐藏真实紧迫性。
        for idx in suppress_indices:
            attacked_aoi[idx] = attacked_aoi[idx] - float(aoi_delta)
        # 低风险 sensor 的观测 AoI 抬高，用于制造诱饵紧迫性。
        for idx in attract_indices:
            attacked_aoi[idx] = attacked_aoi[idx] + float(aoi_delta)

    # AoI 必须保持整数语义，并裁剪到 [1, env.max_aoi]。
    attacked_aoi = project_aoi_part(attacked_aoi, env)
    # 基于投影后的实际差异计算 AoI L0/L1/Linf，clip 掉的维度不会被算入 changed。
    info = _build_aoi_info(attacked_aoi, original_aoi)
    # 记录结构化解释字段：哪些 sensor 被当作高风险隐藏对象。
    info["high_risk_sensors"] = [int(x) for x in suppress_indices]
    # 记录结构化解释字段：哪些 sensor 被当作低风险诱饵对象。
    info["low_risk_sensors"] = [int(x) for x in attract_indices]
    # 返回 AoI 副本和统计信息。
    return attacked_aoi.astype(np.float32, copy=True), info


def _apply_h_mislead(h_part: np.ndarray, aoi_part: np.ndarray, env, config, budget: int) -> tuple[np.ndarray, dict]:
    """
    作用:
        按结构化规则生成 H-only 误导扰动。
    输入格式:
        h_part: np.ndarray[float32], shape=(N*M,)。
        aoi_part: np.ndarray[float32], shape=(N,)。
        env: SemanticSchedulingEnv。
        config: AttackConfig。
        budget: int，H 扰动预算。
    输出格式:
        attacked_h: np.ndarray[float32], shape=(N*M,)。
        info: dict，H 扰动统计和结构化链路信息。
    核心步骤:
        1. 计算 priority = risk * success_prob。
        2. 高 priority 链路 H 降低。
        3. 低风险诱饵链路 H 提高。
        4. 投影后按实际差异统计。
    """
    # 复制原始 H 展平段，后续所有修改都发生在副本上。
    original_h = np.asarray(h_part, dtype=np.float32).reshape(-1).copy()
    # attacked_h 是本函数要返回的观测扰动结果，不会写回 env。
    attacked_h = original_h.copy()
    # 计算链路优先级 priority=risk*success_prob，并同时得到 sensor risk。
    priority, risk, _ = compute_link_priority(aoi_part, original_h, env, risk_mode="reset_gain")
    # 根据 priority、risk 和边界可变性选择 H 降低链路与诱饵抬高链路。
    suppressed, attracted = _select_h_indices(original_h, priority, risk, env, budget)
    # 读取 H 单维扰动幅值，兼容 AttackConfig 和测试配置对象。
    h_delta = int(round(float(_get_config_value(config, "h_delta", 1))))

    # delta<=0 时不施加任何扰动，后续统计会自然得到 total_l0=0。
    if h_delta > 0:
        # 高 priority 链路的观测 H 降低，用于把真实好链路伪装得更差。
        for flat_idx in suppressed:
            attacked_h[flat_idx] = attacked_h[flat_idx] - float(h_delta)
        # 低风险诱饵链路的观测 H 抬高，用于把诱饵链路伪装得更好。
        for flat_idx in attracted:
            attacked_h[flat_idx] = attacked_h[flat_idx] + float(h_delta)

    # H 必须保持离散索引语义，并裁剪到 [0,4]。
    attacked_h = project_h_part(attacked_h)
    # 基于投影后的实际差异计算 H L0/L1/Linf，clip 掉的维度不会被算入 changed。
    info = _build_h_info(attacked_h, original_h)
    # 记录结构化解释字段：被压制的 H 展平下标。
    info["suppressed_h_indices"] = [int(x) for x in suppressed]
    # 记录结构化解释字段：被抬高的 H 展平下标。
    info["attracted_h_indices"] = [int(x) for x in attracted]
    # 将被压制链路转换成 (sensor,channel)，方便读报告时理解。
    info["high_priority_links"] = [tuple(int(v) for v in unflatten_h_index(x, env)) for x in suppressed]
    # 将诱饵链路转换成 (sensor,channel)，方便读报告时理解。
    info["decoy_links"] = [tuple(int(v) for v in unflatten_h_index(x, env)) for x in attracted]
    # 返回 H 副本和统计信息。
    return attacked_h.astype(np.float32, copy=True), info


def _finalize_structural_attack(
    attacked_aoi: np.ndarray,
    attacked_h: np.ndarray,
    original_aoi: np.ndarray,
    original_h: np.ndarray,
    attack_state: dict,
    structural_attack_type: str,
    extra_info: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    作用:
        合并 attacked_state、扰动统计和 attack_state 计数更新。
    输入格式:
        attacked_aoi: np.ndarray[float32], shape=(N,)。
        attacked_h: np.ndarray[float32], shape=(N*M,)。
        original_aoi: np.ndarray[float32], shape=(N,)。
        original_h: np.ndarray[float32], shape=(N*M,)。
        attack_state: dict，episode 内攻击状态。
        structural_attack_type: str，攻击模式名。
        extra_info: dict | None，结构化解释字段。
    输出格式:
        attacked_state: np.ndarray[float32], shape=(N+N*M,)。
        perturb_info: dict。
    核心步骤:
        1. 按投影后实际差异重算 AoI/H 统计。
        2. 合并为标准扰动统计。
        3. 只有实际 total_l0>0 时更新 attack_state。
    """
    # 重新计算 AoI 实际扰动统计，保证统计只反映投影后真实变化。
    aoi_info = _build_aoi_info(attacked_aoi, original_aoi)
    # 重新计算 H 实际扰动统计，保证统计只反映投影后真实变化。
    h_info = _build_h_info(attacked_h, original_h)
    # 合并 AoI/H 统计，并自动得到 total_l0 与 is_attacked。
    perturb_info = _merge_info(aoi_info, h_info)
    # 标明本次扰动来自哪一种结构化攻击模式。
    perturb_info["structural_attack_type"] = structural_attack_type
    # 如果调用方传入结构化解释字段，则合并到 perturb_info。
    if extra_info:
        perturb_info.update(extra_info)

    # 只有实际发生扰动时才消耗 episode 攻击步预算。
    if perturb_info["is_attacked"]:
        # 记录本 episode 已使用攻击步数。
        attack_state["attack_steps_used"] = int(attack_state.get("attack_steps_used", 0)) + 1
        # 记录连续攻击步数，用于 max_consecutive_steps 约束。
        attack_state["consecutive_attack_steps"] = int(attack_state.get("consecutive_attack_steps", 0)) + 1
    # 如果没有实际扰动，则重置连续攻击计数。
    else:
        attack_state["consecutive_attack_steps"] = 0

    # 拼接 AoI 和 H，返回 attacked_state 副本与标准扰动统计。
    return merge_state(attacked_aoi, attacked_h), perturb_info


def semantic_aoi_mislead_attack(state, env, config, rng, attack_state):
    """
    作用:
        生成 AoI-only 结构化误导攻击观测。
    输入格式:
        state: np.ndarray[float32], shape=(N+N*M,)。
        env: SemanticSchedulingEnv。
        config: AttackConfig。
        rng: np.random.Generator。
        attack_state: dict，episode 内攻击预算状态。
    输出格式:
        attacked_state: np.ndarray[float32], shape=(N+N*M,)。
        perturb_info: dict，字段与 random_semantic 对齐，并包含结构化解释字段。
    核心步骤:
        1. 复用 random 攻击触发逻辑。
        2. 高风险 AoI 降低，低风险 AoI 抬高。
        3. 不修改 H 段，不写回 env。
    """
    # 复制输入 state，保证攻击不会原地修改调用方持有的真实状态。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 复用 random 攻击预算判断；不满足触发条件时返回原 state 副本和空统计。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()

    # 拆分状态，AoI-only 攻击只会修改 aoi_part，不会修改 h_part。
    aoi_part, h_part = split_state(state_copy, env)
    # AoI 预算同时受 max_aoi_features、max_total_features 和 sensor 数 N 约束。
    budget = min(
        int(_get_config_value(config, "max_aoi_features", int(env.n))),
        int(_get_config_value(config, "max_total_features", int(env.n))),
        int(env.n),
    )
    # 根据结构化 AoI 规则生成 attacked_aoi 和解释统计。
    attacked_aoi, aoi_extra = _apply_aoi_mislead(aoi_part, env, config, budget)
    # 合并 attacked_state 和统计；H 段保持原样。
    return _finalize_structural_attack(
        attacked_aoi=attacked_aoi,                         # 被扰动后的 AoI 观测。
        attacked_h=h_part,                                 # H 观测保持真实 state 中的原值。
        original_aoi=aoi_part,                             # 原始 AoI，用于统计实际差异。
        original_h=h_part,                                 # 原始 H，用于统计实际差异。
        attack_state=attack_state,                         # episode 内攻击预算状态。
        structural_attack_type="semantic_aoi_mislead",     # 标记攻击类型。
        extra_info=aoi_extra,                              # 附加 high/low risk sensor 解释字段。
    )


def semantic_h_mislead_attack(state, env, config, rng, attack_state):
    """
    作用:
        生成 H-only 结构化误导攻击观测。
    输入格式:
        state: np.ndarray[float32], shape=(N+N*M,)。
        env: SemanticSchedulingEnv。
        config: AttackConfig。
        rng: np.random.Generator。
        attack_state: dict，episode 内攻击预算状态。
    输出格式:
        attacked_state: np.ndarray[float32], shape=(N+N*M,)。
        perturb_info: dict，字段与 random_semantic 对齐，并包含结构化解释字段。
    核心步骤:
        1. 复用 random 攻击触发逻辑。
        2. 高 priority 链路 H 降低，低风险诱饵链路 H 抬高。
        3. 不修改 AoI 段，不写回 env。
    """
    # 复制输入 state，保证攻击不会原地修改调用方持有的真实状态。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 复用 random 攻击预算判断；不满足触发条件时返回原 state 副本和空统计。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()

    # 拆分状态，H-only 攻击只会修改 h_part，不会修改 aoi_part。
    aoi_part, h_part = split_state(state_copy, env)
    # H 预算同时受 max_h_features、max_total_features 和 H 维度数 N*M 约束。
    budget = min(
        int(_get_config_value(config, "max_h_features", h_part.size)),
        int(_get_config_value(config, "max_total_features", h_part.size)),
        h_part.size,
    )
    # 根据结构化 H 规则生成 attacked_h 和解释统计。
    attacked_h, h_extra = _apply_h_mislead(h_part, aoi_part, env, config, budget)
    # 合并 attacked_state 和统计；AoI 段保持原样。
    return _finalize_structural_attack(
        attacked_aoi=aoi_part,                         # AoI 观测保持真实 state 中的原值。
        attacked_h=attacked_h,                         # 被扰动后的 H 观测。
        original_aoi=aoi_part,                         # 原始 AoI，用于统计实际差异。
        original_h=h_part,                             # 原始 H，用于统计实际差异。
        attack_state=attack_state,                     # episode 内攻击预算状态。
        structural_attack_type="semantic_h_mislead",   # 标记攻击类型。
        extra_info=h_extra,                            # 附加 suppressed/decoy link 解释字段。
    )


def semantic_joint_mislead_attack(state, env, config, rng, attack_state):
    """
    作用:
        生成 Joint 结构化误导攻击观测，同时扰动 AoI 和 H。
    输入格式:
        state: np.ndarray[float32], shape=(N+N*M,)。
        env: SemanticSchedulingEnv。
        config: AttackConfig。
        rng: np.random.Generator。
        attack_state: dict，episode 内攻击预算状态。
    输出格式:
        attacked_state: np.ndarray[float32], shape=(N+N*M,)。
        perturb_info: dict，字段与 random_semantic 对齐，并包含结构化解释字段。
    核心步骤:
        1. 复用 random 攻击触发逻辑。
        2. 先按总预算分配 AoI/H 预算。
        3. AoI 与 H 各自执行结构化 suppression/attraction。
        4. 不写回 env。
    """
    # 复制输入 state，保证攻击不会原地修改调用方持有的真实状态。
    state_copy = np.asarray(state, dtype=np.float32).reshape(-1).copy()
    # 复用 random 攻击预算判断；不满足触发条件时返回原 state 副本和空统计。
    if not _can_attack(config, rng, attack_state):
        return state_copy, _empty_perturb_info()

    # 拆分状态，Joint 攻击会分别构造 AoI 和 H 的观测副本。
    aoi_part, h_part = split_state(state_copy, env)
    # 总预算不能超过 max_total_features，也不能超过完整状态维度。
    total_budget = min(
        int(_get_config_value(config, "max_total_features", aoi_part.size + h_part.size)),
        aoi_part.size + h_part.size,
    )
    # AoI 预算按计划先取总预算的 40%，再受 max_aoi_features 和 N 限制。
    aoi_budget = min(
        int(_get_config_value(config, "max_aoi_features", aoi_part.size)),
        int(ceil(0.4 * float(total_budget))),
        aoi_part.size,
    )
    # 先执行 AoI 结构化误导，并得到实际使用的 AoI L0。
    attacked_aoi, aoi_extra = _apply_aoi_mislead(aoi_part, env, config, aoi_budget)

    # H 可用总预算 = 总预算 - AoI 实际变化维度数，避免因 AoI clip 未变化而浪费预算。
    remaining_total = max(0, total_budget - int(aoi_extra["aoi_l0"]))
    # H 预算同时受 max_h_features、剩余总预算和 H 维度数限制。
    h_budget = min(
        int(_get_config_value(config, "max_h_features", h_part.size)),
        remaining_total,
        h_part.size,
    )
    # 再执行 H 结构化误导，并得到 H 解释统计。
    attacked_h, h_extra = _apply_h_mislead(h_part, aoi_part, env, config, h_budget)

    # extra_info 用于把 AoI 和 H 的结构化解释字段放到同一个 perturb_info 中。
    extra_info = {}
    # 写入 AoI 的 high_risk_sensors / low_risk_sensors 等字段。
    extra_info.update(aoi_extra)
    # 写入 H 的 suppressed_h_indices / decoy_links 等字段。
    extra_info.update(h_extra)
    # 合并 attacked_state 和统计；AoI/H 都使用各自扰动后的副本。
    return _finalize_structural_attack(
        attacked_aoi=attacked_aoi,                         # 被扰动后的 AoI 观测。
        attacked_h=attacked_h,                             # 被扰动后的 H 观测。
        original_aoi=aoi_part,                             # 原始 AoI，用于统计实际差异。
        original_h=h_part,                                 # 原始 H，用于统计实际差异。
        attack_state=attack_state,                         # episode 内攻击预算状态。
        structural_attack_type="semantic_joint_mislead",   # 标记攻击类型。
        extra_info=extra_info,                             # 合并 AoI/H 结构化解释字段。
    )
