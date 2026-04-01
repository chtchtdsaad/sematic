import itertools
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

from config import M, N


def generate_unstable_matrix(
    dim: int = 2,
    spectral_radius_range: Tuple[float, float] = (1.0, 1.4),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    作用:
        生成随机方阵 A，并将其谱半径缩放到指定区间内，得到不稳定系统矩阵。

    参数:
        dim: 方阵维度，默认 2。
        spectral_radius_range: 目标谱半径范围 (low, high)。
        rng: 随机数生成器；为 None 时自动创建。

    返回:
        np.ndarray: 缩放后的 dim x dim 矩阵。

    核心步骤:
        1. 生成元素全正的随机矩阵 B。
        2. 构造 A = B^T B + eps*I，确保特征值全为正。
        3. 随机采样目标谱半径并按比例缩放 A。
        4. 验证元素、谱半径、特征值均满足约束后返回。
    """
    if rng is None:
        rng = np.random.default_rng()

    low, high = spectral_radius_range
    if not (0 < low < high):
        raise ValueError('Invalid spectral radius range.')

    if dim != 2:
        raise ValueError('This function is designed for 2x2 matrices in current experiment.')

    eps = 1e-6
    max_trials = 1000
    for _ in range(max_trials):
        b = rng.uniform(1e-3, 1.0, size=(dim, dim))
        a = b.T @ b + eps * np.eye(dim, dtype=np.float64)

        eigenvalues = np.linalg.eigvalsh(a)
        if np.any(eigenvalues <= 0):
            continue

        spectral_radius = float(np.max(eigenvalues))
        if spectral_radius <= 1e-12:
            continue

        target_radius = float(rng.uniform(low, high))
        scaled_a = a * (target_radius / spectral_radius)
        scaled_eigs = np.linalg.eigvalsh(scaled_a)

        if np.all(scaled_a > 0.0) and np.all(scaled_eigs > 0.0):
            return scaled_a

    raise RuntimeError('Failed to generate a valid matrix under positivity and spectral constraints.')


def solve_steady_state_covariance(
    a: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    max_iter: int = 10000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    作用:
        求解稳态卡尔曼估计误差协方差 P_bar。

    参数:
        a: 系统状态矩阵 A。
        c: 观测矩阵 C。
        w: 过程噪声协方差 W。
        v: 观测噪声协方差 V。
        max_iter: 迭代上限。
        tol: 收敛阈值（Frobenius 范数）。

    返回:
        np.ndarray: 对称化后的稳态误差协方差矩阵 P_bar。

    核心步骤:
        1. 采用 Riccati 递推进行迭代更新。
        2. 若相邻迭代差小于阈值则判定收敛并返回。
        3. 若未收敛则回退到 `solve_discrete_are` 求解。
        4. 对结果做对称化处理保证数值稳定性。
    """
    p = w.copy()
    for _ in range(max_iter):
        innovation = c @ p @ c.T + v
        innovation_inv = np.linalg.inv(innovation)
        p_next = a @ p @ a.T + w - a @ p @ c.T @ innovation_inv @ c @ p @ a.T
        if np.linalg.norm(p_next - p, ord='fro') < tol:
            return 0.5 * (p_next + p_next.T)
        p = p_next

    p_dare = solve_discrete_are(a.T, c.T, w, v)
    return 0.5 * (p_dare + p_dare.T)


def compute_mse_from_aoi(
    p_bar: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    aoi: int,
) -> float:
    """
    作用:
        根据 AoI 和论文 Eq.(9) 计算当前协方差迹值（MSE）。

    参数:
        p_bar: 稳态协方差矩阵 P_bar。
        a: 系统状态矩阵 A。
        w: 过程噪声协方差 W。
        aoi: 当前信息年龄 tau，要求 >= 1。

    返回:
        float: Tr(P_t)，即当前时刻的 MSE 指标。

    核心步骤:
        1. 计算 A^tau * P_bar * (A^T)^tau。
        2. 累加噪声传播项 sum_{k=0}^{tau-1} A^k W (A^T)^k。
        3. 对最终 P_t 取 trace 作为 MSE。
    """
    if aoi < 1:
        raise ValueError('AoI must be >= 1.')

    a_pow_tau = np.linalg.matrix_power(a, aoi)
    p_t = a_pow_tau @ p_bar @ a_pow_tau.T

    for k in range(aoi):
        a_pow_k = np.linalg.matrix_power(a, k)#不要改这里
        p_t = p_t + a_pow_k @ w @ a_pow_k.T

    return float(np.trace(p_t))


def generate_action_space(n: int = N, m: int = M) -> List[Tuple[int, ...]]:
    """
    作用:
        生成离散动作空间，枚举所有“n 个传感器里选 m 个并分配到 m 条不同信道”的方案。

    参数:
        n: 传感器数量。
        m: 可用信道数量（每条信道每时隙最多分给一个传感器）。

    返回:
        List[Tuple[int, ...]]: 动作列表，长度为 n!/(n-m)!。
            每个动作是长度 n 的元组：
            - 0 表示该传感器本时隙未调度
            - 1..m 表示分配到对应信道编号

    核心步骤:
        1. 对传感器索引做 m 长度排列。
        2. 将排列位置映射为信道编号 1..m。
        3. 未被选中的传感器填 0。
    """
    actions: List[Tuple[int, ...]] = []
    for scheduled_sensors in itertools.permutations(range(n), m):
        action = [0] * n
        for channel_idx, sensor_idx in enumerate(scheduled_sensors, start=1):
            action[sensor_idx] = channel_idx
        actions.append(tuple(action))
    return actions


def decode_action(action_id: int, action_space: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    作用:
        将离散动作编号解码为具体传感器-信道分配方案。

    参数:
        action_id: 动作编号（从 0 开始）。
        action_space: 由 `generate_action_space` 生成的动作集合。

    返回:
        Tuple[int, ...]: 具体动作向量（长度为 n）。

    核心步骤:
        1. 检查 action_id 是否越界。
        2. 直接按索引返回对应分配方案。
    """
    if action_id < 0 or action_id >= len(action_space):
        raise IndexError('Action id out of range.')
    return action_space[action_id]


if __name__ == '__main__':
    a = generate_unstable_matrix()
    print(a, np.linalg.eigvals(a))
    
