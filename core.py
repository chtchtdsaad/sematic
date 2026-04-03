"""
核心数学与动作空间模块（core.py）
==============================
本模块只包含纯函数，不依赖训练流程。
"""

from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

from config import M, N, MAX_SPECTRAL_RADIUS, MIN_SPECTRAL_RADIUS


def generate_unstable_matrix(
    dim: int = 2,
    spectral_radius_range: Tuple[float, float] = (MIN_SPECTRAL_RADIUS, MAX_SPECTRAL_RADIUS),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    作用:
        生成满足约束的二维随机系统矩阵 A。

    输入格式:
        dim: int，矩阵维度，当前实验固定为 2。
        spectral_radius_range: tuple[float,float]，目标谱半径范围 (low, high)。
        rng: np.random.Generator | None，随机数生成器。

    输出格式:
        np.ndarray[float64]，shape=(2,2)。

    核心步骤:
        1. 采样正矩阵 B，并构造 A0 = B^T B + eps*I（保证特征值全正）。
        2. 采样目标谱半径并按比例缩放 A0。
        3. 检查“元素全正 + 特征值全正 + 谱半径在区间内”，通过后返回。
    """
    # 若外部没有传入 RNG，则创建默认生成器
    if rng is None:
        rng = np.random.default_rng()

    # 解包并检查谱半径区间
    low, high = spectral_radius_range
    if not (0.0 < low < high):
        raise ValueError("Invalid spectral radius range.")

    # 按当前任务要求，只支持 2x2
    if dim != 2:
        raise ValueError("generate_unstable_matrix currently only supports dim=2.")

    # 防止数值病态的微小正则项
    eps = 1e-9

    # 最多尝试若干次，避免极端随机样本导致死循环
    max_trials = 2000

    for _ in range(max_trials):
        # Step-1: 采样正元素矩阵 B，shape=(2,2), dtype=float64
        b = rng.uniform(1e-3, 1.0, size=(dim, dim))

        # Step-2: 构造正定候选矩阵 A0
        a0 = b.T @ b + eps * np.eye(dim, dtype=np.float64)

        # Step-3: 获取当前谱半径（对称正定可用 eigvalsh）
        eigvals0 = np.linalg.eigvalsh(a0)
        spectral_radius0 = float(np.max(eigvals0))
        if spectral_radius0 <= 0.0:
            continue

        # Step-4: 采样目标谱半径并进行线性缩放
        target_radius = float(rng.uniform(low, high))
        a = a0 * (target_radius / spectral_radius0)

        # Step-5: 终检（元素/特征值/谱半径）
        eigvals = np.linalg.eigvalsh(a)
        rho = float(np.max(eigvals))
        if np.all(a > 0.0) and np.all(eigvals > 0.0) and (low <= rho <= high):
            return a

    # 如果若干次都没找到合格矩阵则报错
    raise RuntimeError("Failed to generate valid matrix after many trials.")


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
        求解稳态卡尔曼误差协方差 P_bar。

    输入格式:
        a: np.ndarray[float64], shape=(ln,ln)
        c: np.ndarray[float64], shape=(en,ln)
        w: np.ndarray[float64], shape=(ln,ln)
        v: np.ndarray[float64], shape=(en,en)
        max_iter: int，Riccati 迭代最大次数。
        tol: float，收敛阈值（Frobenius 范数）。

    输出格式:
        np.ndarray[float64], shape=(ln,ln)，对称化后的稳态协方差。

    核心步骤:
        1. 用离散 Riccati 递推进行数值迭代。
        2. 若达到收敛阈值则返回。
        3. 若未收敛，回退到 scipy.solve_discrete_are。
        4. 对结果做对称化，降低数值误差。
    """
    # 初始值常用过程噪声协方差
    p = w.copy()

    # 迭代 Riccati 方程
    for _ in range(max_iter):
        # 创新协方差，shape=(en,en)
        innovation = c @ p @ c.T + v

        # 创新协方差逆
        innovation_inv = np.linalg.inv(innovation)

        # Riccati 一步更新，shape=(ln,ln)
        p_next = a @ p @ a.T + w - a @ p @ c.T @ innovation_inv @ c @ p @ a.T

        # 收敛判据：两次迭代差的 Frobenius 范数
        if np.linalg.norm(p_next - p, ord="fro") < tol:
            # 对称化返回，防止数值非对称
            return 0.5 * (p_next + p_next.T)

        # 进入下一轮
        p = p_next

    # 回退到代数 Riccati 方程闭式求解
    p_dare = solve_discrete_are(a.T, c.T, w, v)

    # 对称化后返回
    return 0.5 * (p_dare + p_dare.T)


def compute_mse_from_aoi(
    p_bar: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    aoi: int,
    a_powers: np.ndarray | None = None,
    noise_cov_sums: np.ndarray | None = None,
) -> float:
    """
    作用:
        根据 AoI 与论文 Eq.(9) 计算当前 MSE（Tr(P_t)）。

    输入格式:
        p_bar: np.ndarray[float64], shape=(ln,ln)
        a: np.ndarray[float64], shape=(ln,ln)
        w: np.ndarray[float64], shape=(ln,ln)
        aoi: int，要求 >= 1。
        a_powers: np.ndarray | None，shape=(max_aoi+1,ln,ln)，缓存 A^k。
        noise_cov_sums: np.ndarray | None，shape=(max_aoi+1,ln,ln)，缓存噪声累计项。

    输出格式:
        float，当前时刻 MSE。

    核心步骤:
        1. 若存在缓存，直接读取 A^aoi 与噪声和进行快速计算。
        2. 若无缓存，按公式用 matrix_power 逐项计算。
        3. 返回 trace(P_t)。
    """
    # AoI 合法性检查
    if aoi < 1:
        raise ValueError("AoI must be >= 1.")

    # 快速路径：使用缓存，减少重复幂运算
    if a_powers is not None and noise_cov_sums is not None:
        # 检查缓存覆盖范围
        if aoi >= a_powers.shape[0] or aoi >= noise_cov_sums.shape[0]:
            raise ValueError("AoI exceeds cached range.")

        # A^aoi，shape=(ln,ln)
        a_pow_tau = a_powers[aoi]

        # P_t = A^tau Pbar (A^T)^tau + noise_sum[tau]
        p_t = a_pow_tau @ p_bar @ a_pow_tau.T + noise_cov_sums[aoi]

        # 输出 MSE = trace(P_t)
        return float(np.trace(p_t))

    # 慢速路径：按定义逐项计算
    a_pow_tau = np.linalg.matrix_power(a, aoi)
    p_t = a_pow_tau @ p_bar @ a_pow_tau.T

    # 噪声累计项求和: sum_{k=0}^{aoi-1} A^k W (A^T)^k
    for k in range(aoi):
        a_pow_k = np.linalg.matrix_power(a, k)
        p_t = p_t + a_pow_k @ w @ a_pow_k.T

    # 返回迹
    return float(np.trace(p_t))


def generate_action_space(n: int = N, m: int = M) -> List[Tuple[int, ...]]:
    """
    作用:
        生成离散动作空间（有序选择 + 信道编号分配）。

    输入格式:
        n: int，传感器数量。
        m: int，信道数量。

    输出格式:
        List[Tuple[int,...]]，长度为 n!/(n-m)!。
        每个动作 tuple 长度为 n，元素含义：
        - 0: 未调度
        - 1..m: 分配到对应信道编号

    核心步骤:
        1. 枚举长度 m 的传感器排列（有序）。
        2. 排列位置映射到信道 1..m。
        3. 其余传感器置 0。
    """
    # 动作容器
    actions: List[Tuple[int, ...]] = []

    # permutations 产生有序选择结果
    for scheduled_sensors in itertools.permutations(range(n), m):
        # 默认全部未调度
        action = [0] * n

        # 第 channel_idx 个位置的传感器分配到 channel_idx
        for channel_idx, sensor_idx in enumerate(scheduled_sensors, start=1):
            action[sensor_idx] = channel_idx

        # 存为不可变 tuple，便于作为哈希键
        actions.append(tuple(action))

    return actions


def decode_action(action_id: int, action_space: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    """
    作用:
        将离散动作编号解码为具体动作 tuple。

    输入格式:
        action_id: int，范围 [0, len(action_space)-1]。
        action_space: Sequence[Tuple[int,...]]，动作表。

    输出格式:
        Tuple[int,...]，长度为 n 的动作向量。

    核心步骤:
        1. 检查 action_id 是否越界。
        2. 直接索引返回对应动作。
    """
    # 越界检查
    if action_id < 0 or action_id >= len(action_space):
        raise IndexError("Action id out of range.")

    # 返回动作 tuple
    return action_space[action_id]


if __name__ == "__main__":
    # 最小自测：打印矩阵与特征值
    a_test = generate_unstable_matrix()
    print(a_test)
    print(np.linalg.eigvals(a_test))
