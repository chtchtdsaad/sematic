"""
工具模块（utils.py）
===================
提供：
- DDPG 连续动作到离散动作映射
- 滑动平均
- MSE / SumAoI 曲线绘图
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 映射缓存: key=id(action_space), value={action_tuple: action_id}
_ACTION_MAP_CACHE: dict[int, dict[tuple[int, ...], int]] = {}

# 静态可视化数值显示范围（用于评估图/默认回退）
MSE_MIN, MSE_MAX = 50.0, 200.0
AOI_MIN, AOI_MAX = 10.0, 30.0

# 动态纵轴规则（训练图/对比图使用）
DYNAMIC_TAIL_RATIO = 0.10
DYNAMIC_MIN_TAIL_POINTS = 5
DYNAMIC_TARGET_RATIO = 0.35


def map_continuous_to_discrete(virtual_action: np.ndarray, action_space: list[tuple[int, ...]]) -> int:
    """
    作用:
        将 DDPG 的连续动作向量映射为离散 action_id。

    输入格式:
        virtual_action: np.ndarray[float], shape=(N,)
        action_space: list[tuple[int,...]]，长度 num_actions

    输出格式:
        int（action_id）

    核心步骤:
        1. 按 virtual_action 降序取 Top-M 传感器索引。
        2. 构建长度 N 的动作 tuple（Top-M 依次分配 1..M）。
        3. 通过哈希表 O(1) 查找对应 action_id。
    """
    # 检查动作空间有效性
    if not action_space:
        raise ValueError("action_space cannot be empty.")

    # 连续动作转 1D float32 向量
    v = np.asarray(virtual_action, dtype=np.float32).reshape(-1)

    # 从 action_space 推断 N 和 M
    n = len(action_space[0])
    m = int(max(action_space[0]))

    # 长度校验
    if v.shape[0] != n:
        raise ValueError(f"virtual_action length must be {n}, got {v.shape[0]}.")

    # 降序取 Top-M 索引
    top_indices = np.argsort(v)[-m:][::-1]

    # 生成目标动作 tuple
    action_tuple = [0] * n
    for rank, sensor_idx in enumerate(top_indices, start=1):
        action_tuple[int(sensor_idx)] = rank
    action_tuple_t = tuple(action_tuple)

    # 构建或读取缓存映射
    cache_key = id(action_space)
    if cache_key not in _ACTION_MAP_CACHE:
        _ACTION_MAP_CACHE[cache_key] = {act: idx for idx, act in enumerate(action_space)}

    # 查找 action_id
    action_map = _ACTION_MAP_CACHE[cache_key]
    if action_tuple_t not in action_map:
        raise KeyError(f"mapped tuple {action_tuple_t} not found in action_space")

    return action_map[action_tuple_t]


def map_continuous_to_assignment(virtual_action: np.ndarray, n: int, m: int) -> tuple[int, ...]:
    """
    作用:
        将 DDPG 连续动作向量直接映射为 assignment（不依赖离散 action_space）。

    输入格式:
        virtual_action: np.ndarray[float], shape=(n,)
        n: int，传感器数量
        m: int，信道数量

    输出格式:
        tuple[int,...]，长度 n
        - 0: 未调度
        - 1..m: 分配到对应信道

    核心步骤:
        1. 对连续动作按值降序排序。
        2. 选择 Top-m 传感器作为被调度集合。
        3. 按排序位次赋予信道编号 1..m。
    """
    # 规模合法性检查。
    if n <= 0 or m <= 0 or m > n:
        raise ValueError(f"Invalid n/m for assignment mapping: n={n}, m={m}.")

    # 连续动作转一维 float32。
    v = np.asarray(virtual_action, dtype=np.float32).reshape(-1)
    if v.shape[0] != n:
        raise ValueError(f"virtual_action length must be {n}, got {v.shape[0]}.")

    # 按值降序选取 Top-m。
    top_indices = np.argsort(v)[-m:][::-1]

    # 生成 assignment：Top-m 依次绑定信道 1..m，其余为 0。
    assignment = [0] * n
    for channel_id, sensor_idx in enumerate(top_indices, start=1):
        assignment[int(sensor_idx)] = int(channel_id)

    return tuple(assignment)


def moving_average(values: list[float] | np.ndarray, window: int = 10) -> np.ndarray:
    """
    作用:
        对序列做滑动平均平滑。

    输入格式:
        values: list[float] | np.ndarray，shape=(T,)
        window: int

    输出格式:
        np.ndarray[float64], shape=(T,)

    核心步骤:
        1. 转 float64 数组。
        2. 计算 valid 卷积平均。
        3. 用首个平滑值补齐前缀长度。
    """
    # 转换输入
    arr = np.asarray(values, dtype=np.float64)

    # 边界情况
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr

    # 均值卷积核
    kernel = np.ones(window, dtype=np.float64) / window

    # valid 段平滑值
    valid = np.convolve(arr, kernel, mode="valid")

    # 前缀补齐
    prefix = np.full(window - 1, valid[0], dtype=np.float64)

    # 返回等长平滑结果
    return np.concatenate([prefix, valid])


def _compute_tail_count(total_len: int, tail_ratio: float, min_tail_points: int) -> int:
    """
    作用:
        根据总长度计算“最后 tail 段”的点数。
    """
    if total_len <= 0:
        return 0
    tail_n = int(np.ceil(float(total_len) * float(tail_ratio)))
    tail_n = max(int(min_tail_points), tail_n)
    return min(total_len, tail_n)


def compute_tail_mean(
    values: list[float] | np.ndarray,
    tail_ratio: float = DYNAMIC_TAIL_RATIO,
    min_tail_points: int = DYNAMIC_MIN_TAIL_POINTS,
) -> float:
    """
    作用:
        计算序列最后 tail 区间的均值（用于收敛段基准）。
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    tail_n = _compute_tail_count(arr.size, tail_ratio=tail_ratio, min_tail_points=min_tail_points)
    return float(np.mean(arr[-tail_n:]))


def compute_dynamic_ylim_from_tail_mean(
    values: list[float] | np.ndarray,
    target_ratio: float = DYNAMIC_TARGET_RATIO,
    tail_ratio: float = DYNAMIC_TAIL_RATIO,
    min_tail_points: int = DYNAMIC_MIN_TAIL_POINTS,
    y_min: float = 0.0,
) -> tuple[float, float]:
    """
    作用:
        根据“最后 tail 段均值”计算动态纵轴范围。

    规则:
        y_max = tail_mean / target_ratio
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(y_min), float(y_min + 1.0)

    # 目标比例数值安全：限制在 (0,1)。
    ratio = float(np.clip(target_ratio, 1e-3, 0.99))

    # 优先使用 tail 均值；异常时回退到全局最大值。
    tail_mean = compute_tail_mean(arr, tail_ratio=tail_ratio, min_tail_points=min_tail_points)
    if not np.isfinite(tail_mean) or tail_mean <= 0.0:
        tail_mean = float(np.max(arr))
    if tail_mean <= 0.0:
        tail_mean = 1.0

    y_max = float(tail_mean / ratio)
    if y_max <= y_min:
        y_max = float(y_min + 1.0)
    return float(y_min), float(y_max)


def compute_dynamic_ylim_from_two_tail_means(
    values_a: list[float] | np.ndarray,
    values_b: list[float] | np.ndarray,
    target_ratio: float = DYNAMIC_TARGET_RATIO,
    tail_ratio: float = DYNAMIC_TAIL_RATIO,
    min_tail_points: int = DYNAMIC_MIN_TAIL_POINTS,
    y_min: float = 0.0,
) -> tuple[float, float]:
    """
    作用:
        对比图动态纵轴：基准取两条曲线 tail 均值中的较大值。
    """
    mean_a = compute_tail_mean(values_a, tail_ratio=tail_ratio, min_tail_points=min_tail_points)
    mean_b = compute_tail_mean(values_b, tail_ratio=tail_ratio, min_tail_points=min_tail_points)
    ref = float(np.nanmax(np.asarray([mean_a, mean_b], dtype=np.float64)))
    if not np.isfinite(ref) or ref <= 0.0:
        # 回退：用两条序列的整体最大值。
        a_arr = np.asarray(values_a, dtype=np.float64).reshape(-1)
        b_arr = np.asarray(values_b, dtype=np.float64).reshape(-1)
        merged = np.concatenate([a_arr, b_arr]) if a_arr.size + b_arr.size > 0 else np.asarray([1.0])
        merged = merged[np.isfinite(merged)]
        ref = float(np.max(merged)) if merged.size > 0 else 1.0
        if ref <= 0.0:
            ref = 1.0

    ratio = float(np.clip(target_ratio, 1e-3, 0.99))
    y_max = float(ref / ratio)
    if y_max <= y_min:
        y_max = float(y_min + 1.0)
    return float(y_min), float(y_max)


def plot_learning_curve(
    mse_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
    ylim: tuple[float, float] | None = None,
    hard_clip: bool = True,
) -> Path:
    """
    作用:
        绘制并保存 MSE 曲线（Episode vs Average Sum MSE）。

    输入格式:
        mse_history: list[float] | np.ndarray，shape=(num_episodes,)
        algo_name: str
        save_path: str | Path
        window: int

    输出格式:
        Path（绝对路径）

    核心步骤:
        1. 转数组并计算滑动平均。
        2. 使用 matplotlib 绘制 raw + smooth 曲线。
        3. 保存 PNG 并返回绝对路径。
    """
    # 原始数据转换。
    mse_raw = np.asarray(mse_history, dtype=np.float64)

    # 纵轴范围：未传入时回退静态默认。
    if ylim is None:
        y_min, y_max = float(MSE_MIN), float(MSE_MAX)
    else:
        y_min, y_max = float(ylim[0]), float(ylim[1])
        if y_max <= y_min:
            y_max = y_min + 1.0

    # 展示序列：hard_clip=True 时进行硬裁剪。
    mse_arr = np.clip(mse_raw, y_min, y_max) if hard_clip else mse_raw
    smooth_arr = moving_average(mse_arr, window=window)

    # 路径处理
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(mse_arr, label=f"{algo_name} Raw", alpha=0.4)
    plt.plot(smooth_arr, label=f"{algo_name} MA(W={window})", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Average Sum MSE")
    plt.title(f"{algo_name} Learning Curve")
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存输出
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path.resolve()


def plot_learning_curve_raw(
    mse_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
) -> Path:
    """
    作用:
        绘制并保存未裁剪 MSE 曲线（用于发散诊断）。

    输入格式:
        mse_history: list[float] | np.ndarray
        algo_name: str
        save_path: str | Path
        window: int

    输出格式:
        Path（绝对路径）
    """
    # 转换为 float64 原始 MSE 数组，shape=(T,)。
    mse_arr = np.asarray(mse_history, dtype=np.float64)
    # 计算滑动平均曲线，shape=(T,)。
    smooth_arr = moving_average(mse_arr, window=window)

    # 统一路径类型并创建父目录。
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建画布并绘制 raw 曲线。
    plt.figure(figsize=(10, 6))
    plt.plot(mse_arr, label=f"{algo_name} Raw(MSE)", alpha=0.35)
    # 绘制平滑曲线，便于观察趋势。
    plt.plot(smooth_arr, label=f"{algo_name} MA(W={window})", linewidth=2.0)
    # 设置横轴为 episode。
    plt.xlabel("Episode")
    # 设置纵轴为原始 MSE（不裁剪）。
    plt.ylabel("Average Sum MSE (Raw)")
    # 设置图标题为诊断用途。
    plt.title(f"{algo_name} Raw MSE Diagnostic")
    # 打开网格以便读数。
    plt.grid(True, alpha=0.3)
    # 显示图例。
    plt.legend()
    # 自适应边距，避免标签截断。
    plt.tight_layout()

    # 保存 PNG 文件。
    plt.savefig(save_path, dpi=200)
    # 关闭画布释放内存。
    plt.close()
    # 返回绝对路径。
    return save_path.resolve()


def plot_log_mse_curve(
    mse_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
    eps: float = 1e-12,
) -> Path:
    """
    作用:
        绘制并保存 log(MSE) 曲线（用于稳定性判据）。

    输入格式:
        mse_history: list[float] | np.ndarray
        algo_name: str
        save_path: str | Path
        window: int
        eps: float

    输出格式:
        Path（绝对路径）
    """
    # 转换为 float64 MSE 数组，shape=(T,)。
    mse_arr = np.asarray(mse_history, dtype=np.float64)
    # 先对 MSE 做下界截断，再取自然对数，避免 log(0)。
    log_arr = np.log(np.clip(mse_arr, eps, None))
    # 对 log(MSE) 做滑动平均，shape=(T,)。
    smooth_arr = moving_average(log_arr, window=window)

    # 统一路径类型并创建父目录。
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建画布并绘制 log(MSE) 原始曲线。
    plt.figure(figsize=(10, 6))
    plt.plot(log_arr, label=f"{algo_name} Raw log(MSE)", alpha=0.35)
    # 绘制平滑曲线。
    plt.plot(smooth_arr, label=f"{algo_name} MA(W={window})", linewidth=2.0)
    # 设置横轴为 episode。
    plt.xlabel("Episode")
    # 设置纵轴为 log(MSE)。
    plt.ylabel("log(Average Sum MSE)")
    # 设置图标题为诊断用途。
    plt.title(f"{algo_name} log(MSE) Diagnostic")
    # 打开网格以便读数。
    plt.grid(True, alpha=0.3)
    # 显示图例。
    plt.legend()
    # 自适应边距，避免标签截断。
    plt.tight_layout()

    # 保存 PNG 文件。
    plt.savefig(save_path, dpi=200)
    # 关闭画布释放内存。
    plt.close()
    # 返回绝对路径。
    return save_path.resolve()


def plot_sum_aoi_curve(
    sum_aoi_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
    ylim: tuple[float, float] | None = None,
    hard_clip: bool = True,
) -> Path:
    """
    作用:
        绘制并保存 SumAoI 曲线（Episode vs Average SumAoI）。

    输入格式:
        sum_aoi_history: list[float] | np.ndarray，shape=(num_episodes,)
        algo_name: str
        save_path: str | Path
        window: int

    输出格式:
        Path（绝对路径）

    核心步骤:
        1. 转数组并计算滑动平均。
        2. 使用 matplotlib 绘制 raw + smooth 曲线。
        3. 保存 PNG 并返回绝对路径。
    """
    # 原始数据转换。
    aoi_raw = np.asarray(sum_aoi_history, dtype=np.float64)

    # 纵轴范围：未传入时回退静态默认。
    if ylim is None:
        y_min, y_max = float(AOI_MIN), float(AOI_MAX)
    else:
        y_min, y_max = float(ylim[0]), float(ylim[1])
        if y_max <= y_min:
            y_max = y_min + 1.0

    # 展示序列：hard_clip=True 时进行硬裁剪。
    aoi_arr = np.clip(aoi_raw, y_min, y_max) if hard_clip else aoi_raw
    smooth_arr = moving_average(aoi_arr, window=window)

    # 路径处理
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(aoi_arr, label=f"{algo_name} Raw", alpha=0.4)
    plt.plot(smooth_arr, label=f"{algo_name} MA(W={window})", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Average SumAoI")
    plt.title(f"{algo_name} SumAoI Curve")
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存输出
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path.resolve()


def plot_compare_mse_curve(
    dqn_mse_history: list[float] | np.ndarray,
    ddpg_mse_history: list[float] | np.ndarray,
    save_path: str | Path,
    window: int = 10,
    ylim: tuple[float, float] | None = None,
    hard_clip: bool = True,
) -> Path:
    """
    作用:
        在同一张图中对比 DQN 与 DDPG 的 Average Sum MSE 曲线。

    输入格式:
        dqn_mse_history: list[float] | np.ndarray，shape=(num_episodes,)
        ddpg_mse_history: list[float] | np.ndarray，shape=(num_episodes,)
        save_path: str | Path
        window: int

    输出格式:
        Path（绝对路径）
    """
    # 原始数据转换。
    dqn_raw = np.asarray(dqn_mse_history, dtype=np.float64)
    ddpg_raw = np.asarray(ddpg_mse_history, dtype=np.float64)

    # 纵轴范围：未传入时回退静态默认。
    if ylim is None:
        y_min, y_max = float(MSE_MIN), float(MSE_MAX)
    else:
        y_min, y_max = float(ylim[0]), float(ylim[1])
        if y_max <= y_min:
            y_max = y_min + 1.0

    # 展示序列：hard_clip=True 时进行硬裁剪。
    dqn_arr = np.clip(dqn_raw, y_min, y_max) if hard_clip else dqn_raw
    ddpg_arr = np.clip(ddpg_raw, y_min, y_max) if hard_clip else ddpg_raw

    # 平滑曲线
    dqn_smooth = moving_average(dqn_arr, window=window)
    ddpg_smooth = moving_average(ddpg_arr, window=window)

    # 路径处理
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_arr, label="DQN Raw", alpha=0.25, color="#1f77b4")
    plt.plot(dqn_smooth, label=f"DQN MA(W={window})", linewidth=2.2, color="#1f77b4")
    plt.plot(ddpg_arr, label="DDPG Raw", alpha=0.25, color="#ff7f0e")
    plt.plot(ddpg_smooth, label=f"DDPG MA(W={window})", linewidth=2.2, color="#ff7f0e")
    plt.xlabel("Episode")
    plt.ylabel("Average Sum MSE")
    plt.title("DQN vs DDPG - Average Sum MSE")
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存输出
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path.resolve()


def plot_compare_sum_aoi_curve(
    dqn_sum_aoi_history: list[float] | np.ndarray,
    ddpg_sum_aoi_history: list[float] | np.ndarray,
    save_path: str | Path,
    window: int = 10,
    ylim: tuple[float, float] | None = None,
    hard_clip: bool = True,
) -> Path:
    """
    作用:
        在同一张图中对比 DQN 与 DDPG 的 Average SumAoI 曲线。

    输入格式:
        dqn_sum_aoi_history: list[float] | np.ndarray，shape=(num_episodes,)
        ddpg_sum_aoi_history: list[float] | np.ndarray，shape=(num_episodes,)
        save_path: str | Path
        window: int

    输出格式:
        Path（绝对路径）
    """
    # 原始数据转换。
    dqn_raw = np.asarray(dqn_sum_aoi_history, dtype=np.float64)
    ddpg_raw = np.asarray(ddpg_sum_aoi_history, dtype=np.float64)

    # 纵轴范围：未传入时回退静态默认。
    if ylim is None:
        y_min, y_max = float(AOI_MIN), float(AOI_MAX)
    else:
        y_min, y_max = float(ylim[0]), float(ylim[1])
        if y_max <= y_min:
            y_max = y_min + 1.0

    # 展示序列：hard_clip=True 时进行硬裁剪。
    dqn_arr = np.clip(dqn_raw, y_min, y_max) if hard_clip else dqn_raw
    ddpg_arr = np.clip(ddpg_raw, y_min, y_max) if hard_clip else ddpg_raw

    # 平滑曲线
    dqn_smooth = moving_average(dqn_arr, window=window)
    ddpg_smooth = moving_average(ddpg_arr, window=window)

    # 路径处理
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_arr, label="DQN Raw", alpha=0.25, color="#1f77b4")
    plt.plot(dqn_smooth, label=f"DQN MA(W={window})", linewidth=2.2, color="#1f77b4")
    plt.plot(ddpg_arr, label="DDPG Raw", alpha=0.25, color="#ff7f0e")
    plt.plot(ddpg_smooth, label=f"DDPG MA(W={window})", linewidth=2.2, color="#ff7f0e")
    plt.xlabel("Episode")
    plt.ylabel("Average SumAoI")
    plt.title("DQN vs DDPG - Average SumAoI")
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存输出
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path.resolve()
