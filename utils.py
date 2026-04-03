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

# 可视化数值显示范围
MSE_MIN, MSE_MAX = 50.0, 150.0
AOI_MIN, AOI_MAX = 5.0, 20.0


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


def plot_learning_curve(
    mse_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
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
    # 数据转换并限制到指定范围 [50, 150]
    mse_arr = np.asarray(mse_history, dtype=np.float64)
    mse_arr = np.clip(mse_arr, MSE_MIN, MSE_MAX)
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
    plt.ylim(MSE_MIN, MSE_MAX)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存输出
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path.resolve()


def plot_sum_aoi_curve(
    sum_aoi_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
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
    # 数据转换并限制到指定范围 [5, 20]
    aoi_arr = np.asarray(sum_aoi_history, dtype=np.float64)
    aoi_arr = np.clip(aoi_arr, AOI_MIN, AOI_MAX)
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
    plt.ylim(AOI_MIN, AOI_MAX)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 保存输出
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path.resolve()
