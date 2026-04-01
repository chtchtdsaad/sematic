from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_ACTION_MAP_CACHE: dict[int, dict[tuple[int, ...], int]] = {}


def map_continuous_to_discrete(virtual_action: np.ndarray, action_space: list[tuple[int, ...]]) -> int:
    """
    作用:
        将 DDPG 输出的连续动作向量映射为环境离散动作 ID。

    参数:
        virtual_action: 连续向量 v（长度为 N）。
        action_space: 环境动作表（长度为 N 的 tuple 列表）。

    返回:
        int: 对应的 action_id。

    核心步骤:
        1. 对连续向量降序排序并取 Top-M 传感器。
        2. 生成长度 N 的动作元组（Top-M 依次分配 1..M，其余为 0）。
        3. 在 action_space 中查找对应索引并返回。
    """
    if not action_space:
        raise ValueError("action_space cannot be empty.")

    v = np.asarray(virtual_action, dtype=np.float32).reshape(-1)
    n = len(action_space[0])
    m = int(max(action_space[0]))
    if v.shape[0] != n:
        raise ValueError(f"virtual_action length must be {n}, got {v.shape[0]}.")

    top_indices = np.argsort(v)[-m:][::-1]
    action_tuple = [0] * n
    for rank, sensor_idx in enumerate(top_indices, start=1):
        action_tuple[int(sensor_idx)] = rank
    action_tuple_t = tuple(action_tuple)

    cache_key = id(action_space)
    if cache_key not in _ACTION_MAP_CACHE:
        _ACTION_MAP_CACHE[cache_key] = {act: idx for idx, act in enumerate(action_space)}

    action_map = _ACTION_MAP_CACHE[cache_key]
    if action_tuple_t not in action_map:
        raise KeyError(f"mapped tuple {action_tuple_t} not found in action_space")
    return action_map[action_tuple_t]


def moving_average(values: list[float] | np.ndarray, window: int = 10) -> np.ndarray:
    """
    作用:
        计算窗口滑动平均用于平滑训练曲线。

    参数:
        values: 输入序列。
        window: 平滑窗口大小。

    返回:
        np.ndarray: 平滑后的序列。

    核心步骤:
        1. 转换为 numpy 数组。
        2. 使用卷积计算滑动平均。
        3. 补齐前缀保证长度一致。
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr

    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(arr, kernel, mode="valid")
    prefix = np.full(window - 1, valid[0], dtype=np.float64)
    return np.concatenate([prefix, valid])


def plot_learning_curve(
    mse_history: list[float] | np.ndarray,
    algo_name: str,
    save_path: str | Path,
    window: int = 10,
) -> Path:
    """
    作用:
        绘制并保存学习曲线（Episode vs Average Sum MSE）。

    参数:
        mse_history: 每轮 Average Sum MSE 序列。
        algo_name: 算法名称。
        save_path: 图片保存路径。
        window: 平滑窗口大小。

    返回:
        Path: 最终保存路径（绝对路径）。

    核心步骤:
        1. 计算原始曲线与滑动平均曲线。
        2. 使用 matplotlib 绘图并标注。
        3. 保存为图片文件。
    """
    mse_arr = np.asarray(mse_history, dtype=np.float64)
    smooth_arr = moving_average(mse_arr, window=window)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(mse_arr, label=f"{algo_name} Raw", alpha=0.4)
    plt.plot(smooth_arr, label=f"{algo_name} MA(W={window})", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Average Sum MSE")
    plt.title(f"{algo_name} Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path.resolve()
