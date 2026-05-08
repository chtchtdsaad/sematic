"""
攻击状态处理模块（state_ops.py）
==============================
本文件只保存阶段 2 所需的状态拆分、拼接和合法范围投影函数。
"""

from __future__ import annotations

import numpy as np


def split_state(state: np.ndarray, env) -> tuple[np.ndarray, np.ndarray]:
    """
    作用:
        将环境状态向量拆分为 AoI 段和 H_t 信道状态段。

    输入格式:
        state: np.ndarray[float], shape=(N + N*M,)
        env: 环境实例，需提供 n 和 m 属性。

    输出格式:
        tuple[np.ndarray, np.ndarray]
        - aoi_part: np.ndarray[float32], shape=(N,)
        - h_part: np.ndarray[float32], shape=(N*M,)

    核心步骤:
        1. 将输入状态转为一维 float32 数组副本。
        2. 根据 env.n / env.m 校验状态长度。
        3. 前 N 维作为 AoI，后 N*M 维作为 H_t 展平段。
    """
    # 将输入统一为一维 float32 副本，避免后续扰动影响原始 state。
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1).copy()

    # 根据环境规模计算理论状态长度。
    n = int(env.n)
    m = int(env.m)
    expected_len = n + n * m

    # 状态长度不匹配时立即报错，避免攻击模块在错误切片上运行。
    if state_arr.size != expected_len:
        raise ValueError(f"state length must be {expected_len}, got {state_arr.size}.")

    # AoI 段为前 N 维，shape=(N,)。
    aoi_part = state_arr[:n].copy()

    # H_t 段为后 N*M 维，shape=(N*M,)。
    h_part = state_arr[n:].copy()

    # 返回两个互不共享写入意图的副本。
    return aoi_part, h_part


def merge_state(aoi_part: np.ndarray, h_part: np.ndarray) -> np.ndarray:
    """
    作用:
        将 AoI 段和 H_t 段拼接回完整状态向量。

    输入格式:
        aoi_part: np.ndarray[float], shape=(N,)
        h_part: np.ndarray[float], shape=(N*M,)

    输出格式:
        np.ndarray[float32], shape=(N + N*M,)

    核心步骤:
        1. 将 AoI 与 H_t 都转为一维 float32。
        2. 按原始状态格式 [AoI, H_t 展平] 拼接。
        3. 返回完整状态副本。
    """
    # AoI 段转为一维 float32。
    aoi_arr = np.asarray(aoi_part, dtype=np.float32).reshape(-1)

    # H_t 段转为一维 float32。
    h_arr = np.asarray(h_part, dtype=np.float32).reshape(-1)

    # 按环境状态格式拼接并返回 float32 副本。
    return np.concatenate([aoi_arr, h_arr]).astype(np.float32, copy=True)


def project_aoi_part(aoi_part: np.ndarray, env) -> np.ndarray:
    """
    作用:
        将 AoI 段投影到环境允许的整数范围。

    输入格式:
        aoi_part: np.ndarray[float], shape=(N,)
        env: 环境实例，需提供 max_aoi 属性。

    输出格式:
        np.ndarray[float32], shape=(N,)，元素范围 [1, env.max_aoi]。

    核心步骤:
        1. 将 AoI 输入转为一维 float32。
        2. 对扰动结果 round 到整数语义。
        3. clip 到 [1, env.max_aoi] 并返回。
    """
    # AoI 输入统一为一维 float32。
    aoi_arr = np.asarray(aoi_part, dtype=np.float32).reshape(-1)

    # AoI 是离散整数语义，先四舍五入再裁剪合法范围。
    projected = np.clip(np.rint(aoi_arr), 1, int(env.max_aoi))

    # 返回 float32，保持与环境 state dtype 一致。
    return projected.astype(np.float32, copy=False)


def project_h_part(h_part: np.ndarray) -> np.ndarray:
    """
    作用:
        将 H_t 信道状态段投影到离散信道索引合法范围。

    输入格式:
        h_part: np.ndarray[float], shape=(N*M,)

    输出格式:
        np.ndarray[float32], shape=(N*M,)，元素范围 [0,4]。

    核心步骤:
        1. 将 H_t 输入转为一维 float32。
        2. 对扰动结果 round 到离散索引。
        3. clip 到 [0,4] 并返回。
    """
    # H_t 输入统一为一维 float32。
    h_arr = np.asarray(h_part, dtype=np.float32).reshape(-1)

    # H_t 信道索引共有 5 档，合法取值为 0..4。
    projected = np.clip(np.rint(h_arr), 0, 4)

    # 返回 float32，保持与环境 state dtype 一致。
    return projected.astype(np.float32, copy=False)
