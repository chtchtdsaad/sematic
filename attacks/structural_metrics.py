"""
结构性资源错配指标模块（structural_metrics.py）
==============================================
本文件计算 clean_action 与 attacked_action 在真实 state 下的资源分配差异，
用于解释攻击是否把调度资源从高风险 sensor 或好信道上诱导开。
"""

from __future__ import annotations

import numpy as np

from attacks.state_ops import split_state
from attacks.structural_common import compute_sensor_risk, decode_eval_action_to_assignment


def _scheduled_sensors(assignment: tuple[int, ...]) -> set[int]:
    """
    作用:
        提取 assignment 中被调度的 sensor 下标。
    参数:
        assignment: 调度动作 tuple，长度为 N，0 表示未调度，1..M 表示信道编号。
    输入输出:
        输入 assignment；输出 set[int]，表示被调度 sensor 集合。
    核心步骤:
        1. 枚举 assignment。
        2. 保留 channel_id > 0 的 sensor 下标。
    """
    # channel_id > 0 表示 sensor 被调度。
    return {int(sensor_idx) for sensor_idx, channel_id in enumerate(assignment) if int(channel_id) > 0}


def _scheduled_risk_sum(assignment: tuple[int, ...], risk: np.ndarray) -> float:
    """
    作用:
        计算被调度 sensor 的真实 risk 总和。
    参数:
        assignment: 调度动作 tuple。
        risk: sensor 风险数组，shape=(N,)。
    输入输出:
        输入 assignment 和 risk；输出 selected risk sum。
    核心步骤:
        1. 提取被调度 sensor。
        2. 累加这些 sensor 的 risk。
    """
    # 取出被调度 sensor 集合。
    selected = _scheduled_sensors(assignment)
    # 累加真实 risk。
    return float(sum(float(risk[idx]) for idx in selected))


def _scheduled_ratio(assignment: tuple[int, ...], sensor_indices: list[int]) -> float:
    """
    作用:
        计算目标 sensor 集合中有多少比例被调度。
    参数:
        assignment: 调度动作 tuple。
        sensor_indices: 目标 sensor 下标列表，例如高风险 top-K 或低风险 bottom-K。
    输入输出:
        输入 assignment 和 sensor 列表；输出 [0,1] 范围内的比例。
    核心步骤:
        1. 提取被调度 sensor 集合。
        2. 统计 sensor_indices 中被调度的数量。
        3. 除以 sensor_indices 长度。
    """
    # 空集合返回 0，避免除零。
    if not sensor_indices:
        return 0.0
    # 取出实际被调度 sensor。
    selected = _scheduled_sensors(assignment)
    # 统计命中个数。
    hit_count = sum(1 for idx in sensor_indices if int(idx) in selected)
    # 返回比例。
    return float(hit_count) / float(len(sensor_indices))


def _high_risk_good_channel_ratio(assignment: tuple[int, ...], high_risk_sensors: list[int], h_matrix: np.ndarray) -> float:
    """
    作用:
        计算高风险 sensor 被分配到真实好信道的比例。
    参数:
        assignment: 调度动作 tuple。
        high_risk_sensors: 高风险 sensor 下标列表。
        h_matrix: 真实 H 矩阵，shape=(N,M)。
    输入输出:
        输入动作、高风险 sensor 和真实 H；输出 [0,1] 比例。
    核心步骤:
        1. 用高风险 sensor 的 H 中位数作为好信道阈值。
        2. 检查每个高风险 sensor 是否被调度到不低于阈值的信道。
        3. 返回命中比例。
    """
    # 空集合返回 0。
    if not high_risk_sensors:
        return 0.0
    # 使用高风险 sensor 上真实 H 的中位数作为好信道阈值。
    threshold = float(np.median(h_matrix[np.asarray(high_risk_sensors, dtype=int), :]))
    # 命中计数初始化。
    hit_count = 0
    # 逐个高风险 sensor 检查其分配信道。
    for sensor_idx in high_risk_sensors:
        # 读取 assignment 中的信道编号。
        channel_id = int(assignment[int(sensor_idx)])
        # 未调度不计入好信道命中。
        if channel_id <= 0:
            continue
        # assignment 信道编号 1..M 转为矩阵下标 0..M-1。
        channel_idx = channel_id - 1
        # 若真实 H 不低于阈值，则认为调到了好信道。
        if 0 <= channel_idx < h_matrix.shape[1] and float(h_matrix[int(sensor_idx), channel_idx]) >= threshold:
            hit_count += 1
    # 返回高风险 sensor 中被好信道服务的比例。
    return float(hit_count) / float(len(high_risk_sensors))


def compute_structural_action_metrics(real_state: np.ndarray, env, algo: str, clean_action, attacked_action) -> dict[str, float]:
    """
    作用:
        计算 clean_action 和 attacked_action 的结构性资源错配指标。
    参数:
        real_state: 真实环境 state，shape=(N+N*M,)。
        env: 环境实例，需要提供 n、m、max_aoi 和 MSE 缓存。
        algo: 算法名，支持 DQN 或 DDPG。
        clean_action: 真实 state 下 victim policy 的动作。
        attacked_action: attacked_state 下 victim policy 的动作。
    输入输出:
        输入真实 state 与两类动作；输出结构性指标 dict[str,float]。
    核心步骤:
        1. 基于真实 AoI 计算 sensor risk。
        2. 将 clean/attack 动作统一解码为 assignment。
        3. 比较两类动作在高风险 sensor 和好信道上的资源分配差异。
    """
    # 拆分真实 state。
    aoi_part, h_part = split_state(real_state, env)
    # 基于真实 AoI 计算风险。
    risk = compute_sensor_risk(aoi_part, env, risk_mode="reset_gain")
    # H 恢复为矩阵。
    h_matrix = np.asarray(h_part, dtype=np.float64).reshape(int(env.n), int(env.m))
    # clean 动作解码为 assignment。
    clean_assignment = decode_eval_action_to_assignment(algo, clean_action, env)
    # attack 动作解码为 assignment。
    attack_assignment = decode_eval_action_to_assignment(algo, attacked_action, env)
    # K 取可调度 sensor 数，当前等于信道数 m。
    k = max(1, min(int(env.m), int(env.n)))
    # 风险降序得到高风险 sensor。
    high_risk = np.argsort(-risk, kind="mergesort")[:k].astype(int).tolist()
    # 风险升序得到低风险 sensor。
    low_risk = np.argsort(risk, kind="mergesort")[:k].astype(int).tolist()
    # clean 被选 sensor risk 总和。
    clean_selected_risk_sum = _scheduled_risk_sum(clean_assignment, risk)
    # attack 被选 sensor risk 总和。
    attack_selected_risk_sum = _scheduled_risk_sum(attack_assignment, risk)
    # 返回所有结构性解释指标。
    return {
        "clean_selected_risk_sum": float(clean_selected_risk_sum),
        "attack_selected_risk_sum": float(attack_selected_risk_sum),
        "resource_misallocation_score": float(clean_selected_risk_sum - attack_selected_risk_sum),
        "high_risk_scheduled_ratio_clean": _scheduled_ratio(clean_assignment, high_risk),
        "high_risk_scheduled_ratio_attack": _scheduled_ratio(attack_assignment, high_risk),
        "low_risk_scheduled_ratio_clean": _scheduled_ratio(clean_assignment, low_risk),
        "low_risk_scheduled_ratio_attack": _scheduled_ratio(attack_assignment, low_risk),
        "high_risk_good_channel_ratio_clean": _high_risk_good_channel_ratio(clean_assignment, high_risk, h_matrix),
        "high_risk_good_channel_ratio_attack": _high_risk_good_channel_ratio(attack_assignment, high_risk, h_matrix),
    }


def update_structural_metric_stats(stats: dict, metrics: dict[str, float]) -> dict:
    """
    作用:
        将单步结构性指标累加到 attack stats。
    参数:
        stats: attack eval 的累计统计字典。
        metrics: compute_structural_action_metrics 返回的单步指标。
    输入输出:
        输入累计 stats 和单步 metrics；输出原 stats 对象，已原地更新。
    核心步骤:
        1. 为每个指标构造 total_* 累计字段。
        2. 将单步值累加到累计字段。
        3. 返回 stats 供调用方继续使用。
    """
    # 每个报告字段对应一个 total_* 累计字段。
    for key, value in metrics.items():
        # 累计字段命名规则。
        total_key = f"total_{key}"
        # 原地累加，缺失时从 0 开始。
        stats[total_key] = float(stats.get(total_key, 0.0)) + float(value)
    # 返回原 stats，便于链式调用。
    return stats
