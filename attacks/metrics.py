"""
攻击评估统计模块（metrics.py）
============================
本文件只负责 attack_step、action_flip、扰动强度与约束违规统计。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def init_attack_stats() -> dict[str, int | float]:
    """
    作用:
        初始化攻击评估阶段使用的累计统计字典。
    输入格式:
        无。
    输出格式:
        dict[str, int | float]，包含步数、攻击次数、扰动强度和违规次数。
    核心步骤:
        1. 将计数字段初始化为 0。
        2. 将累计强度字段初始化为 0.0。
        3. 将最大范数字段初始化为 0.0。
    """
    return {
        "total_steps": 0,
        "attack_step_count": 0,
        "action_flip_count": 0,
        "total_aoi_l0": 0,
        "total_h_l0": 0,
        "total_l0": 0,
        "total_aoi_l1": 0.0,
        "total_h_l1": 0.0,
        "max_aoi_linf": 0.0,
        "max_h_linf": 0.0,
        "constraint_violation_count": 0,
        "total_clean_selected_risk_sum": 0.0,
        "total_attack_selected_risk_sum": 0.0,
        "total_resource_misallocation_score": 0.0,
        "total_high_risk_scheduled_ratio_clean": 0.0,
        "total_high_risk_scheduled_ratio_attack": 0.0,
        "total_low_risk_scheduled_ratio_clean": 0.0,
        "total_low_risk_scheduled_ratio_attack": 0.0,
        "total_high_risk_good_channel_ratio_clean": 0.0,
        "total_high_risk_good_channel_ratio_attack": 0.0,
    }


def _get_number(data: dict, key: str, default: float = 0.0) -> float:
    """
    作用:
        从 perturb_info 中读取数值字段，缺失时返回默认值。
    输入格式:
        data: dict，当前 step 的扰动统计。
        key: str，字段名。
        default: float，缺省值。
    输出格式:
        float。
    核心步骤:
        1. 使用 dict.get 读取字段。
        2. 将 numpy 标量或 Python 数值统一转为 float。
    """
    return float(data.get(key, default))


def _has_constraint_violation(perturb_info: dict, config) -> bool:
    """
    作用:
        检查当前 step 的扰动统计是否违反阶段 5 约束。
    输入格式:
        perturb_info: dict，需包含 aoi_l0 / h_l0 / total_l0 / aoi_linf / h_linf 等字段。
        config: AttackConfig 或同字段对象。
    输出格式:
        bool，True 表示存在至少一个约束违规。
    核心步骤:
        1. 检查 AoI / H / total 的 L0 稀疏预算。
        2. 检查 AoI / H 的 L_inf 单维幅值预算。
        3. 任一条件超限即返回 True。
    """
    # AoI 扰动维度不能超过配置预算。
    if _get_number(perturb_info, "aoi_l0") > float(getattr(config, "max_aoi_features", 0)):
        return True
    # H 扰动维度不能超过配置预算。
    if _get_number(perturb_info, "h_l0") > float(getattr(config, "max_h_features", 0)):
        return True
    # 总扰动维度不能超过配置预算。
    if _get_number(perturb_info, "total_l0") > float(getattr(config, "max_total_features", 0)):
        return True
    # AoI 单维扰动幅度不能超过配置预算。
    if _get_number(perturb_info, "aoi_linf") > float(getattr(config, "aoi_delta", 0)):
        return True
    # H 单维扰动幅度不能超过配置预算。
    if _get_number(perturb_info, "h_linf") > float(getattr(config, "h_delta", 0)):
        return True
    return False


def update_attack_step_stats(stats: dict, perturb_info: dict, config) -> dict:
    """
    作用:
        根据当前 step 的 perturb_info 更新攻击步、扰动强度和约束违规统计。
    输入格式:
        stats: dict，init_attack_stats 返回的累计统计。
        perturb_info: dict，当前 step 的扰动统计，缺失字段按 0 处理。
        config: AttackConfig 或同字段对象，用于检查约束。
    输出格式:
        dict，原 stats 对象，已原地更新。
    核心步骤:
        1. total_steps 加 1。
        2. 若 is_attacked=True，则 attack_step_count 加 1。
        3. 累加 L0 / L1，更新最大 L_inf。
        4. 若约束超限，则 constraint_violation_count 加 1。
    """
    # 每调用一次表示处理一个环境 step。
    stats["total_steps"] = int(stats.get("total_steps", 0)) + 1

    # 读取当前 step 的扰动强度，缺失字段按 clean step 处理。
    aoi_l0 = int(_get_number(perturb_info, "aoi_l0"))
    h_l0 = int(_get_number(perturb_info, "h_l0"))
    total_l0 = int(_get_number(perturb_info, "total_l0"))
    aoi_l1 = _get_number(perturb_info, "aoi_l1")
    h_l1 = _get_number(perturb_info, "h_l1")
    aoi_linf = _get_number(perturb_info, "aoi_linf")
    h_linf = _get_number(perturb_info, "h_linf")

    # 攻击步计数由 perturb_info["is_attacked"] 控制。
    if bool(perturb_info.get("is_attacked", False)):
        stats["attack_step_count"] = int(stats.get("attack_step_count", 0)) + 1

    # 累计 L0 / L1 扰动强度。
    stats["total_aoi_l0"] = int(stats.get("total_aoi_l0", 0)) + aoi_l0
    stats["total_h_l0"] = int(stats.get("total_h_l0", 0)) + h_l0
    stats["total_l0"] = int(stats.get("total_l0", 0)) + total_l0
    stats["total_aoi_l1"] = float(stats.get("total_aoi_l1", 0.0)) + aoi_l1
    stats["total_h_l1"] = float(stats.get("total_h_l1", 0.0)) + h_l1

    # 记录整个评估过程中的最大单维扰动幅度。
    stats["max_aoi_linf"] = max(float(stats.get("max_aoi_linf", 0.0)), aoi_linf)
    stats["max_h_linf"] = max(float(stats.get("max_h_linf", 0.0)), h_linf)

    # 任一约束违规即计数一次，便于最终报告告警。
    if _has_constraint_violation(perturb_info, config):
        stats["constraint_violation_count"] = int(stats.get("constraint_violation_count", 0)) + 1

    return stats


def update_perturbation_stats(stats: dict, perturb_info: dict, config) -> dict:
    """
    作用:
        兼容阶段 5 文档中的函数名，等价调用 update_attack_step_stats。
    输入格式:
        stats: dict，累计统计。
        perturb_info: dict，当前 step 扰动统计。
        config: AttackConfig 或同字段对象。
    输出格式:
        dict，原 stats 对象，已原地更新。
    核心步骤:
        1. 将调用转发给 update_attack_step_stats。
    """
    return update_attack_step_stats(stats, perturb_info, config)


def _canonical_action(action):
    """
    作用:
        将 DQN int 与 DDPG assignment 统一成可稳定比较的 Python 对象。
    输入格式:
        action: int / np.integer / list / tuple / np.ndarray。
    输出格式:
        int 或 tuple[int, ...]。
    核心步骤:
        1. numpy 标量转 Python int。
        2. 序列或数组展平后转 tuple[int,...]。
        3. 其他标量尝试转 int。
    """
    # DQN 常见返回为 int 或 np.int64。
    if isinstance(action, (int, np.integer)):
        return int(action)
    # DDPG assignment 常见返回为 list / tuple / ndarray。
    if isinstance(action, (Sequence, np.ndarray)) and not isinstance(action, (str, bytes)):
        return tuple(int(x) for x in np.asarray(action).reshape(-1).tolist())
    # 兜底处理其他数值标量。
    return int(action)


def update_action_flip_stats(stats: dict, clean_action, attacked_action) -> dict:
    """
    作用:
        比较 clean_action 与 attacked_action 是否不同，并更新动作翻转次数。
    输入格式:
        stats: dict，累计统计。
        clean_action: DQN 为 int，DDPG 为 assignment tuple/list。
        attacked_action: DQN 为 int，DDPG 为 assignment tuple/list。
    输出格式:
        dict，原 stats 对象，已原地更新。
    核心步骤:
        1. 将两个动作规范化为 int 或 tuple。
        2. DQN 按 action_id 比较，DDPG 按 assignment tuple 比较。
        3. 若不同，则 action_flip_count 加 1。
    """
    # total_steps 由 update_attack_step_stats 维护，避免后续接入时重复计步。
    if _canonical_action(clean_action) != _canonical_action(attacked_action):
        stats["action_flip_count"] = int(stats.get("action_flip_count", 0)) + 1
    return stats


def summarize_attack_stats(stats: dict) -> dict[str, float | int]:
    """
    作用:
        将累计统计转换为最终 JSON 报告所需的比例和平均指标。
    输入格式:
        stats: dict，init_attack_stats 返回并持续更新的累计统计。
    输出格式:
        dict[str, float | int]，包含 attack_step_ratio、action_flip_ratio、avg_l0、avg_l1 等字段。
    核心步骤:
        1. 使用 total_steps 计算按 step 的比例和平均 L0。
        2. 使用 attack_step_count 计算每个攻击步的平均 L1。
        3. 原样带出最大 L_inf 与约束违规次数。
    """
    # 防御性分母，避免空评估时除零。
    total_steps = max(1, int(stats.get("total_steps", 0)))
    attack_steps = max(1, int(stats.get("attack_step_count", 0)))

    return {
        "attack_step_ratio": float(stats.get("attack_step_count", 0)) / float(total_steps),
        "action_flip_ratio": float(stats.get("action_flip_count", 0)) / float(total_steps),
        "avg_aoi_l0_per_step": float(stats.get("total_aoi_l0", 0)) / float(total_steps),
        "avg_h_l0_per_step": float(stats.get("total_h_l0", 0)) / float(total_steps),
        "avg_total_l0_per_step": float(stats.get("total_l0", 0)) / float(total_steps),
        "avg_aoi_l1_per_attack": float(stats.get("total_aoi_l1", 0.0)) / float(attack_steps),
        "avg_h_l1_per_attack": float(stats.get("total_h_l1", 0.0)) / float(attack_steps),
        "max_aoi_linf": float(stats.get("max_aoi_linf", 0.0)),
        "max_h_linf": float(stats.get("max_h_linf", 0.0)),
        "constraint_violation_count": int(stats.get("constraint_violation_count", 0)),
        "clean_selected_risk_sum": float(stats.get("total_clean_selected_risk_sum", 0.0)) / float(total_steps),
        "attack_selected_risk_sum": float(stats.get("total_attack_selected_risk_sum", 0.0)) / float(total_steps),
        "resource_misallocation_score": float(stats.get("total_resource_misallocation_score", 0.0)) / float(total_steps),
        "high_risk_scheduled_ratio_clean": float(stats.get("total_high_risk_scheduled_ratio_clean", 0.0)) / float(total_steps),
        "high_risk_scheduled_ratio_attack": float(stats.get("total_high_risk_scheduled_ratio_attack", 0.0)) / float(total_steps),
        "low_risk_scheduled_ratio_clean": float(stats.get("total_low_risk_scheduled_ratio_clean", 0.0)) / float(total_steps),
        "low_risk_scheduled_ratio_attack": float(stats.get("total_low_risk_scheduled_ratio_attack", 0.0)) / float(total_steps),
        "high_risk_good_channel_ratio_clean": float(stats.get("total_high_risk_good_channel_ratio_clean", 0.0)) / float(total_steps),
        "high_risk_good_channel_ratio_attack": float(stats.get("total_high_risk_good_channel_ratio_attack", 0.0)) / float(total_steps),
    }
