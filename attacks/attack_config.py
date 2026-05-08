"""
攻击配置模块（attack_config.py）
==============================
本文件只保存攻击评估所需的配置数据结构与 CLI 参数转配置逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttackConfig:
    """
    作用:
        保存状态观测扰动攻击的全部约束参数。

    输入格式:
        mode: str，攻击模式，例如 "clean" / "random_aoi" / "random_h" / "random_joint"。
        seed: int，攻击随机数种子。
        attack_prob: float，每个时间步触发攻击的概率，建议范围 [0,1]。
        strict_budget: bool，是否严格执行 episode 级攻击预算。
        max_attack_ratio: float，episode 内最多攻击步比例。
        max_consecutive_steps: int，最大连续攻击步数。
        cooldown_steps: int，预留冷却步数字段。
        aoi_delta: int，AoI 单维最大扰动幅值。
        h_delta: int，H 单维最大扰动幅值。
        max_aoi_features: int，每步最多扰动 AoI 维度数。
        max_h_features: int，每步最多扰动 H 维度数。
        max_total_features: int，每步最多扰动总维度数。
        aoi_direction: str，AoI 扰动方向。
        h_direction: str，H 扰动方向。
        record_perturbation: bool，是否记录扰动统计。

    输出格式:
        AttackConfig 实例。

    核心步骤:
        1. 记录攻击模式。
        2. 记录时间预算、幅值预算和稀疏预算。
        3. 记录 AoI/H 扰动方向与统计开关。
    """

    mode: str = "clean"
    seed: int = 42

    attack_prob: float = 0.2
    strict_budget: bool = True
    max_attack_ratio: float = 0.2
    max_consecutive_steps: int = 5
    cooldown_steps: int = 0

    aoi_delta: int = 1
    h_delta: int = 1

    max_aoi_features: int = 1
    max_h_features: int = 1
    max_total_features: int = 2

    aoi_direction: str = "random"
    h_direction: str = "random"

    record_perturbation: bool = True


def build_attack_config_from_args(args) -> AttackConfig:
    """
    作用:
        从 argparse.Namespace 构造 AttackConfig。

    输入格式:
        args: argparse.Namespace，需包含 eval_attack.py 中的攻击 CLI 参数。

    输出格式:
        AttackConfig 实例。

    核心步骤:
        1. 读取 CLI 中的攻击模式、预算、幅值和方向参数。
        2. 将 0/1 整数开关转换为 bool。
        3. 返回后续攻击模块可直接使用的配置对象。
    """
    # 从 CLI 参数读取 attack_mode，内部配置字段统一命名为 mode。
    mode = str(getattr(args, "attack_mode", "clean"))

    # 将 argparse 的 0/1 整数开关转为 bool，便于后续逻辑判断。
    strict_budget = bool(int(getattr(args, "strict_budget", 1)))
    record_perturbation = bool(int(getattr(args, "record_perturbation", 1)))

    # 构造并返回攻击配置对象。
    return AttackConfig(
        mode=mode,
        seed=int(getattr(args, "seed", 42)),
        attack_prob=float(getattr(args, "attack_prob", 0.2)),
        strict_budget=strict_budget,
        max_attack_ratio=float(getattr(args, "max_attack_ratio", 0.2)),
        max_consecutive_steps=int(getattr(args, "max_consecutive_steps", 5)),
        cooldown_steps=int(getattr(args, "cooldown_steps", 0)),
        aoi_delta=int(getattr(args, "aoi_delta", 1)),
        h_delta=int(getattr(args, "h_delta", 1)),
        max_aoi_features=int(getattr(args, "max_aoi_features", 1)),
        max_h_features=int(getattr(args, "max_h_features", 1)),
        max_total_features=int(getattr(args, "max_total_features", 2)),
        aoi_direction=str(getattr(args, "aoi_direction", "random")),
        h_direction=str(getattr(args, "h_direction", "random")),
        record_perturbation=record_perturbation,
    )
