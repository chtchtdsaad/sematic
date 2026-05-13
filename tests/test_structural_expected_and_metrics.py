"""
结构化 expected-cost 与结构指标测试文件（test_structural_expected_and_metrics.py）
===============================================================================
本文件验证结构化公共工具、expected-cost 攻击、结构性资源错配指标和报告字段。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from attacks.attack_config import AttackConfig


def make_env():
    """构造轻量 env，用于结构化攻击单元测试。"""
    # 使用 SimpleNamespace 减少对真实环境的依赖。
    env = SimpleNamespace()
    # sensor 数。
    env.n = 3
    # channel 数。
    env.m = 2
    # 最大 AoI。
    env.max_aoi = 5
    # DQN action_space。
    env.action_space = [(1, 2, 0), (1, 0, 2), (0, 1, 2)]
    # 每个 sensor 的稳态协方差。
    env.p_bars = [np.eye(2, dtype=np.float64) * float(i + 1) for i in range(env.n)]
    # 每个 sensor 的系统矩阵。
    env.a_mats = [np.eye(2, dtype=np.float64) for _ in range(env.n)]
    # 每个 sensor 的过程噪声。
    env.w_mats = [np.eye(2, dtype=np.float64) for _ in range(env.n)]
    # A^k 缓存。
    env.a_powers_cache = [np.repeat(np.eye(2, dtype=np.float64)[None, :, :], env.max_aoi + 1, axis=0) for _ in range(env.n)]
    # 噪声累计缓存。
    env.noise_cov_sums_cache = [
        np.stack([float(k) * np.eye(2, dtype=np.float64) for k in range(env.max_aoi + 1)], axis=0)
        for _ in range(env.n)
    ]
    # 返回 env。
    return env


class DummyDQNAgent:
    """根据 AoI 大小返回 DQN action_id 的测试 agent。"""

    def select_action(self, state, epsilon: float = 0.0):
        # AoI 最大值高时选择 action 1，否则选择 action 0。
        return int(np.max(np.asarray(state, dtype=np.float32)[:3]) >= 4)


class DummyDDPGAgent:
    """返回固定连续动作的测试 DDPG agent。"""

    def select_action(self, state, noise_std: float = 0.0):
        # 三个 sensor 对应映射到 0/1/2 信道编号。
        return np.array([0.1, 0.9, 0.4], dtype=np.float32)


def make_config(mode: str = "semantic_joint_expected_cost", **overrides) -> AttackConfig:
    """构造 expected-cost 测试配置。"""
    # 默认配置保证每次都触发攻击。
    data = {
        "mode": mode,
        "seed": 7,
        "attack_prob": 1.0,
        "strict_budget": True,
        "max_attack_ratio": 1.0,
        "max_consecutive_steps": 10,
        "cooldown_steps": 0,
        "aoi_delta": 2,
        "h_delta": 2,
        "max_aoi_features": 2,
        "max_h_features": 3,
        "max_total_features": 4,
        "expected_cost_mode": "mse",
    }
    # 应用测试覆盖字段。
    data.update(overrides)
    # 返回 AttackConfig。
    return AttackConfig(**data)


def make_attack_state():
    """构造 episode 内攻击状态。"""
    # max_attack_steps 给足，避免预算阻塞。
    return {"attack_steps_used": 0, "max_attack_steps": 10, "consecutive_attack_steps": 0, "cooldown_remaining": 0}


def test_decode_action_and_expected_cost_support_dqn_and_ddpg() -> None:
    from attacks.structural_common import decode_eval_action_to_assignment, estimate_one_step_expected_cost

    env = make_env()
    state = np.array([2, 3, 4, 4, 3, 2, 1, 0, 4], dtype=np.float32)

    assert decode_eval_action_to_assignment("DQN", 1, env) == (1, 0, 2)
    assert decode_eval_action_to_assignment("DDPG", (0, 1, 2), env) == (0, 1, 2)
    assert estimate_one_step_expected_cost(state, env, "DQN", 1, cost_mode="mse") > 0.0
    assert estimate_one_step_expected_cost(state, env, "DDPG", (0, 1, 2), cost_mode="sum_aoi") > 0.0


def test_joint_expected_cost_attack_mutates_copy_and_respects_budget() -> None:
    from attacks.structural_expected import semantic_joint_expected_cost_attack

    env = make_env()
    state = np.array([2, 3, 4, 4, 3, 2, 1, 0, 4], dtype=np.float32)
    env_snapshot = {name: getattr(env, name) for name in ("n", "m", "max_aoi")}
    config = make_config(expected_cost_mode="sum_aoi")
    attacked_state, info = semantic_joint_expected_cost_attack(
        state=state,
        env=env,
        config=config,
        rng=np.random.default_rng(5),
        attack_state=make_attack_state(),
        algo="DQN",
        agent=DummyDQNAgent(),
    )

    assert attacked_state is not state
    np.testing.assert_array_equal(state, np.array([2, 3, 4, 4, 3, 2, 1, 0, 4], dtype=np.float32))
    assert info["is_attacked"] is True
    assert info["total_l0"] == info["aoi_l0"] + info["h_l0"]
    assert info["total_l0"] <= config.max_total_features
    assert info["structural_attack_type"] == "semantic_joint_expected_cost"
    assert "expected_cost_score" in info
    assert {name: getattr(env, name) for name in ("n", "m", "max_aoi")} == env_snapshot


def test_structural_action_metrics_explain_resource_shift() -> None:
    from attacks.structural_metrics import compute_structural_action_metrics

    env = make_env()
    state = np.array([2, 3, 4, 4, 3, 2, 1, 0, 4], dtype=np.float32)
    metrics = compute_structural_action_metrics(state, env, "DQN", clean_action=0, attacked_action=2)

    assert metrics["resource_misallocation_score"] == metrics["clean_selected_risk_sum"] - metrics["attack_selected_risk_sum"]
    assert 0.0 <= metrics["high_risk_scheduled_ratio_clean"] <= 1.0
    assert 0.0 <= metrics["high_risk_scheduled_ratio_attack"] <= 1.0


def test_save_attack_report_includes_expected_cost_and_structural_metrics(tmp_path: Path) -> None:
    from eval_attack import save_attack_report

    env = make_env()
    args = Namespace(algo="DQN", scenario="base", seed=24, device="cpu", eval_episodes=1)
    clean_result = {"mean_mse": 10.0, "mean_sum_aoi": 2.0}
    attack_result = {
        "mean_mse": 12.0,
        "mean_sum_aoi": 3.0,
        "attack_step_ratio": 1.0,
        "action_flip_ratio": 0.5,
        "constraint_violation_count": 0,
        "resource_misallocation_score": 2.0,
        "high_risk_scheduled_ratio_clean": 1.0,
        "high_risk_scheduled_ratio_attack": 0.0,
    }
    config = make_config("semantic_joint_expected_cost", expected_cost_mode="sum_aoi")

    report_path = save_attack_report(
        args=args,
        env=env,
        ckpt_path=tmp_path / "model.pt",
        clean_result=clean_result,
        attack_result=attack_result,
        attack_config=config,
        report_dir=tmp_path,
    )

    payload = report_path.read_text(encoding="utf-8")
    assert '"expected_cost_mode": "sum_aoi"' in payload
    assert '"resource_misallocation_score": 2.0' in payload
    assert '"high_risk_scheduled_ratio_clean": 1.0' in payload
