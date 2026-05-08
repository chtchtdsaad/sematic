from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from attacks.attack_config import AttackConfig


class DummyAttackEnv:
    def __init__(self) -> None:
        self.n = 2
        self.m = 1
        self.max_aoi = 5
        self.episode_length = 3
        self.aoi = np.array([1, 1], dtype=np.int64)
        self.channel_state = np.array([[0], [0]], dtype=np.int64)
        self.channel_loss = np.array([[0.0], [0.0]], dtype=np.float64)
        self.actions: list[int] = []
        self._step = 0

    def reset(self) -> np.ndarray:
        self._step = 0
        self.aoi = np.array([1, 1], dtype=np.int64)
        self.channel_state = np.array([[0], [0]], dtype=np.int64)
        self.channel_loss = np.array([[0.0], [0.0]], dtype=np.float64)
        return np.array([1, 1, 0, 0], dtype=np.float32)

    def step(self, action_id: int):
        self.actions.append(int(action_id))
        self._step += 1
        self.aoi = np.array([1 + int(action_id), 1], dtype=np.int64)
        next_state = np.array([1, 1, 0, 0], dtype=np.float32)
        reward = -float(10 + int(action_id))
        done = self._step >= self.episode_length
        return next_state, reward, done


class ThresholdDQNAgent:
    def select_action(self, state, epsilon: float = 0.0) -> int:
        return int(np.max(np.asarray(state, dtype=np.float32)[:2]) >= 3)


def test_run_attack_eval_uses_attacked_action_and_collects_stats() -> None:
    from eval_attack import run_attack_eval

    env = DummyAttackEnv()
    agent = ThresholdDQNAgent()
    config = AttackConfig(
        mode="random_aoi",
        seed=123,
        attack_prob=1.0,
        strict_budget=True,
        max_attack_ratio=1.0,
        max_consecutive_steps=10,
        aoi_delta=2,
        h_delta=1,
        max_aoi_features=2,
        max_h_features=1,
        max_total_features=2,
        aoi_direction="increase",
        h_direction="random",
    )

    result = run_attack_eval(
        algo="DQN",
        env=env,
        agent=agent,
        attack_config=config,
        eval_episodes=1,
        episode_length=3,
        verbose=False,
    )

    assert env.actions == [1, 1, 1]
    assert result["attack_mode"] == "random_aoi"
    assert result["mean_mse"] == 11.0
    assert result["mean_sum_aoi"] == 3.0
    assert result["attack_step_ratio"] == 1.0
    assert result["action_flip_ratio"] == 1.0
    assert result["constraint_violation_count"] == 0


def test_save_attack_report_writes_flat_attack_payload() -> None:
    from eval_attack import save_attack_report

    report_dir = Path("tests") / "_tmp_attack_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    args = Namespace(
        algo="DQN",
        scenario="base",
        seed=123,
        device="cpu",
        eval_episodes=1,
        save_report=1,
    )
    env = DummyAttackEnv()
    clean_result = {"mean_mse": 10.0, "mean_sum_aoi": 2.0}
    attack_result = {
        "attack_mode": "random_aoi",
        "mean_mse": 11.0,
        "mean_sum_aoi": 3.0,
        "attack_step_ratio": 1.0,
        "action_flip_ratio": 1.0,
        "avg_aoi_l0_per_step": 2.0,
        "avg_h_l0_per_step": 0.0,
        "avg_total_l0_per_step": 2.0,
        "avg_aoi_l1_per_attack": 4.0,
        "avg_h_l1_per_attack": 0.0,
        "max_aoi_linf": 2.0,
        "max_h_linf": 0.0,
        "constraint_violation_count": 0,
    }
    config = AttackConfig(mode="random_aoi", seed=123, attack_prob=1.0, max_attack_ratio=1.0)

    report_path = save_attack_report(
        args=args,
        env=env,
        ckpt_path=report_dir / "model.pt",
        clean_result=clean_result,
        attack_result=attack_result,
        attack_config=config,
        report_dir=report_dir,
    )

    payload = report_path.read_text(encoding="utf-8")
    assert '"clean_mean_mse": 10.0' in payload
    assert '"attack_mean_mse": 11.0' in payload
    assert '"mse_degradation": 1.0' in payload
    assert '"attack_step_ratio": 1.0' in payload
    report_path.unlink()
