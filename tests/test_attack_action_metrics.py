from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class DummyDQNAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, float]] = []

    def select_action(self, state, epsilon=1.0):
        self.calls.append((np.asarray(state, dtype=np.float32).copy(), float(epsilon)))
        return np.int64(2)


class DummyDDPGAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, float]] = []

    def select_action(self, state, noise_std=1.0):
        self.calls.append((np.asarray(state, dtype=np.float32).copy(), float(noise_std)))
        return np.array([0.1, 0.9, 0.4], dtype=np.float32)


class DummyEnv:
    def __init__(self) -> None:
        self.n = 3
        self.m = 2
        self.max_aoi = 10
        self.channel_bins = [0, 1, 2, 3]


def test_select_dqn_action_uses_greedy_policy_and_returns_int() -> None:
    from attacks.action_utils import select_action_for_eval, select_dqn_action

    state = np.array([1, 2, 3], dtype=np.float32)
    agent = DummyDQNAgent()

    action_id = select_dqn_action(state, agent)
    unified_action = select_action_for_eval("dqn", state, agent, env=None)

    assert action_id == 2
    assert isinstance(action_id, int)
    assert unified_action == 2
    assert agent.calls[0][1] == 0.0
    np.testing.assert_array_equal(agent.calls[0][0], state)


def test_select_ddpg_assignment_uses_raw_state_and_maps_to_tuple() -> None:
    import attacks.action_utils as action_utils

    env = DummyEnv()
    agent = DummyDDPGAgent()
    state = np.array([5, 10, 1, 0, 2, 4, 1, 3, 0], dtype=np.float32)

    assignment = action_utils.select_ddpg_assignment(state, agent, env)
    unified_assignment = action_utils.select_action_for_eval("DDPG", state, agent, env)

    assert assignment == (0, 1, 2)
    assert unified_assignment == (0, 1, 2)
    assert isinstance(assignment, tuple)
    assert agent.calls[0][1] == 0.0
    np.testing.assert_array_equal(agent.calls[0][0], state)


def test_update_attack_step_stats_accumulates_perturbation_and_constraint_violation() -> None:
    from attacks.metrics import init_attack_stats, update_attack_step_stats

    config = SimpleNamespace(
        max_aoi_features=1,
        max_h_features=2,
        max_total_features=3,
        aoi_delta=1,
        h_delta=1,
    )
    stats = init_attack_stats()

    update_attack_step_stats(
        stats,
        {
            "is_attacked": True,
            "aoi_l0": 2,
            "h_l0": 1,
            "total_l0": 3,
            "aoi_l1": 4.5,
            "h_l1": 2.0,
            "aoi_linf": 2.0,
            "h_linf": 1.0,
        },
        config,
    )
    update_attack_step_stats(stats, {"is_attacked": False}, config)

    assert stats["total_steps"] == 2
    assert stats["attack_step_count"] == 1
    assert stats["total_aoi_l0"] == 2
    assert stats["total_h_l0"] == 1
    assert stats["total_l0"] == 3
    assert stats["total_aoi_l1"] == 4.5
    assert stats["total_h_l1"] == 2.0
    assert stats["max_aoi_linf"] == 2.0
    assert stats["max_h_linf"] == 1.0
    assert stats["constraint_violation_count"] == 1


def test_action_flip_stats_compare_int_and_tuple_actions() -> None:
    from attacks.metrics import init_attack_stats, update_action_flip_stats

    stats = init_attack_stats()
    stats["total_steps"] = 4

    update_action_flip_stats(stats, 1, np.int64(1))
    update_action_flip_stats(stats, 1, 2)
    update_action_flip_stats(stats, [0, 1, 2], (0, 1, 2))
    update_action_flip_stats(stats, (0, 1, 2), (0, 2, 1))

    assert stats["total_steps"] == 4
    assert stats["action_flip_count"] == 2


def test_summarize_attack_stats_returns_ratios_and_averages() -> None:
    from attacks.metrics import summarize_attack_stats

    summary = summarize_attack_stats(
        {
            "total_steps": 4,
            "attack_step_count": 2,
            "action_flip_count": 1,
            "total_aoi_l0": 6,
            "total_h_l0": 2,
            "total_l0": 8,
            "total_aoi_l1": 10.0,
            "total_h_l1": 4.0,
            "max_aoi_linf": 3.0,
            "max_h_linf": 2.0,
            "constraint_violation_count": 1,
        }
    )

    assert summary == {
        "attack_step_ratio": 0.5,
        "action_flip_ratio": 0.25,
        "avg_aoi_l0_per_step": 1.5,
        "avg_h_l0_per_step": 0.5,
        "avg_total_l0_per_step": 2.0,
        "avg_aoi_l1_per_attack": 5.0,
        "avg_h_l1_per_attack": 2.0,
        "max_aoi_linf": 3.0,
        "max_h_linf": 2.0,
        "constraint_violation_count": 1,
    }
