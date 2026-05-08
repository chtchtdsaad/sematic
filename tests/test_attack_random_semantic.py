from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class DummyEnv:
    def __init__(self) -> None:
        self.n = 3
        self.m = 2
        self.max_aoi = 5
        self.aoi = np.array([1, 2, 3], dtype=np.float32)
        self.channel_state = np.array([[0, 1], [2, 3], [4, 0]], dtype=np.float32)
        self.channel_loss = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)


def make_config(**overrides):
    config = {
        "mode": "random_joint",
        "attack_prob": 1.0,
        "strict_budget": True,
        "aoi_delta": 2,
        "h_delta": 1,
        "max_aoi_features": 2,
        "max_h_features": 2,
        "max_total_features": 3,
        "aoi_direction": "mixed",
        "h_direction": "mixed",
        "max_consecutive_steps": 5,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def make_attack_state(**overrides):
    state = {
        "attack_steps_used": 0,
        "max_attack_steps": 10,
        "consecutive_attack_steps": 0,
    }
    state.update(overrides)
    return state


def test_random_bias_aoi_projects_and_respects_constraints_without_input_mutation() -> None:
    from attacks.random_semantic import random_bias_aoi

    env = DummyEnv()
    config = make_config(max_aoi_features=2, max_total_features=2, aoi_direction="increase")
    rng = np.random.default_rng(7)
    aoi_part = np.array([2, 3, 4], dtype=np.float32)
    original = aoi_part.copy()

    attacked_aoi, info = random_bias_aoi(aoi_part, env, config, rng)

    np.testing.assert_array_equal(aoi_part, original)
    assert attacked_aoi.dtype == np.float32
    assert np.all(attacked_aoi >= 1)
    assert np.all(attacked_aoi <= env.max_aoi)
    assert np.all(attacked_aoi == np.rint(attacked_aoi))
    assert info["aoi_l0"] <= 2
    assert info["aoi_linf"] <= config.aoi_delta
    assert info["aoi_l1"] == float(np.sum(np.abs(attacked_aoi - original)))
    assert info["aoi_changed_indices"] == np.flatnonzero(attacked_aoi != original).astype(int).tolist()


def test_random_bias_h_projects_and_respects_constraints_without_input_mutation() -> None:
    from attacks.random_semantic import random_bias_h

    config = make_config(max_h_features=2, max_total_features=2, h_direction="decrease")
    rng = np.random.default_rng(11)
    h_part = np.array([1, 2, 3, 4, 2, 1], dtype=np.float32)
    original = h_part.copy()

    attacked_h, info = random_bias_h(h_part, config, rng)

    np.testing.assert_array_equal(h_part, original)
    assert attacked_h.dtype == np.float32
    assert np.all(attacked_h >= 0)
    assert np.all(attacked_h <= 4)
    assert np.all(attacked_h == np.rint(attacked_h))
    assert info["h_l0"] <= 2
    assert info["h_linf"] <= config.h_delta
    assert info["h_l1"] == float(np.sum(np.abs(attacked_h - original)))
    assert info["h_changed_indices"] == np.flatnonzero(attacked_h != original).astype(int).tolist()


def test_semantic_random_attack_clean_returns_state_copy_and_empty_stats() -> None:
    from attacks.random_semantic import semantic_random_attack

    env = DummyEnv()
    config = make_config(mode="clean")
    rng = np.random.default_rng(3)
    attack_state = make_attack_state()
    state = np.array([2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.float32)

    attacked_state, info = semantic_random_attack(state, env, config, rng, attack_state)

    assert attacked_state is not state
    np.testing.assert_array_equal(attacked_state, state)
    assert info["is_attacked"] is False
    assert info["total_l0"] == 0
    assert info["aoi_changed_indices"] == []
    assert info["h_changed_indices"] == []
    assert attack_state["attack_steps_used"] == 0


def test_semantic_random_attack_modes_only_modify_requested_state_segments_and_not_env() -> None:
    from attacks.random_semantic import semantic_random_attack

    env = DummyEnv()
    state = np.array([2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.float32)
    env_aoi = env.aoi.copy()
    env_h = env.channel_state.copy()
    env_loss = env.channel_loss.copy()

    for mode in ("random_aoi", "random_h", "random_joint"):
        config = make_config(mode=mode, max_aoi_features=1, max_h_features=2, max_total_features=2)
        attacked_state, info = semantic_random_attack(
            state,
            env,
            config,
            np.random.default_rng(19),
            make_attack_state(),
        )

        np.testing.assert_array_equal(env.aoi, env_aoi)
        np.testing.assert_array_equal(env.channel_state, env_h)
        np.testing.assert_array_equal(env.channel_loss, env_loss)
        np.testing.assert_array_equal(state, np.array([2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.float32))
        assert attacked_state is not state
        assert info["total_l0"] <= config.max_total_features
        assert info["aoi_l0"] <= config.max_aoi_features
        assert info["h_l0"] <= config.max_h_features
        assert info["aoi_linf"] <= config.aoi_delta
        assert info["h_linf"] <= config.h_delta
        assert np.all(attacked_state[: env.n] >= 1)
        assert np.all(attacked_state[: env.n] <= env.max_aoi)
        assert np.all(attacked_state[env.n :] >= 0)
        assert np.all(attacked_state[env.n :] <= 4)

        if mode == "random_aoi":
            np.testing.assert_array_equal(attacked_state[env.n :], state[env.n :])
            assert info["h_l0"] == 0
        elif mode == "random_h":
            np.testing.assert_array_equal(attacked_state[: env.n], state[: env.n])
            assert info["aoi_l0"] == 0


def test_semantic_random_attack_obeys_strict_budget_and_consecutive_limit() -> None:
    from attacks.random_semantic import semantic_random_attack

    env = DummyEnv()
    state = np.array([2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.float32)
    rng = np.random.default_rng(23)

    budget_config = make_config(mode="random_aoi", strict_budget=True)
    budget_state = make_attack_state(attack_steps_used=1, max_attack_steps=1)
    attacked_state, info = semantic_random_attack(state, env, budget_config, rng, budget_state)
    np.testing.assert_array_equal(attacked_state, state)
    assert info["is_attacked"] is False
    assert budget_state["attack_steps_used"] == 1

    consecutive_config = make_config(mode="random_h", max_consecutive_steps=2)
    consecutive_state = make_attack_state(consecutive_attack_steps=2)
    attacked_state, info = semantic_random_attack(state, env, consecutive_config, rng, consecutive_state)
    np.testing.assert_array_equal(attacked_state, state)
    assert info["is_attacked"] is False
    assert consecutive_state["consecutive_attack_steps"] == 0
