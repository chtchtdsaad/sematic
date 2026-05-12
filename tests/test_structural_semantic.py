from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def make_config(**overrides):
    config = {
        "mode": "semantic_joint_mislead",
        "attack_prob": 1.0,
        "strict_budget": True,
        "aoi_delta": 2,
        "h_delta": 2,
        "max_aoi_features": 2,
        "max_h_features": 3,
        "max_total_features": 5,
        "max_consecutive_steps": 10,
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


def make_env():
    env = SimpleNamespace()
    env.n = 3
    env.m = 2
    env.max_aoi = 5
    env.aoi = np.ones(env.n, dtype=np.int64)
    env.channel_state = np.zeros((env.n, env.m), dtype=np.int64)
    env.channel_loss = np.zeros((env.n, env.m), dtype=np.float64)
    env.p_bars = [
        np.eye(2, dtype=np.float64),
        1.5 * np.eye(2, dtype=np.float64),
        2.0 * np.eye(2, dtype=np.float64),
    ]
    env.a_mats = [np.eye(2, dtype=np.float64) for _ in range(env.n)]
    env.w_mats = [np.eye(2, dtype=np.float64) for _ in range(env.n)]
    env.a_powers_cache = [np.repeat(np.eye(2, dtype=np.float64)[None, :, :], env.max_aoi + 1, axis=0) for _ in range(env.n)]
    env.noise_cov_sums_cache = [
        np.stack([float(k) * np.eye(2, dtype=np.float64) for k in range(env.max_aoi + 1)], axis=0)
        for _ in range(env.n)
    ]
    return env


def test_structural_helpers_compute_shapes_and_channel_mapping() -> None:
    from config import PACKET_LOSS_LEVELS
    from attacks.structural_semantic import (
        compute_link_priority,
        compute_sensor_mse_values,
        compute_sensor_risk,
        flatten_h_index,
        h_to_success_prob,
        unflatten_h_index,
    )

    env = make_env()
    aoi_part = np.array([1, 2, 3], dtype=np.float32)
    h_part = np.array([0, 1, 2, 3, 4, 0], dtype=np.float32)

    mse_values = compute_sensor_mse_values(aoi_part, env)
    risk = compute_sensor_risk(aoi_part, env)
    success_prob = h_to_success_prob(h_part)
    priority, priority_risk, priority_success = compute_link_priority(aoi_part, h_part, env)

    assert mse_values.shape == (env.n,)
    assert mse_values.dtype == np.float64
    assert np.all(mse_values > 0.0)
    assert risk.shape == (env.n,)
    assert priority.shape == (env.n, env.m)
    np.testing.assert_allclose(priority_risk, risk)
    np.testing.assert_allclose(priority_success, success_prob.reshape(env.n, env.m))
    np.testing.assert_allclose(success_prob, 1.0 - PACKET_LOSS_LEVELS[h_part.astype(np.int64)])

    for sensor_idx in range(env.n):
        for channel_idx in range(env.m):
            flat_idx = flatten_h_index(sensor_idx, channel_idx, env)
            assert flat_idx == sensor_idx * env.m + channel_idx
            assert unflatten_h_index(flat_idx, env) == (sensor_idx, channel_idx)


def test_structural_mislead_attacks_respect_budgets_and_do_not_mutate_env() -> None:
    from attacks.structural_semantic import (
        semantic_aoi_mislead_attack,
        semantic_h_mislead_attack,
        semantic_joint_mislead_attack,
    )

    env = make_env()
    state = np.array([5, 4, 2, 4, 3, 0, 1, 2, 0], dtype=np.float32)
    env_aoi = env.aoi.copy()
    env_h = env.channel_state.copy()
    env_loss = env.channel_loss.copy()

    cases = [
        (
            "semantic_aoi_mislead",
            semantic_aoi_mislead_attack,
            make_config(mode="semantic_aoi_mislead", max_aoi_features=2, max_total_features=2),
        ),
        (
            "semantic_h_mislead",
            semantic_h_mislead_attack,
            make_config(mode="semantic_h_mislead", max_h_features=3, max_total_features=3),
        ),
        (
            "semantic_joint_mislead",
            semantic_joint_mislead_attack,
            make_config(mode="semantic_joint_mislead", max_aoi_features=2, max_h_features=3, max_total_features=5),
        ),
    ]

    for expected_type, attack_fn, config in cases:
        attack_state = make_attack_state()
        attacked_state, info = attack_fn(
            state=state,
            env=env,
            config=config,
            rng=np.random.default_rng(11),
            attack_state=attack_state,
        )

        np.testing.assert_array_equal(env.aoi, env_aoi)
        np.testing.assert_array_equal(env.channel_state, env_h)
        np.testing.assert_array_equal(env.channel_loss, env_loss)
        np.testing.assert_array_equal(state, np.array([5, 4, 2, 4, 3, 0, 1, 2, 0], dtype=np.float32))

        assert attacked_state is not state
        assert info["structural_attack_type"] == expected_type
        assert info["is_attacked"] is True
        assert info["total_l0"] == info["aoi_l0"] + info["h_l0"]
        assert info["total_l0"] <= config.max_total_features
        assert info["aoi_l0"] <= config.max_aoi_features
        assert info["h_l0"] <= config.max_h_features
        assert info["aoi_linf"] <= config.aoi_delta
        assert info["h_linf"] <= config.h_delta
        assert attack_state["attack_steps_used"] == 1
        assert np.all(attacked_state[: env.n] >= 1)
        assert np.all(attacked_state[: env.n] <= env.max_aoi)
        assert np.all(attacked_state[env.n :] >= 0)
        assert np.all(attacked_state[env.n :] <= 4)

        if expected_type == "semantic_aoi_mislead":
            np.testing.assert_array_equal(attacked_state[env.n :], state[env.n :])
            assert info["h_l0"] == 0
        elif expected_type == "semantic_h_mislead":
            np.testing.assert_array_equal(attacked_state[: env.n], state[: env.n])
            assert info["aoi_l0"] == 0


def test_structural_h_mislead_backfills_clipped_suppression_candidates() -> None:
    from attacks.structural_semantic import semantic_h_mislead_attack

    env = make_env()
    state = np.array([5, 4, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    config = make_config(
        mode="semantic_h_mislead",
        h_delta=2,
        max_h_features=3,
        max_total_features=3,
    )

    attacked_state, info = semantic_h_mislead_attack(
        state=state,
        env=env,
        config=config,
        rng=np.random.default_rng(13),
        attack_state=make_attack_state(),
    )

    assert info["is_attacked"] is True
    assert info["aoi_l0"] == 0
    assert info["h_l0"] == 3
    assert info["total_l0"] == 3
    np.testing.assert_array_equal(attacked_state[: env.n], state[: env.n])
    assert np.all(attacked_state[env.n :] >= state[env.n :])


def test_eval_attack_generate_attacked_state_dispatches_structural_mode() -> None:
    from eval_attack import generate_attacked_state

    env = make_env()
    state = np.array([5, 4, 2, 4, 3, 0, 1, 2, 0], dtype=np.float32)
    config = make_config(mode="semantic_aoi_mislead", max_aoi_features=2, max_total_features=2)
    attack_state = make_attack_state()

    attacked_state, info = generate_attacked_state(
        state=state,
        env=env,
        config=config,
        rng=np.random.default_rng(17),
        attack_state=attack_state,
    )

    assert info["structural_attack_type"] == "semantic_aoi_mislead"
    assert info["is_attacked"] is True
    assert info["h_l0"] == 0
    np.testing.assert_array_equal(attacked_state[env.n :], state[env.n :])
