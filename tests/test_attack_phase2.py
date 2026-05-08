from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pytest


class DummyEnv:
    def __init__(self) -> None:
        self.n = 3
        self.m = 2
        self.max_aoi = 5


def test_attack_config_keeps_phase2_fields() -> None:
    from attacks.attack_config import AttackConfig

    config = AttackConfig(
        mode="random_aoi",
        seed=24,
        attack_prob=0.5,
        strict_budget=True,
        max_attack_ratio=0.25,
        max_consecutive_steps=4,
        cooldown_steps=1,
        aoi_delta=2,
        h_delta=1,
        max_aoi_features=1,
        max_h_features=2,
        max_total_features=3,
        aoi_direction="increase",
        h_direction="random",
        record_perturbation=True,
    )

    assert asdict(config) == {
        "mode": "random_aoi",
        "seed": 24,
        "attack_prob": 0.5,
        "strict_budget": True,
        "max_attack_ratio": 0.25,
        "max_consecutive_steps": 4,
        "cooldown_steps": 1,
        "aoi_delta": 2,
        "h_delta": 1,
        "max_aoi_features": 1,
        "max_h_features": 2,
        "max_total_features": 3,
        "aoi_direction": "increase",
        "h_direction": "random",
        "record_perturbation": True,
    }


def test_build_attack_config_from_args_converts_integer_flags() -> None:
    from attacks.attack_config import build_attack_config_from_args

    args = Namespace(
        attack_mode="random_joint",
        seed=42,
        attack_prob=0.3,
        strict_budget=0,
        max_attack_ratio=0.4,
        max_consecutive_steps=6,
        cooldown_steps=2,
        aoi_delta=2,
        h_delta=3,
        max_aoi_features=2,
        max_h_features=4,
        max_total_features=5,
        aoi_direction="mixed",
        h_direction="decrease",
        record_perturbation=0,
    )

    config = build_attack_config_from_args(args)

    assert config.mode == "random_joint"
    assert config.seed == 42
    assert config.strict_budget is False
    assert config.record_perturbation is False
    assert config.max_total_features == 5
    assert config.aoi_direction == "mixed"
    assert config.h_direction == "decrease"


def test_split_and_merge_state_round_trip() -> None:
    from attacks.state_ops import merge_state, split_state

    env = DummyEnv()
    state = np.array([1, 2, 3, 0, 1, 2, 3, 4, 0], dtype=np.float32)

    aoi_part, h_part = split_state(state, env)
    merged = merge_state(aoi_part, h_part)

    np.testing.assert_array_equal(aoi_part, np.array([1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(h_part, np.array([0, 1, 2, 3, 4, 0], dtype=np.float32))
    np.testing.assert_array_equal(merged, state)
    assert merged.dtype == np.float32


def test_split_state_rejects_wrong_state_length() -> None:
    from attacks.state_ops import split_state

    env = DummyEnv()
    state = np.array([1, 2, 3, 0, 1], dtype=np.float32)

    with pytest.raises(ValueError, match="state length"):
        split_state(state, env)


def test_project_aoi_and_h_parts_clip_to_valid_ranges() -> None:
    from attacks.state_ops import project_aoi_part, project_h_part

    env = DummyEnv()
    aoi_part = np.array([-2.2, 1.4, 6.8], dtype=np.float32)
    h_part = np.array([-1.2, 0.4, 1.6, 3.5, 4.9, 9.1], dtype=np.float32)

    projected_aoi = project_aoi_part(aoi_part, env)
    projected_h = project_h_part(h_part)

    np.testing.assert_array_equal(projected_aoi, np.array([1, 1, 5], dtype=np.float32))
    np.testing.assert_array_equal(projected_h, np.array([0, 0, 2, 4, 4, 4], dtype=np.float32))
    assert projected_aoi.dtype == np.float32
    assert projected_h.dtype == np.float32
