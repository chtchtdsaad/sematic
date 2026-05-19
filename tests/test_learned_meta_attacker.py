from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def make_learned_env():
    env = SimpleNamespace()
    env.n = 3
    env.m = 2
    env.max_aoi = 5
    env.state_dim = env.n + env.n * env.m
    env.aoi = np.ones(env.n, dtype=np.int64)
    env.channel_state = np.zeros((env.n, env.m), dtype=np.int64)
    env.channel_loss = np.zeros((env.n, env.m), dtype=np.float64)
    env.action_space = [(1, 2, 0)]
    env.last_action_id = None
    env.last_assignment = None

    def reset():
        env.aoi = np.ones(env.n, dtype=np.int64)
        env.channel_state = np.zeros((env.n, env.m), dtype=np.int64)
        return np.array([1, 1, 1, 0, 1, 2, 3, 4, 0], dtype=np.float32)

    def step(action_id):
        env.last_action_id = int(action_id)
        return np.array([2, 1, 2, 1, 2, 3, 4, 0, 1], dtype=np.float32), -12.5, False

    def step_assignment(assignment):
        env.last_assignment = tuple(int(x) for x in assignment)
        return np.array([2, 2, 1, 2, 3, 4, 0, 1, 2], dtype=np.float32), -22.0, False

    env.reset = reset
    env.step = step
    env.step_assignment = step_assignment
    return env


class DummyDQNVictim:
    def select_action(self, state, epsilon):
        assert epsilon == 0.0
        assert np.asarray(state).shape == (9,)
        return 0


class DummyDDPGVictim:
    def select_action(self, state, noise_std):
        assert noise_std == 0.0
        assert np.asarray(state).shape == (9,)
        return np.array([0.9, 0.2, -0.1], dtype=np.float32)


class DummyAttackAgent:
    """
    作用:
        提供 run_attacker_training 测试用的最小 attacker agent。

    输入格式:
        action_dim: int，continuous intent action 维度。

    输出格式:
        DummyAttackAgent 实例。

    参数含义:
        action_dim 决定 select_action 返回向量长度；该测试 agent 不更新网络。

    核心步骤:
        1. 保存 action_dim。
        2. select_action 返回全 1 intent，方便触发 deterministic eval 路径。
    """

    def __init__(self, action_dim: int) -> None:
        """
        作用:
            初始化测试 attacker。

        输入格式:
            action_dim: int。

        输出格式:
            None。

        参数含义:
            action_dim 是测试环境的 action 维度。
        """
        self.action_dim = int(action_dim)

    def select_action(self, state, noise_std):
        """
        作用:
            返回确定性 continuous intent action。

        输入格式:
            state: np.ndarray，当前 raw state。
            noise_std: float，测试中 deterministic eval 必须为 0。

        输出格式:
            np.ndarray[float32], shape=(action_dim,)。

        参数含义:
            state 只用于校验调用形状；noise_std 用于确认 deterministic eval 不加噪声。
        """
        assert float(noise_std) == 0.0
        assert np.asarray(state).reshape(-1).shape[0] == self.action_dim
        return np.ones(self.action_dim, dtype=np.float32)

    def update(self, replay_buffer, batch_size):
        """
        作用:
            提供不可达的 update 接口，防止测试误触发训练更新。

        输入格式:
            replay_buffer: ReplayBuffer。
            batch_size: int。

        输出格式:
            tuple[float,float]。

        参数含义:
            本测试通过 warmup/batch 配置避免调用 update。
        """
        raise AssertionError("DummyAttackAgent.update should not be called in this diagnostic test.")


def test_learned_attack_config_from_args_defaults_and_overrides() -> None:
    from attacks.learned_attack_config import build_learned_attack_config_from_args

    args = SimpleNamespace(
        attacker_episodes=7,
        per_step_energy_budget=3.5,
        save_attacker_checkpoints=0,
        victim_algo="DQN",
    )

    cfg = build_learned_attack_config_from_args(args)

    assert cfg.attacker_algo == "DDPG"
    assert cfg.attacker_episodes == 7
    assert cfg.per_step_energy_budget == 3.5
    assert cfg.save_attacker_checkpoints is False
    assert cfg.victim_algo == "DQN"
    assert cfg.alpha_tau == 1.0


def test_intent_mapping_respects_energy_and_does_not_mutate_env() -> None:
    from attacks.learned_attack_config import LearnedAttackConfig
    from attacks.learned_attack_env import LearnedAttackEnv

    env = make_learned_env()
    cfg = LearnedAttackConfig(per_step_energy_budget=2.0, alpha_tau=1.0, alpha_h=0.25, aoi_delta=1, h_delta=1)
    attack_env = LearnedAttackEnv(env, DummyDQNVictim(), "DQN", cfg)
    state = np.array([5, 4, 2, 4, 3, 0, 1, 2, 0], dtype=np.float32)
    intent = np.ones(env.state_dim, dtype=np.float32)
    env_aoi = env.aoi.copy()
    env_h = env.channel_state.copy()
    env_loss = env.channel_loss.copy()

    attacked_state, delta, info = attack_env.map_intent_to_attacked_state(state, intent)

    np.testing.assert_array_equal(env.aoi, env_aoi)
    np.testing.assert_array_equal(env.channel_state, env_h)
    np.testing.assert_array_equal(env.channel_loss, env_loss)
    assert attacked_state.shape == state.shape
    assert delta.shape == state.shape
    assert info["energy_used"] <= cfg.per_step_energy_budget + 1e-6
    assert info["aoi_energy"] + info["h_energy"] == info["energy_used"]
    assert info["constraint_violation"] is False
    assert info["total_l0"] == info["aoi_l0"] + info["h_l0"]
    assert np.all(attacked_state[: env.n] >= 1)
    assert np.all(attacked_state[: env.n] <= env.max_aoi)
    assert np.all(attacked_state[env.n :] >= 0)
    assert np.all(attacked_state[env.n :] <= 4)


def test_learned_attack_env_step_supports_dqn_and_ddpg_victims() -> None:
    from attacks.learned_attack_config import LearnedAttackConfig
    from attacks.learned_attack_env import LearnedAttackEnv

    cfg = LearnedAttackConfig(per_step_energy_budget=2.0)

    dqn_env = make_learned_env()
    dqn_attack_env = LearnedAttackEnv(dqn_env, DummyDQNVictim(), "DQN", cfg)
    dqn_attack_env.reset()
    next_state, reward, done, info = dqn_attack_env.step(np.zeros(dqn_env.state_dim, dtype=np.float32))
    assert dqn_env.last_action_id == 0
    assert reward == 12.5
    assert done is False
    assert info["victim_algo"] == "DQN"
    assert next_state.shape == (dqn_env.state_dim,)

    ddpg_env = make_learned_env()
    ddpg_attack_env = LearnedAttackEnv(ddpg_env, DummyDDPGVictim(), "DDPG", cfg)
    ddpg_attack_env.reset()
    next_state, reward, done, info = ddpg_attack_env.step(np.zeros(ddpg_env.state_dim, dtype=np.float32))
    assert sorted(x for x in ddpg_env.last_assignment if x > 0) == [1, 2]
    assert reward == 22.0
    assert done is False
    assert info["victim_algo"] == "DDPG"
    assert next_state.shape == (ddpg_env.state_dim,)


def test_attack_ddpg_update_and_checkpoint_roundtrip(tmp_path) -> None:
    from agent import ReplayBuffer
    from attacks.attack_agent import AttackDDPGAgent
    from train import load_attacker_checkpoint, save_attacker_checkpoint

    agent = AttackDDPGAgent(state_dim=6, action_dim=6, device="cpu")
    buffer = ReplayBuffer(capacity=16)
    for idx in range(8):
        state = np.full(6, float(idx), dtype=np.float32)
        action = np.clip(np.linspace(-1, 1, 6, dtype=np.float32), -1, 1)
        next_state = state + 1.0
        buffer.push(state, action, 0.1 * idx, next_state, False)

    critic_loss, actor_loss = agent.update(buffer, batch_size=4)
    assert isinstance(critic_loss, float)
    assert isinstance(actor_loss, float)

    ckpt_path = save_attacker_checkpoint(agent, tmp_path / "attacker.pt", meta={"episode": 1})
    restored = AttackDDPGAgent(state_dim=6, action_dim=6, device="cpu")
    meta = load_attacker_checkpoint(restored, ckpt_path, map_location="cpu")
    assert meta["episode"] == 1


def test_attacker_reward_is_clipped_without_scaling() -> None:
    from train import _clip_attacker_reward

    assert _clip_attacker_reward(raw_reward=80.0, reward_clip=5000.0) == 80.0
    assert _clip_attacker_reward(raw_reward=6000.0, reward_clip=5000.0) == 5000.0
    assert _clip_attacker_reward(raw_reward=-3.0, reward_clip=5000.0) == 0.0


def test_attacker_training_reports_split_stats_and_eval_baselines() -> None:
    """
    作用:
        验证 learned attacker 训练报告包含 AoI/H 拆分统计、deterministic eval 和 random intent baseline。

    输入格式:
        无，内部构造轻量 fake env、victim 和 attacker。

    输出格式:
        None，断言失败时 pytest 报错。

    参数含义:
        测试使用同一 LearnedAttackEnv mapping 来保证 learned 和 random baseline 共享能量预算。
    """
    from attacks.learned_attack_config import LearnedAttackConfig
    from attacks.learned_attack_env import LearnedAttackEnv
    from train import run_attacker_training

    env = make_learned_env()
    cfg = LearnedAttackConfig(per_step_energy_budget=2.0, alpha_tau=1.0, alpha_h=0.25, aoi_delta=1, h_delta=1)
    attack_env = LearnedAttackEnv(env, DummyDQNVictim(), "DQN", cfg)
    attacker_agent = DummyAttackAgent(action_dim=env.state_dim)

    result = run_attacker_training(
        attack_env,
        attacker_agent,
        config={
            "episodes": 1,
            "episode_length": 2,
            "batch_size": 999,
            "warmup_steps": 999,
            "attacker_eval_every": 1,
            "attacker_eval_episodes": 1,
            "attacker_random_baseline": True,
            "save_checkpoints": False,
            "save_history": False,
            "save_report": False,
            "seed": 123,
        },
    )

    assert len(result["aoi_l0_history"]) == 1
    assert len(result["h_l0_history"]) == 1
    assert len(result["aoi_energy_history"]) == 1
    assert len(result["h_energy_history"]) == 1
    assert len(result["deterministic_eval_history"]) == 1
    assert len(result["random_intent_baseline_history"]) == 1

    det_eval = result["deterministic_eval_history"][0]
    random_eval = result["random_intent_baseline_history"][0]
    assert det_eval["episode"] == 1
    assert random_eval["episode"] == 1
    assert det_eval["energy_budget"] == cfg.per_step_energy_budget
    assert random_eval["energy_budget"] == cfg.per_step_energy_budget
    assert det_eval["energy_used"] <= cfg.per_step_energy_budget + 1e-6
    assert random_eval["energy_used"] <= cfg.per_step_energy_budget + 1e-6
    assert det_eval["aoi_energy"] + det_eval["h_energy"] == det_eval["energy_used"]
    assert random_eval["aoi_energy"] + random_eval["h_energy"] == random_eval["energy_used"]
