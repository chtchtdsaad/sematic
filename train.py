from __future__ import annotations

from typing import Any

from agent import ReplayBuffer
from config import (
    BATCH_SIZE,
    EPISODE_LENGTH,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    NOISE_STD_DECAY,
    NOISE_STD_MIN,
    NOISE_STD_START,
    NUM_EPISODES,
    REPLAY_BUFFER_CAPACITY,
)
from utils import map_continuous_to_discrete


def run_training(algo: str, env, agent, config: dict[str, Any] | None = None) -> list[float]:
    """
    作用:
        统一训练引擎，按算法类型驱动环境交互与参数更新。

    参数:
        algo: 算法名称，支持 'DQN' 或 'DDPG'。
        env: 环境对象（SemanticSchedulingEnv）。
        agent: 智能体对象（DQNAgent 或 DDPGAgent）。
        config: 配置字典，可覆盖默认训练参数。

    返回:
        list[float]: 每个 episode 的 Average Sum MSE 历史序列。

    核心步骤:
        1. 读取训练配置并初始化回放池。
        2. 每步按算法分支执行动作选择和交互。
        3. DQN 存离散动作，DDPG 存连续动作向量。
        4. 回放池满足条件后调用 agent.train_step。
        5. 每轮记录 Average MSE 并衰减探索参数。
    """
    algo = algo.upper()
    if algo not in {"DQN", "DDPG"}:
        raise ValueError("algo must be 'DQN' or 'DDPG'")

    cfg = config or {}
    num_episodes = int(cfg.get("num_episodes", NUM_EPISODES))
    episode_length = int(cfg.get("episode_length", EPISODE_LENGTH))
    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    buffer_capacity = int(cfg.get("buffer_capacity", REPLAY_BUFFER_CAPACITY))
    verbose_interval = int(cfg.get("verbose_interval", 10))

    epsilon = float(cfg.get("epsilon_start", EPSILON_START))
    epsilon_decay = float(cfg.get("epsilon_decay", EPSILON_DECAY))
    epsilon_min = float(cfg.get("epsilon_min", EPSILON_MIN))

    noise_std = float(cfg.get("noise_std_start", NOISE_STD_START))
    noise_decay = float(cfg.get("noise_std_decay", NOISE_STD_DECAY))
    noise_min = float(cfg.get("noise_std_min", NOISE_STD_MIN))

    buffer = ReplayBuffer(capacity=buffer_capacity)
    mse_history: list[float] = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(episode_length):
            if algo == "DQN":
                action_id = agent.select_action(state, epsilon)
                next_state, reward, done = env.step(action_id)
                buffer.push(state, action_id, reward, next_state, done)
            else:
                virtual_action = agent.select_action(state, noise_std)
                action_id = map_continuous_to_discrete(virtual_action, env.action_space)
                next_state, reward, done = env.step(action_id)
                buffer.push(state, virtual_action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                agent.train_step(buffer, batch_size)

            if done:
                break

        avg_mse = -total_reward / episode_length
        mse_history.append(float(avg_mse))

        if algo == "DQN":
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            info = f"epsilon={epsilon:.4f}"
        else:
            noise_std = max(noise_min, noise_std * noise_decay)
            info = f"noise_std={noise_std:.4f}"

        if (episode + 1) % verbose_interval == 0 or episode == 0:
            print(
                f"[{algo}] Episode {episode + 1}/{num_episodes} | "
                f"Average Sum MSE={avg_mse:.6f} | {info}"
            )

    return mse_history
