from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from agent import DDPGAgent, DQNAgent
from config import (
    BATCH_SIZE,
    DEFAULT_SEED,
    EPISODE_LENGTH,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    N,
    NOISE_STD_DECAY,
    NOISE_STD_MIN,
    NOISE_STD_START,
    NUM_EPISODES,
    REPLAY_BUFFER_CAPACITY,
)
from env import SemanticSchedulingEnv
from train import run_training
from utils import plot_learning_curve


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析命令行参数。

    参数:
        无。

    返回:
        argparse.Namespace: 参数对象。

    核心步骤:
        1. 创建 argparse 解析器。
        2. 注册 algo/seed/episodes/eval 参数。
        3. 执行解析并返回。
    """
    parser = argparse.ArgumentParser(description="Baseline DQN/DDPG training CLI")
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "DDPG"], help="Training algorithm.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Reserved switch for future model-only evaluation mode.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """
    作用:
        固定随机种子，增强可复现性。

    参数:
        seed: 随机种子。

    返回:
        None。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_config(args: argparse.Namespace) -> dict:
    """
    作用:
        组装训练配置字典并传入 run_training。

    参数:
        args: 命令行参数。

    返回:
        dict: 训练配置。
    """
    return {
        "num_episodes": args.episodes,
        "episode_length": EPISODE_LENGTH,
        "batch_size": BATCH_SIZE,
        "buffer_capacity": REPLAY_BUFFER_CAPACITY,
        "epsilon_start": EPSILON_START,
        "epsilon_decay": EPSILON_DECAY,
        "epsilon_min": EPSILON_MIN,
        "noise_std_start": NOISE_STD_START,
        "noise_std_decay": NOISE_STD_DECAY,
        "noise_std_min": NOISE_STD_MIN,
        "verbose_interval": 10,
    }


def main() -> None:
    """
    作用:
        程序入口：解析 CLI、构建环境和智能体、启动训练并保存曲线。

    参数:
        无。

    返回:
        None。

    核心步骤:
        1. 解析参数并设置随机种子。
        2. 初始化环境与对应算法智能体。
        3. 调用 run_training 得到 mse_history。
        4. 绘制并保存学习曲线图。
    """
    args = parse_args()
    set_global_seed(args.seed)

    print(
        f"Starting training -> Algo: {args.algo}, Scale: (N=6, M=3), "
        f"Seed: {args.seed}, Episodes: {args.episodes}, EvalMode: {args.eval}"
    )

    if args.eval:
        print("`--eval` is reserved for future extension. Training will still run in this baseline version.")

    env = SemanticSchedulingEnv(seed=args.seed)
    if args.algo == "DQN":
        agent = DQNAgent(state_dim=env.state_dim, num_actions=env.num_actions)
    else:
        agent = DDPGAgent(state_dim=env.state_dim, action_dim=N)

    train_cfg = build_train_config(args)
    mse_history = run_training(args.algo, env, agent, train_cfg)

    out_path = Path("results") / f"result_{args.algo}_seed{args.seed}.png"
    saved_path = plot_learning_curve(mse_history, args.algo, out_path, window=10)
    print(f"Training finished. Curve saved to: {saved_path}")


if __name__ == "__main__":
    main()
