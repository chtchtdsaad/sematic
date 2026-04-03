"""
程序入口（main.py）
===================
职责：
- 解析 CLI 参数
- 初始化环境与智能体
- 执行训练或评估
- 保存 MSE/SumAoI 曲线
"""

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
    DDPG_ACTOR_LR,
    DDPG_ACTOR_LR_END,
    DDPG_CRITIC_LR,
    DDPG_CRITIC_LR_END,
    DDPG_REWARD_CLIP,
    DDPG_REWARD_SCALE,
    DDPG_WARMUP_STEPS,
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
from train import load_checkpoint, run_evaluation, run_training
from utils import plot_learning_curve, plot_sum_aoi_curve


def _exp_decay_gamma(start_lr: float, end_lr: float, steps: int) -> float:
    """
    作用:
        根据起止学习率计算指数衰减系数 gamma。

    输入格式:
        start_lr: float
        end_lr: float
        steps: int

    输出格式:
        float

    核心步骤:
        1. 处理非法输入与无需衰减情况。
        2. 按 (end/start)^(1/steps) 计算 gamma。
    """
    # 无效参数或不需要衰减时返回 1.0
    if steps <= 0 or start_lr <= 0 or end_lr <= 0:
        return 1.0
    if end_lr >= start_lr:
        return 1.0

    # 指数衰减系数
    return (end_lr / start_lr) ** (1.0 / steps)


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析命令行参数。

    输入格式:
        无

    输出格式:
        argparse.Namespace

    核心步骤:
        1. 注册训练/评估所需 CLI 参数。
        2. 返回解析结果。
    """
    parser = argparse.ArgumentParser(description="Baseline DQN/DDPG training CLI")

    # 算法选择
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "DDPG"], help="Training algorithm.")

    # 随机种子
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")

    # 训练轮数
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes.")

    # 设备选择
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Force training device.")

    # 是否仅评估
    parser.add_argument("--eval", action="store_true", help="Run evaluation only (load model and test without training).")

    # 评估时模型路径
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Checkpoint path used in --eval mode. If empty, use default best checkpoint path.",
    )

    # 评估轮数
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes for evaluation mode.")

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """
    作用:
        固定 Python/NumPy/PyTorch 随机种子。

    输入格式:
        seed: int

    输出格式:
        None

    核心步骤:
        1. 设置 random 与 numpy 种子。
        2. 设置 torch CPU/CUDA 种子。
    """
    # Python 原生随机
    random.seed(seed)

    # NumPy 随机
    np.random.seed(seed)

    # PyTorch CPU 随机
    torch.manual_seed(seed)

    # PyTorch CUDA 随机（若可用）
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_config(args: argparse.Namespace) -> dict[str, object]:
    """
    作用:
        生成训练配置字典并传给 run_training。

    输入格式:
        args: argparse.Namespace

    输出格式:
        dict[str, object]

    核心步骤:
        1. 读取 CLI 参数。
        2. 拼接 config.py 默认超参数。
        3. 返回配置字典。
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
        "ddpg_warmup_steps": DDPG_WARMUP_STEPS,
        "ddpg_reward_clip": DDPG_REWARD_CLIP,
        "ddpg_reward_scale": DDPG_REWARD_SCALE,
        "verbose_interval": 10,
        "save_dir": "results/checkpoints",
        "save_prefix": f"{args.algo.lower()}_seed{args.seed}",
        "save_best": True,
        "save_last": True,
    }


def resolve_device(device_arg: str) -> torch.device:
    """
    作用:
        将设备字符串解析为 torch.device 并校验可用性。

    输入格式:
        device_arg: str，'cpu' 或 'cuda'

    输出格式:
        torch.device

    核心步骤:
        1. 解析字符串为 torch.device。
        2. 若请求 cuda 但不可用则抛错。
    """
    # 解析设备
    device = torch.device(device_arg)

    # 可用性校验
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested by --device=cuda, but CUDA is not available.")

    return device


def main() -> None:
    """
    作用:
        程序主入口：根据参数执行训练或评估。

    输入格式:
        无

    输出格式:
        None

    核心步骤:
        1. 解析参数并设置随机种子。
        2. 初始化环境与智能体。
        3. --eval 分支加载模型并评估。
        4. 否则执行训练并保存曲线。
    """
    # 解析 CLI 参数
    args = parse_args()

    # 设置随机种子
    set_global_seed(args.seed)

    # 解析训练设备
    device = resolve_device(args.device)

    # 启动日志
    print(
        f"Starting training -> Algo: {args.algo}, Scale: (N=6, M=3), "
        f"Seed: {args.seed}, Episodes: {args.episodes}, Device: {device}, EvalMode: {args.eval}"
    )

    # 初始化环境
    env = SemanticSchedulingEnv(seed=args.seed)

    # 初始化智能体
    if args.algo == "DQN":
        # DQN 智能体
        agent = DQNAgent(state_dim=env.state_dim, num_actions=env.num_actions, device=device)
    else:
        # 计算 DDPG actor 的 LR 衰减系数
        actor_gamma = _exp_decay_gamma(
            start_lr=DDPG_ACTOR_LR,
            end_lr=DDPG_ACTOR_LR_END,
            steps=max(1, args.episodes),
        )

        # 计算 DDPG critic 的 LR 衰减系数
        critic_gamma = _exp_decay_gamma(
            start_lr=DDPG_CRITIC_LR,
            end_lr=DDPG_CRITIC_LR_END,
            steps=max(1, args.episodes),
        )

        # DDPG 智能体
        agent = DDPGAgent(
            state_dim=env.state_dim,
            action_dim=N,
            actor_lr_decay_gamma=actor_gamma,
            critic_lr_decay_gamma=critic_gamma,
            device=device,
        )

        # 输出 LR 衰减说明
        print(
            f"DDPG LR decay -> actor: {DDPG_ACTOR_LR:.1e}->{DDPG_ACTOR_LR_END:.1e}, "
            f"critic: {DDPG_CRITIC_LR:.1e}->{DDPG_CRITIC_LR_END:.1e}"
        )

    # 默认 best checkpoint 路径
    default_best_ckpt = Path("results/checkpoints") / f"{args.algo.lower()}_seed{args.seed}_best.pt"

    # ---------- 评估分支 ----------
    if args.eval:
        # 优先使用命令行路径，否则用默认 best
        ckpt_path = Path(args.model_path) if args.model_path else default_best_ckpt

        # 检查 checkpoint 是否存在
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

        # 加载参数
        meta = load_checkpoint(args.algo, agent, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path.resolve()} | meta={meta}")

        # 运行评估
        eval_result = run_evaluation(
            args.algo,
            env,
            agent,
            num_episodes=args.eval_episodes,
            episode_length=EPISODE_LENGTH,
            verbose=True,
        )

        # 评估曲线路径
        eval_mse_path = Path("results") / f"result_{args.algo}_seed{args.seed}_eval_mse.png"
        eval_aoi_path = Path("results") / f"result_{args.algo}_seed{args.seed}_eval_sumaoi.png"

        # 保存评估曲线
        saved_eval_mse = plot_learning_curve(eval_result["mse_history"], args.algo, eval_mse_path, window=10)
        saved_eval_aoi = plot_sum_aoi_curve(eval_result["sum_aoi_history"], args.algo, eval_aoi_path, window=10)

        print(f"Evaluation finished. Curves saved to: {saved_eval_mse} and {saved_eval_aoi}")
        return

    # ---------- 训练分支 ----------
    # 构建训练配置
    train_cfg = build_train_config(args)

    # 执行训练
    train_result = run_training(args.algo, env, agent, train_cfg)

    # 读取训练曲线
    mse_history = train_result["mse_history"]
    sum_aoi_history = train_result["sum_aoi_history"]

    # 曲线输出路径
    mse_path = Path("results") / f"result_{args.algo}_seed{args.seed}.png"
    aoi_path = Path("results") / f"result_{args.algo}_seed{args.seed}_sumaoi.png"

    # 保存训练曲线
    saved_mse_path = plot_learning_curve(mse_history, args.algo, mse_path, window=10)
    saved_aoi_path = plot_sum_aoi_curve(sum_aoi_history, args.algo, aoi_path, window=10)

    # 打印输出路径
    print(f"Training finished. MSE curve saved to: {saved_mse_path}")
    print(f"Training finished. SumAoI curve saved to: {saved_aoi_path}")
    print(
        "Checkpoints -> "
        f"best: {train_result['best_model_path']}, last: {train_result['last_model_path']}"
    )


if __name__ == "__main__":
    main()
