"""
训练编排工具模块（workflow.py）
=============================
职责：
- 训练前公共初始化（seed、device）
- 环境与智能体构建
- 训练配置组装
- 历史文件与对比图保存
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
    NOISE_STD_DECAY,
    NOISE_STD_MIN,
    NOISE_STD_START,
    REPLAY_BUFFER_CAPACITY,
    SCENARIO_PRESETS,
)
from env import SemanticSchedulingEnv
from utils import (
    compute_dynamic_ylim_from_two_tail_means,
    plot_compare_mse_curve,
    plot_compare_sum_aoi_curve,
)


def exp_decay_gamma(start_lr: float, end_lr: float, steps: int) -> float:
    """
    作用:
        根据起止学习率计算指数衰减系数 gamma。

    输入格式:
        start_lr: float
        end_lr: float
        steps: int

    输出格式:
        float
    """
    # 非法输入时不衰减，直接返回 1.0。
    if steps <= 0 or start_lr <= 0 or end_lr <= 0:
        return 1.0
    # 终点学习率高于起点时也不做衰减。
    if end_lr >= start_lr:
        return 1.0
    # 按指数关系计算每轮衰减系数。
    return (end_lr / start_lr) ** (1.0 / steps)


def set_global_seed(seed: int) -> None:
    """
    作用:
        固定 Python/NumPy/PyTorch 随机种子。

    输入格式:
        seed: int

    输出格式:
        None
    """
    # Python 随机种子。
    random.seed(seed)
    # NumPy 随机种子。
    np.random.seed(seed)
    # PyTorch CPU 随机种子。
    torch.manual_seed(seed)
    # 若存在 CUDA，补充设置 CUDA 随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """
    作用:
        将设备字符串解析为 torch.device 并校验可用性。

    输入格式:
        device_arg: str，'cpu' 或 'cuda'

    输出格式:
        torch.device
    """
    # 将字符串转换为 torch.device 对象。
    device = torch.device(device_arg)
    # 当用户请求 CUDA 但本机不可用时，明确抛错提示。
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested by --device=cuda, but CUDA is not available.")
    # 返回校验后的设备对象。
    return device


def build_env(seed: int, scenario: str, algo: str) -> SemanticSchedulingEnv:
    """
    作用:
        构建训练环境实例。

    输入格式:
        seed: int
        scenario: str，场景键（base/s10x5/s20x10）
        algo: str，'DQN' 或 'DDPG'

    输出格式:
        SemanticSchedulingEnv
    """
    # 场景键合法性检查。
    if scenario not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario: {scenario}.")

    # 读取场景对应 n/m。
    n = int(SCENARIO_PRESETS[scenario]["n"])
    m = int(SCENARIO_PRESETS[scenario]["m"])

    # DQN 需要离散 action_space；DDPG 直接走 assignment，可关闭 action_space 构建。
    build_action_space = str(algo).upper() == "DQN"
    return SemanticSchedulingEnv(seed=seed, n=n, m=m, build_action_space=build_action_space)


def build_agent(args: argparse.Namespace, env: SemanticSchedulingEnv, device: torch.device):
    """
    作用:
        根据算法类型构建 DQN/DDPG 智能体。

    输入格式:
        args: argparse.Namespace
        env: SemanticSchedulingEnv
        device: torch.device

    输出格式:
        tuple[object, str]
        - 第 1 项：agent 实例（DQNAgent 或 DDPGAgent）
        - 第 2 项：构建日志字符串
    """
    # 将算法名统一为大写，避免分支判断受到大小写影响。
    algo = str(args.algo).upper()
    # DQN 分支：离散动作直接输出 action_id。
    if algo == "DQN":
        agent = DQNAgent(state_dim=env.state_dim, num_actions=env.num_actions, device=device)
        return agent, "DQN initialized."

    # DDPG 分支：若 CLI 未指定学习率则回退默认配置。
    actor_lr = float(args.ddpg_actor_lr) if args.ddpg_actor_lr is not None else DDPG_ACTOR_LR
    critic_lr = float(args.ddpg_critic_lr) if args.ddpg_critic_lr is not None else DDPG_CRITIC_LR
    # 读取学习率终点，用于指数衰减系数计算。
    actor_end = DDPG_ACTOR_LR_END
    critic_end = DDPG_CRITIC_LR_END
    # 计算 actor 学习率衰减 gamma。
    actor_gamma = exp_decay_gamma(start_lr=actor_lr, end_lr=actor_end, steps=max(1, int(args.episodes)))
    # 计算 critic 学习率衰减 gamma。
    critic_gamma = exp_decay_gamma(start_lr=critic_lr, end_lr=critic_end, steps=max(1, int(args.episodes)))

    # 构建 DDPG 智能体。
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.n,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        actor_lr_decay_gamma=actor_gamma,
        critic_lr_decay_gamma=critic_gamma,
        device=device,
    )
    # 生成构建日志字符串，便于终端确认本次实际参数。
    info = (
        f"DDPG initialized. actor_lr={actor_lr:.2e}->{actor_end:.2e}, "
        f"critic_lr={critic_lr:.2e}->{critic_end:.2e}"
    )
    return agent, info


def build_train_config(args: argparse.Namespace) -> dict[str, object]:
    """
    作用:
        组装训练配置字典，统一传给 run_training。

    输入格式:
        args: argparse.Namespace

    输出格式:
        dict[str, object]
        - 键值全部为 run_training 可识别字段
    """
    # 若 CLI 未覆盖噪声衰减，使用 config 默认值。
    noise_decay = float(args.noise_std_decay) if args.noise_std_decay is not None else NOISE_STD_DECAY
    # 若 CLI 未覆盖 warmup 步数，使用 config 默认值。
    warmup_steps = int(args.ddpg_warmup_steps) if args.ddpg_warmup_steps is not None else DDPG_WARMUP_STEPS

    # 返回训练配置字典；后续直接传给 run_training 使用。
    return {
        "num_episodes": args.episodes,  # int
        "episode_length": EPISODE_LENGTH,  # int
        "batch_size": BATCH_SIZE,  # int
        "buffer_capacity": REPLAY_BUFFER_CAPACITY,  # int
        "epsilon_start": EPSILON_START,  # float
        "epsilon_decay": EPSILON_DECAY,  # float
        "epsilon_min": EPSILON_MIN,  # float
        "noise_std_start": NOISE_STD_START,  # float
        "noise_std_decay": noise_decay,  # float
        "noise_std_min": NOISE_STD_MIN,  # float
        "ddpg_warmup_steps": warmup_steps,  # int
        "ddpg_reward_clip": DDPG_REWARD_CLIP,  # float
        "ddpg_reward_scale": DDPG_REWARD_SCALE,  # float
        "update_interval": max(1, int(args.update_interval)),  # int
        "best_select_mode": str(args.best_select_mode),  # str
        "best_eval_every": max(1, int(args.best_eval_every)),  # int
        "best_eval_episodes": max(1, int(args.best_eval_episodes)),  # int
        "verbose_interval": 10,  # int
        "save_dir": "results/checkpoints",  # str
        "save_prefix": f"{str(args.algo).lower()}_{str(args.scenario)}_seed{args.seed}",  # str
        "save_best": bool(args.save_checkpoints),  # bool
        "save_last": bool(args.save_checkpoints),  # bool
    }


def save_history_and_compare(
    algo: str,
    scenario: str,
    n: int,
    m: int,
    seed: int,
    episodes: int,
    mse_history: list[float],
    sum_aoi_history: list[float],
    save_plots: bool,
) -> None:
    """
    作用:
        保存当前算法历史，并在同配置下自动输出 DQN/DDPG 对比图。

    输入格式:
        algo: str，'DQN' 或 'DDPG'
        scenario: str
        n: int
        m: int
        seed: int
        episodes: int
        mse_history: list[float]，长度=episodes
        sum_aoi_history: list[float]，长度=episodes
        save_plots: bool

    输出格式:
        None
    """
    # 创建 history 目录，保存统一命名的 npz 文件。
    history_dir = Path("results/histories")
    history_dir.mkdir(parents=True, exist_ok=True)
    # 当前算法历史文件路径。
    current_history_path = history_dir / f"{algo}_{scenario}_seed{seed}_ep{episodes}.npz"

    # 将当前算法历史落盘为 npz，便于后续自动对比加载。
    np.savez(
        current_history_path,
        algo=algo,
        scenario=scenario,
        n=n,
        m=m,
        seed=seed,
        episodes=episodes,
        mse_history=np.asarray(mse_history, dtype=np.float64),
        sum_aoi_history=np.asarray(sum_aoi_history, dtype=np.float64),
    )
    print(f"Saved training history: {current_history_path.resolve()}")

    # 反向推导对比算法名称。
    other_algo = "DDPG" if algo == "DQN" else "DQN"
    # 对比算法历史文件路径。
    other_history_path = history_dir / f"{other_algo}_{scenario}_seed{seed}_ep{episodes}.npz"
    # 对比方历史不存在时，直接结束。
    if not other_history_path.exists():
        print(f"Compare curve skipped: counterpart history not found -> {other_history_path.resolve()}")
        return

    # 读取当前与对比算法历史。
    current_data = np.load(current_history_path, allow_pickle=True)
    other_data = np.load(other_history_path, allow_pickle=True)
    # 校验场景元信息（scenario/N/M/seed/episodes）是否一致。
    is_same_setup = (
        str(current_data["scenario"]) == str(other_data["scenario"]) == scenario
        and int(current_data["n"]) == int(other_data["n"]) == n
        and int(current_data["m"]) == int(other_data["m"]) == m
        and int(current_data["seed"]) == int(other_data["seed"]) == seed
        and int(current_data["episodes"]) == int(other_data["episodes"]) == episodes
    )
    if not is_same_setup:
        print("Compare curve skipped: metadata mismatch (scenario/N/M/seed/episodes not equal).")
        return

    # 根据当前文件的 algo 字段，对齐 dqn/ddpg 曲线变量。
    cur_algo = str(current_data["algo"])
    if cur_algo == "DQN":
        dqn_mse = current_data["mse_history"]
        dqn_aoi = current_data["sum_aoi_history"]
        ddpg_mse = other_data["mse_history"]
        ddpg_aoi = other_data["sum_aoi_history"]
    else:
        dqn_mse = other_data["mse_history"]
        dqn_aoi = other_data["sum_aoi_history"]
        ddpg_mse = current_data["mse_history"]
        ddpg_aoi = current_data["sum_aoi_history"]

    # 若用户关闭绘图，仅保留 history，不输出对比图。
    if not save_plots:
        print("Compare curve skipped: --save-plots 0.")
        return

    # 生成 MSE 对比图路径。
    cmp_mse_path = Path("results") / f"compare_DQN_DDPG_{scenario}_seed{seed}_ep{episodes}_mse.png"
    # 生成 SumAoI 对比图路径。
    cmp_aoi_path = Path("results") / f"compare_DQN_DDPG_{scenario}_seed{seed}_ep{episodes}_sumaoi.png"
    # 对比图动态纵轴：基准取 DQN/DDPG 最后 10% 均值中的较大值，目标比例固定 35%。
    cmp_mse_ylim = compute_dynamic_ylim_from_two_tail_means(dqn_mse, ddpg_mse, target_ratio=0.35)
    cmp_aoi_ylim = compute_dynamic_ylim_from_two_tail_means(dqn_aoi, ddpg_aoi, target_ratio=0.35)
    # 保存 MSE 对比图。
    saved_cmp_mse = plot_compare_mse_curve(
        dqn_mse,
        ddpg_mse,
        cmp_mse_path,
        window=10,
        ylim=cmp_mse_ylim,
        hard_clip=True,
    )
    # 保存 SumAoI 对比图。
    saved_cmp_aoi = plot_compare_sum_aoi_curve(
        dqn_aoi,
        ddpg_aoi,
        cmp_aoi_path,
        window=10,
        ylim=cmp_aoi_ylim,
        hard_clip=True,
    )
    print(f"Compare curves saved to: {saved_cmp_mse} and {saved_cmp_aoi}")
