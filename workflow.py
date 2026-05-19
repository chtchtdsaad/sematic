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
from attacks.learned_attack_config import LearnedAttackConfig, build_learned_attack_config_from_args
from config import (
    BATCH_SIZE,
    DDPG_ACTOR_LR,
    DDPG_ACTOR_LR_END,
    DDPG_CRITIC_LR,
    DDPG_CRITIC_LR_END,
    DDPG_REWARD_CLIP,
    DDPG_WARMUP_STEPS,
    DQN_WARMUP_STEPS,
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
    dqn_warmup_steps = int(args.dqn_warmup_steps) if args.dqn_warmup_steps is not None else DQN_WARMUP_STEPS
    ddpg_warmup_steps = int(args.ddpg_warmup_steps) if args.ddpg_warmup_steps is not None else DDPG_WARMUP_STEPS

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
        "dqn_warmup_steps": dqn_warmup_steps,  # int
        "ddpg_warmup_steps": ddpg_warmup_steps,  # int
        "ddpg_reward_clip": DDPG_REWARD_CLIP,  # float
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


def _default_victim_checkpoint_path(victim_algo: str, scenario: str, seed: int) -> Path:
    """
    作用:
        构造 learned attacker 默认使用的 victim best checkpoint 路径。

    输入格式:
        victim_algo: str，"DQN" 或 "DDPG"。
        scenario: str，场景名。
        seed: int，随机种子。

    输出格式:
        Path，形如 results/checkpoints/ddpg_base_seed24_best.pt。

    参数含义:
        victim_algo 决定文件名前缀；scenario/seed 决定具体实验配置。

    核心步骤:
        1. 将算法名转小写。
        2. 按项目既有 checkpoint 命名规则拼接路径。
    """
    # victim checkpoint 继续使用现有 results/checkpoints 目录。
    return Path("results/checkpoints") / f"{str(victim_algo).lower()}_{scenario}_seed{seed}_best.pt"


def _freeze_victim_agent(victim_agent) -> None:
    """
    作用:
        冻结 victim agent 的网络参数并切换 eval 模式。

    输入格式:
        victim_agent: DQNAgent 或 DDPGAgent。

    输出格式:
        None。

    参数含义:
        victim_agent 是被攻击的固定调度器，不允许在 attacker training 中被更新。

    核心步骤:
        1. 遍历 victim 可能包含的网络属性。
        2. 对每个网络关闭梯度。
        3. 对每个网络调用 eval()。
    """
    # 覆盖 DQN 和 DDPG 的所有网络属性。
    module_names = ("q_net", "target_q_net", "actor", "actor_target", "critic", "critic_target")
    # 遍历每个候选属性。
    for module_name in module_names:
        # 某些属性只存在于 DQN 或 DDPG，因此使用 getattr 防御性读取。
        module = getattr(victim_agent, module_name, None)
        # 属性不存在时跳过。
        if module is None:
            continue
        # eval 模式关闭 dropout/bn 等训练行为，虽然当前 MLP 没有这些层，也保持语义明确。
        module.eval()
        # 关闭所有参数梯度，确保 attacker 训练不会更新 victim。
        for param in module.parameters():
            param.requires_grad_(False)


def build_victim_for_attack(args: argparse.Namespace, device: torch.device):
    """
    作用:
        构建并加载固定 victim scheduler，供 learned attacker 训练使用。

    输入格式:
        args: argparse.Namespace，需包含 victim_algo、victim_model_path、scenario、seed。
        device: torch.device。

    输出格式:
        tuple[base_env, victim_agent, victim_algo, victim_meta]。

    参数含义:
        victim_algo 决定 DQN/DDPG；victim_model_path 为空时使用默认 best checkpoint。

    核心步骤:
        1. 用 victim_algo 构建真实环境。
        2. 用现有 build_agent 构建 victim agent。
        3. 加载 victim checkpoint，缺失时明确报错。
        4. 冻结 victim 参数并切换 eval 模式。
        5. 返回环境、agent、算法名和 checkpoint meta。
    """
    # 延迟导入可避免 workflow 顶部依赖 train 的保存逻辑。
    from train import load_checkpoint

    # victim 算法统一大写。
    victim_algo = str(getattr(args, "victim_algo", "DDPG")).upper()
    # 只支持已有 victim DQN/DDPG。
    if victim_algo not in {"DQN", "DDPG"}:
        raise ValueError("victim_algo must be 'DQN' or 'DDPG'.")
    # 复用 build_env，DQN 自动构造 action_space，DDPG 不构造离散动作全集。
    base_env = build_env(seed=int(args.seed), scenario=str(args.scenario), algo=victim_algo)
    # build_agent 读取 args.algo，因此复制 Namespace 并覆盖为 victim_algo。
    victim_args = argparse.Namespace(**vars(args))
    # 设置 algo 字段，让现有 build_agent 无需改动即可构建 victim。
    victim_args.algo = victim_algo
    # build_agent 的 DDPG 分支需要 episodes 字段，这里给一个安全值。
    if not hasattr(victim_args, "episodes") or victim_args.episodes is None:
        victim_args.episodes = max(1, int(getattr(args, "attacker_episodes", 1)))
    # build_agent 的 DDPG 分支读取这些可选学习率字段，缺失时补 None。
    if not hasattr(victim_args, "ddpg_actor_lr"):
        victim_args.ddpg_actor_lr = None
    if not hasattr(victim_args, "ddpg_critic_lr"):
        victim_args.ddpg_critic_lr = None
    # 构建 victim agent。
    victim_agent, _agent_info = build_agent(args=victim_args, env=base_env, device=device)
    # 用户指定路径优先；为空时使用默认 best checkpoint。
    ckpt_path = Path(args.victim_model_path) if str(getattr(args, "victim_model_path", "")) else _default_victim_checkpoint_path(victim_algo, str(args.scenario), int(args.seed))
    # checkpoint 不存在时明确报错，禁止静默训练随机 victim。
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Victim checkpoint not found: {ckpt_path.resolve()}")
    # 加载 victim checkpoint。
    victim_meta = load_checkpoint(victim_algo, victim_agent, ckpt_path, map_location=device)
    # 冻结 victim 网络，确保 attacker 训练只更新 attacker。
    _freeze_victim_agent(victim_agent)
    # 返回构建结果。
    return base_env, victim_agent, victim_algo, victim_meta


def build_learned_attack_env(base_env, victim_agent, victim_algo: str, learned_attack_config: LearnedAttackConfig):
    """
    作用:
        构建 LearnedAttackEnv 包装器。

    输入格式:
        base_env: SemanticSchedulingEnv。
        victim_agent: 固定 victim agent。
        victim_algo: str，"DQN" 或 "DDPG"。
        learned_attack_config: LearnedAttackConfig。

    输出格式:
        LearnedAttackEnv 实例。

    参数含义:
        learned_attack_config 提供映射预算和扰动幅度。

    核心步骤:
        1. 延迟导入 LearnedAttackEnv。
        2. 传入真实环境、victim 和配置。
    """
    # 延迟导入减少 workflow 顶部依赖。
    from attacks.learned_attack_env import LearnedAttackEnv

    # 直接构建 learned attacker 环境包装器。
    return LearnedAttackEnv(base_env=base_env, victim_agent=victim_agent, victim_algo=victim_algo, config=learned_attack_config)


def build_attack_agent(args: argparse.Namespace, attack_env, device: torch.device):
    """
    作用:
        构建 AttackDDPGAgent。

    输入格式:
        args: argparse.Namespace，包含 attacker 学习率、gamma、tau 等字段。
        attack_env: LearnedAttackEnv。
        device: torch.device。

    输出格式:
        tuple[AttackDDPGAgent, str]。

    参数含义:
        attack_env 提供 state_dim/action_dim；args 提供训练超参数。

    核心步骤:
        1. 从 args 构建 LearnedAttackConfig。
        2. 使用 AttackDDPGAgent 构建 attacker。
        3. 返回 agent 和日志字符串。
    """
    # 延迟导入 attacker agent，避免普通 victim 训练无谓加载。
    from attacks.attack_agent import AttackDDPGAgent

    # 从 CLI 参数构造 learned attacker 配置。
    cfg = build_learned_attack_config_from_args(args)
    # 第一轮只支持 DDPG attacker。
    if cfg.attacker_algo != "DDPG":
        raise ValueError("Only DDPG attacker is supported in the first implementation.")
    # 构建 learned attacker agent。
    agent = AttackDDPGAgent(
        state_dim=int(attack_env.state_dim),
        action_dim=int(attack_env.action_dim),
        gamma=float(cfg.attacker_gamma),
        actor_lr=float(cfg.attacker_actor_lr),
        critic_lr=float(cfg.attacker_critic_lr),
        tau=float(cfg.attacker_tau),
        grad_clip_norm=float(cfg.attacker_grad_clip_norm),
        device=device,
    )
    # 返回构建日志。
    return agent, f"AttackDDPG initialized. state_dim={attack_env.state_dim}, action_dim={attack_env.action_dim}"


def build_attacker_train_config(args: argparse.Namespace, learned_attack_config: LearnedAttackConfig) -> dict[str, object]:
    """
    作用:
        构造 run_attacker_training 所需配置字典。

    输入格式:
        args: argparse.Namespace。
        learned_attack_config: LearnedAttackConfig。

    输出格式:
        dict[str,object]，键值全部为 run_attacker_training 可识别字段。

    参数含义:
        args 提供 CLI 可覆盖字段；learned_attack_config 提供 dataclass 默认值。

    核心步骤:
        1. 读取 attacker episode、batch、warmup 等训练参数。
        2. 读取 reward clip 和 deterministic eval 参数。
        3. 写入保存开关和文件命名元信息。
    """
    # 为了简洁，使用局部变量 cfg 表示 learned attacker 配置。
    cfg = learned_attack_config
    # episode length 默认 500，smoke test 可用 --attacker-episode-length 缩短。
    episode_length = int(getattr(args, "attacker_episode_length", EPISODE_LENGTH))
    # 返回训练配置字典。
    return {
        "episodes": int(cfg.attacker_episodes),
        "episode_length": episode_length,
        "batch_size": int(cfg.attacker_batch_size),
        "replay_buffer_capacity": int(cfg.attacker_buffer_capacity),
        "warmup_steps": int(cfg.attacker_warmup_steps),
        "update_interval": max(1, int(cfg.attacker_update_interval)),
        "noise_std_start": float(cfg.attacker_noise_std_start),
        "noise_std_decay": float(cfg.attacker_noise_std_decay),
        "noise_std_min": float(cfg.attacker_noise_std_min),
        "reward_clip": float(getattr(args, "attacker_reward_clip", DDPG_REWARD_CLIP)),
        "attacker_eval_every": max(0, int(cfg.attacker_eval_every)),
        "attacker_eval_episodes": max(1, int(cfg.attacker_eval_episodes)),
        "attacker_random_baseline": bool(cfg.attacker_random_baseline),
        "save_checkpoints": bool(cfg.save_attacker_checkpoints),
        "save_history": bool(cfg.save_attacker_history),
        "save_report": bool(cfg.save_attacker_report),
        "scenario": str(args.scenario),
        "seed": int(args.seed),
        "victim_algo": str(cfg.victim_algo).upper(),
        "attacker_result_dir": str(cfg.attacker_result_dir),
        "verbose_interval": int(getattr(args, "attacker_verbose_interval", 1)),
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
