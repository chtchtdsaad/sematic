"""
训练与评估模块（train.py）
===========================
负责：
- 统一训练循环（DQN / DDPG）
- checkpoint 保存/加载
- 评估流程（不更新参数）
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent import ReplayBuffer
from config import (
    BATCH_SIZE,
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
    NUM_EPISODES,
    REPLAY_BUFFER_CAPACITY,
)
from utils import map_continuous_to_assignment




def _build_checkpoint(algo: str, agent, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    作用:
        组装可持久化 checkpoint 字典。

    输入格式:
        algo: str，'DQN' 或 'DDPG'
        agent: 智能体实例
        meta: dict[str,Any] | None

    输出格式:
        dict[str,Any]

    核心步骤:
        1. 记录算法标识和 meta。
        2. 按算法写入对应网络与优化器状态。
    """
    # 统一大写算法名
    algo = algo.upper()

    # 通用字段
    ckpt: dict[str, Any] = {"algo": algo, "meta": meta or {}}

    # DQN 字段
    if algo == "DQN":
        ckpt.update(
            {
                "q_net": agent.q_net.state_dict(),
                "target_q_net": agent.target_q_net.state_dict(),
                "optimizer": agent.optimizer.state_dict(),
                "update_steps": int(agent.update_steps),
            }
        )

    # DDPG 字段
    elif algo == "DDPG":
        ckpt.update(
            {
                "actor": agent.actor.state_dict(),
                "actor_target": agent.actor_target.state_dict(),
                "critic": agent.critic.state_dict(),
                "critic_target": agent.critic_target.state_dict(),
                "actor_optimizer": agent.actor_optimizer.state_dict(),
                "critic_optimizer": agent.critic_optimizer.state_dict(),
            }
        )

    # 算法非法
    else:
        raise ValueError("algo must be 'DQN' or 'DDPG'")

    return ckpt


def save_checkpoint(algo: str, agent, save_path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    """
    作用:
        保存模型 checkpoint 到本地路径。

    输入格式:
        algo: str
        agent: 智能体实例
        save_path: str | Path
        meta: dict[str,Any] | None

    输出格式:
        Path（绝对路径）

    核心步骤:
        1. 构造 Path 对象并创建父目录。
        2. 构建 checkpoint 字典。
        3. 使用 torch.save 写入文件。
    """
    # 统一路径类型
    save_path = Path(save_path)

    # 创建目录
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 序列化写盘
    torch.save(_build_checkpoint(algo, agent, meta), save_path)

    # 返回绝对路径
    return save_path.resolve()


def load_checkpoint(
    algo: str,
    agent,
    ckpt_path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """
    作用:
        从 checkpoint 恢复智能体参数状态。

    输入格式:
        algo: str，'DQN' 或 'DDPG'
        agent: 智能体实例
        ckpt_path: str | Path
        map_location: str | torch.device | None

    输出格式:
        dict[str,Any]（checkpoint 中的 meta）

    核心步骤:
        1. 读取 checkpoint 并检查算法一致性。
        2. 按算法恢复网络和优化器参数。
        3. 返回 meta 信息。
    """
    # 读取 checkpoint
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # 校验算法名
    ckpt_algo = str(ckpt["algo"]).upper()
    algo = algo.upper()
    if ckpt_algo != algo:
        raise ValueError(f"Checkpoint algo={ckpt_algo} does not match requested algo={algo}")

    # 恢复 DQN
    if algo == "DQN":
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_q_net.load_state_dict(ckpt["target_q_net"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.update_steps = int(ckpt.get("update_steps", 0))

    # 恢复 DDPG
    else:
        agent.actor.load_state_dict(ckpt["actor"])
        agent.actor_target.load_state_dict(ckpt["actor_target"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

    return dict(ckpt.get("meta", {}))


def _build_attacker_checkpoint(attacker_agent, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    作用:
        组装 learned attacker checkpoint 字典。

    输入格式:
        attacker_agent: AttackDDPGAgent。
        meta: dict[str,Any] | None，训练轮次、指标和配置元信息。

    输出格式:
        dict[str,Any]，可直接交给 torch.save。

    参数含义:
        attacker_agent 提供 Actor/Critic/Optimizer 的 state_dict；meta 保存实验上下文。

    核心步骤:
        1. 写入 attacker 算法标识。
        2. 保存在线网络和目标网络参数。
        3. 保存优化器状态和 meta。
    """
    # learned attacker 第一轮只实现 DDPG，因此 checkpoint 明确记录 ATTACK_DDPG。
    return {
        "algo": "ATTACK_DDPG",
        "attacker_algo": "DDPG",
        "actor": attacker_agent.actor.state_dict(),
        "actor_target": attacker_agent.actor_target.state_dict(),
        "critic": attacker_agent.critic.state_dict(),
        "critic_target": attacker_agent.critic_target.state_dict(),
        "actor_optimizer": attacker_agent.actor_optimizer.state_dict(),
        "critic_optimizer": attacker_agent.critic_optimizer.state_dict(),
        "meta": meta or {},
    }


def save_attacker_checkpoint(attacker_agent, save_path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    """
    作用:
        保存 learned attacker checkpoint 到本地文件。

    输入格式:
        attacker_agent: AttackDDPGAgent。
        save_path: str | Path。
        meta: dict[str,Any] | None。

    输出格式:
        Path，保存后的绝对路径。

    参数含义:
        save_path 指定输出文件；meta 保存训练指标和配置。

    核心步骤:
        1. 创建父目录。
        2. 构建 checkpoint 字典。
        3. 使用 torch.save 写盘。
    """
    # 统一路径对象，避免字符串路径在不同系统下拼接出错。
    save_path = Path(save_path)
    # checkpoint 目录不存在时自动创建。
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 保存完整 attacker 状态。
    torch.save(_build_attacker_checkpoint(attacker_agent, meta), save_path)
    # 返回绝对路径，方便日志和 report 使用。
    return save_path.resolve()


def load_attacker_checkpoint(attacker_agent, ckpt_path: str | Path, map_location=None) -> dict[str, Any]:
    """
    作用:
        从 checkpoint 恢复 learned attacker 参数。

    输入格式:
        attacker_agent: AttackDDPGAgent。
        ckpt_path: str | Path。
        map_location: torch load 的设备映射参数。

    输出格式:
        dict[str,Any]，checkpoint 中保存的 meta。

    参数含义:
        ckpt_path 是 learned attacker checkpoint，不是 victim checkpoint。

    核心步骤:
        1. 读取 checkpoint。
        2. 校验算法标识。
        3. 恢复 Actor/Critic/Optimizer。
        4. 返回 meta。
    """
    # 读取本地 checkpoint。
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # 校验 checkpoint 类型，避免误加载 victim checkpoint。
    if str(ckpt.get("algo", "")).upper() != "ATTACK_DDPG":
        raise ValueError("Attacker checkpoint algo must be ATTACK_DDPG.")
    # 恢复在线 Actor。
    attacker_agent.actor.load_state_dict(ckpt["actor"])
    # 恢复目标 Actor。
    attacker_agent.actor_target.load_state_dict(ckpt["actor_target"])
    # 恢复在线 Critic。
    attacker_agent.critic.load_state_dict(ckpt["critic"])
    # 恢复目标 Critic。
    attacker_agent.critic_target.load_state_dict(ckpt["critic_target"])
    # 恢复 Actor 优化器。
    attacker_agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    # 恢复 Critic 优化器。
    attacker_agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    # 返回元信息副本。
    return dict(ckpt.get("meta", {}))


def _clip_attacker_reward(raw_reward: float, reward_clip: float) -> float:
    """
    作用:
        将 raw MSE 奖励裁剪到稳定范围内，作为 DDPG critic 的训练奖励。

    输入格式:
        raw_reward: float，真实下一步 MSE。
        reward_clip: float，奖励上限。

    输出格式:
        float，写入 ReplayBuffer 的 clipped reward。

    参数含义:
        raw_reward 用于 history；clipped reward 用于 DDPG update。

    核心步骤:
        1. 将 raw reward 限制到 [0,reward_clip]。
    """
    # raw MSE 理论上非负；这里防御性截断到非负区间。
    clipped = min(max(0.0, float(raw_reward)), float(reward_clip))
    # 只做上限裁剪，避免额外除法压弱 critic 的奖励信号。
    return float(clipped)


def _write_attacker_json(path: Path, payload: dict[str, Any]) -> Path:
    """
    作用:
        将 learned attacker 训练结果写为 UTF-8 JSON。

    输入格式:
        path: Path，输出路径。
        payload: dict[str,Any]，可序列化结果。

    输出格式:
        Path，写入后的绝对路径。

    参数含义:
        path 指定 report/history JSON 文件；payload 是训练统计。

    核心步骤:
        1. 创建父目录。
        2. 使用 ensure_ascii=False 保存中文可读 JSON。
    """
    # 创建父目录，避免首次运行时写入失败。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入 JSON，保留中文。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 返回绝对路径供日志使用。
    return path.resolve()


def _empty_attacker_metric_totals() -> dict[str, float]:
    """
    作用:
        创建 learned attacker 训练或评估阶段的指标累加器。

    输入格式:
        无。

    输出格式:
        dict[str,float]，包含 reward、MSE、能耗、AoI/H 拆分扰动和约束违规计数。

    参数含义:
        无参数；调用方每一步通过 _accumulate_attacker_step_metrics 写入统计。

    核心步骤:
        1. 为所有需要平均的指标初始化 0。
        2. 将 constraint_violation_count 也纳入同一累加器。
    """
    # 使用统一字段，避免训练 history 和 eval baseline 字段不一致。
    return {
        "raw_reward": 0.0,
        "raw_mse": 0.0,
        "energy_used": 0.0,
        "aoi_energy": 0.0,
        "h_energy": 0.0,
        "total_l0": 0.0,
        "aoi_l0": 0.0,
        "h_l0": 0.0,
        "constraint_violation_count": 0.0,
    }


def _accumulate_attacker_step_metrics(totals: dict[str, float], raw_reward: float, info: dict[str, Any]) -> None:
    """
    作用:
        将单步 learned attacker 交互指标累加到 episode 或 eval totals。

    输入格式:
        totals: dict[str,float]，由 _empty_attacker_metric_totals 创建。
        raw_reward: float，attacker 的 raw MSE reward。
        info: dict[str,Any]，LearnedAttackEnv.step 返回的诊断信息。

    输出格式:
        None，原地更新 totals。

    参数含义:
        raw_reward 与 raw MSE 等价；info 中的 AoI/H 字段来自 mapping_info。

    核心步骤:
        1. 累加 raw reward/MSE。
        2. 累加总能耗和 AoI/H 拆分能耗。
        3. 累加总 L0 和 AoI/H 拆分 L0。
    """
    # raw reward 定义为正的下一步 MSE。
    totals["raw_reward"] += float(raw_reward)
    # 当前 reward 与 raw MSE 等价，保留两个名字便于报告阅读。
    totals["raw_mse"] += float(raw_reward)
    # 累计总能耗。
    totals["energy_used"] += float(info.get("energy_used", 0.0))
    # 累计 AoI 能耗。
    totals["aoi_energy"] += float(info.get("aoi_energy", 0.0))
    # 累计 H 能耗。
    totals["h_energy"] += float(info.get("h_energy", 0.0))
    # 累计总扰动维度数。
    totals["total_l0"] += float(info.get("total_l0", 0.0))
    # 累计 AoI 扰动维度数。
    totals["aoi_l0"] += float(info.get("aoi_l0", 0.0))
    # 累计 H 扰动维度数。
    totals["h_l0"] += float(info.get("h_l0", 0.0))
    # 约束违规按次数统计，不做平均前的布尔压缩。
    totals["constraint_violation_count"] += 1.0 if bool(info.get("constraint_violation", False)) else 0.0


def _average_attacker_metric_totals(
    totals: dict[str, float],
    step_count: int,
    episode: int,
    policy: str,
    energy_budget: float,
    eval_episodes: int,
) -> dict[str, Any]:
    """
    作用:
        将 learned attacker 累加指标转为可写入 report 的平均指标字典。

    输入格式:
        totals: dict[str,float]，累加指标。
        step_count: int，有效交互步数。
        episode: int，当前训练 episode 编号。
        policy: str，评估策略名称。
        energy_budget: float，单步能量预算。
        eval_episodes: int，本次评估 episode 数。

    输出格式:
        dict[str,Any]，包含均值 reward/MSE/energy/L0 和元信息。

    参数含义:
        step_count 用作平均分母；episode/policy/energy_budget 用于报告定位。

    核心步骤:
        1. 使用 step_count 防止除零。
        2. 对逐步指标求平均。
        3. 保留约束违规次数和预算字段。
    """
    # 防止极端情况下除零。
    denom = max(1, int(step_count))
    # 总能耗均值单独计算，后续用 AoI/H 能耗均值相加保证一致。
    aoi_energy = float(totals["aoi_energy"] / denom)
    # H 能耗均值。
    h_energy = float(totals["h_energy"] / denom)
    # 返回统一结构，便于 learned 与 random 横向对比。
    return {
        "episode": int(episode),
        "policy": str(policy),
        "eval_episodes": int(eval_episodes),
        "steps": int(step_count),
        "reward": float(totals["raw_reward"] / denom),
        "mse": float(totals["raw_mse"] / denom),
        "energy_used": float(aoi_energy + h_energy),
        "energy_budget": float(energy_budget),
        "aoi_energy": aoi_energy,
        "h_energy": h_energy,
        "total_l0": float(totals["total_l0"] / denom),
        "aoi_l0": float(totals["aoi_l0"] / denom),
        "h_l0": float(totals["h_l0"] / denom),
        "constraint_violation_count": int(totals["constraint_violation_count"]),
    }


def _run_attacker_policy_evaluation(
    attack_env,
    attacker_agent,
    episode_length: int,
    eval_episodes: int,
    report_episode: int,
    policy: str,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    作用:
        在不更新网络、不写 ReplayBuffer 的前提下评估 learned attacker 当前策略或 random intent baseline。

    输入格式:
        attack_env: LearnedAttackEnv。
        attacker_agent: AttackDDPGAgent 或兼容测试对象。
        episode_length: int，每个 eval episode 最大步数。
        eval_episodes: int，本次评估 episode 数。
        report_episode: int，对应训练 episode 编号。
        policy: str，"deterministic" 或 "random_intent"。
        rng: np.random.Generator | None，random intent 使用的随机数发生器。

    输出格式:
        dict[str,Any]，可直接写入 deterministic_eval_history 或 random_intent_baseline_history。

    参数含义:
        deterministic 使用 Actor 且 `noise_std=0`；random_intent 使用同一 mapping 和同一能量预算。

    核心步骤:
        1. 保存训练环境运行状态。
        2. 重置环境并运行 eval episode。
        3. 按策略生成 intent action。
        4. 累加 raw MSE、能耗和 AoI/H 拆分统计。
        5. 恢复训练环境运行状态。
    """
    # 只接受两种评估策略，避免拼写错误静默进入错误逻辑。
    if policy not in {"deterministic", "random_intent"}:
        raise ValueError("policy must be 'deterministic' or 'random_intent'.")
    # random baseline 需要显式 RNG，保证报告可复现。
    if policy == "random_intent" and rng is None:
        raise ValueError("rng is required when policy='random_intent'.")
    # 评估会 reset/step 真实环境，因此先保存训练现场。
    snapshot = attack_env.snapshot_runtime_state()
    # 初始化累加器。
    totals = _empty_attacker_metric_totals()
    # 有效 step 计数。
    step_count = 0
    try:
        # 外层 eval episode 循环。
        for _ in range(max(1, int(eval_episodes))):
            # reset 返回真实 raw state。
            state = attack_env.reset()
            # 每个 eval episode 内按固定长度推进。
            for _step_idx in range(max(1, int(episode_length))):
                # deterministic eval 使用当前 Actor 的纯策略，不叠加训练噪声。
                if policy == "deterministic":
                    intent_action = attacker_agent.select_action(state, noise_std=0.0)
                # random intent baseline 共享同一 action_dim 和 mapping，仅替换 intent 来源。
                else:
                    intent_action = rng.uniform(-1.0, 1.0, size=attack_env.action_dim).astype(np.float32)
                # 评估也通过 attack_env.step，以保证 victim 与真实环境路径一致。
                next_state, raw_reward, done, info = attack_env.step(intent_action)
                # 累加本步诊断指标。
                _accumulate_attacker_step_metrics(totals, raw_reward=raw_reward, info=info)
                # 更新当前状态。
                state = next_state
                # 记录有效 step。
                step_count += 1
                # 环境终止则结束该 eval episode。
                if done:
                    break
    finally:
        # 无论评估是否异常，都恢复训练环境，避免破坏后续训练。
        attack_env.restore_runtime_state(snapshot)

    # 从配置中读取单步能量预算，写入评估摘要。
    energy_budget = float(getattr(attack_env.config, "per_step_energy_budget", 0.0))
    # 返回平均后的评估指标。
    return _average_attacker_metric_totals(
        totals=totals,
        step_count=step_count,
        episode=int(report_episode),
        policy=policy,
        energy_budget=energy_budget,
        eval_episodes=max(1, int(eval_episodes)),
    )


def run_attacker_training(attack_env, attacker_agent, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    作用:
        训练 learned adversarial DRL attacker。

    输入格式:
        attack_env: LearnedAttackEnv。
        attacker_agent: AttackDDPGAgent。
        config: dict[str,Any] | None，由 workflow.build_attacker_train_config 构造。

    输出格式:
        dict[str,Any]，包含训练曲线、checkpoint 路径、耗时和诊断统计。

    参数含义:
        attack_env 负责真实交互和映射；attacker_agent 负责 Actor/Critic 更新；
        config 控制 episode、warmup、buffer、噪声、reward clip 和保存行为。

    核心步骤:
        1. 初始化 ReplayBuffer 和训练历史。
        2. 每步用 raw state 生成 continuous intent action。
        3. attack_env 内部完成 attacked_state、victim action 和真实环境推进。
        4. ReplayBuffer 存 raw state、intent action、clipped reward。
        5. 满足 warmup/batch/update interval 后更新 attacker。
        6. 保存 best/last checkpoint、history 和 report。
    """
    # 读取配置字典，缺省时使用保守默认值。
    cfg = config or {}
    # episode 数量用于外层训练循环。
    num_episodes = int(cfg.get("episodes", 300))
    # episode length 默认沿用环境配置，也允许 smoke test 覆盖。
    episode_length = int(cfg.get("episode_length", EPISODE_LENGTH))
    # batch size 控制每次更新采样数量。
    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    # 回放池容量控制训练样本历史长度。
    buffer_capacity = int(cfg.get("replay_buffer_capacity", REPLAY_BUFFER_CAPACITY))
    # warmup 阶段使用随机 intent action。
    warmup_steps = int(cfg.get("warmup_steps", DDPG_WARMUP_STEPS))
    # 更新间隔避免每步都强制更新。
    update_interval = max(1, int(cfg.get("update_interval", 1)))
    # 初始探索噪声。
    noise_std = float(cfg.get("noise_std_start", NOISE_STD_START))
    # episode 级噪声衰减。
    noise_decay = float(cfg.get("noise_std_decay", NOISE_STD_DECAY))
    # 噪声下限。
    noise_min = float(cfg.get("noise_std_min", NOISE_STD_MIN))
    # reward clip 上限。
    reward_clip = float(cfg.get("reward_clip", DDPG_REWARD_CLIP))
    # 是否保存 checkpoint。
    save_checkpoints = bool(cfg.get("save_checkpoints", True))
    # 是否保存 history JSON。
    save_history = bool(cfg.get("save_history", True))
    # 是否保存 report JSON。
    save_report = bool(cfg.get("save_report", True))
    # 输出根目录。
    result_dir = Path(str(cfg.get("attacker_result_dir", "results/attacker")))
    # 场景名称用于文件命名。
    scenario = str(cfg.get("scenario", "base"))
    # seed 用于文件命名。
    seed = int(cfg.get("seed", 0))
    # victim 算法用于文件命名。
    victim_algo = str(cfg.get("victim_algo", "DDPG")).upper()
    # 日志间隔默认每个 episode 都打印，长训练可由配置覆盖。
    verbose_interval = max(1, int(cfg.get("verbose_interval", 1)))
    # deterministic eval 间隔；0 表示不做额外评估。
    attacker_eval_every = max(0, int(cfg.get("attacker_eval_every", 0)))
    # 每次 eval 的 episode 数。
    attacker_eval_episodes = max(1, int(cfg.get("attacker_eval_episodes", 1)))
    # 是否在 deterministic eval 同时运行 random intent baseline。
    attacker_random_baseline = bool(cfg.get("attacker_random_baseline", True))
    # random intent baseline 使用独立 RNG，避免污染环境 RNG。
    random_baseline_rng = np.random.default_rng(seed + 7919)

    # ReplayBuffer 存 raw state、continuous intent action、clipped reward。
    buffer = ReplayBuffer(capacity=buffer_capacity)
    # 每个 episode 的 raw attacker reward 均值。
    attacker_reward_history: list[float] = []
    # raw MSE history 与 raw reward 等价，保留独立字段便于画图。
    attacker_mse_history: list[float] = []
    # 每个 episode 的平均能量消耗。
    energy_used_history: list[float] = []
    # 每个 episode 的平均 AoI 能量消耗。
    aoi_energy_history: list[float] = []
    # 每个 episode 的平均 H 能量消耗。
    h_energy_history: list[float] = []
    # 记录每个 episode 的平均扰动维度数量。
    total_l0_history: list[float] = []
    # 记录每个 episode 的平均 AoI 扰动维度数量。
    aoi_l0_history: list[float] = []
    # 记录每个 episode 的平均 H 扰动维度数量。
    h_l0_history: list[float] = []
    # 记录每个 episode 的约束违规次数，正常应保持为 0。
    constraint_violation_count_history: list[int] = []
    # deterministic eval 的 episode 级诊断结果。
    deterministic_eval_history: list[dict[str, Any]] = []
    # random intent baseline 的 episode 级诊断结果。
    random_intent_baseline_history: list[dict[str, Any]] = []
    # 记录训练 loss，便于调试。
    critic_loss_history: list[float] = []
    actor_loss_history: list[float] = []
    # best 按 raw MSE/reward 最大化。
    best_reward = -float("inf")
    # best checkpoint 路径。
    best_model_path: Path | None = None
    # last checkpoint 路径。
    last_model_path: Path | None = None
    # 全局交互步数。
    global_step = 0
    # 训练计时起点。
    train_start_ts = time.perf_counter()

    # 外层 episode 循环。
    for episode in range(num_episodes):
        # reset 返回 raw true state。
        state = attack_env.reset()
        # 当前 episode raw reward 总和。
        total_raw_reward = 0.0
        # 当前 episode 指标累加器用于统一统计 AoI/H 拆分。
        metric_totals = _empty_attacker_metric_totals()
        # 当前 episode 实际步数。
        episode_steps = 0
        # 当前 episode 是否发生过参数更新。
        episode_trained = False

        # 内层 step 循环。
        for _ in range(episode_length):
            # warmup 阶段用随机 intent action 探索映射空间。
            if global_step < warmup_steps:
                intent_action = np.random.uniform(-1.0, 1.0, size=attack_env.action_dim).astype(np.float32)
            # warmup 后使用 attacker Actor 并叠加探索噪声。
            else:
                intent_action = attacker_agent.select_action(state, noise_std=noise_std)

            # 环境包装器内部完成映射、victim 选动作和真实环境推进。
            next_state, raw_reward, done, info = attack_env.step(intent_action)
            # 将 raw MSE clip 后写入 ReplayBuffer。
            reward_train = _clip_attacker_reward(raw_reward, reward_clip=reward_clip)
            # ReplayBuffer 存 continuous intent action，不存离散 delta。
            buffer.push(state, intent_action, reward_train, next_state, done)

            # buffer 样本足够且 warmup 结束后才更新。
            ready_for_update = len(buffer) >= batch_size and global_step >= warmup_steps
            # update_interval 控制更新频率。
            if ready_for_update and (global_step % update_interval == 0):
                # 执行一次 DDPG 更新。
                critic_loss, actor_loss = attacker_agent.update(buffer, batch_size=batch_size)
                # 保存 Critic loss。
                critic_loss_history.append(float(critic_loss))
                # 保存 Actor loss。
                actor_loss_history.append(float(actor_loss))
                # 标记本 episode 已训练。
                episode_trained = True

            # 累计 raw reward，用于 best checkpoint 选择。
            total_raw_reward += float(raw_reward)
            # 累计训练 episode 诊断指标。
            _accumulate_attacker_step_metrics(metric_totals, raw_reward=raw_reward, info=info)
            # 更新 state 为下一步真实状态。
            state = next_state
            # 推进步数。
            episode_steps += 1
            # 推进全局步数。
            global_step += 1
            # 环境终止时结束当前 episode。
            if done:
                break

        # 防止极端情况下除零。
        denom = max(1, episode_steps)
        # 当前 episode 平均 raw reward。
        avg_raw_reward = float(total_raw_reward / denom)
        # 当前 episode 平均 raw MSE。
        avg_raw_mse = avg_raw_reward
        # 当前 episode 平均能耗。
        avg_aoi_energy = float(metric_totals["aoi_energy"] / denom)
        # 当前 episode 平均 H 能耗。
        avg_h_energy = float(metric_totals["h_energy"] / denom)
        # 当前 episode 平均总能耗。
        avg_energy = float(avg_aoi_energy + avg_h_energy)
        # 当前 episode 平均 AoI L0。
        avg_aoi_l0 = float(metric_totals["aoi_l0"] / denom)
        # 当前 episode 平均 H L0。
        avg_h_l0 = float(metric_totals["h_l0"] / denom)
        # 当前 episode 平均 L0。
        avg_total_l0 = float(avg_aoi_l0 + avg_h_l0)
        # 当前 episode 约束违规次数。
        constraint_violation_count = int(metric_totals["constraint_violation_count"])
        # 写入 raw reward history。
        attacker_reward_history.append(avg_raw_reward)
        # 写入 raw MSE history。
        attacker_mse_history.append(avg_raw_mse)
        # 写入能耗 history。
        energy_used_history.append(avg_energy)
        # 写入 AoI 能耗 history。
        aoi_energy_history.append(avg_aoi_energy)
        # 写入 H 能耗 history。
        h_energy_history.append(avg_h_energy)
        # 写入扰动维度 history。
        total_l0_history.append(avg_total_l0)
        # 写入 AoI L0 history。
        aoi_l0_history.append(avg_aoi_l0)
        # 写入 H L0 history。
        h_l0_history.append(avg_h_l0)
        # 写入约束违规次数 history。
        constraint_violation_count_history.append(constraint_violation_count)

        # raw reward 越大代表攻击越强。
        if avg_raw_reward > best_reward:
            # 更新 best 指标。
            best_reward = avg_raw_reward
            # 按开关保存 best checkpoint。
            if save_checkpoints:
                best_model_path = save_attacker_checkpoint(
                    attacker_agent=attacker_agent,
                    save_path=result_dir / "checkpoints" / f"attacker_ddpg_vs_{victim_algo.lower()}_{scenario}_seed{seed}_best.pt",
                    meta={
                        "episode": episode + 1,
                        "avg_raw_reward": avg_raw_reward,
                        "avg_raw_mse": avg_raw_mse,
                        "avg_energy_used": avg_energy,
                        "victim_algo": victim_algo,
                        "scenario": scenario,
                        "seed": seed,
                    },
                )

        # 按间隔运行 deterministic eval 和 random intent baseline。
        if attacker_eval_every > 0 and (episode + 1) % attacker_eval_every == 0:
            # deterministic eval 使用 noise_std=0，评估当前 Actor 纯策略。
            det_eval = _run_attacker_policy_evaluation(
                attack_env=attack_env,
                attacker_agent=attacker_agent,
                episode_length=episode_length,
                eval_episodes=attacker_eval_episodes,
                report_episode=episode + 1,
                policy="deterministic",
            )
            # 保存 deterministic eval 结果。
            deterministic_eval_history.append(det_eval)
            # random intent baseline 使用同一 mapping/预算，只替换 intent 来源。
            if attacker_random_baseline:
                random_eval = _run_attacker_policy_evaluation(
                    attack_env=attack_env,
                    attacker_agent=attacker_agent,
                    episode_length=episode_length,
                    eval_episodes=attacker_eval_episodes,
                    report_episode=episode + 1,
                    policy="random_intent",
                    rng=random_baseline_rng,
                )
                # 保存 random baseline 结果。
                random_intent_baseline_history.append(random_eval)

        # 每个 episode 后衰减探索噪声。
        noise_std = max(noise_min, noise_std * noise_decay)
        # 按间隔打印训练日志。
        if (episode + 1) % verbose_interval == 0 or episode == 0:
            print(
                f"[ATTACK_DDPG vs {victim_algo}] Episode {episode + 1}/{num_episodes} | "
                f"AvgRawMSE={avg_raw_mse:.6f} | AvgEnergy={avg_energy:.4f} | "
                f"AoIE={avg_aoi_energy:.4f} | HE={avg_h_energy:.4f} | "
                f"AvgL0={avg_total_l0:.4f} | noise_std={noise_std:.4f} | trained={episode_trained}"
            )

    # 保存 last checkpoint。
    if save_checkpoints:
        last_model_path = save_attacker_checkpoint(
            attacker_agent=attacker_agent,
            save_path=result_dir / "checkpoints" / f"attacker_ddpg_vs_{victim_algo.lower()}_{scenario}_seed{seed}_last.pt",
            meta={
                "episode": num_episodes,
                "avg_raw_reward": attacker_reward_history[-1] if attacker_reward_history else None,
                "avg_raw_mse": attacker_mse_history[-1] if attacker_mse_history else None,
                "victim_algo": victim_algo,
                "scenario": scenario,
                "seed": seed,
            },
        )

    # 计算训练耗时。
    train_seconds = float(time.perf_counter() - train_start_ts)
    # 汇总返回结果。
    result: dict[str, Any] = {
        "attacker_reward_history": attacker_reward_history,
        "attacker_mse_history": attacker_mse_history,
        "energy_used_history": energy_used_history,
        "aoi_energy_history": aoi_energy_history,
        "h_energy_history": h_energy_history,
        "total_l0_history": total_l0_history,
        "aoi_l0_history": aoi_l0_history,
        "h_l0_history": h_l0_history,
        "constraint_violation_count_history": constraint_violation_count_history,
        "deterministic_eval_history": deterministic_eval_history,
        "random_intent_baseline_history": random_intent_baseline_history,
        "critic_loss_history": critic_loss_history,
        "actor_loss_history": actor_loss_history,
        "best_reward": float(best_reward) if np.isfinite(best_reward) else None,
        "best_model_path": str(best_model_path) if best_model_path is not None else None,
        "last_model_path": str(last_model_path) if last_model_path is not None else None,
        "train_seconds": train_seconds,
        "global_steps": int(global_step),
        "victim_algo": victim_algo,
        "scenario": scenario,
        "seed": seed,
        "reward_clip": reward_clip,
        "attacker_eval_every": attacker_eval_every,
        "attacker_eval_episodes": attacker_eval_episodes,
        "attacker_random_baseline": attacker_random_baseline,
    }

    # history JSON 保存训练曲线。
    if save_history:
        history_path = result_dir / "histories" / f"attacker_ddpg_vs_{victim_algo.lower()}_{scenario}_seed{seed}_history.json"
        result["history_path"] = str(_write_attacker_json(history_path, result))
    # report JSON 保存同一份摘要，后续可扩展配置字段。
    if save_report:
        report_path = result_dir / "reports" / f"attacker_ddpg_vs_{victim_algo.lower()}_{scenario}_seed{seed}_report.json"
        result["report_path"] = str(_write_attacker_json(report_path, result))
    # 返回完整训练结果。
    return result


def run_training(algo: str, env, agent, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    作用:
        统一训练入口，支持 DQN 和 DDPG。

    输入格式:
        algo: str，'DQN' 或 'DDPG'
        env: 环境实例（需有 reset/step/action_space/aoi）
        agent: 智能体实例
        config: dict[str,Any] | None

    输出格式:
        dict[str,Any]
        {
          "mse_history": list[float],
          "sum_aoi_history": list[float],
          "best_mse": float | None,
          "best_model_path": str | None,
          "last_model_path": str | None,
          "best_metric_source": str,
          "best_metric_value": float | None,
          "train_seconds": float,
        }

    核心步骤:
        1. 读取训练配置并初始化回放池。
        2. 逐 episode/step 与环境交互并存储 transition。
        3. 满足条件时执行 agent.train_step。
        4. 记录每轮 Average Sum MSE 与 Average SumAoI。
        5. 保存 best/last checkpoint 并返回统计结果。
    """
    # 统一算法名
    algo = algo.upper()
    if algo not in {"DQN", "DDPG"}:
        raise ValueError("algo must be 'DQN' or 'DDPG'")

    # ---------- 读取配置 ----------
    cfg = config or {}

    # 训练轮次与长度
    num_episodes = int(cfg.get("num_episodes", NUM_EPISODES))
    episode_length = int(cfg.get("episode_length", EPISODE_LENGTH))

    # batch 与回放池容量
    batch_size = int(cfg.get("batch_size", BATCH_SIZE))
    buffer_capacity = int(cfg.get("buffer_capacity", REPLAY_BUFFER_CAPACITY))

    # 日志间隔
    verbose_interval = int(cfg.get("verbose_interval", 10))

    # DQN 探索参数
    epsilon = float(cfg.get("epsilon_start", EPSILON_START))
    epsilon_decay = float(cfg.get("epsilon_decay", EPSILON_DECAY))
    epsilon_min = float(cfg.get("epsilon_min", EPSILON_MIN))

    # DDPG 探索参数
    noise_std = float(cfg.get("noise_std_start", NOISE_STD_START))
    noise_decay = float(cfg.get("noise_std_decay", NOISE_STD_DECAY))
    noise_min = float(cfg.get("noise_std_min", NOISE_STD_MIN))

    # DQN/DDPG 稳定化参数
    dqn_warmup_steps = int(cfg.get("dqn_warmup_steps", DQN_WARMUP_STEPS))
    ddpg_warmup_steps = int(cfg.get("ddpg_warmup_steps", DDPG_WARMUP_STEPS))

    # checkpoint 参数
    save_dir = Path(cfg.get("save_dir", "results/checkpoints"))
    save_prefix = str(cfg.get("save_prefix", algo.lower()))
    save_best = bool(cfg.get("save_best", True))
    save_last = bool(cfg.get("save_last", True))
    # best 选择模式：train=按训练 episode 指标；eval=按 clean-eval 指标。
    best_select_mode = str(cfg.get("best_select_mode", "train")).lower()
    # 防御性修正：若传入非法字符串，默认回退到 train 模式。
    if best_select_mode not in {"train", "eval"}:
        best_select_mode = "train"
    # best-eval 触发间隔（单位：episode），最小为 1。
    best_eval_every = max(1, int(cfg.get("best_eval_every", 10)))
    # 每次 clean-eval 运行的 episode 数，最小为 1。
    best_eval_episodes = max(1, int(cfg.get("best_eval_episodes", 3)))
    # 参数更新间隔（单位：step），1 表示每步更新，2 表示隔步更新。
    update_interval = max(1, int(cfg.get("update_interval", 1)))
    # clean-eval 日志开关，默认关闭避免训练日志过多。
    best_eval_verbose = bool(cfg.get("best_eval_verbose", False))

    # ---------- 训练容器 ----------
    buffer = ReplayBuffer(capacity=buffer_capacity)

    mse_history: list[float] = []
    sum_aoi_history: list[float] = []
    # 训练完成后统计（全训练区间聚合，不分 warmup/post-warmup）
    per_channel_select_count = np.zeros(env.m, dtype=np.int64)            # shape=(m,)
    per_channel_selected_sensor_aoi_sum = np.zeros(env.m, dtype=np.float64)  # shape=(m,)
    per_sensor_selected_count = np.zeros(env.n, dtype=np.int64)           # shape=(n,)
    per_sensor_selected_aoi_sum = np.zeros(env.n, dtype=np.float64)       # shape=(n,)
    sensor_channel_count = np.zeros((env.n, env.m), dtype=np.int64)       # shape=(n,m)

    global_step = 0
    best_mse = float("inf")
    best_metric_source = "train"
    best_metric_value = float("inf")
    best_model_path: Path | None = None
    last_model_path: Path | None = None
    dqn_warmup_end_logged = False
    # 记录训练开始时间戳，最终用于统计 train_seconds。
    train_start_ts = time.perf_counter()

    # ---------- episode 循环 ----------
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()

        # 当前 episode 统计量
        total_reward = 0.0
        total_sum_aoi = 0.0
        episode_steps = 0
        episode_trained = False
        episode_used_dqn_warmup = False

        # ---------- step 循环 ----------
        for _ in range(episode_length):
            # 记录调度前 AoI（用于“被选中时 AoI”统计）
            aoi_before_step = env.aoi.astype(np.float64, copy=True)

            if algo == "DQN":
                # DQN warmup: 前若干步仅随机采样，不做参数更新。
                in_dqn_warmup = global_step < dqn_warmup_steps
                if in_dqn_warmup:
                    action_id = int(np.random.randint(0, env.num_actions))
                    episode_used_dqn_warmup = True
                else:
                    action_id = agent.select_action(state, epsilon)
                # DQN 通过离散动作索引解码 assignment。
                assignment = tuple(int(x) for x in env.action_space[action_id])
            
                # 与环境交互
                next_state, reward, done = env.step(action_id)
                # 存离散动作到 buffer
                buffer.push(state, action_id, reward, next_state, done)
            else:
                # DDPG: 归一化状态

                # 连续动作（shape=(N,)）
                virtual_action = agent.select_action(state, noise_std)

                # 连续动作映射为 assignment（长度 n，值域 0..m）。
                assignment = map_continuous_to_assignment(virtual_action, n=env.n, m=env.m)

                # 环境交互（DDPG 直接走 assignment，不依赖离散 action_space）。
                next_state, reward, done = env.step_assignment(assignment)

                # 存连续动作到 buffer（Critic 需要连续动作）
                buffer.push(state, virtual_action, reward, next_state, done)

            # 聚合“训练完成后”统计（单份，不分 warmup）
            for sensor_idx, channel_id in enumerate(assignment):
                if channel_id <= 0:
                    continue
                c_idx = int(channel_id) - 1
                per_channel_select_count[c_idx] += 1
                per_channel_selected_sensor_aoi_sum[c_idx] += float(aoi_before_step[sensor_idx])
                per_sensor_selected_count[sensor_idx] += 1
                per_sensor_selected_aoi_sum[sensor_idx] += float(aoi_before_step[sensor_idx])
                sensor_channel_count[sensor_idx, c_idx] += 1

            # 更新当前状态
            state = next_state

            # 累计原始奖励（用于 MSE 统计）
            total_reward += reward

            # 累计每步 SumAoI
            total_sum_aoi += float(np.sum(env.aoi))

            # 步数推进
            episode_steps += 1
            global_step += 1

            # 训练触发条件
            if algo == "DQN":
                ready_for_train = (len(buffer) > batch_size) and (global_step > dqn_warmup_steps)
            else:
                ready_for_train = (len(buffer) > batch_size) and (global_step > ddpg_warmup_steps)

            # 执行参数更新
            # update_interval 控制训练更新频率，兼顾稳定性与时长。
            if ready_for_train and (global_step % update_interval == 0):
                agent.train_step(buffer, batch_size)
                episode_trained = True

            # 若到达终止条件则结束本 episode
            if done:
                break

        # ---------- episode 结果计算 ----------
        # denom = max(1, episode_steps)
        # avg_mse = -total_reward / denom
        # avg_sum_aoi = total_sum_aoi / denom
        avg_mse = -total_reward / episode_length
        avg_sum_aoi = total_sum_aoi / episode_length


        # 记录曲线
        mse_history.append(float(avg_mse))
        sum_aoi_history.append(float(avg_sum_aoi))

        # 统计 best 指标（checkpoint 是否落盘由 save_best 决定）
        # eval 模式：使用 clean-eval mean_mse 作为 best 指标来源。
        if best_select_mode == "eval":
            # 到达评估间隔或训练末轮时触发 clean-eval。
            need_eval = ((episode + 1) % best_eval_every == 0) or (episode + 1 == num_episodes)
            if need_eval:
                # 运行不带探索噪声的评估，输出 eval_result 字典。
                eval_result = run_evaluation(
                    algo=algo,
                    env=env,
                    agent=agent,
                    num_episodes=best_eval_episodes,
                    episode_length=episode_length,
                    verbose=best_eval_verbose,
                )
                # 候选指标：clean-eval 的 mean_mse（float）。
                candidate_metric = float(eval_result["mean_mse"])
                # 指标更优则更新 best 记录。
                if candidate_metric < best_metric_value:
                    best_metric_value = candidate_metric
                    best_mse = candidate_metric
                    best_metric_source = "eval"
                    # save_best=True 时，落盘 checkpoint；否则只更新内存指标。
                    if save_best:
                        best_model_path = save_checkpoint(
                            algo=algo,
                            agent=agent,
                            save_path=save_dir / f"{save_prefix}_best.pt",
                            meta={
                                "episode": episode + 1,
                                "avg_mse": float(avg_mse),
                                "avg_sum_aoi": float(avg_sum_aoi),
                                "best_metric_source": "eval",
                                "best_metric_value": candidate_metric,
                                "best_eval_episodes": best_eval_episodes,
                            },
                        )
        else:
            # train 模式：直接使用当前 episode 的 avg_mse 作为候选指标。
            candidate_metric = float(avg_mse)
            # 指标更优则更新 best 记录。
            if candidate_metric < best_metric_value:
                best_metric_value = candidate_metric
                best_mse = candidate_metric
                best_metric_source = "train"
                # save_best=True 时，落盘 checkpoint；否则只更新内存指标。
                if save_best:
                    best_model_path = save_checkpoint(
                        algo=algo,
                        agent=agent,
                        save_path=save_dir / f"{save_prefix}_best.pt",
                        meta={
                            "episode": episode + 1,
                            "avg_mse": float(avg_mse),
                            "avg_sum_aoi": float(avg_sum_aoi),
                            "best_metric_source": "train",
                            "best_metric_value": candidate_metric,
                        },
                        )

        # 更新探索参数
        if algo == "DQN":
            dqn_warmup_on = dqn_warmup_steps > 0 and global_step <= dqn_warmup_steps
            if not dqn_warmup_on:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)
            info = (
                f"epsilon={epsilon:.4f}"

            )
        else:
            noise_std = max(noise_min, noise_std * noise_decay)

            # 本轮有训练时再衰减学习率
            if episode_trained and hasattr(agent, "step_lr_decay"):
                agent.step_lr_decay()

            # 记录当前学习率
            if hasattr(agent, "get_current_lrs"):
                actor_lr, critic_lr = agent.get_current_lrs()
                info = f"noise_std={noise_std:.4f}, actor_lr={actor_lr:.2e}, critic_lr={critic_lr:.2e}"
            else:
                info = f"noise_std={noise_std:.4f}"

        # 打印日志
        if (episode + 1) % verbose_interval == 0 or episode == 0:
            print(
                f"[{algo}] Episode {episode + 1}/{num_episodes} | "
                f"Average Sum MSE={avg_mse:.6f} | Average SumAoI={avg_sum_aoi:.6f} | {info}"
            )

    # 保存 last checkpoint
    if save_last:
        last_model_path = save_checkpoint(
            algo=algo,
            agent=agent,
            save_path=save_dir / f"{save_prefix}_last.pt",
            meta={
                "episode": num_episodes,
                "avg_mse": float(mse_history[-1]) if mse_history else None,
                "avg_sum_aoi": float(sum_aoi_history[-1]) if sum_aoi_history else None,
                "best_metric_source": best_metric_source,
                "best_metric_value": float(best_metric_value) if np.isfinite(best_metric_value) else None,
            },
        )

    # 训练总耗时（秒）。
    train_seconds = float(time.perf_counter() - train_start_ts)

    # ---------- 训练完成后统计汇总 ----------
    per_channel_selected_sensor_aoi_mean = np.divide(
        per_channel_selected_sensor_aoi_sum,
        np.maximum(per_channel_select_count, 1),
        dtype=np.float64,
    ).tolist()
    per_sensor_avg_aoi_when_selected = np.divide(
        per_sensor_selected_aoi_sum,
        np.maximum(per_sensor_selected_count, 1),
        dtype=np.float64,
    ).tolist()

    # 每个 sensor 的 A 矩阵谱半径，shape=(n)
    per_sensor_spectral_radius = [
        float(np.max(np.abs(np.linalg.eigvals(a_mat))))
        for a_mat in env.a_mats
    ]
    # 每个信道的 scale（按 sensor 维度取均值，shape=(m)）
    per_channel_scale = np.mean(env.channel_scales, axis=0).astype(np.float64).tolist()

    return {
        "mse_history": mse_history,
        "sum_aoi_history": sum_aoi_history,
        "best_mse": float(best_mse) if np.isfinite(best_mse) else None,
        "best_model_path": str(best_model_path) if best_model_path is not None else None,
        "last_model_path": str(last_model_path) if last_model_path is not None else None,
        "best_metric_source": best_metric_source,
        "best_metric_value": float(best_metric_value) if np.isfinite(best_metric_value) else None,
        "train_seconds": train_seconds,
        "train_end_selection_stats": {
            "per_channel_selected_sensor_aoi_mean": per_channel_selected_sensor_aoi_mean,
            "per_sensor_selected_count": per_sensor_selected_count.astype(int).tolist(),
            "per_sensor_avg_aoi_when_selected": per_sensor_avg_aoi_when_selected,
            "sensor_channel_count": sensor_channel_count.astype(int).tolist(),
        },
        "per_sensor_spectral_radius": per_sensor_spectral_radius,
        "per_channel_scale": per_channel_scale,
    }


def run_evaluation(
    algo: str,
    env,
    agent,
    num_episodes: int = 10,
    episode_length: int = EPISODE_LENGTH,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    作用:
        评估模式（不更新参数），返回 MSE 与 SumAoI 指标。

    输入格式:
        algo: str，'DQN' 或 'DDPG'
        env: 环境实例
        agent: 智能体实例
        num_episodes: int
        episode_length: int
        verbose: bool

    输出格式:
        dict[str,Any]
        {
          "mse_history": list[float],
          "sum_aoi_history": list[float],
          "mean_mse": float,
          "mean_sum_aoi": float,
        }

    核心步骤:
        1. 每轮 reset 后执行固定步数交互。
        2. DQN 用 epsilon=0；DDPG 用 noise_std=0。
        3. 统计并返回每轮/总体平均指标。
    """
    # 统一算法名
    algo = algo.upper()
    if algo not in {"DQN", "DDPG"}:
        raise ValueError("algo must be 'DQN' or 'DDPG'")

    # 曲线容器
    mse_history: list[float] = []
    sum_aoi_history: list[float] = []

    # episode 循环
    for ep in range(num_episodes):
        # 重置环境
        state = env.reset()

        # 统计量
        total_reward = 0.0
        total_sum_aoi = 0.0
        steps = 0

        # step 循环
        for _ in range(episode_length):
            if algo == "DQN":
                # 评估时 DQN 用贪心动作
                action_id = agent.select_action(state, epsilon=0.0)
                # DQN 使用离散 action_id 接口。
                next_state, reward, done = env.step(action_id)
            else:
                # DDPG 评估时用确定性动作（noise=0）
                virtual_action = agent.select_action(state, noise_std=0.0)
                
                assignment = map_continuous_to_assignment(virtual_action, n=env.n, m=env.m)
                # DDPG 评估同样直接走 assignment 接口。
                next_state, reward, done = env.step_assignment(assignment)
            state = next_state

            # 累积统计
            total_reward += reward
            total_sum_aoi += float(np.sum(env.aoi))
            steps += 1

            if done:
                break

        # 当前 episode 指标
        denom = max(1, steps)
        avg_mse = -total_reward / denom
        avg_sum_aoi = total_sum_aoi / denom

        mse_history.append(float(avg_mse))
        sum_aoi_history.append(float(avg_sum_aoi))

        if verbose:
            print(
                f"[{algo}][Eval] Episode {ep + 1}/{num_episodes} | "
                f"Average Sum MSE={avg_mse:.6f} | Average SumAoI={avg_sum_aoi:.6f}"
            )

    # 计算整体平均
    mean_mse = float(np.mean(mse_history)) if mse_history else float("nan")
    mean_sum_aoi = float(np.mean(sum_aoi_history)) if sum_aoi_history else float("nan")

    if verbose:
        print(
            f"[{algo}][Eval] Mean Average Sum MSE={mean_mse:.6f} | "
            f"Mean Average SumAoI={mean_sum_aoi:.6f}"
        )

    return {
        "mse_history": mse_history,
        "sum_aoi_history": sum_aoi_history,
        "mean_mse": mean_mse,
        "mean_sum_aoi": mean_sum_aoi,
    }
