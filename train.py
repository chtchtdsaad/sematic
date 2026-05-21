"""
训练与评估模块（train.py）
===========================
负责：
- 统一训练循环（DQN / DDPG）
- checkpoint 保存/加载
- 评估流程（不更新参数）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent import ReplayBuffer
from config import (
    BATCH_SIZE,
    DDPG_REWARD_CLIP,
    DDPG_REWARD_SCALE,
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
    ddpg_reward_clip = float(cfg.get("ddpg_reward_clip", DDPG_REWARD_CLIP))
    ddpg_reward_scale = float(cfg.get("ddpg_reward_scale", DDPG_REWARD_SCALE))

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

                # # next_state 同样归一化

                # # 训练奖励做裁剪缩放（降低数值跨度）
                # reward_train = max(-ddpg_reward_clip, min(ddpg_reward_clip, reward)) / ddpg_reward_scale

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
