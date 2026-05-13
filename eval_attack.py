"""
评估攻击脚本（eval_attack.py）
============================
本文件只负责攻击评估主流程：
1. 解析 CLI 参数。
2. 构建环境与智能体。
3. 加载 checkpoint。
4. 调用 attacks/ 模块生成 attacked_state、动作和统计。
5. 输出 clean / attack 对比 JSON 报告。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from attacks.action_utils import select_action_for_eval
from attacks.attack_config import AttackConfig, build_attack_config_from_args
from attacks.metrics import init_attack_stats, summarize_attack_stats, update_action_flip_stats, update_perturbation_stats
from attacks.random_semantic import semantic_random_attack
from attacks.structural_expected import (
    semantic_aoi_expected_cost_attack,
    semantic_h_expected_cost_attack,
    semantic_joint_expected_cost_attack,
)
from attacks.structural_metrics import compute_structural_action_metrics, update_structural_metric_stats
from attacks.structural_semantic import (
    semantic_aoi_mislead_attack,
    semantic_h_mislead_attack,
    semantic_joint_mislead_attack,
)
from config import DEFAULT_SEED, EPISODE_LENGTH, SCENARIO_PRESETS
from train import load_checkpoint
from workflow import build_agent, build_env, resolve_device, set_global_seed


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析 eval_attack.py 命令行参数。

    输入格式:
        无。

    输出格式:
        argparse.Namespace，包含基础评估参数、checkpoint 参数和攻击约束参数。

    核心步骤:
        1. 定义算法、场景、种子、设备和 checkpoint 参数。
        2. 定义攻击模式、概率、预算、幅值和方向参数。
        3. 定义 build_agent 所需的兼容占位参数。
    """
    # 构建参数解析器。
    parser = argparse.ArgumentParser(description="Evaluate clean and semantic-random attacked policies.")

    # 基础评估参数。
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "DDPG"], help="Evaluation algorithm.")
    parser.add_argument("--scenario", type=str, default="base", choices=sorted(SCENARIO_PRESETS.keys()))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--eval-episodes", type=int, default=10)

    # 攻击模式与时间预算参数。
    parser.add_argument(
        "--attack-mode",
        type=str,
        default="clean",
        choices=[
            "clean",
            "random_aoi",
            "random_h",
            "random_joint",
            "semantic_aoi_mislead",
            "semantic_h_mislead",
            "semantic_joint_mislead",
            "semantic_aoi_expected_cost",
            "semantic_h_expected_cost",
            "semantic_joint_expected_cost",
        ],
    )
    parser.add_argument("--attack-prob", type=float, default=0.2)
    parser.add_argument("--max-attack-ratio", type=float, default=0.2)
    parser.add_argument("--strict-budget", type=int, default=1, choices=[0, 1])

    # 幅值约束参数。
    parser.add_argument("--aoi-delta", type=int, default=1)
    parser.add_argument("--h-delta", type=int, default=1)

    # 稀疏约束参数。
    parser.add_argument("--max-aoi-features", type=int, default=1)
    parser.add_argument("--max-h-features", type=int, default=1)
    parser.add_argument("--max-total-features", type=int, default=2)

    # 扰动方向参数。
    parser.add_argument("--aoi-direction", type=str, default="random", choices=["random", "increase", "decrease", "mixed"])
    parser.add_argument("--h-direction", type=str, default="random", choices=["random", "increase", "decrease", "mixed"])

    # 连续攻击与输出控制参数。
    parser.add_argument("--max-consecutive-steps", type=int, default=5)
    parser.add_argument("--cooldown-steps", type=int, default=0)
    parser.add_argument("--record-perturbation", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save-report", type=int, default=1, choices=[0, 1])
    parser.add_argument("--expected-cost-mode", type=str, default="mse", choices=["mse", "sum_aoi"])

    # 为兼容 workflow.build_agent 的参数需求提供占位值。
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--ddpg-actor-lr", type=float, default=None)
    parser.add_argument("--ddpg-critic-lr", type=float, default=None)

    # 返回解析结果。
    return parser.parse_args()


def _default_best_ckpt(algo: str, scenario: str, seed: int) -> Path:
    """
    作用:
        构造默认 best checkpoint 路径。

    输入格式:
        algo: str，'DQN' 或 'DDPG'。
        scenario: str，场景键。
        seed: int，随机种子。

    输出格式:
        Path，例如 results/checkpoints/dqn_base_seed24_best.pt。

    核心步骤:
        1. 算法名转小写。
        2. 按项目 checkpoint 命名规则拼接路径。
    """
    # 默认 checkpoint 目录为 results/checkpoints。
    return Path("results/checkpoints") / f"{str(algo).lower()}_{scenario}_seed{seed}_best.pt"


def _step_env_with_action(algo: str, env, action):
    """
    作用:
        根据算法类型用动作推进真实环境一步。

    输入格式:
        algo: str，'DQN' 或 'DDPG'。
        env: 环境实例。
        action: DQN 为 int action_id；DDPG 为 tuple[int,...] assignment。

    输出格式:
        tuple[np.ndarray, float, bool]，即环境 step 返回值。

    核心步骤:
        1. DQN 调用 env.step(action_id)。
        2. DDPG 调用 env.step_assignment(assignment)。
    """
    # DQN 走离散动作编号接口。
    if str(algo).upper() == "DQN":
        return env.step(int(action))

    # DDPG 走 assignment 接口。
    return env.step_assignment(action)


def generate_attacked_state(
    state,
    env,
    config: AttackConfig,
    rng: np.random.Generator,
    attack_state: dict,
    algo: str | None = None,
    agent=None,
):
    """
    作用:
        根据 attack mode 分发到 random、mislead 或 expected-cost 攻击实现。
    输入格式:
        state: np.ndarray[float32], shape=(N+N*M,)。
        env: 环境实例，只读传入攻击函数。
        config: AttackConfig，攻击预算和模式配置。
        rng: np.random.Generator，攻击随机数发生器。
        attack_state: dict，episode 内攻击预算状态。
        algo: str | None，expected-cost 攻击需要 DQN/DDPG 算法名。
        agent: victim agent，expected-cost 攻击需要用它给候选状态选动作。
    输出格式:
        tuple[np.ndarray, dict]，attacked_state 和扰动统计。
    核心步骤:
        1. clean/random 复用 semantic_random_attack。
        2. mislead 分发到 structural_semantic。
        3. expected-cost 分发到 structural_expected，并额外传入 algo/agent。
    """
    # 从 AttackConfig 读取模式，所有攻击分发都以 config.mode 为准。
    mode = str(config.mode)
    # clean/random 系列保持旧逻辑，避免影响已有 baseline。
    if mode in {"clean", "random_aoi", "random_h", "random_joint"}:
        return semantic_random_attack(state=state, env=env, config=config, rng=rng, attack_state=attack_state)
    # AoI-only 结构化误导攻击。
    if mode == "semantic_aoi_mislead":
        return semantic_aoi_mislead_attack(state=state, env=env, config=config, rng=rng, attack_state=attack_state)
    # H-only 结构化误导攻击。
    if mode == "semantic_h_mislead":
        return semantic_h_mislead_attack(state=state, env=env, config=config, rng=rng, attack_state=attack_state)
    # Joint 结构化误导攻击。
    if mode == "semantic_joint_mislead":
        return semantic_joint_mislead_attack(state=state, env=env, config=config, rng=rng, attack_state=attack_state)
    # expected-cost 攻击必须传入 algo 和 agent，否则无法对候选状态调用 victim policy。
    if algo is None or agent is None:
        raise ValueError("expected-cost attack requires algo and agent.")
    # AoI-only expected-cost 攻击。
    if mode == "semantic_aoi_expected_cost":
        return semantic_aoi_expected_cost_attack(
            state=state,
            env=env,
            config=config,
            rng=rng,
            attack_state=attack_state,
            algo=algo,
            agent=agent,
        )
    # H-only expected-cost 攻击。
    if mode == "semantic_h_expected_cost":
        return semantic_h_expected_cost_attack(
            state=state,
            env=env,
            config=config,
            rng=rng,
            attack_state=attack_state,
            algo=algo,
            agent=agent,
        )
    # Joint expected-cost 攻击。
    if mode == "semantic_joint_expected_cost":
        return semantic_joint_expected_cost_attack(
            state=state,
            env=env,
            config=config,
            rng=rng,
            attack_state=attack_state,
            algo=algo,
            agent=agent,
        )
    # 未知攻击模式直接报错。
    raise ValueError(f"unsupported attack mode: {mode}")


def run_clean_eval(
    algo: str,
    env,
    agent,
    num_episodes: int,
    episode_length: int = EPISODE_LENGTH,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    作用:
        运行 clean baseline 评估，不加入任何攻击。

    输入格式:
        algo: str，'DQN' 或 'DDPG'。
        env: 环境实例。
        agent: 智能体实例。
        num_episodes: int，评估 episode 数。
        episode_length: int，每个 episode 最大步数。
        verbose: bool，是否打印每轮日志。

    输出格式:
        dict[str, Any]，包含 mse_history、sum_aoi_history、mean_mse、mean_sum_aoi。

    核心步骤:
        1. 每轮 reset 环境。
        2. 使用统一动作选择接口生成 clean action。
        3. 用 clean action 推进真实环境并累计 MSE / SumAoI。
    """
    # 算法名统一为大写。
    algo = str(algo).upper()
    if algo not in {"DQN", "DDPG"}:
        raise ValueError("algo must be 'DQN' or 'DDPG'")

    # 每轮平均 MSE 和平均 SumAoI 历史。
    mse_history: list[float] = []
    sum_aoi_history: list[float] = []

    # 外层 episode 循环。
    for ep in range(int(num_episodes)):
        # 重置环境得到真实 state，shape=(N + N*M,)。
        state = env.reset()
        total_reward = 0.0
        total_sum_aoi = 0.0
        steps = 0

        # 内层 step 循环。
        for _ in range(int(episode_length)):
            # clean eval 中 agent 直接观察真实 state。
            action = select_action_for_eval(algo, state, agent, env)
            # 用 clean action 推进真实环境。
            next_state, reward, done = _step_env_with_action(algo, env, action)

            # 累计指标。
            state = next_state
            total_reward += float(reward)
            total_sum_aoi += float(np.sum(env.aoi))
            steps += 1

            # 环境结束时退出当前 episode。
            if done:
                break

        # 每轮平均指标。
        denom = max(1, int(steps))
        avg_mse = -total_reward / float(denom)
        avg_sum_aoi = total_sum_aoi / float(denom)
        mse_history.append(float(avg_mse))
        sum_aoi_history.append(float(avg_sum_aoi))

        # 可选日志输出。
        if bool(verbose):
            print(
                f"[{algo}][Clean] Episode {ep + 1}/{num_episodes} | "
                f"Average Sum MSE={avg_mse:.6f} | Average SumAoI={avg_sum_aoi:.6f}"
            )

    # 返回结构化 clean 结果。
    return {
        "mse_history": mse_history,
        "sum_aoi_history": sum_aoi_history,
        "mean_mse": float(np.mean(mse_history)) if mse_history else float("nan"),
        "mean_sum_aoi": float(np.mean(sum_aoi_history)) if sum_aoi_history else float("nan"),
    }


def run_attack_eval(
    algo: str,
    env,
    agent,
    attack_config: AttackConfig,
    eval_episodes: int,
    episode_length: int = EPISODE_LENGTH,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    作用:
        在攻击状态观测下评估 agent，并统计 clean_action 与 attacked_action 的差异。

    输入格式:
        algo: str，'DQN' 或 'DDPG'。
        env: 环境实例。
        agent: 智能体实例。
        attack_config: AttackConfig，攻击约束配置。
        eval_episodes: int，评估 episode 数。
        episode_length: int，每个 episode 最大步数。
        verbose: bool，是否打印每轮日志。

    输出格式:
        dict[str, Any]，包含 attack mean MSE / SumAoI 与攻击统计 summary。

    核心步骤:
        1. 每个 episode reset 真实环境。
        2. 基于真实 state 计算 clean_action，仅用于动作翻转统计。
        3. 调用 semantic_random_attack 生成 attacked_state。
        4. 基于 attacked_state 计算 attacked_action，并用它推进真实环境。
        5. 累计性能指标、扰动统计和动作翻转统计。
    """
    # 算法名统一为大写。
    algo = str(algo).upper()
    if algo not in {"DQN", "DDPG"}:
        raise ValueError("algo must be 'DQN' or 'DDPG'")

    # 攻击随机数发生器，独立于环境 rng。
    rng = np.random.default_rng(int(attack_config.seed))

    # 历史指标与累计统计。
    mse_history: list[float] = []
    sum_aoi_history: list[float] = []
    stats = init_attack_stats()

    # 外层 episode 循环。
    for ep in range(int(eval_episodes)):
        # 重置真实环境。
        state = env.reset()
        total_reward = 0.0
        total_sum_aoi = 0.0
        steps = 0

        # 每个 episode 单独维护攻击预算。
        attack_state = {
            "attack_steps_used": 0,
            "max_attack_steps": int(np.floor(int(episode_length) * float(attack_config.max_attack_ratio))),
            "consecutive_attack_steps": 0,
            "cooldown_remaining": 0,
        }

        # 内层 step 循环。
        for _ in range(int(episode_length)):
            # clean_action 只用于比较动作翻转，不用于环境执行。
            clean_action = select_action_for_eval(algo, state, agent, env)

            # attacked_state 是 agent 看到的状态副本，不写回 env。
            attacked_state, perturb_info = generate_attacked_state(
                state=state,
                env=env,
                config=attack_config,
                rng=rng,
                attack_state=attack_state,
                algo=algo,
                agent=agent,
            )

            # attacked_action 用于真实环境执行。
            attacked_action = select_action_for_eval(algo, attacked_state, agent, env)

            # 更新扰动统计与动作翻转统计。
            update_perturbation_stats(stats, perturb_info, attack_config)
            update_action_flip_stats(stats, clean_action, attacked_action)
            # 结构性资源错配指标使用真实 state、clean_action 和 attacked_action 计算。
            structural_metrics = compute_structural_action_metrics(state, env, algo, clean_action, attacked_action)
            update_structural_metric_stats(stats, structural_metrics)

            # 真实环境只接收 attacked_action，不能接收 attacked_state。
            next_state, reward, done = _step_env_with_action(algo, env, attacked_action)

            # 累计性能指标。
            state = next_state
            total_reward += float(reward)
            total_sum_aoi += float(np.sum(env.aoi))
            steps += 1

            # 环境结束时退出当前 episode。
            if done:
                break

        # 每轮平均指标。
        denom = max(1, int(steps))
        avg_mse = -total_reward / float(denom)
        avg_sum_aoi = total_sum_aoi / float(denom)
        mse_history.append(float(avg_mse))
        sum_aoi_history.append(float(avg_sum_aoi))

        # 可选日志输出。
        if bool(verbose):
            print(
                f"[{algo}][Attack:{attack_config.mode}] Episode {ep + 1}/{eval_episodes} | "
                f"Average Sum MSE={avg_mse:.6f} | Average SumAoI={avg_sum_aoi:.6f}"
            )

    # 汇总攻击统计。
    summary = summarize_attack_stats(stats)

    # 返回 attack 评估结构化结果。
    return {
        "attack_mode": str(attack_config.mode),
        "mse_history": mse_history,
        "sum_aoi_history": sum_aoi_history,
        "mean_mse": float(np.mean(mse_history)) if mse_history else float("nan"),
        "mean_sum_aoi": float(np.mean(sum_aoi_history)) if sum_aoi_history else float("nan"),
        **summary,
    }


def _zero_attack_summary() -> dict[str, float | int]:
    """
    作用:
        为 clean 模式构造零攻击统计字段。

    输入格式:
        无。

    输出格式:
        dict[str, float | int]，字段与 summarize_attack_stats 对齐。

    核心步骤:
        1. 初始化空 stats。
        2. 直接调用 summarize_attack_stats 返回零值摘要。
    """
    # 空统计会得到全部零攻击指标。
    return summarize_attack_stats(init_attack_stats())


def _safe_ratio(numerator: float, denominator: float) -> float:
    """
    作用:
        计算性能退化比例，避免分母为零。

    输入格式:
        numerator: float，退化绝对值。
        denominator: float，clean baseline 指标。

    输出格式:
        float，分母为 0 时返回 0.0。

    核心步骤:
        1. 检查分母绝对值是否过小。
        2. 正常情况下返回 numerator / denominator。
    """
    # 防止除零和极小分母导致无意义比例。
    if abs(float(denominator)) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _round_floats(obj: Any, ndigits: int = 4) -> Any:
    """
    作用:
        递归将对象中的浮点数保留固定小数位。

    输入格式:
        obj: Any，支持 float / dict / list / 其他基础类型。
        ndigits: int，小数位数。

    输出格式:
        Any，结构不变，float 被 round。

    核心步骤:
        1. float 直接 round。
        2. dict/list 递归处理。
        3. 其他类型保持原样。
    """
    # float 直接保留 ndigits 位。
    if isinstance(obj, float):
        return round(obj, ndigits)
    # dict 递归处理 value。
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    # list 递归处理元素。
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    # 其余类型保持原样。
    return obj


def save_attack_report(
    args: argparse.Namespace,
    env,
    ckpt_path: Path,
    clean_result: dict[str, Any],
    attack_result: dict[str, Any],
    attack_config: AttackConfig,
    report_dir: Path | str = Path("results/attack_reports"),
) -> Path:
    """
    作用:
        保存 clean / attack 对比 JSON 报告。

    输入格式:
        args: argparse.Namespace，命令行参数。
        env: 环境实例，需提供 n / m。
        ckpt_path: Path，实际加载的 checkpoint 路径。
        clean_result: dict[str, Any]，run_clean_eval 返回结果。
        attack_result: dict[str, Any]，run_attack_eval 返回结果。
        attack_config: AttackConfig，攻击配置。
        report_dir: Path | str，报告输出目录。

    输出格式:
        Path，报告绝对路径。

    核心步骤:
        1. 构造扁平 JSON payload。
        2. 计算 MSE / SumAoI 退化值和退化比例。
        3. 写入 results/attack_reports/。
    """
    # 报告目录不存在时自动创建。
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)

    # 报告文件名包含 attack mode，避免不同攻击模式互相覆盖。
    report_path = report_root / (
        f"attack_eval_{str(args.algo).upper()}_{args.scenario}_seed{args.seed}_{attack_config.mode}.json"
    )

    # 读取 clean / attack 核心指标。
    clean_mean_mse = float(clean_result.get("mean_mse", float("nan")))
    attack_mean_mse = float(attack_result.get("mean_mse", float("nan")))
    clean_mean_sum_aoi = float(clean_result.get("mean_sum_aoi", float("nan")))
    attack_mean_sum_aoi = float(attack_result.get("mean_sum_aoi", float("nan")))

    # 计算退化绝对值与比例。
    mse_degradation = attack_mean_mse - clean_mean_mse
    sum_aoi_degradation = attack_mean_sum_aoi - clean_mean_sum_aoi

    # 构造扁平 JSON payload。
    payload: dict[str, Any] = {
        "algo": str(args.algo).upper(),
        "scenario": str(args.scenario),
        "n": int(env.n),
        "m": int(env.m),
        "seed": int(args.seed),
        "device": str(args.device),
        "eval_episodes": int(args.eval_episodes),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "attack_mode": str(attack_config.mode),
        "attack_prob": float(attack_config.attack_prob),
        "max_attack_ratio": float(attack_config.max_attack_ratio),
        "strict_budget": bool(attack_config.strict_budget),
        "aoi_delta": int(attack_config.aoi_delta),
        "h_delta": int(attack_config.h_delta),
        "max_aoi_features": int(attack_config.max_aoi_features),
        "max_h_features": int(attack_config.max_h_features),
        "max_total_features": int(attack_config.max_total_features),
        "aoi_direction": str(attack_config.aoi_direction),
        "h_direction": str(attack_config.h_direction),
        "max_consecutive_steps": int(attack_config.max_consecutive_steps),
        "expected_cost_mode": str(getattr(attack_config, "expected_cost_mode", "mse")),
        "clean_mean_mse": clean_mean_mse,
        "attack_mean_mse": attack_mean_mse,
        "mse_degradation": mse_degradation,
        "mse_degradation_ratio": _safe_ratio(mse_degradation, clean_mean_mse),
        "clean_mean_sum_aoi": clean_mean_sum_aoi,
        "attack_mean_sum_aoi": attack_mean_sum_aoi,
        "sum_aoi_degradation": sum_aoi_degradation,
        "sum_aoi_degradation_ratio": _safe_ratio(sum_aoi_degradation, clean_mean_sum_aoi),
        "attack_step_ratio": float(attack_result.get("attack_step_ratio", 0.0)),
        "action_flip_ratio": float(attack_result.get("action_flip_ratio", 0.0)),
        "avg_aoi_l0_per_step": float(attack_result.get("avg_aoi_l0_per_step", 0.0)),
        "avg_h_l0_per_step": float(attack_result.get("avg_h_l0_per_step", 0.0)),
        "avg_total_l0_per_step": float(attack_result.get("avg_total_l0_per_step", 0.0)),
        "avg_aoi_l1_per_attack": float(attack_result.get("avg_aoi_l1_per_attack", 0.0)),
        "avg_h_l1_per_attack": float(attack_result.get("avg_h_l1_per_attack", 0.0)),
        "max_aoi_linf": float(attack_result.get("max_aoi_linf", 0.0)),
        "max_h_linf": float(attack_result.get("max_h_linf", 0.0)),
        "constraint_violation_count": int(attack_result.get("constraint_violation_count", 0)),
        "clean_selected_risk_sum": float(attack_result.get("clean_selected_risk_sum", 0.0)),
        "attack_selected_risk_sum": float(attack_result.get("attack_selected_risk_sum", 0.0)),
        "resource_misallocation_score": float(attack_result.get("resource_misallocation_score", 0.0)),
        "high_risk_scheduled_ratio_clean": float(attack_result.get("high_risk_scheduled_ratio_clean", 0.0)),
        "high_risk_scheduled_ratio_attack": float(attack_result.get("high_risk_scheduled_ratio_attack", 0.0)),
        "low_risk_scheduled_ratio_clean": float(attack_result.get("low_risk_scheduled_ratio_clean", 0.0)),
        "low_risk_scheduled_ratio_attack": float(attack_result.get("low_risk_scheduled_ratio_attack", 0.0)),
        "high_risk_good_channel_ratio_clean": float(attack_result.get("high_risk_good_channel_ratio_clean", 0.0)),
        "high_risk_good_channel_ratio_attack": float(attack_result.get("high_risk_good_channel_ratio_attack", 0.0)),
    }

    # 统一浮点位数后写入 UTF-8 JSON。
    payload = _round_floats(payload, ndigits=4)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 返回绝对路径。
    return report_path.resolve()


def main() -> None:
    """
    作用:
        eval_attack.py 主流程入口。

    输入格式:
        无，由 CLI 提供参数。

    输出格式:
        None，终端输出评估摘要并按需保存 JSON 报告。

    核心步骤:
        1. 解析参数并初始化设备/随机种子。
        2. 构建环境与 agent 并加载 checkpoint。
        3. 运行 clean baseline。
        4. 按 attack_mode 运行 attack eval 或构造 clean 等价攻击结果。
        5. 保存扁平 JSON 报告并打印摘要。
    """
    # 解析 CLI 参数并构造攻击配置。
    args = parse_args()
    attack_config = build_attack_config_from_args(args)

    # 固定全局随机种子并解析设备。
    set_global_seed(int(args.seed))
    device = resolve_device(str(args.device))

    # 构建 clean 环境和智能体。
    clean_env = build_env(seed=int(args.seed), scenario=str(args.scenario), algo=str(args.algo))
    agent, init_info = build_agent(args, clean_env, device)
    print(init_info)

    # 选择 checkpoint 路径（优先用户指定）。
    ckpt_path = Path(args.model_path) if str(args.model_path) else _default_best_ckpt(args.algo, args.scenario, args.seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    # 加载 checkpoint 到智能体。
    ckpt_meta = load_checkpoint(algo=str(args.algo), agent=agent, ckpt_path=ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path.resolve()}")
    if ckpt_meta:
        print(f"Checkpoint meta: {ckpt_meta}")

    # 运行 clean baseline。
    clean_result = run_clean_eval(
        algo=str(args.algo),
        env=clean_env,
        agent=agent,
        num_episodes=int(args.eval_episodes),
        episode_length=EPISODE_LENGTH,
        verbose=True,
    )

    # clean 模式不重复执行攻击流程，只补零统计字段。
    if attack_config.mode == "clean":
        attack_result = {
            "attack_mode": "clean",
            "mse_history": clean_result["mse_history"],
            "sum_aoi_history": clean_result["sum_aoi_history"],
            "mean_mse": clean_result["mean_mse"],
            "mean_sum_aoi": clean_result["mean_sum_aoi"],
            **_zero_attack_summary(),
        }
        report_env = clean_env
    else:
        # attack 使用同 seed 新环境，避免 clean baseline 消耗环境 rng 后影响攻击评估起点。
        attack_env = build_env(seed=int(args.seed), scenario=str(args.scenario), algo=str(args.algo))
        attack_result = run_attack_eval(
            algo=str(args.algo),
            env=attack_env,
            agent=agent,
            attack_config=attack_config,
            eval_episodes=int(args.eval_episodes),
            episode_length=EPISODE_LENGTH,
            verbose=True,
        )
        report_env = attack_env

    # 约束违规时终端打印 warning。
    if int(attack_result.get("constraint_violation_count", 0)) > 0:
        print(f"[EvalAttack][WARNING] constraint_violation_count={attack_result['constraint_violation_count']}")

    # 按需保存 JSON 报告。
    report_path = None
    if bool(int(args.save_report)):
        report_path = save_attack_report(
            args=args,
            env=report_env,
            ckpt_path=ckpt_path,
            clean_result=clean_result,
            attack_result=attack_result,
            attack_config=attack_config,
        )

    # 输出终端摘要。
    print(
        f"[EvalAttack] mode={attack_config.mode} | algo={args.algo} | scenario={args.scenario} | "
        f"clean_mse={clean_result['mean_mse']:.6f} | attack_mse={attack_result['mean_mse']:.6f} | "
        f"clean_sum_aoi={clean_result['mean_sum_aoi']:.6f} | attack_sum_aoi={attack_result['mean_sum_aoi']:.6f} | "
        f"attack_step_ratio={attack_result.get('attack_step_ratio', 0.0):.6f} | "
        f"action_flip_ratio={attack_result.get('action_flip_ratio', 0.0):.6f}"
    )
    if report_path is not None:
        print(f"[EvalAttack] Report saved to: {report_path}")


if __name__ == "__main__":
    # 启动主流程。
    main()
