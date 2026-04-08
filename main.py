"""
程序入口（main.py）
===================
职责：
- 解析 CLI 参数
- 组装环境/智能体/训练配置
- 调用 train.py 执行训练或评估
- 保存展示图、诊断图与实验报告
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DEFAULT_SEED, EPISODE_LENGTH, NUM_EPISODES, SCENARIO_PRESETS
from train import load_checkpoint, run_evaluation, run_training
from utils import (
    plot_learning_curve,
    plot_learning_curve_raw,
    plot_log_mse_curve,
    plot_sum_aoi_curve,
)
from workflow import (
    build_agent,
    build_env,
    build_train_config,
    resolve_device,
    save_history_and_compare,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    """
    作用:
        解析命令行参数并返回参数对象。

    输入格式:
        无

    输出格式:
        argparse.Namespace
        - 例如：args.algo(str), args.seed(int), args.episodes(int)
    """
    # 创建参数解析器，统一管理训练/评估模式的命令行参数。
    parser = argparse.ArgumentParser(description="Baseline DQN/DDPG training CLI")

    # ---------- 基础参数 ----------
    # 算法类型，离散动作 DQN 或连续动作映射 DDPG。
    parser.add_argument("--algo", type=str, required=True, choices=["DQN", "DDPG"], help="Training algorithm.")
    # 随机种子，输入为 int，影响环境和模型初始化。
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    # 训练轮次，输入为 int。
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes.")
    # 训练设备，输入为字符串 cpu/cuda。
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Force device.")
    # 运行模式，train 为训练，eval 为纯评估。
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Run mode.")
    # 场景预配置，默认 base(6x3)。
    parser.add_argument(
        "--scenario",
        type=str,
        default="base",
        choices=sorted(SCENARIO_PRESETS.keys()),
        help="Preset scale scenario. base keeps original N/M.",
    )

    # ---------- 评估参数 ----------
    # 评估时可指定模型路径；为空则自动走默认 best 路径。
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Checkpoint path used in eval mode. Empty means default best checkpoint.",
    )
    # 评估 episode 数量，输入为 int。
    parser.add_argument("--eval-episodes", type=int, default=10, help="Eval episodes.")

    # ---------- 保存开关 ----------
    # 是否保存 checkpoint（1/0）。
    parser.add_argument("--save-checkpoints", type=int, default=1, choices=[0, 1], help="Save checkpoints.")
    # 是否保存训练历史并尝试生成 DQN/DDPG 对比图（1/0）。
    parser.add_argument("--save-history", type=int, default=1, choices=[0, 1], help="Save history and compare.")
    # 是否保存展示曲线（带裁剪）。（1/0）
    parser.add_argument("--save-plots", type=int, default=1, choices=[0, 1], help="Save display plots.")
    # 是否保存诊断曲线（raw MSE / log(MSE)）。（1/0）
    parser.add_argument(
        "--save-diagnostic-plots",
        type=int,
        default=0,
        choices=[0, 1],
        help="Save raw/log-MSE diagnostic plots.",
    )

    # ---------- DDPG 可调参数（稳定性/时长实验） ----------
    # 覆盖 DDPG warmup step 数，输入 int，可为 None。
    parser.add_argument("--ddpg-warmup-steps", type=int, default=None, help="Override DDPG warmup steps.")
    # 覆盖 DDPG actor 学习率，输入 float，可为 None。
    parser.add_argument("--ddpg-actor-lr", type=float, default=None, help="Override DDPG actor lr.")
    # 覆盖 DDPG critic 学习率，输入 float，可为 None。
    parser.add_argument("--ddpg-critic-lr", type=float, default=None, help="Override DDPG critic lr.")
    # 覆盖 DDPG 噪声衰减系数，输入 float，可为 None。
    parser.add_argument("--noise-std-decay", type=float, default=None, help="Override DDPG noise decay.")
    # 更新间隔，输入 int；1 表示每步更新，2 表示隔一步更新。
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1,
        help="Optimization update interval in steps. 1 means update every step.",
    )

    # ---------- best checkpoint 与稳定性诊断参数 ----------
    # best 选择依据，train=按训练 episode 指标，eval=按 clean-eval 指标。
    parser.add_argument(
        "--best-select-mode",
        type=str,
        default="eval",
        choices=["train", "eval"],
        help="Best checkpoint selection metric source.",
    )
    # 每 K 个 episode 做一次 clean-eval（仅用于 best 选择）。
    parser.add_argument("--best-eval-every", type=int, default=10, help="Run clean-eval every K episodes.")
    # 每次 clean-eval 跑多少个 episode。
    parser.add_argument("--best-eval-episodes", type=int, default=3, help="Episodes per clean-eval.")

    # 输出 argparse.Namespace，供主流程读取。
    return parser.parse_args()


def _default_best_ckpt(algo: str, scenario: str, seed: int) -> Path:
    """
    作用:
        构造默认 best checkpoint 路径。

    输入格式:
        algo: str（例如 "DQN" 或 "DDPG"）
        seed: int

    输出格式:
        pathlib.Path
        - 例如 results/checkpoints/ddpg_s20x10_seed7_best.pt
    """
    # 统一按小写算法名拼接默认 best 文件路径。
    return Path("results/checkpoints") / f"{algo.lower()}_{scenario}_seed{seed}_best.pt"


def _save_run_report(args: argparse.Namespace, train_result: dict, env) -> Path:
    """
    作用:
        保存训练报告（JSON），便于后续复盘和横向对比。

    输入格式:
        args: argparse.Namespace
        train_result: dict（run_training 返回的结果字典）

    输出格式:
        pathlib.Path
        - 报告文件的绝对路径
    """
    # 报告目录：results/reports，不存在时自动创建。
    report_dir = Path("results/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    # 报告命名格式：report_{algo}_seed{seed}_ep{episodes}.json
    report_path = report_dir / f"report_{args.algo}_{args.scenario}_seed{args.seed}_ep{args.episodes}.json"

    # 将本轮关键配置和关键指标写入结构化字典，便于脚本化统计。
    payload = {
        "algo": args.algo,
        "scenario": str(args.scenario),
        "n": int(env.n),
        "m": int(env.m),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "best_select_mode": args.best_select_mode,
        "best_eval_every": int(args.best_eval_every),
        "best_eval_episodes": int(args.best_eval_episodes),
        "update_interval": int(args.update_interval),
        "ddpg_warmup_steps": args.ddpg_warmup_steps,
        "ddpg_actor_lr": args.ddpg_actor_lr,
        "ddpg_critic_lr": args.ddpg_critic_lr,
        "noise_std_decay": args.noise_std_decay,
        "best_metric_source": train_result.get("best_metric_source"),
        "best_metric_value": train_result.get("best_metric_value"),
        "best_model_path": train_result.get("best_model_path"),
        "last_model_path": train_result.get("last_model_path"),
        "train_seconds": train_result.get("train_seconds"),
    }
    # 写盘为 UTF-8 JSON；ensure_ascii=False 保留中文可读性。
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 返回绝对路径，方便上层日志直接打印。
    return report_path.resolve()


def _plot_train_outputs(args: argparse.Namespace, train_result: dict) -> None:
    """
    作用:
        保存训练阶段的展示图和诊断图。

    输入格式:
        args: argparse.Namespace
        train_result: dict（包含 mse_history、sum_aoi_history）

    输出格式:
        None
    """
    # 从训练结果中提取两个历史序列。
    mse_history = train_result["mse_history"]
    sum_aoi_history = train_result["sum_aoi_history"]

    # 展示图关闭时直接返回，避免无意义绘图开销。
    if not bool(args.save_plots):
        print("Training finished. Plot saving is disabled by --save-plots 0.")
        return

    # 展示图路径（裁剪版曲线）。
    mse_path = Path("results") / f"result_{args.algo}_{args.scenario}_seed{args.seed}.png"
    aoi_path = Path("results") / f"result_{args.algo}_{args.scenario}_seed{args.seed}_sumaoi.png"
    # 保存 MSE 展示图。
    saved_mse_path = plot_learning_curve(mse_history, args.algo, mse_path, window=10)
    # 保存 AoI 展示图。
    saved_aoi_path = plot_sum_aoi_curve(sum_aoi_history, args.algo, aoi_path, window=10)
    print(f"Training finished. MSE display curve saved to: {saved_mse_path}")
    print(f"Training finished. SumAoI display curve saved to: {saved_aoi_path}")

    # 诊断图开关打开时，额外保存 raw MSE 与 log(MSE) 曲线。
    if bool(args.save_diagnostic_plots):
        raw_mse_path = Path("results") / f"result_{args.algo}_{args.scenario}_seed{args.seed}_raw_mse.png"
        log_mse_path = Path("results") / f"result_{args.algo}_{args.scenario}_seed{args.seed}_log_mse.png"
        saved_raw_path = plot_learning_curve_raw(mse_history, args.algo, raw_mse_path, window=10)
        saved_log_path = plot_log_mse_curve(mse_history, args.algo, log_mse_path, window=10)
        print(f"Raw/log MSE diagnostic curves saved to: {saved_raw_path} and {saved_log_path}")


def _run_eval_mode(args: argparse.Namespace, env, agent, device) -> None:
    """
    作用:
        执行评估流程，并按配置保存评估曲线。

    输入格式:
        args: argparse.Namespace
        env: 环境实例（支持 reset/step）
        agent: 智能体实例（DQNAgent 或 DDPGAgent）
        device: torch.device

    输出格式:
        None
    """
    # 优先使用用户给定模型路径；为空时回退到默认 best 路径。
    ckpt_path = Path(args.model_path) if args.model_path else _default_best_ckpt(args.algo, args.scenario, args.seed)
    # 提前检查 checkpoint 是否存在，避免运行中断时信息不清晰。
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    # 加载 checkpoint 到当前 agent。
    meta = load_checkpoint(args.algo, agent, ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path.resolve()} | meta={meta}")

    # 执行纯评估（不做训练更新）。
    eval_result = run_evaluation(
        args.algo,
        env,
        agent,
        num_episodes=args.eval_episodes,
        episode_length=EPISODE_LENGTH,
        verbose=True,
    )

    # 根据开关保存评估曲线。
    if bool(args.save_plots):
        eval_mse_path = Path("results") / f"result_{args.algo}_{args.scenario}_seed{args.seed}_eval_mse.png"
        eval_aoi_path = Path("results") / f"result_{args.algo}_{args.scenario}_seed{args.seed}_eval_sumaoi.png"
        saved_eval_mse = plot_learning_curve(eval_result["mse_history"], args.algo, eval_mse_path, window=10)
        saved_eval_aoi = plot_sum_aoi_curve(eval_result["sum_aoi_history"], args.algo, eval_aoi_path, window=10)
        print(f"Evaluation finished. Curves saved to: {saved_eval_mse} and {saved_eval_aoi}")
    else:
        print("Evaluation finished. Plot saving is disabled by --save-plots 0.")


def main() -> None:
    """
    作用:
        程序主入口：执行训练或评估，并输出关键诊断信息。

    输入格式:
        无（参数从 CLI 读取）

    输出格式:
        None
    """
    # 解析命令行参数。
    args = parse_args()
    # 固定随机种子，提升实验复现实验一致性。
    set_global_seed(args.seed)
    # 解析并校验设备参数。
    device = resolve_device(args.device)

    # 启动日志：打印本轮任务配置。
    print(
        f"Run -> Algo: {args.algo}, Scenario: {args.scenario}, Seed: {args.seed}, Episodes: {args.episodes}, "
        f"Device: {device}, Mode: {args.mode}, "
        f"Save(ckpt/history/plots/diag)=({args.save_checkpoints}/{args.save_history}/{args.save_plots}/{args.save_diagnostic_plots})"
    )

    # 当前策略：非 base 场景仅支持 DDPG。
    if args.algo == "DQN" and args.scenario != "base":
        raise ValueError("DQN currently supports only --scenario base. Use DDPG for s10x5/s20x10.")

    # 构建环境对象（按场景规模与算法类型）。
    env = build_env(seed=args.seed, scenario=str(args.scenario), algo=str(args.algo))
    # 构建智能体对象，并获取构建说明字符串。
    agent, agent_info = build_agent(args=args, env=env, device=device)
    print(agent_info)

    # 若是 eval 模式，直接走评估分支并返回。
    if args.mode == "eval":
        _run_eval_mode(args=args, env=env, agent=agent, device=device)
        return

    # 组装训练配置字典。
    train_cfg = build_train_config(args)
    # 执行训练并获取结果。
    train_result = run_training(args.algo, env, agent, train_cfg)

    # 保存展示图/诊断图。
    _plot_train_outputs(args, train_result)
    # 保存 JSON 训练报告。
    report_path = _save_run_report(args, train_result, env)

    # 根据 checkpoint 开关打印不同说明。
    if bool(args.save_checkpoints):
        print(
            "Checkpoints -> "
            f"best: {train_result['best_model_path']}, last: {train_result['last_model_path']}"
        )
    else:
        print("Checkpoint saving is disabled by --save-checkpoints 0.")

    # 打印关键训练摘要。
    print(
        "Summary -> "
        f"best_source={train_result.get('best_metric_source')}, "
        f"best_metric={train_result.get('best_metric_value')}, "
        f"train_seconds={train_result.get('train_seconds'):.3f}"
    )
    print(f"Training report saved to: {report_path}")

    # 历史保存关闭时，不生成跨算法对比图。
    if not bool(args.save_history):
        print("History saving is disabled by --save-history 0. Compare curve skipped.")
        return

    # 保存 history，并尝试生成同配置 DQN/DDPG 对比图。
    save_history_and_compare(
        algo=str(args.algo),
        scenario=str(args.scenario),
        n=int(env.n),
        m=int(env.m),
        seed=int(args.seed),
        episodes=int(args.episodes),
        mse_history=train_result["mse_history"],
        sum_aoi_history=train_result["sum_aoi_history"],
        save_plots=bool(args.save_plots),
    )


if __name__ == "__main__":
    # 脚本入口。
    main()
