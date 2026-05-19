"""
文件作用：保存 learned adversarial DRL attacker 的训练配置、能量约束映射参数和结果保存参数。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LearnedAttackConfig:
    """
    作用:
        保存 learned adversarial DRL attacker 的训练配置与约束参数。

    输入格式:
        通常由 main.py 的 argparse.Namespace 通过 build_learned_attack_config_from_args 构造。

    输出格式:
        LearnedAttackConfig 实例。

    参数含义:
        attacker_algo: 攻击者算法名称，第一轮固定为 DDPG。
        attacker_episodes: 攻击者训练 episode 数。
        attacker_gamma: DDPG 折扣因子。
        attacker_batch_size: ReplayBuffer 每次采样 batch 大小。
        attacker_buffer_capacity: ReplayBuffer 最大容量。
        attacker_warmup_steps: 使用随机 intent action 的 warmup 步数。
        attacker_update_interval: 每隔多少个环境步更新一次参数。
        attacker_actor_lr / attacker_critic_lr: 攻击者 Actor/Critic 学习率。
        attacker_tau: 目标网络软更新系数。
        attacker_noise_std_*: 训练探索噪声的初始值、衰减和下限。
        attacker_grad_clip_norm: 梯度裁剪阈值。
        attacker_eval_every: 每隔多少个 episode 做一次 deterministic eval；0 表示关闭。
        attacker_eval_episodes: 每次 deterministic eval 的 episode 数。
        attacker_random_baseline: deterministic eval 开启时，是否同时运行 random intent baseline。
        per_step_energy_budget: 单步离散扰动能量预算。
        alpha_tau / alpha_h: AoI/H 扰动的能量代价系数。
        aoi_delta / h_delta: AoI/H 单维最大扰动幅度。
        save_attacker_*: checkpoint、history、report 保存开关。
        attacker_result_dir: learned attacker 输出根目录。
        victim_algo / victim_model_path: 被攻击 victim 的算法和 checkpoint 路径。

    核心步骤:
        1. 记录 attacker 训练超参数。
        2. 记录 SDEC-QM 映射所需的能量、幅值和代价系数。
        3. 记录 victim checkpoint 与结果保存配置。
    """

    # 第一轮只实现 DDPG attacker，因此算法默认值固定为 DDPG。
    attacker_algo: str = "DDPG"
    # episode 数默认给正式训练使用，smoke test 可由 CLI 覆盖为 1。
    attacker_episodes: int = 300
    # 折扣因子与 victim DDPG 默认值保持一致，减少额外调参维度。
    attacker_gamma: float = 0.95
    # batch size 控制每次 critic/actor 更新使用多少 transition。
    attacker_batch_size: int = 128
    # 回放池容量保留最近交互样本，避免 critic 只看极短历史。
    attacker_buffer_capacity: int = 20000
    # warmup 阶段先随机探索 intent action，避免初始策略过早主导数据分布。
    attacker_warmup_steps: int = 2000
    # update interval 用于控制训练耗时和稳定性。
    attacker_update_interval: int = 1

    # Actor 学习率通常小于 critic，沿用 DDPG 常见设置。
    attacker_actor_lr: float = 1e-4
    # Critic 学习率略大，用于更快拟合 Q 值。
    attacker_critic_lr: float = 1e-3
    # tau 越小，目标网络变化越平滑。
    attacker_tau: float = 0.005
    # 初始探索噪声用于扩大连续 intent action 覆盖范围。
    attacker_noise_std_start: float = 0.25
    # 每个 episode 后衰减噪声，逐步从探索转向利用。
    attacker_noise_std_decay: float = 0.995
    # 噪声下限保留少量探索。
    attacker_noise_std_min: float = 0.03
    # 梯度裁剪避免 MSE 奖励尺度较大时更新爆炸。
    attacker_grad_clip_norm: float = 5.0
    # deterministic eval 默认关闭，避免正式训练默认额外消耗时间。
    attacker_eval_every: int = 0
    # 每次 eval 默认跑 1 个 episode，先用于轻量诊断。
    attacker_eval_episodes: int = 1
    # 开启 eval 后默认同时输出 random intent baseline，便于同 power 横向比较。
    attacker_random_baseline: bool = True

    # 单步能量预算用于硬约束最终离散扰动。
    per_step_energy_budget: float = 2.0
    # AoI 扰动通常更敏感，因此默认代价高于 H。
    alpha_tau: float = 1.0
    # H 扰动默认代价较低，方便 attacker 探索信道扰动。
    alpha_h: float = 0.25
    # AoI 单维扰动幅度，第一轮保持最小整数幅度。
    aoi_delta: int = 1
    # H 单维扰动幅度，第一轮保持最小整数幅度。
    h_delta: int = 1

    # 默认保存 checkpoint，正式训练时可恢复 best/last。
    save_attacker_checkpoints: bool = True
    # 默认保存 history，方便后续画学习曲线。
    save_attacker_history: bool = True
    # 默认保存 report，方便记录训练配置和结果。
    save_attacker_report: bool = True
    # learned attacker 产物集中放在独立目录，避免混入 victim 结果。
    attacker_result_dir: str = "results/attacker"

    # victim 默认使用 DDPG，因为大规模场景优先支持 DDPG。
    victim_algo: str = "DDPG"
    # 空路径表示 workflow 使用默认 best checkpoint 规则。
    victim_model_path: str = ""


def _get_arg(args, name: str, default):
    """
    作用:
        从 argparse.Namespace 中读取字段，缺失时回退默认值。

    输入格式:
        args: argparse.Namespace 或具备属性访问的对象。
        name: str，需要读取的属性名。
        default: 任意类型，字段缺失时使用的默认值。

    输出格式:
        任意类型，优先为 args.name，否则为 default。

    参数含义:
        args 表示 CLI 参数容器；name 表示参数字段；default 表示配置默认值。

    核心步骤:
        1. 使用 getattr 安全读取属性。
        2. 缺失时返回默认配置值。
    """
    # getattr 带默认值可以兼容测试中的 SimpleNamespace 和真实 argparse.Namespace。
    return getattr(args, name, default)


def build_learned_attack_config_from_args(args) -> LearnedAttackConfig:
    """
    作用:
        从 main.py 的 argparse.Namespace 构造 LearnedAttackConfig。

    输入格式:
        args: argparse.Namespace，需要包含 learned attacker 相关 CLI 参数。

    输出格式:
        LearnedAttackConfig 实例。

    参数含义:
        args 中的 attacker_* 字段控制训练超参数；per_step_energy_budget、alpha_tau、
        alpha_h、aoi_delta、h_delta 控制 SDEC-QM 映射；victim_* 字段控制固定 victim 加载。

    核心步骤:
        1. 先创建默认配置，保证缺省参数可用。
        2. 从 args 读取 learned attacker 训练超参数。
        3. 从 args 读取诊断评估参数。
        4. 从 args 读取 SDEC-QM 映射约束参数。
        5. 将 0/1 保存开关转换为 bool。
    """
    # 先构造默认配置，后续每个字段都以默认值为回退。
    default = LearnedAttackConfig()
    # 返回新的 dataclass，避免修改 default 临时对象。
    return LearnedAttackConfig(
        attacker_algo=str(_get_arg(args, "attacker_algo", default.attacker_algo)).upper(),
        attacker_episodes=int(_get_arg(args, "attacker_episodes", default.attacker_episodes)),
        attacker_gamma=float(_get_arg(args, "attacker_gamma", default.attacker_gamma)),
        attacker_batch_size=int(_get_arg(args, "attacker_batch_size", default.attacker_batch_size)),
        attacker_buffer_capacity=int(_get_arg(args, "attacker_buffer_capacity", default.attacker_buffer_capacity)),
        attacker_warmup_steps=int(_get_arg(args, "attacker_warmup_steps", default.attacker_warmup_steps)),
        attacker_update_interval=int(_get_arg(args, "attacker_update_interval", default.attacker_update_interval)),
        attacker_actor_lr=float(_get_arg(args, "attacker_actor_lr", default.attacker_actor_lr)),
        attacker_critic_lr=float(_get_arg(args, "attacker_critic_lr", default.attacker_critic_lr)),
        attacker_tau=float(_get_arg(args, "attacker_tau", default.attacker_tau)),
        attacker_noise_std_start=float(_get_arg(args, "attacker_noise_std_start", default.attacker_noise_std_start)),
        attacker_noise_std_decay=float(_get_arg(args, "attacker_noise_std_decay", default.attacker_noise_std_decay)),
        attacker_noise_std_min=float(_get_arg(args, "attacker_noise_std_min", default.attacker_noise_std_min)),
        attacker_grad_clip_norm=float(_get_arg(args, "attacker_grad_clip_norm", default.attacker_grad_clip_norm)),
        attacker_eval_every=int(_get_arg(args, "attacker_eval_every", default.attacker_eval_every)),
        attacker_eval_episodes=int(_get_arg(args, "attacker_eval_episodes", default.attacker_eval_episodes)),
        attacker_random_baseline=bool(int(_get_arg(args, "attacker_random_baseline", int(default.attacker_random_baseline)))),
        per_step_energy_budget=float(_get_arg(args, "per_step_energy_budget", default.per_step_energy_budget)),
        alpha_tau=float(_get_arg(args, "alpha_tau", default.alpha_tau)),
        alpha_h=float(_get_arg(args, "alpha_h", default.alpha_h)),
        aoi_delta=int(_get_arg(args, "aoi_delta", default.aoi_delta)),
        h_delta=int(_get_arg(args, "h_delta", default.h_delta)),
        save_attacker_checkpoints=bool(int(_get_arg(args, "save_attacker_checkpoints", int(default.save_attacker_checkpoints)))),
        save_attacker_history=bool(int(_get_arg(args, "save_attacker_history", int(default.save_attacker_history)))),
        save_attacker_report=bool(int(_get_arg(args, "save_attacker_report", int(default.save_attacker_report)))),
        attacker_result_dir=str(_get_arg(args, "attacker_result_dir", default.attacker_result_dir)),
        victim_algo=str(_get_arg(args, "victim_algo", default.victim_algo)).upper(),
        victim_model_path=str(_get_arg(args, "victim_model_path", default.victim_model_path)),
    )
