
# 训练攻击者模型的参数
$attackerArgs = @(
    # 指定任务类型：训练 attacker
    "--task", "attacker"
    # 指定被攻击的 victim 算法
    "--victim-algo", "DDPG"
    # 使用 base 场景配置
    "--scenario", "base"
    # 固定随机种子，保证实验可复现
    "--seed", "24"
    # 使用 CPU 运行
    "--device", "cpu"
    # 加载已经训练好的 DDPG victim 模型
    "--victim-model-path", "results/checkpoints/ddpg_base_seed24_best.pt"
    # attacker 训练轮数
    "--attacker-episodes", "50"
    # 每一步攻击的能量预算
    "--per-step-energy-budget", "10.0"
    # tau 相关攻击权重
    "--alpha-tau", "1.0"
    # h 相关攻击权重
    "--alpha-h", "0.25"
    # AoI 攻击扰动步长
    "--aoi-delta", "3"
    # h 攻击扰动步长
    "--h-delta", "2"
    # 保存 attacker checkpoint
    "--save-attacker-checkpoints", "0"
    # 保存 attacker 训练历史
    "--save-attacker-history", "0"
    # 保存 attacker 实验报告
    "--save-attacker-report", "1"
    # 是否在训练时评估
    "--attacker-eval-every", "1"
    # 平均评估的episode数
    "--attacker-eval-episodes", "1"
    # 同预算下与random做对比
    "--attacker-random-baseline", "1"
)
# 评估语义攻击的参数
$evalAttackArgs = @(
    # 指定被攻击的 victim 算法
    "--algo", "DDPG"
    # 使用 base 场景配置
    "--scenario", "base"
    # 固定随机种子，保证实验可复现
    "--seed", "24"
    # 使用 CPU 运行
    "--device", "cpu"
    # 评估的 episode 数量
    "--eval-episodes", "10"
    # 攻击模式：语义联合期望代价攻击
    "--attack-mode", "random_joint"
    # 攻击概率
    "--attack-prob", "1"
    # 最大攻击比例
    "--max-attack-ratio", "1"
    # AoI 攻击扰动步长
    "--aoi-delta", "3"
    # h 攻击扰动步长
    "--h-delta", "2"
    # 最大 AoI 特征数量
    "--max-aoi-features", "3"
    # 最大 h 特征数量
    "--max-h-features", "18"
    # 最大总特征数量
    "--max-total-features", "21"
    # 是否严格遵循预算
    "--strict-budget", "1"
    # 期望代价模式：均方误差
    "--expected-cost-mode", "mse"
)

# 执行攻击评估
python eval_attack.py @evalAttackArgs
# 执行 attacker 训练
# python main.py @attackerArgs