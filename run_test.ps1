$trainArgs = @(
    "--algo", "DQN",
    "--scenario", "base",
    "--mode", "train",
    "--seed", "24",
    "--episodes", "300",
    "--device", "cpu",
    "--save-checkpoints", "1",
    "--save-history", "1",
    "--save-plots", "1",
    "--best-select-mode", "eval",
    "--best-eval-every", "10",
    "--best-eval-episodes", "3",
    "--update-interval", "2"
)
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
    "--attack-mode", "semantic_h_expected_cost"
    # 攻击概率
    "--attack-prob", "0.5"
    # 最大攻击比例
    "--max-attack-ratio", "0.5"
    # AoI 攻击扰动步长
    "--aoi-delta", "0"
    # h 攻击扰动步长
    "--h-delta", "4"
    # 最大 AoI 特征数量
    "--max-aoi-features", "0"
    # 最大 h 特征数量
    "--max-h-features", "10"
    # 最大总特征数量
    "--max-total-features", "21"
    # 是否严格遵循预算
    "--strict-budget", "1"
    # 期望代价模式：均方误差
    "--expected-cost-mode", "mse"
)

# 执行攻击评估
python eval_attack.py @evalAttackArgs
# python main.py @trainArgs