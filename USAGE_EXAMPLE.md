# Usage Example

本文档重新整理为三类：`通用指令`、`仅 DQN 指令`、`仅 DDPG 指令`。  
你可以直接按分类复制命令，避免参数混用。

## 1. 环境准备（通用）

```bash
# 进入项目目录
cd C:\Users\28681\Desktop\Kalman\sematic

# （可选）创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖
pip install numpy scipy matplotlib torch
```

## 2. 参数适用范围总表（重点）

### 2.1 通用参数（DQN/DDPG 都可用）

- `--algo {DQN,DDPG}`：选择算法
- `--scenario {base,s10x5,s20x10}`：场景预配置（默认 `base`）
- `--mode {train,eval}`：训练或评估
- `--seed <int>`：随机种子
- `--episodes <int>`：训练轮数（train 模式）
- `--device {cpu,cuda}`：训练设备
- `--model-path <path>`：评估模型路径（eval 模式）
- `--eval-episodes <int>`：评估轮数（eval 模式）
- `--save-checkpoints {0,1}`：是否保存 checkpoint
- `--save-history {0,1}`：是否保存 history（用于跨算法对比）
- `--save-plots {0,1}`：是否保存展示曲线图
- `--save-diagnostic-plots {0,1}`：是否保存诊断图（raw MSE / log MSE，默认 `0`）
- `--best-select-mode {train,eval}`：best checkpoint 选择依据
- `--best-eval-every <int>`：每 K 个 episode 做一次 clean-eval（best 选择用）
- `--best-eval-episodes <int>`：每次 clean-eval 的 episode 数
- `--update-interval <int>`：每几步做一次参数更新（1=每步更新，2=隔步更新）

### 2.2 仅 DDPG 参数（DQN 下会被忽略）

- `--ddpg-warmup-steps <int>`：DDPG warmup 步数
- `--ddpg-actor-lr <float>`：DDPG actor 学习率
- `--ddpg-critic-lr <float>`：DDPG critic 学习率
- `--noise-std-decay <float>`：DDPG 探索噪声衰减系数

### 2.3 仅 DQN 相关说明

- `--dqn-warmup-steps <int>`：DQN warmup 步数（warmup 阶段仅随机采样，不更新参数；并冻结 epsilon 衰减）。  
- DQN 的学习率、epsilon 等参数来自 `config.py`（如 `DQN_LR`, `EPSILON_DECAY`）。
- 当前 DQN 仅支持 `--scenario base`。

### 2.4 eval_attack 输入参数说明（攻击评估专用）

- `--attack-mode <mode>`：攻击模式。可选 `clean`、`random_aoi`、`random_h`、`random_joint`、`semantic_aoi_mislead`、`semantic_h_mislead`、`semantic_joint_mislead`、`semantic_aoi_expected_cost`、`semantic_h_expected_cost`、`semantic_joint_expected_cost`。
- `--attack-prob <float>`：每个 step 尝试攻击的概率；即使概率命中，仍会受总攻击步数、连续攻击步数和 cooldown 限制。
- `--max-attack-ratio <float>`：每个 episode 最多允许攻击的 step 比例，例如 `0.2` 表示最多攻击 20% 的步数。
- `--strict-budget {0,1}`：是否严格遵守稀疏预算；推荐 `1`。若为 `1`，扰动 L0 不允许超过配置预算。
- `--aoi-delta <int>`：单个 AoI 特征每次最多改变量。AoI 攻击会在合法范围内 clip。
- `--h-delta <int>`：单个 H 特征每次最多改变量。这里 H 越大表示丢包率越低，因此降低 H 表示伪造更差的信道，提高 H 表示伪造更好的信道。
- `--max-aoi-features <int>`：单次攻击最多扰动多少个 AoI 维度。
- `--max-h-features <int>`：单次攻击最多扰动多少个 H 维度。
- `--max-total-features <int>`：单次 joint 攻击的 AoI+H 总 L0 上限。
- `--aoi-direction {random,increase,decrease,mixed}`：random AoI 攻击方向；结构化 mislead / expected-cost 攻击会按自身策略选择方向。
- `--h-direction {random,increase,decrease,mixed}`：random H 攻击方向；结构化 mislead / expected-cost 攻击会按自身策略选择方向。
- `--max-consecutive-steps <int>`：最多允许连续攻击多少个 step。
- `--cooldown-steps <int>`：连续攻击达到上限后，需要等待多少个 step 才能再次攻击。
- `--record-perturbation {0,1}`：是否统计扰动 L0/L1/Linf 等信息；推荐保持 `1`。
- `--save-report {0,1}`：是否保存 JSON 攻击报告到 `results/attack_reports/`。
- `--expected-cost-mode {mse,sum_aoi}`：expected-cost 攻击的候选状态评分目标。`mse` 使用下一步期望 MSE，`sum_aoi` 使用下一步期望 SumAoI；默认 `mse`。

### 2.5 eval_attack 报告字段说明（clean / attack 输出）

- `algo`、`scenario`、`n`、`m`、`seed`、`device`、`eval_episodes`、`checkpoint_path`：本次评估的算法、场景规模、随机种子、设备、episode 数和模型路径。
- `attack_mode`：实际使用的攻击模式；`clean` 表示无攻击。
- `attack_prob`、`max_attack_ratio`、`strict_budget`、`aoi_delta`、`h_delta`、`max_aoi_features`、`max_h_features`、`max_total_features`、`aoi_direction`、`h_direction`、`max_consecutive_steps`、`expected_cost_mode`：本次攻击配置的扁平记录。
- `clean_mean_mse`：clean 环境下的平均 Sum MSE。
- `attack_mean_mse`：攻击观测下执行策略后的平均 Sum MSE。
- `mse_degradation`：`attack_mean_mse - clean_mean_mse`，越大表示攻击导致 MSE 退化越明显。
- `mse_degradation_ratio`：MSE 退化相对 clean 的比例。
- `clean_mean_sum_aoi`：clean 环境下的平均 SumAoI。
- `attack_mean_sum_aoi`：攻击观测下执行策略后的平均 SumAoI。
- `sum_aoi_degradation`：`attack_mean_sum_aoi - clean_mean_sum_aoi`，越大表示 AoI 退化越明显。
- `sum_aoi_degradation_ratio`：SumAoI 退化相对 clean 的比例。
- `attack_step_ratio`：实际发生攻击的 step 占比。
- `action_flip_ratio`：clean action 与 attacked action 不一致的比例。
- `avg_aoi_l0_per_step`、`avg_h_l0_per_step`、`avg_total_l0_per_step`：按全部 step 平均的 AoI/H/总扰动稀疏度。
- `avg_aoi_l1_per_attack`、`avg_h_l1_per_attack`：按攻击 step 平均的 AoI/H 扰动 L1 强度。
- `max_aoi_linf`、`max_h_linf`：AoI/H 单维最大扰动幅度。
- `constraint_violation_count`：预算或约束违规次数；正常应为 `0`。
- `clean_selected_risk_sum`、`attack_selected_risk_sum`：clean / attack action 选中传感器的风险总和。
- `resource_misallocation_score`：`clean_selected_risk_sum - attack_selected_risk_sum`；越大表示攻击越能把资源从高风险传感器移走。
- `high_risk_scheduled_ratio_clean`、`high_risk_scheduled_ratio_attack`：clean / attack action 对高风险传感器的调度比例。
- `low_risk_scheduled_ratio_clean`、`low_risk_scheduled_ratio_attack`：clean / attack action 对低风险传感器的调度比例。
- `high_risk_good_channel_ratio_clean`、`high_risk_good_channel_ratio_attack`：高风险传感器被分配到好信道的比例，用于观察结构性资源错配。

### 2.6 Learned Attacker 训练参数说明

- `--task {victim,attacker}`：运行任务类型。`victim` 保持原 DQN/DDPG 训练或评估；`attacker` 启动 learned meta-attacker 训练。
- `--victim-algo {DQN,DDPG}`：被攻击的固定 victim scheduler 算法。
- `--victim-model-path <path>`：victim checkpoint 路径；为空时默认读取 `results/checkpoints/{victim_algo_lower}_{scenario}_seed{seed}_best.pt`。
- `--attacker-algo DDPG`：learned attacker 算法，第一轮只支持 DDPG。
- `--attacker-episodes <int>`：attacker 训练 episode 数。
- `--attacker-episode-length <int>`：attacker 每个 episode 的 step 数，默认 `500`；快速验证可设为 `20` 或 `50`。
- `--attacker-gamma <float>`：attacker DDPG 的折扣因子。
- `--attacker-actor-lr <float>`、`--attacker-critic-lr <float>`：attacker Actor/Critic 学习率。
- `--attacker-tau <float>`：target 网络软更新系数。
- `--attacker-warmup-steps <int>`：warmup 期间使用随机 continuous intent action。
- `--attacker-batch-size <int>`、`--attacker-buffer-capacity <int>`、`--attacker-update-interval <int>`：ReplayBuffer 与参数更新控制。
- `--attacker-noise-std-start <float>`、`--attacker-noise-std-decay <float>`、`--attacker-noise-std-min <float>`：Actor 输出 intent action 后叠加的探索噪声配置。
- `--attacker-eval-every <int>`：每隔多少个 attacker 训练 episode 做一次 deterministic eval；`0` 表示关闭，开启后不会写 ReplayBuffer、不会更新网络。
- `--attacker-eval-episodes <int>`：每次 deterministic eval 跑多少个 episode。
- `--attacker-random-baseline {0,1}`：开启 deterministic eval 时，是否同时运行 random intent baseline；该 baseline 使用同一 `per_step_energy_budget`、`alpha_tau`、`alpha_h`、`aoi_delta`、`h_delta`，用于同 power 对照。
- `--attacker-reward-clip <float>`：训练写入 ReplayBuffer 前对 raw MSE reward 做上限裁剪；报告仍记录 raw MSE。
- `--per-step-energy-budget <float>`：每一步离散扰动的能量预算上限。
- `--alpha-tau <float>`、`--alpha-h <float>`：AoI/H 扰动能量代价系数，能耗为 `alpha_i * delta_i^2`。
- `--aoi-delta <int>`、`--h-delta <int>`：AoI/H 单维最大整数扰动幅度。
- `--save-attacker-checkpoints {0,1}`、`--save-attacker-history {0,1}`、`--save-attacker-report {0,1}`：是否保存 attacker checkpoint、history 和 report。
- `--attacker-result-dir <path>`：attacker 输出根目录，默认 `results/attacker`。
- learned attacker 使用 raw state，不做状态归一化。
- learned attacker 不使用 `attack_prob`、`max_attack_ratio`、`max_consecutive_steps`；这些参数只属于 `eval_attack.py` 的规则/启发式攻击评估。

## 3. 训练命令模板（严格分开）

### 3.1 仅 DQN 训练命令

#### DQN-Train-Basic（保存主要结果）

```bash
python main.py --algo DQN --scenario base --mode train --seed 42 --episodes 300 --device cpu --dqn-warmup-steps 2000 --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DQN-Train-Fast（快速回归，不落盘）

```bash
python main.py --algo DQN --scenario base --mode train --seed 1 --episodes 1 --device cpu --dqn-warmup-steps 0 --save-checkpoints 0 --save-history 0 --save-plots 0 --save-diagnostic-plots 0 --best-select-mode eval --best-eval-every 1 --best-eval-episodes 1 --update-interval 1
```

### 3.2 仅 DDPG 训练命令

#### DDPG-B0（基线）

```bash
python main.py --algo DDPG --scenario base --mode train --seed 7 --episodes 300 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DDPG-S1（增大 warmup）

```bash
python main.py --algo DDPG --scenario base --mode train --seed 7 --episodes 300 --device cpu --ddpg-warmup-steps 5000 --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DDPG-S2（在 S1 上降低 actor lr）

```bash
python main.py --algo DDPG --scenario base --mode train --seed 7 --episodes 300 --device cpu --ddpg-warmup-steps 5000 --ddpg-actor-lr 5e-5 --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DDPG-S3a / S3b（噪声衰减两候选）

```bash
python main.py --algo DDPG --scenario base --mode train --seed 7 --episodes 300 --device cpu --ddpg-warmup-steps 5000 --ddpg-actor-lr 5e-5 --noise-std-decay 0.996 --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
python main.py --algo DDPG --scenario base --mode train --seed 7 --episodes 300 --device cpu --ddpg-warmup-steps 5000 --ddpg-actor-lr 5e-5 --noise-std-decay 0.998 --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DDPG-T1（提速：隔步更新）

```bash
python main.py --algo DDPG --scenario base --mode train --seed 7 --episodes 300 --device cpu --ddpg-warmup-steps 5000 --ddpg-actor-lr 5e-5 --noise-std-decay 0.996 --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 2
```

#### DDPG-Scale-10x5（仅 DDPG）

```bash
python main.py --algo DDPG --scenario s10x5 --mode train --seed 7 --episodes 300 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DDPG-Scale-20x10（仅 DDPG）

```bash
python main.py --algo DDPG --scenario s20x10 --mode train --seed 7 --episodes 300 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

## 4. 评估命令模板（分算法）

### 4.1 DQN 评估（仅 DQN）

```bash
python main.py --algo DQN --scenario base --mode eval --seed 42 --device cpu --model-path results\checkpoints\dqn_base_seed42_best.pt --eval-episodes 10 --save-plots 1
```

### 4.2 DDPG 评估（仅 DDPG）

```bash
python main.py --algo DDPG --scenario s20x10 --mode eval --seed 42 --device cpu --model-path results\checkpoints\ddpg_s20x10_seed42_best.pt --eval-episodes 10 --save-plots 1
```

### 4.3 eval_attack（攻击评估）

#### eval_attack-DQN-clean

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --model-path results\checkpoints\dqn_base_seed24_best.pt --eval-episodes 10 --attack-mode clean
```

#### eval_attack-DDPG-clean

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --model-path results\checkpoints\ddpg_base_seed24_best.pt --eval-episodes 10 --attack-mode clean
```

#### eval_attack-默认 best checkpoint（model-path 留空）

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode clean
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode clean
```

## 5. 攻击评估命令模板

本节命令只运行评估，不保存 checkpoint、history、图片；默认只把 JSON 报告保存到 `results/attack_reports/`。

### 5.1 DQN：约束 random_aoi

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode random_aoi --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --max-aoi-features 1 --max-total-features 1 --aoi-direction random --strict-budget 1
```

### 5.2 DQN：约束 random_h

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode random_h --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 1 --max-h-features 2 --max-total-features 2 --h-direction random --strict-budget 1
```

### 5.3 DQN：约束 random_joint

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode random_joint --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --h-delta 1 --max-aoi-features 1 --max-h-features 2 --max-total-features 3 --aoi-direction random --h-direction random --strict-budget 1
```

### 5.4 DDPG：约束 random_joint

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode random_joint --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --h-delta 1 --max-aoi-features 1 --max-h-features 2 --max-total-features 3 --aoi-direction random --h-direction random --strict-budget 1
```

### 5.5 DQN：结构化感知 mislead 攻击

AoI-only mislead：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_aoi_mislead --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --max-aoi-features 2 --max-total-features 2 --strict-budget 1
```

H-only mislead：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_h_mislead --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 4 --max-h-features 4 --max-total-features 4 --strict-budget 1
```

Joint mislead：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_mislead --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1
```

### 5.6 DDPG：结构化感知 mislead 攻击

AoI-only mislead：

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_aoi_mislead --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --max-aoi-features 2 --max-total-features 2 --strict-budget 1
```

H-only mislead：

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_h_mislead --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 4 --max-h-features 4 --max-total-features 4 --strict-budget 1
```

Joint mislead：

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_mislead --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1
```

### 5.7 结构化感知 mislead 的含义

- `semantic_aoi_mislead`：降低高风险 sensor 的观测 AoI，抬高低风险 sensor 的观测 AoI，诱导策略低估真正紧急的 sensor。
- `semantic_h_mislead`：降低高 priority 链路的观测 H，抬高低风险诱饵链路的观测 H。H 越大表示丢包率越低，因此降低 H 表示把关键链路伪装得更差。
- `semantic_joint_mislead`：同时执行 AoI 与 H 的结构化误导，并受 `--max-total-features` 总 L0 预算约束。

### 5.8 DQN：结构化 expected-cost 攻击

AoI-only expected-cost：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_aoi_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --max-aoi-features 2 --max-total-features 2 --strict-budget 1 --expected-cost-mode mse
```

H-only expected-cost：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_h_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 4 --max-h-features 4 --max-total-features 4 --strict-budget 1 --expected-cost-mode mse
```

Joint expected-cost：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1 --expected-cost-mode mse
```

### 5.9 DDPG：结构化 expected-cost 攻击

AoI-only expected-cost：

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_aoi_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --max-aoi-features 2 --max-total-features 2 --strict-budget 1 --expected-cost-mode mse
```

H-only expected-cost：

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_h_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 4 --max-h-features 4 --max-total-features 4 --strict-budget 1 --expected-cost-mode mse
```

Joint expected-cost：

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1 --expected-cost-mode mse
```

### 5.10 expected-cost 目标切换

默认 `--expected-cost-mode mse`，表示攻击者枚举候选扰动后，选择让 victim action 的下一步期望 MSE 最大的候选。  
如果希望用 AoI 作为更轻量的代理目标，可以改为：

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_expected_cost --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1 --expected-cost-mode sum_aoi
```

## 6. 输出文件说明（通用）

### 6.1 展示曲线（`--save-plots 1`）

- `results/result_{algo}_{scenario}_seed{seed}.png`
- `results/result_{algo}_{scenario}_seed{seed}_sumaoi.png`
- 训练展示图采用动态纵轴：按最后 10% episode 的均值构造范围，目标比例固定 35%，并对超出上界的点做硬裁剪。

### 6.2 诊断曲线（`--save-diagnostic-plots 1`）

- `results/result_{algo}_{scenario}_seed{seed}_raw_mse.png`
- `results/result_{algo}_{scenario}_seed{seed}_log_mse.png`
- 诊断图不使用动态纵轴（保持原始诊断行为）。

### 6.3 训练报告（每次 train 都会生成）

- `results/reports/report_{algo}_{scenario}_seed{seed}_ep{episodes}.json`
- 关键字段：
  - `best_metric_source`
  - `best_metric_value`
  - `train_seconds`

### 6.4 eval_attack 攻击评估报告

- `results/attack_reports/attack_eval_{algo}_{scenario}_seed{seed}_{attack_mode}.json`
- 关键字段：
  - `attack_mode`
  - `checkpoint_path`
  - `clean_mean_mse`
  - `attack_mean_mse`
  - `mse_degradation`
  - `clean_mean_sum_aoi`
  - `attack_mean_sum_aoi`
  - `sum_aoi_degradation`
  - `attack_step_ratio`
  - `action_flip_ratio`
  - `constraint_violation_count`

### 6.5 history 与跨算法对比（`--save-history 1`）

- `results/histories/{algo}_{scenario}_seed{seed}_ep{episodes}.npz`
- 若 DQN 和 DDPG 同配置 history 同时存在，且 `--save-plots 1`，会输出：
  - `results/compare_DQN_DDPG_{scenario}_seed{seed}_ep{episodes}_mse.png`
  - `results/compare_DQN_DDPG_{scenario}_seed{seed}_ep{episodes}_sumaoi.png`
- 对比图采用动态纵轴：基准取 DQN/DDPG 最后 10% 均值中的较大值，目标比例固定 35%，并进行硬裁剪。

## 7. 防混用提示（务必看）

1. 你在跑 `--algo DQN` 时，不要加 `--ddpg-*` 参数。  
2. 你在跑 `--algo DDPG` 时，可以用 `--ddpg-*` 组合调稳定性。  
3. 你在跑 `--algo DQN` 时，`--scenario` 只能是 `base`。  
4. `--update-interval` 是通用参数，两种算法都生效。  
5. `--best-select-mode eval` 是通用推荐设置，适合你“best checkpoint 为准”的流程。

## 8. Learned Attacker 训练命令

learned attacker 必须先有 victim checkpoint。若 `--victim-model-path` 为空，程序会按默认 best checkpoint 路径加载；如果文件不存在，会明确报错，不会静默训练随机 victim。

### 8.1 DDPG victim：正式训练

```bash
python main.py --task attacker --victim-algo DDPG --scenario base --seed 24 --device cpu --victim-model-path results/checkpoints/ddpg_base_seed24_best.pt --attacker-episodes 300 --attacker-eval-every 10 --attacker-eval-episodes 3 --attacker-random-baseline 1 --per-step-energy-budget 2.0 --alpha-tau 1.0 --alpha-h 0.25 --aoi-delta 1 --h-delta 1 --save-attacker-checkpoints 1 --save-attacker-history 1 --save-attacker-report 1
```

### 8.2 DDPG victim：快速 smoke test

```bash
python main.py --task attacker --victim-algo DDPG --scenario base --seed 24 --device cpu --victim-model-path results/checkpoints/ddpg_base_seed24_best.pt --attacker-episodes 1 --attacker-episode-length 20 --attacker-warmup-steps 10 --attacker-eval-every 1 --attacker-eval-episodes 1 --attacker-random-baseline 1 --save-attacker-checkpoints 0 --save-attacker-history 0 --save-attacker-report 0
```

### 8.3 DQN victim：轻量验证

```bash
python main.py --task attacker --victim-algo DQN --scenario base --seed 24 --device cpu --victim-model-path results/checkpoints/dqn_base_seed24_best.pt --attacker-episodes 1 --attacker-episode-length 20 --attacker-warmup-steps 10 --attacker-eval-every 1 --attacker-eval-episodes 1 --attacker-random-baseline 1 --save-attacker-checkpoints 0 --save-attacker-history 0 --save-attacker-report 0
```

### 8.4 输出文件

- checkpoint：`results/attacker/checkpoints/attacker_ddpg_vs_{victim_algo}_{scenario}_seed{seed}_best.pt`
- history：`results/attacker/histories/attacker_ddpg_vs_{victim_algo}_{scenario}_seed{seed}_history.json`
- report：`results/attacker/reports/attacker_ddpg_vs_{victim_algo}_{scenario}_seed{seed}_report.json`

新增诊断字段：

- `aoi_energy_history`、`h_energy_history`：训练 episode 内 AoI/H 扰动平均能耗，二者之和等于 `energy_used_history`。
- `aoi_l0_history`、`h_l0_history`：训练 episode 内 AoI/H 平均扰动维度数，二者之和等于 `total_l0_history`。
- `constraint_violation_count_history`：每个训练 episode 的约束违规次数；正常应为 `0`。
- `deterministic_eval_history`：按 `--attacker-eval-every` 触发的纯策略评估结果，Actor 使用 `noise_std=0`。
- `random_intent_baseline_history`：同 power random intent 对照结果；它复用 learned attacker 的 mapping 和能量预算，只把 intent action 换成均匀随机向量。

