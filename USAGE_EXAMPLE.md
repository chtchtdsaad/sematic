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

### 2.4 eval_attack 攻击评估输入参数

以下参数只用于 `eval_attack.py`，用于 clean、random 系列和已实现的结构化 mislead 攻击评估。`eval_attack.py` 不训练模型，不保存 checkpoint、history、图片，只按 `--save-report` 控制是否保存 JSON 报告。

#### 2.4.1 基础评估参数

- `--algo {DQN,DDPG}`：选择被评估的算法。DQN 输出离散 `action_id`，DDPG 输出 `assignment`。
- `--scenario {base,s10x5,s20x10}`：选择环境规模预设。DQN 当前建议使用 `base`；DDPG 可用于更大规模场景。
- `--seed <int>`：随机种子。用于构建环境、加载默认 checkpoint 文件名，并作为攻击随机数种子来源。
- `--device {cpu,cuda}`：模型推理设备。没有 CUDA 时使用 `cpu`。
- `--model-path <path>`：手动指定 checkpoint 路径。留空时默认使用 `results/checkpoints/{algo}_{scenario}_seed{seed}_best.pt`。
- `--eval-episodes <int>`：评估 episode 数。数值越大结果越稳定，但耗时越长。
- `--save-report {0,1}`：是否保存 JSON 报告。`1` 表示保存到 `results/attack_reports/`，`0` 表示只打印终端摘要。

#### 2.4.2 攻击模式参数

- `--attack-mode {clean,random_aoi,random_h,random_joint,semantic_aoi_mislead,semantic_h_mislead,semantic_joint_mislead}`：攻击模式。
  - `clean`：不攻击，`attack_mean_*` 与 `clean_mean_*` 相同，攻击统计为 0。
  - `random_aoi`：只扰动 agent 看到的 AoI 段。
  - `random_h`：只扰动 agent 看到的信道状态 H 段。
  - `random_joint`：同时扰动 AoI 和 H，但仍受总稀疏预算约束。
  - `semantic_aoi_mislead`：只扰动 AoI 观测，降低高风险 sensor 的观测 AoI，并抬高低风险诱饵 sensor 的观测 AoI。
  - `semantic_h_mislead`：只扰动 H 观测，降低高风险高成功率链路的观测 H，并抬高低风险诱饵链路的观测 H。
  - `semantic_joint_mislead`：同时执行 AoI 和 H 的结构化误导扰动。
- `--attack-prob <float>`：每个 step 尝试攻击的概率，范围建议 `[0,1]`。例如 `0.2` 表示每步有 20% 概率尝试攻击。
- `--max-attack-ratio <float>`：episode 内攻击步比例上限，范围建议 `[0,1]`。例如 `0.2` 表示最多攻击约 20% 的 step。
- `--strict-budget {0,1}`：是否严格执行 `max_attack_ratio`。`1` 表示攻击步数达到预算后停止攻击。
- `--max-consecutive-steps <int>`：最大连续攻击步数。达到该值后下一次攻击尝试会被跳过，用于限制攻击连续性。
- `--cooldown-steps <int>`：冷却步数预留参数。当前实现保留该字段，暂不执行复杂冷却逻辑。
- `--record-perturbation {0,1}`：扰动记录开关预留参数。当前报告始终输出汇总统计字段。

#### 2.4.3 AoI 扰动约束参数

- `--aoi-delta <int>`：AoI 单维最大扰动幅值。扰动后会执行 `round + clip`，保证 AoI 在 `[1, env.max_aoi]`。
- `--max-aoi-features <int>`：每个 step 最多扰动多少个 AoI 维度。
- `--aoi-direction {random,increase,decrease,mixed}`：AoI 扰动方向。
  - `random`：在 `[-aoi_delta, aoi_delta]` 中随机取非零整数。
  - `increase`：只增大 AoI。
  - `decrease`：只减小 AoI。
  - `mixed`：每个被选维度随机增大或减小。

#### 2.4.4 H 扰动约束参数

- `--h-delta <int>`：H 单维最大扰动幅值。扰动后会执行 `round + clip`，保证 H 在 `[0,4]`。
- `--max-h-features <int>`：每个 step 最多扰动多少个 H 维度。
- `--h-direction {random,increase,decrease,mixed}`：H 扰动方向，含义与 `--aoi-direction` 相同。
- H 的索引越大表示丢包率越低、信道越好。结构化 H 攻击中，降低 H 表示把优质链路伪装得更差，抬高 H 表示把诱饵链路伪装得更好。

#### 2.4.5 总稀疏约束参数

- `--max-total-features <int>`：每个 step 最多扰动的总维度数。`random_joint` 下必须满足 `aoi_l0 + h_l0 <= max_total_features`。
- 推荐约束关系：`max_total_features >= max_aoi_features` 且 `max_total_features >= max_h_features`；如果更小，实际扰动维度会被总预算截断。

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

### 5.5 DQN：结构化 AoI-only mislead

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_aoi_mislead --attack-prob 1.0 --max-attack-ratio 0.5 --aoi-delta 5 --max-aoi-features 2 --max-total-features 2 --strict-budget 1
```

### 5.6 DQN：结构化 H-only mislead

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_h_mislead --attack-prob 1.0 --max-attack-ratio 0.5 --h-delta 4 --max-h-features 4 --max-total-features 4 --strict-budget 1
```

### 5.7 DQN：结构化 Joint mislead

```bash
python eval_attack.py --algo DQN --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_mislead --attack-prob 1.0 --max-attack-ratio 0.5 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1
```

### 5.8 DDPG：结构化 Joint mislead

```bash
python eval_attack.py --algo DDPG --scenario base --seed 24 --device cpu --eval-episodes 10 --attack-mode semantic_joint_mislead --attack-prob 1.0 --max-attack-ratio 0.5 --aoi-delta 5 --h-delta 4 --max-aoi-features 2 --max-h-features 4 --max-total-features 6 --strict-budget 1
```

### 5.9 小规模三档预算

base 场景下 `N=6, M=3`，可以用以下三档预算比较 random 与结构化 mislead：

```text
轻度：--max-aoi-features 1 --max-h-features 2 --max-total-features 3
中度：--max-aoi-features 2 --max-h-features 4 --max-total-features 6
强度：--max-aoi-features 3 --max-h-features 6 --max-total-features 9
```

建议先固定：

```text
--attack-prob 1.0
--max-attack-ratio 0.5
--strict-budget 1
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
- clean 和 attack 都使用同一份报告结构。`attack_mode=clean` 时，`attack_mean_mse == clean_mean_mse`，`attack_mean_sum_aoi == clean_mean_sum_aoi`，攻击统计字段为 0。

#### 6.4.1 实验元信息字段

- `algo`：被评估算法，取值为 `DQN` 或 `DDPG`。
- `scenario`：环境规模预设，例如 `base`、`s10x5`、`s20x10`。
- `n`：传感器数量。
- `m`：信道数量。
- `seed`：本次评估使用的随机种子。
- `device`：模型推理设备，例如 `cpu` 或 `cuda`。
- `eval_episodes`：评估 episode 数。
- `checkpoint_path`：实际加载的 checkpoint 绝对路径。

#### 6.4.2 攻击配置字段

- `attack_mode`：攻击模式，取值为 `clean`、`random_aoi`、`random_h`、`random_joint`、`semantic_aoi_mislead`、`semantic_h_mislead`、`semantic_joint_mislead`。
- `attack_prob`：每个 step 尝试攻击的概率。
- `max_attack_ratio`：每个 episode 内攻击步比例上限。
- `strict_budget`：是否严格执行攻击步预算。
- `aoi_delta`：AoI 单维最大扰动幅值。
- `h_delta`：H 单维最大扰动幅值。
- `max_aoi_features`：每步最多扰动的 AoI 维度数。
- `max_h_features`：每步最多扰动的 H 维度数。
- `max_total_features`：每步最多扰动的总维度数。
- `aoi_direction`：AoI 扰动方向。
- `h_direction`：H 扰动方向。
- `max_consecutive_steps`：最大连续攻击步数。

#### 6.4.3 clean / attack 性能对比字段

- `clean_mean_mse`：clean baseline 下的平均 Sum MSE。由 clean 动作真实推进环境得到。
- `attack_mean_mse`：攻击评估下的平均 Sum MSE。agent 用 `attacked_state` 选动作，环境仍按真实状态转移。
- `mse_degradation`：MSE 绝对退化量，计算方式为 `attack_mean_mse - clean_mean_mse`。正数表示攻击后 MSE 变大。
- `mse_degradation_ratio`：MSE 相对退化比例，计算方式为 `mse_degradation / clean_mean_mse`。
- `clean_mean_sum_aoi`：clean baseline 下的平均 SumAoI。
- `attack_mean_sum_aoi`：攻击评估下的平均 SumAoI。
- `sum_aoi_degradation`：SumAoI 绝对退化量，计算方式为 `attack_mean_sum_aoi - clean_mean_sum_aoi`。
- `sum_aoi_degradation_ratio`：SumAoI 相对退化比例，计算方式为 `sum_aoi_degradation / clean_mean_sum_aoi`。

#### 6.4.4 攻击行为统计字段

- `attack_step_ratio`：实际发生扰动的 step 比例，计算方式为 `attack_step_count / total_steps`。
- `action_flip_ratio`：动作翻转比例，计算方式为 `action_flip_count / total_steps`。DQN 比较 `action_id`，DDPG 比较 `assignment tuple`。
- `avg_aoi_l0_per_step`：每个 step 平均被扰动的 AoI 维度数。
- `avg_h_l0_per_step`：每个 step 平均被扰动的 H 维度数。
- `avg_total_l0_per_step`：每个 step 平均被扰动的总维度数。
- `avg_aoi_l1_per_attack`：每个实际攻击步的 AoI 扰动 L1 平均强度。
- `avg_h_l1_per_attack`：每个实际攻击步的 H 扰动 L1 平均强度。
- `max_aoi_linf`：整个评估过程中 AoI 单维最大实际扰动幅值，应不超过 `aoi_delta`。
- `max_h_linf`：整个评估过程中 H 单维最大实际扰动幅值，应不超过 `h_delta`。
- `constraint_violation_count`：约束违规次数。正常情况下应为 `0`；若大于 `0`，终端会打印 warning。

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

