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

- 当前 CLI 没有 `dqn-*` 专属参数。  
- DQN 的学习率、epsilon 等参数来自 `config.py`（如 `DQN_LR`, `EPSILON_DECAY`）。
- 当前 DQN 仅支持 `--scenario base`。

## 3. 训练命令模板（严格分开）

### 3.1 仅 DQN 训练命令

#### DQN-Train-Basic（保存主要结果）

```bash
python main.py --algo DQN --scenario base --mode train --seed 42 --episodes 300 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1 --save-diagnostic-plots 1 --best-select-mode eval --best-eval-every 10 --best-eval-episodes 3 --update-interval 1
```

#### DQN-Train-Fast（快速回归，不落盘）

```bash
python main.py --algo DQN --scenario base --mode train --seed 1 --episodes 1 --device cpu --save-checkpoints 0 --save-history 0 --save-plots 0 --save-diagnostic-plots 0 --best-select-mode eval --best-eval-every 1 --best-eval-episodes 1 --update-interval 1
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

## 5. 输出文件说明（通用）

### 5.1 展示曲线（`--save-plots 1`）

- `results/result_{algo}_{scenario}_seed{seed}.png`
- `results/result_{algo}_{scenario}_seed{seed}_sumaoi.png`
- 训练展示图采用动态纵轴：按最后 10% episode 的均值构造范围，目标比例固定 35%，并对超出上界的点做硬裁剪。

### 5.2 诊断曲线（`--save-diagnostic-plots 1`）

- `results/result_{algo}_{scenario}_seed{seed}_raw_mse.png`
- `results/result_{algo}_{scenario}_seed{seed}_log_mse.png`
- 诊断图不使用动态纵轴（保持原始诊断行为）。

### 5.3 训练报告（每次 train 都会生成）

- `results/reports/report_{algo}_{scenario}_seed{seed}_ep{episodes}.json`
- 关键字段：
  - `best_metric_source`
  - `best_metric_value`
  - `train_seconds`

### 5.4 history 与跨算法对比（`--save-history 1`）

- `results/histories/{algo}_{scenario}_seed{seed}_ep{episodes}.npz`
- 若 DQN 和 DDPG 同配置 history 同时存在，且 `--save-plots 1`，会输出：
  - `results/compare_DQN_DDPG_{scenario}_seed{seed}_ep{episodes}_mse.png`
  - `results/compare_DQN_DDPG_{scenario}_seed{seed}_ep{episodes}_sumaoi.png`
- 对比图采用动态纵轴：基准取 DQN/DDPG 最后 10% 均值中的较大值，目标比例固定 35%，并进行硬裁剪。

## 6. 防混用提示（务必看）

1. 你在跑 `--algo DQN` 时，不要加 `--ddpg-*` 参数。  
2. 你在跑 `--algo DDPG` 时，可以用 `--ddpg-*` 组合调稳定性。  
3. 你在跑 `--algo DQN` 时，`--scenario` 只能是 `base`。  
4. `--update-interval` 是通用参数，两种算法都生效。  
5. `--best-select-mode eval` 是通用推荐设置，适合你“best checkpoint 为准”的流程。

