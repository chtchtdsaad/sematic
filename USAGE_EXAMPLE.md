# Usage Example

本文件给出该仓库的最小可运行示例与常用命令模板。

## 1. 环境准备

```bash
# 进入项目目录
cd C:\Users\28681\Desktop\Kalman\sematic

# 建议：创建并激活虚拟环境（可选）
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖（按你的环境调整）
pip install numpy scipy matplotlib torch
```

## 2. 训练模式（`--mode train`）

### 2.1 DQN 训练（保存全部结果）

```bash
python main.py --algo DQN --mode train --seed 42 --episodes 300 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1
```

### 2.2 DDPG 训练（不保存任何文件，快速测试）

```bash
python main.py --algo DDPG --mode train --seed 42 --episodes 5 --device cpu --save-checkpoints 0 --save-history 0 --save-plots 0
```

## 3. 评估模式（`--mode eval`）

> 评估时需要 checkpoint，可用 `--model-path` 指定。

```bash
python main.py --algo DQN --mode eval --seed 42 --device cpu --model-path results\checkpoints\dqn_seed42_best.pt --eval-episodes 10 --save-plots 1
```

如果不想保存评估曲线：

```bash
python main.py --algo DQN --mode eval --seed 42 --device cpu --model-path results\checkpoints\dqn_seed42_best.pt --eval-episodes 10 --save-plots 0
```

## 4. DQN/DDPG 同图对比说明

当满足以下条件时，程序会自动生成 DQN 与 DDPG 同图对比曲线：

- `N`、`M` 相同（来自同一 `config.py`）
- `seed` 相同
- `episodes` 相同
- `--save-history 1`
- `--save-plots 1`

对应输出：

- `results/compare_DQN_DDPG_seed{seed}_ep{episodes}_mse.png`
- `results/compare_DQN_DDPG_seed{seed}_ep{episodes}_sumaoi.png`

## 5. 常用参数说明

- `--algo {DQN,DDPG}`: 选择算法
- `--mode {train,eval}`: 选择训练或评估
- `--seed`: 随机种子
- `--episodes`: 训练轮数（`train` 模式）
- `--eval-episodes`: 评估轮数（`eval` 模式）
- `--device {cpu,cuda}`: 强制设备
- `--model-path`: 评估时 checkpoint 路径
- `--save-checkpoints {0,1}`: 是否保存 checkpoint
- `--save-history {0,1}`: 是否保存历史 `npz`（用于跨算法对比）
- `--save-plots {0,1}`: 是否保存曲线图片

## 6. 快速回归命令（推荐）

### 6.1 1 回合快速跑通（不落盘）

```bash
python main.py --algo DQN --mode train --seed 1 --episodes 1 --device cpu --save-checkpoints 0 --save-history 0 --save-plots 0
```

### 6.2 DQN 与 DDPG 同配置各跑一次并自动对比

```bash
python main.py --algo DQN  --mode train --seed 7 --episodes 100 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1
python main.py --algo DDPG --mode train --seed 7 --episodes 100 --device cpu --save-checkpoints 1 --save-history 1 --save-plots 1
```

第二条命令结束后，若历史文件匹配，会自动生成对比图。
