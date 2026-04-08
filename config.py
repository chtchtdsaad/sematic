"""
全局配置文件（config.py）
========================
本文件只放“不会在训练中频繁变化”的常量，方便其他模块统一读取。

说明：
- 所有数值都按论文实验设定和当前项目约束给出。
- 训练入口（main.py）可覆盖其中部分参数（如 episodes）。
"""

import numpy as np

# -----------------------------
# 系统规模（论文基础设定）
# -----------------------------
N = 6  # 传感器数量
M = 3  # 信道数量

# 预配置场景（用于 CLI --scenario）
# - base: 论文当前基础规模
# - s10x5: 中等规模扩展
# - s20x10: 大规模扩展（DDPG 推荐）
SCENARIO_PRESETS = {
    "base": {"n": N, "m": M},
    "s10x5": {"n": 10, "m": 5},
    "s20x10": {"n": 20, "m": 10},
}

# -----------------------------
# 状态/观测维度
# -----------------------------
ln = 2  # 单个传感器状态维度（x_n,t 维度）
en = 1  # 单个传感器观测维度（y_n,t 维度）

# -----------------------------
# 量化信道对应的丢包率（5档）
# 索引 0..4 对应离散信道状态 h_t
# -----------------------------
PACKET_LOSS_LEVELS = np.array([0.2, 0.15, 0.1, 0.05, 0.01], dtype=np.float64)
NUM_CHANNEL_STATES = int(PACKET_LOSS_LEVELS.size)

# -----------------------------
# 环境与物理过程
# -----------------------------
EPISODE_LENGTH = 500   # 每个 episode 的时长 T
MAX_AOI = 30           # AoI 上限（用于控制状态爆炸）

# Rayleigh 衰落尺度参数范围（每个 sensor-channel 对独立采样一个固定 scale）
RAYLEIGH_SCALE_MIN = 0.5
RAYLEIGH_SCALE_MAX = 2.0

# A 矩阵谱半径限制
MIN_SPECTRAL_RADIUS = 1.0
MAX_SPECTRAL_RADIUS = 1.4

# -----------------------------
# 可复现性
# -----------------------------
DEFAULT_SEED = 42

# -----------------------------
# 通用 RL 超参数
# -----------------------------
NUM_EPISODES = 300
GAMMA = 0.95
BATCH_SIZE = 128
REPLAY_BUFFER_CAPACITY = 20000

# -----------------------------
# DQN 超参数
# -----------------------------
DQN_HIDDEN_DIMS = (256, 256)
DQN_LR = 1e-3
DQN_TARGET_UPDATE_FREQ = 100  # 每 100 个训练 step 硬更新目标网络
EPSILON_START = 1.0
EPSILON_DECAY = 0.98
EPSILON_MIN = 0.01

# -----------------------------
# DDPG 超参数
# -----------------------------
DDPG_ACTOR_HIDDEN_DIMS = (512, 512)
DDPG_CRITIC_HIDDEN_DIMS = (512, 512)
DDPG_ACTOR_LR = 1e-4
DDPG_CRITIC_LR = 1e-3
DDPG_TAU = 0.005  # 目标网络软更新系数

# 300 episode 的学习率衰减目标（指数衰减终点）
DDPG_ACTOR_LR_END = 2e-5
DDPG_CRITIC_LR_END = 1e-4

# 噪声调度（约 300 episode 从 0.25 衰减到 0.03）
NOISE_STD_START = 0.25
NOISE_STD_DECAY = 0.99296
NOISE_STD_MIN = 0.03

# DDPG 稳定化
DDPG_WARMUP_STEPS = 2000
DDPG_REWARD_CLIP = 5000.0
DDPG_REWARD_SCALE = 1000.0
DDPG_GRAD_CLIP_NORM = 5.0
