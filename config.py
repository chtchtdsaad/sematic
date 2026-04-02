import numpy as np

# System scale
N = 6
M = 3

# Dimensions
ln = 2
en = 1

# Quantized packet-loss levels (5 channel states)
PACKET_LOSS_LEVELS = np.array([0.2, 0.15, 0.1, 0.05, 0.01], dtype=np.float64)
NUM_CHANNEL_STATES = PACKET_LOSS_LEVELS.size

# Experiment setup
EPISODE_LENGTH = 500
MAX_AOI = 30
RAYLEIGH_SCALE_MIN = 0.5
RAYLEIGH_SCALE_MAX = 2.0
MAX_SPECTRAL_RADIUS = 1.4
MIN_SPECTRAL_RADIUS = 1.0

# Reproducibility
DEFAULT_SEED = 42

# Baseline RL setup (without SE stage)
NUM_EPISODES = 300
GAMMA = 0.95
BATCH_SIZE = 128
REPLAY_BUFFER_CAPACITY = 20000

# DQN hyperparameters
DQN_HIDDEN_DIMS = (256, 256)
DQN_LR = 1e-3
DQN_TARGET_UPDATE_FREQ = 100
EPSILON_START = 1.0
EPSILON_DECAY = 0.98
EPSILON_MIN = 0.01

# DDPG hyperparameters
DDPG_ACTOR_HIDDEN_DIMS = (512, 512)
DDPG_CRITIC_HIDDEN_DIMS = (512, 512)
DDPG_ACTOR_LR = 1e-4
DDPG_CRITIC_LR = 1e-3
# Learning-rate decay targets for 300 episodes (paper-style decaying LR setting)
DDPG_ACTOR_LR_END = 2e-5
DDPG_CRITIC_LR_END = 1e-4
DDPG_TAU = 0.005
# Noise schedule aligned to 300 episodes:
# sigma_t = max(NOISE_STD_MIN, NOISE_STD_START * NOISE_STD_DECAY^episode)
# 0.25 -> 0.03 over ~300 episodes.
NOISE_STD_START = 0.25
NOISE_STD_DECAY = 0.99296
NOISE_STD_MIN = 0.03

# DDPG stabilization
DDPG_WARMUP_STEPS = 2000
DDPG_REWARD_CLIP = 5000.0
DDPG_REWARD_SCALE = 1000.0
DDPG_GRAD_CLIP_NORM = 5.0
