from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (
    BATCH_SIZE,
    DDPG_ACTOR_HIDDEN_DIMS,
    DDPG_ACTOR_LR,
    DDPG_CRITIC_HIDDEN_DIMS,
    DDPG_CRITIC_LR,
    DDPG_TAU,
    DQN_HIDDEN_DIMS,
    DQN_LR,
    DQN_TARGET_UPDATE_FREQ,
    GAMMA,
)


class ReplayBuffer:
    """
    作用:
        经验回放池，统一支持 DQN 与 DDPG 的样本存储。

    参数:
        capacity: 回放池容量，默认 20000。

    返回:
        None。

    核心步骤:
        1. 使用固定长度 deque 保存转移元组。
        2. push 时写入 (state, action, reward, next_state, done)。
        3. sample 时随机采样并打包成数组。
    """

    def __init__(self, capacity: int = 20000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        作用:
            向回放池压入一条样本。

        参数:
            state: 当前状态。
            action: DQN 使用整数 action_id；DDPG 使用连续动作向量 v。
            reward: 即时奖励。
            next_state: 下一状态。
            done: 终止标志。

        返回:
            None。

        核心步骤:
            1. 将状态数组标准化为 float32。
            2. 动作保持原语义（离散 int 或连续向量）。
            3. 追加到循环队列。
        """
        s = np.asarray(state, dtype=np.float32)
        s_next = np.asarray(next_state, dtype=np.float32)
        if isinstance(action, (int, np.integer)):
            a: int | np.ndarray = int(action)
        else:
            a = np.asarray(action, dtype=np.float32)
        self.buffer.append((s, a, float(reward), s_next, bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        作用:
            随机采样一个 batch。

        参数:
            batch_size: 批量大小。

        返回:
            tuple[np.ndarray, ...]:
                (states, actions, rewards, next_states, dones)。

        核心步骤:
            1. 从回放池均匀随机抽样。
            2. 分别堆叠状态、奖励、终止标志。
            3. 动作根据类型打包为 int 向量或 float 矩阵。
        """
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_arr = np.stack(states).astype(np.float32)
        next_states_arr = np.stack(next_states).astype(np.float32)
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=np.float32)

        if isinstance(actions[0], (int, np.integer)):
            actions_arr = np.asarray(actions, dtype=np.int64)
        else:
            actions_arr = np.stack(actions).astype(np.float32)

        return states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr

    def __len__(self) -> int:
        """
        作用:
            返回当前回放池样本数量。
        """
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    作用:
        DQN 的 Q 网络，隐藏层为 [256, 256]。

    参数:
        state_dim: 状态维度。
        num_actions: 离散动作空间大小。

    返回:
        None。

    核心步骤:
        1. 线性层 + ReLU 两层特征提取。
        2. 输出层直接输出各动作 Q 值。
    """

    def __init__(self, state_dim: int, num_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, DQN_HIDDEN_DIMS[0])
        self.fc2 = nn.Linear(DQN_HIDDEN_DIMS[0], DQN_HIDDEN_DIMS[1])
        self.out = nn.Linear(DQN_HIDDEN_DIMS[1], num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            前向计算 Q 值向量。

        参数:
            x: 状态张量。

        返回:
            torch.Tensor: 每个动作的 Q 值。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    """
    作用:
        基线 DQN 智能体，采用 epsilon-greedy 与目标网络硬更新。
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = GAMMA,
        lr: float = DQN_LR,
        target_update_freq: int = DQN_TARGET_UPDATE_FREQ,
    ) -> None:
        """
        作用:
            初始化 DQN 网络、优化器和目标网络。

        参数:
            state_dim: 状态维度。
            num_actions: 动作数量。
            gamma: 折扣因子。
            lr: 学习率。
            target_update_freq: 目标网络硬更新步频（默认 100）。

        返回:
            None。
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_q_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        作用:
            基于 epsilon-greedy 选择离散动作。

        参数:
            state: 当前状态。
            epsilon: 探索概率。

        返回:
            int: 离散动作 ID。

        核心步骤:
            1. 以 epsilon 概率随机探索。
            2. 否则选择 Q 最大的动作。
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, buffer: ReplayBuffer, batch_size: int = BATCH_SIZE) -> float:
        """
        作用:
            执行一次 DQN 训练更新（TD Error + MSE）。

        参数:
            buffer: 回放池。
            batch_size: 批量大小。

        返回:
            float: 当前损失值。

        核心步骤:
            1. 采样 batch 并构造张量。
            2. 计算当前 Q(s,a) 与目标 Q。
            3. 反向传播优化 Q 网络。
            4. 每 100 step 触发一次目标网络硬更新。
        """
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_current = self.q_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            q_next_max = self.target_q_net(next_states_t).max(dim=1, keepdim=True)[0]
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * q_next_max

        loss = F.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


class ActorNet(nn.Module):
    """
    作用:
        DDPG 的 Actor 网络，隐藏层 [512, 512]，输出经 Tanh 压缩到 [-1, 1]。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, DDPG_ACTOR_HIDDEN_DIMS[0])
        self.fc2 = nn.Linear(DDPG_ACTOR_HIDDEN_DIMS[0], DDPG_ACTOR_HIDDEN_DIMS[1])
        self.out = nn.Linear(DDPG_ACTOR_HIDDEN_DIMS[1], action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            前向输出连续动作向量。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))


class CriticNet(nn.Module):
    """
    作用:
        DDPG 的 Critic 网络，输入为 [state, action] 拼接向量，输出标量 Q 值。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        input_dim = state_dim + action_dim
        self.fc1 = nn.Linear(input_dim, DDPG_CRITIC_HIDDEN_DIMS[0])
        self.fc2 = nn.Linear(DDPG_CRITIC_HIDDEN_DIMS[0], DDPG_CRITIC_HIDDEN_DIMS[1])
        self.out = nn.Linear(DDPG_CRITIC_HIDDEN_DIMS[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        作用:
            前向输出 Q(s,a)。
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DDPGAgent:
    """
    作用:
        基线 DDPG 智能体，连续动作控制并每步软更新目标网络。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = GAMMA,
        tau: float = DDPG_TAU,
        actor_lr: float = DDPG_ACTOR_LR,
        critic_lr: float = DDPG_CRITIC_LR,
    ) -> None:
        """
        作用:
            初始化 Actor/Critic 及其目标网络。

        参数:
            state_dim: 状态维度。
            action_dim: 连续动作维度（N）。
            gamma: 折扣因子。
            tau: 软更新系数（默认 0.005）。
            actor_lr: Actor 学习率。
            critic_lr: Critic 学习率。

        返回:
            None。
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNet(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNet(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNet(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray, noise_std: float) -> np.ndarray:
        """
        作用:
            输出连续动作并叠加高斯噪声后裁剪到 [-1,1]。

        参数:
            state: 当前状态。
            noise_std: 噪声标准差。

        返回:
            np.ndarray: 连续动作向量。

        核心步骤:
            1. Actor 前向得到基础动作。
            2. 加入 Gaussian Noise。
            3. clip 到 [-1,1]。
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        noise = np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)
        action_noisy = np.clip(action + noise, -1.0, 1.0)
        return action_noisy.astype(np.float32)

    def train_step(self, buffer: ReplayBuffer, batch_size: int = BATCH_SIZE) -> tuple[float, float]:
        """
        作用:
            执行一次 DDPG 训练（Critic 更新 + Actor 更新 + 软更新）。

        参数:
            buffer: 回放池。
            batch_size: 批量大小。

        返回:
            tuple[float, float]: (critic_loss, actor_loss)。

        核心步骤:
            1. 更新 Critic：最小化 TD 目标误差。
            2. 更新 Actor：最大化 Q(s, Actor(s))。
            3. 每步执行一次目标网络软更新。
        """
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions_t = self.actor_target(next_states_t)
            q_next_t = self.critic_target(next_states_t, next_actions_t)
            q_target_t = rewards_t + self.gamma * (1.0 - dones_t) * q_next_t

        q_current_t = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q_current_t, q_target_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)

        return float(critic_loss.item()), float(actor_loss.item())

    @staticmethod
    def _soft_update(source_net: nn.Module, target_net: nn.Module, tau: float) -> None:
        """
        作用:
            执行软更新：target <- tau * source + (1 - tau) * target。

        参数:
            source_net: 在线网络。
            target_net: 目标网络。
            tau: 软更新系数。

        返回:
            None。
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
