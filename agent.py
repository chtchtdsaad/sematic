"""
智能体模块（agent.py）
======================
包含：
- ReplayBuffer（DQN/DDPG 共用）
- DQNAgent（离散动作）
- DDPGAgent（连续动作）
"""

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
    DDPG_GRAD_CLIP_NORM,
    DDPG_TAU,
    DQN_HIDDEN_DIMS,
    DQN_LR,
    DQN_TARGET_UPDATE_FREQ,
    GAMMA,
)


class ReplayBuffer:
    """
    作用:
        经验回放池，统一存储 transition=(state, action, reward, next_state, done)。

    输入格式:
        capacity: int，缓存容量。

    输出格式:
        None

    核心步骤:
        1. 使用固定长度 deque 存储样本。
        2. push 写入新样本，超容量时自动覆盖最老样本。
        3. sample 随机采样 batch 并按字段打包。
    """

    def __init__(self, capacity: int = 20000) -> None:
        # 固定容量队列
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
            压入一条经验样本。

        输入格式:
            state: np.ndarray[float], shape=(state_dim,)
            action: int（DQN）或 np.ndarray[float], shape=(action_dim,)（DDPG）
            reward: float
            next_state: np.ndarray[float], shape=(state_dim,)
            done: bool

        输出格式:
            None

        核心步骤:
            1. 统一状态 dtype=float32。
            2. 动作保持算法语义（int 或 float 向量）。
            3. 将 transition 追加到队列。
        """
        # 状态统一为 float32，shape=(state_dim,)
        s = np.asarray(state, dtype=np.float32)
        s_next = np.asarray(next_state, dtype=np.float32)

        # 动作按类型处理
        if isinstance(action, (int, np.integer)):
            a: int | np.ndarray = int(action)
        else:
            a = np.asarray(action, dtype=np.float32)

        # 入队（reward/done 规范化）
        self.buffer.append((s, a, float(reward), s_next, bool(done)))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        作用:
            从回放池随机采样一个 batch。

        输入格式:
            batch_size: int

        输出格式:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - states: np.ndarray[float32], shape=(B,state_dim)
            - actions: np.ndarray[int64], shape=(B,) 或 np.ndarray[float32], shape=(B,action_dim)
            - rewards: np.ndarray[float32], shape=(B,)
            - next_states: np.ndarray[float32], shape=(B,state_dim)
            - dones: np.ndarray[float32], shape=(B,)

        核心步骤:
            1. 随机采样 transition 列表。
            2. 拆分并堆叠为数组。
            3. 动作根据首元素类型决定打包 dtype/shape。
        """
        # 随机采样 B 条样本
        transitions = random.sample(self.buffer, batch_size)

        # 按字段解包
        states, actions, rewards, next_states, dones = zip(*transitions)

        # 状态与下一状态堆叠
        states_arr = np.stack(states).astype(np.float32)
        next_states_arr = np.stack(next_states).astype(np.float32)

        # 奖励和 done
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=np.float32)

        # 动作按类型处理
        if isinstance(actions[0], (int, np.integer)):
            actions_arr = np.asarray(actions, dtype=np.int64)
        else:
            actions_arr = np.stack(actions).astype(np.float32)

        return states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr

    def __len__(self) -> int:
        """
        作用:
            返回当前缓存中的样本数。

        输入格式:
            无

        输出格式:
            int

        核心步骤:
            1. 调用 deque 的长度接口。
        """
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    作用:
        DQN 的 Q 网络（MLP: 256-256）。

    输入格式:
        x: torch.Tensor[float32], shape=(B,state_dim)

    输出格式:
        torch.Tensor[float32], shape=(B,num_actions)

    核心步骤:
        1. 两层 ReLU 隐藏层。
        2. 输出层给出每个离散动作 Q 值。
    """

    def __init__(self, state_dim: int, num_actions: int) -> None:
        super().__init__()

        # 第一层: state_dim -> 256
        self.fc1 = nn.Linear(state_dim, DQN_HIDDEN_DIMS[0])

        # 第二层: 256 -> 256
        self.fc2 = nn.Linear(DQN_HIDDEN_DIMS[0], DQN_HIDDEN_DIMS[1])

        # 输出层: 256 -> num_actions
        self.out = nn.Linear(DQN_HIDDEN_DIMS[1], num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            前向计算 Q 值向量。

        输入格式:
            x: torch.Tensor[float32], shape=(B,state_dim)

        输出格式:
            torch.Tensor[float32], shape=(B,num_actions)

        核心步骤:
            1. x 经过 fc1 + ReLU。
            2. 再经过 fc2 + ReLU。
            3. 经 out 输出 Q 值。
        """
        # 隐藏层 1
        x = F.relu(self.fc1(x))

        # 隐藏层 2
        x = F.relu(self.fc2(x))

        # 输出层
        return self.out(x)


class DQNAgent:
    """
    作用:
        基线 DQN 智能体（epsilon-greedy + 目标网络硬更新）。
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        gamma: float = GAMMA,
        lr: float = DQN_LR,
        target_update_freq: int = DQN_TARGET_UPDATE_FREQ,
        device: str | torch.device | None = None,
    ) -> None:
        """
        作用:
            初始化 DQN 相关网络与优化器。

        输入格式:
            state_dim: int
            num_actions: int
            gamma: float
            lr: float
            target_update_freq: int
            device: str | torch.device | None

        输出格式:
            None

        核心步骤:
            1. 解析并校验 device。
            2. 创建在线 Q 网络与目标网络。
            3. 初始化 Adam 优化器。
        """
        self.num_actions = int(num_actions)
        self.gamma = float(gamma)
        self.target_update_freq = int(target_update_freq)
        self.update_steps = 0

        # 设备解析
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA device is requested but not available.")

        # 在线 Q 网络
        self.q_net = QNetwork(state_dim, self.num_actions).to(self.device)

        # 目标 Q 网络（初始权重与在线网络一致）
        self.target_q_net = QNetwork(state_dim, self.num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        作用:
            epsilon-greedy 动作选择。

        输入格式:
            state: np.ndarray[float], shape=(state_dim,)
            epsilon: float

        输出格式:
            int（离散动作 ID）

        核心步骤:
            1. 以 epsilon 概率随机探索。
            2. 否则用 Q 网络贪心选择 argmax 动作。
        """
        # 随机探索
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        # 状态转 tensor，shape=(1,state_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 前向推理（不求梯度）
        with torch.no_grad():
            q_values = self.q_net(state_t)

        # 返回最大 Q 对应动作
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, buffer: ReplayBuffer, batch_size: int = BATCH_SIZE) -> float:
        """
        作用:
            执行一次 DQN 参数更新。

        输入格式:
            buffer: ReplayBuffer
            batch_size: int

        输出格式:
            float（当前 MSE 损失）

        核心步骤:
            1. 从回放池采样 batch。
            2. 计算当前 Q(s,a) 与目标 Q。
            3. 最小化 TD MSE 损失。
            4. 每 target_update_freq 步硬更新目标网络。
        """
        # 采样 batch
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # 转为 tensor
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 当前 Q(s,a)
        q_current = self.q_net(states_t).gather(1, actions_t)

        # 目标 Q
        with torch.no_grad():
            q_next_max = self.target_q_net(next_states_t).max(dim=1, keepdim=True)[0]
            q_target = rewards_t + self.gamma * (1.0 - dones_t) * q_next_max

        # MSE 损失
        loss = F.mse_loss(q_current, q_target)

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新计数
        self.update_steps += 1

        # 硬更新目标网络
        if self.update_steps % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


class ActorNet(nn.Module):
    """
    作用:
        DDPG Actor 网络（MLP: 512-512，输出 Tanh 到 [-1,1]）。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        # 第一层: state_dim -> 512
        self.fc1 = nn.Linear(state_dim, DDPG_ACTOR_HIDDEN_DIMS[0])

        # 第二层: 512 -> 512
        self.fc2 = nn.Linear(DDPG_ACTOR_HIDDEN_DIMS[0], DDPG_ACTOR_HIDDEN_DIMS[1])

        # 输出层: 512 -> action_dim
        self.out = nn.Linear(DDPG_ACTOR_HIDDEN_DIMS[1], action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        作用:
            Actor 前向输出连续动作向量。

        输入格式:
            x: torch.Tensor[float32], shape=(B,state_dim)

        输出格式:
            torch.Tensor[float32], shape=(B,action_dim)，范围[-1,1]

        核心步骤:
            1. 两层 ReLU 隐藏层。
            2. 输出层后接 Tanh。
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x))


class CriticNet(nn.Module):
    """
    作用:
        DDPG Critic 网络（输入 state+action，输出 Q 标量）。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        # 拼接输入维度
        input_dim = state_dim + action_dim

        # 第一层: (state_dim+action_dim) -> 512
        self.fc1 = nn.Linear(input_dim, DDPG_CRITIC_HIDDEN_DIMS[0])

        # 第二层: 512 -> 512
        self.fc2 = nn.Linear(DDPG_CRITIC_HIDDEN_DIMS[0], DDPG_CRITIC_HIDDEN_DIMS[1])

        # 输出层: 512 -> 1
        self.out = nn.Linear(DDPG_CRITIC_HIDDEN_DIMS[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        作用:
            Critic 前向输出 Q(s,a)。

        输入格式:
            state: torch.Tensor[float32], shape=(B,state_dim)
            action: torch.Tensor[float32], shape=(B,action_dim)

        输出格式:
            torch.Tensor[float32], shape=(B,1)

        核心步骤:
            1. 拼接 state/action。
            2. 两层 ReLU 隐藏层。
            3. 输出标量 Q。
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DDPGAgent:
    """
    作用:
        基线 DDPG 智能体（Actor/Critic + 目标网络软更新）。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = GAMMA,
        tau: float = DDPG_TAU,
        actor_lr: float = DDPG_ACTOR_LR,
        critic_lr: float = DDPG_CRITIC_LR,
        actor_lr_decay_gamma: float = 1.0,
        critic_lr_decay_gamma: float = 1.0,
        grad_clip_norm: float = DDPG_GRAD_CLIP_NORM,
        device: str | torch.device | None = None,
    ) -> None:
        """
        作用:
            初始化 DDPG 网络、目标网络、优化器和学习率调度器。

        输入格式:
            state_dim: int
            action_dim: int
            gamma: float
            tau: float
            actor_lr: float
            critic_lr: float
            actor_lr_decay_gamma: float
            critic_lr_decay_gamma: float
            grad_clip_norm: float
            device: str | torch.device | None

        输出格式:
            None

        核心步骤:
            1. 解析并校验 device。
            2. 初始化 Actor/Critic 及对应 target 网络。
            3. 初始化优化器与学习率调度器。
        """
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.grad_clip_norm = float(grad_clip_norm)

        # 设备解析
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA device is requested but not available.")

        # 在线 Actor
        self.actor = ActorNet(state_dim, self.action_dim).to(self.device)

        # 目标 Actor
        self.actor_target = ActorNet(state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # 在线 Critic
        self.critic = CriticNet(state_dim, self.action_dim).to(self.device)

        # 目标 Critic
        self.critic_target = CriticNet(state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
            self.actor_optimizer,
            gamma=actor_lr_decay_gamma,
        )
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
            self.critic_optimizer,
            gamma=critic_lr_decay_gamma,
        )

    def select_action(self, state: np.ndarray, noise_std: float) -> np.ndarray:
        """
        作用:
            Actor 输出连续动作并叠加高斯噪声。

        输入格式:
            state: np.ndarray[float], shape=(state_dim,)
            noise_std: float

        输出格式:
            np.ndarray[float32], shape=(action_dim,)，范围[-1,1]

        核心步骤:
            1. 前向得到确定性动作。
            2. 加入 N(0, noise_std^2) 噪声。
            3. clip 到 [-1,1]。
        """
        # 状态转 tensor，shape=(1,state_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 前向推理
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()

        # 采样高斯噪声，shape=(action_dim,)
        noise = np.random.normal(0.0, noise_std, size=self.action_dim).astype(np.float32)

        # 叠加噪声并截断
        action_noisy = np.clip(action + noise, -1.0, 1.0)

        return action_noisy.astype(np.float32)

    def train_step(self, buffer: ReplayBuffer, batch_size: int = BATCH_SIZE) -> tuple[float, float]:
        """
        作用:
            执行一次 DDPG 更新（Critic -> Actor -> Soft Update）。

        输入格式:
            buffer: ReplayBuffer
            batch_size: int

        输出格式:
            tuple[float,float] = (critic_loss, actor_loss)

        核心步骤:
            1. 采样 batch 并构建目标 Q。
            2. 更新 Critic（最小化 TD MSE）。
            3. 更新 Actor（最大化 Critic 估计 Q）。
            4. 软更新两个目标网络。
        """
        # 采样 batch
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        # 转 tensor
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # -------- 1) Critic 更新 --------
        with torch.no_grad():
            # 目标网络产生下一动作
            next_actions_t = self.actor_target(next_states_t)

            # 目标 Q 值
            q_next_t = self.critic_target(next_states_t, next_actions_t)
            q_target_t = rewards_t + self.gamma * (1.0 - dones_t) * q_next_t

        # 当前 Q 值
        q_current_t = self.critic(states_t, actions_t)

        # Critic MSE 损失
        critic_loss = F.mse_loss(q_current_t, q_target_t)

        # Critic 反向
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # -------- 2) Actor 更新 --------
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        # -------- 3) 目标网络软更新 --------
        self._soft_update(self.actor, self.actor_target, self.tau)
        self._soft_update(self.critic, self.critic_target, self.tau)

        return float(critic_loss.item()), float(actor_loss.item())

    @staticmethod
    def _soft_update(source_net: nn.Module, target_net: nn.Module, tau: float) -> None:
        """
        作用:
            软更新目标网络：target <- tau*source + (1-tau)*target。

        输入格式:
            source_net: nn.Module
            target_net: nn.Module
            tau: float

        输出格式:
            None

        核心步骤:
            1. 遍历参数对。
            2. 逐参数执行加权更新。
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def step_lr_decay(self) -> None:
        """
        作用:
            让 Actor/Critic 学习率各衰减一步。

        输入格式:
            无

        输出格式:
            None

        核心步骤:
            1. 调用 actor_scheduler.step()。
            2. 调用 critic_scheduler.step()。
        """
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_current_lrs(self) -> tuple[float, float]:
        """
        作用:
            获取当前 Actor/Critic 学习率。

        输入格式:
            无

        输出格式:
            tuple[float,float] = (actor_lr, critic_lr)

        核心步骤:
            1. 从优化器参数组读取 lr。
            2. 返回二元组。
        """
        return (
            float(self.actor_optimizer.param_groups[0]["lr"]),
            float(self.critic_optimizer.param_groups[0]["lr"]),
        )
