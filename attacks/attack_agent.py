"""
文件作用：定义 learned meta-attacker 使用的 DDPG Actor、Critic 和训练更新逻辑。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DDPG_ACTOR_HIDDEN_DIMS, DDPG_CRITIC_HIDDEN_DIMS


class AttackActor(nn.Module):
    """
    作用:
        learned attacker 的 Actor 网络，将 raw state 映射为连续 intent action。

    输入格式:
        state: torch.Tensor[float32], shape=(B,state_dim)。

    输出格式:
        action: torch.Tensor[float32], shape=(B,action_dim)，范围 [-1,1]。

    参数含义:
        state_dim 是 AoI+H 展平后的状态维度；action_dim 与 state_dim 相同。

    核心步骤:
        1. 两层 ReLU 隐藏层提取状态特征。
        2. 输出层给出每个状态维度的扰动意图。
        3. 使用 tanh 将 intent action 限制到 [-1,1]。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        作用:
            初始化 Actor 网络层。

        输入格式:
            state_dim: int，输入状态维度。
            action_dim: int，输出 intent action 维度。

        输出格式:
            None。

        参数含义:
            state_dim/action_dim 均由 LearnedAttackEnv 提供。

        核心步骤:
            1. 创建两层隐藏层。
            2. 创建输出层。
        """
        super().__init__()
        # 第一层将 raw state 映射到隐藏空间。
        self.fc1 = nn.Linear(int(state_dim), DDPG_ACTOR_HIDDEN_DIMS[0])
        # 第二层继续提取非线性特征。
        self.fc2 = nn.Linear(DDPG_ACTOR_HIDDEN_DIMS[0], DDPG_ACTOR_HIDDEN_DIMS[1])
        # 输出层给出每个维度的连续扰动意图。
        self.out = nn.Linear(DDPG_ACTOR_HIDDEN_DIMS[1], int(action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        作用:
            前向计算连续 intent action。

        输入格式:
            state: torch.Tensor[float32], shape=(B,state_dim)。

        输出格式:
            torch.Tensor[float32], shape=(B,action_dim)，范围 [-1,1]。

        参数含义:
            state 是 raw state，不做归一化。

        核心步骤:
            1. state 经过 fc1 + ReLU。
            2. 再经过 fc2 + ReLU。
            3. 输出层后接 tanh。
        """
        # 第一层 ReLU 用于引入非线性。
        x = F.relu(self.fc1(state))
        # 第二层 ReLU 进一步组合 AoI/H 特征。
        x = F.relu(self.fc2(x))
        # tanh 保证 Actor 输出满足映射层设计范围。
        return torch.tanh(self.out(x))


class AttackCritic(nn.Module):
    """
    作用:
        learned attacker 的 Critic 网络，估计 Q(raw_state, intent_action)。

    输入格式:
        state: torch.Tensor[float32], shape=(B,state_dim)。
        action: torch.Tensor[float32], shape=(B,action_dim)。

    输出格式:
        q: torch.Tensor[float32], shape=(B,1)。

    参数含义:
        action 是 continuous intent action，不是离散 delta。

    核心步骤:
        1. 拼接 state 和 action。
        2. 两层 ReLU 隐藏层。
        3. 输出标量 Q 值。
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """
        作用:
            初始化 Critic 网络层。

        输入格式:
            state_dim: int，状态维度。
            action_dim: int，连续 intent action 维度。

        输出格式:
            None。

        参数含义:
            state_dim/action_dim 由 LearnedAttackEnv 提供。

        核心步骤:
            1. 拼接维度为 state_dim + action_dim。
            2. 创建两层隐藏层和一个 Q 输出层。
        """
        super().__init__()
        # Critic 输入是 state 和 continuous intent action 的拼接。
        input_dim = int(state_dim) + int(action_dim)
        # 第一层处理拼接后的状态动作特征。
        self.fc1 = nn.Linear(input_dim, DDPG_CRITIC_HIDDEN_DIMS[0])
        # 第二层继续拟合 Q 函数。
        self.fc2 = nn.Linear(DDPG_CRITIC_HIDDEN_DIMS[0], DDPG_CRITIC_HIDDEN_DIMS[1])
        # 输出层给出单个 Q 值。
        self.out = nn.Linear(DDPG_CRITIC_HIDDEN_DIMS[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        作用:
            前向估计 Q(raw_state, intent_action)。

        输入格式:
            state: torch.Tensor[float32], shape=(B,state_dim)。
            action: torch.Tensor[float32], shape=(B,action_dim)。

        输出格式:
            torch.Tensor[float32], shape=(B,1)。

        参数含义:
            action 是 Actor 输出或 ReplayBuffer 中存储的 continuous intent action。

        核心步骤:
            1. 在特征维拼接 state/action。
            2. 两层 ReLU。
            3. 输出 Q 标量。
        """
        # 拼接 raw state 和 continuous intent action。
        x = torch.cat([state, action], dim=1)
        # 第一层 ReLU 处理状态动作联合特征。
        x = F.relu(self.fc1(x))
        # 第二层 ReLU 提高 Q 函数拟合能力。
        x = F.relu(self.fc2(x))
        # 输出标量 Q 值。
        return self.out(x)


class AttackDDPGAgent:
    """
    作用:
        learned meta-attacker 的 DDPG agent。

    输入格式:
        state_dim: int，raw state 维度。
        action_dim: int，continuous intent action 维度。
        gamma/tau/lr/grad_clip/device: DDPG 训练超参数。

    输出格式:
        AttackDDPGAgent 实例。

    参数含义:
        state_dim = action_dim = N + N*M；Actor 输出 intent action；Critic 估计 intent action 的长期攻击收益。

    核心步骤:
        1. 初始化 Actor/Critic 和 target 网络。
        2. 初始化 Adam 优化器。
        3. select_action 负责推理和探索噪声。
        4. update 负责 Critic、Actor 和 target 软更新。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.95,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        grad_clip_norm: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        """
        作用:
            初始化 learned attacker DDPG 的网络、目标网络和优化器。

        输入格式:
            state_dim: int。
            action_dim: int。
            gamma: float。
            tau: float。
            actor_lr: float。
            critic_lr: float。
            grad_clip_norm: float。
            device: str | torch.device | None。

        输出格式:
            None。

        参数含义:
            gamma 控制长期收益折扣；tau 控制 target 网络软更新；grad_clip_norm 控制梯度裁剪。

        核心步骤:
            1. 解析设备。
            2. 构造在线网络和目标网络。
            3. 同步目标网络初始权重。
            4. 构造优化器。
        """
        # 保存维度，供 select_action 校验和采样噪声使用。
        self.state_dim = int(state_dim)
        # 保存 action 维度，等于 continuous intent action 长度。
        self.action_dim = int(action_dim)
        # 保存折扣因子。
        self.gamma = float(gamma)
        # 保存 target 网络软更新系数。
        self.tau = float(tau)
        # 保存梯度裁剪阈值。
        self.grad_clip_norm = float(grad_clip_norm)
        # 未指定设备时自动选择可用设备。
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # CUDA 不可用但用户指定 cuda 时明确报错。
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device is requested but not available.")

        # 在线 Actor 负责产生当前策略动作。
        self.actor = AttackActor(self.state_dim, self.action_dim).to(self.device)
        # 目标 Actor 用于构造 TD target。
        self.actor_target = AttackActor(self.state_dim, self.action_dim).to(self.device)
        # 初始时目标 Actor 与在线 Actor 完全一致。
        self.actor_target.load_state_dict(self.actor.state_dict())
        # 在线 Critic 负责拟合 Q。
        self.critic = AttackCritic(self.state_dim, self.action_dim).to(self.device)
        # 目标 Critic 用于构造 TD target。
        self.critic_target = AttackCritic(self.state_dim, self.action_dim).to(self.device)
        # 初始时目标 Critic 与在线 Critic 完全一致。
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Actor 优化器只更新在线 Actor。
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        # Critic 优化器只更新在线 Critic。
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=float(critic_lr))

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        """
        作用:
            根据 raw state 输出 continuous intent action，并可叠加高斯探索噪声。

        输入格式:
            state: np.ndarray[float32], shape=(state_dim,)。
            noise_std: float，探索噪声标准差。

        输出格式:
            action: np.ndarray[float32], shape=(action_dim,)，范围 [-1,1]。

        参数含义:
            state 不做归一化；noise_std=0 表示确定性评估。

        核心步骤:
            1. 将 raw state 转为 torch tensor。
            2. Actor 前向输出 action。
            3. 按需叠加高斯噪声。
            4. clip 到 [-1,1]。
        """
        # 将状态统一为一维 float32。
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        # 维度错误时立即报错，避免网络输入错位。
        if state_arr.shape[0] != self.state_dim:
            raise ValueError(f"state shape must be ({self.state_dim},), got {state_arr.shape}.")
        # 转为 batch tensor，shape=(1,state_dim)。
        state_t = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        # 推理阶段不需要梯度。
        with torch.no_grad():
            # Actor 输出 shape=(action_dim,)。
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        # 训练阶段可加入高斯噪声扩大探索。
        if float(noise_std) > 0.0:
            # 噪声 shape 与 action 一致。
            noise = np.random.normal(0.0, float(noise_std), size=self.action_dim).astype(np.float32)
            # 将噪声加入动作。
            action = action + noise
        # 映射层要求 intent action 位于 [-1,1]。
        action = np.clip(action, -1.0, 1.0)
        # 返回 float32，方便 ReplayBuffer 存储。
        return action.astype(np.float32)

    def update(self, replay_buffer, batch_size: int) -> tuple[float, float]:
        """
        作用:
            执行一次 learned attacker DDPG 参数更新。

        输入格式:
            replay_buffer: agent.ReplayBuffer，存储 raw state、continuous intent action、clipped reward。
            batch_size: int。

        输出格式:
            tuple[float,float] = (critic_loss, actor_loss)。

        参数含义:
            replay_buffer 中的 action 必须是 Actor 输出的 continuous intent action，不是离散 delta。

        核心步骤:
            1. 从 ReplayBuffer 采样 batch。
            2. 用 target 网络构造 TD target。
            3. 更新 Critic。
            4. 更新 Actor。
            5. 软更新 target 网络。
        """
        # 从回放池采样 transition。
        states, actions, rewards, next_states, dones = replay_buffer.sample(int(batch_size))
        # raw state batch 转 tensor。
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        # continuous intent action batch 转 tensor。
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        # clipped reward 转列向量。
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        # next raw state batch 转 tensor。
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        # done 标志转列向量。
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # target 计算不需要梯度。
        with torch.no_grad():
            # 目标 Actor 给出下一状态的 intent action。
            next_actions_t = self.actor_target(next_states_t)
            # 目标 Critic 估计下一状态动作价值。
            q_next_t = self.critic_target(next_states_t, next_actions_t)
            # DDPG TD target。
            q_target_t = rewards_t + self.gamma * (1.0 - dones_t) * q_next_t

        # 当前 Critic 对真实采样 action 的估计。
        q_current_t = self.critic(states_t, actions_t)
        # Critic 用 MSE 拟合 TD target。
        critic_loss = F.mse_loss(q_current_t, q_target_t)
        # 清空旧梯度。
        self.critic_optimizer.zero_grad()
        # 反向传播 Critic 损失。
        critic_loss.backward()
        # 裁剪 Critic 梯度，避免 reward 尺度导致爆炸。
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        # 更新 Critic 参数。
        self.critic_optimizer.step()

        # Actor 目标是最大化 Critic 估计 Q，因此损失取负均值。
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        # 清空 Actor 旧梯度。
        self.actor_optimizer.zero_grad()
        # 反向传播 Actor 损失。
        actor_loss.backward()
        # 裁剪 Actor 梯度。
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        # 更新 Actor 参数。
        self.actor_optimizer.step()

        # 更新目标 Actor。
        self.soft_update(self.actor, self.actor_target)
        # 更新目标 Critic。
        self.soft_update(self.critic, self.critic_target)
        # 返回 Python float，便于 history/report 序列化。
        return float(critic_loss.item()), float(actor_loss.item())

    def soft_update(self, source_net: nn.Module, target_net: nn.Module) -> None:
        """
        作用:
            执行目标网络软更新：target <- tau*source + (1-tau)*target。

        输入格式:
            source_net: nn.Module，在线网络。
            target_net: nn.Module，目标网络。

        输出格式:
            None。

        参数含义:
            source_net 提供新参数；target_net 被原地更新。

        核心步骤:
            1. 遍历 target/source 参数对。
            2. 对每个参数做指数滑动更新。
        """
        # zip 保证逐层参数一一对应。
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            # 原地写入软更新结果，避免重新创建网络。
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
