"""
评估动作选择工具模块（action_utils.py）
====================================
本文件只负责统一 DQN / DDPG 在 eval_attack 阶段的动作选择接口。
"""

from __future__ import annotations

import numpy as np

from utils import map_continuous_to_assignment




def select_dqn_action(state: np.ndarray, agent) -> int:
    """
    作用:
        DQN 评估阶段根据输入 state 选择贪心 action_id。
    输入格式:
        state: np.ndarray[float32], shape=(state_dim,)
        agent: DQNAgent，需提供 select_action(state, epsilon)
    输出格式:
        int，离散动作 action_id。
    核心步骤:
        1. 将 state 转为 float32 一维数组。
        2. 调用 agent.select_action，固定 epsilon=0.0。
        3. 将返回值转为 Python int，方便后续比较与 env.step 使用。
    """
    # 评估阶段使用贪心策略，不加入探索噪声。
    action_id = agent.select_action(np.asarray(state, dtype=np.float32).reshape(-1), epsilon=0.0)
    # 兼容 np.int64 等 numpy 标量返回。
    return int(action_id)


def select_ddpg_assignment(state: np.ndarray, agent, env) -> tuple[int, ...]:
    """
    作用:
        DDPG 评估阶段根据输入 state 选择确定性 assignment。
    输入格式:
        state: np.ndarray[float32], shape=(state_dim,)
        agent: DDPGAgent，需提供 select_action(state, noise_std)
        env: 环境实例，需包含 n / m。
    输出格式:
        tuple[int, ...]，长度为 env.n，取值范围为 0..env.m。
    核心步骤:
        1. 直接使用原始 state 调用 agent.select_action。
        2. 固定 noise_std=0.0。
        3. 使用 map_continuous_to_assignment 转为离散 assignment。
    """
    # DDPG 不做额外 normalize，直接使用 eval_attack 传入的原始 state。
    state_for_agent = np.asarray(state, dtype=np.float32).reshape(-1)
    # 评估阶段不加入动作噪声。
    virtual_action = agent.select_action(state_for_agent, noise_std=0.0)
    # 连续动作映射为 assignment，作为 DDPG 后续 env.step_assignment 的输入。
    assignment = map_continuous_to_assignment(virtual_action, n=int(env.n), m=int(env.m))
    return tuple(int(x) for x in assignment)


def select_action_for_eval(algo: str, state: np.ndarray, agent, env):
    """
    作用:
        统一 DQN/DDPG 评估动作选择入口。
    输入格式:
        algo: str，"DQN" 或 "DDPG"，大小写不敏感
        state: np.ndarray[float32], shape=(state_dim,)
        agent: 智能体实例
        env: 环境实例；DQN 可为 None，DDPG 必须提供 n / m
    输出格式:
        DQN: int action_id
        DDPG: tuple[int, ...] assignment
    核心步骤:
        1. 标准化算法名。
        2. DQN 分支调用 select_dqn_action。
        3. DDPG 分支调用 select_ddpg_assignment。
    """
    # 算法名统一大写，便于 eval_attack 调用方传入 CLI 原始字符串。
    algo_upper = str(algo).upper()
    if algo_upper == "DQN":
        return select_dqn_action(state, agent)
    if algo_upper == "DDPG":
        return select_ddpg_assignment(state, agent, env)
    raise ValueError("algo must be 'DQN' or 'DDPG'")
