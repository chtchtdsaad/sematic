"""
文件作用：封装真实调度环境、固定 victim scheduler 和 SDEC-QM 映射层，提供 learned attacker 训练环境。
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from attacks.action_utils import select_action_for_eval
from attacks.learned_attack_config import LearnedAttackConfig


class LearnedAttackEnv:
    """
    作用:
        将原始调度环境、固定 victim agent 和连续-离散攻击映射封装为 attacker 训练环境。

    输入格式:
        base_env: SemanticSchedulingEnv，真实环境。
        victim_agent: 已加载 checkpoint 的 DQNAgent 或 DDPGAgent。
        victim_algo: str，"DQN" 或 "DDPG"。
        config: LearnedAttackConfig。

    输出格式:
        LearnedAttackEnv 实例。

    参数含义:
        base_env 负责真实状态转移；victim_agent 负责在 attacked_state 上选调度动作；
        victim_algo 控制 DQN/DDPG 分支；config 控制能量预算和扰动幅度。

    核心步骤:
        1. reset 时返回真实 state。
        2. step(intent_action) 时将连续 intent 映射为合法离散 delta。
        3. 用 attacked_state 让 victim 选择调度动作。
        4. 用 victim action 推进真实 base_env。
        5. 返回 next_state、attacker_reward、done、info。
    """

    def __init__(self, base_env, victim_agent, victim_algo: str, config: LearnedAttackConfig) -> None:
        """
        作用:
            初始化 learned attacker 训练环境包装器。

        输入格式:
            base_env: 真实 SemanticSchedulingEnv。
            victim_agent: 已加载并冻结的 victim agent。
            victim_algo: str，"DQN" 或 "DDPG"。
            config: LearnedAttackConfig。

        输出格式:
            None。

        参数含义:
            base_env 提供 reset/step；victim_agent 只用于动作选择；config 提供映射约束。

        核心步骤:
            1. 保存依赖对象。
            2. 读取 n/m/state_dim。
            3. 设置 attacker action_dim 与 state_dim 一致。
        """
        # 保存真实环境，后续所有真实转移都必须通过它完成。
        self.base_env = base_env
        # 保存固定 victim agent，只用于在 attacked_state 上选择动作。
        self.victim_agent = victim_agent
        # 算法名统一大写，避免 CLI 大小写影响分支。
        self.victim_algo = str(victim_algo).upper()
        # 保存 learned attacker 配置。
        self.config = config
        # 传感器数量来自真实环境。
        self.n = int(base_env.n)
        # 信道数量来自真实环境。
        self.m = int(base_env.m)
        # 状态维度为 AoI 部分加 H 展平部分。
        self.state_dim = int(base_env.state_dim)
        # attacker actor 输出每个状态维度的扰动意图。
        self.action_dim = self.state_dim
        # 保存最近一次真实 state，避免 step 内部读取 attacked_state。
        self._current_state: np.ndarray | None = None
        # 算法合法性检查，避免静默进入错误分支。
        if self.victim_algo not in {"DQN", "DDPG"}:
            raise ValueError("victim_algo must be 'DQN' or 'DDPG'.")

    def snapshot_runtime_state(self) -> dict[str, Any]:
        """
        作用:
            保存 learned attacker 环境和真实 base_env 的运行时状态。

        输入格式:
            无。

        输出格式:
            dict[str,Any]，包含 `_current_state`、base_env 时间、AoI、H、丢包率和 RNG 状态。

        参数含义:
            无参数；该函数用于 deterministic eval 前保存训练环境状态。

        核心步骤:
            1. 复制 LearnedAttackEnv 当前真实 state 缓存。
            2. 复制 base_env 的可变状态数组。
            3. 深拷贝 numpy Generator 的 bit_generator.state。
        """
        # 先创建快照字典，后续按字段存在性逐项填充。
        snapshot: dict[str, Any] = {}
        # 保存 wrapper 当前真实 state，None 表示尚未 reset。
        snapshot["current_state"] = None if self._current_state is None else self._current_state.copy()
        # 保存真实环境引用，减少重复访问。
        base_env = self.base_env
        # 保存环境时间步，eval 会 reset/step，因此必须恢复。
        if hasattr(base_env, "_t"):
            snapshot["base_t"] = int(getattr(base_env, "_t"))
        # 保存 AoI 数组，避免 eval 后真实 AoI 留在评估轨迹上。
        if hasattr(base_env, "aoi"):
            snapshot["aoi"] = np.asarray(getattr(base_env, "aoi")).copy()
        # 保存离散信道状态。
        if hasattr(base_env, "channel_state"):
            snapshot["channel_state"] = np.asarray(getattr(base_env, "channel_state")).copy()
        # 保存信道丢包率矩阵。
        if hasattr(base_env, "channel_loss"):
            snapshot["channel_loss"] = np.asarray(getattr(base_env, "channel_loss")).copy()
        # 保存 RNG 状态；deepcopy 避免后续原地修改污染快照。
        if hasattr(base_env, "rng") and hasattr(base_env.rng, "bit_generator"):
            snapshot["rng_state"] = copy.deepcopy(base_env.rng.bit_generator.state)
        # 测试 fake env 会记录最近动作，这里一并恢复，避免诊断评估污染断言。
        if hasattr(base_env, "last_action_id"):
            snapshot["last_action_id"] = getattr(base_env, "last_action_id")
        if hasattr(base_env, "last_assignment"):
            snapshot["last_assignment"] = getattr(base_env, "last_assignment")
        # 返回完整快照。
        return snapshot

    def restore_runtime_state(self, snapshot: dict[str, Any]) -> None:
        """
        作用:
            将 learned attacker 环境和真实 base_env 恢复到 snapshot_runtime_state 保存的状态。

        输入格式:
            snapshot: dict[str,Any]，由 snapshot_runtime_state 返回。

        输出格式:
            None。

        参数含义:
            snapshot 保存评估前的训练环境状态，恢复后训练轨迹不受 eval 消耗的 RNG 影响。

        核心步骤:
            1. 恢复 wrapper 的 `_current_state`。
            2. 恢复 base_env 的时间、AoI、H 和丢包率。
            3. 恢复 numpy Generator 状态。
        """
        # 恢复 wrapper 当前真实 state 缓存。
        current_state = snapshot.get("current_state")
        self._current_state = None if current_state is None else np.asarray(current_state, dtype=np.float32).copy()
        # 保存真实环境引用，减少重复访问。
        base_env = self.base_env
        # 恢复时间步。
        if "base_t" in snapshot and hasattr(base_env, "_t"):
            base_env._t = int(snapshot["base_t"])
        # 恢复 AoI 数组。
        if "aoi" in snapshot and hasattr(base_env, "aoi"):
            base_env.aoi = np.asarray(snapshot["aoi"]).copy()
        # 恢复信道状态。
        if "channel_state" in snapshot and hasattr(base_env, "channel_state"):
            base_env.channel_state = np.asarray(snapshot["channel_state"]).copy()
        # 恢复信道丢包率。
        if "channel_loss" in snapshot and hasattr(base_env, "channel_loss"):
            base_env.channel_loss = np.asarray(snapshot["channel_loss"]).copy()
        # 恢复 RNG 状态，保证评估不改变后续训练随机序列。
        if "rng_state" in snapshot and hasattr(base_env, "rng") and hasattr(base_env.rng, "bit_generator"):
            base_env.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        # 恢复 fake env 最近动作字段，真实 env 没有这些字段时自动跳过。
        if "last_action_id" in snapshot and hasattr(base_env, "last_action_id"):
            base_env.last_action_id = snapshot["last_action_id"]
        if "last_assignment" in snapshot and hasattr(base_env, "last_assignment"):
            base_env.last_assignment = snapshot["last_assignment"]

    def reset(self) -> np.ndarray:
        """
        作用:
            重置真实环境，并返回 attacker 观察到的真实状态。

        输入格式:
            无。

        输出格式:
            state: np.ndarray[float32], shape=(N + N*M,)。

        参数含义:
            无参数；该函数只转发真实环境 reset。

        核心步骤:
            1. 调用 base_env.reset()。
            2. 保存真实 state 副本。
            3. 返回真实 state，不构造 attacked_state。
        """
        # 真实环境 reset 会刷新真实 AoI 和信道状态。
        state = self.base_env.reset()
        # 保存 float32 副本，保证后续 step 使用真实观测。
        self._current_state = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        # 返回副本，避免外部误改内部缓存。
        return self._current_state.copy()

    def step(self, intent_action: np.ndarray):
        """
        作用:
            执行 learned attacker 的一步交互。

        输入格式:
            intent_action: np.ndarray[float32], shape=(N + N*M,)，范围建议在 [-1,1]。

        输出格式:
            next_state: np.ndarray[float32], shape=(N + N*M,)
            attacker_reward: float，等于真实下一步 MSE。
            done: bool，真实环境 episode 是否结束。
            info: dict，包含 mapping 信息和 victim 动作信息。

        参数含义:
            intent_action 是 Actor 输出的连续扰动意图，不是最终离散 delta。

        核心步骤:
            1. 读取当前真实 state。
            2. 调用 map_intent_to_attacked_state 生成 attacked_state 和 delta。
            3. 用 attacked_state 让 victim 选择动作。
            4. 用 victim 动作推进真实 base_env。
            5. 将 -env_reward 作为 attacker raw reward。
        """
        # 若调用者没有先 reset，主动报错而不是使用空状态。
        if self._current_state is None:
            raise RuntimeError("LearnedAttackEnv.step() called before reset().")
        # 当前 state 必须是真实环境观测，不允许使用 attacked_state 回写。
        state = self._current_state.copy()
        # 将连续 intent action 投影为合法离散扰动。
        attacked_state, delta, mapping_info = self.map_intent_to_attacked_state(state, intent_action)
        # victim 只在 attacked_state 上选动作，不改变真实环境状态。
        victim_action = select_action_for_eval(self.victim_algo, attacked_state, self.victim_agent, self.base_env)
        # DQN 使用离散 action_id 推进真实环境。
        if self.victim_algo == "DQN":
            next_state, env_reward, done = self.base_env.step(int(victim_action))
        # DDPG 使用 assignment 推进真实环境。
        else:
            next_state, env_reward, done = self.base_env.step_assignment(victim_action)
        # env_reward 是负 MSE，因此 attacker 最大化 raw MSE。
        attacker_reward = -float(env_reward)
        # 缓存下一步真实 state，供下一次 step 使用。
        self._current_state = np.asarray(next_state, dtype=np.float32).reshape(-1).copy()
        # 汇总 info，便于训练循环统计能量和扰动规模。
        info: dict[str, Any] = dict(mapping_info)
        # 记录 victim 分支，方便排查 DQN/DDPG 行为。
        info["victim_algo"] = self.victim_algo
        # DQN 为 int，DDPG 为 tuple；统一保留原始动作语义。
        info["victim_action"] = victim_action
        # 保存离散扰动向量，测试和诊断可直接检查。
        info["delta"] = delta.copy()
        # 保存 raw env reward，方便确认 attacker_reward = -env_reward。
        info["env_reward"] = float(env_reward)
        # 返回真实 next_state 副本，避免外部误写内部缓存。
        return self._current_state.copy(), attacker_reward, bool(done), info

    def map_intent_to_attacked_state(self, state: np.ndarray, intent_action: np.ndarray):
        """
        作用:
            将 attacker actor 输出的连续 intent action 映射为满足单步能耗约束的离散扰动。

        输入格式:
            state: np.ndarray[float32], shape=(N + N*M,)。
            intent_action: np.ndarray[float32], shape=(N + N*M,)。

        输出格式:
            attacked_state: np.ndarray[float32], shape=(N + N*M,)。
            delta: np.ndarray[float32], shape=(N + N*M,)。
            mapping_info: dict，包含能量、L0/L1 和约束状态。

        参数含义:
            state 是真实观测；intent_action 是连续扰动意图；最终 delta 是整数扰动。

        核心步骤:
            1. 为 AoI/H 每个维度计算状态依赖上下界。
            2. 将连续 intent 转为期望扰动 hat_delta。
            3. 枚举每个维度的非零整数候选。
            4. 按单位能耗收益贪心选择候选，直到能量预算耗尽。
            5. 构造合法 attacked_state，并返回统计信息。
        """
        # 真实 state 统一为一维 float32，避免形状广播错误。
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        # intent action 统一为一维 float32，后续会 clip 到 [-1,1]。
        intent_arr = np.asarray(intent_action, dtype=np.float32).reshape(-1)
        # 状态维度不一致时立即报错，避免生成错位扰动。
        if state_arr.shape[0] != self.state_dim:
            raise ValueError(f"state shape must be ({self.state_dim},), got {state_arr.shape}.")
        # intent 维度不一致时立即报错，避免 actor 输出被误解释。
        if intent_arr.shape[0] != self.action_dim:
            raise ValueError(f"intent_action shape must be ({self.action_dim},), got {intent_arr.shape}.")

        # 将 intent 限制在 actor 设计范围内，保证映射层输入稳定。
        intent_arr = np.clip(intent_arr, -1.0, 1.0)
        # 初始化每维离散扰动为 0。
        delta = np.zeros(self.state_dim, dtype=np.float32)
        # AoI 部分的真实观测。
        aoi_part = state_arr[: self.n]
        # H 部分的真实观测。
        h_part = state_arr[self.n :]

        # AoI 下界由单维幅度和最小合法 AoI=1 共同决定。
        aoi_lower = np.maximum(-int(self.config.aoi_delta), 1.0 - aoi_part)
        # AoI 上界由单维幅度和 env.max_aoi 共同决定。
        aoi_upper = np.minimum(int(self.config.aoi_delta), float(self.base_env.max_aoi) - aoi_part)
        # H 下界由单维幅度和最小合法 H=0 共同决定。
        h_lower = np.maximum(-int(self.config.h_delta), -h_part)
        # H 上界由单维幅度和最大合法 H=4 共同决定。
        h_upper = np.minimum(int(self.config.h_delta), 4.0 - h_part)
        # 拼接所有维度的下界。
        lower = np.concatenate([aoi_lower, h_lower]).astype(np.float32)
        # 拼接所有维度的上界。
        upper = np.concatenate([aoi_upper, h_upper]).astype(np.float32)
        # 每个维度的能量代价系数。
        alpha = np.concatenate(
            [
                np.full(self.n, float(self.config.alpha_tau), dtype=np.float32),
                np.full(self.n * self.m, float(self.config.alpha_h), dtype=np.float32),
            ]
        )

        # 正 intent 贴近上界，负 intent 贴近下界。
        hat_delta = np.where(intent_arr >= 0.0, intent_arr * upper, (-intent_arr) * lower)
        # 候选格式：(收益能耗比, 收益, 维度索引, 离散扰动, 能耗)。
        candidates: list[tuple[float, float, int, int, float]] = []
        # 遍历每个维度，枚举该维度所有非零整数扰动。
        for idx in range(self.state_dim):
            # 将连续下界上取整，得到可行整数下界。
            low_i = int(np.ceil(float(lower[idx])))
            # 将连续上界下取整，得到可行整数上界。
            high_i = int(np.floor(float(upper[idx])))
            # 遍历当前维度的每个整数候选。
            for k in range(low_i, high_i + 1):
                # 零扰动不需要进入候选池。
                if k == 0:
                    continue
                # 当前候选的能耗为 alpha_i * k^2。
                cost = float(alpha[idx] * float(k * k))
                # 单个候选超过总预算时不可能被选中。
                if cost > float(self.config.per_step_energy_budget) + 1e-8:
                    continue
                # 收益定义为该维从 0 扰动改为 k 后更贴近期望扰动的距离改善。
                gain = float((hat_delta[idx] ** 2) - ((hat_delta[idx] - float(k)) ** 2))
                # 只保留确实更贴近 intent 的候选。
                if gain <= 0.0:
                    continue
                # 单位能耗收益用于贪心排序。
                ratio = gain / (cost + 1e-12)
                # 保存候选，后续按 ratio/gain 降序选择。
                candidates.append((ratio, gain, idx, int(k), cost))

        # 高单位能耗收益优先；收益并列时更大的绝对收益优先。
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        # 当前已用能量。
        energy_used = 0.0
        # 记录已选择维度，保证每个维度最多选一个 k。
        selected_dims: set[int] = set()
        # 逐个吸收候选，直到预算耗尽。
        for _ratio, _gain, idx, k, cost in candidates:
            # 每个维度只能选择一个离散扰动。
            if idx in selected_dims:
                continue
            # 加入该候选会超预算时跳过。
            if energy_used + cost > float(self.config.per_step_energy_budget) + 1e-8:
                continue
            # 写入该维最终离散扰动。
            delta[idx] = float(k)
            # 累计能量。
            energy_used += cost
            # 标记该维已使用。
            selected_dims.add(idx)

        # 构造 attacked_state，注意这里只是观测副本。
        attacked_state = state_arr + delta
        # AoI 必须是整数并处在合法范围内。
        attacked_state[: self.n] = np.clip(np.rint(attacked_state[: self.n]), 1, int(self.base_env.max_aoi))
        # H 必须是整数并处在合法范围内。
        attacked_state[self.n :] = np.clip(np.rint(attacked_state[self.n :]), 0, 4)
        # 根据实际 attacked_state 反推实际 delta，确保统计与输出一致。
        actual_delta = (attacked_state - state_arr).astype(np.float32)
        # AoI 扰动向量。
        aoi_delta = actual_delta[: self.n]
        # H 扰动向量。
        h_delta = actual_delta[self.n :]
        # AoI 部分能耗单独统计，便于确认 attacker 是否只依赖 AoI 扰动。
        aoi_energy = float(np.sum(alpha[: self.n] * np.square(aoi_delta)))
        # H 部分能耗单独统计，便于确认信道扰动是否真正被使用。
        h_energy = float(np.sum(alpha[self.n :] * np.square(h_delta)))
        # 实际能量用两个拆分项相加，保证 report 中总量与拆分完全一致。
        actual_energy = float(aoi_energy + h_energy)
        # 检查能量和合法范围是否违反约束。
        constraint_violation = bool(
            actual_energy > float(self.config.per_step_energy_budget) + 1e-6
            or np.any(attacked_state[: self.n] < 1)
            or np.any(attacked_state[: self.n] > int(self.base_env.max_aoi))
            or np.any(attacked_state[self.n :] < 0)
            or np.any(attacked_state[self.n :] > 4)
        )
        # 汇总映射统计信息，训练循环和测试都依赖这些字段。
        mapping_info = {
            "energy_used": actual_energy,
            "energy_budget": float(self.config.per_step_energy_budget),
            "aoi_energy": aoi_energy,
            "h_energy": h_energy,
            "aoi_l0": int(np.count_nonzero(aoi_delta)),
            "h_l0": int(np.count_nonzero(h_delta)),
            "total_l0": int(np.count_nonzero(actual_delta)),
            "aoi_l1": float(np.sum(np.abs(aoi_delta))),
            "h_l1": float(np.sum(np.abs(h_delta))),
            "max_abs_delta": float(np.max(np.abs(actual_delta))) if actual_delta.size else 0.0,
            "constraint_violation": constraint_violation,
        }
        # 返回 attacked_state 副本、实际 delta 和统计信息。
        return attacked_state.astype(np.float32), actual_delta, mapping_info
