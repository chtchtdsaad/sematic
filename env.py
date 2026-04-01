from __future__ import annotations

import numpy as np

from config import (
    DEFAULT_SEED,
    EPISODE_LENGTH,
    MAX_AOI,
    M,
    N,
    PACKET_LOSS_LEVELS,
    RAYLEIGH_SCALE_MAX,
    RAYLEIGH_SCALE_MIN,
    en,
    ln,
)
from core import (
    compute_mse_from_aoi,
    decode_action,
    generate_action_space,
    generate_unstable_matrix,
    solve_steady_state_covariance,
)


class SemanticSchedulingEnv:
    def __init__(self, seed: int = DEFAULT_SEED, episode_length: int = EPISODE_LENGTH) -> None:
        """
        作用:
            初始化语义感知调度环境，生成系统参数、动作空间和初始状态容器。

        参数:
            seed: 随机种子，用于复现实验。
            episode_length: 每个 episode 的最大步数。

        返回:
            None

        核心步骤:
            1. 读取全局常量并构建随机数生成器。
            2. 生成离散动作空间并计算动作总数。
            3. 为每个传感器随机生成 A、C，并设定 W、V。
            4. 预计算每个传感器的稳态协方差 P_bar。
            5. 初始化 AoI 上限、信道丢包矩阵与状态维度。
        """
        self.n = N
        self.m = M
        self.ln = ln
        self.en = en
        self.max_aoi = MAX_AOI
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        self.action_space = generate_action_space(self.n, self.m)
        self.num_actions = len(self.action_space)

        self.a_mats: list[np.ndarray] = []
        self.c_mats: list[np.ndarray] = []
        self.w_mats: list[np.ndarray] = []
        self.v_mats: list[np.ndarray] = []
        self.p_bars: list[np.ndarray] = []

        for _ in range(self.n):
            a = generate_unstable_matrix(dim=self.ln, rng=self.rng)
            c = self.rng.uniform(0.0, 1.0, size=(self.en, self.ln))
            w = np.eye(self.ln, dtype=np.float64)
            v = np.eye(self.en, dtype=np.float64)
            p_bar = solve_steady_state_covariance(a, c, w, v)

            self.a_mats.append(a)
            self.c_mats.append(c)
            self.w_mats.append(w)
            self.v_mats.append(v)
            self.p_bars.append(p_bar)

        self.state_dim = self.n + self.n * self.m
        self._t = 0
        self.aoi = np.ones(self.n, dtype=np.int64)
        self.channel_state = np.zeros((self.n, self.m), dtype=np.int64)
        self.channel_loss = np.zeros((self.n, self.m), dtype=np.float64)

        # Fixed channel statistics: each sensor-channel pair has a time-invariant Rayleigh scale.
        self.channel_scales = self.rng.uniform(
            RAYLEIGH_SCALE_MIN, RAYLEIGH_SCALE_MAX, size=(self.n, self.m)
        )
        # Fixed quantization thresholds shared by all pairs.
        self.channel_bins = self._build_global_channel_bins()
        # Fixed per-pair discrete channel-state distributions.
        self.channel_state_probs = self._build_stationary_channel_state_probs()

        self._refresh_channel_states()

    @staticmethod
    def _rayleigh_cdf(x: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        return 1.0 - np.exp(-(x_arr ** 2) / (2.0 * (sigma_arr ** 2)))

    def _build_global_channel_bins(self) -> np.ndarray:
        """
        作用:
            生成全局固定的离散化阈值，用于把连续信道质量量化为 5 档状态。

        参数:
            无。

        返回:
            np.ndarray: 长度为 4 的升序阈值数组。

        核心步骤:
            1. 选取参考 Rayleigh 尺度 sigma=1。
            2. 对分位点 [0.2, 0.4, 0.6, 0.8] 计算逆 CDF。
            3. 将阈值作为全局固定量化边界。
        """
        ref_sigma = 1.0
        quantiles = np.linspace(0.0, 1.0, len(PACKET_LOSS_LEVELS) + 1)[1:-1]
        bins = ref_sigma * np.sqrt(-2.0 * np.log(1.0 - quantiles))
        return bins.astype(np.float64)

    def _build_stationary_channel_state_probs(self) -> np.ndarray:
        """
        作用:
            为每个传感器-信道对构建随时间不变的离散信道状态分布。

        参数:
            无。

        返回:
            np.ndarray: 形状 (N, M, 5) 的概率张量，每个 (n,m,:) 和为 1。

        核心步骤:
            1. 读取每个链路固定的 Rayleigh 尺度参数 sigma_{n,m}。
            2. 利用全局阈值计算各离散区间概率。
            3. 归一化得到每个链路的固定离散分布。
        """
        num_levels = len(PACKET_LOSS_LEVELS)
        probs = np.zeros((self.n, self.m, num_levels), dtype=np.float64)

        cdf_at_bins = self._rayleigh_cdf(self.channel_bins[None, None, :], self.channel_scales[:, :, None])
        probs[:, :, 0] = cdf_at_bins[:, :, 0]
        for i in range(1, num_levels - 1):
            probs[:, :, i] = cdf_at_bins[:, :, i] - cdf_at_bins[:, :, i - 1]
        probs[:, :, -1] = 1.0 - cdf_at_bins[:, :, -1]

        probs = np.clip(probs, 0.0, 1.0)
        probs_sum = probs.sum(axis=2, keepdims=True)
        probs = probs / probs_sum
        return probs

    def _refresh_channel_states(self) -> None:
        """
        作用:
            刷新当前时隙的信道丢包率矩阵（N x M）。

        参数:
            无。

        返回:
            None

        核心步骤:
            1. 随机采样 Rayleigh 分布信道增益。
            2. 用分位数将连续增益量化为 5 档离散等级。
            3. 将等级索引映射为预定义丢包率。
        """
        for i in range(self.n):
            for j in range(self.m):
                self.channel_state[i, j] = self.rng.choice(
                    len(PACKET_LOSS_LEVELS), p=self.channel_state_probs[i, j]
                )
        self.channel_loss = PACKET_LOSS_LEVELS[self.channel_state]

    def _build_state(self) -> np.ndarray:
        """
        作用:
            将 AoI 和离散信道状态索引 H_t 拼接为一维状态向量。

        参数:
            无。

        返回:
            np.ndarray: 形状为 (N + N*M,) 的状态向量。

        核心步骤:
            1. 将 AoI 转为 float32。
            2. 将离散信道状态矩阵 H_t（索引 0~4）展平为一维向量。
            3. 拼接两部分得到最终状态。
        """
        return np.concatenate([
            self.aoi.astype(np.float32),
            self.channel_state.astype(np.float32).reshape(-1),
        ])

    def reset(self) -> np.ndarray:
        """
        作用:
            重置环境到新 episode 起点。

        参数:
            无。

        返回:
            np.ndarray: 初始状态向量。

        核心步骤:
            1. 时间步计数清零。
            2. 所有传感器 AoI 重置为 1。
            3. 刷新信道状态并构建状态向量。
        """
        self._t = 0
        self.aoi = np.ones(self.n, dtype=np.int64)
        self._refresh_channel_states()
        return self._build_state()

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool]:
        """
        作用:
            执行一步状态转移：动作解码、传输成败采样、AoI 更新、奖励计算、信道刷新。

        参数:
            action_id: 离散动作编号（0 ~ num_actions-1）。

        返回:
            tuple[np.ndarray, float, bool]:
                - next_state: 下一状态
                - reward: 当前步奖励（负总 MSE）
                - done: 是否到达 episode 结束

        核心步骤:
            1. 将 action_id 解码为传感器-信道分配。
            2. 对每个传感器按是否调度和丢包结果更新 AoI（并截断到 max_aoi）。
            3. 基于更新后的 AoI 计算所有传感器 MSE 并求和。
            4. 令 reward = -total_mse。
            5. 刷新下一个时隙信道状态并推进时间计数。
        """
        assignment = decode_action(action_id, self.action_space)

        for sensor_idx, channel_id in enumerate(assignment):
            if channel_id == 0:
                self.aoi[sensor_idx] = min(self.aoi[sensor_idx] + 1, self.max_aoi)
                continue

            channel_idx = channel_id - 1
            loss_prob = self.channel_loss[sensor_idx, channel_idx]
            is_success = self.rng.random() > loss_prob
            if is_success:
                self.aoi[sensor_idx] = 1
            else:
                self.aoi[sensor_idx] = min(self.aoi[sensor_idx] + 1, self.max_aoi)

        total_mse = 0.0
        for i in range(self.n):
            total_mse += compute_mse_from_aoi(
                p_bar=self.p_bars[i],
                a=self.a_mats[i],
                w=self.w_mats[i],
                aoi=int(self.aoi[i]),
            )

        reward = -float(total_mse)

        self._refresh_channel_states()
        self._t += 1
        done = self._t >= self.episode_length
        next_state = self._build_state()
        return next_state, reward, done

if __name__ == '__main__':
    env = SemanticSchedulingEnv(seed=22)
    state = env.reset()
    print(env.channel_state_probs)
