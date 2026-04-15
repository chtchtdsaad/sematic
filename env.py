"""
环境模块（env.py）
=================
将核心数学函数封装为标准 MDP：
- 状态 s_t = [AoI 向量, H_t 离散信道状态索引展平]
- 动作 a_t = 离散 action_id
- 奖励 r_t = -clip(sum_n Tr(P_n,t), 0, n*200)
"""

from __future__ import annotations

import numpy as np

from config import (
    DEFAULT_SEED,
    EPISODE_LENGTH,
    MAX_AOI,
    PACKET_LOSS_LEVELS,
    RAYLEIGH_SCALE_MAX,
    RAYLEIGH_SCALE_MIN,
    TRACE_P_CAP_PER_SENSOR,
    en,
    ln,
    M,
    N,
)
from core import (
    compute_mse_from_aoi,
    decode_action,
    generate_action_space,
    generate_unstable_matrix,
    solve_steady_state_covariance,
)


class SemanticSchedulingEnv:
    """
    作用:
        语义感知调度环境。

    状态格式:
        np.ndarray[float32], shape=(N + N*M,)
        - 前 N 维: AoI（int 值转 float32）
        - 后 N*M 维: 离散信道状态索引 H_t（0..4）

    动作格式:
        int，范围 [0, num_actions-1]

    step 输出格式:
        tuple[np.ndarray, float, bool]
        - next_state: np.ndarray[float32], shape=(N + N*M,)
        - reward: float
        - done: bool
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        episode_length: int = EPISODE_LENGTH,
        n: int | None = None,
        m: int | None = None,
        build_action_space: bool = True,
    ) -> None:
        """
        作用:
            初始化环境中的系统模型、动作空间、信道统计和缓存。

        输入格式:
            seed: int，随机种子。
            episode_length: int，每个 episode 的最大步数。
            n: int | None，传感器数量；None 时使用 config.N。
            m: int | None，信道数量；None 时使用 config.M。
            build_action_space: bool，是否构建离散 action_space（DQN 需要，DDPG 可关闭）。

        输出格式:
            None

        核心步骤:
            1. 初始化规模参数、随机数发生器和动作空间。
            2. 为每个传感器生成 A/C/W/V 及稳态 P_bar。
            3. 预计算 MSE 所需的 A^k 与噪声和缓存。
            4. 构建固定信道分布（scale 固定，分布随时间不变）。
            5. 采样初始 H_t。
        """
        # 基础规模参数
        self.n = int(N if n is None else n)
        self.m = int(M if m is None else m)
        self.ln = ln
        self.en = en
        if self.n <= 0 or self.m <= 0 or self.m > self.n:
            raise ValueError(f"Invalid env scale: n={self.n}, m={self.m}.")

        # 环境运行参数
        self.max_aoi = int(MAX_AOI)
        self.episode_length = int(episode_length)
        # 总 Tr(P) 截断上限: n * 每传感器上限（用于抑制训练中指数爆炸）。
        self.total_mse_cap = float(self.n) * float(TRACE_P_CAP_PER_SENSOR)

        # 随机数生成器（统一来源，便于复现）
        self.rng = np.random.default_rng(seed)

        # 离散动作空间（长度: N!/(N-M)!）。
        # - DQN: 需要 action_space + action_id。
        # - DDPG(大场景): 可关闭，改为直接输入 assignment。
        self.action_space = generate_action_space(self.n, self.m) if build_action_space else []
        self.num_actions = len(self.action_space)

        # 各传感器系统参数容器
        self.a_mats: list[np.ndarray] = []
        self.c_mats: list[np.ndarray] = []
        self.w_mats: list[np.ndarray] = []
        self.v_mats: list[np.ndarray] = []
        self.p_bars: list[np.ndarray] = []

        # 各传感器 MSE 计算缓存容器
        self.a_powers_cache: list[np.ndarray] = []
        self.noise_cov_sums_cache: list[np.ndarray] = []

        # 为每个传感器构建独立系统模型
        for _ in range(self.n):
            # A: shape=(2,2), 元素正、特征值正、谱半径在 (1,1.4)
            a = generate_unstable_matrix(dim=self.ln, rng=self.rng)

            # C: shape=(1,2), 均匀分布 (0,1)
            c = self.rng.uniform(0.0, 1.0, size=(self.en, self.ln))

            # W: shape=(2,2), 单位阵
            w = np.eye(self.ln, dtype=np.float64)

            # V: shape=(1,1), 单位阵
            v = np.eye(self.en, dtype=np.float64)

            # 稳态协方差
            p_bar = solve_steady_state_covariance(a, c, w, v)

            # 构建 MSE 快速缓存
            a_powers, noise_cov_sums = self._build_mse_cache(a, w)

            # 写入容器
            self.a_mats.append(a)
            self.c_mats.append(c)
            self.w_mats.append(w)
            self.v_mats.append(v)
            self.p_bars.append(p_bar)
            self.a_powers_cache.append(a_powers)
            self.noise_cov_sums_cache.append(noise_cov_sums)

        # 状态维度: N + N*M
        self.state_dim = self.n + self.n * self.m

        # 时间步计数器
        self._t = 0

        # 运行时状态变量
        self.aoi = np.ones(self.n, dtype=np.int64)                      # shape=(N,)
        self.channel_state = np.zeros((self.n, self.m), dtype=np.int64)  # shape=(N,M), 值域 0..4
        self.channel_loss = np.zeros((self.n, self.m), dtype=np.float64) # shape=(N,M), 值域对应丢包率

        # 每个 sensor-channel 对固定一个 Rayleigh scale（时间不变）
        self.channel_scales = self.rng.uniform(
            RAYLEIGH_SCALE_MIN,
            RAYLEIGH_SCALE_MAX,
            size=(self.n, self.m),
        )

        # 全局固定分箱阈值（用于 5 档离散状态）
        self.channel_bins = self._build_global_channel_bins()

        # 每个 sensor-channel 对的离散分布（固定）
        self.channel_state_probs = self._build_stationary_channel_state_probs()

        # 初始化当前时刻信道状态
        self._refresh_channel_states()

    def _build_mse_cache(self, a: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        作用:
            预计算 MSE 所需幂次缓存，减少 step 中重复矩阵运算。

        输入格式:
            a: np.ndarray[float64], shape=(ln,ln)
            w: np.ndarray[float64], shape=(ln,ln)

        输出格式:
            tuple[np.ndarray, np.ndarray]
            - a_powers: np.ndarray[float64], shape=(max_aoi+1,ln,ln)
              a_powers[k] = A^k
            - noise_cov_sums: np.ndarray[float64], shape=(max_aoi+1,ln,ln)
              noise_cov_sums[k] = sum_{j=0}^{k-1} A^j W (A^T)^j

        核心步骤:
            1. 递推构建 A^k。
            2. 递推构建噪声累计项。
            3. 返回两组缓存。
        """
        # A^k 缓存，k=0..max_aoi
        a_powers = np.zeros((self.max_aoi + 1, self.ln, self.ln), dtype=np.float64)

        # k=0 时 A^0 = I
        a_powers[0] = np.eye(self.ln, dtype=np.float64)

        # 递推: A^k = A^(k-1) * A
        for k in range(1, self.max_aoi + 1):
            a_powers[k] = a_powers[k - 1] @ a

        # 噪声累计缓存
        noise_cov_sums = np.zeros((self.max_aoi + 1, self.ln, self.ln), dtype=np.float64)

        # 递推: sum_k = sum_{k-1} + A^(k-1) W (A^(k-1))^T
        for k in range(1, self.max_aoi + 1):
            a_prev = a_powers[k - 1]
            noise_cov_sums[k] = noise_cov_sums[k - 1] + a_prev @ w @ a_prev.T

        return a_powers, noise_cov_sums

    @staticmethod
    def _rayleigh_cdf(x: np.ndarray | float, sigma: np.ndarray | float) -> np.ndarray:
        """
        作用:
            计算 Rayleigh 分布 CDF。

        输入格式:
            x: np.ndarray | float
            sigma: np.ndarray | float（支持广播）

        输出格式:
            np.ndarray[float64]

        核心步骤:
            1. 输入转为 float64 数组。
            2. 按公式 F(x)=1-exp(-x^2/(2*sigma^2)) 计算。
        """
        x_arr = np.asarray(x, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        return 1.0 - np.exp(-(x_arr**2) / (2.0 * (sigma_arr**2)))

    def _build_global_channel_bins(self) -> np.ndarray:
        """
        作用:
            构建全局固定离散化阈值（5 档状态 -> 4 个边界）。

        输入格式:
            无

        输出格式:
            np.ndarray[float64], shape=(4,)

        核心步骤:
            1. 在参考 sigma=1.0 下取分位点 [0.2,0.4,0.6,0.8]。
            2. 对 Rayleigh 分布反解得到阈值。
        """
        # 参考尺度
        ref_sigma = 1.0

        # 5 档状态对应 4 个内部分位点
        quantiles = np.linspace(0.0, 1.0, len(PACKET_LOSS_LEVELS) + 1)[1:-1]

        # Rayleigh 分布分位点反函数
        bins = ref_sigma * np.sqrt(-2.0 * np.log(1.0 - quantiles))

        return bins.astype(np.float64)

    def _build_stationary_channel_state_probs(self) -> np.ndarray:
        """
        作用:
            构建每个 sensor-channel 对的离散信道状态分布（时间不变）。

        输入格式:
            无

        输出格式:
            np.ndarray[float64], shape=(N,M,5)，每个 (i,j,:) 和为 1。

        核心步骤:
            1. 计算每对链路在 bins 处的 CDF。
            2. 用相邻 CDF 差分得到 5 个区间概率。
            3. 归一化并返回。
        """
        # 离散状态数量（5）
        num_levels = len(PACKET_LOSS_LEVELS)

        # 概率张量容器
        probs = np.zeros((self.n, self.m, num_levels), dtype=np.float64)

        # 每个 bins 对应 CDF，shape=(N,M,4)
        cdf_at_bins = self._rayleigh_cdf(
            self.channel_bins[None, None, :],
            self.channel_scales[:, :, None],
        )

        # 第一段概率 P(X<=b1)
        probs[:, :, 0] = cdf_at_bins[:, :, 0]

        # 中间段概率 P(b_i < X <= b_{i+1})
        for i in range(1, num_levels - 1):
            probs[:, :, i] = cdf_at_bins[:, :, i] - cdf_at_bins[:, :, i - 1]

        # 最后一段概率 P(X > b_last)
        probs[:, :, -1] = 1.0 - cdf_at_bins[:, :, -1]

        # 数值安全：裁剪并归一化
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / probs.sum(axis=2, keepdims=True)

        return probs

    def _refresh_channel_states(self) -> None:
        """
        作用:
            按固定离散分布重采样当前 H_t，并更新对应丢包率矩阵。

        输入格式:
            无

        输出格式:
            None（更新 self.channel_state / self.channel_loss）

        核心步骤:
            1. 对每条链路离散分布构建 CDF。
            2. 采样 Uniform(0,1) 并用 CDF 反演得到离散索引。
            3. 由索引映射到丢包率。
        """
        # CDF 张量，shape=(N,M,5)
        cdf = np.cumsum(self.channel_state_probs, axis=2)

        # 统一采样 U~Uniform(0,1)，shape=(N,M,1)
        uniforms = self.rng.random((self.n, self.m, 1))

        # CDF 反演：统计 U 大于多少个 CDF 边界，即离散索引 0..4
        self.channel_state = np.sum(uniforms > cdf, axis=2).astype(np.int64)

        # 索引映射到真实丢包率矩阵，shape=(N,M)
        self.channel_loss = PACKET_LOSS_LEVELS[self.channel_state]

    def _build_state(self) -> np.ndarray:
        """
        作用:
            将 AoI 与 H_t 展平后拼接成状态向量。

        输入格式:
            无

        输出格式:
            np.ndarray[float32], shape=(N + N*M,)

        核心步骤:
            1. AoI 转 float32。
            2. H_t 索引矩阵展平并转 float32。
            3. 连接成一维状态向量。
        """
        # AoI: shape=(N,), float32
        aoi_part = self.aoi.astype(np.float32)

        # H_t: shape=(N,M)->(N*M,), float32
        channel_part = self.channel_state.astype(np.float32).reshape(-1)

        # 拼接后返回
        return np.concatenate([aoi_part, channel_part])

    def reset(self) -> np.ndarray:
        """
        作用:
            重置环境到新 episode 起点。

        输入格式:
            无

        输出格式:
            np.ndarray[float32], shape=(N + N*M,)

        核心步骤:
            1. 时间步清零。
            2. AoI 全部设为 1。
            3. 重采样信道状态。
            4. 返回初始状态。
        """
        # 时间步归零
        self._t = 0

        # AoI 初始化为 1
        self.aoi = np.ones(self.n, dtype=np.int64)

        # 刷新信道状态
        self._refresh_channel_states()

        # 返回状态向量
        return self._build_state()

    def _validate_assignment(self, assignment: tuple[int, ...]) -> None:
        """
        作用:
            校验 assignment 是否满足调度约束。

        输入格式:
            assignment: tuple[int,...]，长度必须为 n

        输出格式:
            None（非法时抛出 ValueError）
        """
        if len(assignment) != self.n:
            raise ValueError(f"assignment length must be n={self.n}, got {len(assignment)}.")

        # 每个信道编号 1..m 需各出现一次；0 表示未调度。
        channel_counts = np.zeros(self.m + 1, dtype=np.int64)
        for ch in assignment:
            if ch < 0 or ch > self.m:
                raise ValueError(f"assignment channel id out of range: {ch}, valid [0,{self.m}].")
            if ch > 0:
                channel_counts[ch] += 1

        if not np.all(channel_counts[1:] == 1):
            raise ValueError(
                "assignment must schedule exactly one sensor per channel "
                f"(counts={channel_counts[1:].tolist()})."
            )

    def _step_with_assignment(self, assignment: tuple[int, ...]) -> tuple[np.ndarray, float, bool]:
        """
        作用:
            使用 assignment 执行一步环境转移（DDPG 推荐路径）。

        输入格式:
            assignment: tuple[int,...]，长度 n，元素取值 0..m

        输出格式:
            tuple[np.ndarray, float, bool]
        """
        # 按每个传感器更新 AoI。
        for sensor_idx, channel_id in enumerate(assignment):
            # 未调度：AoI + 1 并截断。
            if channel_id == 0:
                self.aoi[sensor_idx] = min(self.aoi[sensor_idx] + 1, self.max_aoi)
                continue

            # 被调度：取对应信道丢包率并采样成功/失败。
            channel_idx = channel_id - 1
            loss_prob = self.channel_loss[sensor_idx, channel_idx]
            is_success = bool(self.rng.random() > loss_prob)

            # 成功则重置为 1，失败则 +1 并截断。
            if is_success:
                self.aoi[sensor_idx] = 1
            else:
                self.aoi[sensor_idx] = min(self.aoi[sensor_idx] + 1, self.max_aoi)

        # 计算总 MSE。
        total_mse = 0.0
        for i in range(self.n):
            total_mse += compute_mse_from_aoi(
                p_bar=self.p_bars[i],
                a=self.a_mats[i],
                w=self.w_mats[i],
                aoi=int(self.aoi[i]),
                a_powers=self.a_powers_cache[i],
                noise_cov_sums=self.noise_cov_sums_cache[i],
            )

        # 对总 MSE 做硬截断，控制 Tr(P) 指数爆炸对训练稳定性的冲击。
        total_mse = min(total_mse, self.total_mse_cap)

        # 奖励定义为负总 MSE。
        reward = -float(total_mse)

        # 刷新下一时刻信道状态。
        self._refresh_channel_states()

        # 时间推进。
        self._t += 1

        # done 判定。
        done = self._t >= self.episode_length

        # 构建 next_state。
        next_state = self._build_state()

        return next_state, reward, done

    def step_assignment(self, assignment: np.ndarray | list[int] | tuple[int, ...]) -> tuple[np.ndarray, float, bool]:
        """
        作用:
            直接使用 assignment 执行一步（用于 DDPG，避免离散动作全集）。

        输入格式:
            assignment: np.ndarray | list[int] | tuple[int,...]
            - 展平后长度 n
            - 元素取值 0..m，且每个 1..m 各出现一次

        输出格式:
            tuple[np.ndarray, float, bool]
        """
        # 输入统一为一维整数 tuple。
        assignment_arr = np.asarray(assignment, dtype=np.int64).reshape(-1)
        assignment_t = tuple(int(x) for x in assignment_arr.tolist())
        self._validate_assignment(assignment_t)
        return self._step_with_assignment(assignment_t)

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool]:
        """
        作用:
            执行一次环境状态转移。

        输入格式:
            action_id: int，范围 [0, num_actions-1]

        输出格式:
            tuple[np.ndarray, float, bool]
            - next_state: np.ndarray[float32], shape=(N + N*M,)
            - reward: float
            - done: bool

        核心步骤:
            1. 将 action_id 解码为传感器-信道分配方案。
            2. 根据调度与丢包结果更新 AoI（并截断到 max_aoi）。
            3. 根据更新后的 AoI 计算 total_mse 并做上限截断。
            4. reward = -total_mse（截断后）。
            5. 采样下一个时刻信道状态，推进时间并返回。
        """
        # action_space 未构建时，禁止 action_id 接口（DDPG 大场景走 step_assignment）。
        if not self.action_space:
            raise RuntimeError("action_space is not built. Use step_assignment for this environment.")

        # 解码动作编号 -> 分配 tuple（长度 N）
        assignment = decode_action(action_id, self.action_space)
        self._validate_assignment(assignment)
        return self._step_with_assignment(assignment)


if __name__ == "__main__":
    # 快速自测
    env = SemanticSchedulingEnv(seed=22)
    state = env.reset()
    print("state shape:", state.shape)
    print("num_actions:", env.num_actions)
