---

# 核心章节：面向离散观测调度器的能量约束 DRL 元攻击者建模
*(Energy-Constrained DRL Meta-Attacker for Discrete Observation Perturbations)*

**【总体逻辑引言】**  
在远程状态估计调度任务中，受害者调度器并不直接观察连续物理过程，而是基于离散化的 AoI 状态与信道状态进行传输决策。因此，攻击者若直接枚举所有可能的联合观测篡改，将面对随 $N+NM$ 指数增长的组合动作空间；而若直接输出连续扰动并简单 round 到离散状态，又会导致大量连续动作映射到同一离散扰动，削弱 Actor-Critic 方法的学习效率。

为此，我们将攻击者建模为一个在连续虚拟动作空间中决策、但通过状态依赖映射层生成合法离散观测扰动的元攻击者。其核心思想是：Actor 不直接输出最终的离散篡改状态，而是输出对 AoI 与信道观测的连续扰动意图；随后由一个**状态依赖的能量约束量化映射（State-Dependent Energy-Constrained Quantization Mapping, SDEC-QM）**将连续意图转化为满足离散语义、状态边界与单步攻击能耗约束的真实观测扰动。

与单步启发式攻击不同，该攻击者每个 step 均允许采取攻击动作，但每一步必须满足固定能耗上限。攻击策略的优化目标不是最大化单步 MSE，而是最大化整个 episode 内的平均真实估计误差。因此，映射层只负责动作合法化，不执行单步 MSE 贪心搜索；跨时域的攻击效果由 DRL 的长期回报学习完成。

---

## 一、 攻击问题建模 (Adversarial Problem Formulation)

我们考虑一个已经训练好的受害者调度策略 $\pi_v$。在每个时刻 $t$，真实系统状态为：

$$
\boldsymbol{s}_t = \left( \boldsymbol{\tau}_t, \boldsymbol{H}_t \right)
$$

其中，$\boldsymbol{\tau}_t \in \{1,2,\dots,\tau_{\max}\}^N$ 表示 $N$ 个传感器的 AoI 状态，$\boldsymbol{H}_t \in \{0,1,2,3,4\}^{N\times M}$ 表示 $N$ 个传感器到 $M$ 个信道的离散信道状态。

攻击者只能篡改受害者调度器的输入观测，而不能改变真实环境状态、真实信道状态或环境奖励。因此攻击过程满足：

$$
\tilde{\boldsymbol{s}}_t = \boldsymbol{s}_t + \boldsymbol{\Delta}_t
$$

$$
\boldsymbol{a}^{v}_t = \pi_v(\tilde{\boldsymbol{s}}_t)
$$

$$
\boldsymbol{s}_{t+1} \sim \mathcal{P}\left(\boldsymbol{s}_{t+1}\mid \boldsymbol{s}_t, \boldsymbol{a}^{v}_t\right)
$$

其中，$\tilde{\boldsymbol{s}}_t$ 是受害者看到的被攻击观测，$\boldsymbol{a}^{v}_t$ 是受害者在被攻击观测下产生的调度动作，真实环境仍然基于真实状态 $\boldsymbol{s}_t$ 和受害者调度动作 $\boldsymbol{a}^{v}_t$ 演化。

---

## 二、 攻击者 MDP 建模 (Meta-Attacker MDP)

固定受害者策略 $\pi_v$ 后，攻击者面对的是一个由“观测篡改—受害者决策—真实环境转移”组成的新 MDP：

$$
\mathcal{M}^{att} = \left\langle \mathcal{S}^{att}, \mathcal{A}^{att}, \mathcal{P}^{att}, \mathcal{R}^{att}, \gamma \right\rangle
$$

### A. 攻击者状态空间 (State Space)

为了保持模型简洁并降低工程实现难度，攻击者第一版直接观察真实调度状态：

$$
\boldsymbol{s}^{att}_t = \boldsymbol{s}_t = \left( \boldsymbol{\tau}_t, \boldsymbol{H}_t \right)
$$

其中：

$$
\boldsymbol{\tau}_t \in \{1,2,\dots,\tau_{\max}\}^N
$$

$$
\boldsymbol{H}_t \in \{0,1,2,3,4\}^{N\times M}
$$

因此攻击者网络的输入维度为：

$$
D = N + NM
$$

---

### B. 连续扰动意图动作空间 (Continuous Perturbation Intent Space)

为了避免直接枚举离散联合扰动动作，攻击者 Actor 输出连续扰动意图：

$$
\boldsymbol{a}^{att}_t = \pi_{\theta}^{att}(\boldsymbol{s}^{att}_t)
$$

其中：

$$
\boldsymbol{a}^{att}_t = \left( \boldsymbol{v}^{\tau}_t, \boldsymbol{V}^{H}_t \right)
$$

$$
\boldsymbol{v}^{\tau}_t \in [-1,1]^N
$$

$$
\boldsymbol{V}^{H}_t \in [-1,1]^{N\times M}
$$

$\boldsymbol{v}^{\tau}_t$ 表示针对 AoI 观测的连续扰动意图，$\boldsymbol{V}^{H}_t$ 表示针对信道观测的连续扰动意图。其符号含义为：

$$
v_{i,t} > 0 \Rightarrow \text{倾向于提高该维观测值}
$$

$$
v_{i,t} < 0 \Rightarrow \text{倾向于降低该维观测值}
$$

$$
|v_{i,t}| \Rightarrow \text{扰动意图强度}
$$

需要强调的是，$\boldsymbol{a}^{att}_t$ 不是最终作用到 victim 观测上的真实离散扰动，而是连续虚拟动作。最终扰动由后续映射算子 $\mathcal{Q}_{\Omega}$ 生成。

---

## 三、 单步能耗约束 (Per-Step Energy Constraint)

不同于随机攻击评估中的 attack-prob、attack-ratio 或最大连续攻击步约束，学习型攻击者每个 step 都允许进行一次攻击决策。攻击行为只需要满足单步攻击能耗约束：

$$
 c_t \le \bar{a}(t), \quad \forall t=1,2,\dots,T
$$

其中 $\bar{a}(t)$ 是攻击者在第 $t$ 个时刻允许使用的最大扰动能量。对于固定预算情形，可以设定：

$$
\bar{a}(t) = \bar{a}, \quad \forall t
$$

给定最终离散扰动：

$$
\boldsymbol{\Delta}_t = \left( \boldsymbol{\Delta}^{\tau}_t, \boldsymbol{\Delta}^{H}_t \right)
$$

单步攻击能耗定义为：

$$
c_t(\boldsymbol{\Delta}_t) = \alpha_{\tau}\sum_{n=1}^{N} \left|\Delta\tau_{n,t}\right|^2 + \alpha_H \sum_{n=1}^{N}\sum_{m=1}^{M}\left|\Delta H_{n,m,t}\right|^2
$$

其中 $\alpha_{\tau}$ 和 $\alpha_H$ 分别表示 AoI 篡改与信道观测篡改的单位能耗系数。该约束表示攻击者不能在单个时刻对过多维度或过大幅度的观测进行篡改，其中$\alpha_{\tau}$的系数应该相较于$\alpha_{H}$会大一些，因为传感器的信息年龄往往直接影响最后的reward，而信道本身也属于瑞利衰落，其信道质量也会不定式改变，所以对应的改变所需能量较小。

---

## 四、 状态依赖的能量约束量化映射 (SDEC-QM)

由于 Actor 输出的是连续扰动意图，而 victim 只能接收离散 AoI 与离散信道状态，因此必须设计连续—离散映射层：

$$
\boldsymbol{\Delta}_t^{\star} = \mathcal{Q}_{\Omega}\left(\boldsymbol{s}_t,\boldsymbol{a}^{att}_t,\bar{a}(t)\right)
$$

最终被攻击观测为：

$$
\tilde{\boldsymbol{s}}_t = \boldsymbol{s}_t + \boldsymbol{\Delta}_t^{\star}
$$

该映射层需要同时满足三个要求：

1. 与 Actor 输出的连续扰动意图尽可能一致；
2. 保证 AoI 和 H 的离散合法性；
3. 保证单步能耗不超过 $\bar{a}(t)$。

---

### A. 状态依赖可行扰动边界 (State-Dependent Feasible Bounds)

对 AoI 维度 $n$，攻击扰动必须满足：

$$
\tilde{\tau}_{n,t} = \tau_{n,t} + \Delta\tau_{n,t} \in \{1,2,\dots,\tau_{\max}\}
$$

同时单维最大扰动幅度不超过 $d_{\tau}$，因此：

$$
L^{\tau}_{n,t} = \max\left(-d_{\tau}, 1-\tau_{n,t}\right)
$$

$$
U^{\tau}_{n,t} = \min\left(d_{\tau}, \tau_{\max}-\tau_{n,t}\right)
$$

$$
\Delta\tau_{n,t} \in \{L^{\tau}_{n,t}, L^{\tau}_{n,t}+1, \dots, U^{\tau}_{n,t}\}
$$

对信道维度 $(n,m)$，攻击扰动必须满足：

$$
\tilde{H}_{n,m,t} = H_{n,m,t} + \Delta H_{n,m,t} \in \{0,1,2,3,4\}
$$

同时单维最大扰动幅度不超过 $d_H$，因此：

$$
L^{H}_{n,m,t} = \max\left(-d_H, -H_{n,m,t}\right)
$$

$$
U^{H}_{n,m,t} = \min\left(d_H, 4-H_{n,m,t}\right)
$$

$$
\Delta H_{n,m,t} \in \{L^{H}_{n,m,t}, L^{H}_{n,m,t}+1, \dots, U^{H}_{n,m,t}\}
$$

该设计避免了简单 clip 带来的动作浪费。例如，当 $\tau_{n,t}=1$ 时，任何负向 AoI 扰动都不可行，映射层会自动将该方向的可行下界调整为 0。

---

### B. 连续意图到期望扰动 (Intent-to-Desired Perturbation)

将 Actor 输出展平为统一向量：

$$
\boldsymbol{v}_t = \left[\boldsymbol{v}^{\tau}_t, \operatorname{vec}(\boldsymbol{V}^{H}_t)\right] \in [-1,1]^D
$$

其中 $D=N+NM$。对每一维 $i$，由当前状态计算其可行整数扰动上下界 $L_{i,t}$ 和 $U_{i,t}$。根据连续意图 $v_{i,t}$ 定义期望扰动：

$$
\hat{\Delta}_{i,t} =
\begin{cases}
 v_{i,t} U_{i,t}, & v_{i,t} \ge 0, \\
 (-v_{i,t}) L_{i,t}, & v_{i,t} < 0.
\end{cases}
$$

该定义具有清晰的物理意义：当 $v_{i,t}>0$ 时，Actor 希望沿正方向扰动，最大可达 $U_{i,t}$；当 $v_{i,t}<0$ 时，Actor 希望沿负方向扰动，最大可达 $L_{i,t}$。因此连续动作天然受到当前状态边界约束。

---

### C. 能量约束整数投影 (Energy-Constrained Integer Projection)

最终离散扰动由以下投影问题给出：

$$
\boldsymbol{\Delta}^{\star}_t = \arg\min_{\boldsymbol{\Delta}_t \in \mathcal{D}(\boldsymbol{s}_t,\bar{a}(t))} \left\| \boldsymbol{\Delta}_t - \hat{\boldsymbol{\Delta}}_t \right\|_2^2
$$

其中可行集合为：

$$
\mathcal{D}(\boldsymbol{s}_t,\bar{a}(t)) = \left\{ \boldsymbol{\Delta}_t:\ \Delta_{i,t}\in\mathbb{Z},\ L_{i,t}\le \Delta_{i,t}\le U_{i,t},\ \sum_{i=1}^{D}\alpha_i\Delta_{i,t}^2 \le \bar{a}(t) \right\}
$$

其中：

$$
\alpha_i =
\begin{cases}
\alpha_{\tau}, & i \in \text{AoI part}, \\
\alpha_H, & i \in \text{Channel part}.
\end{cases}
$$

该映射不是单步贪心攻击，因为其目标不是：

$$
\arg\max_{\boldsymbol{\Delta}_t} \operatorname{MSE}(t+1)
$$

而是：

$$
\arg\min_{\boldsymbol{\Delta}_t} \left\| \boldsymbol{\Delta}_t - \hat{\boldsymbol{\Delta}}_t \right\|_2^2
$$

也就是说，映射层只负责将 Actor 的连续长期策略意图转化为合法的离散扰动；至于哪些扰动有利于提高 500-step 平均 MSE，则由 Critic 在长期回报中学习。

---

### D. 工程近似求解 (Greedy Projection Approximation)

严格求解上述整数投影问题等价于一个带能量约束的整数选择问题。考虑到 $d_{\tau}$ 与 $d_H$ 通常较小，第一版可以采用候选枚举加贪心选择的近似方法。

对每一维 $i$，枚举所有非零可行扰动：

$$
k \in \{L_{i,t},\dots,U_{i,t}\}, \quad k\ne 0
$$

选择扰动 $k$ 相比不扰动带来的意图贴近收益定义为：

$$
G_{i,k} = \left(\hat{\Delta}_{i,t}-0\right)^2 - \left(\hat{\Delta}_{i,t}-k\right)^2
$$

其能耗为：

$$
C_{i,k} = \alpha_i k^2
$$

仅保留 $G_{i,k}>0$ 的候选项，并计算单位能耗收益：

$$
R_{i,k} = \frac{G_{i,k}}{C_{i,k}+\epsilon}
$$

按照 $R_{i,k}$ 从大到小选择候选扰动，直到能耗达到上限：

$$
\sum C_{i,k} \le \bar{a}(t)
$$

同时每个维度最多选择一个候选扰动。该近似方法避免了联合离散动作枚举，复杂度约为：

$$
\mathcal{O}\left((N+NM)d\log((N+NM)d)\right)
$$

其中 $d=\max(d_{\tau},d_H)$。对于 $N=20,M=10,d=2$ 的大规模场景，候选数量约为 $880$，工程上可以接受。

---

## 五、 攻击者优化目标 (Finite-Horizon Attack Objective)

攻击者的目标是最大化整个 episode 内的平均真实估计误差，而非最大化某一步的瞬时误差。设每个 episode 长度为 $T=500$，攻击者优化目标为：

$$
\max_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}^{att}}\left[ \frac{1}{T}\sum_{t=1}^{T} \operatorname{MSE}_{true}(t) \right]
$$

其中 $\operatorname{MSE}_{true}(t)$ 由真实环境状态计算，而不是由被攻击观测 $\tilde{\boldsymbol{s}}_t$ 计算。由于能耗已经由映射层作为硬约束保证，因此第一版攻击奖励可以直接设为：

$$
r_t^{att} = \operatorname{MSE}_{true}(t+1)
$$

若后续希望区分同等 MSE 下的低能耗攻击，也可以加入软惩罚项作为消融版本：

$$
r_t^{att} = \operatorname{MSE}_{true}(t+1) - \lambda c_t
$$

但主模型建议采用硬约束版本，因为其与每步攻击能量约束：

$$
c_t \le \bar{a}(t), \quad \forall t
$$

更加一致。
如果如果攻击者把真实信道状态$~h$伪造成$~\tilde h$​，可以定义一个 plausibility cost：（实现时可后续再加入）
$$
c_t^{H,plaus}=β_H\sum_{n,m}[−log~q_{n,m}(\tilde{H}_{n,m,t})+log~q_{n,m}(H_{n,m,t})]
$$


---

## 六、 训练过程与状态转移 (Training Procedure)

每个 step 的训练交互流程如下：

1. 攻击者观察真实状态：

$$
\boldsymbol{s}^{att}_t = \boldsymbol{s}_t
$$

2. Actor 输出连续扰动意图：

$$
\boldsymbol{a}^{att}_t = \pi_{\theta}^{att}(\boldsymbol{s}^{att}_t)
$$

3. 映射层生成合法离散扰动：

$$
\boldsymbol{\Delta}^{\star}_t = \mathcal{Q}_{\Omega}\left(\boldsymbol{s}_t,\boldsymbol{a}^{att}_t,\bar{a}(t)\right)
$$

4. 构造被攻击观测：

$$
\tilde{\boldsymbol{s}}_t = \boldsymbol{s}_t + \boldsymbol{\Delta}^{\star}_t
$$

5. 固定受害者调度器根据被攻击观测选择动作：

$$
\boldsymbol{a}^{v}_t = \pi_v(\tilde{\boldsymbol{s}}_t)
$$

6. 真实环境根据真实状态与受害者动作转移：

$$
\boldsymbol{s}_{t+1} \sim \mathcal{P}\left(\boldsymbol{s}_{t+1}\mid \boldsymbol{s}_t,\boldsymbol{a}^{v}_t\right)
$$

7. 攻击者获得长期学习奖励：

$$
r_t^{att} = \operatorname{MSE}_{true}(t+1)
$$

8. 经验回放池存储：

$$
\left(\boldsymbol{s}^{att}_t, \boldsymbol{a}^{att}_t, r_t^{att}, \boldsymbol{s}^{att}_{t+1}, done\right)
$$

注意，Replay Buffer 中存储的是 Actor 输出的连续虚拟动作 $\boldsymbol{a}^{att}_t$，而不是最终离散扰动 $\boldsymbol{\Delta}^{\star}_t$。这是因为 Critic 需要学习连续虚拟动作经过映射层、受害者策略与真实环境共同作用后的长期回报。

---

## 七、 算法选择：DDPG / TD3 / SAC 的适配方式

由于攻击者 Actor 的动作空间被定义为连续扰动意图：

$$
\mathcal{A}^{att} = [-1,1]^{N+NM}
$$

因此可以自然使用 DDPG、TD3 或 SAC 等连续控制算法。第一版建议优先使用 DDPG 或 TD3，原因如下：

1. 当前项目中已经存在 DDPG 连续动作到离散 assignment 的工程经验；
2. 攻击者映射层同样属于“连续虚拟动作到离散可执行结果”的范式；
3. TD3 可在 DDPG 基础上缓解 Critic 过估计问题，更适合后续增强。

若采用 TD3，Critic 学习目标可写为：

$$
y_t = r_t^{att} + \gamma \min_{j=1,2} Q_{\phi_j^-}\left(\boldsymbol{s}_{t+1}^{att}, \pi_{\theta^-}^{att}(\boldsymbol{s}_{t+1}^{att})\right)
$$

Critic 损失为：

$$
\mathcal{L}_Q(\phi_j) = \mathbb{E}\left[ \left(Q_{\phi_j}(\boldsymbol{s}_t^{att},\boldsymbol{a}_t^{att}) - y_t\right)^2 \right]
$$

Actor 更新目标为：

$$
\max_{\theta} \mathbb{E}\left[ Q_{\phi_1}\left(\boldsymbol{s}_t^{att},\pi_{\theta}^{att}(\boldsymbol{s}_t^{att})\right) \right]
$$

这里的 $Q$ 函数评价的是连续意图动作经过 $\mathcal{Q}_{\Omega}$、victim policy 与真实环境转移后的长期攻击效果。

---

## 八、 与随机攻击评估模块的区别

当前已有随机攻击评估逻辑主要用于测试阶段鲁棒性分析，其约束包括：

```text
attack_prob
max_attack_ratio
max_consecutive_steps
max_aoi_features
max_h_features
max_total_features
```

而学习型元攻击者不应沿用 attack-prob 与 attack-ratio，因为其本质是每个 step 都做一次策略决策。对于 learned attacker，核心约束应简化为：

```text
per_step_energy_budget = a_bar
alpha_tau
alpha_H
aoi_delta
h_delta
```

也就是说，随机攻击强调“什么时候攻击、随机改哪里”；学习型攻击强调“每一步在能耗上限内如何选择最有长期收益的观测篡改”。因此 learned attacker 的主模型应采用：

$$
 c_t \le \bar{a}(t), \quad \forall t
$$

而不是：

$$
\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}\left[\tilde{\boldsymbol{s}}_t\ne\boldsymbol{s}_t\right] \le \rho
$$

---

## 九、 最终可落地定义 (Final Implementable Definition)

最终，本文提出的元攻击者可以被简洁地定义为：

$$
\boldsymbol{a}^{att}_t = \pi_{\theta}^{att}(\boldsymbol{s}_t)
$$

$$
\boldsymbol{\Delta}^{\star}_t = \mathcal{Q}_{\Omega}\left(\boldsymbol{s}_t,\boldsymbol{a}^{att}_t,\bar{a}(t)\right)
$$

$$
\tilde{\boldsymbol{s}}_t = \boldsymbol{s}_t + \boldsymbol{\Delta}^{\star}_t
$$

$$
\boldsymbol{a}^{v}_t = \pi_v(\tilde{\boldsymbol{s}}_t)
$$

$$
\boldsymbol{s}_{t+1} \sim \mathcal{P}\left(\boldsymbol{s}_{t+1}\mid \boldsymbol{s}_t,\boldsymbol{a}^{v}_t\right)
$$

其中映射算子为：

$$
\mathcal{Q}_{\Omega}\left(\boldsymbol{s}_t,\boldsymbol{a}^{att}_t,\bar{a}(t)\right) = \arg\min_{\boldsymbol{\Delta}_t\in\mathcal{D}(\boldsymbol{s}_t,\bar{a}(t))} \left\| \boldsymbol{\Delta}_t - \hat{\boldsymbol{\Delta}}_t \right\|_2^2
$$

可行集合为：

$$
\mathcal{D}(\boldsymbol{s}_t,\bar{a}(t)) = \left\{ \boldsymbol{\Delta}_t:\ \Delta_{i,t}\in\mathbb{Z},\ L_{i,t}\le\Delta_{i,t}\le U_{i,t},\ \sum_i \alpha_i\Delta_{i,t}^2\le\bar{a}(t) \right\}
$$

优化目标为：

$$
\max_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}^{att}}\left[\frac{1}{T}\sum_{t=1}^{T}\operatorname{MSE}_{true}(t)\right]
$$

该建模的关键优势在于：

1. 避免了 $\mathcal{O}(2^{N+NM})$ 的离散联合动作枚举；
2. 保留了 DDPG / TD3 / SAC 等连续 Actor-Critic 算法的适用性；
3. 映射后的实际扰动严格满足 AoI/H 离散语义；
4. 单步能耗由硬约束保证，不需要额外 attack-ratio；
5. 攻击目标是 500-step 平均 MSE 最大化，而不是单步启发式最优。

---

## 十、 给代码实现的最小任务拆分

### 阶段 1：新增 learned attacker 的配置文件

建议新增：

```text
attacks/learned_attack_config.py
```

核心参数包括：

```text
per_step_energy_budget
alpha_tau
alpha_H
aoi_delta
h_delta
attacker_algo
attacker_gamma
attacker_actor_lr
attacker_critic_lr
```

不建议包含：

```text
attack_prob
max_attack_ratio
max_consecutive_steps
```

---

### 阶段 2：实现连续—离散映射函数

建议新增：

```text
attacks/intent_mapping.py
```

核心函数：

```python
map_intent_to_discrete_delta(state, intent_action, env, config)
```

输入输出：

```text
state: np.ndarray, shape=(N + N*M,)
intent_action: np.ndarray, shape=(N + N*M,)
output delta: np.ndarray, shape=(N + N*M,)
```

该函数只负责：

1. 计算每一维状态依赖边界；
2. 根据连续意图生成期望扰动；
3. 枚举候选整数扰动；
4. 按单位能耗收益贪心选择；
5. 返回满足能耗约束的离散扰动。

---

### 阶段 3：实现 learned attack 环境包装器

建议新增：

```text
attacks/learned_attacker_env.py
```

其作用是把原调度环境、固定 victim agent、映射层封装成攻击者训练环境：

```text
输入：真实 state
攻击者动作：continuous intent action
内部：intent action -> discrete delta -> attacked_state -> victim action -> env step
输出：next_state, attacker_reward, done
```

---

### 阶段 4：训练 DDPG / TD3 attacker

建议新增：

```text
train_attacker.py
```

训练时冻结 victim policy，只更新 attacker policy。训练目标是最大化：

$$
\frac{1}{500}\sum_{t=1}^{500}\operatorname{MSE}_{true}(t)
$$

---

## 十一、 交给 Codex 的 prompt

```text
请基于当前项目实现 learned adversarial DRL attacker 的建模与最小代码骨架。

核心设定：
1. 固定已有训练好的 victim scheduler。
2. attacker 只修改 victim 输入观测，不允许修改 env.aoi、env.channel_state、env.channel_loss。
3. attacker actor 输出连续扰动意图 action，shape=(N+N*M,)，取值范围 [-1,1]。
4. 使用状态依赖的能量约束量化映射 SDEC-QM，将连续意图转为离散扰动 delta。
5. 映射后 attacked_state = state + delta，victim 使用 attacked_state 选择调度动作。
6. 真实环境仍然用真实 state 和 victim action 转移。
7. 每步扰动必须满足 c_t <= per_step_energy_budget。
8. 不使用 attack_prob、max_attack_ratio、max_consecutive_steps。
9. attacker reward = true MSE at next step。
10. ReplayBuffer 存储 attacker 的连续 intent action，而不是离散 delta。

请先只完成以下文件：
- attacks/learned_attack_config.py
- attacks/intent_mapping.py
- attacks/learned_attacker_env.py

暂时不要实现完整训练脚本，先保证映射层和环境包装器可单步运行。
```

