# Codex 分阶段任务文档：能量约束 DRL 元攻击者训练框架

> 目标：在当前 `Remote_Estimation_RL` 项目中，基于已有 victim scheduler 的训练框架，新增一个 learned adversarial DRL attacker。该 attacker 只篡改 victim 的输入观测，不修改真实环境状态；Actor 输出连续扰动意图，并通过状态依赖的能量约束量化映射生成合法离散扰动；训练目标是最大化 500-step episode 内的平均真实 MSE。

---

## 0. 总体工程原则

### 0.1 本任务不做的事情

本任务不是重写当前调度器训练框架，也不是改动原始环境动力学。Codex 必须严格遵守以下边界：

```text
禁止修改 env.aoi、env.channel_state、env.channel_loss 的真实含义。
禁止让 attacker 直接写回真实环境状态。
禁止把 attacked_state 传入 env.step 作为真实状态。
禁止在 learned attacker 主模型中继续使用 attack_prob、max_attack_ratio、max_consecutive_steps。
禁止把映射层写成单步 MSE 贪心搜索。
禁止修改已有 victim DQN/DDPG 的训练逻辑，除非是为了新增独立入口且保持默认行为不变。
```

### 0.2 本任务必须实现的事情

```text
1. 固定已有 victim scheduler。（根据已有的checkpoint，对其调度器进行攻击）
2. attacker 观察真实 state = [AoI, H]。
3. attacker actor 输出 continuous intent action，shape=(N+N*M,)。
4. learned attack env 内部将连续的intent action 映射为离散 delta。
5. attacked_state = state + delta，只用于 victim 选动作。
6. 真实 env 使用 victim 在 attacked_state 下选择出的 action 推进一步。
7. 每一步扰动满足 c_t <= per_step_energy_budget（在一开始可以设置我们的budget尽可能大，探索攻击边界，后续再慢慢回调）。
8. attacker reward = true MSE at next step，即可先用 -env_reward。
9. ReplayBuffer 存 attacker 的 continuous intent action，不存最终离散 delta。
10. 最终通过 main.py 的命令行入口启动 attacker 训练。
```

---

## 1. 当前项目结构与新增结构

### 1.1 当前核心结构

当前项目已经包含 victim scheduler 训练相关文件：

```text
Remote_Estimation_RL/
├── config.py          # 全局配置
├── env.py             # 原始调度环境 SemanticSchedulingEnv
├── agent.py           # DQNAgent / DDPGAgent / ReplayBuffer
├── train.py           # victim 训练与评估函数
├── workflow.py        # victim 环境、agent、训练配置构建
├── main.py            # victim 训练/评估命令行入口
├── utils.py           # 连续动作到 assignment 的映射等工具
└── attacks/           # 已有随机攻击评估相关模块
```

### 1.2 本任务新增或修改后的结构

本任务采用“尽量贴近 victim DDPG 训练框架”的工程组织方式。第一版建议新增：

```text
Remote_Estimation_RL/
├── attacks/
│   ├── learned_attack_config.py     # learned attacker 配置 dataclass
│   ├── learned_attack_env.py        # learned attacker 环境包装器，第一版 mapping 写在这里
│   └── attack_agent.py              # DDPG 风格 attacker agent
├── train.py                         # 新增 run_attacker_training / save_attacker_checkpoint 等函数
├── workflow.py                      # 新增 build_learned_attack_env / build_attack_agent / build_attacker_train_config
└── main.py                          # 新增 train-attacker 命令行入口
```

> 注意：虽然后续可以把 mapping 单独拆成 `attacks/intent_mapping.py`，或加入util，但第一版为了减少文件耦合，先把 mapping 写进 `learned_attack_env.py`，后续稳定后再拆分。

---

## 2. 数学建模与代码对象对应关系

### 2.1 真实状态

原始调度状态为：

$$
\boldsymbol{s}_t = \left(\boldsymbol{\tau}_t, \boldsymbol{H}_t\right)
$$

其中：

$$
\boldsymbol{\tau}_t \in \{1,2,\dots,\tau_{\max}\}^N
$$

$$
\boldsymbol{H}_t \in \{0,1,2,3,4\}^{N\times M}
$$

代码中对应：

```text
state: np.ndarray[float32], shape=(N + N*M,)
state[:N]      -> AoI
state[N:]      -> flattened H
```

### 2.2 attacker 连续动作

攻击者 Actor 输出：

$$
\boldsymbol{a}^{att}_t = \pi^{att}_{\theta}(\boldsymbol{s}_t)
$$

其中：

$$
\boldsymbol{a}^{att}_t \in [-1,1]^{N+NM}
$$

代码中对应：

```text
intent_action: np.ndarray[float32], shape=(N + N*M,)
```

该动作不是最终离散扰动，而是连续扰动意图。

### 2.3 映射层

映射层定义为：

$$
\boldsymbol{\Delta}^{\star}_t
=
\mathcal{Q}_{\Omega}\left(\boldsymbol{s}_t,\boldsymbol{a}^{att}_t,\bar{a}\right)
$$

输出：

$$
\boldsymbol{\Delta}^{\star}_t \in \mathbb{Z}^{N+NM}
$$

并满足：

$$
\sum_i \alpha_i\left(\Delta^{\star}_{i,t}\right)^2 \le \bar{a}
$$

代码中对应：

```python
attacked_state, delta, mapping_info = attack_env.map_intent_to_attacked_state(state, intent_action)
```

### 2.4 attacked observation 与真实环境转移

构造 attacked observation：

$$
\tilde{\boldsymbol{s}}_t = \boldsymbol{s}_t + \boldsymbol{\Delta}^{\star}_t
$$

victim 根据 attacked observation 选动作：

$$
\boldsymbol{a}^{v}_t = \pi_v(\tilde{\boldsymbol{s}}_t)
$$

真实环境转移：

$$
\boldsymbol{s}_{t+1}\sim
\mathcal{P}\left(
\boldsymbol{s}_{t+1}\mid \boldsymbol{s}_t,\boldsymbol{a}^{v}_t
\right)
$$

代码中必须保证：

```text
attacked_state 只用于 victim action selection。
真实 env.step / env.step_assignment 仍然推进真实 env。
```

### 2.5 attacker reward

第一版使用硬能耗约束，因此奖励不再减能耗惩罚：

$$
r_t^{att}=\operatorname{MSE}_{true}(t+1)
$$

如果当前 env.step 返回：

$$
r_t^{env}=-\operatorname{MSE}_{true}(t+1)
$$

则代码中第一版可用：

```python
attacker_reward = -float(env_reward)
```

---

## 3. 阶段执行总览

请让 Codex 按以下顺序逐步执行。每个阶段完成后必须能单独运行最小验收，不要一次性修改所有文件。

```text
阶段 1：新增 learned attacker 配置文件。
阶段 2：新增 learned attacker 环境包装器，并在 env 中实现 mapping。
阶段 3：新增 DDPG 风格 attack_agent.py。
阶段 4：在 train.py 中封装 run_attacker_training。
阶段 5：在 workflow.py 中封装 attacker 构建流程。
阶段 6：在 main.py 中接入 train-attacker 命令行。
阶段 7：补充 USAGE_EXAMPLE.md 使用命令和最小验收流程。

```

---

# 阶段 1：新增 learned attacker 配置文件

## 目标

新增：

```text
attacks/learned_attack_config.py
```

该文件只负责 learned attacker 的配置，不要复用随机攻击的 `AttackConfig`，因为 learned attacker 不使用 attack probability 或 episode attack ratio。

## 文件职责

```text
learned_attack_config.py
    保存 learned attacker 的全部训练参数、映射约束参数、保存路径参数。
```

## 必须包含的 dataclass

```python
from dataclasses import dataclass


@dataclass
class LearnedAttackConfig:
    """
    作用:
        保存 learned adversarial DRL attacker 的训练配置与约束参数。

    输入格式:
        通常由 main.py 的 argparse.Namespace 构造。

    输出格式:
        LearnedAttackConfig 实例。

    核心步骤:
        1. 记录 attacker 算法与训练超参数。
        2. 记录 SDEC-QM 映射所需的能量、幅值和代价系数。
        3. 记录 victim checkpoint 与结果保存配置。
    """

    # attacker 算法与训练参数
    attacker_algo: str = "DDPG"
    attacker_episodes: int = 300
    attacker_gamma: float = 0.95
    attacker_batch_size: int = 128
    attacker_buffer_capacity: int = 20000
    attacker_warmup_steps: int = 2000
    attacker_update_interval: int = 1

    # attacker DDPG 网络与优化参数
    attacker_actor_lr: float = 1e-4
    attacker_critic_lr: float = 1e-3
    attacker_tau: float = 0.005
    attacker_noise_std_start: float = 0.25
    attacker_noise_std_decay: float = 0.995
    attacker_noise_std_min: float = 0.03
    attacker_grad_clip_norm: float = 5.0

    # SDEC-QM 映射约束
    per_step_energy_budget: float = 2.0
    alpha_tau: float = 1.0
    alpha_h: float = 0.25
    aoi_delta: int = 1
    h_delta: int = 1

    # 训练与保存
    save_attacker_checkpoints: bool = True
    save_attacker_history: bool = True
    save_attacker_report: bool = True
    attacker_result_dir: str = "results/attacker"

    # victim 加载信息
    victim_algo: str = "DDPG"
    victim_model_path: str = ""
```

## 必须实现的函数

```python
def build_learned_attack_config_from_args(args) -> LearnedAttackConfig:
    """
    作用:
        从 main.py 的 argparse.Namespace 构造 LearnedAttackConfig。

    输入格式:
        args: argparse.Namespace，需要包含 attacker 相关 CLI 参数。

    输出格式:
        LearnedAttackConfig 实例。

    核心步骤:
        1. 从 args 读取 attacker 训练超参数。
        2. 从 args 读取 SDEC-QM 映射约束参数。
        3. 将 0/1 保存开关转为 bool。
    """
```

## 边界要求

```text
1. 不要引入 attack_prob。
2. 不要引入 max_attack_ratio。
3. 不要引入 max_consecutive_steps。
4. 不要修改原 attacks/attack_config.py。
5. 不要修改 random_semantic.py。
```

## 交给 Codex 的 prompt

```text
请只完成阶段 1：新增 learned attacker 配置文件。

要求：
1. 新建 attacks/learned_attack_config.py。
2. 实现 LearnedAttackConfig dataclass。
3. 实现 build_learned_attack_config_from_args(args)。
4. 参数只包含 learned attacker 训练参数、SDEC-QM 映射参数、victim checkpoint 参数和保存参数。
5. 不要包含 attack_prob、max_attack_ratio、max_consecutive_steps。
6. 不要修改已有 random attack 配置文件。
7. 保持项目风格：中文 docstring，关键代码中文注释。
```

## 验收标准

```bash
python -m py_compile attacks/learned_attack_config.py
```

并且可以在 Python 中导入：

```bash
python - <<'PY'
from attacks.learned_attack_config import LearnedAttackConfig
cfg = LearnedAttackConfig()
print(cfg.per_step_energy_budget, cfg.alpha_tau, cfg.alpha_h)
PY
```

---

# 阶段 2：新增 learned attacker 环境包装器与映射层

## 目标

新增：

```text
attacks/learned_attack_env.py
```

该文件是 learned attacker 的核心。它包装已有 `SemanticSchedulingEnv`、固定 victim agent 和 SDEC-QM 映射逻辑，使 attacker 可以像普通连续控制环境一样训练。

## 类设计

实现：

```python
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

    核心步骤:
        1. reset 时返回真实 state。
        2. step(intent_action) 时将连续 intent 映射为合法离散 delta。
        3. 用 attacked_state 让 victim 选择调度动作。
        4. 用 victim action 推进真实 base_env。
        5. 返回 next_state、attacker_reward、done、info。
    """
```

## 必须包含的属性

```python
self.base_env
self.victim_agent
self.victim_algo
self.config
self.n
self.m
self.state_dim
self.action_dim
```

其中：

```text
state_dim = base_env.state_dim = N + N*M
action_dim = state_dim
```

## reset 函数

```python
def reset(self) -> np.ndarray:
    """
    作用:
        重置真实环境，并返回 attacker 观察到的真实状态。

    输出格式:
        state: np.ndarray[float32], shape=(N + N*M,)
    """
```

要求：

```text
1. 调用 self.base_env.reset()。
2. 返回真实 state，不做 attacked_state。
3. 不修改 victim agent。
```

## step 函数

```python
def step(self, intent_action: np.ndarray):
    """
    作用:
        执行 attacker 的一步交互。

    输入格式:
        intent_action: np.ndarray[float32], shape=(N + N*M,), range [-1,1]

    输出格式:
        next_state: np.ndarray[float32], shape=(N + N*M,)
        attacker_reward: float
        done: bool
        info: dict
    """
```

核心流程：

```text
1. 读取当前真实 state。
2. 调用 map_intent_to_attacked_state 生成 attacked_state、delta、mapping_info。
3. 调用 victim action selection：
   - DQN: victim 在 attacked_state 上输出 action_id。
   - DDPG: victim 在 attacked_state 上输出 continuous action，再映射为 assignment。
4. 使用 victim action 推进真实 base_env。
5. attacker_reward = -env_reward。
6. 返回 next_state, attacker_reward, done, info。
```

## victim action selection 要求

优先复用已有：

```python
from attacks.action_utils import select_action_for_eval
```

step 中使用：

```python
victim_action = select_action_for_eval(self.victim_algo, attacked_state, self.victim_agent, self.base_env)
```

然后：

```python
if self.victim_algo == "DQN":
    next_state, env_reward, done = self.base_env.step(int(victim_action))
else:
    next_state, env_reward, done = self.base_env.step_assignment(victim_action)
```

## 映射函数：map_intent_to_attacked_state

第一版直接写在 `LearnedAttackEnv` 内部：

```python
def map_intent_to_attacked_state(self, state: np.ndarray, intent_action: np.ndarray):
    """
    作用:
        将 attacker actor 输出的连续 intent action 映射为满足单步能耗约束的离散扰动，并构造 attacked_state。

    输入格式:
        state: np.ndarray[float32], shape=(N + N*M,)
        intent_action: np.ndarray[float32], shape=(N + N*M,)

    输出格式:
        attacked_state: np.ndarray[float32], shape=(N + N*M,)
        delta: np.ndarray[float32], shape=(N + N*M,)
        mapping_info: dict
    """
```

### 映射公式

对每一维计算状态依赖边界：

AoI 维度：

$$
L^{\tau}_{n,t}=\max(-d_{\tau},1-\tau_{n,t})
$$

$$
U^{\tau}_{n,t}=\min(d_{\tau},\tau_{\max}-\tau_{n,t})
$$

H 维度：

$$
L^H_{n,m,t}=\max(-d_H,-H_{n,m,t})
$$

$$
U^H_{n,m,t}=\min(d_H,4-H_{n,m,t})
$$

连续意图转期望扰动：

$$
\hat{\Delta}_{i,t}=
\begin{cases}
 v_{i,t}U_{i,t}, & v_{i,t}\ge 0,\\
 (-v_{i,t})L_{i,t}, & v_{i,t}<0.
\end{cases}
$$

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

#### 对于工程上我们可以这样实现：

枚举每个维度所有非零可行整数扰动：


$$
k\in\{L_{i,t},\dots,U_{i,t}\},\quad k\ne 0
$$



意图贴近收益：
$$
G_{i,k}=(\hat{\Delta}_{i,t}-0)^2-(\hat{\Delta}_{i,t}-k)^2
$$

能耗：

$$
C_{i,k}=\alpha_i k^2
$$



单位能耗收益：
$$
R_{i,k}=\frac{G_{i,k}}{C_{i,k}+\epsilon}
$$



贪心选择规则：

```text
1. 仅保留 G_{i,k} > 0 的候选。
2. 按 R_{i,k} 从大到小排序。
3. 逐个选择候选，直到总能耗超过 per_step_energy_budget。
4. 每个维度最多选择一个 k。
5. 输出 delta。
```

### attacked_state 合法性

```python
attacked_state = state + delta
attacked_state[:n] = np.clip(np.rint(attacked_state[:n]), 1, base_env.max_aoi)
attacked_state[n:] = np.clip(np.rint(attacked_state[n:]), 0, 4)
```

### mapping_info 字段

必须返回：

```python
mapping_info = {
    "energy_used": float,
    "energy_budget": float,
    "aoi_l0": int,
    "h_l0": int,
    "total_l0": int,
    "aoi_l1": float,
    "h_l1": float,
    "max_abs_delta": float,
    "constraint_violation": bool,
}
```

其中 `constraint_violation` 正常必须为 False。

## 边界要求

```text

2. 映射层只根据 state、intent_action 和 energy budget 生成合法 delta。
3. 不要枚举联合扰动空间。
4. 不要调用 env.step 来测试候选扰动效果（就类似贪心算法了）。
5. 不要修改 base_env 内部变量，只能通过 base_env.step 推进。
```

## 交给 Codex 的 prompt

```text
请只完成阶段 2：新增 learned attacker 环境包装器和 SDEC-QM 映射层。

要求：
1. 新建 attacks/learned_attack_env.py。
2. 实现 LearnedAttackEnv 类。
3. __init__ 接收 base_env、victim_agent、victim_algo、config。
4. reset 返回 base_env.reset() 的真实 state。
5. step(intent_action) 内部完成：
   - map_intent_to_attacked_state
   - victim 在 attacked_state 上选动作
   - base_env 使用 victim 动作推进真实环境
   - attacker_reward = -env_reward
6. 第一版把 map_intent_to_attacked_state 写在 LearnedAttackEnv 内部，不单独拆文件。
7. 映射层必须满足单步能耗 per_step_energy_budget。
8. 映射层必须保证 AoI 在 [1, env.max_aoi]，H 在 [0,4]。
9. 映射层不得做单步 MSE 贪心，不得调用 env.step 测试候选。
10. 保持中文 docstring 和中文关键注释。
```

## 验收标准

```bash
python -m py_compile attacks/learned_attack_env.py
```

最小单步测试思路：

```text
1. 构建 base_env。
2. 加载 victim checkpoint。
3. 构建 LearnedAttackEnv。
4. reset 得到 state。
5. 随机生成 intent_action，shape=(state_dim,)。
6. 调用 step。
7. 检查 info["energy_used"] <= info["energy_budget"]。
8. 检查 attacked_state 没有写回 base_env.aoi / base_env.channel_state。
```

---

# 阶段 3：新增 DDPG 风格 attack_agent.py

## 目标

新增：

```text
attacks/attack_agent.py
```

实现一个和当前 victim DDPG 风格接近的 continuous attacker agent。

## 文件职责

```text
attack_agent.py
    定义 attacker actor、attacker critic、AttackDDPGAgent。
```

## 网络输入输出

```text
state_dim = N + N*M
action_dim = N + N*M
actor 输出范围 [-1,1]
critic 输入 concat(state, action)
```

## 必须实现的类

```python
class AttackActor(nn.Module):
    """
    输入: state tensor, shape=(B,state_dim)
    输出: intent action tensor, shape=(B,action_dim), range [-1,1]
    """
```

Actor 最后一层必须使用：

```python
return torch.tanh(self.out(x))
```

```python
class AttackCritic(nn.Module):
    """
    输入:
        state: torch.Tensor, shape=(B,state_dim)
        action: torch.Tensor, shape=(B,action_dim)
    输出:
        q: torch.Tensor, shape=(B,1)
    """
```

```python
class AttackDDPGAgent:
    """
    作用:
        learned meta-attacker 的 DDPG agent。
    """
```

## AttackDDPGAgent 必须包含

```text
actor
actor_target
critic
critic_target
actor_optimizer
critic_optimizer
select_action(...)
update(...)
soft_update(...)
save/load 可先不做，统一交给 train.py 的 checkpoint 函数。
```

## select_action 规则

```python
def select_action(self, state, noise_std=0.0):
    """
    输入:
        state: np.ndarray[float32], shape=(state_dim,)
    输出:
        action: np.ndarray[float32], shape=(action_dim,), clipped to [-1,1]
    """
```

```text
1. eval/train 时都先转 torch。
2. actor 输出 action。
3. 若 noise_std > 0，加高斯探索噪声。
4. 最后 clip 到 [-1,1]。
```

## update 规则

```python
def update(self, replay_buffer, batch_size):
```

损失：

$$
y_t=r_t+\gamma(1-done)Q_{\phi^-}(s_{t+1},\pi_{\theta^-}(s_{t+1}))
$$

$$
\mathcal{L}_Q=\mathbb{E}\left[(Q_{\phi}(s_t,a_t)-y_t)^2\right]
$$

$$
\mathcal{L}_{actor}=-\mathbb{E}\left[Q_{\phi}(s_t,\pi_{\theta}(s_t))\right]
$$

注意：

```text
1. ReplayBuffer 中 action 是 continuous intent action。
2. update 不需要知道最终离散 delta。
3. mapping/env 只发生在交互阶段，不发生在 update 阶段。
```

## 是否复用 agent.py 中 ReplayBuffer

第一版建议直接复用：

```python
from agent import ReplayBuffer
```

不要重复写 ReplayBuffer，避免项目不系统。

## 交给 Codex 的 prompt

```text
请只完成阶段 3：新增 DDPG 风格 learned attacker agent。

要求：
1. 新建 attacks/attack_agent.py。
2. 实现 AttackActor、AttackCritic、AttackDDPGAgent。
3. state_dim = action_dim = N + N*M，由外部传入。
4. Actor 最后一层使用 tanh，保证输出 [-1,1]。
5. Critic 输入为 concat(state, action)。
6. AttackDDPGAgent 实现 select_action、update、soft_update。
7. update 使用 ReplayBuffer 中的 continuous intent action。
8. 不要在 agent.py 中改动 victim DQN/DDPG。
9. 不要在 update 中调用 mapping 或 env.step。
10. 保持中文 docstring 和中文关键注释。
```

## 验收标准

```bash
python -m py_compile attacks/attack_agent.py
```

最小 forward 测试：

```bash
python - <<'PY'
import numpy as np
from attacks.attack_agent import AttackDDPGAgent
agent = AttackDDPGAgent(state_dim=24, action_dim=24, device='cpu')
s = np.zeros(24, dtype=np.float32)
a = agent.select_action(s, noise_std=0.1)
print(a.shape, a.min(), a.max())
PY
```

---

# 阶段 4：在 train.py 中封装 attacker 训练函数

## 目标

修改：

```text
train.py
```

新增 learned attacker 训练入口，但不破坏已有 `run_training` 和 `run_evaluation`。

## 新增函数 1：checkpoint 构建

```python
def _build_attacker_checkpoint(attacker_agent, meta: dict | None = None) -> dict:
    """
    作用:
        组装 learned attacker checkpoint。

    输入格式:
        attacker_agent: AttackDDPGAgent
        meta: dict | None

    输出格式:
        dict[str, Any]
    """
```

保存内容：

```text
attacker_algo
actor
actor_target
critic
critic_target
actor_optimizer
critic_optimizer
meta
```

## 新增函数 2：保存 checkpoint

```python
def save_attacker_checkpoint(attacker_agent, save_path, meta=None) -> Path:
```

## 新增函数 3：加载 checkpoint

```python
def load_attacker_checkpoint(attacker_agent, ckpt_path, map_location=None) -> dict:
```

## 新增函数 4：run_attacker_training

```python
def run_attacker_training(attack_env, attacker_agent, config: dict | None = None) -> dict[str, Any]:
    """
    作用:
        训练 learned adversarial DRL attacker。

    输入格式:
        attack_env: LearnedAttackEnv
        attacker_agent: AttackDDPGAgent
        config: dict[str, Any]，由 workflow 构造

    输出格式:
        dict[str, Any]，包含训练曲线、best checkpoint 路径和训练耗时。
    """
```

## 训练循环逻辑

每个 episode：

```text
state = attack_env.reset()
for t in range(episode_length):
    if global_step < warmup_steps:
        intent_action = np.random.uniform(-1, 1, size=attack_env.action_dim)
    else:
        intent_action = attacker_agent.select_action(state_for_agent, noise_std=current_noise)

    next_state, attacker_reward, done, info = attack_env.step(intent_action)

    replay_buffer.push(state_for_agent, intent_action, attacker_reward_scaled, next_state_for_agent, done)

    if len(replay_buffer) >= batch_size and global_step % update_interval == 0:
        attacker_agent.update(replay_buffer, batch_size)

    state = next_state
```



## 训练输出字段

返回字典至少包含：

```python
{
    "attacker_reward_history": list[float],
    "attacker_mse_history": list[float],
    "energy_used_history": list[float],
    "best_reward": float,
    "best_model_path": str | None,
    "last_model_path": str | None,
    "train_seconds": float,
}
```

其中：

```text
attacker_reward_history 记录每个 episode 的平均 raw attacker_reward。
attacker_mse_history 可以等同于 average raw reward。
energy_used_history 记录每个 episode 平均 energy_used。
```

## 保存规则

```text
1. 若 save_attacker_checkpoints=1，则保存 best 和 last。
2. 保存目录：results/attacker/checkpoints/
3. 文件名建议：attacker_ddpg_vs_{victim_algo}_{scenario}_seed{seed}_best.pt
4. 测试时可通过 save_attacker_checkpoints=0 禁止落盘。
```

## 交给 Codex 的 prompt

```text
请只完成阶段 4：在 train.py 中新增 learned attacker 训练函数。

要求：
1. 不要破坏已有 run_training / run_evaluation。
3. 新增 _build_attacker_checkpoint、save_attacker_checkpoint、load_attacker_checkpoint。
4. 新增 run_attacker_training(attack_env, attacker_agent, config)。
5. 训练循环参考现有 DDPG：warmup、replay buffer、noise decay、update interval、soft update。
6. ReplayBuffer 存 normalized state 和 continuous intent action。
7. attack_env.step 接收原始 intent_action，并内部完成 victim 调度与真实环境推进。
8. attacker reward 使用 -env_reward 的 raw MSE，训练更新前可做 clip。
9. 返回训练 history dict。
10. 保持中文 docstring 和中文关键注释。
```

## 验收标准

```bash
python -m py_compile train.py
```

并且已有 victim 命令仍能解析和运行，不因新增 attacker 函数报错。

---

# 阶段 5：在 workflow.py 中封装 attacker 构建流程

## 目标

修改：

```text
workflow.py
```

新增和 victim 训练对齐的 attacker 构建函数。

## 新增函数 1：build_victim_for_attack

```python
def build_victim_for_attack(args, device):
    """
    作用:
        构建并加载固定 victim scheduler，供 learned attacker 训练使用。

    输入格式:
        args: argparse.Namespace
        device: torch.device

    输出格式:
        tuple[base_env, victim_agent, victim_algo, victim_meta]
    """
```

核心流程：

```text
1. 使用 build_env(seed=args.seed, scenario=args.scenario, algo=args.victim_algo) 构建 base_env。
2. 使用 build_agent 构建 victim_agent。
3. 加载 args.victim_model_path；若为空，则使用默认 best checkpoint 路径。
4. 冻结 victim 网络参数。
5. 将 victim 网络设为 eval 模式。
6. 返回 base_env 和 victim_agent。
```

冻结逻辑：

```python
for p in victim_agent.xxx.parameters():
    p.requires_grad_(False)
```

DQN 和 DDPG 分支分别处理。

## 新增函数 2：build_learned_attack_env

```python
def build_learned_attack_env(base_env, victim_agent, victim_algo, learned_attack_config):
    """
    作用:
        构建 LearnedAttackEnv 包装器。
    """
```

内部：

```python
from attacks.learned_attack_env import LearnedAttackEnv
return LearnedAttackEnv(base_env=base_env, victim_agent=victim_agent, victim_algo=victim_algo, config=learned_attack_config)
```

## 新增函数 3：build_attack_agent

```python
def build_attack_agent(args, attack_env, device):
    """
    作用:
        构建 AttackDDPGAgent。
    """
```

内部：

```python
from attacks.attack_agent import AttackDDPGAgent
agent = AttackDDPGAgent(
    state_dim=attack_env.state_dim,
    action_dim=attack_env.action_dim,
    gamma=cfg.attacker_gamma,
    actor_lr=cfg.attacker_actor_lr,
    critic_lr=cfg.attacker_critic_lr,
    tau=cfg.attacker_tau,
    device=device,
)
```

## 新增函数 4：build_attacker_train_config

```python
def build_attacker_train_config(args, learned_attack_config) -> dict:
    """
    作用:
        构造 run_attacker_training 所需 dict。
    """
```

字段至少包括：

```python
{
    "episodes": args.attacker_episodes,
    "episode_length": EPISODE_LENGTH,
    "batch_size": cfg.attacker_batch_size,
    "replay_buffer_capacity": cfg.attacker_buffer_capacity,
    "warmup_steps": cfg.attacker_warmup_steps,
    "update_interval": cfg.attacker_update_interval,
    "noise_std_start": cfg.attacker_noise_std_start,
    "noise_std_decay": cfg.attacker_noise_std_decay,
    "noise_std_min": cfg.attacker_noise_std_min,
    "reward_clip": args.attacker_reward_clip,
    "reward_scale": args.attacker_reward_scale,
    "save_checkpoints": cfg.save_attacker_checkpoints,
    "save_history": cfg.save_attacker_history,
    "scenario": args.scenario,
    "seed": args.seed,
    "victim_algo": args.victim_algo,
    "attacker_result_dir": cfg.attacker_result_dir,
}
```

## 边界要求

```text
1. 不要影响已有 build_env / build_agent / build_train_config 默认行为。
2. 新增函数名必须清晰，不覆盖 victim 函数。
3. victim checkpoint 加载失败时要明确报错，不要静默训练随机 victim。
4. victim 冻结后不可在 attacker training 中更新。
```

## 交给 Codex 的 prompt

```text
请只完成阶段 5：在 workflow.py 中新增 learned attacker 构建流程。

要求：
1. 不要破坏已有 build_env、build_agent、build_train_config。
2. 新增 build_victim_for_attack(args, device)。
3. 新增 build_learned_attack_env(base_env, victim_agent, victim_algo, learned_attack_config)。
4. 新增 build_attack_agent(args, attack_env, device)。
5. 新增 build_attacker_train_config(args, learned_attack_config)。
6. build_victim_for_attack 必须加载 checkpoint，若路径不存在必须报错。
7. victim 加载后必须冻结参数并设为 eval 模式。
8. build_attack_agent 使用 attacks/attack_agent.py 中的 AttackDDPGAgent。
9. 保持中文 docstring 和中文关键注释。
```

## 验收标准

```bash
python -m py_compile workflow.py
```

并且原 victim 训练命令不受影响。

---

# 阶段 6：在 main.py 中接入 train-attacker 命令行

## 目标

修改：

```text
main.py
```

新增 learned attacker 训练入口。

## CLI 设计建议

为了不破坏原有 victim 训练逻辑，建议新增一个顶层参数：

```python
parser.add_argument(
    "--task",
    type=str,
    default="victim",
    choices=["victim", "attacker"],
    help="Run victim scheduler training/eval or learned attacker training.",
)
```

保留原有：

```text
--algo DQN/DDPG
--mode train/eval
```

对于 attacker 训练，新增：

```python
parser.add_argument("--victim-algo", type=str, default="DDPG", choices=["DQN", "DDPG"])
parser.add_argument("--victim-model-path", type=str, default="")
parser.add_argument("--attacker-algo", type=str, default="DDPG", choices=["DDPG"])
parser.add_argument("--attacker-episodes", type=int, default=300)
parser.add_argument("--attacker-actor-lr", type=float, default=1e-4)
parser.add_argument("--attacker-critic-lr", type=float, default=1e-3)
parser.add_argument("--attacker-warmup-steps", type=int, default=2000)
parser.add_argument("--attacker-batch-size", type=int, default=128)
parser.add_argument("--attacker-buffer-capacity", type=int, default=20000)
parser.add_argument("--attacker-update-interval", type=int, default=1)
parser.add_argument("--attacker-reward-clip", type=float, default=5000.0)
parser.add_argument("--attacker-reward-scale", type=float, default=1000.0)
parser.add_argument("--per-step-energy-budget", type=float, default=2.0)
parser.add_argument("--alpha-tau", type=float, default=1.0)
parser.add_argument("--alpha-h", type=float, default=0.25)
parser.add_argument("--aoi-delta", type=int, default=1)
parser.add_argument("--h-delta", type=int, default=1)
parser.add_argument("--save-attacker-checkpoints", type=int, default=1, choices=[0,1])
parser.add_argument("--save-attacker-history", type=int, default=1, choices=[0,1])
```

## main 分支逻辑

```python
def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    if args.task == "victim":
        # 完全保留原有 victim 训练/评估逻辑
        ...
        return

    if args.task == "attacker":
        # learned attacker 训练逻辑
        learned_cfg = build_learned_attack_config_from_args(args)
        base_env, victim_agent, victim_algo, victim_meta = build_victim_for_attack(args, device)
        attack_env = build_learned_attack_env(base_env, victim_agent, victim_algo, learned_cfg)
        attacker_agent, log_msg = build_attack_agent(args, attack_env, device)
        train_cfg = build_attacker_train_config(args, learned_cfg)
        history = run_attacker_training(attack_env, attacker_agent, train_cfg)
        print(json.dumps(history, ensure_ascii=False, indent=2))
        return
```

## 默认 checkpoint 路径

若 `--victim-model-path` 为空，复用项目默认 best checkpoint 规则：

```text
results/checkpoints/{victim_algo_lower}_{scenario}_seed{seed}_best.pt
```

注意：这里是 victim checkpoint，不是 attacker checkpoint。

## 交给 Codex 的 prompt

```text
请只完成阶段 6：在 main.py 中接入 learned attacker 训练入口。

要求：
1. 新增 --task 参数，choices=["victim", "attacker"]，默认 victim。
2. task=victim 时保持原有 main.py 行为不变。
3. task=attacker 时执行 learned attacker 训练流程。
4. 添加 victim checkpoint 参数：--victim-algo、--victim-model-path。
5. 添加 attacker 训练参数：episodes、lr、warmup、batch、buffer、noise、reward scale/clip。
6. 添加 SDEC-QM 参数：per_step_energy_budget、alpha_tau、alpha_h、aoi_delta、h_delta。
7. 使用 learned_attack_config、workflow 中新增函数和 train.py 中 run_attacker_training。
8. victim_model_path 为空时按默认 best checkpoint 路径加载。
9. 打印训练结果 JSON。
10. 保持中文注释，不破坏原 victim 命令。
```

## 验收标准

```bash
python -m py_compile main.py
```

原 victim 命令仍可运行：

```bash
python main.py --task victim --algo DDPG --scenario base --mode train --seed 7 --episodes 1 --device cpu --save-checkpoints 0 --save-history 0 --save-plots 0
```

attacker 命令至少能解析：

```bash
python main.py --task attacker --victim-algo DDPG --scenario base --seed 7 --device cpu --attacker-episodes 1 --save-attacker-checkpoints 0 --save-attacker-history 0
```

如果没有 victim checkpoint，应明确报错，而不是随机训练。

---

# 阶段 7：补充 USAGE_EXAMPLE.md

## 目标

修改：

```text
USAGE_EXAMPLE.md
```

新增 learned attacker 训练命令说明。

## 必须新增章节

```markdown
## 4. Learned Attacker 训练命令
```

### 7.2 训练 learned attacker

```bash
python main.py --task attacker --victim-algo DDPG --scenario base --seed 7 --device cpu --victim-model-path results/checkpoints/ddpg_base_seed7_best.pt --attacker-episodes 300 --per-step-energy-budget 2.0 --alpha-tau 1.0 --alpha-h 0.25 --aoi-delta 1 --h-delta 1 --save-attacker-checkpoints 1 --save-attacker-history 1
```

### 7.3 快速 smoke test

```bash
python main.py --task attacker --victim-algo DDPG --scenario base --seed 7 --device cpu --victim-model-path results/checkpoints/ddpg_base_seed7_best.pt --attacker-episodes 1 --attacker-warmup-steps 10 --save-attacker-checkpoints 0 --save-attacker-history 0
```

## 交给 Codex 的 prompt

```text
请只完成阶段 7：补充 USAGE_EXAMPLE.md 的 learned attacker 使用说明。

要求：
1. 新增 Learned Attacker 训练章节。
2. 说明必须先有 victim checkpoint。

4. 给出 learned attacker 正式训练命令。

6. 明确 learned attacker 不使用 attack_prob / attack_ratio / max_consecutive_steps。
7. 不要改动已有 DQN/DDPG 训练命令含义。
```

---

# 阶段 8：可选增强任务

以下任务不属于第一版必须实现内容。只有当前 1-7 阶段全部跑通后再做。

## 8.2 增加 TD3 attacker

新增：

```text
AttackTD3Agent
```

双 critic 目标：

$$
y_t=r_t+\gamma\min_{j=1,2}Q_{\phi_j^-}(s_{t+1},\pi_{\theta^-}(s_{t+1}))
$$

## 8.3 增加 mapping 诊断报告

每个 episode 统计：

```text
avg_energy_used
avg_energy_ratio
avg_aoi_l0
avg_h_l0
avg_total_l0
avg_aoi_l1
avg_h_l1
constraint_violation_count
```

## 8.4 加入专家 warm start

如果已有单步启发式攻击，可以构造专家 intent action，加入行为克隆损失：

$$
\mathcal{L}_{BC}=\left\|\pi_{\theta}^{att}(s_t)-a_t^{expert}\right\|_2^2
$$

但第一版不要做，避免影响主链路跑通。

```text
实现要求：
1. 每个新增函数都写中文 docstring，说明作用、输入格式、输出格式、核心步骤。
2. 关键代码行写中文注释。
3. 不破坏已有 victim DQN/DDPG 训练和评估命令。
4. 若 victim checkpoint 不存在，必须明确报错。
5. 提供 smoke test 命令。
```
