### **代码规范部分：**

1.用中文在每个函数前后补充参数说明，函数作用，返回结果，以及简要的函数核心步骤
2.你每次写代码时，测试时保存的checkpoints、history、图片记得删除

3.用中文尽量对于每一行说明该行代码的作用，同时标清楚数据输入输出格式

在USAGE_EXAMPLE中补上使用案例

## 0. 总体目标

本次实现的攻击模式是：

\[
s_t = [\tau_t, H_t]
\]

攻击者生成：

\[
\tilde{s}_t = [\tilde{\tau}_t, \tilde{H}_t]
\]

然后 agent 使用 `attacked_state = \tilde{s}_t` 选择动作，但环境仍然根据真实状态和真实动力学转移。

需要完成：

1. 新增或更新 `eval_attack.py`。
2. 新建 `attacks/` 模块目录。
3. 实现 `clean / random_aoi / random_h / random_joint` 四种评估模式。
4. 对随机扰动加入：
   - 离散合法性约束；
   - 单维幅值约束；
   - 每步稀疏约束；
   - episode 级时间预算约束；
   - 最大连续攻击步约束；
   - 扰动统计与约束违规统计。
5. 输出 clean 与 attack 对比 JSON 报告。
6. 不保存 checkpoint / history / 图片。
7. 在 `USAGE_EXAMPLE.md` 中补充攻击评估命令。

---

## 1. 代码风格要求

请严格遵守当前项目代码风格：

1. 每个函数前写中文 docstring，说明：
   - 函数作用；
   - 输入格式；
   - 输出格式；
   - 核心步骤。
2. 关键代码行写中文注释。
3. 所有状态、动作、统计字段都要尽量标注 shape 或数据类型。
4. 测试过程中不要保存 checkpoint、history、图片。
5. 本任务只允许输出 JSON 报告到：

```text
results/attack_reports/
```

---

## 2. 最终文件结构

完成后项目新增结构如下：

```text
Remote_Estimation_RL/
├── eval_attack.py
└── attacks/
    ├── __init__.py
    ├── attack_config.py
    ├── state_ops.py
    ├── random_semantic.py
    ├── action_utils.py
    └── metrics.py
```

各文件职责：

```text
eval_attack.py
    只负责攻击评估流程，不堆攻击细节。

attacks/attack_config.py
    存放 AttackConfig dataclass，以及从 args 构造配置对象的函数。

attacks/state_ops.py
    存放状态拆分、合并、合法性投影函数。

attacks/random_semantic.py
    存放 random_aoi / random_h / random_joint 的约束随机扰动逻辑。

attacks/action_utils.py
    统一 DQN / DDPG 的评估动作选择接口。

attacks/metrics.py
    统计 attack_step、action_flip、扰动强度和约束违规。
```

---

## 3. 攻击约束定义

### 3.1 攻击对象约束

攻击者只能修改 agent 输入状态副本：

```python
attacked_state = attack_fn(state.copy())
```

禁止：

```python
env.aoi = ...
env.channel_state = ...
env.channel_loss = ...
```

也就是说：

- 真实环境状态不被攻击者改写；
- 真实信道丢包率不被攻击者改写；
- reward 不被攻击者改写；
- 只改变策略网络看到的观测。

---

### 3.2 AoI 合法性约束

AoI 扰动后必须满足：

\[
\tilde{\tau}_{i,t} \in \{1,2,\dots,\tau_{\max}\}
\]

实现必须使用：

```python
attacked_aoi = np.round(attacked_aoi)
attacked_aoi = np.clip(attacked_aoi, 1, env.max_aoi)
```

---

### 3.3 H 合法性约束

信道状态扰动后必须满足：

\[
\tilde{H}_{n,m,t} \in \{0,1,2,3,4\}
\]

实现必须使用：

```python
attacked_h = np.round(attacked_h)
attacked_h = np.clip(attacked_h, 0, 4)
```

---

### 3.4 单维幅值约束

AoI 单维扰动幅值：

\[
|\tilde{\tau}_{i,t} - \tau_{i,t}| \le d_\tau
\]

H 单维扰动幅值：

\[
|\tilde{H}_{n,m,t} - H_{n,m,t}| \le d_H
\]

对应参数：

```text
aoi_delta = d_tau
h_delta   = d_H
```

---

### 3.5 每步稀疏约束

每个 step 最多扰动若干个 AoI 维度：

\[
\|\tilde{\tau}_t - \tau_t\|_0 \le B_\tau
\]

每个 step 最多扰动若干个 H 维度：

\[
\|\tilde{H}_t - H_t\|_0 \le B_H
\]

每个 step 最多扰动总维度：

\[
\|\tilde{s}_t - s_t\|_0 \le B
\]

对应参数：

```text
max_aoi_features   = B_tau
max_h_features     = B_H
max_total_features = B
```

---

### 3.6 episode 级时间预算约束

整个 episode 中攻击步比例不超过：

\[
\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\tilde{s}_t \ne s_t] \le \rho
\]

对应参数：

```text
max_attack_ratio = rho
strict_budget    = 1 or 0
```

若 `strict_budget=1`，一旦本 episode 攻击步数达到：

```python
max_attack_steps = floor(episode_length * max_attack_ratio)
```

后续 step 必须停止攻击。

---

### 3.7 最大连续攻击约束

为避免攻击过于明显，连续攻击步数不得超过：

\[
N_{consecutive} \le C
\]

对应参数：

```text
max_consecutive_steps = C
cooldown_steps        = 冷却步数，默认可以先实现为 0
```

第一版可以只实现 `max_consecutive_steps`，`cooldown_steps` 保留字段但不做复杂逻辑。

---

## 4. 分阶段执行总览

请让 Codex 按以下阶段逐步执行：

```text
阶段 1：只实现 clean eval 骨架。
阶段 2：实现攻击配置和状态操作文件。
阶段 3：实现带约束的 random_aoi。
阶段 4：扩展带约束的 random_h / random_joint。
阶段 5：实现统一动作选择和统计模块。
阶段 6：接入完整 attack eval 与 JSON 输出。
阶段 7：补充 USAGE_EXAMPLE.md 使用命令。
```

每个阶段完成后必须能单独验收，不要一次性改完所有逻辑。

---

# 阶段 1：实现 clean eval 骨架

## 目标

新建 `eval_attack.py`，先只跑 clean eval，不实现任何攻击。

## 修改文件

```text
eval_attack.py
```

## eval_attack.py 只保留这些函数

```python
parse_args()
run_clean_eval(...)
save_attack_report(...)
main()
```

## 复用项目现有接口

必须复用：

```python
from workflow import build_env, build_agent, resolve_device, set_global_seed, build_train_config
from train import load_checkpoint, _normalize_state_for_ddpg
from utils import map_continuous_to_assignment
```

注意：若 `build_train_config` 在本阶段暂时不需要，可以不使用。

## DQN clean eval 规则

DQN 使用原始 state：

```python
action_id = agent.select_action(state, epsilon=0.0)
next_state, reward, done = env.step(action_id)
```

## DDPG clean eval 规则

DDPG 先归一化 state，再输出 continuous action，最后映射 assignment：

```python
state_for_agent = _normalize_state_for_ddpg(state, env)
virtual_action = agent.select_action(state_for_agent, noise_std=0.0)
assignment = map_continuous_to_assignment(virtual_action, env.n, env.m)
next_state, reward, done = env.step_assignment(assignment)
```

如果当前项目里的 DDPG action 接口函数名不同，请按照 `agent.py` 中真实接口适配，但不要改变 agent 本身。

## clean eval 统计

每个 episode 统计：

```text
mean_mse
mean_sum_aoi
```

说明：

- 若环境 step 返回 info 中已有 mse / sum_aoi，则优先使用 info。
- 若没有，则使用现有训练评估逻辑中的计算方式。
- 不要新建复杂计算逻辑，优先复用已有 run_evaluation 的做法。

## JSON 输出字段

第一阶段 JSON 至少包含：

```json
{
  "algo": "DQN",
  "scenario": "base",
  "seed": 42,
  "eval_episodes": 10,
  "attack_mode": "clean",
  "clean_mean_mse": 0.0,
  "clean_mean_sum_aoi": 0.0
}
```

保存路径：

```text
results/attack_reports/attack_eval_{algo}_{scenario}_seed{seed}_clean.json
```

## 交给 Codex 的 prompt

```text
请只完成阶段 1：实现 clean eval 骨架。

要求：
1. 新建 eval_attack.py。
2. 只实现 clean eval，不加入任何攻击逻辑。
3. eval_attack.py 只保留 parse_args、run_clean_eval、save_attack_report、main。
4. 复用 build_env、build_agent、resolve_device、set_global_seed、load_checkpoint、_normalize_state_for_ddpg、map_continuous_to_assignment。
5. DQN 使用 epsilon=0.0 的贪心动作。
6. DDPG 使用归一化 state，并使用 noise_std=0.0。
7. 输出 JSON 到 results/attack_reports/。
8. 不保存 checkpoint、history、图片。
9. 保持中文 docstring 和中文关键注释。
```

## 验收命令

```bash
python eval_attack.py --algo DQN --scenario base --mode eval --seed 42 --device cpu --eval-episodes 2 --attack-mode clean --save-report 1
```

如果当前 `eval_attack.py` 不设计 `--mode eval`，可以省略 `--mode`，但命令行参数必须清晰。

## 验收标准

- 程序能跑通。
- 能加载 checkpoint。
- 能输出 clean mean MSE / SumAoI。
- 能生成 JSON。
- 没有保存 checkpoint / history / 图片。

---

# 阶段 2：实现攻击配置与状态操作

## 目标

新建 `attacks/` 目录，并实现：

```text
attacks/__init__.py
attacks/attack_config.py
attacks/state_ops.py
```

暂时不要接入 `eval_attack.py` 主流程。

---

## 2.1 attacks/attack_config.py

实现 `AttackConfig` dataclass。

```python
from dataclasses import dataclass


@dataclass
class AttackConfig:
    """
    作用:
        保存状态观测扰动攻击的全部约束参数。

    输入格式:
        由 eval_attack.py 的 CLI 参数构造。

    输出格式:
        AttackConfig 实例。

    核心步骤:
        1. 记录攻击模式。
        2. 记录时间预算、幅值预算、稀疏预算。
        3. 记录 AoI/H 扰动方向。
    """

    mode: str = "clean"
    seed: int = 42

    attack_prob: float = 0.2
    strict_budget: bool = True
    max_attack_ratio: float = 0.2
    max_consecutive_steps: int = 5
    cooldown_steps: int = 0

    aoi_delta: int = 1
    h_delta: int = 1

    max_aoi_features: int = 1
    max_h_features: int = 1
    max_total_features: int = 2

    aoi_direction: str = "random"  # random / increase / decrease / mixed
    h_direction: str = "random"    # random / increase / decrease / mixed

    record_perturbation: bool = True
```

同时实现：

```python
def build_attack_config_from_args(args) -> AttackConfig:
    """
    作用:
        从 argparse.Namespace 构造 AttackConfig。
    """
```

---

## 2.2 attacks/state_ops.py

实现以下函数：

```python
def split_state(state, env):
    """
    作用:
        将状态拆分为 AoI 段和 H 段。

    输入:
        state: np.ndarray, shape=(N + N*M,)
        env: 环境对象，需包含 env.n 和 env.m

    输出:
        aoi_part: np.ndarray, shape=(N,)
        h_part: np.ndarray, shape=(N*M,)
    """
```

```python
def merge_state(aoi_part, h_part):
    """
    作用:
        将 AoI 段和 H 段合并为完整状态。

    输出:
        state: np.ndarray, shape=(N + N*M,), dtype=float32
    """
```

```python
def project_aoi_part(aoi_part, env):
    """
    作用:
        将 AoI 投影回合法离散范围 [1, env.max_aoi]。
    """
```

```python
def project_h_part(h_part):
    """
    作用:
        将 H 投影回合法离散范围 [0, 4]。
    """
```

要求：

- 所有输出统一为 `np.float32`。
- `split_state` 不允许原地修改输入。
- `merge_state` 返回新数组。

## 交给 Codex 的 prompt

```text
请只完成阶段 2：攻击配置和状态操作模块。

要求：
1. 新建 attacks/__init__.py、attacks/attack_config.py、attacks/state_ops.py。
2. attack_config.py 中实现 AttackConfig 和 build_attack_config_from_args(args)。
3. state_ops.py 中实现 split_state、merge_state、project_aoi_part、project_h_part。
4. 不要接入 eval_attack.py 主流程。
5. 不要实现 random attack。
6. 所有函数写中文 docstring 和中文关键注释。
7. 状态输出 dtype 使用 np.float32。
```

## 验收标准

可以写一个临时小测试，但不要提交测试文件：

```python
state2 = merge_state(*split_state(state, env))
assert state2.shape == state.shape
```

---

# 阶段 3：实现带约束 random_aoi

## 目标

新建：

```text
attacks/random_semantic.py
```

第一版只实现：

```text
clean
random_aoi
```

不实现 `random_h` 和 `random_joint`。

---

## 3.1 attack_state 运行时状态

在每个 episode 开始时，`eval_attack.py` 后续会创建：

```python
attack_state = {
    "attack_steps_used": 0,
    "max_attack_steps": int(np.floor(env.episode_length * config.max_attack_ratio)),
    "consecutive_attack_steps": 0,
}
```

阶段 3 中 `semantic_random_attack` 需要读取和更新这个字典。

---

## 3.2 random_bias_aoi 规则

实现：

```python
def random_bias_aoi(aoi_part, env, config, rng):
    """
    作用:
        对 AoI 段施加满足幅值、稀疏和合法性约束的随机扰动。

    输入格式:
        aoi_part: np.ndarray[float32], shape=(N,)
        env: 环境对象，需包含 env.max_aoi
        config: AttackConfig
        rng: np.random.Generator

    输出格式:
        attacked_aoi: np.ndarray[float32], shape=(N,)
        info: dict
    """
```

逻辑要求：

1. 复制 `aoi_part`，禁止原地修改。
2. 本步最多扰动：

```python
num_candidates = min(
    config.max_aoi_features,
    config.max_total_features,
    attacked_aoi.shape[0],
)
```

3. 从 AoI 下标中随机选择 `num_candidates` 个。
4. 扰动方向由 `config.aoi_direction` 控制：
   - `increase`：只加；
   - `decrease`：只减；
   - `mixed`：每个被选维度随机加或减；
   - `random`：在 `[-aoi_delta, aoi_delta]` 中随机取非零整数。
5. 扰动后必须调用 `project_aoi_part`。
6. 返回扰动统计：

```python
info = {
    "aoi_l0": int,
    "aoi_l1": float,
    "aoi_linf": float,
    "aoi_changed_indices": list[int],
}
```

---

## 3.3 semantic_random_attack 规则

实现：

```python
def semantic_random_attack(state, env, config, rng, attack_state):
    """
    作用:
        根据攻击配置生成 attacked_state，并返回扰动统计。

    输入格式:
        state: np.ndarray[float32], shape=(N + N*M,)
        env: 环境对象
        config: AttackConfig
        rng: np.random.Generator
        attack_state: dict，记录 episode 内攻击预算使用情况

    输出格式:
        attacked_state: np.ndarray[float32], shape=(N + N*M,)
        perturb_info: dict
    """
```

目前只支持：

```text
clean
random_aoi
```

默认空统计：

```python
{
    "is_attacked": False,
    "aoi_l0": 0,
    "h_l0": 0,
    "total_l0": 0,
    "aoi_l1": 0.0,
    "h_l1": 0.0,
    "aoi_linf": 0.0,
    "h_linf": 0.0,
    "aoi_changed_indices": [],
    "h_changed_indices": [],
}
```

攻击判断顺序：

1. 若 `mode == clean`，直接返回原始状态副本和空统计。
2. 根据 `attack_prob` 判断是否尝试攻击。
3. 若 `strict_budget=True` 且 `attack_steps_used >= max_attack_steps`，不攻击。
4. 若 `consecutive_attack_steps >= max_consecutive_steps`，不攻击，并将连续攻击计数重置为 0。
5. 若不攻击，返回原始状态副本和空统计。
6. 若攻击，调用 `random_bias_aoi`。
7. 合并状态并更新 `attack_state`。
8. 返回 `attacked_state` 和 `perturb_info`。

## 交给 Codex 的 prompt

```text
请只完成阶段 3：实现带约束 random_aoi。

要求：
1. 新建 attacks/random_semantic.py。
2. 实现 random_bias_aoi 和 semantic_random_attack。
3. semantic_random_attack 目前只支持 clean 和 random_aoi。
4. random_aoi 必须满足：
   - 只改 AoI 段；
   - 只改 state 副本；
   - 不修改 env.aoi、env.channel_state、env.channel_loss；
   - AoI round + clip 到 [1, env.max_aoi]；
   - 每步满足 max_aoi_features 和 max_total_features；
   - 单维扰动幅值不超过 aoi_delta；
   - episode 内满足 max_attack_ratio；
   - 连续攻击步不超过 max_consecutive_steps。
5. semantic_random_attack 返回 attacked_state 和 perturb_info。
6. 不要实现 random_h 和 random_joint。
7. 保持中文 docstring 和中文关键注释。
```

## 验收标准

- `random_aoi` 每次输出的 AoI 都在 `[1, env.max_aoi]`。
- `aoi_l0 <= max_aoi_features`。
- `total_l0 <= max_total_features`。
- `aoi_linf <= aoi_delta`。
- 不修改环境真实变量。

---

# 阶段 4：扩展 random_h 和 random_joint

## 目标

在 `attacks/random_semantic.py` 中继续实现：

```text
random_h
random_joint
```

---

## 4.1 random_bias_h 规则

实现：

```python
def random_bias_h(h_part, config, rng):
    """
    作用:
        对 H 段施加满足幅值、稀疏和合法性约束的随机扰动。

    输入格式:
        h_part: np.ndarray[float32], shape=(N*M,)
        config: AttackConfig
        rng: np.random.Generator

    输出格式:
        attacked_h: np.ndarray[float32], shape=(N*M,)
        info: dict
    """
```

逻辑要求：

1. 复制 `h_part`，禁止原地修改。
2. 本步最多扰动：

```python
num_candidates = min(
    config.max_h_features,
    config.max_total_features,
    attacked_h.shape[0],
)
```

3. 随机选择 H 下标。
4. 扰动方向由 `config.h_direction` 控制：
   - `increase`：只加；
   - `decrease`：只减；
   - `mixed`：随机加或减；
   - `random`：在 `[-h_delta, h_delta]` 中随机取非零整数。
5. 扰动后必须调用 `project_h_part`。
6. 返回统计：

```python
info = {
    "h_l0": int,
    "h_l1": float,
    "h_linf": float,
    "h_changed_indices": list[int],
}
```

---

## 4.2 random_joint 规则

`random_joint` 同时攻击 AoI 与 H，但必须满足总稀疏约束：

\[
aoi\_l0 + h\_l0 \le max\_total\_features
\]

建议实现方法：

1. 先计算本步可用总预算：

```python
budget_total = config.max_total_features
```

2. AoI 使用：

```python
aoi_budget = min(config.max_aoi_features, budget_total)
```

3. H 使用剩余预算：

```python
h_budget = min(config.max_h_features, budget_total - actual_aoi_l0)
```

4. 为避免大改结构，可以让 `random_bias_aoi` / `random_bias_h` 支持可选参数 `max_features_override`。

推荐函数签名更新为：

```python
def random_bias_aoi(aoi_part, env, config, rng, max_features_override=None):
    ...

def random_bias_h(h_part, config, rng, max_features_override=None):
    ...
```

## 交给 Codex 的 prompt

```text
请只完成阶段 4：扩展 random_h 和 random_joint。

要求：
1. 在 attacks/random_semantic.py 中新增 random_bias_h。
2. 扩展 semantic_random_attack，使其支持 clean、random_aoi、random_h、random_joint。
3. H 扰动必须 round + clip 到 [0,4]。
4. random_h 必须满足 max_h_features、max_total_features、h_delta。
5. random_joint 必须满足 total_l0 <= max_total_features。
6. 不能重复写大量 AoI 逻辑，尽量复用 random_bias_aoi 和 random_bias_h。
7. 仍然只能攻击 state 副本，不能修改环境真实变量。
8. 保持中文 docstring 和中文关键注释。
```

## 验收标准

- `random_h` 输出 H 全在 `[0,4]`。
- `h_l0 <= max_h_features`。
- `h_linf <= h_delta`。
- `random_joint` 满足 `total_l0 <= max_total_features`。

---

# 阶段 5：实现统一动作选择与统计模块

## 目标

新建：

```text
attacks/action_utils.py
attacks/metrics.py
```

---

## 5.1 attacks/action_utils.py

实现：

```python
def select_dqn_action(state, agent):
    """
    作用:
        DQN 评估阶段根据输入 state 选择贪心 action_id。

    输入:
        state: np.ndarray[float32], shape=(state_dim,)
        agent: DQNAgent

    输出:
        int, action_id
    """
```

```python
def select_ddpg_assignment(state, agent, env):
    """
    作用:
        DDPG 评估阶段根据输入 state 选择 assignment。

    输入:
        state: np.ndarray[float32], shape=(state_dim,)
        agent: DDPGAgent
        env: 环境对象

    输出:
        tuple[int, ...], assignment
    """
```

```python
def select_action_for_eval(algo, state, agent, env):
    """
    作用:
        统一 DQN/DDPG 评估动作选择。

    输出:
        DQN: int action_id
        DDPG: tuple[int, ...] assignment
    """
```

要求：

- DQN 使用 `epsilon=0.0`。
- DDPG 使用 `_normalize_state_for_ddpg` 和 `noise_std=0.0`。
- 如果 DDPG agent 的 `select_action` 参数不同，按项目真实接口适配。

---

## 5.2 attacks/metrics.py

实现统计初始化：

```python
def init_attack_stats():
    return {
        "total_steps": 0,
        "attack_step_count": 0,
        "action_flip_count": 0,
        "total_aoi_l0": 0,
        "total_h_l0": 0,
        "total_l0": 0,
        "total_aoi_l1": 0.0,
        "total_h_l1": 0.0,
        "max_aoi_linf": 0.0,
        "max_h_linf": 0.0,
        "constraint_violation_count": 0,
    }
```

实现扰动统计更新：

```python
def update_perturbation_stats(stats, perturb_info, config):
    """
    作用:
        根据当前 step 的 perturb_info 更新扰动统计，并检查约束是否违反。
    """
```

约束检查：

```python
if perturb_info["aoi_l0"] > config.max_aoi_features:
    violated = True
if perturb_info["h_l0"] > config.max_h_features:
    violated = True
if perturb_info["total_l0"] > config.max_total_features:
    violated = True
if perturb_info["aoi_linf"] > config.aoi_delta:
    violated = True
if perturb_info["h_linf"] > config.h_delta:
    violated = True
```

实现动作翻转统计：

```python
def update_action_flip_stats(stats, clean_action, attacked_action):
    """
    作用:
        比较 clean_action 和 attacked_action 是否不同。
    """
```

注意：

- DQN 比较 int。
- DDPG 比较 tuple。

实现汇总：

```python
def summarize_attack_stats(stats):
    """
    作用:
        将累计统计转换为比例和平均指标。
    """
```

summary 至少包含：

```json
{
  "attack_step_ratio": 0.0,
  "action_flip_ratio": 0.0,
  "avg_aoi_l0_per_step": 0.0,
  "avg_h_l0_per_step": 0.0,
  "avg_total_l0_per_step": 0.0,
  "avg_aoi_l1_per_attack": 0.0,
  "avg_h_l1_per_attack": 0.0,
  "max_aoi_linf": 0.0,
  "max_h_linf": 0.0,
  "constraint_violation_count": 0
}
```

## 交给 Codex 的 prompt

```text
请只完成阶段 5：实现统一动作选择和攻击统计模块。

要求：
1. 新建 attacks/action_utils.py 和 attacks/metrics.py。
2. action_utils.py 中实现 select_dqn_action、select_ddpg_assignment、select_action_for_eval。
3. DQN 使用 epsilon=0.0。
4. DDPG 使用 _normalize_state_for_ddpg、noise_std=0.0、map_continuous_to_assignment。
5. metrics.py 中实现 init_attack_stats、update_perturbation_stats、update_action_flip_stats、summarize_attack_stats。
6. 统计 attack_step_ratio、action_flip_ratio、avg_l0、avg_l1、max_linf、constraint_violation_count。
7. 不要改 eval_attack.py 主流程。
8. 保持中文 docstring 和中文关键注释。
```

## 验收标准

- DQN 动作比较不报错。
- DDPG assignment tuple 比较不报错。
- 统计字段完整。
- 约束违规能被计数。

---

# 阶段 6：接入完整 attack eval 与 JSON 输出

## 目标

修改 `eval_attack.py`，接入完整攻击流程。

---

## 6.1 eval_attack.py 新增 CLI 参数

在 `parse_args()` 中增加：

```python
parser.add_argument("--attack-mode", type=str, default="clean",
                    choices=["clean", "random_aoi", "random_h", "random_joint"])

parser.add_argument("--attack-prob", type=float, default=0.2)
parser.add_argument("--max-attack-ratio", type=float, default=0.2)
parser.add_argument("--strict-budget", type=int, default=1, choices=[0, 1])

parser.add_argument("--aoi-delta", type=int, default=1)
parser.add_argument("--h-delta", type=int, default=1)

parser.add_argument("--max-aoi-features", type=int, default=1)
parser.add_argument("--max-h-features", type=int, default=1)
parser.add_argument("--max-total-features", type=int, default=2)

parser.add_argument("--aoi-direction", type=str, default="random",
                    choices=["random", "increase", "decrease", "mixed"])

parser.add_argument("--h-direction", type=str, default="random",
                    choices=["random", "increase", "decrease", "mixed"])

parser.add_argument("--max-consecutive-steps", type=int, default=5)
parser.add_argument("--cooldown-steps", type=int, default=0)
parser.add_argument("--record-perturbation", type=int, default=1, choices=[0, 1])
parser.add_argument("--save-report", type=int, default=1, choices=[0, 1])
```

---

## 6.2 eval_attack.py 增加 run_attack_eval

实现：

```python
def run_attack_eval(algo, env, agent, attack_config, eval_episodes):
    """
    作用:
        在攻击状态观测下评估 agent 性能，并同时统计 clean_action 与 attacked_action 的差异。

    核心步骤:
        1. 每个 episode reset 环境。
        2. 每个 step 基于真实 state 生成 attacked_state。
        3. 分别计算 clean_action 和 attacked_action。
        4. 用 attacked_action 与真实环境交互。
        5. 更新 MSE、SumAoI、扰动统计、动作翻转统计。
    """
```

关键逻辑：

```python
clean_action = select_action_for_eval(algo, state, agent, env)

attacked_state, perturb_info = semantic_random_attack(
    state=state,
    env=env,
    config=attack_config,
    rng=rng,
    attack_state=attack_state,
)

attacked_action = select_action_for_eval(algo, attacked_state, agent, env)

update_perturbation_stats(stats, perturb_info, attack_config)
update_action_flip_stats(stats, clean_action, attacked_action)
```

环境执行：

```python
if algo == "DQN":
    next_state, reward, done = env.step(attacked_action)
else:
    next_state, reward, done = env.step_assignment(attacked_action)
```

注意：

- clean_action 只用于比较动作翻转，不用于环境执行。
- attacked_action 用于环境执行。
- attacked_state 只用于 agent 选动作，不进入环境内部。

---

## 6.3 JSON 报告字段

最终 JSON 至少包含：

```json
{
  "algo": "DQN",
  "scenario": "base",
  "seed": 42,
  "eval_episodes": 10,

  "attack_mode": "random_joint",
  "attack_prob": 0.2,
  "max_attack_ratio": 0.2,
  "strict_budget": true,
  "aoi_delta": 1,
  "h_delta": 1,
  "max_aoi_features": 1,
  "max_h_features": 2,
  "max_total_features": 3,
  "aoi_direction": "random",
  "h_direction": "random",
  "max_consecutive_steps": 5,

  "clean_mean_mse": 0.0,
  "attack_mean_mse": 0.0,
  "mse_degradation": 0.0,
  "mse_degradation_ratio": 0.0,

  "clean_mean_sum_aoi": 0.0,
  "attack_mean_sum_aoi": 0.0,
  "sum_aoi_degradation": 0.0,
  "sum_aoi_degradation_ratio": 0.0,

  "attack_step_ratio": 0.0,
  "action_flip_ratio": 0.0,
  "avg_aoi_l0_per_step": 0.0,
  "avg_h_l0_per_step": 0.0,
  "avg_total_l0_per_step": 0.0,
  "avg_aoi_l1_per_attack": 0.0,
  "avg_h_l1_per_attack": 0.0,
  "max_aoi_linf": 0.0,
  "max_h_linf": 0.0,
  "constraint_violation_count": 0
}
```

如果 `constraint_violation_count > 0`，终端必须打印 warning。

---

## 交给 Codex 的 prompt

```text
请只完成阶段 6：在 eval_attack.py 中接入完整攻击评估流程。

要求：
1. 在 parse_args 中加入攻击相关 CLI 参数。
2. 使用 build_attack_config_from_args(args) 构造 AttackConfig。
3. 新增 run_attack_eval。
4. clean_action 只用于动作翻转统计，attacked_action 用于真实环境执行。
5. attacked_state 只传给 agent 选动作，不得写回 env。
6. 攻击流程调用 semantic_random_attack。
7. 统计流程调用 metrics.py 中的函数。
8. 输出 clean / attack 对比 JSON。
9. JSON 中必须包含攻击配置、性能退化、扰动强度、动作翻转率、约束违规次数。
10. 若 constraint_violation_count > 0，在终端打印 warning。
11. 不保存 checkpoint、history、图片。
12. 保持中文 docstring 和中文关键注释。
```

## 验收命令

### DQN random_aoi

```bash
python eval_attack.py --algo DQN --scenario base --seed 42 --device cpu --eval-episodes 2 --attack-mode random_aoi --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --max-aoi-features 1 --max-total-features 1 --aoi-direction random --strict-budget 1
```

### DQN random_h

```bash
python eval_attack.py --algo DQN --scenario base --seed 42 --device cpu --eval-episodes 2 --attack-mode random_h --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 1 --max-h-features 2 --max-total-features 2 --h-direction random --strict-budget 1
```

### DQN random_joint

```bash
python eval_attack.py --algo DQN --scenario base --seed 42 --device cpu --eval-episodes 2 --attack-mode random_joint --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --h-delta 1 --max-aoi-features 1 --max-h-features 2 --max-total-features 3 --aoi-direction random --h-direction random --strict-budget 1
```

## 验收标准

- clean 和 attack 都能跑完。
- JSON 正常生成。
- `constraint_violation_count == 0`。
- `attack_step_ratio <= max_attack_ratio + 1e-6`，若 strict_budget=1。
- `max_aoi_linf <= aoi_delta`。
- `max_h_linf <= h_delta`。
- 没有保存 checkpoint / history / 图片。

---

# 阶段 7：补充 USAGE_EXAMPLE.md

## 目标

在 `USAGE_EXAMPLE.md` 中增加攻击评估命令模板。

## 交给 Codex 的 prompt

```text
请只完成阶段 7：更新 USAGE_EXAMPLE.md。

要求：
1. 新增“攻击评估命令模板”章节。
2. 添加 DQN random_aoi、DQN random_h、DQN random_joint、DDPG random_joint 四条命令。
3. 每条命令都使用约束参数：attack_prob、max_attack_ratio、aoi_delta、h_delta、max_aoi_features、max_h_features、max_total_features、strict_budget。
4. 说明本实验不会保存 checkpoint、history、图片，只保存 JSON 报告。
5. 保持原文档中文风格。
```

## 建议追加内容

```markdown
## 攻击评估命令模板

### DQN：约束 random_aoi

```bash
python eval_attack.py --algo DQN --scenario base --seed 42 --device cpu --eval-episodes 10 --attack-mode random_aoi --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --max-aoi-features 1 --max-total-features 1 --aoi-direction random --strict-budget 1
```

### DQN：约束 random_h

```bash
python eval_attack.py --algo DQN --scenario base --seed 42 --device cpu --eval-episodes 10 --attack-mode random_h --attack-prob 0.2 --max-attack-ratio 0.2 --h-delta 1 --max-h-features 2 --max-total-features 2 --h-direction random --strict-budget 1
```

### DQN：约束 random_joint

```bash
python eval_attack.py --algo DQN --scenario base --seed 42 --device cpu --eval-episodes 10 --attack-mode random_joint --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --h-delta 1 --max-aoi-features 1 --max-h-features 2 --max-total-features 3 --aoi-direction random --h-direction random --strict-budget 1
```

### DDPG：约束 random_joint

```bash
python eval_attack.py --algo DDPG --scenario base --seed 7 --device cpu --eval-episodes 10 --attack-mode random_joint --attack-prob 0.2 --max-attack-ratio 0.2 --aoi-delta 1 --h-delta 1 --max-aoi-features 1 --max-h-features 2 --max-total-features 3 --aoi-direction random --h-direction random --strict-budget 1
```

```
---

# 8. 最终检查清单

完成全部阶段后，请逐项检查：

```text
[ ] eval_attack.py 不包含具体攻击细节。
[ ] attacks/attack_config.py 只负责配置。
[ ] attacks/state_ops.py 只负责状态拆分、合并、投影。
[ ] attacks/random_semantic.py 只负责随机语义扰动。
[ ] attacks/action_utils.py 只负责动作选择。
[ ] attacks/metrics.py 只负责统计。
[ ] 攻击只改 attacked_state，不改 env 内部真实变量。
[ ] AoI 始终在 [1, env.max_aoi]。
[ ] H 始终在 [0,4]。
[ ] 每步扰动满足 max_aoi_features / max_h_features / max_total_features。
[ ] episode 攻击比例满足 max_attack_ratio。
[ ] constraint_violation_count == 0。
[ ] JSON 报告字段完整。
[ ] USAGE_EXAMPLE.md 已补充命令。
[ ] 没有保存 checkpoint、history、图片。
```

---

# 9. 后续扩展预留

本实验只实现约束随机扰动。后续可以在不改变 `eval_attack.py` 主流程的情况下继续增加：

```text
greedy_aoi_hide
    隐藏高 AoI 传感器风险。

greedy_h_fake_better
    将差的真实信道伪装成更好信道。

q_margin_timed_attack
    只在 Q 值间隔较小时攻击，提高攻击效率和隐蔽性。
```

这些后续攻击应继续复用：

```text
AttackConfig
state_ops
metrics
action_utils
```

不要破坏当前模块解耦结构。
