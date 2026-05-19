文件作用：保存 learned meta-attacker 实施过程中的代码事实、技术决策、接口约定和验证发现。

# Findings: Learned Meta-Attacker

## Repository Facts

- 当前分支：`codex/metadrl`。
- 现有 victim 入口：
  - `main.py` 解析训练/评估 CLI。
  - `workflow.py` 构建 env、agent、训练配置。
  - `train.py` 提供 `run_training`、`run_evaluation`、checkpoint save/load。
- 现有动作选择工具：
  - `attacks/action_utils.py::select_action_for_eval`
  - DQN 返回 action_id。
  - DDPG 直接使用 raw state，输出连续动作后由 `map_continuous_to_assignment` 转 assignment。
- 现有 checkpoint：
  - `results/checkpoints/ddpg_base_seed24_best.pt`
  - `results/checkpoints/dqn_base_seed24_best.pt`

## Design Decisions

- Learned attacker 的配置不复用 `attacks/attack_config.py`。
- Learned attacker 不使用 `attack_prob`、`max_attack_ratio`、`max_consecutive_steps`。
- SDEC-QM 映射第一版放在 `LearnedAttackEnv` 内部。
- 映射层只追踪 intent action 的离散投影，不调用 `env.step` 评估候选扰动效果。
- `attacked_state` 只传入 victim action selection，真实环境推进始终使用 `base_env` 当前真实状态。
- 训练 loop 使用 raw state；不得引入 `_normalize_state_for_ddpg` 或类似归一化函数。

## Interface Notes

- `LearnedAttackEnv.state_dim = base_env.state_dim`
- `LearnedAttackEnv.action_dim = state_dim`
- `intent_action.shape == (N + N*M,)`
- `delta.shape == (N + N*M,)`
- AoI 攻击后合法范围 `[1, env.max_aoi]`
- H 攻击后合法范围 `[0, 4]`
- `mapping_info` 至少包含：
  - `energy_used`
  - `energy_budget`
  - `aoi_l0`
  - `h_l0`
  - `total_l0`
  - `aoi_l1`
  - `h_l1`
  - `max_abs_delta`
  - `constraint_violation`

## Implemented Notes

- `attacks/learned_attack_config.py` 已新增，配置字段独立于随机攻击 `AttackConfig`。
- `attacks/learned_attack_env.py` 已新增，`LearnedAttackEnv` 内部维护 `_current_state`，避免从 attacked observation 回写真实环境。
- SDEC-QM 当前实现按 intent 距离改善和单位能耗收益选择每维整数候选。
- `attacks/attack_agent.py` 已新增，`AttackDDPGAgent` 使用 raw state 和 continuous intent action。
- `AttackDDPGAgent.update` 不调用 mapping，也不调用 env，只消费 ReplayBuffer。
- `train.py::run_attacker_training` 已新增，ReplayBuffer 写入 clipped reward，history 保留 raw MSE。
- `workflow.py::build_victim_for_attack` 已新增，会加载 checkpoint、冻结 victim 并切换 eval。
- `main.py` 已新增 `--task {victim,attacker}`；`task=victim` 时仍要求显式提供 `--algo`。
- 额外增加 `--attacker-episode-length`，用于把 smoke test 从 500 step 缩短到更快的验证长度。

## Current Risks

- `main.py --algo` 当前为 required；接入 `--task attacker` 时需要避免 attacker 命令被 `--algo` 阻塞。
- `run_attacker_training` 若默认跑完整 500 step，smoke 时间可能偏长；测试配置需要允许覆盖 `episode_length`。
- 当前工作区已有 `.pyc` 改动和 untracked pyc，属于验证产物；本任务不依赖它们。
- `LearnedAttackEnv.map_intent_to_attacked_state` 已通过 py_compile，仍需要单元测试覆盖无写回和能耗边界。
- `run_attacker_training` 已通过 py_compile，仍需要真实 checkpoint smoke 验证。
- 完整 `pytest -q` 结果为 30 passed。
- DDPG victim attacker smoke 已通过，使用 `ddpg_base_seed24_best.pt`。
- DQN victim attacker smoke 已通过，使用 `dqn_base_seed24_best.pt`。
- 缺失 victim checkpoint 会明确抛出 `FileNotFoundError`。

## Verification Notes

- Windows 本地 smoke 需要在进程级设置 `KMP_DUPLICATE_LIB_OK=TRUE` 才能绕过 OpenMP runtime 重复加载问题。
- 该环境变量没有写入代码，也没有写入用户环境，仅用于本次命令验证。
- victim smoke 会按现有 `main.py` 行为保存训练 report；本次生成的 `results/reports/report_DDPG_base_seed7_ep1.json` 已作为临时产物删除。

## 2026-05-19 Diagnostic Design Notes

- 用户明确要求本轮不处理 action flip，因为此前实验中动作翻转概率本身不高。
- deterministic eval 的目标是区分“训练探索噪声下的 episode 表现”和“当前 Actor 纯策略表现”，因此 eval 阶段必须使用 `noise_std=0`，且不能写入 ReplayBuffer。
- random intent baseline 的公平性来自复用同一个 `LearnedAttackEnv.map_intent_to_attacked_state`，即同一 `per_step_energy_budget`、`alpha_tau`、`alpha_h`、`aoi_delta`、`h_delta` 下比较 learned intent 与随机 intent。
- 为避免 eval 改变后续训练轨迹，评估前后需要保存并恢复 `base_env.rng`、`base_env._t`、`base_env.aoi`、`base_env.channel_state`、`base_env.channel_loss` 和 `LearnedAttackEnv._current_state`。
- AoI/H 拆分统计应同时包含 L0 与能耗：`aoi_l0`、`h_l0`、`aoi_energy`、`h_energy`，便于判断模型是否只是在某一类扰动上耗尽预算。

## 2026-05-19 Diagnostic Implementation Findings

- `LearnedAttackEnv.map_intent_to_attacked_state` 现在返回 `aoi_energy` 和 `h_energy`；`energy_used = aoi_energy + h_energy`。
- `run_attacker_training` 现在返回 `aoi_energy_history`、`h_energy_history`、`aoi_l0_history`、`h_l0_history`、`constraint_violation_count_history`。
- `deterministic_eval_history` 使用当前 attacker Actor 且 `noise_std=0`；它不写 ReplayBuffer，也不调用 `attacker_agent.update`。
- `random_intent_baseline_history` 使用 `np.random.uniform(-1,1)` 生成 intent，并复用同一个 `LearnedAttackEnv.step` 与 mapping，因此对照组 power 约束与 learned attacker 完全一致。
- 评估函数 `_run_attacker_policy_evaluation` 会调用 `snapshot_runtime_state` / `restore_runtime_state`，恢复 base env 的 `_t`、AoI、H、丢包率和 RNG 状态，避免 eval 改变后续训练轨迹。
- `rse_dq3` 环境当前没有 pytest 包；使用该环境 Python 直接调用测试函数完成了本轮 red/green 验证。
