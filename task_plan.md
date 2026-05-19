文件作用：记录 learned meta-attacker 长周期任务的阶段计划、验收闸门、已锁定决策和当前执行状态。

# Task Plan: Learned Meta-Attacker 第一轮实现

## Goal

在当前 `Remote_Estimation_RL` 项目中新增能量约束 learned DDPG attacker。攻击者只篡改 victim scheduler 的输入观测，不修改真实环境状态；Actor 输出连续扰动意图，环境包装器通过状态依赖能量约束映射生成合法离散扰动；训练目标是提高真实下一步 MSE。

## Locked Decisions

- 第一轮只实现阶段 1-7，不实现 TD3、专家 warm start、完整 mapping 诊断报告。
- attacker 使用 raw state，不做归一化。
- victim 覆盖 DQN + DDPG；重点 smoke DDPG，DQN 做轻量可用性验证。
- ReplayBuffer 存 continuous intent action，不存最终离散 delta。
- ReplayBuffer 中 reward 只存 clip 后的训练值；history/report 记录 raw MSE。
- 执行节奏采用阶段闸门式：实现、验证、更新规划文件后再进入下一阶段。
- 不清理或提交现有 attack report JSON，除非用户单独确认。

## Current Status

Phase 8: complete，已补充 learned attacker 训练诊断：deterministic eval、AoI/H 拆分统计、同 power random intent 对照；明确不做 action flip 统计。

## Phases

### Phase 0: Planning Files

- [x] 运行 `session-catchup.py`
- [x] 检查当前 git 状态
- [x] 重建 `task_plan.md`
- [x] 重建 `findings.md`
- [x] 重建 `progress.md`

### Phase 1: Learned Attack Config

- [x] 新增 `attacks/learned_attack_config.py`
- [x] 实现 `LearnedAttackConfig`
- [x] 实现 `build_learned_attack_config_from_args(args)`
- [x] 验证 `python -m py_compile attacks/learned_attack_config.py`

### Phase 2: Learned Attack Env And SDEC-QM Mapping

- [x] 新增 `attacks/learned_attack_env.py`
- [x] 实现 `LearnedAttackEnv.reset`
- [x] 实现 `LearnedAttackEnv.step`
- [x] 实现 `map_intent_to_attacked_state`
- [x] 验证 py_compile；shape/clip/energy/env 写回将在单元测试中继续覆盖

### Phase 3: Attack DDPG Agent

- [x] 新增 `attacks/attack_agent.py`
- [x] 实现 `AttackActor`
- [x] 实现 `AttackCritic`
- [x] 实现 `AttackDDPGAgent`
- [x] 验证 select_action shape/range；update loss 将在单元测试中覆盖

### Phase 4: Training Loop

- [x] 在 `train.py` 新增 attacker checkpoint 函数
- [x] 在 `train.py` 新增 `run_attacker_training`
- [x] 训练过程使用 raw state 和 continuous intent action
- [x] history 记录 raw MSE、energy；ReplayBuffer 存 clipped reward

### Phase 5: Workflow Builders

- [x] 在 `workflow.py` 新增 `build_victim_for_attack`
- [x] 在 `workflow.py` 新增 `build_learned_attack_env`
- [x] 在 `workflow.py` 新增 `build_attack_agent`
- [x] 在 `workflow.py` 新增 `build_attacker_train_config`
- [x] victim checkpoint 缺失时明确报错
- [x] victim 参数冻结并切换 eval

### Phase 6: Main CLI

- [x] 在 `main.py` 新增 `--task {victim,attacker}`
- [x] `task=victim` 保持原训练/评估主链路
- [x] `task=attacker` 接入 learned attacker 训练
- [x] 缺失 checkpoint 明确报错

### Phase 7: Usage And Tests

- [x] 更新 `USAGE_EXAMPLE.md`
- [x] 新增 learned attacker 单元测试
- [x] 运行 py_compile
- [x] 运行 pytest
- [x] 运行 victim smoke
- [x] 运行 DDPG attacker smoke
- [x] 运行 DQN attacker smoke 或轻量验证

### Phase 8: Learned Attacker Diagnostics

- [x] 在 mapping info 中补充 `aoi_energy` / `h_energy`
- [x] 在 attacker 训练 history/report 中记录 AoI/H 拆分 L0 与能耗
- [x] 新增 deterministic eval 诊断，Actor 使用 `noise_std=0`
- [x] 新增 random intent baseline，对照使用同一 `per_step_energy_budget`、`alpha_tau`、`alpha_h`、`aoi_delta`、`h_delta`
- [x] deterministic eval 与 random baseline 不写入 ReplayBuffer，不更新网络
- [x] 评估前后恢复环境运行状态，避免诊断评估消耗训练环境 RNG
- [x] 更新 CLI、Usage 文档和单元测试

## Acceptance Commands

```powershell
python -m py_compile attacks/learned_attack_config.py attacks/learned_attack_env.py attacks/attack_agent.py train.py workflow.py main.py
python -m pytest -q
python main.py --task victim --algo DDPG --scenario base --mode train --seed 7 --episodes 1 --device cpu --save-checkpoints 0 --save-history 0 --save-plots 0
python main.py --task attacker --victim-algo DDPG --scenario base --seed 24 --device cpu --attacker-episodes 1 --attacker-warmup-steps 10 --save-attacker-checkpoints 0 --save-attacker-history 0
python main.py --task attacker --victim-algo DQN --scenario base --seed 24 --device cpu --attacker-episodes 1 --attacker-warmup-steps 10 --save-attacker-checkpoints 0 --save-attacker-history 0
python main.py --task attacker --victim-algo DDPG --scenario base --seed 24 --device cpu --attacker-episodes 1 --attacker-episode-length 20 --attacker-eval-every 1 --attacker-eval-episodes 1 --attacker-random-baseline 1 --save-attacker-checkpoints 0 --save-attacker-history 0 --save-attacker-report 0
```

## Errors Encountered

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-05-18 | `task_plan.md/findings.md/progress.md` 显示为 deleted | 1 | 按本任务重建专题规划文件 |
| 2026-05-18 | Windows OpenMP runtime 重复加载导致 smoke 首次失败 | 1 | 仅在 smoke 进程内设置 `KMP_DUPLICATE_LIB_OK=TRUE` 后验证通过，不写入代码 |
| 2026-05-18 | `git restore` 恢复 tracked `.pyc` 时 `.git/index.lock` 权限失败 | 2 | 使用 `git checkout-index -f -- ...` 恢复 tracked `.pyc` 验证产物 |
| 2026-05-19 | `conda activate rse_dq3; python -m pytest ...` 落到系统 Python 且无 pytest | 1 | 改用 `C:\Users\28681\anaconda3\envs\rse_dq3\python.exe` 直接调用测试函数；记录 pytest 环境缺失 |
