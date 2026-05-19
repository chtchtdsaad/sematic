文件作用：按时间记录 learned meta-attacker 的执行进度、验证结果、错误和后续恢复点。

# Progress: Learned Meta-Attacker

## 2026-05-18

### Session Start

- 用户要求实施 learned meta-attacker 第一轮计划。
- 已按 `planning-with-files` 运行 session catchup。
- 当前分支为 `codex/metadrl`。
- 当前规划文件此前处于 deleted 状态，已重建为 learned attacker 专题版。

### Current Checklist

- [x] 重建 `task_plan.md`
- [x] 重建 `findings.md`
- [x] 重建 `progress.md`
- [x] 实现阶段 1：配置文件
- [x] 实现阶段 2：环境包装与 mapping
- [x] 实现阶段 3：AttackDDPGAgent
- [x] 实现阶段 4：训练循环
- [x] 实现阶段 5：workflow 构建
- [x] 实现阶段 6：main CLI
- [x] 实现阶段 7：Usage 文档和测试

### Verification Log

| Time | Command | Result |
| --- | --- | --- |
| 2026-05-18 | `python ... session-catchup.py` | exit 0 |
| 2026-05-18 | `git status --short --branch` | branch `codex/metadrl`; planning files deleted before rebuild |
| 2026-05-18 | `python -m py_compile attacks/learned_attack_config.py attacks/learned_attack_env.py attacks/attack_agent.py` | exit 0 |
| 2026-05-18 | `python -c "...AttackDDPGAgent..."` | exit 0；select_action 输出 shape=(24,) 且范围在 [-1,1] |
| 2026-05-18 | `python -m py_compile train.py workflow.py main.py attacks/learned_attack_config.py attacks/learned_attack_env.py attacks/attack_agent.py` | exit 0 |
| 2026-05-18 | `python -m py_compile attacks/learned_attack_config.py attacks/learned_attack_env.py attacks/attack_agent.py train.py workflow.py main.py tests/test_learned_meta_attacker.py` | exit 0 |
| 2026-05-18 | `python main.py --help` | exit 0 |
| 2026-05-18 | `python -m pytest tests/test_learned_meta_attacker.py -q` | 4 passed |
| 2026-05-18 | `python -m pytest -q` | 30 passed |
| 2026-05-18 | victim DDPG smoke，进程内设置 `KMP_DUPLICATE_LIB_OK=TRUE` | exit 0 |
| 2026-05-18 | DDPG victim attacker smoke，进程内设置 `KMP_DUPLICATE_LIB_OK=TRUE` | exit 0 |
| 2026-05-18 | DQN victim attacker smoke，进程内设置 `KMP_DUPLICATE_LIB_OK=TRUE` | exit 0 |
| 2026-05-18 | 缺失 victim checkpoint 验证 | exit 0；确认原命令非零并抛 `FileNotFoundError` |

### Error Log

| Time | Error | Attempt | Resolution |
| --- | --- | --- | --- |
| 2026-05-18 | `OMP: Error #15` | victim smoke 首次运行 | 仅在验证命令进程内设置 `KMP_DUPLICATE_LIB_OK=TRUE` 后通过 |
| 2026-05-18 | `git restore` 无法创建 `.git/index.lock` | 清理 tracked `.pyc` | 改用 `git checkout-index -f -- ...` 恢复 tracked pyc |

### Recovery Notes

- 如果会话中断，从 `task_plan.md` 的 Current Status 和 Phases 继续。
- 不要清理 `results/attack_reports/*.json`，除非用户单独确认。
- 新代码必须保持中文 docstring 和关键步骤中文注释。

## 2026-05-19

### Reward Clip Update

- 按用户要求删除 learned attacker 训练链路中的 reward scale。
- ReplayBuffer 现在写入 `clip(raw_reward, 0, reward_clip)`，不再除以 `reward_scale`。
- 删除 `--attacker-reward-scale` CLI 参数和相关 workflow/report 字段。
- 清理 Usage 和规划文件中的 scaled reward 描述。

### Learned Attacker Diagnostics Start

- 用户要求本轮不做 action flip，重点加入 deterministic eval、AoI/H 拆分统计和同 power random intent 对照。
- 已检查 `train.py::run_attacker_training`、`attacks/learned_attack_env.py`、`workflow.py::build_attacker_train_config`、`main.py` learned attacker CLI。
- 当前发现：mapping 已返回 `aoi_l0/h_l0`，但未返回 AoI/H energy；训练 report 只记录总能耗和总 L0；尚无 deterministic eval 或 random intent baseline。

### Learned Attacker Diagnostics Complete

- 已实现 `aoi_energy/h_energy` mapping 统计和训练 history 拆分字段。
- 已实现 deterministic eval：`--attacker-eval-every`、`--attacker-eval-episodes`。
- 已实现 random intent baseline：`--attacker-random-baseline`，与 learned attacker 共用同一 energy budget 和 mapping。
- 已实现 eval 前后环境快照恢复，避免 eval 消耗训练环境 RNG。
- 已更新 `USAGE_EXAMPLE.md` learned attacker 参数说明、正式命令、smoke 命令和新增报告字段。

### Verification Log

| Time | Command | Result |
| --- | --- | --- |
| 2026-05-19 | `conda activate rse_dq3; python -m pytest ...` | failed：未进入 rse_dq3，落到 `C:\Python314\python.exe` 且无 pytest |
| 2026-05-19 | `C:\Users\28681\anaconda3\envs\rse_dq3\python.exe -c "...test_intent_mapping..."` | red：`KeyError: 'aoi_energy'` |
| 2026-05-19 | `C:\Users\28681\anaconda3\envs\rse_dq3\python.exe -c "...test_attacker_training_reports..."` | red：`KeyError: 'aoi_l0_history'` |
| 2026-05-19 | `C:\Users\28681\anaconda3\envs\rse_dq3\python.exe -m py_compile attacks\learned_attack_config.py attacks\learned_attack_env.py train.py workflow.py main.py tests\test_learned_meta_attacker.py` | exit 0 |
| 2026-05-19 | direct call all functions in `tests/test_learned_meta_attacker.py` | passed |
| 2026-05-19 | DDPG victim attacker smoke with eval/random baseline, save off | exit 0 |
| 2026-05-19 | DQN victim attacker smoke with eval/random baseline, save off | exit 0 |

### Verification Log

| Time | Command | Result |
| --- | --- | --- |
| 2026-05-19 | `python -m pytest tests/test_learned_meta_attacker.py::test_attacker_reward_is_clipped_without_scaling -q` | 先失败后通过，确认 TDD red/green |
| 2026-05-19 | `python -m py_compile train.py workflow.py main.py attacks/attack_agent.py tests/test_learned_meta_attacker.py` | exit 0 |
| 2026-05-19 | `python -m pytest tests/test_learned_meta_attacker.py -q` | 5 passed |
| 2026-05-19 | `python -m pytest -q` | 31 passed |
