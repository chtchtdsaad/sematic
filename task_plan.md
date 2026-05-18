文件作用：记录结构化感知 mislead 与 expected-cost 攻击工作的阶段计划、当前完成状态、关键决策和遗留注意点，方便后续继续开发或审查。

# Task Plan: 结构化感知攻击与 Expected-Cost 基线整理

## Goal

在 `codex/attack` 分支上完成并固化结构化感知 mislead 攻击、expected-cost 攻击、结构性资源错配指标、CLI 接入、Usage 文档和测试验证。

## Current Phase

Phase 7: 已完成实现与验证，当前处于交付总结和计划文件维护阶段。

## Phases

### Phase 1: 结构化感知 Mislead 攻击

- [x] 支持 `semantic_aoi_mislead`
- [x] 支持 `semantic_h_mislead`
- [x] 支持 `semantic_joint_mislead`
- [x] DQN/DDPG 都可通过 `eval_attack.py --attack-mode` 调用
- [x] 攻击只修改 agent 看到的 `attacked_state`，不写回环境真实状态
- **Status:** complete

### Phase 2: 公共结构化工具抽取

- [x] 新增 `attacks/structural_common.py`
- [x] 抽取 sensor MSE、sensor risk、H 到成功率映射
- [x] 抽取 H flatten/unflatten 下标转换
- [x] 抽取 DQN/DDPG eval action 到 assignment 的统一解码
- [x] 抽取一步 expected-cost 估计
- [x] 重写 `attacks/structural_semantic.py`，让它只负责 mislead 攻击，公共逻辑复用 common
- **Status:** complete

### Phase 3: Expected-Cost 攻击

- [x] 新增 `attacks/structural_expected.py`
- [x] 支持 `semantic_aoi_expected_cost`
- [x] 支持 `semantic_h_expected_cost`
- [x] 支持 `semantic_joint_expected_cost`
- [x] 支持 `--expected-cost-mode mse`
- [x] 支持 `--expected-cost-mode sum_aoi`
- [x] DQN/DDPG 都支持
- **Status:** complete

### Phase 4: 结构性资源错配指标

- [x] 新增 `attacks/structural_metrics.py`
- [x] 计算 clean / attack selected risk sum
- [x] 计算 `resource_misallocation_score`
- [x] 计算 high/low risk scheduled ratio
- [x] 计算 high-risk good-channel ratio
- [x] 接入 `attacks/metrics.py` 汇总
- [x] 接入 `eval_attack.py` 报告字段
- **Status:** complete

### Phase 5: CLI 与文档

- [x] `eval_attack.py --attack-mode` 增加 mislead 和 expected-cost 模式
- [x] 新增 `--expected-cost-mode {mse,sum_aoi}`，默认 `mse`
- [x] `USAGE_EXAMPLE.md` 增加攻击参数说明
- [x] `USAGE_EXAMPLE.md` 增加 clean / attack 报告字段说明
- [x] `USAGE_EXAMPLE.md` 增加结构化感知 mislead 的 DQN/DDPG 命令
- [x] `USAGE_EXAMPLE.md` 增加 expected-cost 的 DQN/DDPG 命令
- **Status:** complete

### Phase 6: 测试与验证

- [x] 新增 `tests/test_structural_expected_and_metrics.py`
- [x] 更新 DDPG raw-state 动作选择测试
- [x] 更新 AttackConfig 字段测试
- [x] 更新 cooldown 行为测试
- [x] 更新 eval_attack 集成测试
- [x] 完整 pytest 通过
- [x] py_compile 通过
- [x] DQN/DDPG joint mislead smoke 通过
- [x] DQN/DDPG joint expected-cost smoke 通过
- **Status:** complete

### Phase 7: 分支整理与计划文件维护

- [x] 当前分支确认：`codex/attack`
- [x] 当前分支相对 `origin/codex/attack` ahead 2
- [x] 两个本地提交记录了本轮核心实现
- [x] 创建并维护 `task_plan.md`
- [x] 创建并维护 `findings.md`
- [x] 创建并维护 `progress.md`
- **Status:** complete

## Key Questions

1. 结构化感知 mislead 攻击是否还能用？
   - 是。`eval_attack.py` 仍支持 `semantic_aoi_mislead`、`semantic_h_mislead`、`semantic_joint_mislead`，并且 smoke 已验证 joint mislead 可运行。
2. `USAGE_EXAMPLE.md` 里结构化感知攻击为什么一度看不到？
   - 是文档整理时漏掉了专门章节，不是代码不可用。现在已补回 DQN/DDPG mislead 命令和含义说明。
3. DDPG 是否使用 `_normalize_state_for_ddpg`？
   - 不使用。`attacks/action_utils.py` 直接使用 eval_attack 传入的 raw state。
4. 当前是否已经提交？
   - 当前分支 `codex/attack` 已 ahead 2，最新提交包括 `5a9e7f2` 和 `c59fbf9`。但当前仍有 8 个 attack report JSON 处于 staged 状态，另有 2 个 joint expected-cost report JSON 未跟踪。

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 将公共结构化逻辑抽到 `structural_common.py` | 避免 mislead 和 expected-cost 各写一套 risk/H/action decode |
| `structural_semantic.py` 只负责 mislead 攻击 | 文件职责更清楚，后续调试更容易 |
| `structural_expected.py` 单独保存 expected-cost 攻击 | expected-cost 候选生成和 victim policy scoring 复杂度较高，需要隔离 |
| `structural_metrics.py` 单独保存解释指标 | 指标计算与攻击生成解耦，报告字段更清晰 |
| expected-cost 默认 `mse`，可选 `sum_aoi` | `mse` 对齐主评价指标，`sum_aoi` 作为轻量代理目标 |
| DDPG 使用 raw state | 用户明确要求，且测试已覆盖 |
| cooldown 逻辑实际生效 | 文档和 CLI 已暴露 `--cooldown-steps`，代码必须一致 |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Prompt 4-8 实现最初未完整落在 `codex/attack` | 1 | 在 `codex/attack` 上重新整理 common/expected/metrics 和测试 |
| `main` 上曾有不完整提交，import 了缺失模块 | 1 | 不直接合并 broken 状态，改为在 `codex/attack` 完整补齐 |
| `USAGE_EXAMPLE.md` 漏掉结构化感知 mislead 专门章节 | 1 | 已补回 DQN/DDPG mislead 命令和含义说明 |
| `git reset` / `git restore --staged` 曾遇到 `.git/index.lock` 权限问题 | 1 | 保留代码内容，后续由用户决定是否调整 staged 状态 |
| 验证生成 `.pyc` 和临时报告 | 1 | `.pyc` 已清理；当前仍有 8 个 attack report JSON staged，2 个 joint expected-cost report JSON untracked |

## Notes

- 当前 `codex/attack` 分支 ahead 2。
- 当前 staged 区域有 8 个 `results/attack_reports/*.json` 文件，另有 2 个 joint expected-cost report JSON 未跟踪，是否保留进提交需要用户决定。
- 当前代码实现和文档已经可用，结构化感知 mislead 并没有被废弃。
