文件作用：保存结构化攻击整理过程中的需求、技术发现、实现内容、验证结论和当前仓库状态，作为后续恢复上下文的知识库。

# Findings & Decisions

## Requirements

- 在 `codex/attack` 分支整理并保留本轮结构化攻击实现。
- 支持 DQN 和 DDPG。
- DDPG 不使用 `_normalize_state_for_ddpg`。
- 核心代码需要中文注释。
- 每个函数开头需要说明功能、输入输出和参数含义。
- `USAGE_EXAMPLE.md` 需要包含结构化感知 mislead 和 expected-cost 的命令说明。
- 使用 `planning-with-files` 技能维护 `task_plan.md`、`findings.md`、`progress.md`。

## Research Findings

- 当前分支：`codex/attack`。
- 当前本地提交：
  - `5a9e7f2 2026.5.13 one-step`
  - `c59fbf9 2026.5.13 one-step`
- 当前分支相对 `origin/codex/attack`：ahead 2。
- `eval_attack.py` 当前支持的结构化攻击模式：
  - `semantic_aoi_mislead`
  - `semantic_h_mislead`
  - `semantic_joint_mislead`
  - `semantic_aoi_expected_cost`
  - `semantic_h_expected_cost`
  - `semantic_joint_expected_cost`
- `USAGE_EXAMPLE.md` 当前已经补回：
  - `5.5 DQN：结构化感知 mislead 攻击`
  - `5.6 DDPG：结构化感知 mislead 攻击`
  - `5.7 结构化感知 mislead 的含义`
  - `5.8 DQN：结构化 expected-cost 攻击`
  - `5.9 DDPG：结构化 expected-cost 攻击`
  - `5.10 expected-cost 目标切换`

## Implemented Files

| File | Purpose |
|------|---------|
| `attacks/structural_common.py` | 公共结构化工具：MSE/risk、H 成功率、link priority、H 下标转换、action decode、expected-cost 估计 |
| `attacks/structural_semantic.py` | AoI/H/Joint 结构化感知 mislead 攻击 |
| `attacks/structural_expected.py` | AoI/H/Joint expected-cost 攻击 |
| `attacks/structural_metrics.py` | 结构性资源错配指标 |
| `eval_attack.py` | CLI attack-mode 分发、expected-cost 分发、结构指标统计和报告 |
| `attacks/action_utils.py` | DQN/DDPG eval 动作选择；DDPG 使用 raw state |
| `attacks/attack_config.py` | 新增 `expected_cost_mode` |
| `attacks/metrics.py` | 汇总结构性指标 |
| `attacks/random_semantic.py` | cooldown 实际逻辑 |
| `tests/test_structural_expected_and_metrics.py` | expected-cost 与结构指标测试 |
| `USAGE_EXAMPLE.md` | 攻击参数、报告字段、mislead/expected-cost 示例命令 |

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| `structural_common.py` 作为公共层 | 避免公共 risk/action decode/expected-cost 逻辑重复 |
| `structural_semantic.py` 只做 mislead | 结构化感知攻击与 expected-cost 攻击职责分开 |
| `structural_expected.py` 做有限候选 expected-cost | 避免大状态空间全枚举，同时提供更强 baseline |
| `structural_metrics.py` 做解释指标 | 指标独立于攻击生成，便于报告和论文分析 |
| H 越大表示丢包率越低 | 因此 H 降低表示伪造更差信道，H 提高表示伪造更好诱饵信道 |
| expected-cost 默认 `mse` | 与主性能指标一致 |
| DDPG 不 normalize | 用户明确要求，测试已覆盖 |

## Verification Findings

- 完整测试通过：`python -m pytest -q` -> `26 passed`。
- 语法检查通过：`python -m py_compile ...`。
- smoke 已通过：
  - DQN `semantic_joint_expected_cost`
  - DQN `semantic_joint_mislead`
  - DDPG `semantic_joint_expected_cost`
  - DDPG `semantic_joint_mislead`
- 文档检查通过：
  - `USAGE_EXAMPLE.md` 包含结构化感知 mislead 的 DQN/DDPG 命令。
  - `USAGE_EXAMPLE.md` 包含 expected-cost 的 DQN/DDPG 命令。

## Current Git State

- Branch: `codex/attack`
- Ahead: 2 commits ahead of `origin/codex/attack`
- Current staged files:
  - 8 个 staged 的 `results/attack_reports/attack_eval_*_semantic_*_*.json`
  - 2 个 untracked 的 joint expected-cost report JSON
- Important note:
  - 这些 JSON 是攻击评估报告产物，不是源码。是否纳入最终提交需要用户决定。

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 结构化感知攻击文档章节缺失 | 已补回 mislead 命令和含义说明 |
| Prompt 4-8 曾未完整落到 `codex/attack` | 已在 `codex/attack` 整理完整实现 |
| DDPG normalize 旧描述残留风险 | 已改为 raw state，并有测试覆盖 |
| cooldown 只有参数但逻辑不一致 | 已在 `_can_attack()` 中实现并测试 |
| `.pyc` 验证产物污染 status | 已清理 |

## Follow-Up Notes

- 下一步建议先决定 8 个 staged report JSON 和 2 个 untracked joint expected-cost report JSON 是否保留。
- 若要提交源码，建议单独提交源码/测试/文档，报告 JSON 另行决定是否作为实验产物归档。
- H-only 攻击在 smoke 中退化通常小于 AoI/Joint，后续实验需要重点分析 action flip 和 resource misallocation 是否解释充分。
