# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research codebase for remote state estimation, DRL scheduling, and semantic attack experiments. Core runtime modules live at the repository root: `main.py` is the CLI entry point, `train.py` contains training/evaluation loops, `workflow.py` assembles environments and agents, and `config.py` stores defaults and scenario presets. Attack-specific code is under `attacks/`, with shared helpers, random attacks, structural semantic attacks, learned attacker environments, and metrics. Tests live in `tests/`. Generated artifacts belong in `results/`, including `checkpoints/`, `histories/`, reports, and plots. Usage notes and experiment context are documented in `USAGE_EXAMPLE.md` and related Markdown files.

## Build, Test, and Development Commands

Activate the required conda environment before tests or local runs:

```powershell
conda activate rse_dq3
```

## Coding Style & Naming Conventions

Use Python 3 type hints where they clarify interfaces. Keep module, function, and variable names in `snake_case`; use `UPPER_CASE` for constants. For every function, start the docstring by explaining its purpose, input/output, and each parameter. For core algorithm steps, write concise Chinese comments explaining why the line or block exists. Prefer structured data and existing helpers over ad hoc parsing. Do not commit generated checkpoints, histories, or plots unless the change explicitly needs reproducible artifacts.

## Testing Guidelines

Tests use `pytest` and are named `tests/test_*.py`. Add focused regression tests beside the behavior being changed, especially for attack constraints, action metrics, environment transitions, and CLI integration. Before reporting test results, run them freshly inside `rse_dq3`.

## Commit & Pull Request Guidelines

Recent history uses short experiment-oriented subjects, often date-prefixed, such as `2026.5.18 before MDPattack`. Keep commits concise and scoped: mention the affected area and experiment or behavior. Pull requests should include the goal, key implementation changes, test command/output, affected result files, and screenshots only when plots or visual outputs change.

## Security & Configuration Tips

Keep machine-specific paths, credentials, and large private datasets out of git. Prefer configurable paths through CLI arguments or `config.py`. Validate checkpoint paths before long evaluations so missing files fail early.
