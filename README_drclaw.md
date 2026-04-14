# UAgent-DrClaw: Autonomous ML Pipeline Optimization Prototype

UAgent-DrClaw is a standalone, headless autonomous research loop for optimizing ML input pipeline parameters.

## What it does

- Reads experiment configuration from YAML.
- Plans parameter updates using a heuristic policy.
- Executes a pipeline simulation.
- Evaluates metrics and tracks best configuration.
- Iterates automatically with failure recovery.
- Writes reproducible run artifacts.

## Quick start

```bash
python -m uagent_drclaw.experiment_runner
```

Optional args:

```bash
python -m uagent_drclaw.experiment_runner \
  --config uagent_drclaw/experiments/config.yaml \
  --output-root uagent_drclaw/results
```

## Output artifacts
Each run generates a timestamped folder in `uagent_drclaw/results/` containing:
- `metrics.json` — per-iteration configs, metrics, score, and decisions.
- `logs.txt` — textual plan/execute/evaluate trace.
- `summary.md` — concise best-result summary.

## Architecture
See:
- `docs/drclaw_migration.md`
- `docs/drclaw_architecture.md`
- `docs/experiment_design.md`

## Extending DrClaw
- Replace planner with LLM/RL strategy while keeping planner interface.
- Replace simulator with real training input-pipeline benchmark.
- Add persistent memory backend (SQLite or vector store).
