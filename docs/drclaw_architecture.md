# UAgent-DrClaw Architecture

## Module layout

```text
uagent_drclaw/
  planner/      # Proposes next configuration changes (policy layer)
  executor/     # Applies config and executes pipeline simulation
  evaluator/    # Converts metrics to comparable objective scores
  memory/       # Stores history, best config, and bad-config blacklist
  experiments/  # Config files and experiment definitions
  results/      # Per-run outputs
  experiment_runner.py  # End-to-end autonomous loop entrypoint
```

## Interfaces

### `planner`
- Input: current config, previous metrics, memory snapshot.
- Output: `PlanDecision(candidate_config, reason, reverted)`.
- Responsibility: propose next experiment with reversible, explainable changes.

### `executor`
- Input: candidate `PipelineConfig`.
- Output: `ExperimentMetrics` (or failed metrics with reason).
- Responsibility: run pipeline and surface bottlenecks + KPI measurements.

### `evaluator`
- Input: `ExperimentMetrics`.
- Output: scalar score for improvement checks.
- Responsibility: objective shaping (throughput + utilization - latency penalty).

### `memory`
- Input: `(config, metrics, score)` from each iteration.
- Output: current best config/score and bad-config lookup.
- Responsibility: persist optimization trajectory and prevent repeated poor attempts.

### `experiments`
- Contains declarative config (`config.yaml`) for search bounds, initial config, and loop controls.

## Control flow

1. Load experiment config.
2. Initialize planner/executor/evaluator/memory.
3. Iterate until convergence criterion:
   - Plan next config.
   - Execute experiment.
   - Evaluate metrics into score.
   - Update memory.
   - Revert or continue based on score/failure history.
4. Write reproducible artifacts (`metrics.json`, `logs.txt`, `summary.md`).

## Failure recovery strategy

- Executor catches runtime errors and emits failed metrics.
- Planner reacts to failed runs by reverting to `best_config`.
- Memory marks failed configs as bad to avoid repeating.

## Extension points

- Replace `HeuristicPlanner` with LLM planner or RL agent.
- Swap `PipelineSimulator` with real benchmark harness.
- Extend evaluator with multi-objective Pareto scoring.
- Persist memory into SQLite for long-horizon optimization.
