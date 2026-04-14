# Experiment Design for UAgent-DrClaw

## Objective
Optimize ML input pipeline behavior by tuning:
- `batch_size`
- `num_workers`
- `prefetch_size`
- `cache_usage`
- `parallelism`

Metrics:
- throughput (samples/sec)
- approximate GPU utilization
- latency per batch (ms)

## Simulated pipeline stages
1. Data loading
2. Decode
3. Transform
4. Batch
5. Prefetch

The simulator models CPU preprocessing pressure and GPU idle conditions by comparing producer-side latency with GPU compute latency.

## Decision policy v1
- Increase `batch_size` as primary lever.
- If CPU bottleneck, increase `num_workers`.
- If GPU idle, increase `prefetch_size`.
- If a run fails (e.g., OOM), revert to best known configuration.

## Convergence
Stop when there is no improvement for `no_improvement_patience` consecutive iterations, or when `max_iterations` is reached.

## Reproducibility
- Config-driven run.
- Deterministic random seed in simulator.
- All iteration artifacts logged per run directory.
