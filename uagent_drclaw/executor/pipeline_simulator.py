from __future__ import annotations

import random

from uagent_drclaw.types import ExperimentMetrics, PipelineConfig


class PipelineSimulator:
    """Deterministic-ish simulator for ML input pipeline behavior."""

    def __init__(self, seed: int = 7):
        self.seed = seed

    def run(self, config: PipelineConfig) -> ExperimentMetrics:
        if config.batch_size > 1024:
            return ExperimentMetrics(
                throughput_samples_per_sec=0,
                gpu_utilization=0,
                latency_ms_per_batch=0,
                cpu_bottleneck=True,
                gpu_idle=True,
                failed=True,
                failure_reason="OutOfMemory: batch_size exceeded simulated GPU memory budget",
            )

        # Stage model (ms)
        load_ms = max(1.0, 8.0 - (1.5 if config.cache_usage else 0.0))
        decode_ms = max(1.0, 9.5 / max(config.num_workers, 1))
        transform_ms = max(1.0, 12.0 / max(config.parallelism, 1))
        batch_ms = max(1.0, 4.0 + config.batch_size / 100.0)
        prefetch_gain = 1.0 + (config.prefetch_size * 0.08)

        cpu_stage_ms = load_ms + decode_ms + transform_ms
        producer_ms = (cpu_stage_ms + batch_ms) / prefetch_gain

        # Simulated GPU consumption time per batch
        gpu_compute_ms = max(2.5, 6.5 + (config.batch_size / 85.0))
        latency_ms = max(producer_ms, gpu_compute_ms)

        throughput = config.batch_size / (latency_ms / 1000.0)
        gpu_utilization = min(99.0, (gpu_compute_ms / latency_ms) * 100.0)
        signature = (
            config.batch_size * 31
            + config.num_workers * 17
            + config.prefetch_size * 13
            + int(config.cache_usage) * 7
            + config.parallelism * 19
        )
        local_rng = random.Random(self.seed + signature)
        jitter = local_rng.uniform(-0.03, 0.03)
        throughput *= (1 + jitter)

        cpu_bottleneck = producer_ms > gpu_compute_ms * 1.10
        gpu_idle = gpu_utilization < 68.0

        return ExperimentMetrics(
            throughput_samples_per_sec=round(throughput, 3),
            gpu_utilization=round(gpu_utilization, 2),
            latency_ms_per_batch=round(latency_ms, 3),
            cpu_bottleneck=cpu_bottleneck,
            gpu_idle=gpu_idle,
            raw_stage_times_ms={
                "data_loading": round(load_ms, 3),
                "decode": round(decode_ms, 3),
                "transform": round(transform_ms, 3),
                "batch": round(batch_ms, 3),
                "prefetch_effective_divisor": round(prefetch_gain, 3),
                "producer_total": round(producer_ms, 3),
                "gpu_compute": round(gpu_compute_ms, 3),
            },
        )
