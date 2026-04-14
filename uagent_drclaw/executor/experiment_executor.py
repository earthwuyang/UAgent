from __future__ import annotations

import traceback

from uagent_drclaw.executor.pipeline_simulator import PipelineSimulator
from uagent_drclaw.types import ExperimentMetrics, PipelineConfig


class ExperimentExecutor:
    def __init__(self, simulator: PipelineSimulator | None = None):
        self.simulator = simulator or PipelineSimulator()

    def execute(self, config: PipelineConfig) -> ExperimentMetrics:
        try:
            return self.simulator.run(config)
        except Exception as exc:  # Failure recovery path
            return ExperimentMetrics(
                throughput_samples_per_sec=0,
                gpu_utilization=0,
                latency_ms_per_batch=0,
                cpu_bottleneck=True,
                gpu_idle=True,
                failed=True,
                failure_reason=f"ExecutorError: {exc}\n{traceback.format_exc()}"[:1000],
            )
