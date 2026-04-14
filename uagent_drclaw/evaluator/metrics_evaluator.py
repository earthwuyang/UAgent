from __future__ import annotations

from uagent_drclaw.types import ExperimentMetrics


class MetricsEvaluator:
    """Weighted scoring for pipeline optimization objective."""

    def __init__(self, throughput_weight: float = 1.0, gpu_weight: float = 0.25, latency_weight: float = 0.2):
        self.throughput_weight = throughput_weight
        self.gpu_weight = gpu_weight
        self.latency_weight = latency_weight

    def score(self, metrics: ExperimentMetrics) -> float:
        if metrics.failed:
            return float("-inf")

        return (
            self.throughput_weight * metrics.throughput_samples_per_sec
            + self.gpu_weight * metrics.gpu_utilization
            - self.latency_weight * metrics.latency_ms_per_batch
        )

    def improved(self, current: float, best: float) -> bool:
        return current > best
