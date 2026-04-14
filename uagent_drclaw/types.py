from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    batch_size: int = 32
    num_workers: int = 2
    prefetch_size: int = 2
    cache_usage: bool = False
    parallelism: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_size": self.prefetch_size,
            "cache_usage": self.cache_usage,
            "parallelism": self.parallelism,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            batch_size=int(data.get("batch_size", 32)),
            num_workers=int(data.get("num_workers", 2)),
            prefetch_size=int(data.get("prefetch_size", 2)),
            cache_usage=bool(data.get("cache_usage", False)),
            parallelism=int(data.get("parallelism", 1)),
        )


@dataclass
class ExperimentMetrics:
    throughput_samples_per_sec: float
    gpu_utilization: float
    latency_ms_per_batch: float
    cpu_bottleneck: bool
    gpu_idle: bool
    failed: bool = False
    failure_reason: str | None = None
    raw_stage_times_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class PlanDecision:
    candidate: PipelineConfig
    reason: str
    reverted: bool = False
