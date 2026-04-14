from __future__ import annotations

from dataclasses import dataclass, field

from uagent_drclaw.types import ExperimentMetrics, PipelineConfig


@dataclass
class MemoryEntry:
    config: PipelineConfig
    metrics: ExperimentMetrics


@dataclass
class ExperimentMemory:
    history: list[MemoryEntry] = field(default_factory=list)
    best_config: PipelineConfig | None = None
    best_score: float = float("-inf")
    _bad_configs: set[tuple] = field(default_factory=set)
    _seen_configs: set[tuple] = field(default_factory=set)

    def record(self, config: PipelineConfig, metrics: ExperimentMetrics, score: float) -> None:
        self._seen_configs.add(self._signature(config))
        self.history.append(MemoryEntry(config=config, metrics=metrics))
        if metrics.failed:
            self._bad_configs.add(self._signature(config))
            return

        if score > self.best_score:
            self.best_score = score
            self.best_config = config

    def is_known_bad(self, config: PipelineConfig) -> bool:
        return self._signature(config) in self._bad_configs

    def has_seen(self, config: PipelineConfig) -> bool:
        return self._signature(config) in self._seen_configs

    @staticmethod
    def _signature(config: PipelineConfig) -> tuple:
        return (
            config.batch_size,
            config.num_workers,
            config.prefetch_size,
            config.cache_usage,
            config.parallelism,
        )
