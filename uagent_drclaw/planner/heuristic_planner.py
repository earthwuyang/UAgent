from __future__ import annotations

from dataclasses import replace

from uagent_drclaw.memory.experiment_memory import ExperimentMemory
from uagent_drclaw.types import ExperimentMetrics, PipelineConfig, PlanDecision


class HeuristicPlanner:
    """Rule-based planning policy designed for later replacement by learned policies."""

    def __init__(
        self,
        max_batch_size: int,
        max_workers: int,
        max_prefetch: int,
        max_parallelism: int,
    ):
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.max_prefetch = max_prefetch
        self.max_parallelism = max_parallelism

    def propose(
        self,
        current: PipelineConfig,
        last_metrics: ExperimentMetrics | None,
        memory: ExperimentMemory,
    ) -> PlanDecision:
        candidate = current
        reason = "No-op fallback to preserve stability"

        if last_metrics and last_metrics.failed:
            if "OutOfMemory" in (last_metrics.failure_reason or "") and current.batch_size > 1:
                reduced = replace(current, batch_size=max(1, current.batch_size // 2))
                if not memory.is_known_bad(reduced):
                    return PlanDecision(
                        candidate=reduced,
                        reason="OOM detected; reducing batch_size for recovery",
                        reverted=True,
                    )
            candidate = memory.best_config or current
            reason = f"Last run failed ({last_metrics.failure_reason}); reverting to known-good config"
            return PlanDecision(candidate=candidate, reason=reason, reverted=True)

        proposals: list[tuple[PipelineConfig, str]] = []

        if last_metrics is None:
            proposals.append(
                (
                    replace(current, batch_size=min(current.batch_size * 2, self.max_batch_size)),
                    "Bootstrap: increase batch_size to probe throughput ceiling",
                )
            )
        else:
            if last_metrics.cpu_bottleneck and current.num_workers < self.max_workers:
                proposals.append(
                    (
                        replace(current, num_workers=min(current.num_workers + 1, self.max_workers)),
                        "CPU bottleneck detected; increasing num_workers",
                    )
                )
            if last_metrics.gpu_idle and current.prefetch_size < self.max_prefetch:
                proposals.append(
                    (
                        replace(current, prefetch_size=min(current.prefetch_size + 1, self.max_prefetch)),
                        "GPU idle detected; increasing prefetch_size",
                    )
                )

        if current.batch_size < self.max_batch_size:
            proposals.append(
                (
                    replace(current, batch_size=min(current.batch_size * 2, self.max_batch_size)),
                    "Trying larger batch_size for better device occupancy",
                )
            )
        if current.parallelism < self.max_parallelism:
            proposals.append(
                (
                    replace(current, parallelism=min(current.parallelism + 1, self.max_parallelism)),
                    "Exploring higher transform parallelism",
                )
            )
        proposals.append((replace(current, cache_usage=not current.cache_usage), "Exploring cache toggle"))

        for proposal, proposal_reason in proposals:
            if not memory.is_known_bad(proposal) and not memory.has_seen(proposal):
                candidate = proposal
                reason = proposal_reason
                break

        if memory.is_known_bad(candidate):
            fallback = memory.best_config or current
            return PlanDecision(
                candidate=fallback,
                reason="Proposed configuration was previously bad; reverting to best known config",
                reverted=True,
            )

        return PlanDecision(candidate=candidate, reason=reason)
