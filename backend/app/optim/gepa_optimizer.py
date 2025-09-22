"""Thin wrapper around DSPy's GEPA optimizer for host-mode execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple

try:  # pragma: no cover - exercised when dependency is present
    import dspy
    from dspy import Module, Example  # type: ignore
    DSPY_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful degradation when dspy not installed yet
    dspy = None  # type: ignore
    Module = object  # type: ignore
    DSPY_AVAILABLE = False


@dataclass
class GEPAConfig:
    """Configuration for GEPA prompt/program optimisation."""

    max_metric_calls: int = 150
    reflection_lm: str = "openai/gpt-5"
    task_lm: str = "openai/gpt-4.1-mini"
    min_delta: float = 0.02  # Accept changes if validation gain >= 2%


class GEPAOptimizer:
    """Host-mode wrapper over :class:`dspy.GEPA` with defensive guards."""

    def __init__(self, cfg: GEPAConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        if not DSPY_AVAILABLE:
            raise ImportError(
                "dspy-ai is required for GEPA optimisation. Install dspy-ai>=2.6.0 and gepa>=0.2.0"
            )

    def _configure_lms(self, lm_conf: Dict[str, Any]) -> None:
        """Configure the DSPy global settings for the optimisation run."""

        lm_name = lm_conf.get("lm", self.cfg.task_lm)
        prompt_lm = lm_conf.get("prompt_lm", self.cfg.reflection_lm)
        dspy.settings.configure(lm=lm_name, prompt_lm=prompt_lm)  # type: ignore[attr-defined]

    def optimize_program(
        self,
        program_ctor: Callable[[], Module],
        trainset: Iterable[Any],
        valset: Iterable[Any],
        metric_fn: Callable[[Any, Any], float],
        lm_conf: Dict[str, Any],
    ) -> Tuple[Module, Dict[str, Any]]:
        """Compile a program using GEPA and decide whether to adopt the result."""

        self._configure_lms(lm_conf)
        program = program_ctor()

        trainset_list = list(trainset)
        valset_list = list(valset)

        if not trainset_list or not valset_list:
            self.logger.debug("[GEPA] Skipping optimisation (train/val set empty)")
            return program, {"accepted": False, "reason": "empty_dataset"}

        def _prepare(data: Iterable[Any]) -> Iterable[Any]:
            prepared = []
            for item in data:
                if isinstance(item, dict):
                    example = Example(question=item.get("question", ""))
                    example.prompt_used = item.get("prompt")
                    example.target_score = item.get("score", 0.0)
                    example.generated_ideas = item.get("ideas", [])
                    prepared.append(example)
                else:
                    prepared.append(item)
            return prepared

        trainset_prepared = list(_prepare(trainset_list))
        valset_prepared = list(_prepare(valset_list))

        gepa = dspy.GEPA(  # type: ignore[attr-defined]
            max_metric_calls=self.cfg.max_metric_calls,
            reflection_lm=self.cfg.reflection_lm,
            task_lm=self.cfg.task_lm,
        )

        compiled = gepa.compile(
            program=program,
            trainset=trainset_prepared,
            valset=valset_prepared,
            metric=metric_fn,
        )

        val_gain = float(getattr(compiled, "val_gain", 0.0))
        accepted = val_gain >= self.cfg.min_delta
        info = {"accepted": accepted, "val_gain": val_gain}
        if accepted:
            self.logger.info("[GEPA] Accepted compiled program with validation gain %.4f", val_gain)
            return compiled, info

        self.logger.info(
            "[GEPA] Rejected compiled program (gain %.4f < min_delta %.4f)",
            val_gain,
            self.cfg.min_delta,
        )
        return program, info
