"""Meta-optimizer facade coordinating GEPA with runtime heuristics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from .gepa_optimizer import GEPAOptimizer

logger = logging.getLogger(__name__)


@dataclass
class MetaOptConfig:
    """High-level knobs for triggering GEPA optimisation."""

    trigger_threshold: float = 0.50
    enabled: bool = True


class MetaOptimizer:
    """Simple policy for invoking GEPA when recent rewards are low."""

    def __init__(self, gepa: Optional[GEPAOptimizer], cfg: MetaOptConfig):
        self.gepa = gepa
        self.cfg = cfg

    def maybe_improve_program(
        self,
        last_reward: float,
        program_ctor,
        trainset: Iterable[Any],
        valset: Iterable[Any],
        metric_fn,
        lm_conf: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Return a compiled program + info if GEPA ran and improvements accepted."""

        if not self.cfg.enabled:
            logger.debug("[GEPA] Meta-optimizer disabled via configuration")
            return None, None
        if not self.gepa:
            logger.debug("[GEPA] Optimizer unavailable (dspy/gepa not installed)")
            return None, None
        if last_reward >= self.cfg.trigger_threshold:
            logger.debug(
                "[GEPA] Skipping optimisation (last_reward %.3f >= threshold %.3f)",
                last_reward,
                self.cfg.trigger_threshold,
            )
            return None, None

        logger.info(
            "[GEPA] Triggered optimisation (last_reward %.3f < threshold %.3f)",
            last_reward,
            self.cfg.trigger_threshold,
        )
        program, info = self.gepa.optimize_program(program_ctor, trainset, valset, metric_fn, lm_conf)
        return program, info
