"""Debate triggering policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DebatePolicy:
    """Policy thresholds controlling when multi-agent debate should run."""

    min_confidence: float = 0.65
    stakes: str = "high"
    max_cost_usd: float = 1.0


def should_debate(node_signals: Dict[str, Any], policy: DebatePolicy) -> bool:
    """Return True if debate should be triggered for a node given signals."""

    confidence = float(node_signals.get("confidence", 0.0))
    if confidence < policy.min_confidence:
        return True

    stakes = node_signals.get("stakes")
    if stakes and str(stakes).lower() == policy.stakes.lower():
        return True

    disagreement = float(node_signals.get("disagreement_score", 0.0))
    if disagreement >= 0.4:
        return True

    rl_uncertainty = float(node_signals.get("rl_uncertainty", 0.0))
    if rl_uncertainty >= 0.35:
        return True

    projected_cost = float(node_signals.get("projected_cost_usd", 0.0))
    if projected_cost >= policy.max_cost_usd:
        return True

    return False
