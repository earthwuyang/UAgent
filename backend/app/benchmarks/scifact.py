"""SciFact claim verification benchmark."""

from __future__ import annotations

import logging
from typing import Iterable

from ..models import Claim
from ..pipelines import ClaimVerifier
from .base import BenchmarkExample, BenchmarkResult

LOGGER = logging.getLogger(__name__)


class SciFactBenchmark:
    """Evaluate the claim verifier against SciFact-style data."""

    def __init__(self, verifier: ClaimVerifier):
        self.verifier = verifier

    async def evaluate(self, examples: Iterable[BenchmarkExample]) -> BenchmarkResult:
        total = 0
        correct = 0
        details = []

        for example in examples:
            total += 1
            claim = example.claim
            candidate_spans = example.evidence
            try:
                result = await self.verifier.verify_claim(claim, candidate_spans)
            except Exception as exc:  # pragma: no cover - evaluation best effort
                LOGGER.error("Verifier failed for example %s: %s", example.id, exc)
                details.append({
                    "id": example.id,
                    "error": str(exc),
                    "expected": example.label,
                })
                continue

            is_correct = result.label == example.label
            if is_correct:
                correct += 1

            details.append({
                "id": example.id,
                "predicted": result.label,
                "expected": example.label,
                "confidence": result.confidence,
                "citations": result.citations,
            })

        return BenchmarkResult.from_counts("SciFact", total, correct, details)
