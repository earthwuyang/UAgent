"""Base data structures for benchmark evaluations."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from ..models import Claim, EvidenceSpan, VerificationLabel


class BenchmarkExample(BaseModel):
    """Single verification example used in benchmarking."""

    id: str
    claim: Claim
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    label: VerificationLabel


class BenchmarkResult(BaseModel):
    """Aggregate metrics from a benchmark run."""

    name: str
    total: int
    correct: int
    accuracy: float
    details: List[Dict[str, object]] = Field(default_factory=list)

    @classmethod
    def from_counts(cls, name: str, total: int, correct: int, details: List[Dict[str, object]] | None = None) -> "BenchmarkResult":
        accuracy = (correct / total) if total else 0.0
        return cls(name=name, total=total, correct=correct, accuracy=accuracy, details=details or [])
