"""Typed models shared across the scientific research pipeline."""

from .paper import Author, Paper
from .evidence import (
    Claim,
    EvidenceSpan,
    EvidenceTableRow,
    VerificationLabel,
    VerificationResult,
)

__all__ = [
    "Author",
    "Paper",
    "Claim",
    "EvidenceSpan",
    "EvidenceTableRow",
    "VerificationLabel",
    "VerificationResult",
]
