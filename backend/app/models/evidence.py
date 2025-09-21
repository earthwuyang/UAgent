"""Evidence and claim tracking models for scientific research."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


VerificationLabel = Literal["SUPPORTS", "REFUTES", "NEI"]


class EvidenceSpan(BaseModel):
    """Grounded passage tied to a specific paper and page."""

    paper_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    snippet_start: Optional[int] = None
    snippet_end: Optional[int] = None
    score: Optional[float] = None


class Claim(BaseModel):
    """Declarative statement evaluated against collected evidence."""

    text: str
    normalized_question: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Structured verdict for a claim."""

    claim: Claim
    label: VerificationLabel
    confidence: float
    rationales: List[EvidenceSpan] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    reasoning: str = ""
    verifier_model: Optional[str] = None


class EvidenceTableRow(BaseModel):
    """Row in an evidence summary table for quick synthesis."""

    paper_id: str
    study_design: Optional[str] = None
    population: Optional[str] = None
    intervention: Optional[str] = None
    outcome: Optional[str] = None
    key_findings: Optional[str] = None
    limitations: Optional[str] = None
    citation: Optional[str] = None
