"""Report synthesis for the scientific research agent."""

from __future__ import annotations

import json
from typing import Iterable, List, Optional

from pydantic import BaseModel, Field

from ..models import EvidenceTableRow, Paper, VerificationResult


class SynthesisResult(BaseModel):
    """Structured output for downstream presentation."""

    summary: str
    recommendations: List[str] = Field(default_factory=list)
    provenance: List[dict] = Field(default_factory=list)


class EvidenceSynthesizer:
    """Generate grounded scientific summaries using an LLM."""

    def __init__(self, llm_client, max_tokens: int = 900):
        self.llm_client = llm_client
        self.max_tokens = max_tokens

    async def synthesize(
        self,
        query: str,
        papers: Iterable[Paper],
        verifications: Iterable[VerificationResult],
        evidence_table: Optional[Iterable[EvidenceTableRow]] = None,
    ) -> SynthesisResult:
        prompt = self._build_prompt(query, papers, verifications, evidence_table)
        raw_response = await self.llm_client.generate(prompt, max_tokens=self.max_tokens, temperature=0.4)
        parsed = self._parse_json(raw_response)

        if parsed:
            summary = parsed.get("summary") or parsed.get("synthesis") or ""
            recommendations = parsed.get("recommendations") or []
            provenance = parsed.get("provenance") or []
            return SynthesisResult(
                summary=summary.strip() or raw_response.strip(),
                recommendations=[str(rec) for rec in recommendations if rec],
                provenance=provenance if isinstance(provenance, list) else [],
            )

        return SynthesisResult(summary=raw_response.strip())

    def _build_prompt(
        self,
        query: str,
        papers: Iterable[Paper],
        verifications: Iterable[VerificationResult],
        evidence_table: Optional[Iterable[EvidenceTableRow]] = None,
    ) -> str:
        paper_list = list(papers)
        verification_list = list(verifications)
        table_list = list(evidence_table) if evidence_table is not None else []

        paper_lines = []
        for paper in paper_list[:8]:
            authors = ", ".join(author.name for author in paper.authors[:3])
            paper_lines.append(
                f"- {paper.title} ({paper.year or 'n.d.'}) by {authors or 'Unknown'} [id={paper.id}]"
            )

        verification_lines = []
        for result in verification_list:
            verification_lines.append(
                f"- Claim: {result.claim.text}\n  Verdict: {result.label} (confidence={result.confidence:.2f})\n  Citations: {', '.join(result.citations) or 'none'}"
            )

        table_lines = []
        if table_list:
            for row in table_list:
                table_lines.append(
                    f"- {row.paper_id}: design={row.study_design or 'unknown'}, outcome={row.outcome or 'n/a'}, key findings={row.key_findings or 'n/a'}"
                )

        return f"""
You are preparing the final report for a scientific research investigation.

Question: {query}

Key papers:
{chr(10).join(paper_lines) if paper_lines else 'No papers available.'}

Claim verifications:
{chr(10).join(verification_lines) if verification_lines else 'No claims verified yet.'}

Structured evidence table rows:
{chr(10).join(table_lines) if table_lines else 'No structured evidence rows.'}

Write a rigorous but concise summary (<= 5 paragraphs) that cites evidence inline using [paper_id] notation. Also suggest next-step recommendations if appropriate.
Return JSON with keys `summary`, `recommendations` (list), and `provenance` (list of objects with paper_id, quote, and page if available).
""".strip()

    @staticmethod
    def _parse_json(raw_response: str) -> Optional[dict]:
        raw_response = raw_response.strip()
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(raw_response[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


__all__ = ["EvidenceSynthesizer", "SynthesisResult"]
