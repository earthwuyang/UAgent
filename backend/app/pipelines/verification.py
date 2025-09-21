"""LLM-backed scientific claim verification."""

from __future__ import annotations

import json
import logging
from typing import List, cast

from ..core.llm_client import LLMClient
from ..models import Claim, EvidenceSpan, VerificationLabel, VerificationResult

LOGGER = logging.getLogger(__name__)


class ClaimVerifier:
    """Turn evidence spans into structured claim verdicts."""

    def __init__(self, llm_client: LLMClient, max_rationales: int = 3):
        self.llm_client = llm_client
        self.max_rationales = max_rationales

    async def verify_claim(self, claim: Claim, candidate_spans: List[EvidenceSpan]) -> VerificationResult:
        prompt = self._build_prompt(claim, candidate_spans)
        raw_response = await self.llm_client.generate(prompt, max_tokens=800, temperature=0.2)

        parsed = self._parse_response(raw_response)
        label_value = parsed.get("label", "NEI").upper()
        if label_value not in {"SUPPORTS", "REFUTES", "NEI"}:
            label_value = "NEI"

        confidence = float(parsed.get("confidence", parsed.get("confidence_score", 0.0)))
        try:
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        rationales_payload = parsed.get("rationales", [])[: self.max_rationales]
        rationales: List[EvidenceSpan] = []
        for item in rationales_payload:
            if isinstance(item, dict):
                try:
                    rationales.append(EvidenceSpan(**item))
                except TypeError:
                    LOGGER.debug("Skipping malformed rationale payload: %s", item)

        citations = parsed.get("citations", [])
        if not isinstance(citations, list):
            citations = []

        reasoning = parsed.get("reasoning") or parsed.get("explanation") or ""

        return VerificationResult(
            claim=claim,
            label=cast(VerificationLabel, label_value),
            confidence=confidence,
            rationales=rationales,
            citations=[str(cite) for cite in citations],
            reasoning=reasoning,
            verifier_model=self.llm_client.__class__.__name__,
        )

    def _build_prompt(self, claim: Claim, spans: List[EvidenceSpan]) -> str:
        highlighted_spans = spans[: self.max_rationales * 2]
        span_lines = []
        for idx, span in enumerate(highlighted_spans, start=1):
            snippet = span.text.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = f"{snippet[:400]}..."
            span_lines.append(
                f"[{idx}] paper_id={span.paper_id} page={span.page or '?'} section={span.section or 'unknown'}\n{snippet}"
            )
        evidence_block = "\n\n".join(span_lines) if span_lines else "(no supporting evidence provided)"

        return f"""
You are an expert scientific fact checker. Determine whether the claim is supported by the provided evidence excerpts.

Return a compact JSON object with keys: label (SUPPORTS, REFUTES, NEI), confidence (0-1 float), reasoning, rationales (list of objects with paper_id, text, page, section, snippet_start, snippet_end), and citations (list of paper_ids).

Claim: {claim.text}

Evidence excerpts:
{evidence_block}

Respond with JSON only.
""".strip()

    def _parse_response(self, raw_response: str) -> dict:
        raw_response = raw_response.strip()
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and start < end:
            snippet = raw_response[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                LOGGER.debug("Failed to parse JSON snippet from verifier response: %s", raw_response)

        LOGGER.warning("Verifier returned non-JSON response: %s", raw_response)
        return {}


__all__ = ["ClaimVerifier"]
