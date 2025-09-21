"""Science-focused API endpoints."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..core.app_state import get_app_state
from ..core.websocket_manager import progress_tracker
from ..models import Claim, EvidenceSpan, Paper, VerificationResult
from ..pipelines import EvidenceRetriever, SynthesisResult
from ..services import ArtifactStore, PlaywrightCaptureService, QwenVisionAnalyzer

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/science", tags=["science"])


class ScienceRequest(BaseModel):
    """Incoming request payload for scientific research."""

    query: str
    session_id: Optional[str] = None
    max_papers: int = Field(12, ge=1, le=50)
    include_pdf_parsing: bool = False
    include_experiments: bool = False


class ScienceResponse(BaseModel):
    """Structured response from the scientific research agent."""

    session_id: str
    plan: Dict[str, Any]
    papers: List[Paper]
    claims: List[VerificationResult]
    summary: SynthesisResult
    provenance: List[Dict[str, Any]] = Field(default_factory=list)


async def _query_connector(name: str, client, query: str, max_results: int) -> List[Paper]:
    try:
        LOGGER.info("Querying %s connector for '%s'", name, query)
        return await client.search(query, max_results=max_results)
    except Exception as exc:
        LOGGER.warning("%s connector failed: %s", name, exc)
        return []


def _safe_artifact_name(identifier: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in identifier)[:80]


async def _collect_visual_evidence(
    paper: Paper,
    session_id: str,
    question: str,
    browser_service: PlaywrightCaptureService,
    vision_analyzer: QwenVisionAnalyzer,
    artifact_store: Optional[ArtifactStore],
    max_pages: int = 3,
) -> List[EvidenceSpan]:
    target_url = paper.pdf_url or paper.url
    if not target_url:
        return []

    try:
        screenshots = await browser_service.capture_pdf(target_url, max_pages=max_pages)
    except Exception as exc:  # pragma: no cover - network/Playwright variability
        LOGGER.warning("Playwright capture failed for %s: %s", target_url, exc)
        return []

    if not screenshots:
        return []

    analyses = await vision_analyzer.batch_describe(
        screenshots,
        f"Extract key evidence that answers the research question: {question}",
    )

    spans: List[EvidenceSpan] = []
    safe_name = _safe_artifact_name(paper.id or paper.title)
    for idx, (image_bytes, analysis) in enumerate(zip(screenshots, analyses), start=1):
        if artifact_store:
            image_path = artifact_store.save_bytes(
                session_id,
                f"{safe_name}_page{idx}",
                image_bytes,
                suffix=".png",
            )
            artifact_store.record_metadata(
                session_id,
                image_path,
                {"paper_id": paper.id, "type": "screenshot", "page": idx},
            )

        if not analysis:
            continue

        spans.append(
            EvidenceSpan(
                paper_id=paper.id,
                text=analysis,
                section=f"visual_page_{idx}",
                page=idx,
            )
        )

    return spans


@router.post("/ask", response_model=ScienceResponse, status_code=status.HTTP_200_OK)
async def ask_scientific_agent(request: ScienceRequest) -> ScienceResponse:
    app_state = get_app_state()
    science_tools = app_state.get("science_tools")
    if not science_tools:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scientific tooling not initialized")

    session_id = request.session_id or f"science_{uuid.uuid4().hex[:8]}"

    await progress_tracker.log_research_started(
        session_id=session_id,
        request=request.query,
        engine="scientific_research"
    )

    connectors: Dict[str, Any] = science_tools.get("connectors", {})
    retriever_factory = science_tools.get("retriever_factory", EvidenceRetriever)
    claim_verifier = science_tools.get("claim_verifier")
    synthesizer = science_tools.get("synthesizer")
    research_graph = science_tools.get("research_graph")
    artifact_store: Optional[ArtifactStore] = science_tools.get("artifact_store")
    browser_service: Optional[PlaywrightCaptureService] = science_tools.get("browser_service")
    vision_analyzer: Optional[QwenVisionAnalyzer] = science_tools.get("vision_analyzer")

    if not claim_verifier or not synthesizer:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scientific verifier not configured")

    tasks = [
        _query_connector(name, connector, request.query, request.max_papers)
        for name, connector in connectors.items()
    ]
    connector_results = await asyncio.gather(*tasks)

    all_papers: Dict[str, Paper] = {}
    for results in connector_results:
        for paper in results:
            if not paper or not paper.id:
                continue
            if paper.id not in all_papers:
                all_papers[paper.id] = paper
                if research_graph:
                    try:
                        research_graph.upsert_paper(paper)
                    except Exception as exc:  # pragma: no cover - persistence best effort
                        LOGGER.debug("Failed to persist paper %s: %s", paper.id, exc)

    papers = list(all_papers.values())[: request.max_papers]

    await progress_tracker.log_research_progress(
        session_id=session_id,
        engine="scientific_research",
        phase="literature_search",
        progress=20.0,
        message=f"Collected {len(papers)} papers",
        metadata={"paper_ids": [paper.id for paper in papers]}
    )

    retriever = retriever_factory()
    for paper in papers:
        if paper.abstract:
            spans = retriever.build_spans(
                paper_id=paper.id,
                blocks=[{"text": paper.abstract, "section": "abstract", "page": None}],
            )
            retriever.add_spans(spans)

    await progress_tracker.log_research_progress(
        session_id=session_id,
        engine="scientific_research",
        phase="evidence_collection",
        progress=40.0,
        message="Indexed abstracts for retrieval",
    )

    visual_spans = 0
    parsed_papers = 0
    if request.include_pdf_parsing:
        if not browser_service or not browser_service.available:
            LOGGER.warning("Playwright capture unavailable; skipping visual parsing")
        elif not vision_analyzer:
            LOGGER.warning("Qwen-VL analyzer unavailable; skipping visual parsing")
        else:
            max_visual_papers = min(len(papers), 4)
            for paper in papers[:max_visual_papers]:
                spans = await _collect_visual_evidence(
                    paper,
                    session_id,
                    request.query,
                    browser_service,
                    vision_analyzer,
                    artifact_store,
                )
                if spans:
                    retriever.add_spans(spans)
                    visual_spans += len(spans)
                    parsed_papers += 1

            await progress_tracker.log_research_progress(
                session_id=session_id,
                engine="scientific_research",
                phase="visual_parsing",
                progress=55.0,
                message=f"Captured visual evidence for {parsed_papers} papers",
                metadata={"parsed_papers": parsed_papers, "spans_added": visual_spans},
            )

    claim = Claim(text=request.query, normalized_question=request.query)
    candidate_spans = retriever.search(request.query, limit=6)

    await progress_tracker.log_research_progress(
        session_id=session_id,
        engine="scientific_research",
        phase="claim_verification",
        progress=70.0,
        message=f"Evaluating claim against {len(candidate_spans)} evidence spans",
    )

    verification = await claim_verifier.verify_claim(claim, candidate_spans)
    if research_graph:
        try:
            claim_id = research_graph.record_claim(claim)
            research_graph.record_verification(verification, claim_id=claim_id)
        except Exception as exc:  # pragma: no cover - persistence best effort
            LOGGER.debug("Failed to persist verification for %s: %s", claim.text, exc)

    await progress_tracker.log_research_progress(
        session_id=session_id,
        engine="scientific_research",
        phase="synthesis",
        progress=85.0,
        message="Synthesizing report",
    )

    synthesis_result = await synthesizer.synthesize(
        query=request.query,
        papers=papers,
        verifications=[verification],
        evidence_table=None,
    )

    plan = {
        "question": request.query,
        "steps": [
            {"phase": "literature_search", "status": "completed", "sources": list(connectors.keys())},
            {"phase": "claim_verification", "status": "completed", "claims": 1},
            {"phase": "synthesis", "status": "completed"},
        ],
        "include_experiments": request.include_experiments,
    }

    await progress_tracker.log_research_completed(
        session_id=session_id,
        engine="scientific_research",
        result_summary=synthesis_result.summary,
        metadata={
            "papers_considered": len(papers),
            "claims_verified": 1,
            "confidence": verification.confidence,
            "visual_spans": visual_spans,
        }
    )

    response = ScienceResponse(
        session_id=session_id,
        plan=plan,
        papers=papers,
        claims=[verification],
        summary=synthesis_result,
        provenance=synthesis_result.provenance,
    )

    return response


__all__ = ["router"]
