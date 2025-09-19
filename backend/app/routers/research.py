"""Research API endpoints"""

import logging
from typing import Dict, Any, Optional, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.app_state import get_app_state
from ..core.research_engines.deep_research import ResearchResult as DeepResearchResult
from ..core.research_engines.code_research import CodeResearchResult
from ..core.research_engines.scientific_research import ScientificResearchResult


logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response models
class ResearchRequest(BaseModel):
    """Base research request model"""
    query: str = Field(..., description="Research query or question")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class DeepResearchRequest(ResearchRequest):
    """Deep research request model"""
    sources: Optional[List[str]] = Field(["web", "academic", "technical"], description="Source types to use")
    max_sources_per_type: Optional[int] = Field(10, description="Maximum sources per type")


class CodeResearchRequest(ResearchRequest):
    """Code research request model"""
    language: Optional[str] = Field(None, description="Programming language filter")
    include_analysis: bool = Field(True, description="Include detailed repository analysis")
    max_repositories: Optional[int] = Field(10, description="Maximum repositories to analyze")


class ScientificResearchRequest(ResearchRequest):
    """Scientific research request model"""
    include_literature_review: bool = Field(True, description="Include literature review")
    include_code_analysis: bool = Field(True, description="Include code analysis")
    enable_iteration: bool = Field(True, description="Enable iterative refinement")
    max_iterations: Optional[int] = Field(3, description="Maximum research iterations")
    confidence_threshold: Optional[float] = Field(0.8, description="Confidence threshold for completion")


class ResearchResponse(BaseModel):
    """Base research response model"""
    research_id: str = Field(..., description="Unique research identifier")
    status: str = Field(..., description="Research status")
    query: str = Field(..., description="Original research query")


class DeepResearchResponse(ResearchResponse):
    """Deep research response model"""
    sources_count: int = Field(..., description="Number of sources analyzed")
    key_findings: List[str] = Field(..., description="Key research findings")
    confidence_score: float = Field(..., description="Overall confidence score")
    analysis_summary: str = Field(..., description="Analysis summary")
    recommendations: List[str] = Field(..., description="Research recommendations")


class CodeResearchResponse(ResearchResponse):
    """Code research response model"""
    repositories_count: int = Field(..., description="Number of repositories analyzed")
    languages_found: List[str] = Field(..., description="Programming languages found")
    best_practices_count: int = Field(..., description="Number of best practices identified")
    confidence_score: float = Field(..., description="Overall confidence score")
    integration_guide_length: int = Field(..., description="Integration guide length")


class ScientificResearchResponse(ResearchResponse):
    """Scientific research response model"""
    hypotheses_count: int = Field(..., description="Number of hypotheses generated")
    experiments_count: int = Field(..., description="Number of experiments conducted")
    iterations_completed: int = Field(..., description="Number of iterations completed")
    confidence_score: float = Field(..., description="Overall confidence score")
    has_literature_review: bool = Field(..., description="Whether literature review was included")
    has_code_analysis: bool = Field(..., description="Whether code analysis was included")
    publication_draft_length: int = Field(..., description="Publication draft length")


# In-memory storage for research sessions (in production, use database)
research_sessions: Dict[str, Dict[str, Any]] = {}


@router.post("/deep", response_model=DeepResearchResponse)
async def conduct_deep_research(request: DeepResearchRequest, background_tasks: BackgroundTasks):
    """Conduct deep research using the Deep Research Engine"""
    try:
        app_state = get_app_state()
        deep_engine = app_state["engines"]["deep"]

        research_id = str(uuid4())
        logger.info(f"Starting deep research: {research_id} - {request.query}")

        # Execute deep research
        result = await deep_engine.research(
            request.query,
            sources=request.sources
        )

        # Store result
        research_sessions[research_id] = {
            "type": "deep_research",
            "request": request.dict(),
            "result": result,
            "status": "completed"
        }

        # Create response
        response = DeepResearchResponse(
            research_id=research_id,
            status="completed",
            query=request.query,
            sources_count=len(result.sources),
            key_findings=result.key_findings,
            confidence_score=result.confidence_score,
            analysis_summary=result.analysis,
            recommendations=result.recommendations
        )

        logger.info(f"Deep research completed: {research_id}")
        return response

    except Exception as e:
        logger.error(f"Deep research failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deep research failed: {str(e)}"
        )


@router.post("/code", response_model=CodeResearchResponse)
async def conduct_code_research(request: CodeResearchRequest, background_tasks: BackgroundTasks):
    """Conduct code research using the Code Research Engine"""
    try:
        app_state = get_app_state()
        code_engine = app_state["engines"]["code"]

        research_id = str(uuid4())
        logger.info(f"Starting code research: {research_id} - {request.query}")

        # Execute code research
        result = await code_engine.research_code(
            request.query,
            language=request.language,
            include_analysis=request.include_analysis
        )

        # Store result
        research_sessions[research_id] = {
            "type": "code_research",
            "request": request.dict(),
            "result": result,
            "status": "completed"
        }

        # Extract languages found
        languages_found = list(set(repo.language for repo in result.repositories))

        # Create response
        response = CodeResearchResponse(
            research_id=research_id,
            status="completed",
            query=request.query,
            repositories_count=len(result.repositories),
            languages_found=languages_found,
            best_practices_count=len(result.best_practices),
            confidence_score=result.confidence_score,
            integration_guide_length=len(result.integration_guide)
        )

        logger.info(f"Code research completed: {research_id}")
        return response

    except Exception as e:
        logger.error(f"Code research failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code research failed: {str(e)}"
        )


@router.post("/scientific", response_model=ScientificResearchResponse)
async def conduct_scientific_research(request: ScientificResearchRequest, background_tasks: BackgroundTasks):
    """Conduct scientific research using the Scientific Research Engine"""
    try:
        app_state = get_app_state()
        scientific_engine = app_state["engines"]["scientific"]

        research_id = str(uuid4())
        logger.info(f"Starting scientific research: {research_id} - {request.query}")

        # Update engine configuration if provided
        if request.max_iterations:
            scientific_engine.max_iterations = request.max_iterations
        if request.confidence_threshold:
            scientific_engine.confidence_threshold = request.confidence_threshold

        # Execute scientific research
        result = await scientific_engine.conduct_research(
            request.query,
            include_literature_review=request.include_literature_review,
            include_code_analysis=request.include_code_analysis,
            enable_iteration=request.enable_iteration
        )

        # Store result
        research_sessions[research_id] = {
            "type": "scientific_research",
            "request": request.dict(),
            "result": result,
            "status": "completed"
        }

        # Create response
        response = ScientificResearchResponse(
            research_id=research_id,
            status="completed",
            query=request.query,
            hypotheses_count=len(result.hypotheses),
            experiments_count=len(result.experiments),
            iterations_completed=result.iteration_count,
            confidence_score=result.confidence_score,
            has_literature_review=result.literature_review is not None,
            has_code_analysis=result.code_analysis is not None,
            publication_draft_length=len(result.publication_draft)
        )

        logger.info(f"Scientific research completed: {research_id}")
        return response

    except Exception as e:
        logger.error(f"Scientific research failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scientific research failed: {str(e)}"
        )


@router.get("/sessions")
async def list_research_sessions():
    """List all research sessions"""
    app_state = get_app_state()
    session_manager = app_state.get("session_manager")

    sessions: List[Dict[str, Any]] = []

    if session_manager:
        manager_sessions = await session_manager.list_sessions()
        for record in manager_sessions:
            sessions.append({
                "session_id": record["session_id"],
                "query": record["request"],
                "status": record["status"],
                "type": record["classification"].get("primary_engine"),
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
                "has_result": record["result"] is not None,
                "error": record["error"]
            })

    # Include legacy sessions created through direct research endpoints
    for research_id, session_data in research_sessions.items():
        sessions.append({
            "session_id": research_id,
            "query": session_data["request"]["query"],
            "status": session_data["status"],
            "type": session_data["type"],
            "created_at": session_data.get("created_at", "unknown"),
            "updated_at": session_data.get("updated_at", "unknown"),
            "has_result": True,
            "error": None
        })

    return {"sessions": sessions, "total": len(sessions)}


@router.get("/sessions/{research_id}")
async def get_research_session(research_id: str):
    """Get specific research session details"""
    app_state = get_app_state()
    session_manager = app_state.get("session_manager")

    if session_manager:
        record = await session_manager.get_session(research_id)
        if record:
            return {
                "session_id": record["session_id"],
                "request": record["request"],
                "classification": record["classification"],
                "status": record["status"],
                "result": record["result"],
                "error": record["error"],
                "created_at": record["created_at"],
                "updated_at": record["updated_at"]
            }
        return {
            "session_id": research_id,
            "status": "pending",
            "result": None,
            "error": None
        }

    if research_id in research_sessions:
        session_data = research_sessions[research_id]
        return {
            "session_id": research_id,
            "type": session_data["type"],
            "request": session_data["request"],
            "status": session_data["status"],
            "summary": _create_session_summary(session_data),
            "result": _serialize_result(session_data["result"]),
            "created_at": session_data.get("created_at", "unknown"),
            "updated_at": session_data.get("updated_at", "unknown")
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Research session {research_id} not found"
    )


@router.get("/sessions/{research_id}/full")
async def get_full_research_result(research_id: str):
    """Get full research result including all data"""
    app_state = get_app_state()
    session_manager = app_state.get("session_manager")

    if session_manager:
        record = await session_manager.get_session(research_id)
        if record:
            return {
                "session_id": record["session_id"],
                "classification": record["classification"],
                "result": record["result"],
                "status": record["status"],
                "error": record["error"]
            }
        return {
            "session_id": research_id,
            "classification": None,
            "result": None,
            "status": "pending",
            "error": None
        }

    if research_id in research_sessions:
        session_data = research_sessions[research_id]
        return {
            "session_id": research_id,
            "classification": {
                "primary_engine": session_data["type"],
                "user_request": session_data["request"].get("query")
            },
            "result": _serialize_result(session_data["result"]),
            "status": session_data["status"]
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Research session {research_id} not found"
    )


@router.delete("/sessions/{research_id}")
async def delete_research_session(research_id: str):
    """Delete a research session"""
    if research_id not in research_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research session {research_id} not found"
        )

    del research_sessions[research_id]
    return {"message": f"Research session {research_id} deleted successfully"}


def _create_session_summary(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary of research session"""
    result = session_data["result"]
    session_type = session_data["type"]

    if session_type == "deep_research":
        return {
            "sources_analyzed": len(result.sources),
            "key_findings_count": len(result.key_findings),
            "confidence_score": result.confidence_score,
            "recommendations_count": len(result.recommendations)
        }
    elif session_type == "code_research":
        return {
            "repositories_analyzed": len(result.repositories),
            "best_practices_count": len(result.best_practices),
            "confidence_score": result.confidence_score,
            "languages_found": list(set(repo.language for repo in result.repositories))
        }
    elif session_type == "scientific_research":
        return {
            "hypotheses_generated": len(result.hypotheses),
            "experiments_conducted": len(result.experiments),
            "iterations_completed": result.iteration_count,
            "confidence_score": result.confidence_score,
            "has_publication_draft": len(result.publication_draft) > 0
        }
    else:
        return {"type": "unknown"}


def _serialize_result(result: Any) -> Dict[str, Any]:
    """Serialize research result for JSON response"""
    if hasattr(result, "__dict__"):
        # Convert dataclass to dict, handling nested objects
        result_dict = {}
        for key, value in result.__dict__.items():
            if hasattr(value, "__dict__"):
                # Nested dataclass
                result_dict[key] = _serialize_result(value)
            elif isinstance(value, list) and value and hasattr(value[0], "__dict__"):
                # List of dataclasses
                result_dict[key] = [_serialize_result(item) for item in value]
            else:
                # Regular value
                result_dict[key] = value
        return result_dict
    else:
        return result
