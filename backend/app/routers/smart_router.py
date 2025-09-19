"""Smart Router API endpoints"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..core.app_state import get_app_state
from ..core.smart_router import ClassificationRequest, ClassificationResult
from ..core.websocket_manager import progress_tracker


logger = logging.getLogger(__name__)
router = APIRouter()


class RouterRequest(BaseModel):
    """Smart router request model"""
    user_request: str = Field(..., description="User request to classify and route")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    context: Optional[Dict[str, Any]] = Field({}, description="Additional context")


class RouterResponse(BaseModel):
    """Smart router response model"""
    primary_engine: str = Field(..., description="Primary engine recommendation")
    confidence_score: float = Field(..., description="Classification confidence score")
    sub_components: Dict[str, bool] = Field(..., description="Sub-component breakdown")
    reasoning: str = Field(..., description="Reasoning for the classification")
    workflow_plan: Dict[str, Any] = Field(..., description="Workflow plan")
    user_request: str = Field(..., description="Original user request")


@router.post("/classify", response_model=RouterResponse)
async def classify_request(request: RouterRequest):
    """Classify user request and recommend appropriate research engine"""
    try:
        app_state = get_app_state()
        smart_router = app_state["smart_router"]

        logger.info(f"Classifying request: {request.user_request[:100]}...")

        # Create classification request
        classification_request = ClassificationRequest(
            user_request=request.user_request,
            context=request.context
        )

        # Classify and route
        result = await smart_router.classify_and_route(classification_request)

        # Create response
        response = RouterResponse(
            primary_engine=result.primary_engine,
            confidence_score=result.confidence_score,
            sub_components=result.sub_components,
            reasoning=result.reasoning,
            workflow_plan=result.workflow_plan,
            user_request=request.user_request
        )

        logger.info(f"Classification completed: {result.primary_engine} (confidence: {result.confidence_score:.3f})")
        return response

    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/route-and-execute")
async def route_and_execute(request: RouterRequest):
    """Classify request and execute using the recommended engine"""
    try:
        app_state = get_app_state()
        smart_router = app_state["smart_router"]
        engines = app_state["engines"]

        # Generate session ID for tracking
        import uuid
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"

        logger.info(f"Routing and executing request: {request.user_request[:100]}...")

        # Classify request
        classification_request = ClassificationRequest(
            user_request=request.user_request,
            context=request.context
        )

        classification_result = await smart_router.classify_and_route(classification_request)

        # Log research started
        await progress_tracker.log_research_started(
            session_id=session_id,
            request=request.user_request,
            engine=classification_result.primary_engine
        )

        # Execute based on classification
        engine_name = classification_result.primary_engine.lower().replace("_", "")

        if engine_name == "deepresearch":
            engine = engines["deep"]
            execution_result = await engine.research(request.user_request)

            return {
                "classification": {
                    "primary_engine": classification_result.primary_engine,
                    "confidence_score": classification_result.confidence_score,
                    "reasoning": classification_result.reasoning
                },
                "execution": {
                    "engine_used": "deep_research",
                    "query": execution_result.query,
                    "sources_count": len(execution_result.sources),
                    "key_findings": execution_result.key_findings,
                    "confidence_score": execution_result.confidence_score,
                    "summary": execution_result.summary[:500] + "..." if len(execution_result.summary) > 500 else execution_result.summary
                }
            }

        elif engine_name == "coderesearch":
            engine = engines["code"]
            execution_result = await engine.research_code(request.user_request)

            return {
                "classification": {
                    "primary_engine": classification_result.primary_engine,
                    "confidence_score": classification_result.confidence_score,
                    "reasoning": classification_result.reasoning
                },
                "execution": {
                    "engine_used": "code_research",
                    "query": execution_result.query,
                    "repositories_count": len(execution_result.repositories),
                    "languages_found": list(set(repo.language for repo in execution_result.repositories)),
                    "best_practices_count": len(execution_result.best_practices),
                    "confidence_score": execution_result.confidence_score,
                    "integration_guide_preview": execution_result.integration_guide[:500] + "..." if len(execution_result.integration_guide) > 500 else execution_result.integration_guide,
                    "recommendations_count": len(execution_result.recommendations)
                }
            }

        elif engine_name == "scientificresearch":
            engine = engines["scientific"]
            params = classification_result.workflow_plan

            execution_result = await engine.conduct_research(
                request.user_request,
                include_literature_review=params.get("include_literature_review", True),
                include_code_analysis=params.get("include_code_analysis", True),
                enable_iteration=params.get("enable_iteration", True)
            )

            return {
                "classification": {
                    "primary_engine": classification_result.primary_engine,
                    "confidence_score": classification_result.confidence_score,
                    "reasoning": classification_result.reasoning
                },
                "execution": {
                    "engine_used": "scientific_research",
                    "query": execution_result.query,
                    "hypotheses_count": len(execution_result.hypotheses),
                    "experiments_count": len(execution_result.experiments),
                    "iterations_completed": execution_result.iteration_count,
                    "confidence_score": execution_result.confidence_score,
                    "has_literature_review": execution_result.literature_review is not None,
                    "has_code_analysis": execution_result.code_analysis is not None,
                    "summary": execution_result.synthesis if isinstance(execution_result.synthesis, str) else str(execution_result.synthesis)[:500] + "..."
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown engine type: {classification_result.primary_engine}"
            )

    except Exception as e:
        logger.error(f"Route and execute failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route and execute failed: {str(e)}"
        )


@router.get("/engines")
async def list_available_engines():
    """List all available research engines and their capabilities"""
    return {
        "engines": {
            "DEEP_RESEARCH": {
                "name": "Deep Research Engine",
                "description": "Comprehensive multi-source research with web, academic, and technical sources",
                "capabilities": [
                    "Web search and analysis",
                    "Academic paper research",
                    "Technical documentation analysis",
                    "Multi-source synthesis",
                    "Key findings extraction"
                ],
                "best_for": [
                    "Market research",
                    "Literature reviews",
                    "Industry analysis",
                    "Competitive intelligence",
                    "General knowledge queries"
                ]
            },
            "CODE_RESEARCH": {
                "name": "Code Research Engine",
                "description": "Specialized research for code analysis, repositories, and development patterns",
                "capabilities": [
                    "GitHub repository analysis",
                    "Code pattern identification",
                    "Best practices extraction",
                    "Library and framework research",
                    "Integration guides generation"
                ],
                "best_for": [
                    "Technology evaluation",
                    "Library selection",
                    "Code architecture research",
                    "Development methodologies",
                    "Technical implementation guides"
                ]
            },
            "SCIENTIFIC_RESEARCH": {
                "name": "Scientific Research Engine",
                "description": "Advanced research engine with hypothesis generation, experimental design, and iterative refinement",
                "capabilities": [
                    "Hypothesis generation and testing",
                    "Experimental design",
                    "Literature review integration",
                    "Code analysis integration",
                    "Iterative refinement",
                    "Publication-ready output"
                ],
                "best_for": [
                    "Academic research",
                    "Scientific investigations",
                    "Experimental validation",
                    "Research proposals",
                    "Complex multi-faceted studies"
                ]
            }
        },
        "routing_logic": {
            "description": "Requests are automatically classified using AI to determine the most appropriate engine",
            "factors": [
                "Query complexity and scope",
                "Research methodology requirements",
                "Output format expectations",
                "Domain expertise needed",
                "Integration requirements"
            ]
        }
    }


@router.get("/status")
async def router_status():
    """Get smart router system status"""
    try:
        app_state = get_app_state()
        smart_router = app_state["smart_router"]
        engines = app_state["engines"]

        # Check engine availability
        engine_status = {}
        for name, engine in engines.items():
            engine_status[name] = {
                "available": engine is not None,
                "type": engine.__class__.__name__ if engine else None
            }

        return {
            "status": "operational",
            "smart_router": {
                "available": smart_router is not None,
                "type": smart_router.__class__.__name__ if smart_router else None
            },
            "engines": engine_status,
            "total_engines": len(engines),
            "cache_available": app_state.get("cache") is not None,
            "llm_client_available": app_state.get("llm_client") is not None
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "smart_router": {"available": False},
            "engines": {},
            "total_engines": 0
        }