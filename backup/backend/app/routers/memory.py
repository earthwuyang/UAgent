"""
Memory Management API Router
Provides endpoints for memory management functionality
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..core.memory_service import MemoryService, MemoryEventKind, MemorySnapshotType
from ..core.memory_aware_orchestrator import MemoryAwareOrchestrator
from ..core.semantic_memory import RetrievalConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])

# Initialize memory services
memory_service = MemoryService()
memory_orchestrator = MemoryAwareOrchestrator()


# Pydantic models for API
class MemoryEventRequest(BaseModel):
    run_id: str
    node_id: str
    source: str = Field(..., description="Source of the event (e.g., 'llm', 'tool:search', 'user')")
    kind: str = Field(..., description="Event kind: observation, decision, error, note, tool_call, tool_result")
    body: Dict[str, Any] = Field(..., description="Event body content")
    importance: float = Field(0.0, ge=0.0, le=1.0, description="Importance score 0.0-1.0")


class SemanticPromotionRequest(BaseModel):
    text: str = Field(..., description="Text content to promote")
    title: Optional[str] = Field(None, description="Optional title")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    org_scope: str = Field("default", description="Organization scope")


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(10, ge=1, le=50, description="Number of results to return")
    org_scope: str = Field("default", description="Organization scope")
    use_mmr: bool = Field(True, description="Use MMR for diversification")
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0, description="MMR lambda parameter")


class ContextRequest(BaseModel):
    run_id: str
    node_id: str
    query_terms: List[str] = Field(default_factory=list, description="Query terms for context")
    max_tokens: int = Field(8000, ge=1000, le=32000, description="Maximum tokens for context")


class DecisionLogRequest(BaseModel):
    run_id: str
    node_id: str
    decision: str
    reasoning: str
    context: Dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(0.7, ge=0.0, le=1.0)


class ToolInteractionRequest(BaseModel):
    run_id: str
    node_id: str
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    execution_time: float


class MilestoneRequest(BaseModel):
    run_id: str
    node_id: str
    milestone_name: str
    achievements: List[str]
    next_steps: List[str]


class PostmortemRequest(BaseModel):
    run_id: str
    node_id: str
    final_results: Dict[str, Any]
    lessons_learned: List[str]


# Event logging endpoints
@router.post("/events/log")
async def log_event(request: MemoryEventRequest):
    """Log a memory event"""
    try:
        # Validate event kind
        try:
            event_kind = MemoryEventKind(request.kind)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event kind: {request.kind}")

        event_id = await memory_service.log_event(
            run_id=request.run_id,
            node_id=request.node_id,
            source=request.source,
            kind=event_kind,
            body=request.body,
            importance=request.importance,
            maybe_artifact=len(str(request.body)) > 1000
        )

        if event_id < 0:
            raise HTTPException(status_code=500, detail="Failed to log event")

        return {
            "event_id": event_id,
            "status": "logged",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to log event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decisions/log")
async def log_decision(request: DecisionLogRequest):
    """Log a decision to memory"""
    try:
        await memory_orchestrator.log_decision(
            run_id=request.run_id,
            node_id=request.node_id,
            decision=request.decision,
            reasoning=request.reasoning,
            context=request.context,
            importance=request.importance
        )

        return {"status": "logged", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Failed to log decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/log")
async def log_tool_interaction(request: ToolInteractionRequest):
    """Log tool interaction to memory"""
    try:
        await memory_orchestrator.log_tool_interaction(
            run_id=request.run_id,
            node_id=request.node_id,
            tool_name=request.tool_name,
            parameters=request.parameters,
            result=request.result,
            success=request.success,
            execution_time=request.execution_time
        )

        return {"status": "logged", "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Failed to log tool interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory retrieval endpoints
@router.get("/events/{run_id}/{node_id}")
async def get_events(
    run_id: str,
    node_id: str,
    limit: int = Query(50, ge=1, le=1000),
    min_importance: float = Query(0.0, ge=0.0, le=1.0)
):
    """Get memory events for a run/node"""
    try:
        events = await memory_service.get_salient_events(
            run_id=run_id,
            node_id=node_id,
            min_importance=min_importance,
            limit=limit
        )

        return {
            "events": events,
            "count": len(events),
            "run_id": run_id,
            "node_id": node_id
        }

    except Exception as e:
        logger.error(f"Failed to get events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshots/{run_id}/{node_id}/latest")
async def get_latest_snapshot(run_id: str, node_id: str):
    """Get the latest memory snapshot"""
    try:
        snapshot = await memory_service.get_latest_snapshot(run_id, node_id)

        if not snapshot:
            raise HTTPException(status_code=404, detail="No snapshot found")

        return snapshot

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshots/create")
async def create_summary_snapshot(
    run_id: str,
    node_id: str,
    summary_type: str = "rolling"
):
    """Create a memory summary snapshot"""
    try:
        # Validate summary type
        try:
            snapshot_type = MemorySnapshotType(summary_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid summary type: {summary_type}")

        summary_id = await memory_service.create_rolling_summary(run_id, node_id, snapshot_type)

        if summary_id < 0:
            raise HTTPException(status_code=500, detail="Failed to create summary")

        return {
            "summary_id": summary_id,
            "type": summary_type,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create summary snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Context assembly endpoints
@router.post("/context/assemble")
async def assemble_context(request: ContextRequest):
    """Assemble memory context for LLM prompts"""
    try:
        from ..core.memory_service import ContextConfig

        config = ContextConfig(max_tokens=request.max_tokens)

        context = await memory_service.assemble_context(
            run_id=request.run_id,
            node_id=request.node_id,
            query_terms=request.query_terms,
            config=config
        )

        return context

    except Exception as e:
        logger.error(f"Failed to assemble context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/{run_id}/{node_id}")
async def get_memory_context(
    run_id: str,
    node_id: str,
    query: str = Query("", description="Query for context retrieval"),
    max_tokens: int = Query(4000, ge=1000, le=16000)
):
    """Get memory-enhanced context"""
    try:
        context = await memory_orchestrator.get_memory_enhanced_context(
            run_id=run_id,
            node_id=node_id,
            query=query,
            max_tokens=max_tokens
        )

        return context

    except Exception as e:
        logger.error(f"Failed to get memory context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Semantic memory endpoints
@router.post("/semantic/promote")
async def promote_to_semantic_memory(request: SemanticPromotionRequest):
    """Promote content to semantic memory"""
    try:
        chunk_id = await memory_service.promote_to_semantic_memory(
            text=request.text,
            title=request.title,
            meta=request.meta,
            org_scope=request.org_scope
        )

        if chunk_id < 0:
            raise HTTPException(status_code=500, detail="Failed to promote content")

        return {
            "chunk_id": chunk_id,
            "status": "promoted",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to promote to semantic memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic/search")
async def search_semantic_memory(request: SemanticSearchRequest):
    """Search semantic memory"""
    try:
        query_terms = request.query.split()

        results = await memory_service.search_semantic_memory(
            query_terms=query_terms,
            k=request.k,
            org_scope=request.org_scope,
            use_mmr=request.use_mmr,
            mmr_lambda=request.mmr_lambda
        )

        return {
            "results": results,
            "count": len(results),
            "query": request.query,
            "parameters": {
                "k": request.k,
                "use_mmr": request.use_mmr,
                "mmr_lambda": request.mmr_lambda
            }
        }

    except Exception as e:
        logger.error(f"Failed to search semantic memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/semantic/similar")
async def search_similar_experiences(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0)
):
    """Search for similar past experiences"""
    try:
        results = await memory_orchestrator.search_similar_experiences(
            query=query,
            k=k,
            min_confidence=min_confidence
        )

        return {
            "results": results,
            "count": len(results),
            "query": query,
            "parameters": {"k": k, "min_confidence": min_confidence}
        }

    except Exception as e:
        logger.error(f"Failed to search similar experiences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Milestone and postmortem endpoints
@router.post("/milestones/create")
async def create_milestone(request: MilestoneRequest):
    """Create a milestone summary"""
    try:
        summary_id = await memory_orchestrator.create_milestone_summary(
            run_id=request.run_id,
            node_id=request.node_id,
            milestone_name=request.milestone_name,
            achievements=request.achievements,
            next_steps=request.next_steps
        )

        if summary_id < 0:
            raise HTTPException(status_code=500, detail="Failed to create milestone")

        return {
            "summary_id": summary_id,
            "milestone_name": request.milestone_name,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create milestone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/postmortems/create")
async def create_postmortem(request: PostmortemRequest):
    """Create a postmortem summary"""
    try:
        summary_id = await memory_orchestrator.create_postmortem(
            run_id=request.run_id,
            node_id=request.node_id,
            final_results=request.final_results,
            lessons_learned=request.lessons_learned
        )

        if summary_id < 0:
            raise HTTPException(status_code=500, detail="Failed to create postmortem")

        return {
            "summary_id": summary_id,
            "run_id": request.run_id,
            "node_id": request.node_id,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create postmortem: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System status and metrics endpoints
@router.get("/metrics")
async def get_memory_metrics(run_id: Optional[str] = Query(None)):
    """Get memory system metrics"""
    try:
        metrics = await memory_service.get_memory_system_metrics(run_id)
        return metrics

    except Exception as e:
        logger.error(f"Failed to get memory metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_memory_status():
    """Get comprehensive memory system status"""
    try:
        status = await memory_orchestrator.get_memory_system_status()
        return status

    except Exception as e:
        logger.error(f"Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/embedding")
async def get_embedding_stats():
    """Get embedding system statistics"""
    try:
        stats = memory_service.semantic_engine.get_embedding_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Maintenance endpoints
@router.post("/maintenance/cleanup")
async def cleanup_memory(max_age_days: int = Query(30, ge=1, le=365)):
    """Clean up old memory data"""
    try:
        await memory_service.cleanup_expired_memory(max_age_days)

        return {
            "status": "cleanup_completed",
            "max_age_days": max_age_days,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to cleanup memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/recompute-embeddings")
async def recompute_embeddings(batch_size: int = Query(50, ge=10, le=200)):
    """Recompute embeddings for chunks that don't have them"""
    try:
        with memory_service._get_connection() as conn:
            await memory_service.semantic_engine.recompute_all_embeddings(conn, batch_size)

        return {
            "status": "embeddings_recomputed",
            "batch_size": batch_size,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to recompute embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Database initialization endpoint
@router.post("/admin/init-database")
async def initialize_database():
    """Initialize the memory database tables"""
    try:
        from ..core.database import ResearchDatabase

        # Initialize the database - this will create all tables including memory tables
        db = ResearchDatabase()

        return {
            "status": "database_initialized",
            "timestamp": datetime.now().isoformat(),
            "tables_created": [
                "memory_events", "memory_snapshots", "semantic_chunks",
                "kv_memory", "kg_triples", "memory_stats"
            ]
        }

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Debugging endpoints
@router.get("/debug/events-since-snapshot/{run_id}/{node_id}")
async def get_events_since_snapshot(run_id: str, node_id: str):
    """Get events since last snapshot (for debugging)"""
    try:
        events = await memory_service.get_events_since_last_snapshot(run_id, node_id)

        return {
            "events": events,
            "count": len(events),
            "run_id": run_id,
            "node_id": node_id
        }

    except Exception as e:
        logger.error(f"Failed to get events since snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))