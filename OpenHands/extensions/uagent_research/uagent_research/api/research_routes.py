"""
Research API Routes

FastAPI routes for research experiment management.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select as sql_select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Experiment,
    ExperimentStatus,
    ExperimentType,
    ResearchSession,
    Idea,
    Hypothesis,
)
from ..models.base import get_session

logger = logging.getLogger(__name__)

# Import control and session management
try:
    from ...control.control_bus import ControlBus, ControlMessage
    from ...services.research_session_manager import ResearchSessionManager
    from ...orchestrator.event_bus import get_event_bus
    CONTROL_BUS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Control bus not available: {e}")
    CONTROL_BUS_AVAILABLE = False
    ControlBus = None
    ControlMessage = None
    ResearchSessionManager = None

router = APIRouter(prefix="/api/research", tags=["research"])

# Import orchestrator and related components for background execution
try:
    from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
    from extensions.uagent_research.orchestrator.event_bus import get_event_bus
    from extensions.uagent_research.uagent_research.models.research_tree import Budget
    ORCHESTRATOR_AVAILABLE = True
    print(f"[DEBUG] Orchestrator import successful, ORCHESTRATOR_AVAILABLE={ORCHESTRATOR_AVAILABLE}", flush=True)
except ImportError as e:
    logger.warning(f"Orchestrator not available: {e}")
    print(f"[DEBUG] Orchestrator import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    ORCHESTRATOR_AVAILABLE = False
    TreeSearchOrchestrator = None

# Global storage for active orchestrators
_active_orchestrators: Dict[str, TreeSearchOrchestrator] = {}


async def run_experiment_async(experiment_id: str, goal: str, config: Optional[Dict[str, Any]] = None):
    """
    Run research experiment in background.

    Args:
        experiment_id: Experiment ID
        goal: Research goal
        config: Optional configuration
    """
    print(f"[DEBUG] run_experiment_async called for {experiment_id}", flush=True)
    if not ORCHESTRATOR_AVAILABLE:
        logger.error(f"Cannot run experiment {experiment_id}: Orchestrator not available")
        print(f"[DEBUG] ORCHESTRATOR_AVAILABLE is False, exiting", flush=True)
        return

    try:
        logger.info(f"Starting background execution for experiment {experiment_id}")
        print(f"[DEBUG] Starting background execution for experiment {experiment_id}", flush=True)

        # Get config parameters
        config = config or {}
        max_iterations = config.get('max_iterations', 50)
        max_cost = config.get('max_cost', 10.0)
        max_parallel = config.get('max_parallel', 3)

        # Create budget
        budget = Budget(
            max_iterations=max_iterations,
            max_cost=max_cost,
            max_tokens=config.get('max_tokens', 100000),
            deadline=None,
        )

        # Get event bus
        event_bus = get_event_bus()

        # Create orchestrator
        orchestrator = TreeSearchOrchestrator(
            max_parallel=max_parallel,
            budget=budget,
            event_bus=event_bus,
        )

        # Store orchestrator
        _active_orchestrators[experiment_id] = orchestrator

        # Update experiment status to RUNNING
        async for db_session in get_session():
            result = await db_session.execute(
                sql_select(Experiment).where(Experiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            if experiment:
                experiment.status = ExperimentStatus.RUNNING
                experiment.started_at = datetime.utcnow()
                await db_session.commit()
            break  # Only need one iteration

        logger.info(f"Running orchestrator for experiment {experiment_id}")

        # Run orchestrator
        tree = await orchestrator.run(
            goal=goal,
            max_iterations=max_iterations,
            research_id=experiment_id,
        )

        logger.info(f"Experiment {experiment_id} completed successfully")

        # Update experiment status to COMPLETE
        async for db_session in get_session():
            result = await db_session.execute(
                sql_select(Experiment).where(Experiment.id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            if experiment:
                experiment.status = ExperimentStatus.COMPLETE
                experiment.completed_at = datetime.utcnow()
                experiment.results = {
                    'total_nodes': len(tree.nodes) if tree else 0,
                    'stats': tree.stats if tree and hasattr(tree, 'stats') else {}
                }
                await db_session.commit()
            break

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)

        # Update experiment status to FAILED
        try:
            async for db_session in get_session():
                result = await db_session.execute(
                    sql_select(Experiment).where(Experiment.id == experiment_id)
                )
                experiment = result.scalar_one_or_none()
                if experiment:
                    experiment.status = ExperimentStatus.FAILED
                    experiment.error_message = str(e)
                    experiment.completed_at = datetime.utcnow()
                    await db_session.commit()
                break
        except Exception as db_error:
            logger.error(f"Failed to update experiment status: {db_error}")

    finally:
        # Cleanup orchestrator
        _active_orchestrators.pop(experiment_id, None)


# Request/Response Models
class StartResearchRequest(BaseModel):
    """Request to start research experiment"""
    goal: str
    session_id: str
    research_type: str = "scientific"  # scientific, code, roma
    config: Optional[dict] = None


class ExperimentResponse(BaseModel):
    """Experiment response"""
    id: str
    session_id: str
    experiment_type: str
    goal: str
    status: str
    progress_percentage: float
    current_step: Optional[str] = None
    results: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class IdeaGenerationRequest(BaseModel):
    """Request to generate ideas"""
    topic: str
    context: Optional[str] = None
    num_ideas: int = 5
    creativity: float = 0.7


class HypothesisGenerationRequest(BaseModel):
    """Request to generate hypotheses"""
    idea: str
    background: Optional[str] = None
    num_hypotheses: int = 3


# Endpoints

@router.post("/experiments/start", response_model=ExperimentResponse)
async def start_experiment(
    request: StartResearchRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """
    Start a new research experiment.

    Creates an experiment record and initiates background execution.
    """
    logger.info(f"Starting experiment: {request.research_type}")
    logger.info(f"Goal: {request.goal}")

    # Create experiment ID
    experiment_id = f"exp_{request.session_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # Validate research type
    try:
        exp_type = ExperimentType(request.research_type)
    except ValueError:
        raise HTTPException(400, f"Invalid research type: {request.research_type}")

    # Create experiment record
    experiment = Experiment(
        id=experiment_id,
        session_id=request.session_id,
        experiment_type=exp_type,
        goal=request.goal,
        config=request.config,
        status=ExperimentStatus.PENDING,
        created_at=datetime.utcnow(),
    )

    session.add(experiment)
    await session.commit()
    await session.refresh(experiment)

    logger.info(f"Created experiment: {experiment_id}")

    # Start experiment in background
    if ORCHESTRATOR_AVAILABLE:
        background_tasks.add_task(
            run_experiment_async,
            experiment_id,
            request.goal,
            request.config
        )
        logger.info(f"Background task scheduled for experiment {experiment_id}")
        print(f"[DEBUG] Background task added for experiment {experiment_id}, ORCHESTRATOR_AVAILABLE={ORCHESTRATOR_AVAILABLE}", flush=True)
    else:
        logger.warning(f"Orchestrator not available, experiment {experiment_id} will not execute")
        print(f"[DEBUG] Orchestrator not available, experiment {experiment_id} NOT executing", flush=True)

    return ExperimentResponse(
        id=experiment.id,
        session_id=experiment.session_id,
        experiment_type=experiment.experiment_type.value,
        goal=experiment.goal,
        status=experiment.status.value,
        progress_percentage=experiment.progress_percentage,
        current_step=experiment.current_step,
        results=experiment.results,
        error_message=experiment.error_message,
        created_at=experiment.created_at.isoformat(),
        started_at=experiment.started_at.isoformat() if experiment.started_at else None,
        completed_at=experiment.completed_at.isoformat() if experiment.completed_at else None,
    )


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get experiment status and results.
    """
    result = await session.execute(
        sql_select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    return ExperimentResponse(
        id=experiment.id,
        session_id=experiment.session_id,
        experiment_type=experiment.experiment_type.value,
        goal=experiment.goal,
        status=experiment.status.value,
        progress_percentage=experiment.progress_percentage,
        current_step=experiment.current_step,
        results=experiment.results,
        error_message=experiment.error_message,
        created_at=experiment.created_at.isoformat(),
        started_at=experiment.started_at.isoformat() if experiment.started_at else None,
        completed_at=experiment.completed_at.isoformat() if experiment.completed_at else None,
    )


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session)
):
    """
    List experiments with optional filters.
    """
    query = sql_select(Experiment)

    if session_id:
        query = query.where(Experiment.session_id == session_id)

    if status:
        try:
            status_enum = ExperimentStatus(status)
            query = query.where(Experiment.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    # Order by creation time descending
    query = query.order_by(Experiment.created_at.desc())

    # Apply pagination
    query = query.limit(limit).offset(offset)

    result = await session.execute(query)
    experiments = result.scalars().all()

    return [
        ExperimentResponse(
            id=exp.id,
            session_id=exp.session_id,
            experiment_type=exp.experiment_type.value,
            goal=exp.goal,
            status=exp.status.value,
            progress_percentage=exp.progress_percentage,
            current_step=exp.current_step,
            results=exp.results,
            error_message=exp.error_message,
            created_at=exp.created_at.isoformat(),
            started_at=exp.started_at.isoformat() if exp.started_at else None,
            completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
        )
        for exp in experiments
    ]


@router.delete("/experiments/{experiment_id}")
async def cancel_experiment(
    experiment_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Cancel running experiment.
    """
    result = await session.execute(
        sql_select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    if experiment.status not in [ExperimentStatus.PENDING, ExperimentStatus.RUNNING]:
        raise HTTPException(
            400,
            f"Cannot cancel experiment in status {experiment.status.value}"
        )

    # Update status to cancelled
    experiment.status = ExperimentStatus.CANCELLED
    experiment.completed_at = datetime.utcnow()
    await session.commit()

    logger.info(f"Cancelled experiment: {experiment_id}")

    return {"status": "cancelled", "experiment_id": experiment_id}


@router.get("/sessions/{session_id}/tree")
async def get_research_tree(
    session_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get ROMA research tree structure.
    """
    try:
        result = await session.execute(
            sql_select(ResearchSession).where(ResearchSession.id == session_id)
        )
        research_session = result.scalar_one_or_none()

        # Return empty tree if session not found (graceful handling)
        if not research_session:
            return {
                'session_id': session_id,
                'tree': {},
                'updated_at': None,
            }

        # Ensure research_tree is a dict and handle None case
        research_tree = research_session.research_tree if research_session.research_tree is not None else {}

        return {
            'session_id': session_id,
            'tree': research_tree,
            'updated_at': research_session.updated_at.isoformat() if research_session.updated_at else None,
        }
    except Exception as e:
        logger.error(f"Error fetching research tree for session {session_id}: {str(e)}")
        # Return empty tree on error
        return {
            'session_id': session_id,
            'tree': {},
            'updated_at': None,
        }


@router.post("/ideas/generate")
async def generate_ideas(
    request: IdeaGenerationRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Generate research ideas using AI.
    """
    # In production, would use LLM to generate ideas
    # For now, return placeholder response

    logger.info(f"Generating {request.num_ideas} ideas for topic: {request.topic}")

    # TODO: Implement actual idea generation with LLM
    ideas = []
    for i in range(min(request.num_ideas, 10)):
        idea = Idea(
            id=f"idea_{uuid.uuid4().hex[:8]}",
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            title=f"Research Idea {i+1} for {request.topic}",
            description="Placeholder idea description",
            topic=request.topic,
            novelty_score=0.7,
            feasibility_score=0.8,
            impact_score=0.6,
        )
        ideas.append(idea.to_dict())

    return {
        'topic': request.topic,
        'ideas': ideas,
        'generated_at': datetime.utcnow().isoformat(),
    }


@router.post("/hypotheses/generate")
async def generate_hypotheses(
    request: HypothesisGenerationRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Generate testable hypotheses from an idea.
    """
    logger.info(f"Generating {request.num_hypotheses} hypotheses for idea")

    # TODO: Implement actual hypothesis generation with LLM
    hypotheses = []
    for i in range(min(request.num_hypotheses, 10)):
        hypothesis = Hypothesis(
            id=f"hyp_{uuid.uuid4().hex[:8]}",
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            statement=f"Hypothesis {i+1} based on: {request.idea[:50]}...",
            null_hypothesis=f"Null hypothesis {i+1}",
            testability_score=0.8,
        )
        hypotheses.append(hypothesis.to_dict())

    return {
        'idea': request.idea,
        'hypotheses': hypotheses,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "extension": "uagent_research",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Phase 2: Research Tree Endpoints
# ============================================================================

# Import middleware to access orchestrators
try:
    from ...middleware.research_middleware import research_middleware
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False
    logger.warning("Research middleware not available for API routes")

# Global storage for active orchestrators (in production, use Redis/DB)
_active_orchestrators = {}

# Global session manager for research state tracking
_session_manager: Optional[ResearchSessionManager] = None

def get_session_manager() -> Optional[ResearchSessionManager]:
    """Get or create global session manager"""
    global _session_manager

    if not CONTROL_BUS_AVAILABLE:
        return None

    if _session_manager is None:
        try:
            event_bus = get_event_bus()
            control_bus = ControlBus()
            _session_manager = ResearchSessionManager(
                event_bus=event_bus,
                control_bus=control_bus
            )
            logger.info("ResearchSessionManager initialized for API")
        except Exception as e:
            logger.error(f"Failed to initialize session manager: {e}")
            return None

    return _session_manager


class TreeNodeResponse(BaseModel):
    """Tree node response"""
    id: str
    type: str
    title: str
    content: str
    status: str
    visits: int
    prior: float
    avg_value: float
    cost: float
    tokens_used: int
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class TreeSnapshotResponse(BaseModel):
    """Research tree snapshot response"""
    version: int
    timestamp: str
    experiment_id: str
    data: dict


@router.get("/experiments/{experiment_id}/tree", response_model=TreeSnapshotResponse)
async def get_experiment_tree(
    experiment_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get research tree snapshot for experiment.

    Returns full tree state with version for incremental updates.
    """
    try:
        # Check experiment exists
        result = await session.execute(
            sql_select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()

        # Even if experiment doesn't exist yet, return empty tree for graceful handling
        # This prevents 404 errors when accessing research tree before experiment starts
        if not experiment:
            return TreeSnapshotResponse(
                version=0,
                timestamp=datetime.utcnow().isoformat(),
                experiment_id=experiment_id,
                data={
                    "nodes": [],
                    "edges": [],
                    "stats": {}
                }
            )

        # Get orchestrator (if running) - check middleware first
        orchestrator = None
        if MIDDLEWARE_AVAILABLE:
            orchestrator = research_middleware.get_orchestrator(experiment_id)
        if not orchestrator:
            orchestrator = _active_orchestrators.get(experiment_id)

        if not orchestrator or not hasattr(orchestrator, 'tree') or not orchestrator.tree:
            # Return empty tree
            return TreeSnapshotResponse(
                version=0,
                timestamp=datetime.utcnow().isoformat(),
                experiment_id=experiment_id,
                data={
                    "nodes": [],
                    "edges": [],
                    "stats": {}
                }
            )

        tree = orchestrator.tree

        # Build tree snapshot with proper error handling
        nodes = []
        edges = []
        stats = {}

        try:
            # Build nodes list with proper error handling
            if hasattr(tree, 'nodes') and tree.nodes:
                nodes = [
                    {
                        "id": getattr(node, 'id', ''),
                        "type": getattr(node, 'type', {}).value if hasattr(getattr(node, 'type', None), 'value') else 'unknown',
                        "title": getattr(node, 'title', ''),
                        "content": getattr(node, 'content', ''),
                        "status": getattr(node, 'status', {}).value if hasattr(getattr(node, 'status', None), 'value') else 'unknown',
                        "visits": getattr(node, 'visits', 0),
                        "prior": getattr(node, 'prior', 0.0),
                        "avg_value": getattr(node, 'avg_value', 0.0),
                        "cost": getattr(node, 'cost', 0.0),
                        "tokens_used": getattr(node, 'tokens_used', 0),
                        "created_at": node.created_at.isoformat() if hasattr(node, 'created_at') and node.created_at else None,
                        "completed_at": node.completed_at.isoformat() if hasattr(node, 'completed_at') and node.completed_at else None,
                    }
                    for node in tree.nodes.values()
                    if hasattr(node, 'id')  # Only include nodes with valid IDs
                ]

            # Build edges list with proper error handling
            if hasattr(tree, 'children') and tree.children:
                edges = [
                    {"parent_id": str(parent_id), "child_id": str(child_id)}
                    for parent_id, children in tree.children.items()
                    if parent_id is not None
                    for child_id in children
                    if child_id is not None
                ]

            # Get stats with proper error handling
            stats = getattr(tree, 'stats', {}) if hasattr(tree, 'stats') else {}

        except Exception as e:
            logger.error(f"Error building tree snapshot for experiment {experiment_id}: {str(e)}")
            # Return empty tree on error
            return TreeSnapshotResponse(
                version=0,
                timestamp=datetime.utcnow().isoformat(),
                experiment_id=experiment_id,
                data={
                    "nodes": [],
                    "edges": [],
                    "stats": {}
                }
            )

        return TreeSnapshotResponse(
            version=getattr(orchestrator, 'version', 0),
            timestamp=datetime.utcnow().isoformat(),
            experiment_id=experiment_id,
            data={
                "nodes": nodes,
                "edges": edges,
                "stats": stats
            }
        )
    except Exception as e:
        logger.error(f"Error fetching research tree for experiment {experiment_id}: {str(e)}")
        # Return empty tree on any error
        return TreeSnapshotResponse(
            version=0,
            timestamp=datetime.utcnow().isoformat(),
            experiment_id=experiment_id,
            data={
                "nodes": [],
                "edges": [],
                "stats": {}
            }
        )


class ExperimentControlRequest(BaseModel):
    """Experiment control request"""
    action: str  # pause, resume, cancel, cancel_node, reprioritize, steer, add_node
    target: Optional[Dict[str, str]] = Field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)


@router.patch("/experiments/{experiment_id}")
async def control_experiment(
    experiment_id: str,
    request: ExperimentControlRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Control experiment execution.

    Supported actions:
    - pause: Pause research (stop scheduling new nodes)
    - resume: Resume paused research
    - cancel: Cancel entire experiment
    - cancel_node: Cancel specific node (requires target.node_id)
    - reprioritize: Adjust priorities (requires target and payload.delta)
    - steer: Send guidance to adapter (requires target and payload.text)
    - add_node: Add new research direction (requires payload.parent_id and payload.node)
    """
    # Check experiment exists
    result = await session.execute(
        sql_select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    # Try to use session manager first (if available)
    session_mgr = get_session_manager()

    if session_mgr and CONTROL_BUS_AVAILABLE:
        # Use ControlBus for all control actions
        try:
            control_msg = ControlMessage(
                action=request.action,
                target=request.target or {},
                payload=request.payload or {},
                sender="api"
            )

            await session_mgr.send_control(experiment_id, control_msg)

            # Update database for terminal actions
            if request.action == "cancel":
                experiment.status = ExperimentStatus.CANCELLED
                experiment.completed_at = datetime.utcnow()
                await session.commit()
                _active_orchestrators.pop(experiment_id, None)

            return {
                "status": "acknowledged",
                "experiment_id": experiment_id,
                "action": request.action,
                "message": f"Control command '{request.action}' sent successfully"
            }

        except Exception as e:
            logger.error(f"Error sending control command: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to send control command: {str(e)}")

    else:
        # Fallback to direct orchestrator control (legacy)
        orchestrator = _active_orchestrators.get(experiment_id)

        if not orchestrator:
            raise HTTPException(400, f"Experiment {experiment_id} not running")

        # Only support cancel in legacy mode
        if request.action == "cancel":
            await orchestrator.cancel()
            experiment.status = ExperimentStatus.CANCELLED
            experiment.completed_at = datetime.utcnow()
            await session.commit()
            _active_orchestrators.pop(experiment_id, None)

            return {"status": "cancelled", "experiment_id": experiment_id}
        else:
            raise HTTPException(501, f"Action '{request.action}' requires ControlBus (not available)")


@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status(
    experiment_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Get real-time experiment status with detailed progress information.

    Returns:
        - experiment_id: Experiment ID
        - status: Current status (running/paused/complete/failed)
        - stats: Node statistics (total, completed, failed, running, etc.)
        - adapters: Status of each adapter
        - active_branches: Currently running branches
        - created_at: Creation timestamp
        - last_update: Last status update timestamp
    """
    # Check experiment exists
    result = await session.execute(
        sql_select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    # Try to get status from session manager
    session_mgr = get_session_manager()

    if session_mgr:
        try:
            status = session_mgr.get_status(experiment_id)
            return status
        except KeyError:
            # Experiment not registered in session manager
            pass
        except Exception as e:
            logger.error(f"Error getting status from session manager: {e}")

    # Fallback: return basic status from database
    return {
        "experiment_id": experiment_id,
        "status": experiment.status.value if experiment.status else "unknown",
        "stats": {
            "total_nodes": 0,
            "completed": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "total_cost": 0.0,
            "total_tokens": 0
        },
        "adapters": {},
        "active_branches": [],
        "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
        "last_update": experiment.updated_at.isoformat() if hasattr(experiment, 'updated_at') and experiment.updated_at else None,
        "message": "Detailed status not available (session manager not tracking this experiment)"
    }


@router.get("/experiments/{experiment_id}/events")
async def get_experiment_events(
    experiment_id: str,
    since_version: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_session)
):
    """
    Get experiment events since version (incremental fetch fallback to WebSocket).

    This is a fallback for when WebSocket is not available.
    In production, events would be stored in a database or event log.
    """
    # Check experiment exists
    result = await session.execute(
        sql_select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

    # TODO: Implement event storage and retrieval
    # For now, return empty list
    return {
        "events": [],
        "current_version": since_version,
        "has_more": False
    }
