"""Experiment management API endpoints"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.app_state import get_app_state
from ..core.experiment_manager import get_experiment_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ExperimentResponse(BaseModel):
    """Response model for experiment information"""
    session_id: str
    original_query: str
    status: str
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    readable_name: Optional[str]
    success: bool
    arxiv_path: str
    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]
    archived_at: str


class ExperimentListResponse(BaseModel):
    """Response model for experiment list"""
    experiments: List[ExperimentResponse]
    total_count: int
    successful_count: int
    failed_count: int
    interrupted_count: int


class ActiveExperimentResponse(BaseModel):
    """Response model for active experiment information"""
    session_id: str
    original_query: str
    status: str
    start_time: str
    workspace_path: Optional[str]


@router.get("/archived", response_model=ExperimentListResponse)
async def list_archived_experiments(
    status_filter: Optional[str] = Query(None, description="Filter by status: successful, failed, interrupted"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of experiments to return")
):
    """List archived experiments

    Args:
        status_filter: Optional filter by experiment status
        limit: Maximum number of experiments to return

    Returns:
        List of archived experiments with metadata
    """
    try:
        experiment_manager = get_experiment_manager()
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment manager not available")

        # Get archived experiments
        experiments = experiment_manager.list_archived_experiments(status_filter)

        # Apply limit
        limited_experiments = experiments[:limit]

        # Count by status
        successful_count = sum(1 for exp in experiments if exp.get("status") == "success")
        failed_count = sum(1 for exp in experiments if exp.get("status") == "failed")
        interrupted_count = sum(1 for exp in experiments if exp.get("status") == "interrupted")

        # Convert to response format
        response_experiments = []
        for exp in limited_experiments:
            response_experiments.append(ExperimentResponse(
                session_id=exp["session_id"],
                original_query=exp["original_query"],
                status=exp["status"],
                start_time=exp["start_time"],
                end_time=exp.get("end_time"),
                duration_seconds=exp.get("duration_seconds"),
                readable_name=exp.get("readable_name"),
                success=exp.get("success", False),
                arxiv_path=exp["arxiv_path"],
                final_result=exp.get("final_result"),
                error_message=exp.get("error_message"),
                archived_at=exp["archived_at"]
            ))

        return ExperimentListResponse(
            experiments=response_experiments,
            total_count=len(experiments),
            successful_count=successful_count,
            failed_count=failed_count,
            interrupted_count=interrupted_count
        )

    except Exception as e:
        logger.error(f"Error listing archived experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def list_active_experiments():
    """List currently active experiments

    Returns:
        List of active experiments
    """
    try:
        experiment_manager = get_experiment_manager()
        if not experiment_manager:
            return {"active_experiments": [], "count": 0}

        active_experiments = []
        for session_id, experiment in experiment_manager.active_experiments.items():
            active_experiments.append(ActiveExperimentResponse(
                session_id=experiment.session_id,
                original_query=experiment.original_query,
                status=experiment.status.value,
                start_time=experiment.start_time.isoformat(),
                workspace_path=str(experiment.workspace_path) if experiment.workspace_path else None
            ))

        return {
            "active_experiments": active_experiments,
            "count": len(active_experiments)
        }

    except Exception as e:
        logger.error(f"Error listing active experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_experiment_stats():
    """Get experiment statistics

    Returns:
        Statistics about experiments
    """
    try:
        experiment_manager = get_experiment_manager()
        if not experiment_manager:
            return {
                "active_count": 0,
                "archived_count": 0,
                "successful_count": 0,
                "failed_count": 0,
                "interrupted_count": 0,
                "arxiv_path": None
            }

        # Get active experiments
        active_count = len(experiment_manager.active_experiments)

        # Get archived experiments
        archived = experiment_manager.list_archived_experiments()

        successful_count = sum(1 for exp in archived if exp.get("status") == "success")
        failed_count = sum(1 for exp in archived if exp.get("status") == "failed")
        interrupted_count = sum(1 for exp in archived if exp.get("status") == "interrupted")

        return {
            "active_count": active_count,
            "archived_count": len(archived),
            "successful_count": successful_count,
            "failed_count": failed_count,
            "interrupted_count": interrupted_count,
            "arxiv_path": str(experiment_manager.arxiv_dir)
        }

    except Exception as e:
        logger.error(f"Error getting experiment stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-old")
async def cleanup_old_experiments(days_old: int = Query(30, ge=1, le=365, description="Remove experiments older than this many days")):
    """Clean up old archived experiments

    Args:
        days_old: Remove experiments older than this many days

    Returns:
        Cleanup result
    """
    try:
        experiment_manager = get_experiment_manager()
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment manager not available")

        await experiment_manager.cleanup_old_experiments(days_old)

        return {
            "message": f"Cleaned up experiments older than {days_old} days",
            "days_old": days_old
        }

    except Exception as e:
        logger.error(f"Error cleaning up old experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
async def get_experiment_details(session_id: str):
    """Get details for a specific experiment

    Args:
        session_id: Session identifier

    Returns:
        Experiment details
    """
    try:
        experiment_manager = get_experiment_manager()
        if not experiment_manager:
            raise HTTPException(status_code=503, detail="Experiment manager not available")

        # Check active experiments first
        if session_id in experiment_manager.active_experiments:
            experiment = experiment_manager.active_experiments[session_id]
            return {
                "session_id": experiment.session_id,
                "original_query": experiment.original_query,
                "status": experiment.status.value,
                "start_time": experiment.start_time.isoformat(),
                "workspace_path": str(experiment.workspace_path) if experiment.workspace_path else None,
                "is_active": True
            }

        # Check archived experiments
        archived = experiment_manager.list_archived_experiments()
        for exp in archived:
            if exp["session_id"] == session_id:
                return {
                    **exp,
                    "is_active": False
                }

        raise HTTPException(status_code=404, detail=f"Experiment {session_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment details for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))