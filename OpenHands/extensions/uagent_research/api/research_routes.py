"""Research API routes for tree visualization and control."""

import logging
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

# In-memory storage for active experiments
# In production, this should be in a database or Redis
_active_trees: Dict[str, dict] = {}


class TreeSnapshotResponse(BaseModel):
    """Response model for tree snapshot"""
    version: int
    timestamp: str
    experiment_id: str
    data: dict


class ExperimentControlRequest(BaseModel):
    """Request model for experiment control"""
    action: str  # pause, resume, cancel


@router.get("/experiments/{experiment_id}/tree")
async def get_experiment_tree(experiment_id: str) -> TreeSnapshotResponse:
    """
    Get research tree snapshot for an experiment.

    This endpoint provides the current state of the research tree including:
    - All nodes with their status, PUCT metrics, and content
    - All edges connecting nodes
    - Aggregate statistics (total cost, tokens, etc.)

    Args:
        experiment_id: The unique identifier for the experiment

    Returns:
        TreeSnapshotResponse with version-tracked tree state
    """
    logger.info(f"Fetching tree for experiment {experiment_id}")

    # Check if experiment has an active tree
    if experiment_id not in _active_trees:
        # Return empty tree for now
        # In production, you would fetch from database or create new tree
        logger.info(f"No active tree for experiment {experiment_id}, returning empty tree")
        return TreeSnapshotResponse(
            version=0,
            timestamp=datetime.utcnow().isoformat(),
            experiment_id=experiment_id,
            data={
                "nodes": [],
                "edges": [],
                "stats": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "completed_nodes": 0,
                    "failed_nodes": 0,
                }
            }
        )

    tree_data = _active_trees[experiment_id]
    return TreeSnapshotResponse(**tree_data)


@router.patch("/experiments/{experiment_id}")
async def control_experiment(
    experiment_id: str,
    request: ExperimentControlRequest
) -> dict:
    """
    Control experiment execution (pause, resume, cancel).

    Args:
        experiment_id: The unique identifier for the experiment
        request: Control request with action to perform

    Returns:
        Status message
    """
    logger.info(f"Control request for experiment {experiment_id}: {request.action}")

    valid_actions = {"start", "pause", "resume", "cancel"}
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action. Must be one of: {valid_actions}"
        )

    if experiment_id not in _active_trees:
        if request.action == "start":
            logger.info(
                "Initializing empty tree for experiment %s on start action",
                experiment_id,
            )
            _active_trees[experiment_id] = {
                "version": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "experiment_id": experiment_id,
                "data": {
                    "nodes": [],
                    "edges": [],
                    "stats": {
                        "total_nodes": 0,
                        "total_edges": 0,
                        "total_cost": 0.0,
                        "total_tokens": 0,
                        "completed_nodes": 0,
                        "failed_nodes": 0,
                    },
                },
            }
        else:
            raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": experiment_id,
        "action": request.action,
        "status": "acknowledged",
        "message": f"Experiment {request.action} request acknowledged"
    }


@router.get("/experiments/{experiment_id}/events")
async def get_experiment_events(
    experiment_id: str,
    since_version: int = 0,
    limit: int = 100
) -> dict:
    """
    Get incremental events for an experiment since a given version.

    This is a fallback to WebSocket for clients that need to poll
    or recover from connection drops.

    Args:
        experiment_id: The unique identifier for the experiment
        since_version: Get events after this version number
        limit: Maximum number of events to return

    Returns:
        List of events and current version
    """
    logger.info(
        f"Fetching events for experiment {experiment_id} "
        f"since version {since_version}, limit {limit}"
    )

    # TODO: Implement event log retrieval
    # For now, return empty list
    return {
        "experiment_id": experiment_id,
        "since_version": since_version,
        "current_version": since_version,
        "events": [],
        "has_more": False
    }


# Helper function to update tree state (used by orchestrator)
def update_tree_state(experiment_id: str, tree_data: dict):
    """
    Update the tree state for an experiment.

    This should be called by the tree orchestrator when tree state changes.
    """
    _active_trees[experiment_id] = tree_data
    logger.debug(f"Updated tree state for experiment {experiment_id}")


# Helper function to clear tree state
def clear_tree_state(experiment_id: str):
    """Remove tree state when experiment completes."""
    if experiment_id in _active_trees:
        del _active_trees[experiment_id]
        logger.debug(f"Cleared tree state for experiment {experiment_id}")
