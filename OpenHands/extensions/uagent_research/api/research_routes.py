"""Research API routes for tree visualization and control."""

import logging
import traceback
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

# In-memory storage for active experiments
# In production, this should be in a database or Redis
_active_trees: Dict[str, dict] = {}

# Track API initialization
_api_initialized = False


class TreeSnapshotResponse(BaseModel):
    """Response model for tree snapshot"""
    version: int
    timestamp: str
    experiment_id: str
    data: dict


class ExperimentControlRequest(BaseModel):
    """Request model for experiment control"""
    action: str  # pause, resume, cancel


def initialize_research_api():
    """Initialize the research API with diagnostic logging."""
    global _api_initialized
    if not _api_initialized:
        logger.info("🚀 Initializing Research API routes")
        logger.debug(f"📊 Active trees storage initialized: {_active_trees}")
        _api_initialized = True
        logger.info("✅ Research API initialization complete")
    else:
        logger.debug("ℹ️ Research API already initialized")


@router.get("/diagnostics")
async def get_research_diagnostics():
    """Get diagnostic information about the research API state."""
    try:
        logger.info("📊 Research API diagnostics requested")
        
        diagnostics = {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "api_initialized": _api_initialized,
            "active_experiments": list(_active_trees.keys()),
            "total_experiments": len(_active_trees),
            "experiment_details": {}
        }
        
        # Add details for each experiment
        for exp_id, tree_data in _active_trees.items():
            try:
                data = tree_data.get('data', {})
                stats = data.get('stats', {})
                diagnostics["experiment_details"][exp_id] = {
                    "version": tree_data.get('version', 0),
                    "nodes_count": len(data.get('nodes', [])),
                    "edges_count": len(data.get('edges', [])),
                    "stats": stats,
                    "last_update": tree_data.get('timestamp', 'unknown')
                }
            except Exception as e:
                logger.error(f"❌ Error getting details for experiment {exp_id}: {e}")
                diagnostics["experiment_details"][exp_id] = {"error": str(e)}
        
        logger.debug(f"📊 Diagnostics: {diagnostics}")
        return diagnostics
        
    except Exception as e:
        logger.error(f"❌ Failed to get research diagnostics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat()
        }


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
    try:
        logger.info(f"📡 Tree requested for experiment {experiment_id}")
        logger.debug(f"🔍 Checking active trees: {list(_active_trees.keys())}")

        # Check if experiment has an active tree
        if experiment_id not in _active_trees:
            logger.info(f"ℹ️ No active tree for experiment {experiment_id}, returning empty tree")
            # Return empty tree for now
            # In production, you would fetch from database or create new tree
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
        logger.info(
            f"✅ Returning tree for {experiment_id}: "
            f"version={tree_data.get('version', 0)}, "
            f"nodes={len(tree_data.get('data', {}).get('nodes', []))}"
        )
        return TreeSnapshotResponse(**tree_data)
        
    except Exception as e:
        logger.error(f"❌ Error fetching tree for {experiment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch tree: {str(e)}"
        )


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
    try:
        logger.info(f"🎮 Control request for experiment {experiment_id}: {request.action}")

        valid_actions = {"start", "pause", "resume", "cancel"}
        if request.action not in valid_actions:
            logger.warning(f"⚠️ Invalid action '{request.action}' for experiment {experiment_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {valid_actions}"
            )

        if experiment_id not in _active_trees:
            if request.action == "start":
                logger.info(f"🚀 Initializing empty tree for experiment {experiment_id} on start action")
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
                logger.debug(f"✅ Empty tree initialized for {experiment_id}")
            else:
                logger.warning(f"⚠️ Experiment {experiment_id} not found for action {request.action}")
                raise HTTPException(status_code=404, detail="Experiment not found")

        response = {
            "experiment_id": experiment_id,
            "action": request.action,
            "status": "acknowledged",
            "message": f"Experiment {request.action} request acknowledged",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Control request acknowledged for {experiment_id}: {request.action}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error controlling experiment {experiment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to control experiment: {str(e)}"
        )


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
    try:
        logger.info(
            f"📡 Fetching events for experiment {experiment_id} "
            f"since version {since_version}, limit {limit}"
        )

        # TODO: Implement event log retrieval
        # For now, return empty list
        response = {
            "experiment_id": experiment_id,
            "since_version": since_version,
            "current_version": since_version,
            "events": [],
            "has_more": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"✅ Events response for {experiment_id}: {len(response['events'])} events")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error fetching events for {experiment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch events: {str(e)}"
        )


# Helper function to update tree state (used by orchestrator)
def update_tree_state(experiment_id: str, tree_data: dict):
    """Update the tree state for an experiment."""
    try:
        if not experiment_id:
            logger.error("❌ Cannot update tree state: experiment_id is empty")
            return
        
        if not tree_data or not isinstance(tree_data, dict):
            logger.error(f"❌ Cannot update tree state for {experiment_id}: invalid tree_data (type: {type(tree_data)})")
            return
        
        logger.debug(f"🔄 Updating tree state for {experiment_id}")
        _active_trees[experiment_id] = tree_data
        
        # Log summary
        nodes_count = len(tree_data.get('data', {}).get('nodes', []))
        edges_count = len(tree_data.get('data', {}).get('edges', []))
        version = tree_data.get('version', 0)
        
        logger.info(
            f"✅ Tree state updated for {experiment_id}: "
            f"version={version}, nodes={nodes_count}, edges={edges_count}"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to update tree state for {experiment_id}: {e}", exc_info=True)


def clear_tree_state(experiment_id: str):
    """Remove tree state when experiment completes."""
    try:
        if experiment_id in _active_trees:
            nodes_count = len(_active_trees[experiment_id].get('data', {}).get('nodes', []))
            logger.info(f"🧹 Clearing tree state for experiment {experiment_id} (had {nodes_count} nodes)")
            del _active_trees[experiment_id]
            logger.debug(f"✅ Tree state cleared for experiment {experiment_id}")
        else:
            logger.debug(f"ℹ️ No tree state to clear for experiment {experiment_id}")
    except Exception as e:
        logger.error(f"❌ Error clearing tree state for {experiment_id}: {e}", exc_info=True)


# Initialize on module load
logger.info("📦 research_routes module loaded")
