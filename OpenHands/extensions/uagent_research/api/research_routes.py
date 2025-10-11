"""Research API routes for tree visualization and control."""

import logging
import os
import secrets
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from ..config import CONFIG_SUMMARY
from ..utils.security import SlidingWindowRateLimiter, sanitize_identifier

logger = logging.getLogger(__name__)



def _parse_positive_int(env_var: str, default: int) -> int:
    """Parse a positive integer environment variable with sane fallback."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "Invalid integer for %s=%r; falling back to default %s",
            env_var,
            value,
            default,
        )
        return default
    if parsed <= 0:
        logger.warning(
            "%s must be positive (got %s); using default %s",
            env_var,
            parsed,
            default,
        )
        return default
    return parsed


_TREE_TTL_SECONDS = _parse_positive_int("UAGENT_ACTIVE_TREE_TTL_SECONDS", 3600)
_TREE_MAX_ENTRIES = _parse_positive_int("UAGENT_MAX_ACTIVE_TREES", 200)

_API_RATE_LIMIT = _parse_positive_int("UAGENT_API_RATE_LIMIT", 120)
_API_RATE_WINDOW_SECONDS = _parse_positive_int("UAGENT_API_RATE_WINDOW_SECONDS", 60)
_RESEARCH_API_TOKEN = os.getenv("UAGENT_RESEARCH_API_TOKEN")


def _create_rate_limiter() -> SlidingWindowRateLimiter:
    try:
        return SlidingWindowRateLimiter(_API_RATE_LIMIT, _API_RATE_WINDOW_SECONDS)
    except ValueError as exc:  # pragma: no cover - defensive fallback
        logger.warning("Invalid API rate limit configuration: %s", exc)
        return SlidingWindowRateLimiter(120, 60)


_RATE_LIMITER = _create_rate_limiter()


def _authorize_request(x_research_token: Optional[str] = Header(default=None)) -> None:
    """Optional header-based authentication for research endpoints."""
    if not _RESEARCH_API_TOKEN:
        return
    if not x_research_token or not secrets.compare_digest(x_research_token, _RESEARCH_API_TOKEN):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def _validate_experiment_id(experiment_id: str) -> str:
    try:
        return sanitize_identifier("experiment_id", experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def _enforce_rate_limit(endpoint: str, experiment_id: str) -> None:
    key = f"{endpoint}:{experiment_id}"
    if not _RATE_LIMITER.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for experiment",
        )


router = APIRouter(
    prefix="/api/research",
    tags=["research"],
    dependencies=[Depends(_authorize_request)],
)


@dataclass
class _TreeEntry:
    data: dict
    updated_at: datetime

    @property
    def expires_at(self) -> datetime:
        return self.updated_at + timedelta(seconds=_TREE_TTL_SECONDS)


class _ActiveTreeStore:
    """Track active experiments with TTL + LRU eviction."""

    def __init__(self) -> None:
        self._entries: "OrderedDict[str, _TreeEntry]" = OrderedDict()
        self._lock = RLock()

    def _purge_locked(self, now: datetime) -> None:
        if _TREE_TTL_SECONDS:
            expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
            for key in expired:
                data = self._entries.pop(key)
                logger.info(
                    "🧹 Expired tree state for experiment %s removed; last update %s",
                    key,
                    data.updated_at.isoformat(),
                )
        while _TREE_MAX_ENTRIES and len(self._entries) > _TREE_MAX_ENTRIES:
            key, data = self._entries.popitem(last=False)
            logger.warning(
                "🧹 Evicted oldest tree state for experiment %s to enforce max entries %s",
                key,
                _TREE_MAX_ENTRIES,
            )

    def get(self, experiment_id: str) -> Optional[dict]:
        now = datetime.utcnow()
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(experiment_id)
            if not entry:
                return None
            # Maintain LRU order
            self._entries.move_to_end(experiment_id)
            return entry.data

    def set(self, experiment_id: str, tree_data: dict) -> None:
        now = datetime.utcnow()
        with self._lock:
            self._entries[experiment_id] = _TreeEntry(tree_data, now)
            self._entries.move_to_end(experiment_id)
            self._purge_locked(now)

    def delete(self, experiment_id: str) -> None:
        with self._lock:
            removed = self._entries.pop(experiment_id, None)
        if removed:
            logger.info(
                "🧹 Cleared tree state for experiment %s (nodes=%s)",
                experiment_id,
                len(removed.data.get("data", {}).get("nodes", [])),
            )

    def keys(self) -> List[str]:
        now = datetime.utcnow()
        with self._lock:
            self._purge_locked(now)
            return list(self._entries.keys())

    def items(self) -> List[Tuple[str, dict]]:
        now = datetime.utcnow()
        with self._lock:
            self._purge_locked(now)
            return [(key, entry.data) for key, entry in self._entries.items()]

    def stats(self) -> Dict[str, int]:
        now = datetime.utcnow()
        with self._lock:
            self._purge_locked(now)
            return {
                "count": len(self._entries),
                "max_entries": _TREE_MAX_ENTRIES,
                "ttl_seconds": _TREE_TTL_SECONDS,
            }


_active_tree_store = _ActiveTreeStore()

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
        logger.debug(
            "📊 Active trees storage initialized: %s",
            _active_tree_store.stats(),
        )
        _api_initialized = True
        logger.info("✅ Research API initialization complete")
    else:
        logger.debug("ℹ️ Research API already initialized")


@router.get("/diagnostics")
async def get_research_diagnostics():
    """Get diagnostic information about the research API state."""
    try:
        logger.info("📊 Research API diagnostics requested")
        
        store_stats = _active_tree_store.stats()
        diagnostics = {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "api_initialized": _api_initialized,
            "active_experiments": _active_tree_store.keys(),
            "total_experiments": store_stats["count"],
            "storage_limits": {
                "ttl_seconds": _TREE_TTL_SECONDS,
                "max_entries": _TREE_MAX_ENTRIES,
            },
            "rate_limit": {
                "limit": _API_RATE_LIMIT,
                "window_seconds": _API_RATE_WINDOW_SECONDS,
            },
            "configuration": CONFIG_SUMMARY,
            "experiment_details": {}
        }

        # Add details for each experiment
        for exp_id, tree_data in _active_tree_store.items():
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
        experiment_id = _validate_experiment_id(experiment_id)
        _enforce_rate_limit("tree", experiment_id)
        logger.info(f"📡 Tree requested for experiment {experiment_id}")
        logger.debug(f"🔍 Checking active trees: {_active_tree_store.keys()}")

        # Check if experiment has an active tree
        tree_data = _active_tree_store.get(experiment_id)
        if tree_data is None:
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
        experiment_id = _validate_experiment_id(experiment_id)
        _enforce_rate_limit("control", experiment_id)
        logger.info(f"🎮 Control request for experiment {experiment_id}: {request.action}")

        valid_actions = {"start", "pause", "resume", "cancel"}
        if request.action not in valid_actions:
            logger.warning(f"⚠️ Invalid action '{request.action}' for experiment {experiment_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action. Must be one of: {valid_actions}"
            )

        if _active_tree_store.get(experiment_id) is None:
            if request.action == "start":
                logger.info(f"🚀 Initializing empty tree for experiment {experiment_id} on start action")
                _active_tree_store.set(experiment_id, {
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
                })
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
        experiment_id = _validate_experiment_id(experiment_id)
        _enforce_rate_limit("events", experiment_id)
        limit = max(1, min(limit, 500))
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
        try:
            experiment_id = sanitize_identifier("experiment_id", experiment_id)
        except ValueError as exc:
            logger.error("❌ Cannot update tree state: %s", exc)
            return

        if not tree_data or not isinstance(tree_data, dict):
            logger.error(f"❌ Cannot update tree state for {experiment_id}: invalid tree_data (type: {type(tree_data)})")
            return

        logger.debug(f"🔄 Updating tree state for {experiment_id}")
        _active_tree_store.set(experiment_id, tree_data)

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
        try:
            experiment_id = sanitize_identifier("experiment_id", experiment_id)
        except ValueError as exc:
            logger.error("❌ Invalid experiment_id during clear: %s", exc)
            return

        if _active_tree_store.get(experiment_id) is None:
            logger.debug(f"ℹ️ No tree state to clear for experiment {experiment_id}")
            return
        _active_tree_store.delete(experiment_id)
    except Exception as e:
        logger.error(f"❌ Error clearing tree state for {experiment_id}: {e}", exc_info=True)


# Initialize on module load
logger.info("📦 research_routes module loaded")
