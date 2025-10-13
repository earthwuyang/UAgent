"""
Tree State Publisher

Shared functions for updating and broadcasting tree state.
This module breaks the circular import between orchestrator and research_routes.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global storage for tree snapshots (shared with research_routes)
_active_tree_snapshots: Dict[str, dict] = {}

def update_tree_state(experiment_id: str, tree_data: Dict[str, Any]) -> None:
    """
    Update tree state in global storage.
    
    Args:
        experiment_id: Experiment ID
        tree_data: Tree data dictionary
    """
    try:
        _active_tree_snapshots[experiment_id] = tree_data
        
        # Also index by session_id so UI polling with conversation_id works
        if experiment_id.startswith('exp_'):
            parts = experiment_id.split('_')
            if len(parts) >= 3:
                session_id = parts[1]
                _active_tree_snapshots[session_id] = tree_data
        
        logger.debug(f"Updated tree state for experiment {experiment_id}")
    except Exception as e:
        logger.error(f"Failed to update tree state for {experiment_id}: {e}")

def get_tree_state(experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Get tree state from global storage.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        Tree data dictionary or None if not found
    """
    return _active_tree_snapshots.get(experiment_id)

def clear_tree_state(experiment_id: str) -> None:
    """
    Clear tree state from global storage.
    
    Args:
        experiment_id: Experiment ID
    """
    _active_tree_snapshots.pop(experiment_id, None)
    logger.debug(f"Cleared tree state for experiment {experiment_id}")

def broadcast_tree_update(experiment_id: str, tree_data: Dict[str, Any]) -> None:
    """
    Broadcast tree update (placeholder for now).
    
    This function can be enhanced later to actually broadcast via WebSocket.
    For now, it just logs the update.
    
    Args:
        experiment_id: Experiment ID
        tree_data: Tree data dictionary
    """
    try:
        # For now, just log the broadcast
        # In the future, this could send WebSocket messages
        logger.debug(f"Broadcasting tree update for experiment {experiment_id}")
        
        # Update the tree state as well
        update_tree_state(experiment_id, tree_data)
        
    except Exception as e:
        logger.error(f"Failed to broadcast tree update for {experiment_id}: {e}")

def get_all_tree_snapshots() -> Dict[str, dict]:
    """
    Get all active tree snapshots.
    
    Returns:
        Dictionary of experiment_id -> tree_data
    """
    return _active_tree_snapshots.copy()