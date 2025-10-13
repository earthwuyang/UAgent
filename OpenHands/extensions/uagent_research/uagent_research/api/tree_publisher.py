"""
Tree State Publisher

Shared functions for updating and broadcasting tree state.
This module breaks the circular import between orchestrator and research_routes.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global storage for tree snapshots (shared with research_routes)
# Using session manager for proper experiment isolation - fallback for direct access
_active_tree_snapshots: Dict[str, dict] = {}

# Try to import session manager for experiment-specific storage
try:
    from ...services.research_session_manager import get_global_session_manager
    _session_manager_available = True
    logger.info("✅ Session manager available for tree publisher")
except ImportError:
    _session_manager_available = False
    logger.warning("⚠️ Session manager not available, using global storage")
    get_global_session_manager = None

def update_tree_state(experiment_id: str, tree_data: Dict[str, Any]) -> None:
    """
    Update tree state in global storage.
    
    Args:
        experiment_id: Experiment ID
        tree_data: Tree data dictionary
    """
    try:
        # Try to use session manager for proper experiment isolation
        if _session_manager_available and get_global_session_manager:
            try:
                session_manager = get_global_session_manager()
                if hasattr(session_manager, 'experiments') and experiment_id in session_manager.experiments:
                    # Store tree data in experiment state
                    experiment_state = session_manager.experiments[experiment_id]
                    if hasattr(experiment_state, 'tree_data'):
                        experiment_state.tree_data = tree_data
                    else:
                        # Add tree_data attribute to experiment state
                        setattr(experiment_state, 'tree_data', tree_data)
                    logger.debug(f"Updated tree state for experiment {experiment_id} using session manager")
                    return  # Successfully stored in session manager
            except Exception as session_error:
                logger.warning(f"Session manager storage failed, falling back to global storage: {session_error}")
        
        # Fallback to global storage
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
    # Try to use session manager for proper experiment isolation
    if _session_manager_available and get_global_session_manager:
        try:
            session_manager = get_global_session_manager()
            if hasattr(session_manager, 'experiments') and experiment_id in session_manager.experiments:
                # Get tree data from experiment state
                experiment_state = session_manager.experiments[experiment_id]
                if hasattr(experiment_state, 'tree_data'):
                    return getattr(experiment_state, 'tree_data', None)
                else:
                    # Check if tree_data is stored in a custom attribute
                    for attr_name in ['tree_data', '_tree_data', 'tree_snapshot']:
                        if hasattr(experiment_state, attr_name):
                            return getattr(experiment_state, attr_name, None)
        except Exception as session_error:
            logger.warning(f"Session manager retrieval failed, falling back to global storage: {session_error}")
    
    # Fallback to global storage
    return _active_tree_snapshots.get(experiment_id)

def clear_tree_state(experiment_id: str) -> None:
    """
    Clear tree state from global storage.
    
    Args:
        experiment_id: Experiment ID
    """
    # Try to use session manager for proper experiment isolation
    if _session_manager_available and get_global_session_manager:
        try:
            session_manager = get_global_session_manager()
            if hasattr(session_manager, 'experiments') and experiment_id in session_manager.experiments:
                # Clear tree data from experiment state
                experiment_state = session_manager.experiments[experiment_id]
                if hasattr(experiment_state, 'tree_data'):
                    delattr(experiment_state, 'tree_data')
                logger.debug(f"Cleared tree state for experiment {experiment_id} from session manager")
        except Exception as session_error:
            logger.warning(f"Session manager clear failed: {session_error}")
    
    # Also clear from global storage (for backward compatibility)
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
    # Try to get tree snapshots from session manager
    combined_snapshots = {}
    
    if _session_manager_available and get_global_session_manager:
        try:
            session_manager = get_global_session_manager()
            if hasattr(session_manager, 'experiments'):
                for experiment_id, experiment_state in session_manager.experiments.items():
                    if hasattr(experiment_state, 'tree_data'):
                        tree_data = getattr(experiment_state, 'tree_data', None)
                        if tree_data is not None:
                            combined_snapshots[experiment_id] = tree_data
        except Exception as session_error:
            logger.warning(f"Session manager retrieval failed: {session_error}")
    
    # Merge with global storage (for backward compatibility)
    combined_snapshots.update(_active_tree_snapshots)
    
    return combined_snapshots