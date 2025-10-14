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
except ImportError as e:
    # Try alternative import path (relative from extension root)
    try:
        from extensions.uagent_research.services.research_session_manager import get_global_session_manager
        _session_manager_available = True
        logger.info("✅ Session manager available for tree publisher (fallback import)")
    except ImportError:
        _session_manager_available = False
        logger.warning(f"⚠️ Session manager not available, using global storage. Import error: {e}")
        get_global_session_manager = None

def update_tree_state(experiment_id: str, tree_data: Dict[str, Any]) -> None:
    """
    Update tree state in global storage.

    Args:
        experiment_id: Experiment ID
        tree_data: Tree data dictionary
    """
    logger.info(f"🔄 [CACHE UPDATE] Called for experiment_id: {experiment_id}")
    logger.info(f"🔄 [CACHE UPDATE] _active_tree_snapshots dict ID: {id(_active_tree_snapshots)}")
    logger.info(f"🔄 [CACHE UPDATE] Current keys in cache: {list(_active_tree_snapshots.keys())}")
    logger.info(f"🔄 [CACHE UPDATE] Tree data has {len(tree_data.get('data', {}).get('nodes', []))} nodes")

    try:
        # Try to use session manager for proper experiment isolation
        if _session_manager_available and get_global_session_manager:
            try:
                logger.info(f"🔄 [CACHE UPDATE] Attempting session manager storage...")
                session_manager = get_global_session_manager()
                logger.info(f"🔄 [CACHE UPDATE] Session manager ID: {id(session_manager)}")
                logger.info(f"🔄 [CACHE UPDATE] Session manager has {len(session_manager.experiments)} experiments")
                logger.info(f"🔄 [CACHE UPDATE] Session manager experiment IDs: {list(session_manager.experiments.keys())}")

                if hasattr(session_manager, 'experiments') and experiment_id in session_manager.experiments:
                    # Store tree data in experiment state
                    experiment_state = session_manager.experiments[experiment_id]
                    logger.info(f"🔄 [CACHE UPDATE] Found experiment state, type: {type(experiment_state)}")

                    if hasattr(experiment_state, 'tree_data'):
                        experiment_state.tree_data = tree_data
                        logger.info(f"✅ [CACHE UPDATE] Updated existing tree_data attribute")
                    else:
                        # Add tree_data attribute to experiment state
                        setattr(experiment_state, 'tree_data', tree_data)
                        logger.info(f"✅ [CACHE UPDATE] Created new tree_data attribute")

                    logger.info(f"✅ [CACHE UPDATE] Successfully stored in session manager for {experiment_id}")
                    # Don't return - also store in global fallback for after unregister
                else:
                    logger.warning(f"⚠️ [CACHE UPDATE] Experiment {experiment_id} NOT in session manager experiments dict")
                    logger.warning(f"⚠️ [CACHE UPDATE] Will fall back to global storage")
            except Exception as session_error:
                logger.warning(f"⚠️ [CACHE UPDATE] Session manager storage failed: {session_error}", exc_info=True)
                logger.warning(f"⚠️ [CACHE UPDATE] Falling back to global storage")
        else:
            logger.info(f"ℹ️ [CACHE UPDATE] Session manager not available, using global storage directly")

        # Fallback to global storage
        _active_tree_snapshots[experiment_id] = tree_data
        logger.info(f"✅ [CACHE UPDATE] Stored in global _active_tree_snapshots[{experiment_id}]")

        # Also index by session_id so UI polling with conversation_id works
        if experiment_id.startswith('exp_'):
            parts = experiment_id.split('_')
            if len(parts) >= 3:
                session_id = parts[1]
                _active_tree_snapshots[session_id] = tree_data
                logger.info(f"✅ [CACHE UPDATE] Also stored under session_id: {session_id}")

        logger.info(f"✅ [CACHE UPDATE] Cache now has {len(_active_tree_snapshots)} keys: {list(_active_tree_snapshots.keys())}")
    except Exception as e:
        logger.error(f"❌ [CACHE UPDATE] Failed to update tree state for {experiment_id}: {e}", exc_info=True)

def get_tree_state(experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Get tree state from global storage.

    Args:
        experiment_id: Experiment ID

    Returns:
        Tree data dictionary or None if not found
    """
    logger.info(f"🔍 [CACHE GET] Called for experiment_id: {experiment_id}")
    logger.info(f"🔍 [CACHE GET] _active_tree_snapshots dict ID: {id(_active_tree_snapshots)}")
    logger.info(f"🔍 [CACHE GET] Current keys in cache: {list(_active_tree_snapshots.keys())}")
    logger.info(f"🔍 [CACHE GET] Cache has {len(_active_tree_snapshots)} entries")

    # Try to use session manager for proper experiment isolation
    if _session_manager_available and get_global_session_manager:
        try:
            logger.info(f"🔍 [CACHE GET] Attempting session manager retrieval...")
            session_manager = get_global_session_manager()
            logger.info(f"🔍 [CACHE GET] Session manager ID: {id(session_manager)}")
            logger.info(f"🔍 [CACHE GET] Session manager has {len(session_manager.experiments)} experiments")
            logger.info(f"🔍 [CACHE GET] Session manager experiment IDs: {list(session_manager.experiments.keys())}")

            # Try exact match first
            experiment_state = None
            if hasattr(session_manager, 'experiments') and experiment_id in session_manager.experiments:
                experiment_state = session_manager.experiments[experiment_id]
                logger.info(f"🔍 [CACHE GET] Found exact match: {experiment_id}")
            elif hasattr(session_manager, 'experiments') and experiment_id.startswith('exp_'):
                # Try prefix match: find any experiment for this session
                parts = experiment_id.split('_')
                if len(parts) >= 3:
                    session_id = parts[1]
                    logger.info(f"🔍 [CACHE GET] Exact match not found, trying prefix match for session: {session_id}")
                    for key in session_manager.experiments.keys():
                        if key.startswith(f"exp_{session_id}_"):
                            experiment_state = session_manager.experiments[key]
                            logger.info(f"✅ [CACHE GET] Found using prefix match: {key}")
                            break

            if experiment_state:
                logger.info(f"🔍 [CACHE GET] Found experiment state, type: {type(experiment_state)}")
                logger.info(f"🔍 [CACHE GET] Experiment state attributes: {dir(experiment_state)}")

                if hasattr(experiment_state, 'tree_data'):
                    tree_data = getattr(experiment_state, 'tree_data', None)
                    if tree_data:
                        nodes_count = len(tree_data.get('data', {}).get('nodes', []))
                        logger.info(f"✅ [CACHE GET] Found tree_data with {nodes_count} nodes")
                        return tree_data
                    else:
                        logger.warning(f"⚠️ [CACHE GET] tree_data attribute exists but is None")
                else:
                    logger.info(f"🔍 [CACHE GET] No tree_data attribute, checking alternatives...")
                    # Check if tree_data is stored in a custom attribute
                    for attr_name in ['tree_data', '_tree_data', 'tree_snapshot']:
                        if hasattr(experiment_state, attr_name):
                            tree_data = getattr(experiment_state, attr_name, None)
                            if tree_data:
                                logger.info(f"✅ [CACHE GET] Found data in {attr_name} attribute")
                                return tree_data
                    logger.warning(f"⚠️ [CACHE GET] No tree data in any attribute")
            else:
                logger.warning(f"⚠️ [CACHE GET] Experiment {experiment_id} NOT in session manager experiments dict")
                logger.warning(f"⚠️ [CACHE GET] Will fall back to global storage")
        except Exception as session_error:
            logger.warning(f"⚠️ [CACHE GET] Session manager retrieval failed: {session_error}", exc_info=True)
            logger.warning(f"⚠️ [CACHE GET] Falling back to global storage")
    else:
        logger.info(f"ℹ️ [CACHE GET] Session manager not available, using global storage directly")

    # Fallback to global storage
    result = _active_tree_snapshots.get(experiment_id)

    # If exact match not found, try to find by session_id prefix
    if not result and experiment_id.startswith('exp_'):
        parts = experiment_id.split('_')
        if len(parts) >= 3:
            session_id = parts[1]
            logger.info(f"🔍 [CACHE GET] Exact match not found, trying session_id: {session_id}")

            # Try session_id directly
            result = _active_tree_snapshots.get(session_id)
            if result:
                logger.info(f"✅ [CACHE GET] Found using session_id: {session_id}")
            else:
                # Try prefix match: find any experiment for this session
                logger.info(f"🔍 [CACHE GET] Trying prefix match for session: {session_id}")
                for key in _active_tree_snapshots.keys():
                    if key.startswith(f"exp_{session_id}_"):
                        result = _active_tree_snapshots[key]
                        logger.info(f"✅ [CACHE GET] Found using prefix match: {key}")
                        break

    if result:
        nodes_count = len(result.get('data', {}).get('nodes', []))
        logger.info(f"✅ [CACHE GET] Found in global storage with {nodes_count} nodes")
    else:
        logger.warning(f"❌ [CACHE GET] NOT FOUND in global storage for {experiment_id}")
        logger.warning(f"❌ [CACHE GET] Available keys: {list(_active_tree_snapshots.keys())}")

    return result

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

async def broadcast_tree_update(experiment_id: str, tree_data: Dict[str, Any]) -> None:
    """
    Broadcast tree update (placeholder for now).

    This function can be enhanced later to actually broadcast via WebSocket.
    For now, it just logs the update.

    Args:
        experiment_id: Experiment ID
        tree_data: Tree data dictionary
    """
    try:
        logger.info(f"📡 [BROADCAST] Called for experiment {experiment_id}")
        nodes_count = len(tree_data.get('data', {}).get('nodes', []))
        logger.info(f"📡 [BROADCAST] Broadcasting tree with {nodes_count} nodes")

        # Update the tree state as well
        update_tree_state(experiment_id, tree_data)
        logger.info(f"✅ [BROADCAST] Tree state updated successfully")

    except Exception as e:
        logger.error(f"❌ [BROADCAST] Failed to broadcast tree update for {experiment_id}: {e}", exc_info=True)

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