"""UAgent Research API routes with comprehensive module loading diagnostics."""

import logging
import sys
import traceback
from datetime import datetime

# Initialize logger early
logger = logging.getLogger(__name__)
logger.info("📦 Loading UAgent Research API module")

# Track module loading status
_module_load_status = {
    "started_at": datetime.now().isoformat(),
    "research_routes": {"loaded": False, "error": None},
    "websocket_routes": {"loaded": False, "error": None},
}

# Load research_routes with diagnostics
try:
    logger.debug("🔄 Importing research_routes module")
    from .research_routes import router
    _module_load_status["research_routes"]["loaded"] = True
    logger.info("✅ research_routes module loaded successfully")
        
except Exception as e:
    error_msg = f"Failed to import research_routes: {e}"
    logger.error(f"❌ {error_msg}", exc_info=True)
    _module_load_status["research_routes"]["error"] = error_msg
    _module_load_status["research_routes"]["traceback"] = traceback.format_exc()
    # Re-raise to prevent broken module from being used
    raise

# Load websocket_routes with diagnostics
try:
    logger.debug("🔄 Importing websocket_routes module")
    from .websocket_routes import router as ws_router, manager, broadcast_tree_update
    _module_load_status["websocket_routes"]["loaded"] = True
    logger.info("✅ websocket_routes module loaded successfully")
    
    # Verify connection manager
    if manager:
        logger.debug(f"✅ WebSocket ConnectionManager available: {type(manager)}")
    else:
        logger.warning("⚠️ WebSocket ConnectionManager is None")
        
except Exception as e:
    error_msg = f"Failed to import websocket_routes: {e}"
    logger.error(f"❌ {error_msg}", exc_info=True)
    _module_load_status["websocket_routes"]["error"] = error_msg
    _module_load_status["websocket_routes"]["traceback"] = traceback.format_exc()
    # Re-raise to prevent broken module from being used
    raise

# Export all routers and utilities
__all__ = [
    'router', 
    'ws_router', 
    'manager', 
    'broadcast_tree_update',
    'get_module_diagnostics',
]

def get_module_diagnostics() -> dict:
    """
    Get diagnostic information about the API module loading status.
    
    Returns:
        dict: Module loading diagnostics including status, errors, and metadata
    """
    diagnostics = {
        "module_name": __name__,
        "python_version": sys.version,
        "load_status": _module_load_status.copy(),
        "all_exports": __all__,
        "router_info": {
            "research_router_available": 'router' in globals(),
            "websocket_router_available": 'ws_router' in globals(),
        },
    }
    
    # Add router details if available
    if 'router' in globals():
        try:
            diagnostics["router_info"]["research_router_prefix"] = router.prefix
            diagnostics["router_info"]["research_router_tags"] = router.tags
            diagnostics["router_info"]["research_routes_count"] = len(router.routes)
        except Exception as e:
            diagnostics["router_info"]["research_router_error"] = str(e)
    
    if 'ws_router' in globals():
        try:
            diagnostics["router_info"]["websocket_router_prefix"] = ws_router.prefix
            diagnostics["router_info"]["websocket_router_tags"] = ws_router.tags
            diagnostics["router_info"]["websocket_routes_count"] = len(ws_router.routes)
        except Exception as e:
            diagnostics["router_info"]["websocket_router_error"] = str(e)
    
    if 'manager' in globals():
        try:
            diagnostics["websocket_manager"] = manager.get_diagnostics()
        except Exception as e:
            diagnostics["websocket_manager_error"] = str(e)
    
    return diagnostics

# Log final module loading status
_module_load_status["completed_at"] = datetime.now().isoformat()
logger.info(
    f"✅ UAgent Research API module loaded successfully. "
    f"Routers available: research={_module_load_status['research_routes']['loaded']}, "
    f"websocket={_module_load_status['websocket_routes']['loaded']}"
)
logger.debug(f"📊 Module diagnostics: {get_module_diagnostics()}")
