"""API routes for UAgent Research Extension"""

from .research_routes import router
from .websocket_routes import router as ws_router, manager as ws_manager

__all__ = ['router', 'ws_router', 'ws_manager']
