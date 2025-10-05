"""UAgent Research API routes."""

from .research_routes import router
from .websocket_routes import ws_router

__all__ = ['router', 'ws_router']
