"""API routes for UAgent Research Extension"""

import logging

# Configure debug logging for research API
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Enable debug logging for the research API
logging.getLogger('extensions.uagent_research.api').setLevel(logging.DEBUG)

from .research_routes import router
from .websocket_routes import router as ws_router, manager as ws_manager

__all__ = ['router', 'ws_router', 'ws_manager']
