import contextlib
import warnings
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi.routing import Mount

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import JSONResponse

import openhands.agenthub  # noqa F401 (we import this to get the agents registered)
from openhands import __version__
from openhands.integrations.service_types import AuthenticationError
from openhands.server.routes.conversation import app as conversation_api_router
from openhands.server.routes.feedback import app as feedback_api_router
from openhands.server.routes.files import app as files_api_router
from openhands.server.routes.git import app as git_api_router
from openhands.server.routes.health import add_health_endpoints
from openhands.server.routes.manage_conversations import (
    app as manage_conversation_api_router,
)
from openhands.server.routes.mcp import mcp_server
from openhands.server.routes.public import app as public_api_router
from openhands.server.routes.secrets import app as secrets_router
from openhands.server.routes.security import app as security_api_router
from openhands.server.routes.settings import app as settings_router
from openhands.server.routes.trajectory import app as trajectory_router
from openhands.server.shared import conversation_manager, server_config
from openhands.server.types import AppMode

# UAgent Research Extension - Import directly from source
try:
    import sys
    from pathlib import Path

    # Add extension to Python path
    extension_dir = Path(__file__).parent.parent.parent / 'extensions' / 'uagent_research'
    if extension_dir.exists():
        sys.path.insert(0, str(extension_dir))

        # Import from source (not installed package)
        from uagent_research.api import router as research_router, ws_router as research_ws_router
        from uagent_research.models.base import init_database, close_database

        RESEARCH_EXTENSION_AVAILABLE = True
        print("✅ UAgent Research Extension loaded from source")
    else:
        RESEARCH_EXTENSION_AVAILABLE = False
        research_router = None
        research_ws_router = None
        print(f"⚠️ UAgent Research Extension not found at {extension_dir}")
except ImportError as e:
    print(f"❌ Failed to load UAgent Research Extension: {e}")
    import traceback
    traceback.print_exc()
    RESEARCH_EXTENSION_AVAILABLE = False
    research_router = None
    research_ws_router = None

mcp_app = mcp_server.http_app(path='/mcp')


def combine_lifespans(*lifespans):
    # Create a combined lifespan to manage multiple session managers
    @contextlib.asynccontextmanager
    async def combined_lifespan(app):
        async with contextlib.AsyncExitStack() as stack:
            for lifespan in lifespans:
                await stack.enter_async_context(lifespan(app))
            yield

    return combined_lifespan


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize UAgent Research Extension database if available
    if RESEARCH_EXTENSION_AVAILABLE:
        import os
        import logging
        logger = logging.getLogger(__name__)

        # Get database URL from environment or use default SQLite
        research_db_url = os.getenv(
            'RESEARCH_DATABASE_URL',
            'sqlite+aiosqlite:///./openhands_research.db'
        )

        try:
            await init_database(research_db_url, echo=False)
            logger.info(f"✅ Research database initialized: {research_db_url}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize research database: {e}")

    async with conversation_manager:
        yield

    # Cleanup research database connections
    if RESEARCH_EXTENSION_AVAILABLE:
        try:
            await close_database()
        except Exception:
            pass


app = FastAPI(
    title='OpenHands',
    description='OpenHands: Code Less, Make More',
    version=__version__,
    lifespan=combine_lifespans(_lifespan, mcp_app.lifespan),
    routes=[Mount(path='/mcp', app=mcp_app)],
)


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=401,
        content=str(exc),
    )


app.include_router(public_api_router)
app.include_router(files_api_router)
app.include_router(security_api_router)
app.include_router(feedback_api_router)
app.include_router(conversation_api_router)
app.include_router(manage_conversation_api_router)
app.include_router(settings_router)
app.include_router(secrets_router)
if server_config.app_mode == AppMode.OSS:
    app.include_router(git_api_router)
app.include_router(trajectory_router)

# Include UAgent Research Extension routes if available
if RESEARCH_EXTENSION_AVAILABLE and research_router is not None:
    app.include_router(research_router)
    if research_ws_router is not None:
        app.include_router(research_ws_router)
    print("✅ UAgent Research Extension routes registered")

add_health_endpoints(app)
