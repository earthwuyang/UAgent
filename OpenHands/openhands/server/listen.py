import os
import sys
from pathlib import Path

import socketio

# Add UAgent Research Extension to sys.path BEFORE importing app
# This ensures that modules like conversation_service.py can import research middleware
# We add the 'extensions' directory so that 'uagent_research' can be imported as a package
try:
    # Add the extensions directory to sys.path so 'uagent_research' can be imported
    extension_parent_dir = Path(__file__).parent.parent.parent / 'extensions'
    if extension_parent_dir.exists():
        extension_dir_str = str(extension_parent_dir)
        if extension_dir_str not in sys.path:
            use_source = os.getenv('USE_UAGENT_RESEARCH_FROM_SOURCE', 'true').lower() == 'true'
            if use_source:
                sys.path.insert(0, extension_dir_str)  # Use insert(0) for higher priority
                print(f"🔧 [listen.py] Added extension directory to sys.path: {extension_parent_dir}")
except Exception as e:
    print(f"⚠️ [listen.py] Failed to add extension directory to sys.path: {e}")

from openhands.server.app import app as base_app
from openhands.server.listen_socket import sio
from openhands.server.middleware import (
    CacheControlMiddleware,
    InMemoryRateLimiter,
    LocalhostCORSMiddleware,
    RateLimitMiddleware,
)
from openhands.server.static import SPAStaticFiles

# Note: Database initialization is handled by the lifespan context manager in app.py
# No need to initialize here - the FastAPI lifespan ensures proper initialization

if os.getenv('SERVE_FRONTEND', 'true').lower() == 'true':
    # Mount static files at root, but this should be done AFTER all API routes
    # are registered in app.py to ensure API routes have priority
    base_app.mount(
        '/', SPAStaticFiles(directory='./frontend/build', html=True), name='dist'
    )

base_app.add_middleware(LocalhostCORSMiddleware)
base_app.add_middleware(CacheControlMiddleware)
base_app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=InMemoryRateLimiter(requests=10, seconds=1),
)

# Configure SocketIO to allow WebSocket routes to pass through to FastAPI
app = socketio.ASGIApp(
    sio,
    other_asgi_app=base_app,
    socketio_path='/socket.io/'  # Explicitly set SocketIO path to avoid conflicts
)
