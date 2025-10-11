import os

import socketio

from openhands.server.app import app as base_app
from openhands.server.listen_socket import sio
from openhands.server.middleware import (
    CacheControlMiddleware,
    InMemoryRateLimiter,
    LocalhostCORSMiddleware,
    RateLimitMiddleware,
)
from openhands.server.static import SPAStaticFiles

if os.getenv('SERVE_FRONTEND', 'true').lower() == 'true':
    # Mount static files at root, but this should be done AFTER all API routes
    # are registered in app.py to ensure API routes have priority
    base_app.mount(
        '/', SPAStaticFiles(directory='./OpenHands/frontend/build', html=True), name='dist'
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
