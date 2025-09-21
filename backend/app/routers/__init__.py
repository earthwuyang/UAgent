"""API routers exposed by the backend."""

from . import research, science, smart_router, websocket  # noqa: F401

__all__ = [
    "research",
    "science",
    "smart_router",
    "websocket",
]
