"""API routers exposed by the backend."""

from . import openhands, research, science, smart_router, websocket  # noqa: F401

__all__ = [
    "openhands",
    "research",
    "science",
    "smart_router",
    "websocket",
]
