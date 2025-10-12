"""Compatibility shim that re-exports the canonical research routes.

Historically the extension exposed routers from ``extensions.uagent_research.api``.
The authoritative implementation now lives under
``extensions.uagent_research.uagent_research.api``.  This shim keeps legacy import
paths working while ensuring both orchestrator updates and the FastAPI app share
the same module state (tree snapshots, session managers, etc.).
"""

from ..uagent_research.api.research_routes import *  # noqa: F401,F403

