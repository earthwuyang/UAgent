"""Centralized container image references for UAgent integrations.

This avoids duplicating image names across modules. Override via env vars.
"""

from __future__ import annotations

import os

# Default runtime image used for executing OpenHands workloads.
# Can be overridden with UAGENT_OPENHANDS_IMAGE environment variable.
DEFAULT_OPENHANDS_IMAGE: str = os.getenv("UAGENT_OPENHANDS_IMAGE", "uagent:v0.1")


def get_openhands_image() -> str:
    """Return the container image to use for OpenHands runtimes.

    Priority: env var UAGENT_OPENHANDS_IMAGE, otherwise DEFAULT_OPENHANDS_IMAGE.
    """
    return os.getenv("UAGENT_OPENHANDS_IMAGE", DEFAULT_OPENHANDS_IMAGE)

