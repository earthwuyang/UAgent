"""Configuration validation and helpers (fail-fast for critical env)."""

from __future__ import annotations

import os


class ConfigError(SystemExit):
    pass


def _get_int(name: str) -> int | None:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        raise ConfigError(f"[config] {name} must be an integer, got: {val!r}")


def validate_env() -> None:
    """Fail fast on missing critical configuration.

    Rules:
    - Require a backend port from BACKEND_PORT or UAGENT_BACKEND_PORT.
    - Validate port range.
    """
    port = _get_int("BACKEND_PORT") or _get_int("UAGENT_BACKEND_PORT")
    if port is None:
        raise ConfigError("[config] BACKEND_PORT (or UAGENT_BACKEND_PORT) is required")
    if port < 1024 or port > 65535:
        raise ConfigError(f"[config] BACKEND_PORT must be 1024..65535, got {port}")

