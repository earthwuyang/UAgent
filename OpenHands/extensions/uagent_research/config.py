"""UAgent Research Configuration with validation."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .exceptions import ConfigurationError


_CONFIG_ERRORS: List[str] = []
_CONFIG_VALUES: Dict[str, Any] = {}


def _record(name: str, value: Any) -> Any:
    _CONFIG_VALUES[name] = value
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return _record(name, default)

    normalized = raw.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return _record(name, True)
    if normalized in {"false", "0", "no", "off"}:
        return _record(name, False)

    _CONFIG_ERRORS.append(f"{name} must be boolean (got '{raw}')")
    return _record(name, default)


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    value = default
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            _CONFIG_ERRORS.append(f"{name} must be an integer (got '{raw}')")
            return _record(name, default)

    if min_value is not None and value < min_value:
        _CONFIG_ERRORS.append(f"{name} must be >= {min_value} (got {value})")
    if max_value is not None and value > max_value:
        _CONFIG_ERRORS.append(f"{name} must be <= {max_value} (got {value})")

    return _record(name, value)


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = os.getenv(name)
    value = default
    if raw is not None:
        try:
            value = float(raw)
        except ValueError:
            _CONFIG_ERRORS.append(f"{name} must be a float (got '{raw}')")
            return _record(name, default)

    if min_value is not None and value < min_value:
        _CONFIG_ERRORS.append(f"{name} must be >= {min_value} (got {value})")
    if max_value is not None and value > max_value:
        _CONFIG_ERRORS.append(f"{name} must be <= {max_value} (got {value})")

    return _record(name, value)


# Research Middleware Configuration
ENABLE_AUTO_RESEARCH_TRIGGER = _env_bool('ENABLE_AUTO_RESEARCH_TRIGGER', True)
RESEARCH_CONFIDENCE_THRESHOLD = _env_float(
    'RESEARCH_CONFIDENCE_THRESHOLD',
    0.7,
    min_value=0.0,
    max_value=1.0,
)
RESEARCH_MAX_ITERATIONS = _env_int('RESEARCH_MAX_ITERATIONS', 50, min_value=1)
RESEARCH_MAX_COST = _env_float('RESEARCH_MAX_COST', 10.0, min_value=0.0)
RESEARCH_MAX_PARALLEL = _env_int('RESEARCH_MAX_PARALLEL', 3, min_value=1, max_value=32)

# Agent coordination configuration
ENABLE_AGENT_COORDINATION = _env_bool('ENABLE_AGENT_COORDINATION', True)
RESEARCH_POLL_INTERVAL = _env_int('RESEARCH_POLL_INTERVAL', 10, min_value=1)
PROGRESS_CACHE_TTL = _env_float('PROGRESS_CACHE_TTL', 2.0, min_value=0.1)

# Node Generation Configuration
ENABLE_INTELLIGENT_EXPANSION = _env_bool('RESEARCH_ENABLE_INTELLIGENT_EXPANSION', True)
MAX_RESEARCH_IDEAS = _env_int('RESEARCH_MAX_IDEAS', 3, min_value=1, max_value=10)
MAX_HYPOTHESES_PER_IDEA = _env_int('RESEARCH_MAX_HYPOTHESES', 2, min_value=1, max_value=10)
MAX_EXPERIMENTS_PER_HYPOTHESIS = _env_int('RESEARCH_MAX_EXPERIMENTS', 1, min_value=1, max_value=10)
IDEA_GENERATION_RETRY_COUNT = _env_int('RESEARCH_IDEA_RETRY_COUNT', 2, min_value=0, max_value=5)

# To disable automatic research triggering:
# export ENABLE_AUTO_RESEARCH_TRIGGER=false

# To disable intelligent expansion:
# export RESEARCH_ENABLE_INTELLIGENT_EXPANSION=false

# Research auto-trigger is now ENABLED by default

# Logging configuration
# Enable verbose debug logging for research flow
RESEARCH_DEBUG_LOGGING = _env_bool('RESEARCH_DEBUG_LOGGING', False)

# Log when tree state is updated and published
# Set to 'true' to see when tree snapshots are created and sent to API/WebSocket
# Useful for debugging "disconnected" issues in frontend
RESEARCH_LOG_TREE_UPDATES = _env_bool('RESEARCH_LOG_TREE_UPDATES', True)

# Log WebSocket connection and broadcast events
# Set to 'true' to see WebSocket client connections and message broadcasts
# Useful for debugging real-time update issues
RESEARCH_LOG_WEBSOCKET = _env_bool('RESEARCH_LOG_WEBSOCKET', True)


CONFIG_SUMMARY: Dict[str, Any] = {
    'AUTO_TRIGGER': ENABLE_AUTO_RESEARCH_TRIGGER,
    'CONFIDENCE_THRESHOLD': RESEARCH_CONFIDENCE_THRESHOLD,
    'MAX_ITERATIONS': RESEARCH_MAX_ITERATIONS,
    'MAX_COST': RESEARCH_MAX_COST,
    'MAX_PARALLEL': RESEARCH_MAX_PARALLEL,
    'AGENT_COORDINATION': ENABLE_AGENT_COORDINATION,
    'POLL_INTERVAL': RESEARCH_POLL_INTERVAL,
    'PROGRESS_CACHE_TTL': PROGRESS_CACHE_TTL,
    'INTELLIGENT_EXPANSION': ENABLE_INTELLIGENT_EXPANSION,
    'MAX_IDEAS': MAX_RESEARCH_IDEAS,
    'MAX_HYPOTHESES': MAX_HYPOTHESES_PER_IDEA,
    'MAX_EXPERIMENTS': MAX_EXPERIMENTS_PER_HYPOTHESIS,
    'IDEA_RETRY_COUNT': IDEA_GENERATION_RETRY_COUNT,
    'DEBUG_LOGGING': RESEARCH_DEBUG_LOGGING,
    'LOG_TREE_UPDATES': RESEARCH_LOG_TREE_UPDATES,
    'LOG_WEBSOCKET': RESEARCH_LOG_WEBSOCKET,
}

if _CONFIG_ERRORS:
    error_block = "\n - ".join(_CONFIG_ERRORS)
    raise ConfigurationError(f"Invalid UAgent research configuration:\n - {error_block}")

# Log configuration on import (only if debug logging enabled)
if RESEARCH_DEBUG_LOGGING:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("🔧 Research Extension Configuration:")
    logger.info(f"  Auto-trigger: {ENABLE_AUTO_RESEARCH_TRIGGER}")
    logger.info(f"  Confidence threshold: {RESEARCH_CONFIDENCE_THRESHOLD}")
    logger.info(f"  Max iterations: {RESEARCH_MAX_ITERATIONS}")
    logger.info(f"  Max cost: ${RESEARCH_MAX_COST}")
    logger.info(f"  Max parallel: {RESEARCH_MAX_PARALLEL}")
    logger.info(f"  Debug logging: {RESEARCH_DEBUG_LOGGING}")
    logger.info(f"  Log tree updates: {RESEARCH_LOG_TREE_UPDATES}")
    logger.info(f"  Log WebSocket: {RESEARCH_LOG_WEBSOCKET}")
