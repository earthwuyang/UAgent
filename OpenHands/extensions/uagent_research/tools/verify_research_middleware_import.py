"""Utility script to verify research middleware import chain."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS: Dict[str, Any] = {}


def check_env_flag() -> None:
    value = os.getenv("ENABLE_AUTO_RESEARCH_TRIGGER", "<not-set>")
    RESULTS["env_flag"] = value
    logger.info("ENABLE_AUTO_RESEARCH_TRIGGER=%s", value)


def attempt_import(label: str, module: str, attr: str | None = None) -> None:
    try:
        imported = importlib.import_module(module)
        RESULTS[label] = "ok"
        logger.info("Imported module '%s'", module)
        if attr:
            getattr(imported, attr)
            RESULTS[f"{label}_attr"] = "ok"
            logger.info("Verified attribute '%s.%s'", module, attr)
    except Exception as exc:  # noqa: BLE001
        RESULTS[label] = {
            "error": type(exc).__name__,
            "message": str(exc),
        }
        logger.error("Failed to import %s (%s): %s", module, type(exc).__name__, exc)


def main() -> int:
    check_env_flag()
    attempt_import("config", "extensions.uagent_research.config", "ENABLE_AUTO_RESEARCH_TRIGGER")
    attempt_import("middleware", "extensions.uagent_research.middleware.research_middleware", "research_middleware")
    attempt_import("session", "openhands.server.session.session", "RESEARCH_MIDDLEWARE_AVAILABLE")

    summary_path = Path.cwd() / "research_import_verification.json"
    summary_path.write_text(json.dumps(RESULTS, indent=2))
    logger.info("Wrote verification summary to %s", summary_path)

    failures = [key for key, value in RESULTS.items() if isinstance(value, dict)]
    if failures:
        logger.error("Import chain verification failed: %s", ", ".join(failures))
        return 1
    logger.info("Import chain verified successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
