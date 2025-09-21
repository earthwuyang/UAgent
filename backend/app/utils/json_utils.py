"""Utilities for safely parsing JSON from LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_BLOCK_RE = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)
_GENERIC_BLOCK_RE = re.compile(r"```\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_LIKE_RE = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])")


class JsonParseError(ValueError):
    """Raised when safe_json_loads cannot recover a JSON payload."""


def _extract_json_candidate(text: str) -> str | None:
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)

    match = _GENERIC_BLOCK_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate

    match = _JSON_LIKE_RE.search(text)
    if match:
        return match.group(1)

    return None


def safe_json_loads(raw: str) -> Any:
    """Parse JSON from arbitrary LLM output, raising JsonParseError on failure."""

    if raw is None:
        raise JsonParseError("LLM response was None")

    text = raw.strip()
    if not text:
        raise JsonParseError("LLM response was empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = _extract_json_candidate(text)
        if candidate is None:
            raise JsonParseError("No JSON candidate found in response")

        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            # Attempt minimal repairs for trailing commas or single quotes
            repaired = candidate.replace("'", '"').replace(",}", "}").replace(",]", "]")
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc2:
                raise JsonParseError(f"Unable to parse JSON candidate: {exc2}") from exc2
    except Exception as exc:  # pragma: no cover - unexpected exceptions
        raise JsonParseError(f"Unexpected error parsing JSON: {exc}") from exc
