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


def sanitize_json_strings(payload: str) -> str:
    """Replace literal newlines within quoted JSON strings with spaces."""

    if not payload:
        return payload

    payload = (
        payload.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    result: list[str] = []
    in_string = False
    escape = False

    for ch in payload:
        if in_string:
            if escape:
                result.append(ch)
                escape = False
                continue
            if ch == "\\":
                result.append(ch)
                escape = True
                continue
            if ch in ("\n", "\r"):
                result.append(" ")
                continue
            if ch == '"':
                in_string = False
            result.append(ch)
        else:
            result.append(ch)
            if ch == '"':
                in_string = True
            else:
                escape = False

    return "".join(result)


def _balance_json_delimiters(candidate: str) -> str:
    """Append missing closing braces/brackets to balance delimiters."""

    if not candidate:
        return candidate

    open_brace = candidate.count("{")
    close_brace = candidate.count("}")
    open_bracket = candidate.count("[")
    close_bracket = candidate.count("]")

    balanced = candidate
    if close_brace < open_brace:
        balanced += "}" * (open_brace - close_brace)
    if close_bracket < open_bracket:
        balanced += "]" * (open_bracket - close_bracket)
    return balanced


def safe_json_loads(raw: str) -> Any:
    """Parse JSON from arbitrary LLM output, raising JsonParseError on failure."""

    if raw is None:
        raise JsonParseError("LLM response was None")

    text = (raw or "").strip()
    if not text:
        raise JsonParseError("LLM response was empty")

    try:
        return json.loads(sanitize_json_strings(text))
    except json.JSONDecodeError:
        candidate = _extract_json_candidate(text)
        if candidate is None:
            stripped = text.strip()
            if stripped.startswith("```"):
                lines = stripped.splitlines()
                if len(lines) > 1:
                    candidate = "\n".join(lines[1:])
                    if candidate.endswith("```"):
                        candidate = candidate[: candidate.rfind("```")]
                    candidate = candidate.strip()
                    if candidate:
                        candidate = sanitize_json_strings(candidate)
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
            raise JsonParseError("No JSON candidate found in response")

        candidate = sanitize_json_strings(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            # Attempt minimal repairs for trailing commas or single quotes
            repaired = candidate.replace("'", '"').replace(",}", "}").replace(",]", "]")
            try:
                return json.loads(sanitize_json_strings(repaired))
            except json.JSONDecodeError:
                brace_fixed = _balance_json_delimiters(repaired)
                try:
                    return json.loads(sanitize_json_strings(brace_fixed))
                except json.JSONDecodeError as exc2:
                    raise JsonParseError(f"Unable to parse JSON candidate: {exc2}") from exc2
    except Exception as exc:  # pragma: no cover - unexpected exceptions
        raise JsonParseError(f"Unexpected error parsing JSON: {exc}") from exc
