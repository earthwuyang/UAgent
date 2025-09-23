#!/usr/bin/env python3
"""Helper script to inspect raw responses from GLM-4.5 via LiteLLM.

Usage:
    python scripts/inspect_glm_response.py \
        --prompt "Return valid JSON with keys foo and bar." --model glm-4.5

Environment variables respected:
    LITELLM_API_KEY or ZHIPU_API_KEY  - API token
    LITELLM_API_BASE or ZHIPU_BASE_URL - API base URL (e.g. https://open.bigmodel.cn/api/paas/v4)
    LITELLM_MODEL                      - default model if --model not provided
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict

try:
    import litellm
except ImportError:  # pragma: no cover
    print("litellm is not installed. Install it with `pip install litellm`.", file=sys.stderr)
    sys.exit(1)


def _as_json(data: Any) -> str:
    """Return a prettified JSON string when possible."""

    if isinstance(data, (str, bytes)):
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        return json.dumps(parsed, indent=2, ensure_ascii=False)

    if hasattr(data, "model_dump"):
        try:
            return json.dumps(data.model_dump(), indent=2, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            pass

    if isinstance(data, dict):
        return json.dumps(data, indent=2, ensure_ascii=False)

    if hasattr(data, "dict"):
        try:
            return json.dumps(data.dict(), indent=2, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            pass

    return str(data)


async def _call_glm(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("LITELLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
    api_base = args.api_base or os.getenv("LITELLM_API_BASE") or os.getenv("ZHIPU_BASE_URL")
    model = args.model or os.getenv("LITELLM_MODEL") or "glm-4.5"

    if not api_key:
        print("Missing API key. Set LITELLM_API_KEY / ZHIPU_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    messages = [{"role": "user", "content": args.prompt}]

    request_payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if api_key:
        request_payload["api_key"] = api_key
    if api_base:
        request_payload["api_base"] = api_base

    print("Dispatching request with payload:\n" + json.dumps({k: v for k, v in request_payload.items() if k != "api_key"}, indent=2))

    try:
        response = await litellm.acompletion(**request_payload)
    except Exception as exc:  # pragma: no cover - networking errors
        print(f"LiteLLM request failed: {exc}", file=sys.stderr)
        return 1

    print("\nRaw response:\n" + _as_json(response))

    # Extract first message content for quick inspection
    content = None
    if isinstance(response, dict):
        choices = response.get("choices")
    else:
        choices = getattr(response, "choices", None)

    if choices:
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

    if content is not None:
        print("\nFirst message content:\n" + _as_json(content))

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Respond with valid JSON summarizing your capabilities.")
    parser.add_argument("--model", help="Override model name (default uses LITELLM_MODEL or glm-4.5).")
    parser.add_argument("--api-key", help="Override API key (defaults to env).")
    parser.add_argument("--api-base", help="Override API base URL (defaults to env).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=51200)

    args = parser.parse_args()
    exit_code = asyncio.run(_call_glm(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

