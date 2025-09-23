"""Shared helpers for obtaining real LLM clients in tests."""

from __future__ import annotations

import os
import pytest

from app.core.llm_client import create_llm_client, LLMClient


def require_litellm_client(
    provider: str | None = None,
    *,
    model: str | None = None,
) -> LLMClient:
    """Return a LiteLLM-backed client or skip tests if unavailable."""
    resolved_provider = (
        provider
        or os.getenv("DEFAULT_API_PROVIDER")
        or os.getenv("LLM_PROVIDER")
        or "litellm"
    ).strip().lower()

    api_key = (
        os.getenv("LITELLM_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        pytest.skip("LITELLM_API_KEY not available for real LLM testing")

    resolved_model = (
        model
        or os.getenv("LITELLM_MODEL")
        or os.getenv("LLM_MODEL")
    )

    return create_llm_client(resolved_provider, api_key=api_key, model=resolved_model)
