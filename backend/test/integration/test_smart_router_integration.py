"""Integration tests for SmartRouter using a LiteLLM-backed client."""

import asyncio
import time

import pytest

from app.core.smart_router import (
    SmartRouter,
    ClassificationRequest,
    InvalidRequestError,
    ClassificationError,
)
from app.core.cache import MemoryCache
from ..llm_test_utils import require_litellm_client


class TestSmartRouterLLMIntegration:
    """Integration tests exercising router behaviour with a real LLM."""

    @pytest.mark.asyncio
    async def test_with_real_llm_client(self):
        llm_client = require_litellm_client()
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        request = ClassificationRequest(
            user_request="Design and execute experiments to test neural network performance"
        )

        result = await router.classify_and_route(request)

        assert result.primary_engine in {"SCIENTIFIC_RESEARCH", "CODE_RESEARCH", "DEEP_RESEARCH"}
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.sub_components, dict)
        assert isinstance(result.workflow_plan, dict)

    @pytest.mark.asyncio
    async def test_caching_with_memory_cache(self):
        llm_client = require_litellm_client()
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        request = ClassificationRequest(user_request="Test caching with real LLM integration")

        result1 = await router.classify_and_route(request)
        result2 = await router.classify_and_route(request)

        assert result1.primary_engine == result2.primary_engine
        assert result1.confidence_score == result2.confidence_score
        assert cache.size() > 0

    @pytest.mark.asyncio
    async def test_error_resilience_integration(self):
        llm_client = require_litellm_client()
        router = SmartRouter(llm_client=llm_client, config={"max_retries": 1})

        edge_case_requests = [
            "Short",
            "A" * 512,
            "Request with special characters: !@#$%^&*()",
            "多语言 mixed content",
        ]

        for request_text in edge_case_requests:
            request = ClassificationRequest(user_request=request_text)
            try:
                result = await router.classify_and_route(request)
                assert result.primary_engine in {
                    "DEEP_RESEARCH",
                    "SCIENTIFIC_RESEARCH",
                    "CODE_RESEARCH",
                }
            except Exception as exc:  # pragma: no cover - defensive handling
                assert isinstance(exc, (InvalidRequestError, ClassificationError))


@pytest.mark.integration
class TestSmartRouterPerformanceIntegration:
    """Performance-oriented integration tests with real LLM."""

    @pytest.mark.asyncio
    async def test_concurrent_classification_performance(self):
        llm_client = require_litellm_client()
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        requests = [
            ClassificationRequest(user_request=f"Analyze AI applications in domain {name}")
            for name in ["healthcare", "finance", "transport", "education", "manufacturing"]
        ]

        start_time = time.time()
        results = await asyncio.gather(*(router.classify_and_route(req) for req in requests))
        total_time = time.time() - start_time

        assert len(results) == len(requests)
        assert total_time < 60.0

    @pytest.mark.asyncio
    async def test_cache_performance_real_llm(self):
        llm_client = require_litellm_client()
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        request = ClassificationRequest(
            user_request="Analyze machine learning algorithms for time series prediction"
        )

        start_time = time.time()
        _ = await router.classify_and_route(request)
        first_elapsed = time.time() - start_time

        start_time = time.time()
        _ = await router.classify_and_route(request)
        second_elapsed = time.time() - start_time

        assert second_elapsed < first_elapsed / 5


@pytest.mark.integration
class TestSmartRouterRealWorldScenarios:
    """High-level scenario tests to ensure router robustness."""

    @pytest.mark.asyncio
    async def test_research_workflow_scenarios(self):
        llm_client = require_litellm_client()
        router = SmartRouter(llm_client=llm_client)

        scenarios = [
            "Research renewable energy adoption and identify market opportunities",
            "Compare Python web frameworks for high-performance APIs",
            "Design experiments to validate a new neural network architecture",
            "Assess quantum computing impact on cryptography",
        ]

        for text in scenarios:
            result = await router.classify_and_route(ClassificationRequest(user_request=text))
            assert result.primary_engine in {
                "DEEP_RESEARCH",
                "SCIENTIFIC_RESEARCH",
                "CODE_RESEARCH",
            }
            assert isinstance(result.workflow_plan, dict)
