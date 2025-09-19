"""Integration tests for Smart Router with real LLM clients"""

import os
import pytest
import asyncio
import time

from app.core.smart_router import SmartRouter, ClassificationRequest
from app.core.llm_client import create_llm_client
from app.core.cache import MemoryCache


class TestSmartRouterLLMIntegration:
    """Integration tests with real LLM clients"""

    @pytest.mark.asyncio
    async def test_with_real_llm_client(self):
        """Test smart router with real LLM client"""
        # Arrange
        # Use DashScope if API key available, otherwise skip
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        # Test scientific research classification
        request = ClassificationRequest(
            user_request="Design and execute experiments to test neural network performance on image classification tasks"
        )

        # Act
        result = await router.classify_and_route(request)

        # Assert
        assert result.primary_engine in ["SCIENTIFIC_RESEARCH", "CODE_RESEARCH", "DEEP_RESEARCH"]
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.sub_components, dict)
        assert isinstance(result.workflow_plan, dict)

    @pytest.mark.asyncio
    async def test_caching_with_memory_cache(self):
        """Test caching functionality with memory cache"""
        # Arrange
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        request = ClassificationRequest(
            user_request="Test caching with real LLM integration"
        )

        # Act - First request
        result1 = await router.classify_and_route(request)

        # Act - Second request (should hit cache)
        result2 = await router.classify_and_route(request)

        # Assert
        assert result1.primary_engine == result2.primary_engine
        assert result1.confidence_score == result2.confidence_score
        assert result1.reasoning == result2.reasoning
        assert cache.size() > 0

    @pytest.mark.asyncio
    async def test_workflow_generation_integration(self):
        """Test end-to-end workflow generation"""
        # Arrange
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        router = SmartRouter(llm_client=llm_client)

        # Test different request types
        requests = [
            "Research AI trends and market developments",
            "Find Python machine learning libraries and compare their features",
            "Test hypothesis about deep learning model performance with different optimizers"
        ]

        for request_text in requests:
            request = ClassificationRequest(user_request=request_text)

            # Act
            result = await router.classify_and_route(request)

            # Assert
            assert result.primary_engine in ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH", "CODE_RESEARCH"]
            assert isinstance(result.workflow_plan, dict)
            assert "primary_engine" in result.workflow_plan
            assert "sub_workflows" in result.workflow_plan

    @pytest.mark.asyncio
    async def test_error_resilience_integration(self):
        """Test error handling in real integration scenarios"""
        # Arrange
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        router = SmartRouter(llm_client=llm_client, config={"max_retries": 2})

        # Test with various edge cases
        edge_case_requests = [
            "Very short request",
            "A" * 1000,  # Very long request
            "Request with special characters: !@#$%^&*()",
            "Mixed language request 中文 English française"
        ]

        for request_text in edge_case_requests:
            request = ClassificationRequest(user_request=request_text)

            try:
                # Act
                result = await router.classify_and_route(request)

                # Assert basic structure if successful
                assert result.primary_engine in ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH", "CODE_RESEARCH"]
                assert 0.0 <= result.confidence_score <= 1.0

            except Exception as e:
                # Allow some failures for edge cases, but they should be handled gracefully
                assert isinstance(e, (InvalidRequestError, ClassificationError))


@pytest.mark.integration
class TestSmartRouterPerformanceIntegration:
    """Performance integration tests with real LLM"""

    @pytest.mark.asyncio
    async def test_concurrent_classification_performance(self):
        """Test performance with concurrent real LLM requests"""
        # Arrange
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        # Use different requests to test true concurrency
        requests = [
            ClassificationRequest(user_request=f"Analyze AI applications in domain {i}")
            for i in ["healthcare", "finance", "transportation", "education", "manufacturing"]
        ]

        # Act
        start_time = time.time()
        results = await asyncio.gather(*[
            router.classify_and_route(req) for req in requests
        ])
        end_time = time.time()

        # Assert
        assert len(results) == 5
        for result in results:
            assert result.primary_engine in ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH", "CODE_RESEARCH"]
            assert 0.0 <= result.confidence_score <= 1.0

        # Should complete in reasonable time (allowing for API latency)
        total_time = end_time - start_time
        assert total_time < 60.0  # Allow up to 60 seconds for real API calls

    @pytest.mark.asyncio
    async def test_cache_performance_real_llm(self):
        """Test cache performance with real LLM calls"""
        # Arrange
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        cache = MemoryCache()
        router = SmartRouter(llm_client=llm_client, cache=cache)

        request = ClassificationRequest(user_request="Analyze machine learning algorithms for time series prediction")

        # Act - First request (miss)
        start_time = time.time()
        result1 = await router.classify_and_route(request)
        first_request_time = time.time() - start_time

        # Act - Second request (hit)
        start_time = time.time()
        result2 = await router.classify_and_route(request)
        second_request_time = time.time() - start_time

        # Assert
        assert result1.primary_engine == result2.primary_engine
        assert result1.confidence_score == result2.confidence_score
        assert result1.reasoning == result2.reasoning

        # Cache hit should be significantly faster than API call
        assert second_request_time < first_request_time / 5  # At least 5x faster

    @pytest.mark.asyncio
    async def test_different_llm_providers(self):
        """Test with different LLM providers if available"""
        # Test DashScope
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        test_request = ClassificationRequest(
            user_request="Research artificial intelligence applications in robotics"
        )

        results = {}

        # Test DashScope if available
        if dashscope_key:
            llm_client = create_llm_client("dashscope", api_key=dashscope_key)
            router = SmartRouter(llm_client=llm_client)
            results["dashscope"] = await router.classify_and_route(test_request)

        # Test Anthropic if available
        if anthropic_key:
            llm_client = create_llm_client("anthropic", api_key=anthropic_key)
            router = SmartRouter(llm_client=llm_client)
            results["anthropic"] = await router.classify_and_route(test_request)

        # Test OpenAI if available
        if openai_key:
            llm_client = create_llm_client("openai", api_key=openai_key)
            router = SmartRouter(llm_client=llm_client)
            results["openai"] = await router.classify_and_route(test_request)

        # Assert that all providers give reasonable results
        if not results:
            pytest.skip("No LLM API keys available")

        for provider, result in results.items():
            assert result.primary_engine in ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH", "CODE_RESEARCH"]
            assert 0.0 <= result.confidence_score <= 1.0
            assert isinstance(result.workflow_plan, dict)


@pytest.mark.integration
class TestSmartRouterRealWorldScenarios:
    """Real-world scenario tests"""

    @pytest.mark.asyncio
    async def test_research_workflow_scenarios(self):
        """Test realistic research workflow scenarios"""
        # Arrange
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            pytest.skip("DASHSCOPE_API_KEY not available")

        llm_client = create_llm_client("dashscope", api_key=api_key)
        router = SmartRouter(llm_client=llm_client)

        # Real-world scenarios
        scenarios = [
            {
                "request": "I need to understand the current state of renewable energy adoption and identify market opportunities",
                "expected_types": ["DEEP_RESEARCH"]
            },
            {
                "request": "Help me find and analyze Python web frameworks, compare their performance, and recommend the best one for my project",
                "expected_types": ["CODE_RESEARCH", "SCIENTIFIC_RESEARCH"]
            },
            {
                "request": "Design experiments to validate whether my new neural network architecture performs better than existing solutions",
                "expected_types": ["SCIENTIFIC_RESEARCH"]
            },
            {
                "request": "Research the latest developments in quantum computing and assess their impact on cryptography",
                "expected_types": ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH"]
            }
        ]

        for scenario in scenarios:
            request = ClassificationRequest(user_request=scenario["request"])
            result = await router.classify_and_route(request)

            # Assert reasonable classification
            assert result.primary_engine in (scenario["expected_types"] + ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH", "CODE_RESEARCH"])
            assert 0.0 <= result.confidence_score <= 1.0

            # Check workflow plan appropriateness
            workflow = result.workflow_plan
            assert "sub_workflows" in workflow
            assert len(workflow["sub_workflows"]) > 0

            # Scientific research should have higher complexity
            if result.primary_engine == "SCIENTIFIC_RESEARCH":
                assert workflow.get("complexity_level", 0) > 0.7