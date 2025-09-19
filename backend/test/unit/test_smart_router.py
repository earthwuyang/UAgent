"""
Unit tests for smart router functionality
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.smart_router import (
    SmartRouter,
    EngineType,
    ClassificationRequest,
    ClassificationResult,
    ClassificationError,
    InvalidRequestError,
    ThresholdError
)


class TestEngineType:
    """Test EngineType enum"""

    def test_engine_type_values(self):
        """Test engine type values"""
        assert EngineType.DEEP_RESEARCH == "DEEP_RESEARCH"
        assert EngineType.SCIENTIFIC_RESEARCH == "SCIENTIFIC_RESEARCH"
        assert EngineType.CODE_RESEARCH == "CODE_RESEARCH"

    def test_engine_type_list(self):
        """Test getting all engine types"""
        all_engines = [e.value for e in EngineType]
        assert len(all_engines) == 3
        assert "DEEP_RESEARCH" in all_engines
        assert "SCIENTIFIC_RESEARCH" in all_engines
        assert "CODE_RESEARCH" in all_engines


class TestClassificationRequest:
    """Test ClassificationRequest dataclass"""

    def test_classification_request_creation(self):
        """Test creating classification request"""
        request = ClassificationRequest(user_request="Test request")

        assert request.user_request == "Test request"
        assert request.context is None
        assert request.override_engine is None
        assert request.confidence_threshold == 0.7

    def test_classification_request_with_all_fields(self):
        """Test creating classification request with all fields"""
        context = {"session_id": "123"}
        request = ClassificationRequest(
            user_request="Test request",
            context=context,
            override_engine="DEEP_RESEARCH",
            confidence_threshold=0.8
        )

        assert request.user_request == "Test request"
        assert request.context == context
        assert request.override_engine == "DEEP_RESEARCH"
        assert request.confidence_threshold == 0.8


class TestClassificationResult:
    """Test ClassificationResult dataclass"""

    def test_classification_result_creation(self):
        """Test creating classification result"""
        result = ClassificationResult(
            primary_engine="DEEP_RESEARCH",
            confidence_score=0.85,
            sub_components={"deep_research": True},
            reasoning="Test reasoning",
            workflow_plan={"phases": ["research"]}
        )

        assert result.primary_engine == "DEEP_RESEARCH"
        assert result.confidence_score == 0.85
        assert result.sub_components == {"deep_research": True}
        assert result.reasoning == "Test reasoning"
        assert result.workflow_plan == {"phases": ["research"]}


class TestSmartRouter:
    """Test SmartRouter functionality"""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client"""
        mock_client = AsyncMock()
        return mock_client

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache"""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Default to cache miss
        return mock_cache

    @pytest.fixture
    def router(self, mock_llm_client, mock_cache):
        """Create SmartRouter instance for testing"""
        return SmartRouter(mock_llm_client, mock_cache)

    def test_router_initialization(self, mock_llm_client, mock_cache):
        """Test router initialization"""
        router = SmartRouter(mock_llm_client, mock_cache)

        assert router.llm_client == mock_llm_client
        assert router.cache == mock_cache
        assert router.config == {}
        assert router.max_retries == 3
        assert router.cache_ttl == 86400

    def test_router_initialization_with_config(self, mock_llm_client, mock_cache):
        """Test router initialization with custom config"""
        config = {"max_retries": 5, "cache_ttl": 3600}
        router = SmartRouter(mock_llm_client, mock_cache, config)

        assert router.config == config
        assert router.max_retries == 5
        assert router.cache_ttl == 3600

    def test_router_initialization_without_cache(self, mock_llm_client):
        """Test router initialization without cache"""
        router = SmartRouter(mock_llm_client)

        assert router.llm_client == mock_llm_client
        assert router.cache is None

    @pytest.mark.asyncio
    async def test_classify_and_route_success(self, router, mock_llm_client):
        """Test successful classification and routing"""
        mock_llm_client.classify.return_value = {
            "engine": "DEEP_RESEARCH",
            "confidence_score": 0.85,
            "reasoning": "This is a deep research request",
            "sub_components": {"deep_research": True}
        }

        request = ClassificationRequest(user_request="Research quantum computing")
        result = await router.classify_and_route(request)

        assert isinstance(result, ClassificationResult)
        assert result.primary_engine == "DEEP_RESEARCH"
        assert result.confidence_score == 0.85
        assert result.reasoning == "This is a deep research request"
        assert result.sub_components == {"deep_research": True}
        assert "primary_engine" in result.workflow_plan

    @pytest.mark.asyncio
    async def test_classify_and_route_empty_request(self, router):
        """Test classification with empty request"""
        request = ClassificationRequest(user_request="")

        with pytest.raises(InvalidRequestError):
            await router.classify_and_route(request)

    @pytest.mark.asyncio
    async def test_classify_and_route_whitespace_request(self, router):
        """Test classification with whitespace-only request"""
        request = ClassificationRequest(user_request="   ")

        with pytest.raises(InvalidRequestError):
            await router.classify_and_route(request)

    @pytest.mark.asyncio
    async def test_classify_and_route_with_override(self, router):
        """Test classification with engine override"""
        request = ClassificationRequest(
            user_request="Test request",
            override_engine="SCIENTIFIC_RESEARCH"
        )

        result = await router.classify_and_route(request)

        assert result.primary_engine == "SCIENTIFIC_RESEARCH"
        assert result.confidence_score == 1.0
        assert result.reasoning == "Manual override to SCIENTIFIC_RESEARCH"
        assert result.sub_components == {"manual_override": True}

    @pytest.mark.asyncio
    async def test_classify_and_route_invalid_override(self, router):
        """Test classification with invalid engine override"""
        request = ClassificationRequest(
            user_request="Test request",
            override_engine="INVALID_ENGINE"
        )

        with pytest.raises(InvalidRequestError):
            await router.classify_and_route(request)

    @pytest.mark.asyncio
    async def test_classify_and_route_low_confidence(self, router, mock_llm_client):
        """Test classification with confidence below threshold"""
        mock_llm_client.classify.return_value = {
            "engine": "DEEP_RESEARCH",
            "confidence_score": 0.5,
            "reasoning": "Low confidence classification",
            "sub_components": {"deep_research": True}
        }

        request = ClassificationRequest(
            user_request="Ambiguous request",
            confidence_threshold=0.7
        )

        with pytest.raises(ThresholdError):
            await router.classify_and_route(request)

    @pytest.mark.asyncio
    async def test_classify_and_route_cache_hit(self, router, mock_cache):
        """Test classification with cache hit"""
        cached_result = {
            "primary_engine": "CODE_RESEARCH",
            "confidence_score": 0.9,
            "sub_components": {"code_research": True},
            "reasoning": "Cached result",
            "workflow_plan": {"phases": ["analysis"]}
        }
        mock_cache.get.return_value = cached_result

        request = ClassificationRequest(user_request="Test request")
        result = await router.classify_and_route(request)

        assert result.primary_engine == "CODE_RESEARCH"
        assert result.confidence_score == 0.9
        # LLM should not be called for cache hit
        router.llm_client.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_classify_and_route_cache_miss_then_set(self, router, mock_llm_client, mock_cache):
        """Test classification with cache miss followed by cache set"""
        mock_cache.get.return_value = None  # Cache miss

        mock_llm_client.classify.return_value = {
            "engine": "SCIENTIFIC_RESEARCH",
            "confidence_score": 0.92,
            "reasoning": "Scientific research request",
            "sub_components": {"experimentation": True}
        }

        request = ClassificationRequest(user_request="Design ML experiments")
        result = await router.classify_and_route(request)

        # Should call LLM for cache miss
        mock_llm_client.classify.assert_called_once()

        # Should set result in cache
        mock_cache.set.assert_called_once()

        assert result.primary_engine == "SCIENTIFIC_RESEARCH"
        assert result.confidence_score == 0.92

    @pytest.mark.asyncio
    async def test_classify_with_retry_success(self, router, mock_llm_client):
        """Test successful classification with retry mechanism"""
        mock_llm_client.classify.return_value = {
            "engine": "DEEP_RESEARCH",
            "confidence_score": 0.8,
            "reasoning": "Test reasoning",
            "sub_components": {"deep_research": True}
        }

        request = ClassificationRequest(user_request="Test request")
        result = await router._classify_with_retry(request)

        assert result["engine"] == "DEEP_RESEARCH"
        assert result["confidence_score"] == 0.8

    @pytest.mark.asyncio
    async def test_classify_with_retry_failure_then_success(self, router, mock_llm_client):
        """Test classification retry after initial failure"""
        # First call fails, second succeeds
        mock_llm_client.classify.side_effect = [
            Exception("Network error"),
            {
                "engine": "CODE_RESEARCH",
                "confidence_score": 0.75,
                "reasoning": "Retry success",
                "sub_components": {"code_research": True}
            }
        ]

        request = ClassificationRequest(user_request="Test request")
        result = await router._classify_with_retry(request)

        assert result["engine"] == "CODE_RESEARCH"
        assert mock_llm_client.classify.call_count == 2

    @pytest.mark.asyncio
    async def test_classify_with_retry_invalid_response(self, router, mock_llm_client):
        """Test classification with invalid LLM response format"""
        mock_llm_client.classify.return_value = {
            "engine": "DEEP_RESEARCH",
            # Missing required fields
        }

        request = ClassificationRequest(user_request="Test request")

        with pytest.raises(ClassificationError):
            await router._classify_with_retry(request)

    @pytest.mark.asyncio
    async def test_classify_with_retry_max_attempts(self, router, mock_llm_client):
        """Test classification failing after max retry attempts"""
        mock_llm_client.classify.side_effect = Exception("Persistent error")

        request = ClassificationRequest(user_request="Test request")

        with pytest.raises(ClassificationError, match="Classification failed after 3 attempts"):
            await router._classify_with_retry(request)

        assert mock_llm_client.classify.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_scientific_workflow(self, router):
        """Test generating scientific research workflow"""
        sub_components = {
            "deep_research": True,
            "code_research": True,
            "experimentation": True,
            "iteration": True
        }

        workflow = await router._generate_scientific_workflow("Test request", sub_components)

        assert workflow["primary_engine"] == "scientific_research"
        assert workflow["complexity_level"] == 0.9
        assert workflow["iteration_enabled"] is True
        assert workflow["feedback_loops"] is True
        assert workflow["max_iterations"] == 5

        # Should have multiple sub-workflows
        assert len(workflow["sub_workflows"]) >= 4

        # Check for specific phases
        phases = [w["phase"] for w in workflow["sub_workflows"]]
        assert "literature_review" in phases
        assert "implementation_analysis" in phases
        assert "hypothesis_generation" in phases
        assert "experimental_design" in phases

    @pytest.mark.asyncio
    async def test_generate_deep_research_workflow(self, router):
        """Test generating deep research workflow"""
        workflow = await router._generate_deep_research_workflow("Test request", {})

        assert workflow["primary_engine"] == "deep_research"
        assert workflow["complexity_level"] == 0.5
        assert workflow["iteration_enabled"] is False
        assert workflow["feedback_loops"] is False

        # Check for specific phases
        phases = [w["phase"] for w in workflow["sub_workflows"]]
        assert "multi_source_search" in phases
        assert "synthesis_and_analysis" in phases
        assert "report_generation" in phases

    @pytest.mark.asyncio
    async def test_generate_code_research_workflow(self, router):
        """Test generating code research workflow"""
        workflow = await router._generate_code_research_workflow("Test request", {})

        assert workflow["primary_engine"] == "code_research"
        assert workflow["complexity_level"] == 0.6
        assert workflow["iteration_enabled"] is False
        assert workflow["feedback_loops"] is False
        assert workflow["includes_openhands"] is True

        # Check for specific phases
        phases = [w["phase"] for w in workflow["sub_workflows"]]
        assert "repository_discovery" in phases
        assert "code_analysis" in phases
        assert "documentation_generation" in phases

    def test_generate_cache_key(self, router):
        """Test cache key generation"""
        request = "Test request for caching"
        cache_key = router._generate_cache_key(request)

        assert isinstance(cache_key, str)
        assert cache_key.startswith("router:classification:")
        assert len(cache_key.split(":")[-1]) == 16  # Hash should be 16 chars

    def test_generate_cache_key_normalization(self, router):
        """Test cache key normalization"""
        request1 = "Test request"
        request2 = "  Test request  "
        request3 = "TEST REQUEST"

        key1 = router._generate_cache_key(request1)
        key2 = router._generate_cache_key(request2)
        key3 = router._generate_cache_key(request3)

        # Should normalize whitespace and case for consistent caching
        assert key1 == key2  # Whitespace normalized
        assert key1 == key3  # Case normalized

    @pytest.mark.asyncio
    async def test_get_classification_stats(self, router):
        """Test getting classification statistics"""
        stats = await router.get_classification_stats()

        assert isinstance(stats, dict)
        assert "total_classifications" in stats
        assert "accuracy_rate" in stats
        assert "cache_hit_rate" in stats
        assert "average_response_time" in stats
        assert "engine_distribution" in stats

        # Engine distribution should have all engine types
        engine_dist = stats["engine_distribution"]
        assert "DEEP_RESEARCH" in engine_dist
        assert "SCIENTIFIC_RESEARCH" in engine_dist
        assert "CODE_RESEARCH" in engine_dist

    @pytest.mark.asyncio
    async def test_workflow_plan_generation_invalid_engine(self, router):
        """Test workflow plan generation with invalid engine"""
        with pytest.raises(ValueError):
            await router._generate_workflow_plan("INVALID_ENGINE", "test", {})

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, router, mock_llm_client, mock_cache):
        """Test handling of cache errors (should not break classification)"""
        # Cache operations fail but classification should still work
        mock_cache.get.side_effect = Exception("Cache unavailable")
        mock_cache.set.side_effect = Exception("Cache unavailable")

        mock_llm_client.classify.return_value = {
            "engine": "DEEP_RESEARCH",
            "confidence_score": 0.8,
            "reasoning": "Classification works despite cache errors",
            "sub_components": {"deep_research": True}
        }

        request = ClassificationRequest(user_request="Test request")
        result = await router.classify_and_route(request)

        assert result.primary_engine == "DEEP_RESEARCH"
        assert result.confidence_score == 0.8

    def test_classification_prompt_content(self, router):
        """Test classification prompt contains expected content"""
        prompt = router.classification_prompt

        assert "DEEP_RESEARCH" in prompt
        assert "SCIENTIFIC_RESEARCH" in prompt
        assert "CODE_RESEARCH" in prompt
        assert "MOST COMPLEX" in prompt
        assert "JSON format" in prompt
        assert "confidence_score" in prompt
        assert "sub_components" in prompt

    @pytest.mark.asyncio
    async def test_concurrent_classifications(self, router, mock_llm_client, mock_cache):
        """Test concurrent classification requests"""
        mock_cache.get.return_value = None  # Cache miss

        def mock_classify(request, prompt):
            if "research" in request.lower():
                return {
                    "engine": "DEEP_RESEARCH",
                    "confidence_score": 0.85,
                    "reasoning": "Research request",
                    "sub_components": {"deep_research": True}
                }
            else:
                return {
                    "engine": "CODE_RESEARCH",
                    "confidence_score": 0.8,
                    "reasoning": "Code request",
                    "sub_components": {"code_research": True}
                }

        mock_llm_client.classify.side_effect = mock_classify

        # Run multiple classifications concurrently
        import asyncio
        requests = [
            ClassificationRequest(user_request="Research quantum computing"),
            ClassificationRequest(user_request="Analyze code quality"),
            ClassificationRequest(user_request="Find research papers"),
            ClassificationRequest(user_request="Review code patterns")
        ]

        tasks = [router.classify_and_route(req) for req in requests]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(isinstance(result, ClassificationResult) for result in results)
        assert all(result.confidence_score > 0.7 for result in results)