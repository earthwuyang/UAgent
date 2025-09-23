"""Unit tests for Deep Research Engine"""

import pytest
from unittest.mock import Mock, AsyncMock

from app.core.research_engines.deep_research import (
    DeepResearchEngine,
    WebSearchEngine,
    AcademicSearchEngine,
    TechnicalSearchEngine,
    SearchResult,
    SearchSource
)
from ..llm_test_utils import require_litellm_client


class TestDeepResearchEngine:
    """Test deep research engine functionality"""

    @pytest.fixture
    def llm_client(self):
        """Create real LiteLLM client for testing."""
        return require_litellm_client()

    @pytest.fixture
    def deep_research_engine(self, llm_client):
        """Create deep research engine for testing"""
        return DeepResearchEngine(llm_client)

    @pytest.mark.asyncio
    async def test_web_search_engine(self, llm_client):
        """Test web search engine"""
        # Arrange
        engine = WebSearchEngine(llm_client)
        query = "artificial intelligence trends"

        # Act
        results = await engine.search(query, limit=5)

        # Assert
        assert isinstance(results, list)
        assert len(results) <= 5
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.title
            assert result.url
            assert result.content
            assert result.source == "web"
            assert 0.0 <= result.relevance_score <= 1.0

    @pytest.mark.asyncio
    async def test_academic_search_engine(self, llm_client):
        """Test academic search engine"""
        # Arrange
        engine = AcademicSearchEngine(llm_client)
        query = "machine learning algorithms"

        # Act
        results = await engine.search(query, limit=3)

        # Assert
        assert isinstance(results, list)
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.source == "academic"
            assert result.relevance_score > 0.0

    @pytest.mark.asyncio
    async def test_technical_search_engine(self, llm_client):
        """Test technical search engine"""
        # Arrange
        engine = TechnicalSearchEngine(llm_client)
        query = "Python FastAPI documentation"

        # Act
        results = await engine.search(query, limit=5)

        # Assert
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.source == "technical"

    @pytest.mark.asyncio
    async def test_deep_research_comprehensive(self, deep_research_engine):
        """Test comprehensive deep research"""
        # Arrange
        query = "renewable energy market trends"

        # Act
        result = await deep_research_engine.research(query)

        # Assert
        assert result.query == query
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert isinstance(result.key_findings, list)
        assert len(result.key_findings) > 0
        assert isinstance(result.sources, list)
        assert len(result.sources) > 0
        assert isinstance(result.analysis, str)
        assert len(result.analysis) > 0
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0
        assert 0.0 <= result.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_deep_research_specific_sources(self, deep_research_engine):
        """Test research with specific sources"""
        # Arrange
        query = "blockchain technology"
        sources = ["web", "academic"]

        # Act
        result = await deep_research_engine.research(query, sources=sources)

        # Assert
        assert result.query == query
        assert len(result.sources) > 0
        # Verify only requested sources are used
        source_types = set(source.source for source in result.sources)
        assert source_types.issubset(set(sources))

    @pytest.mark.asyncio
    async def test_search_specific_source(self, deep_research_engine):
        """Test searching a specific source type"""
        # Arrange
        query = "neural networks"
        source_type = "academic"

        # Act
        results = await deep_research_engine.search_specific_source(
            query, source_type, limit=5
        )

        # Assert
        assert isinstance(results, list)
        assert len(results) <= 5
        for result in results:
            assert result.source == source_type

    @pytest.mark.asyncio
    async def test_search_invalid_source(self, deep_research_engine):
        """Test searching with invalid source type"""
        # Arrange
        query = "test query"
        invalid_source = "invalid_source"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await deep_research_engine.search_specific_source(query, invalid_source)

        assert "Unknown source type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_configure_sources(self, deep_research_engine):
        """Test configuring search sources"""
        # Arrange
        custom_sources = [
            SearchSource("web", "web", enabled=True, priority=1),
            SearchSource("academic", "academic", enabled=False, priority=2)
        ]

        # Act
        deep_research_engine.configure_sources(custom_sources)

        # Assert
        assert deep_research_engine.search_sources == custom_sources

    @pytest.mark.asyncio
    async def test_validate_sources(self, deep_research_engine):
        """Test source validation"""
        # Arrange
        test_results = [
            SearchResult(
                title="Test Article",
                url="https://example.com/test",
                content="Test content",
                source="web",
                relevance_score=0.8
            )
        ]

        # Act
        validated_results = await deep_research_engine.validate_sources(test_results)

        # Assert
        assert len(validated_results) == len(test_results)
        assert validated_results[0].title == test_results[0].title

    @pytest.mark.asyncio
    async def test_synthesis_with_real_llm(self, deep_research_engine):
        """Test synthesis with real LLM integration"""
        # Arrange
        query = "artificial intelligence in healthcare"

        # Act
        result = await deep_research_engine.research(query)

        # Assert - Real LLM should provide comprehensive results
        assert result.query == query
        assert isinstance(result.summary, str)
        assert len(result.summary) > 20  # Real LLM should provide substantial summary
        assert len(result.key_findings) >= 3  # Should identify multiple key findings
        assert result.confidence_score > 0.0
        assert len(result.analysis) > 50  # Should provide detailed analysis
        assert len(result.recommendations) >= 3  # Should provide multiple recommendations


class TestSearchEngineIntegration:
    """Test search engine integration"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        import os
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if dashscope_key:
            return create_llm_client("litellm", api_key=dashscope_key)
        else:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, llm_client):
        """Test concurrent searches across multiple engines"""
        # Arrange
        engines = [
            WebSearchEngine(llm_client),
            AcademicSearchEngine(llm_client),
            TechnicalSearchEngine(llm_client)
        ]
        query = "artificial intelligence"

        # Act
        import asyncio
        tasks = [engine.search(query, limit=3) for engine in engines]
        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 3
        for engine_results in results:
            assert isinstance(engine_results, list)
            assert len(engine_results) <= 3

    @pytest.mark.asyncio
    async def test_result_ranking(self, llm_client):
        """Test result ranking by relevance score"""
        # Arrange
        engine = WebSearchEngine(llm_client)
        query = "machine learning"

        # Act
        results = await engine.search(query, limit=10)

        # Assert
        # Results should be sorted by relevance score (descending)
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score


@pytest.mark.integration
class TestDeepResearchIntegration:
    """Integration tests for deep research engine"""

    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self):
        """Test complete research workflow"""
        # Arrange
        import os
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if not dashscope_key:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("litellm", api_key=dashscope_key)
        engine = DeepResearchEngine(llm_client)
        query = "quantum computing applications"

        # Act
        result = await engine.research(query)

        # Assert complete workflow
        assert result.query == query
        assert len(result.sources) > 0
        assert len(result.key_findings) > 0
        assert len(result.recommendations) > 0

        # Verify sources from multiple engines
        source_types = set(source.source for source in result.sources)
        assert len(source_types) > 1  # Should have multiple source types

    @pytest.mark.asyncio
    async def test_research_quality_metrics(self):
        """Test research quality and completeness"""
        # Arrange
        import os
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if not dashscope_key:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("litellm", api_key=dashscope_key)
        engine = DeepResearchEngine(llm_client)
        query = "sustainable energy solutions"

        # Act
        result = await engine.research(query)

        # Assert quality metrics
        assert result.confidence_score > 0.5
        assert len(result.key_findings) >= 3
        assert len(result.recommendations) >= 3
        assert len(result.summary) > 50  # Reasonable summary length
        assert len(result.analysis) > 100  # Detailed analysis
