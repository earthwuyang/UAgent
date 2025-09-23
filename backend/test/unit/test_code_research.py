"""Unit tests for Code Research Engine"""

import pytest
import os

from app.core.research_engines.code_research import (
    CodeResearchEngine,
    GitHubSearchEngine,
    CodePatternEngine,
    CodeRepository,
    CodePattern,
    CodeAnalysis
)
from app.core.llm_client import create_llm_client


class TestCodeResearchEngine:
    """Test code research engine functionality"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if dashscope_key:
            return create_llm_client("litellm", api_key=dashscope_key)
        else:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

    @pytest.fixture
    def code_research_engine(self, llm_client):
        """Create code research engine for testing"""
        return CodeResearchEngine(llm_client)

    @pytest.fixture
    def github_engine(self, llm_client):
        """Create GitHub search engine for testing"""
        return GitHubSearchEngine(llm_client)

    @pytest.fixture
    def pattern_engine(self, llm_client):
        """Create code pattern engine for testing"""
        return CodePatternEngine(llm_client)

    @pytest.mark.asyncio
    async def test_github_search_repositories(self, github_engine):
        """Test GitHub repository search"""
        # Arrange
        query = "FastAPI REST API"
        language = "Python"

        # Act
        repositories = await github_engine.search_repositories(query, language, limit=5)

        # Assert
        assert isinstance(repositories, list)
        assert len(repositories) <= 5
        for repo in repositories:
            assert isinstance(repo, CodeRepository)
            assert repo.name
            assert repo.url
            assert repo.description
            assert repo.language == language
            assert repo.stars >= 0
            assert repo.forks >= 0

    @pytest.mark.asyncio
    async def test_github_analyze_repository(self, github_engine):
        """Test repository analysis"""
        # Arrange
        repository = CodeRepository(
            name="test-fastapi-app",
            url="https://github.com/test/test-fastapi-app",
            description="A modern FastAPI application with authentication and database",
            language="Python",
            stars=1500,
            forks=200,
            last_updated="2024-01-15",
            topics=["fastapi", "python", "api", "authentication"]
        )

        # Act
        analysis = await github_engine.analyze_repository(repository)

        # Assert
        assert isinstance(analysis, CodeAnalysis)
        assert analysis.repository == repository.name
        assert isinstance(analysis.summary, str)
        assert len(analysis.summary) > 0
        assert isinstance(analysis.key_features, list)
        assert len(analysis.key_features) > 0
        assert isinstance(analysis.technologies_used, list)
        assert isinstance(analysis.quality_metrics, dict)
        assert isinstance(analysis.recommendations, list)

    @pytest.mark.asyncio
    async def test_pattern_identification(self, pattern_engine):
        """Test code pattern identification"""
        # Arrange
        code_content = """
        from fastapi import FastAPI, Depends
        from sqlalchemy.orm import Session

        app = FastAPI()

        class ItemRepository:
            def __init__(self, db: Session):
                self.db = db

            async def get_all(self):
                return self.db.query(Item).all()

        @app.get("/items")
        async def get_items(repo: ItemRepository = Depends()):
            return await repo.get_all()
        """
        language = "Python"

        # Act
        patterns = await pattern_engine.identify_patterns(code_content, language)

        # Assert
        assert isinstance(patterns, list)
        # Real LLM should identify patterns in the code
        for pattern in patterns:
            assert isinstance(pattern, CodePattern)
            assert pattern.language == language
            assert pattern.name
            assert pattern.description

    @pytest.mark.asyncio
    async def test_extract_best_practices(self, pattern_engine):
        """Test best practices extraction"""
        # Arrange
        repositories = [
            CodeRepository(
                name="fastapi-best-practices",
                url="https://github.com/test/fastapi-best-practices",
                description="FastAPI application with security best practices",
                language="Python",
                stars=2000,
                forks=300,
                last_updated="2024-01-15",
                topics=["fastapi", "security", "best-practices"]
            ),
            CodeRepository(
                name="async-patterns",
                url="https://github.com/test/async-patterns",
                description="Async programming patterns in Python",
                language="Python",
                stars=1200,
                forks=150,
                last_updated="2024-01-10",
                topics=["async", "patterns", "python"]
            )
        ]

        # Act
        best_practices = await pattern_engine.extract_best_practices(repositories)

        # Assert
        assert isinstance(best_practices, list)
        for practice in best_practices:
            assert isinstance(practice, CodePattern)
            assert practice.category in ['design_pattern', 'best_practice', 'optimization', 'security']

    @pytest.mark.asyncio
    async def test_comprehensive_code_research(self, code_research_engine):
        """Test comprehensive code research workflow"""
        # Arrange
        query = "machine learning pipelines"
        language = "Python"

        # Act
        result = await code_research_engine.research_code(query, language, include_analysis=True)

        # Assert
        assert result.query == query
        assert isinstance(result.repositories, list)
        assert len(result.repositories) > 0
        assert isinstance(result.analysis, list)
        assert isinstance(result.best_practices, list)
        assert isinstance(result.integration_guide, str)
        assert len(result.integration_guide) > 50  # Should provide substantial guide
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0
        assert 0.0 <= result.confidence_score <= 1.0

        # Verify repository structure
        for repo in result.repositories:
            assert isinstance(repo, CodeRepository)
            assert repo.language == language

        # Verify analysis structure if available
        for analysis in result.analysis:
            assert isinstance(analysis, CodeAnalysis)
            assert len(analysis.key_features) > 0

    @pytest.mark.asyncio
    async def test_research_without_analysis(self, code_research_engine):
        """Test code research without detailed analysis"""
        # Arrange
        query = "web scraping"
        language = "Python"

        # Act
        result = await code_research_engine.research_code(query, language, include_analysis=False)

        # Assert
        assert result.query == query
        assert isinstance(result.repositories, list)
        assert len(result.analysis) == 0  # No analysis requested
        assert isinstance(result.best_practices, list)
        assert isinstance(result.integration_guide, str)

    @pytest.mark.asyncio
    async def test_analyze_specific_repository(self, code_research_engine):
        """Test analysis of specific repository by URL"""
        # Arrange
        repository_url = "https://github.com/tiangolo/fastapi"

        # Act
        analysis = await code_research_engine.analyze_specific_repository(repository_url)

        # Assert
        assert isinstance(analysis, CodeAnalysis)
        assert analysis.repository == "fastapi"  # Extracted from URL
        assert isinstance(analysis.summary, str)
        assert len(analysis.summary) > 0
        assert isinstance(analysis.key_features, list)

    @pytest.mark.asyncio
    async def test_find_implementation_examples(self, code_research_engine):
        """Test finding implementation examples"""
        # Arrange
        concept = "singleton pattern"
        language = "Python"

        # Act
        examples = await code_research_engine.find_implementation_examples(concept, language)

        # Assert
        assert isinstance(examples, list)
        for example in examples:
            assert isinstance(example, CodePattern)
            assert example.language == language

    @pytest.mark.asyncio
    async def test_compare_implementations(self, code_research_engine):
        """Test comparing different implementations"""
        # Arrange
        implementations = ["FastAPI", "Flask", "Django REST"]
        language = "Python"

        # Act
        comparison = await code_research_engine.compare_implementations(implementations, language)

        # Assert
        assert isinstance(comparison, dict)
        for impl in implementations:
            if impl in comparison:
                assert "repository" in comparison[impl]
                assert "analysis" in comparison[impl]
                assert "quality_score" in comparison[impl]
                assert isinstance(comparison[impl]["quality_score"], float)

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, code_research_engine):
        """Test confidence score calculation"""
        # Arrange
        repositories = [
            CodeRepository(
                name="high-quality-repo",
                url="https://github.com/test/high-quality",
                description="High quality implementation",
                language="Python",
                stars=5000,  # High stars
                forks=1000,
                last_updated="2024-01-15",
                topics=["quality", "production"]
            )
        ]

        analyses = [
            CodeAnalysis(
                repository="high-quality-repo",
                summary="High quality repository",
                architecture_overview="Well structured",
                key_features=["Feature 1", "Feature 2"],
                technologies_used=["Python"],
                patterns_identified=[],
                quality_metrics={"maintainability": 0.9, "test_coverage": 0.8},  # High quality
                recommendations=["Use in production"]
            )
        ]

        # Act
        confidence = code_research_engine._calculate_confidence_score(repositories, analyses)

        # Assert
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to good metrics

    @pytest.mark.asyncio
    async def test_error_handling_empty_results(self, code_research_engine):
        """Test error handling with no repositories found"""
        # Arrange
        query = "extremely_specific_nonexistent_library_xyz123"

        # Act
        result = await code_research_engine.research_code(query)

        # Assert
        # Should handle gracefully even with no results
        assert result.query == query
        assert isinstance(result.repositories, list)
        assert isinstance(result.analysis, list)
        assert isinstance(result.integration_guide, str)
        assert isinstance(result.recommendations, list)
        assert 0.0 <= result.confidence_score <= 1.0


@pytest.mark.integration
class TestCodeResearchIntegration:
    """Integration tests for code research engine"""

    @pytest.mark.asyncio
    async def test_end_to_end_code_research_workflow(self):
        """Test complete code research workflow"""
        # Arrange
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if not dashscope_key:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("litellm", api_key=dashscope_key)
        engine = CodeResearchEngine(llm_client)
        query = "microservices architecture Python"

        # Act
        result = await engine.research_code(query, language="Python", include_analysis=True)

        # Assert complete workflow
        assert result.query == query
        assert len(result.repositories) > 0
        assert len(result.analysis) > 0
        assert len(result.best_practices) > 0
        assert len(result.integration_guide) > 100  # Substantial guide
        assert len(result.recommendations) >= 3

        # Verify real LLM provides quality analysis
        first_analysis = result.analysis[0]
        assert len(first_analysis.summary) > 50
        assert len(first_analysis.key_features) >= 2
        assert len(first_analysis.technologies_used) >= 1

    @pytest.mark.asyncio
    async def test_real_llm_pattern_extraction(self):
        """Test pattern extraction with real LLM"""
        # Arrange
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if not dashscope_key:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("litellm", api_key=dashscope_key)
        pattern_engine = CodePatternEngine(llm_client)

        code_content = """
        from abc import ABC, abstractmethod

        class PaymentProcessor(ABC):
            @abstractmethod
            def process_payment(self, amount: float) -> bool:
                pass

        class CreditCardProcessor(PaymentProcessor):
            def process_payment(self, amount: float) -> bool:
                # Process credit card payment
                return True

        class PayPalProcessor(PaymentProcessor):
            def process_payment(self, amount: float) -> bool:
                # Process PayPal payment
                return True
        """

        # Act
        patterns = await pattern_engine.identify_patterns(code_content, "Python")

        # Assert real LLM identifies patterns
        assert len(patterns) > 0
        pattern_names = [p.name.lower() for p in patterns]
        # Should identify strategy pattern or abstract base class pattern
        assert any("strategy" in name or "abstract" in name or "interface" in name for name in pattern_names)

    @pytest.mark.asyncio
    async def test_quality_metrics_with_real_analysis(self):
        """Test quality metrics generation with real LLM"""
        # Arrange
        dashscope_key = os.getenv("LITELLM_API_KEY")
        if not dashscope_key:
            pytest.skip("LITELLM_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("litellm", api_key=dashscope_key)
        engine = CodeResearchEngine(llm_client)

        # Act
        result = await engine.research_code("data visualization Python", include_analysis=True)

        # Assert quality metrics from real LLM analysis
        if result.analysis:
            analysis = result.analysis[0]
            quality_metrics = analysis.quality_metrics

            # Should have meaningful quality metrics
            assert "maintainability" in quality_metrics
            assert isinstance(quality_metrics["maintainability"], (int, float))
            assert 0.0 <= quality_metrics["maintainability"] <= 1.0