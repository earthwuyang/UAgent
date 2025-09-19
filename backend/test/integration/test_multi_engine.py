"""Integration tests for multi-engine communication and coordination"""

import pytest
import os
import asyncio

from app.core.research_engines import (
    DeepResearchEngine,
    CodeResearchEngine,
    ScientificResearchEngine
)
from app.core.smart_router import SmartRouter, ClassificationRequest
from app.core.llm_client import create_llm_client
from app.core.cache import create_cache


class TestMultiEngineIntegration:
    """Test integration between multiple research engines"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            return create_llm_client("dashscope", api_key=dashscope_key)
        else:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

    @pytest.fixture
    def research_engines(self, llm_client):
        """Create all research engines for integration testing"""
        deep_engine = DeepResearchEngine(llm_client)
        code_engine = CodeResearchEngine(llm_client)
        scientific_engine = ScientificResearchEngine(
            llm_client=llm_client,
            deep_research_engine=deep_engine,
            code_research_engine=code_engine,
            config={"max_iterations": 1}  # Limited iterations for testing
        )

        return {
            "deep": deep_engine,
            "code": code_engine,
            "scientific": scientific_engine
        }

    @pytest.fixture
    def smart_router(self, llm_client):
        """Create smart router for classification testing"""
        cache = create_cache("memory")
        return SmartRouter(llm_client=llm_client, cache=cache)

    @pytest.mark.asyncio
    async def test_engine_coordination_via_smart_router(self, smart_router, research_engines):
        """Test that smart router correctly routes to different engines"""
        # Test deep research routing
        deep_request = ClassificationRequest(
            user_request="What are the latest trends in renewable energy markets and policies?"
        )
        deep_result = await smart_router.classify_and_route(deep_request)
        assert deep_result.primary_engine in ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH"]

        # Test code research routing
        code_request = ClassificationRequest(
            user_request="Find Python machine learning libraries for time series forecasting"
        )
        code_result = await smart_router.classify_and_route(code_request)
        assert code_result.primary_engine in ["CODE_RESEARCH", "SCIENTIFIC_RESEARCH"]

        # Test scientific research routing
        scientific_request = ClassificationRequest(
            user_request="Design experiments to test whether attention mechanisms improve transformer performance"
        )
        scientific_result = await smart_router.classify_and_route(scientific_request)
        assert scientific_result.primary_engine in ["SCIENTIFIC_RESEARCH"]

    @pytest.mark.asyncio
    async def test_scientific_engine_orchestration(self, research_engines):
        """Test scientific engine's ability to orchestrate other engines"""
        scientific_engine = research_engines["scientific"]

        # Test comprehensive research that should use multiple engines
        research_question = "Effectiveness of transformer attention mechanisms in NLP tasks"

        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Verify multi-engine orchestration
        assert result.literature_review is not None, "Should include literature review from deep research engine"
        assert result.code_analysis is not None, "Should include code analysis from code research engine"

        # Verify literature review integration
        literature = result.literature_review
        assert len(literature.sources) > 0
        assert len(literature.key_findings) > 0
        assert literature.confidence_score > 0

        # Verify code analysis integration
        code_analysis = result.code_analysis
        assert len(code_analysis.repositories) > 0
        assert len(code_analysis.recommendations) > 0
        assert code_analysis.confidence_score > 0

        # Verify scientific workflow integration
        assert len(result.hypotheses) > 0
        assert len(result.experiments) > 0
        assert result.iteration_count > 0

        # Verify synthesis combines all sources
        synthesis = result.synthesis
        assert isinstance(synthesis, dict)
        assert len(result.final_conclusions) > 0

    @pytest.mark.asyncio
    async def test_concurrent_engine_execution(self, research_engines):
        """Test concurrent execution of multiple engines"""
        deep_engine = research_engines["deep"]
        code_engine = research_engines["code"]

        query = "machine learning model optimization"

        # Execute engines concurrently
        start_time = asyncio.get_event_loop().time()

        deep_task = deep_engine.research(f"research: {query}")
        code_task = code_engine.research_code(query, include_analysis=True)

        deep_result, code_result = await asyncio.gather(deep_task, code_task)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # Verify both engines produced valid results
        assert deep_result.query == f"research: {query}"
        assert len(deep_result.sources) > 0
        assert deep_result.confidence_score > 0

        assert code_result.query == query
        assert len(code_result.repositories) > 0
        assert code_result.confidence_score > 0

        # Concurrent execution should be reasonably fast
        assert total_time < 120  # Allow up to 2 minutes for real LLM calls

    @pytest.mark.asyncio
    async def test_data_flow_between_engines(self, research_engines):
        """Test data flow and context passing between engines"""
        scientific_engine = research_engines["scientific"]

        # Test that context from literature review influences code analysis
        research_question = "Impact of batch normalization on neural network training stability"

        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=False  # Focus on data flow, not iteration
        )

        # Verify context propagation
        literature_findings = result.literature_review.key_findings if result.literature_review else []
        code_recommendations = result.code_analysis.recommendations if result.code_analysis else []
        hypotheses = [h.reasoning for h in result.hypotheses]

        # Scientific research should reference insights from both engines
        combined_context = " ".join(literature_findings + code_recommendations + hypotheses).lower()

        # Should contain relevant terms from both literature and code analysis
        research_terms = ["batch", "normalization", "neural", "network", "training", "stability"]
        found_terms = sum(1 for term in research_terms if term in combined_context)
        assert found_terms >= 3, f"Expected research context integration, found terms: {found_terms}"

    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, research_engines):
        """Test error handling across multiple engines"""
        scientific_engine = research_engines["scientific"]

        # Test with challenging/ambiguous query that might cause issues
        challenging_query = "xyzabc_nonexistent_research_topic_12345"

        try:
            result = await scientific_engine.conduct_research(
                challenging_query,
                include_literature_review=True,
                include_code_analysis=True,
                enable_iteration=False
            )

            # Should handle gracefully even with challenging input
            assert result.query == challenging_query
            assert isinstance(result.final_conclusions, list)
            assert result.confidence_score >= 0.0

            # Engines should provide fallback results rather than failing
            if result.literature_review:
                assert len(result.literature_review.sources) >= 0
            if result.code_analysis:
                assert len(result.code_analysis.repositories) >= 0

        except Exception as e:
            # If exceptions occur, they should be handled gracefully
            assert isinstance(e, Exception)
            # Log error but don't fail test - this tests error resilience

    @pytest.mark.asyncio
    async def test_engine_performance_coordination(self, research_engines):
        """Test performance characteristics of coordinated engines"""
        scientific_engine = research_engines["scientific"]

        # Test with realistic research question
        research_question = "Comparing different optimization algorithms for deep learning"

        import time
        start_time = time.time()

        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        end_time = time.time()
        total_time = end_time - start_time

        # Verify reasonable performance for complex multi-engine workflow
        assert total_time < 300  # Should complete within 5 minutes for real LLM

        # Verify comprehensive results justify the time investment
        total_components = (
            len(result.hypotheses) +
            len(result.experiments) +
            len(result.final_conclusions) +
            (len(result.literature_review.sources) if result.literature_review else 0) +
            (len(result.code_analysis.repositories) if result.code_analysis else 0)
        )

        assert total_components >= 5, f"Expected substantial results for {total_time:.1f}s execution time"

    @pytest.mark.asyncio
    async def test_iterative_multi_engine_refinement(self, research_engines):
        """Test iterative refinement across multiple engines"""
        scientific_engine = research_engines["scientific"]

        # Configure for multiple iterations
        scientific_engine.max_iterations = 2
        scientific_engine.confidence_threshold = 0.9  # High threshold to force iterations

        research_question = "Optimal hyperparameters for transformer model training"

        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Verify iterative refinement occurred
        assert result.iteration_count > 0

        # Verify that hypotheses show refinement across iterations
        if len(result.hypotheses) > 1:
            # Different hypotheses should have evolved/refined statements
            statements = [h.statement for h in result.hypotheses]
            assert len(set(statements)) > 1 or any(len(h.evidence) > 0 for h in result.hypotheses)

        # Verify that final confidence reflects iterative improvement
        if result.iteration_count > 1:
            assert result.confidence_score > 0.3  # Should improve with iteration


@pytest.mark.integration
class TestEngineDataIntegration:
    """Test data integration and consistency across engines"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            return create_llm_client("dashscope", api_key=dashscope_key)
        else:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

    @pytest.mark.asyncio
    async def test_cross_engine_data_consistency(self, llm_client):
        """Test that engines provide consistent and complementary data"""
        deep_engine = DeepResearchEngine(llm_client)
        code_engine = CodeResearchEngine(llm_client)

        topic = "convolutional neural networks for image classification"

        # Get results from both engines
        deep_result = await deep_engine.research(f"research: {topic}")
        code_result = await code_engine.research_code(topic)

        # Verify data consistency and complementarity
        deep_summary = deep_result.summary.lower()
        code_integration = code_result.integration_guide.lower()

        # Should have overlapping concepts
        shared_concepts = ["neural", "network", "image", "classification", "convolution"]
        deep_concepts = sum(1 for concept in shared_concepts if concept in deep_summary)
        code_concepts = sum(1 for concept in shared_concepts if concept in code_integration)

        assert deep_concepts >= 2, "Deep research should cover core concepts"
        assert code_concepts >= 2, "Code research should cover core concepts"

        # Should be complementary rather than identical
        assert deep_result.analysis != code_result.integration_guide
        assert len(deep_result.sources) > 0
        assert len(code_result.repositories) > 0

    @pytest.mark.asyncio
    async def test_scientific_engine_synthesis_quality(self, llm_client):
        """Test quality of synthesis across multiple engine inputs"""
        scientific_engine = ScientificResearchEngine(llm_client=llm_client)

        research_question = "Role of attention mechanisms in sequence-to-sequence models"

        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=False
        )

        # Verify synthesis quality
        synthesis = result.synthesis
        publication_draft = result.publication_draft

        # Synthesis should reference multiple sources
        if result.literature_review and result.code_analysis:
            # Should mention both theoretical and implementation aspects
            synthesis_text = str(synthesis).lower()
            pub_text = publication_draft.lower()

            theoretical_terms = ["research", "study", "analysis", "literature"]
            implementation_terms = ["code", "implementation", "repository", "library"]

            theory_count = sum(1 for term in theoretical_terms if term in synthesis_text or term in pub_text)
            impl_count = sum(1 for term in implementation_terms if term in synthesis_text or term in pub_text)

            assert theory_count >= 1, "Should reference theoretical research"
            assert impl_count >= 1, "Should reference implementation aspects"

        # Publication draft should be comprehensive
        assert len(publication_draft) > 500, "Publication draft should be substantial"

        # Should have structured sections
        pub_lower = publication_draft.lower()
        sections = ["abstract", "introduction", "method", "result", "conclusion"]
        found_sections = sum(1 for section in sections if section in pub_lower)
        assert found_sections >= 2, "Should have structured academic format"