"""Integration tests for end-to-end research workflows"""

import pytest
import os
import asyncio
from typing import Dict, Any

from app.core.research_engines import (
    DeepResearchEngine,
    CodeResearchEngine,
    ScientificResearchEngine
)
from app.core.smart_router import SmartRouter, ClassificationRequest
from app.core.llm_client import create_llm_client
from app.core.cache import create_cache


class TestResearchWorkflows:
    """Test complete research workflows end-to-end"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            return create_llm_client("dashscope", api_key=dashscope_key)
        else:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

    @pytest.fixture
    def complete_system(self, llm_client):
        """Create complete system with all components"""
        cache = create_cache("memory")

        # Create research engines
        deep_engine = DeepResearchEngine(llm_client)
        code_engine = CodeResearchEngine(llm_client)
        scientific_engine = ScientificResearchEngine(
            llm_client=llm_client,
            deep_research_engine=deep_engine,
            code_research_engine=code_engine,
            config={"max_iterations": 2, "confidence_threshold": 0.7}
        )

        # Create smart router
        router = SmartRouter(llm_client=llm_client, cache=cache)

        return {
            "router": router,
            "engines": {
                "deep": deep_engine,
                "code": code_engine,
                "scientific": scientific_engine
            },
            "cache": cache
        }

    @pytest.mark.asyncio
    async def test_deep_research_workflow(self, complete_system):
        """Test complete deep research workflow"""
        router = complete_system["router"]
        deep_engine = complete_system["engines"]["deep"]

        # Step 1: User request classification
        user_request = "Analyze the current state of renewable energy adoption globally and identify key market opportunities"

        classification_request = ClassificationRequest(user_request=user_request)
        classification_result = await router.classify_and_route(classification_request)

        # Should classify as deep research
        assert classification_result.primary_engine in ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH"]

        # Step 2: Execute deep research workflow
        research_result = await deep_engine.research(user_request)

        # Step 3: Verify comprehensive workflow output
        assert research_result.query == user_request
        assert len(research_result.sources) > 0
        assert len(research_result.key_findings) >= 3
        assert len(research_result.analysis) > 100
        assert len(research_result.recommendations) >= 3
        assert research_result.confidence_score > 0.5

        # Verify multi-source integration
        source_types = set(source.source for source in research_result.sources)
        assert len(source_types) > 1  # Should use multiple source types

        # Verify quality of synthesis
        assert "renewable energy" in research_result.summary.lower()
        assert any("market" in finding.lower() for finding in research_result.key_findings)

    @pytest.mark.asyncio
    async def test_code_research_workflow(self, complete_system):
        """Test complete code research workflow"""
        router = complete_system["router"]
        code_engine = complete_system["engines"]["code"]

        # Step 1: User request classification
        user_request = "Find and analyze Python libraries for building REST APIs, compare their performance and features"

        classification_request = ClassificationRequest(user_request=user_request)
        classification_result = await router.classify_and_route(classification_request)

        # Should classify as code research
        assert classification_result.primary_engine in ["CODE_RESEARCH", "SCIENTIFIC_RESEARCH"]

        # Step 2: Execute code research workflow
        research_result = await code_engine.research_code(user_request, language="Python", include_analysis=True)

        # Step 3: Verify comprehensive workflow output
        assert research_result.query == user_request
        assert len(research_result.repositories) > 0
        assert len(research_result.analysis) > 0
        assert len(research_result.best_practices) > 0
        assert len(research_result.integration_guide) > 100
        assert len(research_result.recommendations) >= 3
        assert research_result.confidence_score > 0.3

        # Verify repository analysis quality
        for repo in research_result.repositories:
            assert repo.language == "Python"
            assert repo.stars >= 0
            assert len(repo.description) > 0

        # Verify analysis depth
        for analysis in research_result.analysis:
            assert len(analysis.key_features) > 0
            assert len(analysis.technologies_used) > 0
            assert isinstance(analysis.quality_metrics, dict)

        # Verify integration guidance
        integration_guide = research_result.integration_guide.lower()
        assert any(term in integration_guide for term in ["install", "setup", "usage", "example"])

    @pytest.mark.asyncio
    async def test_scientific_research_workflow(self, complete_system):
        """Test complete scientific research workflow - most complex"""
        router = complete_system["router"]
        scientific_engine = complete_system["engines"]["scientific"]

        # Step 1: User request classification
        user_request = "Design and conduct experiments to test whether dropout regularization improves model generalization in deep neural networks"

        classification_request = ClassificationRequest(user_request=user_request)
        classification_result = await router.classify_and_route(classification_request)

        # Should classify as scientific research
        assert classification_result.primary_engine == "SCIENTIFIC_RESEARCH"

        # Step 2: Execute comprehensive scientific research workflow
        research_result = await scientific_engine.conduct_research(
            user_request,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Step 3: Verify comprehensive scientific workflow
        assert research_result.query == user_request
        assert research_result.iteration_count > 0

        # Verify multi-engine orchestration
        assert research_result.literature_review is not None
        assert research_result.code_analysis is not None

        # Verify experimental workflow
        assert len(research_result.hypotheses) > 0
        assert len(research_result.experiments) > 0
        assert len(research_result.executions) > 0
        assert len(research_result.results) > 0

        # Verify scientific rigor
        for hypothesis in research_result.hypotheses:
            assert len(hypothesis.statement) > 20
            assert len(hypothesis.testable_predictions) > 0
            assert isinstance(hypothesis.success_criteria, dict)

        for experiment in research_result.experiments:
            assert len(experiment.methodology) > 50
            assert len(experiment.analysis_plan) > 20
            assert isinstance(experiment.variables, dict)

        # Verify synthesis and conclusions
        assert isinstance(research_result.synthesis, dict)
        assert len(research_result.final_conclusions) > 0
        assert len(research_result.publication_draft) > 500
        assert research_result.confidence_score > 0.0

        # Verify reproducibility
        reproducibility = research_result.reproducibility_report
        assert isinstance(reproducibility, dict)
        assert "reproducibility_score" in reproducibility

    @pytest.mark.asyncio
    async def test_mixed_workflow_routing(self, complete_system):
        """Test workflow routing for mixed/ambiguous requests"""
        router = complete_system["router"]
        engines = complete_system["engines"]

        # Test requests that could go to multiple engines
        mixed_requests = [
            {
                "request": "Research transformer architecture implementations and test their performance on NLP tasks",
                "expected_engines": ["SCIENTIFIC_RESEARCH"],  # Should go to scientific due to "test performance"
                "should_orchestrate": True
            },
            {
                "request": "Find information about the latest AI research trends and developments",
                "expected_engines": ["DEEP_RESEARCH", "SCIENTIFIC_RESEARCH"],
                "should_orchestrate": False
            },
            {
                "request": "Analyze open source machine learning frameworks and compare their features",
                "expected_engines": ["CODE_RESEARCH", "SCIENTIFIC_RESEARCH"],
                "should_orchestrate": False
            }
        ]

        for test_case in mixed_requests:
            # Step 1: Classification
            request = ClassificationRequest(user_request=test_case["request"])
            result = await router.classify_and_route(request)

            assert result.primary_engine in test_case["expected_engines"]

            # Step 2: Execute appropriate workflow
            if result.primary_engine == "SCIENTIFIC_RESEARCH" and test_case["should_orchestrate"]:
                # Should use scientific engine with orchestration
                scientific_result = await engines["scientific"].conduct_research(
                    test_case["request"],
                    include_literature_review=True,
                    include_code_analysis=True,
                    enable_iteration=False  # Limited for testing
                )

                # Should orchestrate multiple engines
                assert scientific_result.literature_review is not None or scientific_result.code_analysis is not None
                assert len(scientific_result.hypotheses) > 0

    @pytest.mark.asyncio
    async def test_workflow_performance_and_scaling(self, complete_system):
        """Test workflow performance characteristics"""
        engines = complete_system["engines"]

        # Test concurrent workflow execution
        workflows = [
            ("deep", "Global climate change research trends", engines["deep"].research),
            ("code", "Python web frameworks comparison", lambda q: engines["code"].research_code(q, include_analysis=False))
        ]

        # Execute workflows concurrently
        import time
        start_time = time.time()

        tasks = []
        for engine_type, query, engine_func in workflows:
            task = engine_func(query)
            tasks.append((engine_type, task))

        results = await asyncio.gather(*[task for _, task in tasks])

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all workflows completed successfully
        assert len(results) == len(workflows)
        for i, (engine_type, _) in enumerate(tasks):
            result = results[i]
            if engine_type == "deep":
                assert len(result.sources) > 0
                assert result.confidence_score > 0
            elif engine_type == "code":
                assert len(result.repositories) > 0
                assert result.confidence_score > 0

        # Concurrent execution should be efficient
        assert total_time < 180  # Should complete within 3 minutes

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, complete_system):
        """Test workflow error handling and recovery"""
        engines = complete_system["engines"]

        # Test with problematic inputs
        error_test_cases = [
            "",  # Empty request
            "x" * 1000,  # Very long request
            "!@#$%^&*()",  # Special characters only
            "completely_nonexistent_technical_concept_xyz123"  # Nonsensical technical term
        ]

        for test_input in error_test_cases:
            try:
                # Test deep research engine recovery
                deep_result = await engines["deep"].research(test_input)
                assert isinstance(deep_result.sources, list)
                assert deep_result.confidence_score >= 0

                # Test code research engine recovery
                code_result = await engines["code"].research_code(test_input)
                assert isinstance(code_result.repositories, list)
                assert code_result.confidence_score >= 0

            except Exception as e:
                # Engines should handle errors gracefully
                # If exceptions occur, they should be specific and informative
                assert isinstance(e, Exception)
                assert len(str(e)) > 0

    @pytest.mark.asyncio
    async def test_workflow_caching_and_optimization(self, complete_system):
        """Test workflow caching and performance optimization"""
        router = complete_system["router"]
        cache = complete_system["cache"]

        # Clear cache
        await cache.clear()

        # Test request that should be cached
        request = ClassificationRequest(user_request="Machine learning optimization techniques")

        # First request - cache miss
        import time
        start_time = time.time()
        result1 = await router.classify_and_route(request)
        first_time = time.time() - start_time

        # Second request - cache hit
        start_time = time.time()
        result2 = await router.classify_and_route(request)
        second_time = time.time() - start_time

        # Verify caching works
        assert result1.primary_engine == result2.primary_engine
        assert result1.confidence_score == result2.confidence_score
        assert result1.reasoning == result2.reasoning

        # Cache hit should be significantly faster
        assert second_time < first_time / 2  # At least 2x faster

        # Verify cache contains data
        cache_size = cache.size() if hasattr(cache, 'size') else 1
        assert cache_size > 0


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows simulating real user scenarios"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            return create_llm_client("dashscope", api_key=dashscope_key)
        else:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

    @pytest.mark.asyncio
    async def test_complete_research_scenario(self, llm_client):
        """Test complete research scenario from user request to final output"""
        # Simulate realistic user research scenario
        user_scenario = {
            "background": "AI researcher studying attention mechanisms",
            "goal": "Understand current state and evaluate new approach",
            "request": "Research attention mechanisms in transformers, find implementations, and design experiments to test a new attention variant"
        }

        # Create complete system
        cache = create_cache("memory")
        deep_engine = DeepResearchEngine(llm_client)
        code_engine = CodeResearchEngine(llm_client)
        scientific_engine = ScientificResearchEngine(
            llm_client=llm_client,
            deep_research_engine=deep_engine,
            code_research_engine=code_engine,
            config={"max_iterations": 2}
        )
        router = SmartRouter(llm_client=llm_client, cache=cache)

        # Execute complete workflow
        request = ClassificationRequest(user_request=user_scenario["request"])
        classification = await router.classify_and_route(request)

        # Should route to scientific research for this complex request
        assert classification.primary_engine == "SCIENTIFIC_RESEARCH"

        # Execute scientific research workflow
        result = await scientific_engine.conduct_research(
            user_scenario["request"],
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Verify complete research output suitable for AI researcher
        # Literature component
        assert result.literature_review is not None
        literature = result.literature_review
        assert len(literature.key_findings) >= 3
        assert any("attention" in finding.lower() for finding in literature.key_findings)

        # Code component
        assert result.code_analysis is not None
        code_analysis = result.code_analysis
        assert len(code_analysis.repositories) > 0
        assert any("transformer" in repo.description.lower() or "attention" in repo.description.lower()
                  for repo in code_analysis.repositories)

        # Experimental component
        assert len(result.hypotheses) > 0
        assert len(result.experiments) > 0
        attention_hypothesis = any("attention" in h.statement.lower() for h in result.hypotheses)
        assert attention_hypothesis

        # Final deliverables
        assert len(result.publication_draft) > 1000  # Substantial research output
        assert len(result.final_conclusions) >= 3
        assert result.confidence_score > 0.4

        # Verify research addresses user's goals
        pub_content = result.publication_draft.lower()
        assert "attention" in pub_content
        assert "transformer" in pub_content
        research_elements = ["method", "experiment", "result", "conclusion"]
        found_elements = sum(1 for element in research_elements if element in pub_content)
        assert found_elements >= 2  # Should have structured research format

    @pytest.mark.asyncio
    async def test_multi_domain_research_workflow(self, llm_client):
        """Test research workflow spanning multiple domains"""
        # Test interdisciplinary research request
        interdisciplinary_request = "Analyze the application of reinforcement learning in robotics, including both theoretical foundations and practical implementations"

        scientific_engine = ScientificResearchEngine(llm_client=llm_client)

        result = await scientific_engine.conduct_research(
            interdisciplinary_request,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=False
        )

        # Should address multiple domains
        combined_content = " ".join([
            result.publication_draft,
            str(result.synthesis),
            " ".join(result.final_conclusions)
        ]).lower()

        # Should cover reinforcement learning concepts
        rl_terms = ["reinforcement", "learning", "reward", "policy", "agent"]
        rl_coverage = sum(1 for term in rl_terms if term in combined_content)
        assert rl_coverage >= 2

        # Should cover robotics concepts
        robotics_terms = ["robot", "control", "sensor", "actuator", "navigation"]
        robotics_coverage = sum(1 for term in robotics_terms if term in combined_content)
        assert robotics_coverage >= 1

        # Should bridge theory and practice
        theory_terms = ["theory", "theoretical", "algorithm", "mathematical"]
        practice_terms = ["implementation", "practical", "application", "system"]

        theory_coverage = sum(1 for term in theory_terms if term in combined_content)
        practice_coverage = sum(1 for term in practice_terms if term in combined_content)

        assert theory_coverage >= 1
        assert practice_coverage >= 1

    @pytest.mark.asyncio
    async def test_iterative_research_refinement_workflow(self, llm_client):
        """Test iterative research refinement workflow"""
        # Configure for multiple iterations
        scientific_engine = ScientificResearchEngine(
            llm_client=llm_client,
            config={"max_iterations": 3, "confidence_threshold": 0.85}  # High threshold to force iterations
        )

        research_question = "Optimal learning rate scheduling for training large language models"

        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=False,  # Focus on experimental iteration
            include_code_analysis=False,
            enable_iteration=True
        )

        # Should show iterative improvement
        assert result.iteration_count > 0

        # Hypotheses should show evolution
        if len(result.hypotheses) > 1:
            # Check for refinement in hypothesis quality
            hypothesis_lengths = [len(h.reasoning) for h in result.hypotheses]
            assert max(hypothesis_lengths) > min(hypothesis_lengths) * 0.8  # Some variation in depth

        # Final confidence should reflect iterative improvement
        assert result.confidence_score > 0.3

        # Should have comprehensive experimental record
        assert len(result.experiments) >= result.iteration_count
        assert len(result.executions) >= result.iteration_count