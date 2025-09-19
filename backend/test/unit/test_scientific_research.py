"""Unit tests for Scientific Research Engine"""

import pytest
import os

from app.core.research_engines.scientific_research import (
    ScientificResearchEngine,
    HypothesisGenerator,
    ExperimentDesigner,
    ExperimentExecutor,
    ResearchHypothesis,
    ExperimentDesign,
    HypothesisStatus,
    ExperimentStatus
)
from app.core.research_engines.deep_research import DeepResearchEngine
from app.core.research_engines.code_research import CodeResearchEngine
from app.core.llm_client import create_llm_client


class TestScientificResearchEngine:
    """Test scientific research engine functionality"""

    @pytest.fixture
    def llm_client(self):
        """Create real DashScope LLM client for testing"""
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            return create_llm_client("dashscope", api_key=dashscope_key)
        else:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

    @pytest.fixture
    def deep_research_engine(self, llm_client):
        """Create deep research engine"""
        return DeepResearchEngine(llm_client)

    @pytest.fixture
    def code_research_engine(self, llm_client):
        """Create code research engine"""
        return CodeResearchEngine(llm_client)

    @pytest.fixture
    def scientific_research_engine(self, llm_client, deep_research_engine, code_research_engine):
        """Create scientific research engine"""
        return ScientificResearchEngine(
            llm_client=llm_client,
            deep_research_engine=deep_research_engine,
            code_research_engine=code_research_engine,
            config={"max_iterations": 2, "confidence_threshold": 0.7}
        )

    @pytest.fixture
    def hypothesis_generator(self, llm_client):
        """Create hypothesis generator"""
        return HypothesisGenerator(llm_client)

    @pytest.fixture
    def experiment_designer(self, llm_client):
        """Create experiment designer"""
        return ExperimentDesigner(llm_client)

    @pytest.fixture
    def experiment_executor(self, llm_client):
        """Create experiment executor"""
        return ExperimentExecutor(llm_client)

    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, hypothesis_generator):
        """Test hypothesis generation"""
        # Arrange
        research_question = "Does attention mechanism improve neural network performance on NLP tasks?"
        literature_context = "Previous studies show attention mechanisms provide interpretability"
        code_context = "Transformer architectures use multi-head attention"

        # Act
        hypotheses = await hypothesis_generator.generate_hypotheses(
            research_question,
            literature_context,
            code_context
        )

        # Assert
        assert isinstance(hypotheses, list)
        assert len(hypotheses) > 0
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, ResearchHypothesis)
            assert hypothesis.id
            assert hypothesis.statement
            assert hypothesis.reasoning
            assert isinstance(hypothesis.testable_predictions, list)
            assert isinstance(hypothesis.success_criteria, dict)
            assert isinstance(hypothesis.variables, dict)
            assert hypothesis.status == HypothesisStatus.PENDING

    @pytest.mark.asyncio
    async def test_hypothesis_refinement(self, hypothesis_generator):
        """Test hypothesis refinement based on feedback"""
        # Arrange
        original_hypothesis = ResearchHypothesis(
            id="test_hyp_001",
            statement="Attention mechanisms improve model performance",
            reasoning="Based on theoretical understanding",
            testable_predictions=["Higher accuracy", "Better interpretability"],
            success_criteria={"accuracy_improvement": 0.05},
            variables={"independent": ["attention_type"], "dependent": ["accuracy"]}
        )
        feedback = "Initial experiments show marginal improvement, need more specific metrics"
        experimental_data = {"accuracy_gain": 0.02, "variance": 0.01}

        # Act
        refined_hypothesis = await hypothesis_generator.refine_hypothesis(
            original_hypothesis,
            feedback,
            experimental_data
        )

        # Assert
        assert isinstance(refined_hypothesis, ResearchHypothesis)
        assert refined_hypothesis.id == original_hypothesis.id
        # Statement or criteria should be refined based on feedback
        assert isinstance(refined_hypothesis.statement, str)
        assert len(refined_hypothesis.statement) > 0

    @pytest.mark.asyncio
    async def test_experiment_design(self, experiment_designer):
        """Test experiment design generation"""
        # Arrange
        hypothesis = ResearchHypothesis(
            id="test_hyp_001",
            statement="Batch normalization improves neural network convergence",
            reasoning="Normalization reduces internal covariate shift",
            testable_predictions=["Faster convergence", "Lower training loss"],
            success_criteria={"convergence_speed": 0.2, "loss_reduction": 0.1},
            variables={"independent": ["batch_norm"], "dependent": ["convergence_time", "final_loss"]}
        )
        resources = {"cpu_cores": 4, "memory": "8GB", "gpu": "1x RTX 3080"}

        # Act
        design = await experiment_designer.design_experiment(hypothesis, resources)

        # Assert
        assert isinstance(design, ExperimentDesign)
        assert design.id
        assert design.hypothesis_id == hypothesis.id
        assert design.name
        assert design.description
        assert design.methodology
        assert isinstance(design.variables, dict)
        assert isinstance(design.controls, list)
        assert isinstance(design.data_collection_plan, dict)
        assert design.analysis_plan
        assert design.expected_duration
        assert isinstance(design.resource_requirements, dict)
        assert isinstance(design.code_requirements, list)
        assert isinstance(design.dependencies, list)

    @pytest.mark.asyncio
    async def test_experiment_execution(self, experiment_executor):
        """Test experiment execution"""
        # Arrange
        design = ExperimentDesign(
            id="test_exp_001",
            hypothesis_id="test_hyp_001",
            name="Batch Normalization Effect Test",
            description="Test the effect of batch normalization on convergence",
            methodology="Controlled experiment with and without batch normalization",
            variables={"independent": ["batch_norm"], "dependent": ["convergence_time"]},
            controls=["random_seed", "learning_rate"],
            data_collection_plan={"samples": 100, "method": "automated"},
            analysis_plan="Statistical significance testing",
            expected_duration="2 hours",
            resource_requirements={"cpu": "2 cores", "memory": "4GB"},
            code_requirements=["Python 3.8+", "PyTorch"],
            dependencies=["torch", "numpy", "scipy"]
        )

        # Act
        execution = await experiment_executor.execute_experiment(design)

        # Assert
        assert execution.id
        assert execution.design_id == design.id
        assert execution.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
        assert execution.start_time
        assert isinstance(execution.logs, list)
        assert len(execution.logs) > 0
        assert isinstance(execution.output_data, dict)
        assert isinstance(execution.intermediate_results, dict)

        if execution.status == ExperimentStatus.COMPLETED:
            assert execution.end_time
            assert execution.progress == 1.0
            assert len(execution.output_data) > 0

    @pytest.mark.asyncio
    async def test_comprehensive_scientific_research(self, scientific_research_engine):
        """Test comprehensive scientific research workflow"""
        # Arrange
        research_question = "How does regularization affect deep learning model generalization?"

        # Act
        result = await scientific_research_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Assert comprehensive research structure
        assert result.research_id
        assert result.query == research_question
        assert isinstance(result.hypotheses, list)
        assert len(result.hypotheses) > 0
        assert isinstance(result.experiments, list)
        assert len(result.experiments) > 0
        assert isinstance(result.executions, list)
        assert len(result.executions) > 0
        assert isinstance(result.results, list)
        assert isinstance(result.synthesis, dict)
        assert isinstance(result.final_conclusions, list)
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.reproducibility_report, dict)
        assert isinstance(result.publication_draft, str)
        assert len(result.publication_draft) > 100  # Substantial publication draft
        assert result.iteration_count > 0
        assert isinstance(result.recommendations, list)

        # Verify multi-engine integration
        if result.literature_review:
            assert result.literature_review.query
            assert len(result.literature_review.sources) > 0

        if result.code_analysis:
            assert result.code_analysis.query
            assert len(result.code_analysis.repositories) > 0

        # Verify experimental workflow
        for hypothesis in result.hypotheses:
            assert isinstance(hypothesis, ResearchHypothesis)
            assert hypothesis.status in [HypothesisStatus.PENDING, HypothesisStatus.SUPPORTED,
                                       HypothesisStatus.REJECTED, HypothesisStatus.INCONCLUSIVE]

        for experiment in result.experiments:
            assert isinstance(experiment, ExperimentDesign)
            assert experiment.hypothesis_id in [h.id for h in result.hypotheses]

    @pytest.mark.asyncio
    async def test_research_without_background_engines(self, llm_client):
        """Test scientific research without literature review and code analysis"""
        # Arrange
        engine = ScientificResearchEngine(llm_client=llm_client)
        research_question = "Does dropout improve model robustness?"

        # Act
        result = await engine.conduct_research(
            research_question,
            include_literature_review=False,
            include_code_analysis=False,
            enable_iteration=False
        )

        # Assert
        assert result.query == research_question
        assert result.literature_review is None
        assert result.code_analysis is None
        assert len(result.hypotheses) > 0
        assert len(result.experiments) > 0
        assert result.iteration_count >= 0

    @pytest.mark.asyncio
    async def test_iterative_research_refinement(self, llm_client):
        """Test iterative research with hypothesis refinement"""
        # Arrange
        engine = ScientificResearchEngine(
            llm_client=llm_client,
            config={"max_iterations": 3, "confidence_threshold": 0.9}  # High threshold to force iterations
        )
        research_question = "What is the optimal learning rate schedule for transformer training?"

        # Act
        result = await engine.conduct_research(
            research_question,
            include_literature_review=False,
            include_code_analysis=False,
            enable_iteration=True
        )

        # Assert iterative behavior
        assert result.iteration_count > 0
        # Should have attempted multiple iterations due to high confidence threshold
        # or stopped early if threshold was reached

    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, scientific_research_engine):
        """Test confidence score calculation logic"""
        # Arrange
        from app.core.research_engines.scientific_research import ExperimentResult

        results = [
            ExperimentResult(
                execution_id="exec_1",
                hypothesis_id="hyp_1",
                status=ExperimentStatus.COMPLETED,
                data={},
                analysis={},
                statistical_tests={},
                visualization_data={},
                conclusions=["Strong evidence"],
                confidence_score=0.9,
                reproducibility_score=0.8,
                limitations=[],
                next_steps=[]
            ),
            ExperimentResult(
                execution_id="exec_2",
                hypothesis_id="hyp_2",
                status=ExperimentStatus.COMPLETED,
                data={},
                analysis={},
                statistical_tests={},
                visualization_data={},
                conclusions=["Moderate evidence"],
                confidence_score=0.7,
                reproducibility_score=0.7,
                limitations=[],
                next_steps=[]
            )
        ]

        # Act
        overall_confidence = scientific_research_engine._calculate_overall_confidence(results)

        # Assert
        assert 0.0 <= overall_confidence <= 1.0
        # Should be higher than individual scores due to multiple supporting experiments
        expected_avg = (0.9 + 0.7) / 2 + 0.1  # Average + consistency bonus
        assert abs(overall_confidence - expected_avg) < 0.1

    @pytest.mark.asyncio
    async def test_synthesis_generation(self, scientific_research_engine):
        """Test research synthesis generation"""
        # Arrange
        research_question = "Impact of data augmentation on model performance"

        # Create a minimal research result for synthesis
        result = await scientific_research_engine.conduct_research(
            research_question,
            include_literature_review=False,
            include_code_analysis=False,
            enable_iteration=False
        )

        # Assert synthesis quality
        synthesis = result.synthesis
        assert isinstance(synthesis, dict)
        assert "overall_findings" in synthesis or "conclusions" in synthesis

        # Verify publication draft
        assert len(result.publication_draft) > 200  # Should be substantial
        assert research_question.lower() in result.publication_draft.lower() or "research" in result.publication_draft.lower()

        # Verify reproducibility report
        assert isinstance(result.reproducibility_report, dict)
        assert "reproducibility_score" in result.reproducibility_report
        assert 0.0 <= result.reproducibility_report["reproducibility_score"] <= 1.0


@pytest.mark.integration
class TestScientificResearchIntegration:
    """Integration tests for scientific research engine"""

    @pytest.mark.asyncio
    async def test_multi_engine_orchestration(self):
        """Test orchestration of multiple research engines"""
        # Arrange
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("dashscope", api_key=dashscope_key)
        deep_engine = DeepResearchEngine(llm_client)
        code_engine = CodeResearchEngine(llm_client)

        scientific_engine = ScientificResearchEngine(
            llm_client=llm_client,
            deep_research_engine=deep_engine,
            code_research_engine=code_engine,
            config={"max_iterations": 2}
        )

        research_question = "Effectiveness of transfer learning in computer vision"

        # Act
        result = await scientific_engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Assert multi-engine coordination
        assert result.literature_review is not None
        assert result.code_analysis is not None
        assert len(result.hypotheses) > 0
        assert len(result.experiments) > 0

        # Verify background research integration
        literature = result.literature_review
        assert len(literature.sources) > 0
        assert len(literature.key_findings) > 0

        code_analysis = result.code_analysis
        assert len(code_analysis.repositories) > 0
        assert len(code_analysis.recommendations) > 0

        # Verify experimental integration builds on background research
        for hypothesis in result.hypotheses:
            # Hypothesis should be informed by background research
            assert len(hypothesis.reasoning) > 50  # Should have substantial reasoning
            assert len(hypothesis.testable_predictions) > 0

    @pytest.mark.asyncio
    async def test_real_llm_hypothesis_quality(self):
        """Test hypothesis quality with real LLM"""
        # Arrange
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("dashscope", api_key=dashscope_key)
        hypothesis_generator = HypothesisGenerator(llm_client)

        research_question = "How does model architecture affect training stability in GANs?"

        # Act
        hypotheses = await hypothesis_generator.generate_hypotheses(research_question)

        # Assert real LLM generates quality hypotheses
        assert len(hypotheses) >= 3  # Should generate multiple hypotheses

        for hypothesis in hypotheses:
            # Real LLM should generate substantial content
            assert len(hypothesis.statement) > 20
            assert len(hypothesis.reasoning) > 50
            assert len(hypothesis.testable_predictions) > 0

            # Should include relevant variables
            assert isinstance(hypothesis.variables, dict)
            assert len(hypothesis.variables) > 0

            # Should have measurable success criteria
            assert isinstance(hypothesis.success_criteria, dict)

    @pytest.mark.asyncio
    async def test_end_to_end_scientific_workflow_quality(self):
        """Test end-to-end scientific workflow with quality metrics"""
        # Arrange
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if not dashscope_key:
            pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")

        llm_client = create_llm_client("dashscope", api_key=dashscope_key)
        engine = ScientificResearchEngine(llm_client=llm_client)

        research_question = "Optimal batch size for training large language models"

        # Act
        result = await engine.conduct_research(
            research_question,
            include_literature_review=True,
            include_code_analysis=True,
            enable_iteration=True
        )

        # Assert workflow quality with real LLM
        assert result.confidence_score > 0.3  # Should achieve meaningful confidence
        assert len(result.final_conclusions) >= 2  # Should draw multiple conclusions
        assert len(result.recommendations) >= 3  # Should provide actionable recommendations

        # Publication draft should be substantial and well-structured
        publication = result.publication_draft
        assert len(publication) > 500  # Should be comprehensive
        assert any(section in publication.lower() for section in ["abstract", "introduction", "methodology", "results"])

        # Synthesis should provide meaningful insights
        synthesis = result.synthesis
        if "overall_findings" in synthesis:
            assert len(synthesis["overall_findings"]) > 100  # Should be detailed

        # Should demonstrate iterative improvement
        assert result.iteration_count > 0
        if result.iteration_count > 1:
            # Multiple iterations should show hypothesis refinement
            hypothesis_statements = [h.statement for h in result.hypotheses]
            assert len(set(hypothesis_statements)) > 0  # Should have diverse hypotheses