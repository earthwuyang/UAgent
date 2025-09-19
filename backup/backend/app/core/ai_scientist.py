"""
AI-Scientist Integration - Automated research and hypothesis testing
Combines automated research workflows with experimental design and validation
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

from .meta_agent import Task, TaskType


class ResearchPhase(Enum):
    """Research phases in AI-Scientist workflow"""
    TOPIC_IDENTIFICATION = "topic_identification"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENTAL_DESIGN = "experimental_design"
    EXPERIMENT_EXECUTION = "experiment_execution"
    RESULT_ANALYSIS = "result_analysis"
    PAPER_WRITING = "paper_writing"
    PEER_REVIEW = "peer_review"


class ExperimentType(Enum):
    """Types of experiments AI-Scientist can conduct"""
    COMPUTATIONAL = "computational"
    SIMULATION = "simulation"
    DATA_ANALYSIS = "data_analysis"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    ABLATION_STUDY = "ablation_study"


@dataclass
class ResearchPaper:
    """Research paper representation"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    content: str = ""
    references: List[str] = field(default_factory=list)
    citations: int = 0
    publication_date: datetime = field(default_factory=datetime.now)
    venue: str = ""
    doi: str = ""


@dataclass
class Hypothesis:
    """Research hypothesis with testable components"""
    id: str
    statement: str
    variables: Dict[str, Any]
    predictions: List[str]
    testable_aspects: List[str]
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Experiment:
    """Experimental design and execution"""
    id: str
    name: str
    hypothesis_id: str
    experiment_type: ExperimentType
    methodology: str
    parameters: Dict[str, Any]
    variables: Dict[str, Any]
    controls: List[str]
    metrics: List[str]
    expected_results: Dict[str, Any] = field(default_factory=dict)
    actual_results: Dict[str, Any] = field(default_factory=dict)
    status: str = "designed"
    execution_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchProject:
    """Complete research project with all components"""
    id: str
    title: str
    description: str
    research_questions: List[str]
    current_phase: ResearchPhase = ResearchPhase.TOPIC_IDENTIFICATION
    literature: List[ResearchPaper] = field(default_factory=list)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    experiments: List[Experiment] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    generated_papers: List[ResearchPaper] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class LiteratureSearchEngine:
    """Advanced literature search and analysis"""

    def __init__(self):
        self.paper_database: Dict[str, ResearchPaper] = {}
        self.semantic_index: Dict[str, List[str]] = {}

    async def search_papers(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        limit: int = 50
    ) -> List[ResearchPaper]:
        """Search for relevant research papers"""
        # Simulate advanced paper search with semantic understanding
        relevant_papers = []

        # Simple keyword-based search simulation
        query_terms = query.lower().split()

        for paper in self.paper_database.values():
            relevance_score = self._calculate_relevance(paper, query_terms)
            if relevance_score > 0.3:
                relevant_papers.append((paper, relevance_score))

        # Sort by relevance and return top results
        relevant_papers.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, _ in relevant_papers[:limit]]

    def _calculate_relevance(self, paper: ResearchPaper, query_terms: List[str]) -> float:
        """Calculate relevance score between paper and query"""
        text_fields = [
            paper.title.lower(),
            paper.abstract.lower(),
            " ".join(paper.keywords).lower()
        ]

        total_score = 0.0
        for field in text_fields:
            field_score = sum(1 for term in query_terms if term in field)
            total_score += field_score / len(query_terms)

        return min(total_score / len(text_fields), 1.0)

    async def analyze_literature_trends(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze trends in literature"""
        if not papers:
            return {"error": "No papers to analyze"}

        # Analyze publication trends
        years = [paper.publication_date.year for paper in papers]
        year_counts = {}
        for year in years:
            year_counts[year] = year_counts.get(year, 0) + 1

        # Extract common keywords
        all_keywords = []
        for paper in papers:
            all_keywords.extend(paper.keywords)

        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        # Identify emerging topics
        emerging_topics = await self._identify_emerging_topics(papers)

        return {
            "total_papers": len(papers),
            "publication_trends": year_counts,
            "top_keywords": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "emerging_topics": emerging_topics,
            "average_citations": sum(paper.citations for paper in papers) / len(papers)
        }

    async def _identify_emerging_topics(self, papers: List[ResearchPaper]) -> List[str]:
        """Identify emerging research topics"""
        # Simple heuristic: topics appearing in recent papers
        recent_papers = [
            paper for paper in papers
            if (datetime.now() - paper.publication_date).days < 365
        ]

        recent_keywords = []
        for paper in recent_papers:
            recent_keywords.extend(paper.keywords)

        keyword_frequency = {}
        for keyword in recent_keywords:
            keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1

        # Return keywords that appear frequently in recent papers
        emerging = [
            keyword for keyword, count in keyword_frequency.items()
            if count >= 2
        ]

        return emerging[:5]


class HypothesisGenerator:
    """Automated hypothesis generation from literature"""

    def __init__(self):
        self.hypothesis_templates = self._initialize_templates()

    def _initialize_templates(self) -> List[Dict[str, str]]:
        """Initialize hypothesis generation templates"""
        return [
            {
                "type": "comparative",
                "template": "Algorithm A will outperform Algorithm B on metric M when condition C is met",
                "variables": ["algorithm_a", "algorithm_b", "metric", "condition"]
            },
            {
                "type": "causal",
                "template": "Factor F causes effect E in domain D",
                "variables": ["factor", "effect", "domain"]
            },
            {
                "type": "optimization",
                "template": "Parameter P can be optimized to value V to improve performance on task T",
                "variables": ["parameter", "value", "task"]
            },
            {
                "type": "correlation",
                "template": "Variable X is correlated with variable Y in context C",
                "variables": ["variable_x", "variable_y", "context"]
            }
        ]

    async def generate_hypotheses(
        self,
        research_context: Dict[str, Any],
        literature: List[ResearchPaper],
        num_hypotheses: int = 5
    ) -> List[Hypothesis]:
        """Generate testable hypotheses based on research context"""
        hypotheses = []

        for i in range(num_hypotheses):
            template = self.hypothesis_templates[i % len(self.hypothesis_templates)]

            hypothesis = await self._instantiate_hypothesis_template(
                template, research_context, literature
            )

            hypotheses.append(hypothesis)

        return hypotheses

    async def _instantiate_hypothesis_template(
        self,
        template: Dict[str, str],
        context: Dict[str, Any],
        literature: List[ResearchPaper]
    ) -> Hypothesis:
        """Instantiate a hypothesis template with specific values"""
        hypothesis_id = f"hyp_{datetime.now().timestamp()}_{hash(template['template']) % 1000}"

        # Extract relevant concepts from literature and context
        concepts = await self._extract_concepts(literature, context)

        # Fill template variables
        variables = {}
        for var in template["variables"]:
            if var in context:
                variables[var] = context[var]
            elif concepts:
                variables[var] = concepts[hash(var) % len(concepts)]
            else:
                variables[var] = f"placeholder_{var}"

        # Generate testable predictions
        predictions = await self._generate_predictions(template, variables)

        # Identify testable aspects
        testable_aspects = await self._identify_testable_aspects(template, variables)

        return Hypothesis(
            id=hypothesis_id,
            statement=self._fill_template(template["template"], variables),
            variables=variables,
            predictions=predictions,
            testable_aspects=testable_aspects,
            confidence=0.6
        )

    async def _extract_concepts(self, literature: List[ResearchPaper], context: Dict[str, Any]) -> List[str]:
        """Extract key concepts from literature and context"""
        concepts = []

        # Extract from context
        for key, value in context.items():
            if isinstance(value, str):
                concepts.extend(value.split())

        # Extract from literature keywords
        for paper in literature[:10]:  # Limit to recent papers
            concepts.extend(paper.keywords)

        # Remove duplicates and return unique concepts
        return list(set(concepts))

    def _fill_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Fill template with variable values"""
        result = template
        for var, value in variables.items():
            placeholder = var.upper()
            result = result.replace(placeholder, str(value))
        return result

    async def _generate_predictions(self, template: Dict[str, str], variables: Dict[str, Any]) -> List[str]:
        """Generate testable predictions from hypothesis"""
        predictions = []

        if template["type"] == "comparative":
            predictions.append(f"Performance difference will be measurable and significant")
            predictions.append(f"Results will be reproducible across multiple trials")

        elif template["type"] == "causal":
            predictions.append(f"Effect will be observed when factor is present")
            predictions.append(f"Effect will not occur when factor is absent")

        elif template["type"] == "optimization":
            predictions.append(f"Performance improvement will be quantifiable")
            predictions.append(f"Optimal value will be stable across different conditions")

        return predictions

    async def _identify_testable_aspects(self, template: Dict[str, str], variables: Dict[str, Any]) -> List[str]:
        """Identify aspects that can be empirically tested"""
        testable = []

        if template["type"] == "comparative":
            testable.extend([
                "Performance metrics comparison",
                "Statistical significance testing",
                "Cross-validation results"
            ])

        elif template["type"] == "causal":
            testable.extend([
                "Controlled experiment design",
                "Before-after analysis",
                "Confounding variable control"
            ])

        return testable


class ExperimentDesigner:
    """Automated experimental design and execution"""

    def __init__(self):
        self.experiment_templates = self._initialize_experiment_templates()

    def _initialize_experiment_templates(self) -> Dict[ExperimentType, Dict]:
        """Initialize experiment design templates"""
        return {
            ExperimentType.ALGORITHM_COMPARISON: {
                "methodology": "Compare multiple algorithms on same dataset with same metrics",
                "required_parameters": ["algorithms", "dataset", "metrics", "evaluation_method"],
                "controls": ["same_dataset", "same_preprocessing", "same_evaluation_criteria"],
                "typical_metrics": ["accuracy", "precision", "recall", "f1_score", "runtime"]
            },
            ExperimentType.PARAMETER_OPTIMIZATION: {
                "methodology": "Optimize hyperparameters using systematic search",
                "required_parameters": ["parameter_space", "optimization_method", "objective_function"],
                "controls": ["fixed_dataset", "fixed_model_architecture"],
                "typical_metrics": ["validation_performance", "convergence_time", "stability"]
            },
            ExperimentType.ABLATION_STUDY: {
                "methodology": "Remove components systematically to assess contribution",
                "required_parameters": ["base_model", "components_to_ablate", "evaluation_dataset"],
                "controls": ["same_training_procedure", "same_evaluation_metrics"],
                "typical_metrics": ["performance_drop", "component_importance", "interaction_effects"]
            }
        }

    async def design_experiments(self, hypothesis: Hypothesis) -> List[Experiment]:
        """Design experiments to test a hypothesis"""
        experiments = []

        # Determine appropriate experiment types
        experiment_types = await self._select_experiment_types(hypothesis)

        for exp_type in experiment_types:
            experiment = await self._design_single_experiment(hypothesis, exp_type)
            experiments.append(experiment)

        return experiments

    async def _select_experiment_types(self, hypothesis: Hypothesis) -> List[ExperimentType]:
        """Select appropriate experiment types for hypothesis"""
        hypothesis_text = hypothesis.statement.lower()

        selected_types = []

        if "outperform" in hypothesis_text or "better than" in hypothesis_text:
            selected_types.append(ExperimentType.ALGORITHM_COMPARISON)

        if "parameter" in hypothesis_text or "optimize" in hypothesis_text:
            selected_types.append(ExperimentType.PARAMETER_OPTIMIZATION)

        if "component" in hypothesis_text or "factor" in hypothesis_text:
            selected_types.append(ExperimentType.ABLATION_STUDY)

        # Default to computational experiment if no specific type identified
        if not selected_types:
            selected_types.append(ExperimentType.COMPUTATIONAL)

        return selected_types

    async def _design_single_experiment(
        self,
        hypothesis: Hypothesis,
        exp_type: ExperimentType
    ) -> Experiment:
        """Design a single experiment"""
        experiment_id = f"exp_{datetime.now().timestamp()}_{hash(hypothesis.id) % 1000}"

        template = self.experiment_templates.get(exp_type, {})

        # Extract parameters from hypothesis
        parameters = await self._extract_experiment_parameters(hypothesis, exp_type)

        return Experiment(
            id=experiment_id,
            name=f"Test {hypothesis.statement[:50]}...",
            hypothesis_id=hypothesis.id,
            experiment_type=exp_type,
            methodology=template.get("methodology", "Custom experimental methodology"),
            parameters=parameters,
            variables=hypothesis.variables,
            controls=template.get("controls", []),
            metrics=template.get("typical_metrics", ["accuracy", "performance"]),
            expected_results=await self._generate_expected_results(hypothesis, exp_type)
        )

    async def _extract_experiment_parameters(
        self,
        hypothesis: Hypothesis,
        exp_type: ExperimentType
    ) -> Dict[str, Any]:
        """Extract experiment parameters from hypothesis"""
        parameters = {
            "hypothesis_variables": hypothesis.variables,
            "experiment_type": exp_type.value
        }

        # Add type-specific parameters
        if exp_type == ExperimentType.ALGORITHM_COMPARISON:
            parameters.update({
                "num_trials": 10,
                "cross_validation_folds": 5,
                "significance_level": 0.05
            })

        elif exp_type == ExperimentType.PARAMETER_OPTIMIZATION:
            parameters.update({
                "search_method": "grid_search",
                "parameter_ranges": {"learning_rate": [0.001, 0.01, 0.1]},
                "optimization_iterations": 100
            })

        return parameters

    async def _generate_expected_results(
        self,
        hypothesis: Hypothesis,
        exp_type: ExperimentType
    ) -> Dict[str, Any]:
        """Generate expected results based on hypothesis"""
        expected = {
            "hypothesis_supported": hypothesis.confidence > 0.5,
            "confidence_level": hypothesis.confidence,
            "expected_trends": []
        }

        # Add type-specific expectations
        if exp_type == ExperimentType.ALGORITHM_COMPARISON:
            expected["expected_trends"].append("Performance difference between algorithms")

        elif exp_type == ExperimentType.PARAMETER_OPTIMIZATION:
            expected["expected_trends"].append("Performance improvement with optimal parameters")

        return expected


class AIScientist:
    """
    Main AI-Scientist orchestrator combining all research components
    Automates the complete research workflow from idea to publication
    """

    def __init__(self):
        self.literature_engine = LiteratureSearchEngine()
        self.hypothesis_generator = HypothesisGenerator()
        self.experiment_designer = ExperimentDesigner()
        self.active_projects: Dict[str, ResearchProject] = {}

    async def start_research_project(
        self,
        title: str,
        description: str,
        research_questions: List[str],
        initial_context: Dict[str, Any] = None
    ) -> str:
        """Start a new automated research project"""
        project_id = f"proj_{datetime.now().timestamp()}"

        project = ResearchProject(
            id=project_id,
            title=title,
            description=description,
            research_questions=research_questions
        )

        self.active_projects[project_id] = project

        # Initialize with seed literature if context provided
        if initial_context and "seed_papers" in initial_context:
            project.literature.extend(initial_context["seed_papers"])

        return project_id

    async def execute_research_phase(
        self,
        project_id: str,
        phase: ResearchPhase = None
    ) -> Dict[str, Any]:
        """Execute a specific research phase or continue from current phase"""
        project = self.active_projects.get(project_id)
        if not project:
            return {"error": "Project not found"}

        target_phase = phase or project.current_phase

        if target_phase == ResearchPhase.LITERATURE_REVIEW:
            return await self._execute_literature_review(project)

        elif target_phase == ResearchPhase.HYPOTHESIS_GENERATION:
            return await self._execute_hypothesis_generation(project)

        elif target_phase == ResearchPhase.EXPERIMENTAL_DESIGN:
            return await self._execute_experimental_design(project)

        elif target_phase == ResearchPhase.EXPERIMENT_EXECUTION:
            return await self._execute_experiments(project)

        elif target_phase == ResearchPhase.RESULT_ANALYSIS:
            return await self._execute_result_analysis(project)

        else:
            return {"error": f"Phase {target_phase} not implemented"}

    async def _execute_literature_review(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute automated literature review"""
        # Search for relevant papers
        all_papers = []
        for question in project.research_questions:
            papers = await self.literature_engine.search_papers(question, limit=20)
            all_papers.extend(papers)

        # Remove duplicates
        unique_papers = list({paper.id: paper for paper in all_papers}.values())
        project.literature.extend(unique_papers)

        # Analyze literature trends
        trends = await self.literature_engine.analyze_literature_trends(project.literature)

        # Advance to next phase
        project.current_phase = ResearchPhase.HYPOTHESIS_GENERATION
        project.updated_at = datetime.now()

        return {
            "phase": "literature_review_completed",
            "papers_found": len(unique_papers),
            "literature_trends": trends,
            "next_phase": project.current_phase.value
        }

    async def _execute_hypothesis_generation(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute automated hypothesis generation"""
        research_context = {
            "research_questions": project.research_questions,
            "domain": project.title,
            "literature_insights": "extracted_from_papers"
        }

        hypotheses = await self.hypothesis_generator.generate_hypotheses(
            research_context, project.literature, num_hypotheses=3
        )

        project.hypotheses.extend(hypotheses)
        project.current_phase = ResearchPhase.EXPERIMENTAL_DESIGN
        project.updated_at = datetime.now()

        return {
            "phase": "hypothesis_generation_completed",
            "hypotheses_generated": len(hypotheses),
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "confidence": h.confidence
                }
                for h in hypotheses
            ],
            "next_phase": project.current_phase.value
        }

    async def _execute_experimental_design(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute automated experimental design"""
        all_experiments = []

        for hypothesis in project.hypotheses:
            experiments = await self.experiment_designer.design_experiments(hypothesis)
            all_experiments.extend(experiments)

        project.experiments.extend(all_experiments)
        project.current_phase = ResearchPhase.EXPERIMENT_EXECUTION
        project.updated_at = datetime.now()

        return {
            "phase": "experimental_design_completed",
            "experiments_designed": len(all_experiments),
            "experiments": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.experiment_type.value,
                    "hypothesis_id": e.hypothesis_id
                }
                for e in all_experiments
            ],
            "next_phase": project.current_phase.value
        }

    async def _execute_experiments(self, project: ResearchProject) -> Dict[str, Any]:
        """Execute designed experiments"""
        execution_results = []

        for experiment in project.experiments:
            if experiment.status == "designed":
                result = await self._run_single_experiment(experiment)
                execution_results.append(result)
                experiment.status = "completed"
                experiment.actual_results = result

        project.current_phase = ResearchPhase.RESULT_ANALYSIS
        project.updated_at = datetime.now()

        return {
            "phase": "experiment_execution_completed",
            "experiments_executed": len(execution_results),
            "results_summary": execution_results,
            "next_phase": project.current_phase.value
        }

    async def _run_single_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Run a single experiment (simulated)"""
        start_time = datetime.now()

        # Simulate experiment execution
        await asyncio.sleep(0.2)

        # Generate simulated results based on experiment type
        if experiment.experiment_type == ExperimentType.ALGORITHM_COMPARISON:
            results = {
                "algorithm_a_performance": 0.85 + np.random.normal(0, 0.05),
                "algorithm_b_performance": 0.82 + np.random.normal(0, 0.05),
                "statistical_significance": "p < 0.05",
                "effect_size": 0.3
            }

        elif experiment.experiment_type == ExperimentType.PARAMETER_OPTIMIZATION:
            results = {
                "optimal_parameters": {"learning_rate": 0.01, "batch_size": 32},
                "best_performance": 0.89,
                "improvement_over_baseline": 0.07,
                "convergence_epochs": 45
            }

        else:
            results = {
                "metric_value": 0.78 + np.random.normal(0, 0.1),
                "baseline_comparison": "significant_improvement",
                "confidence_interval": [0.72, 0.84]
            }

        execution_time = (datetime.now() - start_time).total_seconds()
        experiment.execution_time = execution_time

        return {
            "experiment_id": experiment.id,
            "execution_time": execution_time,
            "results": results,
            "hypothesis_supported": results.get("metric_value", 0.8) > 0.75
        }

    async def _execute_result_analysis(self, project: ResearchProject) -> Dict[str, Any]:
        """Analyze experimental results and generate insights"""
        findings = []

        # Analyze results for each hypothesis
        for hypothesis in project.hypotheses:
            hypothesis_experiments = [
                exp for exp in project.experiments
                if exp.hypothesis_id == hypothesis.id
            ]

            if hypothesis_experiments:
                analysis = await self._analyze_hypothesis_results(
                    hypothesis, hypothesis_experiments
                )
                findings.append(analysis)

        project.findings.extend(findings)
        project.current_phase = ResearchPhase.PAPER_WRITING
        project.updated_at = datetime.now()

        return {
            "phase": "result_analysis_completed",
            "findings": findings,
            "supported_hypotheses": len([f for f in findings if f.get("supported", False)]),
            "next_phase": project.current_phase.value
        }

    async def _analyze_hypothesis_results(
        self,
        hypothesis: Hypothesis,
        experiments: List[Experiment]
    ) -> Dict[str, Any]:
        """Analyze results for a specific hypothesis"""
        support_count = 0
        total_experiments = len(experiments)

        for experiment in experiments:
            if experiment.actual_results.get("hypothesis_supported", False):
                support_count += 1

        support_ratio = support_count / total_experiments if total_experiments > 0 else 0

        return {
            "hypothesis_id": hypothesis.id,
            "hypothesis_statement": hypothesis.statement,
            "experiments_count": total_experiments,
            "support_ratio": support_ratio,
            "supported": support_ratio > 0.5,
            "confidence": support_ratio,
            "key_findings": f"Hypothesis tested through {total_experiments} experiments",
            "recommendations": await self._generate_recommendations(hypothesis, support_ratio)
        }

    async def _generate_recommendations(self, hypothesis: Hypothesis, support_ratio: float) -> List[str]:
        """Generate recommendations based on hypothesis testing results"""
        recommendations = []

        if support_ratio > 0.7:
            recommendations.append("Strong evidence supports this hypothesis - consider for publication")
            recommendations.append("Design follow-up studies to explore broader implications")

        elif support_ratio > 0.4:
            recommendations.append("Mixed results - refine hypothesis and experimental design")
            recommendations.append("Investigate confounding factors or experimental conditions")

        else:
            recommendations.append("Hypothesis not supported - consider alternative explanations")
            recommendations.append("Review experimental methodology and literature basis")

        return recommendations

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive status of research project"""
        project = self.active_projects.get(project_id)
        if not project:
            return {"error": "Project not found"}

        return {
            "project_id": project_id,
            "title": project.title,
            "current_phase": project.current_phase.value,
            "progress": {
                "literature_papers": len(project.literature),
                "hypotheses_generated": len(project.hypotheses),
                "experiments_designed": len(project.experiments),
                "experiments_completed": len([e for e in project.experiments if e.status == "completed"]),
                "findings": len(project.findings)
            },
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "estimated_completion": await self._estimate_completion_time(project)
        }

    async def _estimate_completion_time(self, project: ResearchProject) -> str:
        """Estimate time to complete remaining phases"""
        remaining_phases = []

        phase_order = [
            ResearchPhase.TOPIC_IDENTIFICATION,
            ResearchPhase.LITERATURE_REVIEW,
            ResearchPhase.HYPOTHESIS_GENERATION,
            ResearchPhase.EXPERIMENTAL_DESIGN,
            ResearchPhase.EXPERIMENT_EXECUTION,
            ResearchPhase.RESULT_ANALYSIS,
            ResearchPhase.PAPER_WRITING
        ]

        current_index = phase_order.index(project.current_phase)
        remaining_phases = phase_order[current_index + 1:]

        # Estimate time per phase (in hours)
        phase_times = {
            ResearchPhase.LITERATURE_REVIEW: 2,
            ResearchPhase.HYPOTHESIS_GENERATION: 1,
            ResearchPhase.EXPERIMENTAL_DESIGN: 1,
            ResearchPhase.EXPERIMENT_EXECUTION: 3,
            ResearchPhase.RESULT_ANALYSIS: 2,
            ResearchPhase.PAPER_WRITING: 4
        }

        total_hours = sum(phase_times.get(phase, 1) for phase in remaining_phases)
        return f"{total_hours} hours estimated"