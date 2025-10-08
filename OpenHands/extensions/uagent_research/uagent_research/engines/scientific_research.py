"""
Scientific Research Engine - Adapted for OpenHands

This is a production implementation that integrates UAgent's scientific research
capabilities with OpenHands' infrastructure.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# OpenHands imports
from openhands.llm.llm import LLM
from openhands.runtime.runtime import Runtime
from openhands.events.stream import EventStream
from ...orchestrator.event_bus import EventBus, get_event_bus
from ...uagent_research.models.events import StepEvent, CompleteEvent, ErrorEvent
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation

# Extension imports
from ..models import Experiment, ExperimentStatus, ExperimentType
from ..models.base import get_session

logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    """Hypothesis validation status"""
    PENDING = "pending"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ResearchHypothesis:
    """Research hypothesis with validation criteria"""
    id: str
    statement: str
    reasoning: str
    testable_predictions: List[str]
    success_criteria: Dict[str, Any]
    status: HypothesisStatus = HypothesisStatus.PENDING
    evidence: List[str] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


@dataclass
class ExperimentPlan:
    """Detailed experiment execution plan"""
    id: str
    hypothesis_id: str
    title: str
    description: str
    methodology: str
    expected_outcomes: List[str]
    code_to_execute: Optional[str] = None
    data_requirements: List[str] = None

    def __post_init__(self):
        if self.data_requirements is None:
            self.data_requirements = []


class ScientificResearchEngine:
    """
    Scientific Research Engine for OpenHands.

    Conducts scientific experiments with:
    - Hypothesis generation and testing
    - Automated experiment execution
    - Result validation and analysis
    - Iterative refinement
    """

    def __init__(self, llm: LLM, config: Optional[Dict[str, Any]] = None, event_bus: Optional[EventBus] = None):
        """
        Initialize research engine.

        Args:
            llm: OpenHands LLM instance
            config: Optional configuration dictionary
        """
        self.llm = llm
        self.event_bus = event_bus or get_event_bus()
        self.config = config or {}

        # Configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.validation_strict = self.config.get('validation_strict', True)
        self.simulation_detection = self.config.get('simulation_detection', True)
        self.max_iterations = self.config.get('max_iterations', 10)


        logger.info(f"ScientificResearchEngine initialized with config: {self.config}")

    async def run_experiment(
        self,
        goal: str,
        runtime: Runtime,
        event_stream: EventStream,
        session_id: str,
        experiment_id: Optional[str] = None,
        branch_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete scientific research experiment.

        Args:
            goal: Research objective/question
            runtime: OpenHands runtime for code execution
            event_stream: Event stream for progress updates
            session_id: Current session ID
            experiment_id: Optional experiment ID (creates new if not provided)

        Returns:
            Dictionary with experiment results
        """
        if not experiment_id:
            experiment_id = f"exp_{session_id}_{uuid.uuid4().hex[:8]}"

        bid = branch_id or experiment_id

        logger.info(f"Starting scientific research experiment: {experiment_id}")
        logger.info(f"Goal: {goal}")

        await self.event_bus.publish(
            StepEvent(
                branch_id=bid,
                action="experiment_started",
                reasoning=f"Starting: {goal}",
            )
        )

        try:
            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Generating hypotheses",
                    reasoning="Enumerating candidate hypotheses",
                )
            )
            hypotheses = await self._generate_hypotheses(goal)
            logger.info(f"Generated {len(hypotheses)} hypotheses")

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Designing experiments",
                    reasoning="Translating hypotheses into experiment plans",
                )
            )
            experiment_plans = await self._design_experiments(hypotheses)
            logger.info(f"Designed {len(experiment_plans)} experiments")

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Executing experiments",
                    reasoning="Running experiments via runtime",
                )
            )
            results = await self._execute_experiments(
                experiment_plans,
                runtime,
                event_stream,
                experiment_id,
            )

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Analyzing results",
                    reasoning="Interpreting experiment outputs",
                )
            )
            analysis = await self._analyze_results(hypotheses, results)

            await self.event_bus.publish(
                StepEvent(
                    branch_id=bid,
                    action="Validating findings",
                    reasoning="Checking for simulated or low-confidence results",
                )
            )
            validated = await self._validate_findings(analysis)

            final_results = {
                "experiment_id": experiment_id,
                "goal": goal,
                "hypotheses": [self._hypothesis_to_dict(h) for h in hypotheses],
                "experiment_plans": [self._plan_to_dict(p) for p in experiment_plans],
                "results": results,
                "analysis": analysis,
                "validated": validated,
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
            }

            summary_text = (
                analysis.get("overall_findings", "Experiment completed")
                if isinstance(analysis, dict)
                else "Experiment completed"
            )

            await self.event_bus.publish(
                CompleteEvent(
                    branch_id=bid,
                    summary=summary_text,
                    artifacts=[],
                )
            )

            logger.info(f"Experiment {experiment_id} completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)

            await self.event_bus.publish(
                ErrorEvent(
                    branch_id=bid,
                    message=str(e),
                )
            )
            raise

    async def _generate_hypotheses(self, goal: str) -> List[ResearchHypothesis]:
        """Generate testable hypotheses from research goal"""

        prompt = f"""Given this research goal:
{goal}

Generate 2-3 testable hypotheses that could answer this research question.

For each hypothesis, provide:
1. A clear, testable statement
2. The reasoning behind this hypothesis
3. Specific testable predictions
4. Success criteria (how to validate)

Respond with ONLY valid JSON in this format:
{{
    "hypotheses": [
        {{
            "id": "hyp1",
            "statement": "clear testable statement",
            "reasoning": "why this is worth testing",
            "testable_predictions": ["prediction 1", "prediction 2"],
            "success_criteria": {{
                "metric": "what to measure",
                "threshold": "acceptable value"
            }}
        }}
    ]
}}
"""

        response = await self.llm.completion(
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse response
        content = response.choices[0].message.content

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(content)

        # Create hypothesis objects
        hypotheses = []
        for h_data in data.get('hypotheses', []):
            hypothesis = ResearchHypothesis(
                id=h_data.get('id', f"hyp_{len(hypotheses)+1}"),
                statement=h_data['statement'],
                reasoning=h_data['reasoning'],
                testable_predictions=h_data['testable_predictions'],
                success_criteria=h_data['success_criteria'],
            )
            hypotheses.append(hypothesis)

        return hypotheses

    async def _design_experiments(
        self,
        hypotheses: List[ResearchHypothesis]
    ) -> List[ExperimentPlan]:
        """Design experiments to test hypotheses"""

        plans = []

        for hypothesis in hypotheses:
            prompt = f"""Design an experiment to test this hypothesis:

Hypothesis: {hypothesis.statement}
Reasoning: {hypothesis.reasoning}
Predictions: {', '.join(hypothesis.testable_predictions)}
Success Criteria: {json.dumps(hypothesis.success_criteria)}

Design a concrete, executable experiment including:
1. Detailed methodology
2. Code to execute (if applicable)
3. Expected outcomes
4. Data requirements

Respond with ONLY valid JSON:
{{
    "title": "experiment title",
    "description": "what this experiment does",
    "methodology": "step-by-step approach",
    "code_to_execute": "# Python code to run\\nprint('test')",
    "expected_outcomes": ["outcome 1", "outcome 2"],
    "data_requirements": ["requirement 1"]
}}
"""

            response = await self.llm.completion(
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            plan = ExperimentPlan(
                id=f"plan_{hypothesis.id}",
                hypothesis_id=hypothesis.id,
                title=data['title'],
                description=data['description'],
                methodology=data['methodology'],
                code_to_execute=data.get('code_to_execute'),
                expected_outcomes=data['expected_outcomes'],
                data_requirements=data.get('data_requirements', []),
            )
            plans.append(plan)

        return plans

    async def _execute_experiments(
        self,
        plans: List[ExperimentPlan],
        runtime: Runtime,
        event_stream: EventStream,
        experiment_id: str
    ) -> List[Dict[str, Any]]:
        """Execute experiment plans using OpenHands runtime"""

        results = []

        for i, plan in enumerate(plans):
            logger.info(f"Executing experiment plan: {plan.title}")

            try:
                if plan.code_to_execute:
                    # Execute code using OpenHands runtime
                    action = CmdRunAction(
                        command=f"python3 -c '{plan.code_to_execute.replace(chr(39), chr(34))}'",
                        thought=f"Executing experiment: {plan.title}"
                    )

                    observation = await runtime.run_action(action)

                    result = {
                        'plan_id': plan.id,
                        'title': plan.title,
                        'success': not isinstance(observation, CmdOutputObservation) or observation.exit_code == 0,
                        'output': observation.content if hasattr(observation, 'content') else str(observation),
                        'exit_code': observation.exit_code if isinstance(observation, CmdOutputObservation) else 0,
                    }
                else:
                    # Theoretical experiment (no code)
                    result = {
                        'plan_id': plan.id,
                        'title': plan.title,
                        'success': True,
                        'output': 'Theoretical analysis completed',
                        'note': 'No executable code provided',
                    }

                results.append(result)

            except Exception as e:
                logger.error(f"Experiment execution failed: {e}")
                results.append({
                    'plan_id': plan.id,
                    'title': plan.title,
                    'success': False,
                    'error': str(e),
                })

        return results

    async def _analyze_results(
        self,
        hypotheses: List[ResearchHypothesis],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze experiment results against hypotheses"""

        # Build context for LLM
        context = {
            'hypotheses': [self._hypothesis_to_dict(h) for h in hypotheses],
            'results': results,
        }

        prompt = f"""Analyze these experimental results:

{json.dumps(context, indent=2)}

For each hypothesis, determine:
1. Whether it was supported, rejected, or inconclusive
2. Key evidence from the results
3. Confidence level (0-1)
4. Recommendations for next steps

Respond with ONLY valid JSON:
{{
    "hypothesis_evaluations": [
        {{
            "hypothesis_id": "hyp1",
            "status": "supported|rejected|inconclusive",
            "evidence": ["evidence 1", "evidence 2"],
            "confidence": 0.85,
            "recommendation": "what to do next"
        }}
    ],
    "overall_findings": "summary of key findings",
    "limitations": ["limitation 1", "limitation 2"]
}}
"""

        response = await self.llm.completion(
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = json.loads(content)

        # Update hypothesis statuses
        for eval_data in analysis.get('hypothesis_evaluations', []):
            hyp_id = eval_data['hypothesis_id']
            for hypothesis in hypotheses:
                if hypothesis.id == hyp_id:
                    status_str = eval_data['status'].upper()
                    hypothesis.status = HypothesisStatus[status_str]
                    hypothesis.evidence = eval_data.get('evidence', [])

        return analysis

    async def _validate_findings(self, analysis: Dict[str, Any]) -> bool:
        """Validate that findings are real, not simulated"""

        if not self.validation_strict:
            return True

        # Check for simulation keywords
        simulation_keywords = [
            'simulated', 'simulation', 'mock', 'placeholder',
            'demonstration', 'sample data', 'fake', 'dummy'
        ]

        analysis_str = json.dumps(analysis).lower()

        for keyword in simulation_keywords:
            if keyword in analysis_str:
                logger.warning(f"Validation failed: found simulation keyword '{keyword}'")
                if self.simulation_detection:
                    return False

        return True

    # Helper methods

    def _hypothesis_to_dict(self, hypothesis: ResearchHypothesis) -> Dict[str, Any]:
        """Convert hypothesis to dictionary"""
        return {
            'id': hypothesis.id,
            'statement': hypothesis.statement,
            'reasoning': hypothesis.reasoning,
            'testable_predictions': hypothesis.testable_predictions,
            'success_criteria': hypothesis.success_criteria,
            'status': hypothesis.status.value,
            'evidence': hypothesis.evidence,
        }

    def _plan_to_dict(self, plan: ExperimentPlan) -> Dict[str, Any]:
        """Convert experiment plan to dictionary"""
        return {
            'id': plan.id,
            'hypothesis_id': plan.hypothesis_id,
            'title': plan.title,
            'description': plan.description,
            'methodology': plan.methodology,
            'expected_outcomes': plan.expected_outcomes,
            'data_requirements': plan.data_requirements,
        }


# Required for JSON parsing
import re
