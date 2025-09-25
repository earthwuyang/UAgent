"""Scientific Research Engine - Experimental research with hypothesis testing (MOST COMPLEX)"""

import asyncio
import json
import logging
import os
import shutil
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..llm_client import LLMClient
from ..openhands import OpenHandsClient, CodeGenerationRequest
from ..websocket_manager import progress_tracker
from .deep_research import DeepResearchEngine, ResearchResult as DeepResearchResult
from .code_research import CodeResearchEngine, CodeResearchResult
from ...integrations.openhands_bridge import (
    OpenHandsGoalPlanBridge,
    GoalPlan,
    GoalPlanStep,
)
from ...utils.json_utils import (
    JsonParseError,
    safe_json_loads,
    sanitize_json_strings,
)
from ...debate import (
    DebateManager,
    DebateConfig,
    DebaterConfig,
    DebatePolicy,
    should_debate,
)
from ...debate.debate_manager import DEFAULT_RUBRIC as DEFAULT_DEBATE_RUBRIC
from ...memory import AgentMemory

IDEA_PROMPT_TEMPLATE = (
    "Generate up to {max_ideas} distinct, high-impact research ideas that could address the "
    "following scientific problem:\n\n\"{research_question}\"\n\n"
    "Respond **only** with JSON array. Each idea object must provide:\n"
    "- \"id\": short slug (lowercase, hyphen/underscore allowed)\n"
    "- \"title\": concise descriptive title\n"
    "- \"summary\": 2-3 sentence overview\n"
    "- \"objective\": primary scientific objective or hypothesis\n"
    "- \"plan\": proposed experimental or analytical plan in 3-4 bullet sentences\n"
    "- \"search_queries\": list of 2-4 search queries that should be investigated further\n\n"
    "Ensure ideas are mutually distinct and feasible for an academic research lab. Do not include "
    "narrative outside JSON."
)

try:  # GEPA is optional and only instantiated when dependencies exist
    from ...optim.gepa_optimizer import GEPAOptimizer, GEPAConfig
    from ...optim.meta_optimizer import MetaOptimizer, MetaOptConfig
except ImportError:  # pragma: no cover - graceful fallback when dspy/gepa absent
    GEPAOptimizer = None  # type: ignore
    GEPAConfig = None  # type: ignore
    MetaOptimizer = None  # type: ignore
    MetaOptConfig = None  # type: ignore

IdeaGenerationProgram = None


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


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
    variables: Dict[str, Any]
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence_level: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class ExperimentDesign:
    """Experimental design specification"""
    id: str
    hypothesis_id: str
    name: str
    description: str
    methodology: str
    variables: Dict[str, Any]
    controls: List[str]
    data_collection_plan: Dict[str, Any]
    analysis_plan: str
    expected_duration: str
    resource_requirements: Dict[str, Any]
    code_requirements: List[str]
    dependencies: List[str]


@dataclass
class ExperimentExecution:
    """Experiment execution tracking"""
    id: str
    design_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0
    output_data: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    goal_plan: Optional[GoalPlan] = None
    workspace_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class ExperimentResult:
    """Comprehensive experiment result"""
    execution_id: str
    hypothesis_id: str
    status: ExperimentStatus
    data: Dict[str, Any]
    analysis: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    visualization_data: Dict[str, Any]
    conclusions: List[str]
    confidence_score: float
    reproducibility_score: float
    limitations: List[str]
    next_steps: List[str]


@dataclass
class IdeaEvaluation:
    """Evaluation and scoring for a research idea"""

    idea_id: str
    overall_score: float
    scores: Dict[str, float]
    decision: str
    strengths: List[str]
    weaknesses: List[str]
    comments: str


@dataclass
class OpenHandsSessionContext:
    """Metadata for an acquired OpenHands runtime session."""

    session_id: str
    workspace_id: str
    research_session_id: str
    idea_id: str
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchIdea:
    """Candidate research idea explored during scientific research"""

    id: str
    title: str
    summary: str
    objective: str
    plan: str
    search_queries: List[str]
    node_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    literature_review: Optional[DeepResearchResult] = None
    code_analysis: Optional[CodeResearchResult] = None
    hypotheses: List[ResearchHypothesis] = field(default_factory=list)
    experiments: List[ExperimentDesign] = field(default_factory=list)
    executions: List[ExperimentExecution] = field(default_factory=list)
    results: List[ExperimentResult] = field(default_factory=list)
    evaluation: Optional[IdeaEvaluation] = None
    confidence_score: float = 0.0
    iteration_count: int = 0


@dataclass
class ScientificResearchResult:
    """Comprehensive scientific research result"""
    research_id: str
    query: str
    literature_review: Optional[DeepResearchResult]
    code_analysis: Optional[CodeResearchResult]
    hypotheses: List[ResearchHypothesis]
    experiments: List[ExperimentDesign]
    executions: List[ExperimentExecution]
    results: List[ExperimentResult]
    synthesis: Dict[str, Any]
    final_conclusions: List[str]
    confidence_score: float
    reproducibility_report: Dict[str, Any]
    publication_draft: str
    iteration_count: int
    debates: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    ideas: List[ResearchIdea] = field(default_factory=list)
    idea_evaluations: Dict[str, IdeaEvaluation] = field(default_factory=dict)
    selected_idea_id: Optional[str] = None


class HypothesisGenerator:
    """Generate and refine research hypotheses"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.max_json_retries = 3
        self.max_generation_tokens = int(os.getenv("HYPOTHESIS_MAX_TOKENS", "3200"))

    async def generate_hypotheses(
        self,
        research_question: str,
        literature_context: Optional[str] = None,
        code_context: Optional[str] = None
    ) -> List[ResearchHypothesis]:
        """Generate testable hypotheses for research question"""

        context = f"Research Question: {research_question}\n"
        if literature_context:
            context += f"Literature Context: {literature_context[:1000]}...\n"
        if code_context:
            context += f"Code Context: {code_context[:1000]}...\n"

        hypothesis_prompt = f"""
        Based on the research question and context, generate 3-5 testable hypotheses.

        {context}

        For each hypothesis, provide:
        1. "statement": Clear, testable hypothesis statement
        2. "reasoning": Scientific reasoning behind the hypothesis
        3. "testable_predictions": List of specific predictions that can be tested
        4. "success_criteria": Quantitative criteria for validation
        5. "variables": Independent and dependent variables to measure

        Respond with JSON array of hypothesis objects.
        """

        prompt = hypothesis_prompt.rstrip() + "\n\nRespond with a valid JSON array only. Do not include code fences or prose."

        hypotheses_data: Any = None
        last_response: str | None = None
        for attempt in range(1, self.max_json_retries + 1):
            response = await self.llm_client.generate(
                prompt,
                max_tokens=self.max_generation_tokens,
                temperature=0.3 if attempt > 1 else 0.4,
            )
            self.logger.debug(
                "Hypothesis prompt response (attempt %s): %r",
                attempt,
                response,
            )

            raw_response = str(response)
            sanitized_response = sanitize_json_strings(raw_response)
            last_response = raw_response

            if not raw_response.strip():
                self.logger.warning(
                    "Hypothesis generation returned empty response on attempt %s",
                    attempt,
                )
                if attempt == self.max_json_retries:
                    raise RuntimeError(
                        "Hypothesis generation returned no content after multiple attempts"
                    )
                continue

            try:
                hypotheses_data = safe_json_loads(sanitized_response)
                if isinstance(hypotheses_data, dict):
                    hypotheses_data = [hypotheses_data]
                break
            except JsonParseError as exc:
                self.logger.warning(
                    "Failed to parse hypothesis JSON on attempt %s: %s. Raw response: %s",
                    attempt,
                    exc,
                    raw_response,
                )
                if attempt < self.max_json_retries:
                    prompt = (
                        hypothesis_prompt
                        + "\n\nIMPORTANT: Your previous answer was not valid JSON and raised the following parsing error\n"
                        f"{exc}.\n"
                        "Return ONLY a valid JSON array (no backticks, no extra commentary) following the schema above."
                    )

        if hypotheses_data is None and last_response:
            sanitized_text = sanitize_json_strings(last_response)
            self.logger.warning(
                "Hypothesis generation falling back to text coercion; raw response: %s",
                last_response,
            )
            hypotheses_data = self._coerce_hypotheses_from_text(sanitized_text)

        if not isinstance(hypotheses_data, list):
            raise RuntimeError(
                "Hypothesis generation did not return a list; received type "
                f"{type(hypotheses_data).__name__}"
            )

        hypotheses = []
        for i, hyp_data in enumerate(hypotheses_data):
            try:
                hypothesis = ResearchHypothesis(
                    id=f"hyp_{uuid.uuid4().hex[:8]}",
                    statement=hyp_data.get("statement", f"Hypothesis {i+1}"),
                    reasoning=hyp_data.get("reasoning", ""),
                    testable_predictions=hyp_data.get("testable_predictions", []),
                    success_criteria=hyp_data.get("success_criteria", {}),
                    variables=hyp_data.get("variables", {}),
                )
                hypotheses.append(hypothesis)
            except Exception as item_exc:
                self.logger.debug("Failed to parse hypothesis item %s: %s", hyp_data, item_exc)

        if not hypotheses:
            self.logger.error(
                "Hypothesis generation produced zero valid hypotheses; using structured fallback"
            )
            fallback_payload = self._fallback_hypotheses(research_question)
            for i, hyp_data in enumerate(fallback_payload):
                try:
                    hypothesis = ResearchHypothesis(
                        id=f"hyp_{uuid.uuid4().hex[:8]}",
                        statement=hyp_data.get("statement", f"Fallback hypothesis {i+1}"),
                        reasoning=hyp_data.get("reasoning", ""),
                        testable_predictions=hyp_data.get("testable_predictions", []),
                        success_criteria=hyp_data.get("success_criteria", {}),
                        variables=hyp_data.get("variables", {}),
                    )
                    hypotheses.append(hypothesis)
                except Exception as item_exc:
                    self.logger.debug(
                        "Failed to parse fallback hypothesis item %s: %s", hyp_data, item_exc
                    )

        if not hypotheses:
            raise RuntimeError("Hypothesis generation produced zero valid hypotheses")

        return hypotheses

    def _coerce_hypotheses_from_text(self, raw: str) -> List[Dict[str, Any]]:
        """Best-effort parsing for non-JSON hypothesis responses."""

        text = (raw or "").strip()
        if not text:
            return []

        blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
        results: List[Dict[str, Any]] = []

        def _extract_field(block: str, label: str) -> Optional[str]:
            pattern = re.compile(rf"{label}\s*[:\-]\s*(.+)", re.IGNORECASE)
            match = pattern.search(block)
            return match.group(1).strip() if match else None

        for block in blocks:
            statement = _extract_field(block, "statement")
            reasoning = _extract_field(block, "reasoning") or ""
            predictions_text = _extract_field(block, "testable_predictions") or ""
            success_text = _extract_field(block, "success_criteria") or ""
            variables_text = _extract_field(block, "variables") or ""

            if not statement:
                # Use the first non-empty line as statement fallback.
                first_line = block.splitlines()[0].strip()
                statement = first_line

            predictions: List[str] = []
            if predictions_text:
                predictions = [item.strip("-• ") for item in predictions_text.split("\n") if item.strip()]

            success_criteria: Dict[str, Any] = {}
            if success_text:
                success_criteria = {"detail": success_text}

            variables: Dict[str, Any] = {}
            if variables_text:
                variables = {"description": variables_text}

            results.append(
                {
                    "statement": statement,
                    "reasoning": reasoning or "Additional reasoning not provided",
                    "testable_predictions": predictions,
                    "success_criteria": success_criteria,
                    "variables": variables,
                }
            )

        if not results and text:
            results.append(
                {
                    "statement": text.splitlines()[0],
                    "reasoning": "Hypothesis generated from unstructured response",
                    "testable_predictions": [],
                    "success_criteria": {},
                    "variables": {},
                }
            )

        return results

    def _fallback_hypotheses(self, research_question: str) -> List[Dict[str, Any]]:
        """Fallback structure when the LLM response is invalid."""

        return [
            {
                "statement": f"Applying a targeted intervention improves outcomes for '{research_question}'.",
                "reasoning": "Based on analogous studies, controlled interventions typically yield measurable improvements.",
                "testable_predictions": [
                    "Treatment group performance exceeds baseline by statistically significant margin",
                    "Observed variance remains within acceptable range",
                ],
                "success_criteria": {
                    "p_value": "< 0.05",
                    "effect_size": "> 0.2",
                },
                "variables": {
                    "independent": ["intervention"],
                    "dependent": ["primary_metric"],
                },
            }
        ]

    async def refine_hypothesis(
        self,
        hypothesis: ResearchHypothesis,
        feedback: str,
        experimental_data: Optional[Dict[str, Any]] = None
    ) -> ResearchHypothesis:
        """Refine hypothesis based on feedback or experimental results"""

        refinement_prompt = f"""
        Refine the following hypothesis based on feedback and experimental data:

        Current Hypothesis: {hypothesis.statement}
        Reasoning: {hypothesis.reasoning}
        Feedback: {feedback}
        Experimental Data: {experimental_data or "None available"}

        Provide refined hypothesis in JSON format with:
        1. "statement": Refined hypothesis statement
        2. "reasoning": Updated reasoning
        3. "testable_predictions": Updated predictions
        4. "success_criteria": Refined success criteria
        5. "variables": Updated variable definitions

        Respond with valid JSON only.
        """

        response = await self.llm_client.generate(refinement_prompt)

        try:
            refined_data = safe_json_loads(sanitize_json_strings(response))
        except JsonParseError as exc:
            self.logger.error("Hypothesis refinement returned invalid JSON: %s", exc)
            raise RuntimeError(
                f"Hypothesis refinement returned invalid JSON: {exc}"
            ) from exc

        if not isinstance(refined_data, dict):
            raise RuntimeError("Hypothesis refinement must return a JSON object")

        hypothesis.statement = refined_data.get("statement", hypothesis.statement)
        hypothesis.reasoning = refined_data.get("reasoning", hypothesis.reasoning)
        hypothesis.testable_predictions = refined_data.get(
            "testable_predictions", hypothesis.testable_predictions
        )
        hypothesis.success_criteria = refined_data.get(
            "success_criteria", hypothesis.success_criteria
        )
        hypothesis.variables = refined_data.get("variables", hypothesis.variables)

        return hypothesis


class ExperimentDesigner:
    """Design and plan experiments for hypothesis testing"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.max_generation_tokens = int(os.getenv("EXPERIMENT_DESIGN_MAX_TOKENS", "2800"))

    @staticmethod
    def _safe_loads(payload: str) -> Dict[str, Any]:
        """Parse experiment design JSON or raise a descriptive error."""

        sanitized = sanitize_json_strings(payload)
        try:
            data = safe_json_loads(sanitized)
        except JsonParseError as exc:
            preview = (payload or "").strip().replace("\n", " ")
            if len(preview) > 500:
                preview = preview[:497] + "..."
            logging.getLogger(__name__).error(
                "Experiment design JSON parse failed: %s | raw=%s",
                exc,
                preview or "<empty>",
            )
            raise RuntimeError(f"Experiment design returned invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "Experiment design must return a JSON object with named fields"
            )
        return data

    async def design_experiment(
        self,
        hypothesis: ResearchHypothesis,
        resources: Optional[Dict[str, Any]] = None
    ) -> ExperimentDesign:
        """Design experiment to test hypothesis"""

        design_prompt = f"""
        Design a rigorous experiment to test the following hypothesis:

        Hypothesis: {hypothesis.statement}
        Reasoning: {hypothesis.reasoning}
        Variables: {hypothesis.variables}
        Success Criteria: {hypothesis.success_criteria}
        Available Resources: {resources or "Standard computational resources"}

        Provide experimental design in JSON format with:
        1. "name": Descriptive experiment name
        2. "description": Detailed experiment description
        3. "methodology": Step-by-step methodology
        4. "variables": Detailed variable definitions and measurement plans
        5. "controls": Control conditions and variables
        6. "data_collection_plan": Data collection strategy
        7. "analysis_plan": Statistical analysis plan
        8. "expected_duration": Estimated experiment duration
        9. "resource_requirements": Computational and data requirements
        10. "code_requirements": Programming/implementation requirements
        11. "dependencies": External dependencies needed

        Respond with a raw JSON object only (no markdown fences, no commentary, no code blocks).
        """

        response = await self.llm_client.generate(
            design_prompt,
            max_tokens=self.max_generation_tokens,
        )
        design_data = self._safe_loads(response)

        design = ExperimentDesign(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            hypothesis_id=hypothesis.id,
            name=design_data.get("name", f"Experiment for {hypothesis.id}"),
            description=design_data.get("description", "Experimental validation"),
            methodology=design_data.get("methodology", "Standard experimental methodology"),
            variables=design_data.get("variables", hypothesis.variables),
            controls=design_data.get("controls", []),
            data_collection_plan=design_data.get("data_collection_plan", {}),
            analysis_plan=design_data.get("analysis_plan", "Statistical analysis of results"),
            expected_duration=design_data.get("expected_duration", "1-2 hours"),
            resource_requirements=design_data.get("resource_requirements", {}),
            code_requirements=design_data.get("code_requirements", []),
            dependencies=design_data.get("dependencies", []),
        )

        return design


class ExperimentExecutor:
    """Execute experiments and collect results"""

    def __init__(
        self,
        llm_client: LLMClient,
        openhands_client=None,
        config: Optional[Dict[str, Any]] = None,
        progress_logger: Optional[Callable[[str, str, float, str, Optional[Dict[str, Any]], Optional[str], str], Awaitable[str]]] = None,
    ):
        self.llm_client = llm_client
        self.openhands_client = openhands_client  # For actual code execution
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self._progress_logger = progress_logger
        if self.config.get("allow_simulated_experiments"):
            self.logger.warning(
                "allow_simulated_experiments is deprecated and ignored; simulated runs are no longer supported"
            )
        default_forbidden = [
            "random.uniform",
            "random.gauss",
            "np.random",
            "fake_data",
            "synthetic_data",
        ]
        self.forbidden_code_tokens: List[str] = [
            token.lower()
            for token in self.config.get("forbidden_code_tokens", default_forbidden)
        ]
        self.strict_simulation_guard: bool = bool(
            self.config.get("strict_simulation_guard", False)
        )
        self.ras_spec_path: Optional[Path] = None
        raw_ras = self.config.get("ras_spec_path")
        if isinstance(raw_ras, str) and raw_ras.strip():
            path = Path(raw_ras.strip())
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                self.ras_spec_path = path
            else:
                self.logger.warning("Configured ras_spec_path %s does not exist", path)

    @staticmethod
    def _clean_code_block(raw_code: str) -> str:
        if not raw_code:
            return ""
        cleaned = raw_code.strip()
        fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced[0].strip()
        return cleaned

    def _validate_generated_code(self, code: str) -> None:
        if not code:
            raise RuntimeError("Generated experiment code is empty")
        lower = code.lower()
        flagged = [token for token in self.forbidden_code_tokens if token in lower]
        if flagged:
            message = (
                "Generated experiment code contains potential simulation markers: "
                + ", ".join(flagged)
            )
            if self.strict_simulation_guard:
                raise RuntimeError(message)
            self.logger.warning(message)

    async def execute_experiment(
        self,
        design: ExperimentDesign,
        session_context: Optional[OpenHandsSessionContext] = None,
        prior_errors: Optional[List[str]] = None,
        attempt_number: int = 1,
        attempt_context: Optional[Dict[str, Any]] = None,
    ) -> ExperimentExecution:
        """Execute experiment according to design"""

        execution = ExperimentExecution(
            id=f"exec_{uuid.uuid4().hex[:8]}",
            design_id=design.id,
            status=ExperimentStatus.IN_PROGRESS,
            start_time=datetime.now(),
            workspace_id=session_context.workspace_id if session_context else None,
            session_id=session_context.session_id if session_context else None,
        )
        attempt_context = attempt_context or {}

        prior_errors = list(prior_errors or [])
        try:
            if not self.openhands_client:
                raise RuntimeError(
                    "OpenHands client not configured; simulated experiments are disabled"
                )

            self.logger.info(f"Starting experiment execution: {design.name}")
            if session_context:
                self.logger.debug(
                    "Using OpenHands session %s (workspace=%s) for experiment %s",
                    session_context.session_id,
                    session_context.workspace_id,
                    design.id,
                )

            # Phase 1: Setup and preparation
            execution.logs.append(f"Attempt {attempt_number}: setting up experiment environment")
            if prior_errors:
                execution.logs.append(f"Previous errors to address: {prior_errors[-3:]}")
            execution.progress = 0.1

            if self.openhands_client and session_context:
                # Use OpenHands for actual code execution
                setup_result = await self._setup_experiment_environment(design, session_context)
                execution.logs.append(f"Environment setup: {setup_result}")
            else:
                raise RuntimeError(
                    "OpenHands session unavailable for experiment execution and simulations are not permitted"
                )

            execution.progress = 0.3

            # Phase 2: Data collection
            execution.logs.append(f"Attempt {attempt_number}: collecting experimental data")

            if self.openhands_client and session_context:
                # Execute data collection code (RAS plan preferred)
                data_result = await self._collect_experimental_data(
                    design,
                    execution,
                    session_context,
                    prior_errors,
                    attempt_context=attempt_context,
                )
                execution.output_data.update(data_result or {})
            else:
                raise RuntimeError(
                    "Experiment execution requires an OpenHands session and cannot proceed without one"
                )

            success_flag = False
            if data_result:
                if "success" in data_result:
                    success_flag = bool(data_result["success"])
                elif data_result.get("measurements"):
                    success_flag = True
                else:
                    success_flag = not bool(self.openhands_client)

            if not success_flag:
                execution.progress = 1.0
                execution.status = ExperimentStatus.FAILED
                error_message = (
                    data_result.get("error")
                    if isinstance(data_result, dict)
                    else "Data collection failed"
                )
                if error_message:
                    execution.errors.append(str(error_message))
                execution.logs.append("Data collection failed; skipping analysis phase")
                execution.end_time = datetime.now()
                return execution

            execution.progress = 0.7

            # Phase 3: Analysis
            execution.logs.append("Phase 3: Analyzing collected data")
            analysis_result = await self._analyze_experimental_data(design, execution.output_data)
            execution.intermediate_results = analysis_result

            execution.progress = 1.0
            execution.status = ExperimentStatus.COMPLETED
            execution.end_time = datetime.now()

            self.logger.info(f"Experiment execution completed: {execution.id}")

        except Exception as e:
            execution.status = ExperimentStatus.FAILED
            execution.errors.append(str(e))
            execution.end_time = datetime.now()
            execution.logs.append(
                f"Experiment execution failed: {str(e)}"
            )
            if prior_errors:
                execution.logs.append(
                    f"Prior errors when failing: {prior_errors[-3:]}"
                )
            self.logger.exception(
                "Experiment execution failed for %s (%s)",
                design.id,
                design.name,
            )

        execution.output_data.setdefault("attempt_number", attempt_number)
        if prior_errors:
            execution.output_data.setdefault("prior_errors", prior_errors[-5:])
        return execution

    async def _setup_experiment_environment(
        self,
        design: ExperimentDesign,
        session_context: OpenHandsSessionContext,
    ) -> str:
        """Setup experiment environment using OpenHands"""
        try:
            workspace_id = session_context.workspace_id
            workspace_path = self.openhands_client.workspace_manager.get_workspace_path(workspace_id)
            if not workspace_path:
                # Attempt to recreate the workspace via ensure_session
                await self.openhands_client.ensure_session(
                    research_type="scientific_research",
                    session_id=session_context.session_id,
                    config=session_context.resource_requirements,
                )
                workspace_path = self.openhands_client.workspace_manager.get_workspace_path(workspace_id)

            if not workspace_path:
                raise RuntimeError(f"Workspace not found for OpenHands session {session_context.session_id}")

            # Serialize design metadata safely for the setup script
            design_payload = {
                "id": design.id,
                "name": design.name,
                "description": design.description,
                "methodology": design.methodology,
                "variables": design.variables,
                "controls": design.controls,
                "data_collection_plan": design.data_collection_plan,
                "analysis_plan": design.analysis_plan,
                "resource_requirements": design.resource_requirements,
                "code_requirements": design.code_requirements,
                "dependencies": design.dependencies,
            }
            design_json = json.dumps(design_payload)

            # Perform setup by writing files directly into the workspace (no hardcoded script)
            workspace_path = self.openhands_client.workspace_manager.get_workspace_path(workspace_id)
            if not workspace_path:
                raise RuntimeError("Workspace path missing after ensure_session")

            base_dir = f"experiments/{design.id}"
            # Ensure directories exist by writing .gitkeep files
            await self.openhands_client.workspace_manager.write_file(workspace_id, f"{base_dir}/.gitkeep", "")
            await self.openhands_client.workspace_manager.write_file(workspace_id, f"{base_dir}/data/.gitkeep", "")
            await self.openhands_client.workspace_manager.write_file(workspace_id, f"{base_dir}/results/.gitkeep", "")
            await self.openhands_client.workspace_manager.write_file(workspace_id, f"{base_dir}/logs/.gitkeep", "")
            # Write design.json
            await self.openhands_client.workspace_manager.write_file(
                workspace_id,
                f"{base_dir}/design.json",
                json.dumps(design_payload, indent=2),
                overwrite=True,
            )

            return (
                f"Environment configured for {design.name}: "
                f"{workspace_path / base_dir}"
            )

        except Exception as e:
            self.logger.error(f"Error setting up experiment environment: {e}")
            return f"Environment setup error: {str(e)}"

    async def _collect_experimental_data(
        self,
        design: ExperimentDesign,
        execution: ExperimentExecution,
        session_context: OpenHandsSessionContext,
        prior_errors: Optional[List[str]] = None,
        attempt_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect experimental data using OpenHands code execution."""
        prior_errors = prior_errors or []
        attempt_context = attempt_context or {}

        parent_node_id = attempt_context.get("parent_node_id")
        parent_phase = attempt_context.get("parent_phase")
        progress_anchor = float(attempt_context.get("progress_anchor", execution.progress or 0.0))
        progress_increment = float(attempt_context.get("progress_increment", 5.0))
        phase_prefix = attempt_context.get("phase_prefix") or f"{execution.id}_data_collection"
        codeact_node_id = attempt_context.get("codeact_node_id")

        async def _log_collection_event(
            suffix: str,
            message: str,
            offset: float,
            metadata_overrides: Optional[Dict[str, Any]] = None,
            node_type: str = "step",
        ) -> None:
            nonlocal codeact_node_id
            if not self._progress_logger or not execution.session_id or parent_node_id is None:
                return

            metadata = dict(metadata_overrides or {})
            metadata.setdefault("parent_id", parent_node_id)
            metadata.setdefault("node_type", node_type)
            metadata.setdefault("title", message)
            metadata.setdefault("phase", suffix)
            if codeact_node_id:
                metadata.setdefault("node_id", codeact_node_id)

            phase_name = f"{phase_prefix}_{suffix}"
            progress_value = max(0.0, min(100.0, progress_anchor + (offset * progress_increment)))

            try:
                new_node_id = await self._progress_logger(
                    execution.session_id,
                    phase_name,
                    progress_value,
                    message,
                    metadata,
                    parent_phase=parent_phase,
                    node_type=node_type,
                )
                if not codeact_node_id and new_node_id:
                    codeact_node_id = new_node_id
            except Exception as exc:  # pragma: no cover - logging best effort
                self.logger.debug("Failed to log data collection event %s: %s", suffix, exc)

        codeact_failures: List[Dict[str, Any]] = []

        try:
            await _log_collection_event(
                suffix="start",
                message="Preparing experimental data collection",
                offset=0.0,
            )

            error_context = ""
            if prior_errors:
                error_context = (
                    "\nKnown issues to resolve (latest first):\n"
                    f"{json.dumps(prior_errors[-3:], indent=2)}\n"
                    "Update the implementation to fix these problems explicitly."
                )

            data_collection_prompt = f"""
Generate pure Python (no markdown fences) that collects **real** experimental data for the following study.

Experiment: {design.name}
Description: {design.description}
Methodology: {design.methodology}
Variables: {design.variables}
Controls: {design.controls}
Data Collection Plan: {design.data_collection_plan}

Non-negotiable constraints:
1. Interact with the actual workspace or databases (PostgreSQL, DuckDB, etc.); do not fabricate or simulate results.
2. Do not use random generators, synthetic stubs, or placeholder measurements.
3. Run real commands or database queries to gather metrics, logging each step.
4. Save collected metrics under 'experiments/{design.id}/results'.
5. Use only standard libraries or dependencies already available.
6. Avoid external network calls.
7. If the experiment requires multi-step command execution, author a ResearchActionSpec JSON (following backend/app/core/exec/ras.py) at 'experiments/{design.id}/ras_spec.json' that lists every fetch/build/run step plus required assertions so the orchestrator can execute it deterministically.
{error_context}

The script must emit detailed logs, gracefully handle errors, and return a dict summarising the collected metrics derived from actual execution.
"""
            execution.intermediate_results.setdefault("data_collection_prompt", data_collection_prompt)


            ras_result: Optional[Dict[str, Any]] = None
            if self.ras_spec_path:
                try:
                    ras_result = await self._run_ras_plan(design, execution, session_context)
                    if ras_result:
                        await _log_collection_event(
                            suffix="ras_complete",
                            message="ResearchActionSpec executed successfully",
                            offset=0.35,
                            metadata_overrides={"node_type": "step", "status": "completed"},
                        )
                        return ras_result
                except Exception as ras_exc:
                    self.logger.error("RAS execution failed: %s", ras_exc)
                    prior_errors.append(str(ras_exc))
                    await _log_collection_event(
                        suffix="ras_error",
                        message=f"RAS execution failed: {ras_exc}",
                        offset=0.3,
                        metadata_overrides={"status": "error", "error": str(ras_exc)},
                    )

            codeact_enabled = (
                self.config.get("use_codeact", True)
                and self.openhands_client
                and getattr(self.openhands_client, "action_runner", None)
                and self.openhands_client.action_runner.is_available
            )

            if codeact_enabled:
                try:
                    workspace_path = self.openhands_client.workspace_manager.get_workspace_path(
                        session_context.workspace_id
                    )
                    if not workspace_path:
                        raise RuntimeError("OpenHands workspace path unavailable for CodeAct execution")

                    from ...services.codeact_runner import CodeActRunner  # lazy import

                    runner = CodeActRunner(self.llm_client, self.openhands_client.action_runner)

                    def _extract_final_result(steps_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                        obs_text = "\n".join(
                            [step.get("observation", "") for step in steps_dict.get("steps", [])]
                        ) if isinstance(steps_dict, dict) else ""
                        if ("Final result:" in obs_text) or ("Result:" in obs_text):
                            line = next(
                                (l for l in obs_text.splitlines() if "Final result:" in l),
                                None,
                            ) or next((l for l in obs_text.splitlines() if "Result:" in l), None)
                            if line:
                                json_str = line.split(":", 1)[1].strip()
                                try:
                                    data = safe_json_loads(sanitize_json_strings(json_str))
                                except JsonParseError as exc:
                                    self.logger.debug(
                                        "CodeAct final result JSON parse failed: %s", exc
                                    )
                                    return None
                                return data if isinstance(data, dict) else None
                        return None

                    event_offsets = {
                        "planning": 0.4,
                        "tool_call": 0.5,
                        "tool_result": 0.6,
                        "tool_result_update": 0.65,
                        "finish": 0.75,
                        "error": 0.7,
                    }

                    async def _cb(event: str, data: Dict[str, Any]) -> None:
                        await _log_collection_event(
                            suffix=event,
                            message=f"CodeAct {event}",
                            offset=event_offsets.get(event, 0.55),
                            metadata_overrides={
                                "node_type": "step",
                                "node_id": codeact_node_id,
                                "parent_id": parent_node_id,
                                "status": data.get("success"),
                                "preview": data.get("observation_preview"),
                                "tool": data.get("tool"),
                            },
                        )

                    base_goal = (
                        f"Create a Python script at code/collect_data_{execution.id}.py implementing collect_data() to collect real measurements per the plan. "
                        f"Run it via bash (python3 code/collect_data_{execution.id}.py) and ensure it prints a single line starting with 'Final result: ' "
                        f"followed by a compact JSON dict of results. Avoid placeholder/boilerplate; write production-ready code with concrete logic, error handling, and provenance logging. "
                        f"Avoid using random number generators (e.g. random.uniform, np.random) unless the protocol explicitly requires them. "
                        f"Do not clone unrelated repositories (e.g., OpenHands); work within the current workspace. "
                        f"Interact with the actual environment (files, processes, databases, or benchmarks) and save metrics under experiments/{design.id}/results. "
                        f"If dependencies are missing, write commands to install or initialize them within the workspace and re-run. "
                        f"If a ResearchActionSpec JSON does not already exist at experiments/{design.id}/ras_spec.json, author one describing each command sequence, required capabilities, artifacts, and assertions so the orchestrator can execute the workflow autonomously."
                    )

                    outer_rounds = int(self.config.get("codeact_rounds", 3))
                    last_obs_preview = ""
                    for round_idx in range(1, outer_rounds + 1):
                        goal_text = base_goal
                        if last_obs_preview:
                            goal_text += (
                                "\n\nPrevious attempt feedback (summarized):\n" + last_obs_preview[:800]
                                + "\nAddress these issues explicitly before running again."
                            )
                        result = await runner.run(
                            workspace_path=workspace_path,
                            goal=goal_text,
                            max_steps=int(self.config.get("codeact_max_steps", 10)),
                            timeout_per_action=int(self.config.get("codeact_action_timeout", 180)),
                            progress_cb=_cb,
                        )
                        execution.intermediate_results.setdefault("codeact_rounds", []).append(result)
                        final_data = _extract_final_result(result)
                        if final_data:
                            await _log_collection_event(
                                suffix="finish",
                                message="CodeAct completed successfully",
                                offset=0.85,
                                metadata_overrides={"node_type": "result", "status": "completed"},
                                node_type="result",
                            )
                            payload: Dict[str, Any] = {"success": True}
                            payload.update(final_data)
                            return payload

                        try:
                            obs_list = [s.get("observation", "") for s in result.get("steps", [])]
                            last_obs_preview = "\n".join(obs_list[-5:]) if obs_list else ""
                        except Exception:
                            last_obs_preview = ""

                        failure_msg = (
                            result.get("message")
                            or result.get("error")
                            or (last_obs_preview[:400] if last_obs_preview else "No result emitted")
                        )
                        failure_entry = {
                            "round": round_idx,
                            "message": failure_msg,
                            "observation_preview": last_obs_preview[:800] if last_obs_preview else "",
                            "success": result.get("success"),
                        }
                        codeact_failures.append(failure_entry)
                        execution.logs.append(
                            f"CodeAct round {round_idx} did not produce a final result: {failure_msg}"
                        )
                        execution.output_data.setdefault("codeact_failures", []).append(failure_entry)
                        await _log_collection_event(
                            suffix=f"round_{round_idx}_failed",
                            message=f"CodeAct round {round_idx} failed: {failure_msg[:160]}",
                            offset=0.7,
                            metadata_overrides={
                                "status": "error",
                                "error": failure_msg,
                                "round": round_idx,
                            },
                        )

                    summary_parts: List[str] = []
                    for failure in codeact_failures[-3:]:
                        round_id = failure.get("round")
                        snippet_source = (failure.get("message") or "")
                        if not snippet_source:
                            snippet_source = failure.get("observation_preview") or ""
                        snippet = (snippet_source or "").strip().replace("\n", " ")[:180]
                        summary_parts.append(
                            f"round {round_id}: {snippet}" if round_id else snippet
                        )
                    failure_summary = "; ".join(summary_parts) or "no diagnostic output captured"
                    raise RuntimeError(
                        "CodeAct retries exhausted after "
                        f"{outer_rounds} round(s). Last diagnostics: {failure_summary}"
                    )
                except Exception as exc:
                    await _log_collection_event(
                        suffix="error",
                        message=f"CodeAct flow error: {exc}",
                        offset=0.8,
                        metadata_overrides={"status": "error", "error": str(exc)},
                    )
                    raise RuntimeError(f"CodeAct flow error: {exc}") from exc

            raise RuntimeError(
                "CodeAct runtime did not produce a final result or is unavailable; refusing direct LLM code generation."
            )
        except Exception as exc:
            await _log_collection_event(
                suffix="failure",
                message=f"Data collection error: {str(exc)[:160]}",
                offset=0.95,
                metadata_overrides={"status": "error", "error": str(exc)},
            )
            self.logger.exception(
                "Data collection failed for experiment %s (%s)",
                design.id,
                design.name,
            )
            failure_payload: Dict[str, Any] = {"success": False, "error": str(exc)}
            if codeact_failures:
                failure_payload["codeact_failures"] = codeact_failures
            return failure_payload

    async def _run_ras_plan(
        self,
        design: ExperimentDesign,
        execution: ExperimentExecution,
        session_context: OpenHandsSessionContext,
    ) -> Optional[Dict[str, Any]]:
        if not self.ras_spec_path:
            return None
        workspace_path = self.openhands_client.workspace_manager.get_workspace_path(session_context.workspace_id)
        if not workspace_path:
            raise RuntimeError("OpenHands workspace path unavailable for RAS execution")

        spec_payload: Optional[Dict[str, Any]] = None
        spec_origin = None

        if self.ras_spec_path:
            try:
                spec_payload = json.loads(self.ras_spec_path.read_text(encoding="utf-8"))
                spec_origin = str(self.ras_spec_path)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid RAS spec JSON at {self.ras_spec_path}: {exc}") from exc
        else:
            candidates = [
                f"experiments/{design.id}/ras_spec.json",
                f"experiments/{design.id}/plan/ras_spec.json",
                "experiments/ras_spec.json",
                "ras_spec.json",
            ]
            for rel_path in candidates:
                try:
                    content = await self.openhands_client.workspace_manager.read_file(
                        session_context.workspace_id,
                        rel_path,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.debug("Failed to read candidate RAS spec %s: %s", rel_path, exc)
                    continue
                if not content:
                    continue
                try:
                    spec_payload = json.loads(content)
                    spec_origin = rel_path
                    break
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"RAS spec at {rel_path} is not valid JSON: {exc}"
                    ) from exc

        if spec_payload is None:
            raise RuntimeError(
                f"No ResearchActionSpec found. Create a JSON spec (see backend/app/core/exec/ras.py) "
                f"at experiments/{design.id}/ras_spec.json describing the exact fetch/build/run steps "
                "and required assertions before retrying."
            )

        ras_spec = ResearchActionSpec.parse_obj(spec_payload)

        design_context = {
            "description": getattr(design, "description", None),
            "methodology": getattr(design, "methodology", None),
            "analysis_plan": getattr(design, "analysis_plan", None),
            "data_collection_plan": getattr(design, "data_collection_plan", None),
            "code_requirements": getattr(design, "code_requirements", None),
            "resource_requirements": getattr(design, "resource_requirements", None),
        }
        try:
            validate_research_action_spec(
                ras_spec,
                design_context=design_context,
            )
        except RASValidationError as exc:
            origin_label = spec_origin or f"experiments/{design.id}/ras_spec.json"
            message = f"RAS spec at {origin_label} failed validation: {exc}"
            execution.intermediate_results.setdefault("ras_validation_errors", []).append(message)
            self.logger.warning(message)
            raise RuntimeError(message) from exc

        ras_executor = RASExecutor(ws_mgr=self.openhands_client.workspace_manager)
        run_dir = workspace_path

        results = await ras_executor.execute(
            ras=ras_spec,
            run_dir=run_dir,
            goal_id=design.id,
            node_id=execution.id,
        )

        observations_path = run_dir / "artifacts" / "results" / "observations.jsonl"
        model_path = run_dir / "artifacts" / "model.pkl"
        if not observations_path.exists():
            raise RASExecutionError("RAS plan completed without observations.jsonl")

        bench_summary: Dict[str, Any] = {
            "observations_path": str(observations_path),
            "model_path": str(model_path) if model_path.exists() else None,
            "ras_steps": results,
            "ras_spec_origin": spec_origin,
        }

        assertions_path = run_dir / "artifacts" / "assertions.json"
        if assertions_path.exists():
            try:
                bench_summary["assertions"] = json.loads(assertions_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning("Failed to parse RAS assertions: %s", exc)

        try:
            latencies = []
            for line in observations_path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                latencies.append({
                    "sql_id": payload.get("sql_id"),
                    "winner": payload.get("winner"),
                    "pg_ms": payload.get("latency_ms", {}).get("pg"),
                    "duck_ms": payload.get("latency_ms", {}).get("duck"),
                })
            bench_summary["latencies"] = latencies
        except Exception as exc:  # pragma: no cover - defensive parsing
            self.logger.warning("Failed to parse observations: %s", exc)

        # Determine success based on assertions (if present)
        assertions = bench_summary.get("assertions")
        if assertions:
            failed = [item for item in assertions if item.get("status") != "passed"]
            if failed:
                raise RuntimeError(
                    "RAS assertions failed: "
                    + ", ".join(
                        f"{entry.get('type', 'unknown')}={entry.get('status')}" for entry in failed
                    )
                )
        bench_summary["success"] = not assertions or not failed
        return bench_summary

    async def _analyze_experimental_data(self, design: ExperimentDesign, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experimental data"""

        analysis_prompt = f"""
        Analyze the following experimental data according to the analysis plan:

        Analysis Plan: {design.analysis_plan}
        Data Summary: {len(data.get('measurements', []))} data points collected
        Variables: {design.variables}

        Provide analysis in JSON format with:
        1. "descriptive_stats": Basic descriptive statistics
        2. "statistical_tests": Results of statistical tests
        3. "effect_size": Measured effect size
        4. "confidence_interval": Confidence intervals
        5. "p_value": Statistical significance
        6. "interpretation": Interpretation of results

        Respond with valid JSON only.
        """

        try:
            response = await self.llm_client.generate(analysis_prompt)

            import json, re, statistics
            analysis = None
            try:
                analysis = json.loads(response)
            except Exception:
                m = re.search(r"\{[\s\S]*\}\s*$", str(response))
                if m:
                    try:
                        analysis = json.loads(m.group(0))
                    except Exception:
                        analysis = None
            if analysis is None:
                analysis = {}

            # Add computed statistics if we have actual data
            measurements = data.get("measurements") if isinstance(data, dict) else None
            if isinstance(measurements, list) and measurements:
                analysis.setdefault("descriptive_stats", {})
                analysis["descriptive_stats"].update({
                    "count": len(measurements),
                    "min": min(measurements),
                    "max": max(measurements),
                    "mean": statistics.fmean(measurements),
                })

            return analysis

        except Exception as e:
            self.logger.error(f"Error in data analysis: {e}")
            # Fallback analysis computed from data if possible
            import statistics
            measurements = data.get("measurements") if isinstance(data, dict) else None
            if isinstance(measurements, list) and measurements:
                return {
                    "descriptive_stats": {
                        "count": len(measurements),
                        "min": min(measurements),
                        "max": max(measurements),
                        "mean": statistics.fmean(measurements),
                    },
                    "statistical_tests": {},
                    "effect_size": None,
                    "confidence_interval": None,
                    "p_value": None,
                    "interpretation": "LLM analysis failed; returning descriptive stats only.",
                }
            return {
                "descriptive_stats": {},
                "statistical_tests": {},
                "effect_size": None,
                "confidence_interval": None,
                "p_value": None,
                "interpretation": "LLM analysis failed and no measurements available.",
            }


class ScientificResearchEngine:
    """Scientific Research Engine - Most Complex Engine for Experimental Research"""

    def __init__(
        self,
        llm_client: LLMClient,
        deep_research_engine: Optional[DeepResearchEngine] = None,
        code_research_engine: Optional[CodeResearchEngine] = None,
        openhands_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize scientific research engine

        Args:
            llm_client: LLM client for analysis and synthesis
            deep_research_engine: Deep research engine for literature review
            code_research_engine: Code research engine for implementation analysis
            openhands_client: OpenHands client for code execution
            config: Configuration options
        """
        self.llm_client = llm_client
        self.deep_research_engine = deep_research_engine or DeepResearchEngine(llm_client)
        self.code_research_engine = code_research_engine or CodeResearchEngine(llm_client)
        self.openhands_client = openhands_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize sub-engines
        self.hypothesis_generator = HypothesisGenerator(llm_client)
        self.experiment_designer = ExperimentDesigner(llm_client)
        self.experiment_executor = ExperimentExecutor(
            llm_client,
            openhands_client,
            config=self.config,
            progress_logger=self._log_progress,
        )
        self.goal_bridge = (
            OpenHandsGoalPlanBridge(openhands_client, llm_client)
            if openhands_client and llm_client
            else None
        )

        # Optional GEPA meta-optimizer (requires dspy/gepa packages)
        self.meta_optimizer = None
        self._gepa_enabled = self._to_bool(self._env_or_config("gepa_enabled", "GEPA_ENABLED", False))
        if self._gepa_enabled and all(x is not None for x in (GEPAOptimizer, GEPAConfig, MetaOptimizer, MetaOptConfig)):
            try:
                gepa_cfg = GEPAConfig(
                    max_metric_calls=self._to_int(
                        self._env_or_config("gepa_max_metric_calls", "GEPA_MAX_METRIC_CALLS", 150)
                    ),
                    reflection_lm=str(
                        self._env_or_config("gepa_reflection_lm", "GEPA_REFLECTION_LM", "openai/gpt-5")
                    ),
                    task_lm=str(self._env_or_config("gepa_task_lm", "GEPA_TASK_LM", "openai/gpt-4.1-mini")),
                    min_delta=self._to_float(
                        self._env_or_config("gepa_min_delta", "GEPA_MIN_DELTA", 0.02)
                    ),
                )
                meta_cfg = MetaOptConfig(
                    trigger_threshold=self._to_float(
                        self._env_or_config("gepa_trigger_threshold", "GEPA_TRIGGER_THRESHOLD", 0.5)
                    ),
                    enabled=True,
                )
                self.meta_optimizer = MetaOptimizer(GEPAOptimizer(gepa_cfg), meta_cfg)
                self.logger.info(
                    "GEPA meta-optimizer initialised (max_metric_calls=%s, min_delta=%.3f)",
                    gepa_cfg.max_metric_calls,
                    gepa_cfg.min_delta,
                )
            except Exception as exc:  # pragma: no cover - defensive, dependency issues
                self.logger.warning("GEPA optimizer unavailable: %s", exc)
                self.meta_optimizer = None
        elif self._gepa_enabled:
            self.logger.warning(
                "GEPA requested but dependencies missing; install dspy-ai>=2.6.0 and gepa>=0.2.0"
            )

        self._idea_prompt_template = IDEA_PROMPT_TEMPLATE
        self._idea_program = None
        self._gepa_traces: Dict[str, List[Dict[str, Any]]] = {"idea_generation": []}
        self._gepa_last_reward: Dict[str, float] = {"idea_generation": 1.0}
        self._gepa_dataset_limit = self._to_int(self.config.get("gepa_dataset_limit", 64))

        # Configuration
        self.max_iterations = self.config.get("max_iterations", 3)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.8)
        # Load from environment variables with fallback to config
        self.experiments_per_hypothesis = max(1, int(os.getenv("EXPERIMENTS_PER_HYPOTHESIS", self.config.get("experiments_per_hypothesis", 2))))
        self.max_attempts_per_experiment = max(1, int(self.config.get("max_attempts_per_experiment", 5)))

        # Log the parallelism settings for debugging
        self.logger.info(
            "Research parallelism settings: experiments_per_hypothesis=%d, "
            "MAX_PARALLEL_IDEAS=%s, MAX_RESEARCH_IDEAS=%s",
            self.experiments_per_hypothesis,
            os.getenv("MAX_PARALLEL_IDEAS", "not set"),
            os.getenv("MAX_RESEARCH_IDEAS", "not set")
        )
        default_resources = self.config.get("default_openhands_resources") or {}
        if not isinstance(default_resources, dict):
            default_resources = {}
        self.default_openhands_resources = default_resources

        self._session_phase_nodes: Dict[str, Dict[str, str]] = {}
        self._session_root_nodes: Dict[str, str] = {}

        self.preserve_successful_workspaces = self._to_bool(
            self._env_or_config("preserve_successful_workspaces", "PRESERVE_SUCCESSFUL_WORKSPACES", True)
        )
        archive_dir_cfg = self._env_or_config(
            "success_workspace_archive_dir",
            "SUCCESS_WORKSPACE_ARCHIVE_DIR",
            "./artifacts/successful_experiments",
        )
        self.success_workspace_archive_dir = Path(archive_dir_cfg).expanduser().resolve()
        if self.preserve_successful_workspaces:
            self.success_workspace_archive_dir.mkdir(parents=True, exist_ok=True)

        pool_limit = int(self.config.get("openhands_session_limit", 4))
        self._openhands_session_lock = asyncio.Lock()
        self._openhands_session_semaphore = asyncio.Semaphore(max(1, pool_limit))
        self._openhands_sessions: Dict[str, OpenHandsSessionContext] = {}

        # Debate & memory integration
        memory_store = self.config.get("memory_store")
        self.memory_store: Optional[AgentMemory] = memory_store if isinstance(memory_store, AgentMemory) else None

        self._debate_enabled = self._to_bool(self.config.get("debate_enabled", True))
        self._debate_policy = DebatePolicy(
            min_confidence=float(self.config.get("debate_trigger_confidence", 0.65)),
            stakes=str(self.config.get("debate_trigger_stakes", "high")),
        )
        self._debate_defaults = {
            "num_agents": int(self.config.get("debate_max_agents", 4)),
            "num_rounds": int(self.config.get("debate_max_rounds", 2)),
            "groups": int(self.config.get("debate_groups", 1)),
        }
        try:
            self.debate_manager: Optional[DebateManager] = DebateManager(
                self.llm_client,
                memory=self.memory_store,
                websocket_manager=None,
            ) if self._debate_enabled else None
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Debate manager unavailable: %s", exc)
            self.debate_manager = None

    def _get_root_node_id(self, session_id: Optional[str]) -> str:
        if not session_id:
            return "scientific_research-root"
        if session_id not in self._session_root_nodes:
            self._session_root_nodes[session_id] = f"{session_id}-scientific_research-root"
        return self._session_root_nodes[session_id]

    def _openhands_session_key(self, research_session_id: Optional[str], idea_id: str) -> str:
        return f"{research_session_id or 'global'}::{idea_id}"

    async def _acquire_openhands_session(
        self,
        research_session_id: Optional[str],
        idea_id: str,
        resource_requirements: Optional[Dict[str, Any]] = None,
    ) -> Optional[OpenHandsSessionContext]:
        if not self.openhands_client:
            return None

        key = self._openhands_session_key(research_session_id, idea_id)

        async with self._openhands_session_lock:
            existing = self._openhands_sessions.get(key)
            if existing:
                return existing

        await self._openhands_session_semaphore.acquire()
        try:
            # Use a unified session identifier that matches both OpenHands and Goal Bridge usage
            # Format: "experiment_{research_session_id}_{idea_id}"
            if research_session_id:
                session_identifier = f"experiment_{research_session_id}_{idea_id}"
            else:
                session_identifier = f"experiment_{idea_id}"

            session_config = await self.openhands_client.ensure_session(
                research_type="scientific_research",
                session_id=session_identifier,
                config=resource_requirements,
            )

            workspace_id = session_config.workspace_config.get("workspace_id")
            context = OpenHandsSessionContext(
                session_id=session_identifier,
                workspace_id=workspace_id,
                research_session_id=research_session_id or "global",
                idea_id=idea_id,
                resource_requirements=resource_requirements or {},
            )

            async with self._openhands_session_lock:
                self._openhands_sessions[key] = context

            self.logger.debug(
                "Acquired OpenHands session %s (workspace=%s) for idea %s",
                session_identifier,
                workspace_id,
                idea_id,
            )
            return context
        except Exception:
            self._openhands_session_semaphore.release()
            raise

    async def _release_openhands_session(self, context: OpenHandsSessionContext) -> None:
        if not context:
            return

        key = self._openhands_session_key(context.research_session_id, context.idea_id)

        async with self._openhands_session_lock:
            existing = self._openhands_sessions.pop(key, None)

        try:
            if existing:
                try:
                    await self.openhands_client.close_session(existing.session_id)
                    self.logger.debug(
                        "Closed OpenHands session %s (workspace=%s)",
                        existing.session_id,
                        existing.workspace_id,
                    )
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    self.logger.warning(
                        "Failed to close OpenHands session %s: %s",
                        existing.session_id,
                        exc,
                    )
        finally:
            if existing:
                self._openhands_session_semaphore.release()

    async def _release_all_openhands_sessions(self, research_session_id: Optional[str]) -> None:
        if not self.openhands_client:
            return

        async with self._openhands_session_lock:
            keys = [
                key
                for key, ctx in self._openhands_sessions.items()
                if ctx.research_session_id == (research_session_id or "global")
            ]

        for key in keys:
            async with self._openhands_session_lock:
                ctx = self._openhands_sessions.pop(key, None)
            if ctx:
                try:
                    try:
                        await self.openhands_client.close_session(ctx.session_id)
                        self.logger.debug(
                            "Closed OpenHands session %s during cleanup",
                            ctx.session_id,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.warning(
                            "Failed to close OpenHands session %s during cleanup: %s",
                            ctx.session_id,
                            exc,
                        )
                finally:
                    self._openhands_session_semaphore.release()

    @staticmethod
    def _safe_parse_json_block(content: str) -> Any:
        if not content:
            return None
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}\s*$", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _normalize_idea_id(title: str, fallback: Optional[str] = None) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", (title or "idea").lower()).strip("-")
        if not base:
            base = fallback or f"idea-{uuid.uuid4().hex[:6]}"
        return base[:40]

    def _env_or_config(self, key: str, env_key: str, default: Any) -> Any:
        if isinstance(self.config, dict) and key in self.config:
            return self.config[key]
        env_val = os.getenv(env_key)
        return env_val if env_val is not None else default

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _format_idea_prompt(self, question: str, max_ideas: int) -> str:
        return self._idea_prompt_template.format(
            max_ideas=max(1, max_ideas),
            research_question=question,
        )

    @staticmethod
    def _idea_response_to_dicts(response: str, max_ideas: int) -> List[Dict[str, Any]]:
        parsed = ScientificResearchEngine._safe_parse_json_block(response)
        if not isinstance(parsed, list):
            try:
                parsed = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                parsed = None
        ideas_raw: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            for item in parsed[:max_ideas]:
                if isinstance(item, dict):
                    ideas_raw.append({
                        "id": item.get("id"),
                        "title": item.get("title"),
                        "summary": item.get("summary"),
                        "objective": item.get("objective"),
                        "plan": item.get("plan"),
                        "search_queries": item.get("search_queries"),
                    })
        return ideas_raw

    def _build_research_ideas_from_dicts(
        self,
        idea_dicts: List[Dict[str, Any]],
        research_question: str,
        max_ideas: int,
    ) -> List[ResearchIdea]:
        ideas: List[ResearchIdea] = []
        for idx, item in enumerate(idea_dicts[:max_ideas], start=1):
            raw_id = item.get("id") or item.get("title") or item.get("summary")
            idea_id = self._normalize_idea_id(raw_id or research_question, fallback=f"idea-{idx}")
            raw_queries = item.get("search_queries")
            if not isinstance(raw_queries, list):
                raw_queries = []
            queries = [str(q).strip() for q in raw_queries if str(q).strip()]
            if not queries:
                queries = [item.get("title") or research_question]
            ideas.append(
                ResearchIdea(
                    id=idea_id,
                    title=item.get("title", f"Idea {idx}"),
                    summary=item.get("summary", "Proposed research direction"),
                    objective=item.get("objective", research_question),
                    plan=self._normalize_plan(item.get("plan")),
                    search_queries=queries,
                )
            )

        if not ideas:
            fallback_id = self._normalize_idea_id(research_question, fallback="idea-core")
            ideas = [
                ResearchIdea(
                    id=fallback_id,
                    title=f"Core exploration: {research_question[:60]}",
                    summary="Baseline investigation derived from the original question.",
                    objective=research_question,
                    plan="Perform literature review, derive hypotheses, run experiments using available datasets.",
                    search_queries=[research_question, f"state of the art {research_question}"],
                )
            ]
        return ideas

    @staticmethod
    def _idea_to_dict(idea: ResearchIdea) -> Dict[str, Any]:
        return {
            "title": idea.title,
            "summary": idea.summary,
            "objective": idea.objective,
            "plan": idea.plan,
            "search_queries": list(idea.search_queries),
        }

    @staticmethod
    def _score_research_ideas(ideas: List[ResearchIdea], max_ideas: int) -> float:
        if not ideas:
            return 0.0
        max_ideas = max(1, max_ideas)
        quantity = min(1.0, len(ideas) / max_ideas)
        plan_coverage = sum(1 for idea in ideas if idea.plan and idea.plan.strip()) / len(ideas)
        query_coverage = sum(1 for idea in ideas if idea.search_queries) / len(ideas)
        summary_quality = sum(
            1 for idea in ideas if idea.summary and len(idea.summary.split()) >= 10
        ) / len(ideas)
        uniqueness = len({idea.title.lower() for idea in ideas if idea.title}) / len(ideas)
        score = (
            0.35 * quantity
            + 0.25 * plan_coverage
            + 0.20 * summary_quality
            + 0.20 * uniqueness
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_idea_dicts(ideas: List[Dict[str, Any]], max_ideas: int) -> float:
        temp_ideas = [
            ResearchIdea(
                id=f"tmp-{idx}",
                title=data.get("title", ""),
                summary=data.get("summary", ""),
                objective=data.get("objective", ""),
                plan=ScientificResearchEngine._normalize_plan(data.get("plan")),
                search_queries=[
                    str(q).strip() for q in data.get("search_queries", []) if str(q).strip()
                ],
            )
            for idx, data in enumerate(ideas, start=1)
        ]
        return ScientificResearchEngine._score_research_ideas(temp_ideas, max_ideas)

    @staticmethod
    def _normalize_plan(raw_plan: Any) -> str:
        if isinstance(raw_plan, str):
            return raw_plan.strip()
        if isinstance(raw_plan, list):
            return " ".join(
                str(item).strip()
                for item in raw_plan
                if str(item).strip()
            )
        if isinstance(raw_plan, dict):
            return " ".join(
                f"{key}: {value}".strip()
                for key, value in raw_plan.items()
                if str(value).strip()
            )
        if raw_plan is None:
            return ""
        return str(raw_plan).strip()

    def _record_gepa_example(
        self,
        stage: str,
        question: str,
        prompt_used: str,
        ideas: List[Dict[str, Any]],
        score: float,
    ) -> None:
        if not self.meta_optimizer:
            return
        entry = {
            "question": question,
            "prompt": prompt_used,
            "ideas": ideas,
            "score": float(score),
        }
        traces = self._gepa_traces.setdefault(stage, [])
        traces.append(entry)
        if len(traces) > self._gepa_dataset_limit:
            del traces[0 : len(traces) - self._gepa_dataset_limit]

    async def _generate_ideas_with_prompt(
        self,
        question: str,
        max_ideas: int,
    ) -> Tuple[str, str, bool]:
        if self._idea_program is not None:
            try:
                result = await asyncio.to_thread(self._idea_program_forward, question)
                if isinstance(result, dict):
                    raw = result.get("raw", "")
                    prompt_used = result.get("prompt", self._idea_prompt_template)
                else:
                    raw = str(result)
                    prompt_used = self._idea_prompt_template
                if raw:
                    return raw, prompt_used, True
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("GEPA idea program failed, falling back: %s", exc)

        prompt_used = self._format_idea_prompt(question, max_ideas)
        response = await self.llm_client.generate(
            prompt_used,
            max_tokens=6000,  # Increased to prevent truncation of complex hypotheses
            temperature=self.config.get("idea_generation_temperature", 0.6),
        )
        return response, prompt_used, False

    def _idea_program_forward(self, question: str) -> Dict[str, Any]:
        sample = {"question": question}
        result = self._idea_program(sample)  # type: ignore[operator]
        if isinstance(result, dict):
            return result
        return {"raw": str(result), "prompt": self._idea_prompt_template, "ideas": []}

    def _run_gepa_opt(
        self,
        stage: str,
        dataset: List[Dict[str, Any]],
        last_reward: float,
        max_ideas: int,
    ) -> None:
        if not self.meta_optimizer or stage != "idea_generation":
            return
        min_examples = self._to_int(self.config.get("gepa_min_examples", 6))
        if len(dataset) < min_examples:
            return

        val_size = max(1, len(dataset) // 5)
        valset = dataset[:val_size]
        trainset = dataset[val_size:]
        if not trainset or not valset:
            return

        def program_ctor():
            if IdeaGenerationProgram is None:
                raise RuntimeError("GEPA optimizer unavailable (dspy not installed)")
            return IdeaGenerationProgram(self.llm_client, max_ideas, self._idea_prompt_template)

        def metric_fn(example: Any, prediction: Any) -> float:
            ideas = prediction.get("ideas", []) if isinstance(prediction, dict) else []
            return ScientificResearchEngine._score_idea_dicts(ideas, max_ideas)

        compiled, info = self.meta_optimizer.maybe_improve_program(
            last_reward=last_reward,
            program_ctor=program_ctor,
            trainset=trainset,
            valset=valset,
            metric_fn=metric_fn,
            lm_conf={},
        )
        if compiled is not None and info and info.get("accepted"):
            self._idea_program = compiled
            self.logger.info(
                "[GEPA] Adopted idea-generation program (val_gain=%.4f)",
                float(info.get("val_gain", 0.0)),
            )

    async def _maybe_run_gepa(self, stage: str, last_reward: float, max_ideas: int) -> None:
        """Trigger GEPA optimisation asynchronously when sufficient traces exist."""

        if not self.meta_optimizer:
            return

        dataset = self._gepa_traces.get(stage)
        if not dataset:
            return

        if len(dataset) < self._to_int(self.config.get("gepa_min_examples", 6)):
            return

        await asyncio.to_thread(
            self._run_gepa_opt,
            stage,
            list(dataset),
            float(last_reward),
            max_ideas,
        )

    def _max_parallel_ideas(self, total: int) -> int:
        # Load from environment variables with fallback to config
        configured = int(os.getenv("MAX_PARALLEL_IDEAS", self.config.get("max_parallel_ideas", 2)))
        result = max(1, min(configured, max(1, total)))
        self.logger.info("Processing %d ideas with parallelism=%d (MAX_PARALLEL_IDEAS=%d)", total, result, configured)
        return result

    async def _generate_research_ideas(
        self,
        research_question: str,
        session_id: Optional[str],
    ) -> List[ResearchIdea]:
        # Load from environment variables with fallback to config
        max_ideas = int(os.getenv("MAX_RESEARCH_IDEAS", self.config.get("max_ideas", 3)))
        self.logger.info("Generating up to %d research ideas (MAX_RESEARCH_IDEAS)", max_ideas)
        await self._maybe_run_gepa(
            stage="idea_generation",
            last_reward=self._gepa_last_reward.get("idea_generation", 0.0),
            max_ideas=max_ideas,
        )

        response, prompt_used, _ = await self._generate_ideas_with_prompt(research_question, max_ideas)
        idea_dicts = self._idea_response_to_dicts(response, max_ideas)
        ideas = self._build_research_ideas_from_dicts(idea_dicts, research_question, max_ideas)

        idea_dataset_entry = [self._idea_to_dict(idea) for idea in ideas]
        idea_score = self._score_research_ideas(ideas, max_ideas)
        self._gepa_last_reward["idea_generation"] = idea_score
        self._record_gepa_example(
            "idea_generation",
            research_question,
            prompt_used,
            idea_dataset_entry,
            idea_score,
        )

        if session_id:
            parent_id = await self._log_progress(
                session_id,
                phase="ideation",
                progress=15.0,
                message=f"Generated {len(ideas)} research ideas",
                metadata={
                    "node_type": "group",
                    "title": "Idea Generation",
                    "idea_count": len(ideas),
                },
                parent_phase="initializing",
            )
            for index, idea in enumerate(ideas, start=1):
                idea.parent_node_id = parent_id
                idea.node_id = await self._log_progress(
                    session_id,
                    phase=f"idea_{idea.id}",
                    progress=18.0 + index,
                    message=idea.title,
                    metadata={
                        "parent_id": parent_id,
                        "node_type": "idea",
                        "title": idea.title,
                        "summary": idea.summary,
                        "objective": idea.objective,
                        "plan": idea.plan,
                    },
                    parent_phase="ideation",
                )

        return ideas

    async def _log_progress(
        self,
        session_id: Optional[str],
        phase: str,
        progress: float,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_phase: Optional[str] = None,
        node_type: str = "step"
    ) -> str:
        if not session_id:
            return ""

        metadata = dict(metadata or {})
        node_id = metadata.get("node_id")

        phase_nodes = self._session_phase_nodes.setdefault(session_id, {})
        if node_id is None:
            node_id = phase_nodes.get(phase) or f"{session_id}-scientific-{phase}-{uuid.uuid4().hex[:6]}"
        parent_id = metadata.get("parent_id")
        if not parent_id:
            if parent_phase and parent_phase in phase_nodes:
                parent_id = phase_nodes[parent_phase]
            else:
                parent_id = self._get_root_node_id(session_id)
        metadata["parent_id"] = parent_id
        phase_nodes[phase] = node_id

        metadata["node_id"] = node_id
        metadata.setdefault("node_type", node_type)
        metadata.setdefault("title", message)
        metadata.setdefault("phase", phase)

        try:
            await progress_tracker.log_research_progress(
                session_id=session_id or "unknown",
                engine="scientific_research",
                phase=phase,
                progress=progress,
                message=message,
                metadata=metadata
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.debug(
                "Failed to log scientific research progress for %s (phase=%s): %s",
                session_id,
                phase,
                exc
            )

        return metadata["node_id"]

    async def _gather_idea_context(
        self,
        idea: ResearchIdea,
        session_id: Optional[str],
        base_progress: float,
        progress_window: float,
        include_literature: bool,
        include_code: bool,
    ) -> Tuple[Optional[DeepResearchResult], Optional[CodeResearchResult]]:
        query = idea.search_queries[0] if idea.search_queries else idea.title

        literature_result: Optional[DeepResearchResult] = None
        code_result: Optional[CodeResearchResult] = None

        if include_literature:
            try:
                literature_result = await self.deep_research_engine.research(query, session_id=None)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Deep research failed for idea %s: %s", idea.id, exc)

        if include_code:
            try:
                code_result = await self.code_research_engine.research_code(
                    query,
                    include_analysis=True,
                    session_id=None,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Code research failed for idea %s: %s", idea.id, exc)

        if session_id and idea.node_id:
            if literature_result:
                await self._log_progress(
                    session_id,
                    phase=f"{idea.id}_literature",
                    progress=base_progress + progress_window * 0.15,
                    message="Literature context gathered",
                    metadata={
                        "parent_id": idea.node_id,
                        "node_type": "analysis",
                        "title": "Literature Review",
                        "summary": literature_result.summary,
                        "key_findings": literature_result.key_findings[:3],
                    },
                    parent_phase=f"idea_{idea.id}",
                )
            if code_result:
                await self._log_progress(
                    session_id,
                    phase=f"{idea.id}_code",
                    progress=base_progress + progress_window * 0.25,
                    message="Code landscape analyzed",
                    metadata={
                        "parent_id": idea.node_id,
                        "node_type": "analysis",
                        "title": "Code Research",
                        "summary": code_result.integration_guide[:200],
                        "repositories": [repo.name for repo in code_result.repositories[:3]],
                    },
                    parent_phase=f"idea_{idea.id}",
                )

        return literature_result, code_result

    async def _log_goal_plan(
        self,
        session_id: Optional[str],
        idea: ResearchIdea,
        plan: GoalPlan,
        base_progress: float,
        progress_window: float,
    ) -> Dict[str, str]:
        if not session_id or not idea.node_id or not plan:
            return {}

        plan_node = await self._log_progress(
            session_id,
            phase=f"{idea.id}_goal_plan",
            progress=base_progress,
            message="Goal plan generated",
            metadata={
                "parent_id": idea.node_id,
                "node_type": "plan",
                "title": "Goal Plan",
                "summary": plan.summary,
            },
            parent_phase=f"idea_{idea.id}",
        )

        step_nodes: Dict[str, str] = {"__plan__": plan_node}
        for idx, step in enumerate(plan.steps, start=1):
            offset = progress_window * (0.1 + 0.05 * idx)
            step_node = await self._log_progress(
                session_id,
                phase=f"{idea.id}_goal_plan_step_{idx}",
                progress=base_progress + offset,
                message=step.description,
                metadata={
                    "parent_id": plan_node,
                    "node_type": "step",
                    "title": step.description,
                    "expected_output": step.expected_output,
                    "status": step.status,
                },
                parent_phase=f"{idea.id}_goal_plan",
            )
            step_nodes[step.id] = step_node

        return step_nodes

    async def _log_goal_plan_step_update(
        self,
        session_id: Optional[str],
        idea: ResearchIdea,
        step: GoalPlanStep,
        node_map: Dict[str, str],
        progress_value: float,
    ) -> None:
        if not session_id or not idea.node_id:
            return
        node_id = node_map.get(step.id)
        if not node_id:
            return
        stdout = ""
        stderr = ""
        if step.code_result and step.code_result.execution_result:
            stdout = step.code_result.execution_result.stdout
            stderr = step.code_result.execution_result.stderr

        await self._log_progress(
            session_id,
            phase=f"{idea.id}_goal_plan_step_{step.id}_update",
            progress=progress_value,
            message=f"{step.description} [{step.status}]",
            metadata={
                "node_id": node_id,
                "parent_id": node_map.get("__plan__") or idea.node_id,
                "node_type": "step",
                "status": step.status,
                "stdout": stdout,
                "stderr": stderr,
            },
            parent_phase=f"idea_{idea.id}_goal_plan",
        )

    async def _run_experiments_for_idea(
        self,
        idea: ResearchIdea,
        session_id: Optional[str],
        base_progress: float,
        progress_window: float,
        max_iterations: int,
        openhands_context: Optional[OpenHandsSessionContext],
    ) -> Tuple[float, int]:
        iteration = 0
        max_confidence = 0.0
        per_iteration_increment = progress_window / max(1, max_iterations)

        while iteration < max_iterations:
            pending_hypotheses = [
                hyp
                for hyp in idea.hypotheses
                if hyp.status in {HypothesisStatus.PENDING, HypothesisStatus.INCONCLUSIVE}
            ]
            if not pending_hypotheses:
                break

            iteration += 1
            idea.iteration_count = iteration
            iteration_progress = base_progress + per_iteration_increment * (iteration - 1)

            if session_id and idea.node_id:
                await self._log_progress(
                    session_id,
                    phase=f"{idea.id}_experiments_iter_{iteration}",
                    progress=iteration_progress,
                    message=f"Experiments iteration {iteration}",
                    metadata={
                        "parent_id": idea.node_id,
                        "node_type": "step",
                        "iteration": iteration,
                        "pending_hypotheses": len(pending_hypotheses),
                    },
                    parent_phase=f"idea_{idea.id}",
                )

            iteration_results: List[ExperimentResult] = []

            for hypothesis in pending_hypotheses:
                self.logger.info("Running %d experiments per hypothesis (EXPERIMENTS_PER_HYPOTHESIS)", self.experiments_per_hypothesis)
                for exp_round in range(1, self.experiments_per_hypothesis + 1):
                    design = await self.experiment_designer.design_experiment(hypothesis)
                    idea.experiments.append(design)

                    if session_id and idea.node_id:
                        await self._log_progress(
                            session_id,
                            phase=f"{idea.id}_design_{design.id}_round_{exp_round}",
                            progress=iteration_progress + per_iteration_increment * 0.2,
                            message=f"Designed experiment (round {exp_round}): {design.name}",
                            metadata={
                                "parent_id": idea.node_id,
                                "node_type": "step",
                                "title": design.name,
                                "methodology": design.methodology,
                                "round": exp_round,
                            },
                            parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
                        )

                    goal_plan_nodes: Dict[str, str] = {}
                    goal_plan: Optional[GoalPlan] = None

                    if self.goal_bridge:
                        # Pass the unified session ID from the OpenHands context
                        # This ensures goal_bridge uses the same workspace as the scientific research
                        plan_context = {
                            "hypothesis": hypothesis.statement,
                            "variables": hypothesis.variables,
                            "analysis_plan": design.analysis_plan,
                            "resource_requirements": design.resource_requirements,
                        }

                        # Add the session_id from openhands_context if available
                        if openhands_context:
                            plan_context["session_id"] = openhands_context.session_id

                        goal_plan = await self.goal_bridge.generate_goal_plan(design.description or design.name, plan_context)
                        goal_plan_nodes = await self._log_goal_plan(
                            session_id,
                            idea,
                            goal_plan,
                            iteration_progress + per_iteration_increment * 0.22,
                            per_iteration_increment * 0.15,
                        )

                        async def _plan_callback(event_type: str, payload: Dict[str, Any]) -> None:
                            if not payload:
                                return
                            step_id = payload.get("step_id")
                            if not step_id:
                                return
                            matching = next((s for s in goal_plan.steps if s.id == step_id), None)
                            if not matching:
                                return
                            progress_point = iteration_progress + per_iteration_increment * (0.25 if event_type == "step_started" else 0.28)
                            await self._log_goal_plan_step_update(
                                session_id,
                                idea,
                                matching,
                                goal_plan_nodes,
                                progress_point,
                            )

                        goal_plan = await self.goal_bridge.execute_goal_plan(
                            goal_plan,
                            plan_context,
                            progress_callback=_plan_callback,
                        )

                    final_execution, execution_history = await self._execute_experiment_with_retries(
                        idea,
                        hypothesis,
                        design,
                        session_id,
                        iteration_progress,
                        per_iteration_increment,
                        openhands_context,
                        iteration,
                        exp_round,
                    )

                    if goal_plan:
                        for exec_record in execution_history:
                            exec_record.goal_plan = goal_plan

                    idea.executions.extend(execution_history)

                    if final_execution.status == ExperimentStatus.COMPLETED:
                        experiment_result = await self._analyze_experiment_result(hypothesis, design, final_execution)
                        idea.results.append(experiment_result)
                        iteration_results.append(experiment_result)
                        await self._update_hypothesis_status(hypothesis, experiment_result)

                        if session_id and idea.node_id:
                            await self._log_progress(
                                session_id,
                                phase=f"{idea.id}_result_{experiment_result.execution_id}",
                                progress=iteration_progress + per_iteration_increment * 0.65,
                                message=f"Result for {design.name}",
                                metadata={
                                    "parent_id": idea.node_id,
                                    "node_type": "result",
                                    "conclusions": experiment_result.conclusions[:2],
                                    "confidence_score": experiment_result.confidence_score,
                                },
                                parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
                            )
                        break
                    else:
                        hypothesis.evidence.append("Experiment execution failed after retries")
                        if session_id and idea.node_id:
                            await self._log_progress(
                                session_id,
                                phase=f"{idea.id}_result_{final_execution.id}",
                                progress=iteration_progress + per_iteration_increment * 0.65,
                                message=f"Experiment {design.name} failed after retries",
                                metadata={
                                    "parent_id": idea.node_id,
                                    "node_type": "result",
                                    "status": final_execution.status.value,
                                    "errors": final_execution.errors[:1] if final_execution.errors else [],
                                },
                                parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
                            )

            current_confidence = self._calculate_overall_confidence(idea.results)
            max_confidence = max(max_confidence, current_confidence)

            if current_confidence >= self.confidence_threshold:
                break

            if iteration < max_iterations and iteration_results:
                await self._refine_hypotheses_for_next_iteration(idea.hypotheses, iteration_results)

        idea.confidence_score = max_confidence
        return max_confidence, iteration

    async def _execute_experiment_with_retries(
        self,
        idea: ResearchIdea,
        hypothesis: ResearchHypothesis,
        design: ExperimentDesign,
        session_id: Optional[str],
        iteration_progress: float,
        per_iteration_increment: float,
        session_context: Optional[OpenHandsSessionContext],
        iteration_number: int,
        experiment_round: int,
    ) -> Tuple[ExperimentExecution, List[ExperimentExecution]]:
        """Execute an experiment with multiple attempts to recover from programming errors."""

        prior_errors: List[str] = []
        execution_history: List[ExperimentExecution] = []
        attempt_base_progress = iteration_progress + per_iteration_increment * 0.4
        parent_phase = f"idea_{idea.id}_experiments_iter_{iteration_number}"

        for attempt in range(1, self.max_attempts_per_experiment + 1):
            attempt_progress = attempt_base_progress + per_iteration_increment * min(0.15, 0.05 * attempt)
            attempt_phase = f"{idea.id}_execution_attempt_{attempt}"
            attempt_node_id: Optional[str] = None

            if session_id and idea.node_id:
                attempt_node_id = await self._log_progress(
                    session_id,
                    phase=attempt_phase,
                    progress=attempt_progress,
                    message=f"Attempt {attempt} for {design.name} (starting)",
                    metadata={
                        "parent_id": idea.node_id,
                        "node_type": "step",
                        "attempt": attempt,
                        "round": experiment_round,
                        "status": "starting",
                    },
                    parent_phase=parent_phase,
                ) or attempt_node_id

            execution = await self.experiment_executor.execute_experiment(
                design,
                session_context,
                prior_errors=prior_errors,
                attempt_number=attempt,
                attempt_context={
                    "parent_node_id": attempt_node_id,
                    "parent_phase": attempt_phase,
                    "progress_anchor": attempt_progress,
                    "progress_increment": per_iteration_increment,
                    "phase_prefix": f"{attempt_phase}_collect",
                },
            )
            execution_history.append(execution)

            if session_id and idea.node_id and attempt_node_id:
                await self._log_progress(
                    session_id,
                    phase=attempt_phase,
                    progress=attempt_progress,
                    message=f"Attempt {attempt} for {design.name} ({execution.status.value})",
                    metadata={
                        "node_id": attempt_node_id,
                        "parent_id": idea.node_id,
                        "node_type": "step",
                        "attempt": attempt,
                        "round": experiment_round,
                        "status": execution.status.value,
                        "execution_id": execution.id,
                        "errors": execution.errors[:2] if execution.errors else [],
                    },
                    parent_phase=parent_phase,
                )

            execution.output_data.setdefault("attempt", attempt)
            execution.output_data.setdefault("round", experiment_round)

            if execution.status == ExperimentStatus.COMPLETED:
                await self._archive_successful_execution(
                    execution,
                    design,
                    iteration_number,
                    attempt,
                    session_context,
                )
                return execution, execution_history

            if execution.errors:
                prior_errors.extend(execution.errors)

        return execution_history[-1], execution_history

    async def _archive_successful_execution(
        self,
        execution: ExperimentExecution,
        design: ExperimentDesign,
        iteration_number: int,
        attempt_number: int,
        session_context: Optional[OpenHandsSessionContext],
    ) -> None:
        if not self.preserve_successful_workspaces:
            return
        if not self.openhands_client:
            return

        workspace_id = execution.workspace_id or (
            session_context.workspace_id if session_context else None
        )
        if not workspace_id:
            return

        workspace_path = self.openhands_client.workspace_manager.get_workspace_path(workspace_id)
        if not workspace_path or not workspace_path.exists():
            return

        session_label = execution.session_id or (
            session_context.session_id if session_context else "unknown_session"
        )
        archive_root = self.success_workspace_archive_dir / session_label / design.id
        try:
            archive_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.logger.warning("Unable to prepare archive directory %s: %s", archive_root, exc)
            return

        archive_basename = archive_root / f"{execution.id}_iter{iteration_number}_attempt{attempt_number}"

        try:
            archive_path = await asyncio.to_thread(
                shutil.make_archive,
                str(archive_basename),
                "zip",
                root_dir=str(workspace_path),
            )
            execution.output_data["workspace_archive"] = archive_path
            execution.logs.append(f"Workspace archived to {archive_path}")
        except Exception as exc:
            self.logger.warning(
                "Failed to archive workspace %s for execution %s: %s",
                workspace_id,
                execution.id,
                exc,
            )

    async def _evaluate_idea(
        self,
        idea: ResearchIdea,
        session_id: Optional[str],
        progress_value: float,
    ) -> IdeaEvaluation:
        results_summary = "\n".join(
            [
                f"- {res.conclusions[0]} (confidence {res.confidence_score:.2f})"
                for res in idea.results[:5]
                if res.conclusions
            ]
        ) or "No definitive experimental conclusions yet."

        literature_summary = idea.literature_review.summary if idea.literature_review else "No literature synthesis available."
        code_summary = (
            idea.code_analysis.integration_guide[:400]
            if idea.code_analysis and idea.code_analysis.integration_guide
            else "No implementation notes available."
        )

        evaluation_prompt = f"""
You are a panel of expert reviewers scoring a scientific research idea.

Idea Title: {idea.title}
Summary: {idea.summary}
Objective: {idea.objective}
Plan: {idea.plan}

Literature Context:
{literature_summary}

Code / Implementation Context:
{code_summary}

Experimental Findings So Far:
{results_summary}

Provide a JSON object with the following fields:
- "overall_score": float between 0 and 10 assessing overall promise
- "novelty": float between 0 and 10 (novelty of idea)
- "feasibility": float between 0 and 10 (practical feasibility)
- "impact": float between 0 and 10 (potential impact)
- "clarity": float between 0 and 10 (clarity of approach)
- "strengths": list of 2-4 bullet point strengths
- "weaknesses": list of 2-4 bullet point weaknesses or risks
- "decision": "accept" if worth prioritizing or "reject" otherwise
- "comments": concise reviewer-style paragraph summarizing reasoning

Return JSON only.
"""

        response = await self.llm_client.generate(
            evaluation_prompt,
            max_tokens=4000,  # Increased to prevent truncation of evaluation
            temperature=self.config.get("idea_evaluation_temperature", 0.3),
        )

        parsed = self._safe_parse_json_block(response) or {}

        def _score(value: Any, default: float = 5.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        idea_scores = {
            "novelty": _score(parsed.get("novelty"), 6.0),
            "feasibility": _score(parsed.get("feasibility"), 6.0),
            "impact": _score(parsed.get("impact"), 6.0),
            "clarity": _score(parsed.get("clarity"), 6.0),
        }
        overall_score = _score(parsed.get("overall_score"), sum(idea_scores.values()) / len(idea_scores))
        evaluation = IdeaEvaluation(
            idea_id=idea.id,
            overall_score=overall_score,
            scores=idea_scores,
            decision=str(parsed.get("decision", "accept")).lower(),
            strengths=[str(item).strip() for item in parsed.get("strengths", []) if str(item).strip()],
            weaknesses=[str(item).strip() for item in parsed.get("weaknesses", []) if str(item).strip()],
            comments=str(parsed.get("comments", "")),
        )

        if session_id and idea.node_id:
            await self._log_progress(
                session_id,
                phase=f"{idea.id}_evaluation",
                progress=progress_value,
                message=f"Idea scored: {overall_score:.2f}",
                metadata={
                    "parent_id": idea.node_id,
                    "node_type": "result",
                    "overall_score": overall_score,
                    "scores": evaluation.scores,
                    "decision": evaluation.decision,
                },
                parent_phase=f"idea_{idea.id}",
            )

        return evaluation

    def _select_best_idea(self, ideas: List[ResearchIdea]) -> Optional[ResearchIdea]:
        if not ideas:
            return None
        evaluated = [idea for idea in ideas if idea.evaluation]
        if not evaluated:
            return ideas[0]

        def _idea_sort_key(idea: ResearchIdea) -> Tuple[float, float, float]:
            evaluation = idea.evaluation
            return (
                evaluation.overall_score,
                idea.confidence_score,
                idea.iteration_count,
            )

        evaluated.sort(key=_idea_sort_key, reverse=True)
        return evaluated[0]

    def _compute_disagreement(self, ideas: List[ResearchIdea]) -> float:
        evaluated = [idea for idea in ideas if idea.evaluation]
        if len(evaluated) < 2:
            return 0.0
        scores = sorted(
            (idea.evaluation.overall_score for idea in evaluated if idea.evaluation),
            reverse=True,
        )
        if len(scores) < 2:
            return 0.0
        top, second = scores[0], scores[1]
        if top == 0:
            return 0.0
        diff = abs(top - second) / max(top, 1e-6)
        return float(1.0 - min(diff, 1.0))  # large diff => low disagreement

    def _default_debaters(self) -> List[DebaterConfig]:
        return [
            DebaterConfig(role="proposer", style="thorough", temperature=0.5),
            DebaterConfig(role="critic", style="concise", temperature=0.6),
            DebaterConfig(role="skeptic", style="concise", temperature=0.6),
            DebaterConfig(role="safety", style="conservative", temperature=0.4),
        ]

    def _make_debate_config(self) -> DebateConfig:
        return DebateConfig(
            num_agents=self._debate_defaults.get("num_agents", 4),
            num_rounds=self._debate_defaults.get("num_rounds", 2),
            groups=max(1, self._debate_defaults.get("groups", 1)),
            rubric=DEFAULT_DEBATE_RUBRIC,
        )

    async def _process_idea(
        self,
        idea: ResearchIdea,
        research_question: str,
        session_id: Optional[str],
        semaphore: asyncio.Semaphore,
        idx: int,
        total: int,
        include_literature: bool,
        include_code: bool,
        enable_iteration: bool,
    ) -> ResearchIdea:
        async with semaphore:
            progress_window = 45.0 / max(1, total)
            base_progress = 22.0 + progress_window * idx

            openhands_context: Optional[OpenHandsSessionContext] = None
            try:
                openhands_context = await self._acquire_openhands_session(
                    research_session_id=session_id,
                    idea_id=idea.id,
                    resource_requirements=self.default_openhands_resources,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to acquire OpenHands session for idea %s: %s",
                    idea.id,
                    exc,
                )
                openhands_context = None

            try:
                literature_result, code_result = await self._gather_idea_context(
                    idea,
                    session_id,
                    base_progress,
                    progress_window,
                    include_literature,
                    include_code,
                )
                idea.literature_review = literature_result
                idea.code_analysis = code_result

                literature_context = literature_result.summary if literature_result else None
                code_context = code_result.integration_guide if code_result else None

                hypotheses = await self.hypothesis_generator.generate_hypotheses(
                    research_question,
                    literature_context=literature_context,
                    code_context=code_context,
                )

                if not hypotheses:
                    hypotheses = [
                        ResearchHypothesis(
                            id=f"hyp_{uuid.uuid4().hex[:8]}",
                            statement=f"Evaluate core assumption behind {idea.title}",
                            reasoning="Fallback hypothesis when generation fails",
                            testable_predictions=["Primary metric improves"],
                            success_criteria={"confidence_threshold": 0.6},
                            variables={"independent": ["intervention"], "dependent": ["metric"]},
                        )
                    ]

                idea.hypotheses = hypotheses

                if session_id and idea.node_id:
                    for h_idx, hypothesis in enumerate(hypotheses, start=1):
                        await self._log_progress(
                            session_id,
                            phase=f"{idea.id}_hypothesis_{h_idx}",
                            progress=base_progress + progress_window * 0.35,
                            message=f"Hypothesis {h_idx}",
                            metadata={
                                "parent_id": idea.node_id,
                                "node_type": "result",
                                "title": hypothesis.statement,
                                "reasoning": hypothesis.reasoning,
                            },
                            parent_phase=f"idea_{idea.id}",
                        )

                max_iterations = max(1, self.max_iterations) if enable_iteration else 1
                confidence, iterations = await self._run_experiments_for_idea(
                    idea,
                    session_id,
                    base_progress + progress_window * 0.4,
                    progress_window * 0.45,
                    max_iterations,
                    openhands_context,
                )

                idea.confidence_score = confidence
                idea.iteration_count = iterations

                idea.evaluation = await self._evaluate_idea(
                    idea,
                    session_id,
                    base_progress + progress_window * 0.95,
                )
                return idea
            finally:
                if openhands_context:
                    await self._release_openhands_session(openhands_context)

    async def conduct_research(
        self,
        research_question: str,
        include_literature_review: bool = True,
        include_code_analysis: bool = True,
        enable_iteration: bool = True,
        session_id: Optional[str] = None
    ) -> ScientificResearchResult:
        """Conduct comprehensive scientific research

        This is the main orchestration method that coordinates all research phases:
        1. Literature Review (Deep Research Engine)
        2. Code Discovery & Analysis (Code Research Engine)
        3. Hypothesis Generation & Experimental Design
        4. Iterative Experimentation with Feedback Loops
        5. Analysis & Validation
        6. Synthesis & Documentation

        Args:
            research_question: Research question to investigate
            include_literature_review: Whether to conduct literature review
            include_code_analysis: Whether to analyze existing code implementations
            enable_iteration: Whether to enable iterative refinement

        Returns:
            Comprehensive scientific research result
        """
        include_code_analysis = False
        include_literature_review = False
        
        research_id = f"research_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Starting scientific research: {research_question}")

        root_id = self._get_root_node_id(session_id)

        await self._log_progress(
            session_id,
            phase="initializing",
            progress=5.0,
            message="Planning multi-engine scientific workflow",
            metadata={"query": research_question, "parent_id": root_id}
        )

        # Initialize result structure
        result = ScientificResearchResult(
            research_id=research_id,
            query=research_question,
            literature_review=None,
            code_analysis=None,
            hypotheses=[],
            experiments=[],
            executions=[],
            results=[],
            synthesis={},
            final_conclusions=[],
            confidence_score=0.0,
            reproducibility_report={},
            publication_draft="",
            iteration_count=0,
            recommendations=[],
            debates=[]
        )

        enable_iteration = bool(enable_iteration and self.config.get("enable_iteration", True))

        try:
            self.logger.info("Phase 1: Generating candidate research ideas")

            ideas = await self._generate_research_ideas(research_question, session_id)
            result.ideas = ideas

            idea_count = len(ideas)
            semaphore = asyncio.Semaphore(self._max_parallel_ideas(idea_count))

            processed_ideas = await asyncio.gather(
                *[
                    self._process_idea(
                        idea,
                        research_question,
                        session_id,
                        semaphore,
                        idx,
                        idea_count,
                        include_literature_review,
                        include_code_analysis,
                        enable_iteration,
                    )
                    for idx, idea in enumerate(ideas)
                ]
            )

            result.ideas = processed_ideas
            result.idea_evaluations = {
                idea.id: idea.evaluation
                for idea in processed_ideas
                if idea.evaluation is not None
            }

            ranking = sorted(
                processed_ideas,
                key=lambda idea: (
                    idea.evaluation.overall_score if idea.evaluation else 0.0,
                    idea.confidence_score,
                    idea.iteration_count,
                ),
                reverse=True,
            )

            debate_record: Optional[Dict[str, Any]] = None
            if self.debate_manager and self._debate_enabled and ranking:
                top_score = ranking[0].evaluation.overall_score if ranking[0].evaluation else 0.0
                signals = {
                    "confidence": float(top_score) / 10.0 if top_score else 0.0,
                    "stakes": "high",
                    "disagreement_score": self._compute_disagreement(ranking),
                    "rl_uncertainty": 0.0,
                }
                if should_debate(signals, self._debate_policy):
                    topic = f"Select the best research idea for: {research_question}"
                    context_payload = {
                        "ideas": [
                            {
                                "id": idea.id,
                                "title": idea.title,
                                "summary": idea.summary,
                                "overall_score": idea.evaluation.overall_score if idea.evaluation else None,
                                "confidence": idea.confidence_score,
                            }
                            for idea in ranking
                        ],
                    }
                    debate_cfg = self._make_debate_config()
                    debaters = self._default_debaters()
                    try:
                        debate_record = await self.debate_manager.run(
                            topic,
                            context_payload,
                            debate_cfg,
                            debaters,
                            session_id=session_id,
                        )
                        if result:
                            result.debates.append(debate_record)
                    except Exception as exc:
                        self.logger.warning("Debate execution failed: %s", exc)

            if session_id:
                await self._log_progress(
                    session_id,
                    phase="idea_evaluation_summary",
                    progress=82.0,
                    message="Idea evaluations completed",
                    metadata={
                        "parent_id": self._session_phase_nodes.get(session_id, {}).get("ideation"),
                        "node_type": "analysis",
                        "ranking": [
                            {
                                "idea_id": idea.id,
                                "title": idea.title,
                                "overall_score": idea.evaluation.overall_score if idea.evaluation else None,
                                "confidence": idea.confidence_score,
                            }
                            for idea in ranking
                        ],
                    },
                    parent_phase="ideation",
                )

            best_idea = self._select_best_idea(processed_ideas)
            if not best_idea and processed_ideas:
                best_idea = processed_ideas[0]

            if best_idea:
                result.selected_idea_id = best_idea.id
                result.literature_review = best_idea.literature_review
                result.code_analysis = best_idea.code_analysis
                result.hypotheses = best_idea.hypotheses
                result.experiments = best_idea.experiments
                result.executions = best_idea.executions
                result.results = best_idea.results
                result.iteration_count = best_idea.iteration_count
                result.confidence_score = best_idea.confidence_score

                if not result.hypotheses:
                    raise RuntimeError(
                        "INSUFFICIENT_HYPOTHESES: the selected idea did not produce any testable hypotheses"
                    )

                if not result.experiments:
                    raise RuntimeError(
                        "INSUFFICIENT_EXPERIMENTS: no experiments were designed for the selected idea"
                    )

                if debate_record:
                    verdict = debate_record.get("verdict", {})
                    scores = verdict.get("scores", {}) or {}
                    if scores:
                        score_avg = sum(float(v) for v in scores.values() if isinstance(v, (int, float)))
                        score_avg /= max(len(scores), 1)
                        result.confidence_score = max(result.confidence_score, min(0.95, score_avg))
                    recommendation = str(verdict.get("recommendation", "")).lower()
                    if recommendation == "reject":
                        result.confidence_score = min(result.confidence_score, 0.3)

                if session_id and best_idea.node_id:
                    await self._log_progress(
                        session_id,
                        phase="idea_selection",
                        progress=84.0,
                        message=f"Selected best idea: {best_idea.title}",
                        metadata={
                            "parent_id": self._session_phase_nodes.get(session_id, {}).get("ideation"),
                            "node_type": "result",
                            "selected_idea": best_idea.title,
                            "overall_score": best_idea.evaluation.overall_score if best_idea.evaluation else None,
                            "confidence": best_idea.confidence_score,
                        },
                        parent_phase="ideation",
                    )

            # Phase 2: Synthesis and Final Analysis based on selected idea
            self.logger.info("Phase 2: Synthesizing results and generating conclusions")

            await self._log_progress(
                session_id,
                phase="synthesis",
                progress=88.0,
                message="Synthesizing research findings",
                metadata={
                    "executions": len(result.executions),
                    "results": len(result.results),
                    "selected_idea": result.selected_idea_id,
                },
                parent_phase="ideation",
            )

            result.synthesis = await self._synthesize_research_results(result)
            result.reproducibility_report = await self._generate_reproducibility_report(result)
            result.publication_draft = await self._generate_publication_draft(result)
            result.recommendations = await self._generate_research_recommendations(result)

            successful_results = [
                res
                for res in result.results
                if res.status == ExperimentStatus.COMPLETED and res.data.get("success")
            ]

            if not result.results:
                raise RuntimeError(
                    "INSUFFICIENT_RESULTS: no experiment executions were recorded"
                )

            if not successful_results:
                raise RuntimeError(
                    "INSUFFICIENT_EVIDENCE: no completed experiments produced success metrics"
                )

            if successful_results:
                result.final_conclusions = result.synthesis.get("conclusions", [])

                if session_id and result.final_conclusions:
                    for idx, conclusion in enumerate(result.final_conclusions[:5]):
                        await self._log_progress(
                            session_id,
                            phase=f"conclusion_{idx + 1}",
                            progress=94.0,
                            message=f"Conclusion {idx + 1}",
                            metadata={
                                "title": conclusion,
                                "parent_id": self._session_phase_nodes.get(session_id, {}).get("synthesis"),
                            },
                            parent_phase="synthesis",
                            node_type="result",
                        )
                if self.memory_store and result.final_conclusions:
                    await self.memory_store.add_semantic(
                        "\n".join(result.final_conclusions[:3]),
                        importance=0.65,
                        tags={"research_id": research_id, "scope": "scientific_research"},
                    )
            else:
                result.final_conclusions = []

            if not successful_results:
                result.confidence_score = min(result.confidence_score, 0.2)

            self.logger.info("Scientific research completed: %s", research_id)

            await self._log_progress(
                session_id,
                phase="synthesis",
                progress=96.0,
                message="Final analysis prepared",
                metadata={
                    "conclusions": len(result.final_conclusions),
                    "recommendations": len(result.recommendations),
                    "selected_idea": result.selected_idea_id,
                },
                parent_phase="ideation",
            )

            if session_id:
                summary_text = result.synthesis.get("overall_findings") if result.synthesis else "Scientific research completed"
                await progress_tracker.log_research_completed(
                    session_id=session_id,
                    engine="scientific_research",
                    result_summary=summary_text or "Scientific research completed",
                    metadata={
                        "selected_idea": result.selected_idea_id,
                        "idea_rankings": [
                            {
                                "idea_id": idea.id,
                                "title": idea.title,
                                "overall_score": idea.evaluation.overall_score if idea.evaluation else None,
                                "confidence": idea.confidence_score,
                            }
                            for idea in ranking
                        ],
                    },
                )
            else:
                failure_message = "All experimental attempts failed; research remains incomplete."
                result.final_conclusions = [failure_message]
                result.confidence_score = 0.0
                result.recommendations = [
                    "Resolve programming errors reported during OpenHands execution",
                    "Re-run data collection after fixing scripts",
                    "Increase max_attempts_per_experiment if failures persist",
                ]
                result.synthesis = {
                    "overall_findings": failure_message,
                    "evidence_strength": "No empirical evidence collected",
                    "conclusions": result.final_conclusions,
                    "limitations": [
                        "All OpenHands execution attempts failed",
                        "No query runtime data or ML model produced",
                    ],
                    "future_research": result.recommendations,
                    "confidence_assessment": "Very low",
                }

                if session_id:
                    await self._log_progress(
                        session_id,
                        phase="synthesis",
                        progress=92.0,
                        message="Research incomplete – experiments did not succeed",
                        metadata={
                            "selected_idea": result.selected_idea_id,
                            "failed_experiments": len(result.executions),
                        },
                        parent_phase="ideation",
                    )

        except Exception as e:
            self.logger.error(f"Error in scientific research: {e}")
            result.final_conclusions = [f"Research encountered error: {str(e)}"]
            result.confidence_score = 0.1

        finally:
            if session_id:
                self._session_phase_nodes.pop(session_id, None)
                self._session_root_nodes.pop(session_id, None)
            await self._release_all_openhands_sessions(session_id)

        return result

    async def _analyze_experiment_result(
        self,
        hypothesis: ResearchHypothesis,
        design: ExperimentDesign,
        execution: ExperimentExecution
    ) -> ExperimentResult:
        """Analyze experiment result and determine hypothesis support"""

        analysis_data = execution.intermediate_results or {}
        statistical_tests = analysis_data.get("statistical_tests", {})

        # If the execution failed or produced no analysable output, surface that fact directly.
        if execution.status != ExperimentStatus.COMPLETED:
            primary_error = execution.errors[0] if execution.errors else "Experiment did not complete"
            return ExperimentResult(
                execution_id=execution.id,
                hypothesis_id=hypothesis.id,
                status=execution.status,
                data=execution.output_data,
                analysis=analysis_data,
                statistical_tests=statistical_tests,
                visualization_data={},
                conclusions=[f"Experiment failed: {primary_error}"],
                confidence_score=0.0,
                reproducibility_score=0.0,
                limitations=["Execution failed", "No empirical evidence collected"],
                next_steps=[
                    "Inspect execution logs and errors",
                    "Fix environment or code issues before re-running",
                ],
            )

        # Determine if hypothesis is supported when experiment completed successfully
        p_value = statistical_tests.get("t_test", {}).get("p_value", 1.0)
        effect_size = analysis_data.get("effect_size", 0.0)

        conclusions = []
        confidence_score = 0.0

        if p_value < 0.05 and effect_size > 0.2:
            conclusions.append("Hypothesis is supported by experimental evidence")
            confidence_score = 0.8
        elif p_value < 0.1:
            conclusions.append("Hypothesis has marginal support")
            confidence_score = 0.6
        else:
            conclusions.append("Hypothesis is not supported by current evidence")
            confidence_score = 0.3

        limitations = analysis_data.get("limitations", [])
        if not limitations:
            limitations = ["Limited sample size"]

        return ExperimentResult(
            execution_id=execution.id,
            hypothesis_id=hypothesis.id,
            status=execution.status,
            data=execution.output_data,
            analysis=analysis_data,
            statistical_tests=statistical_tests,
            visualization_data={},  # Would include plots in production
            conclusions=conclusions,
            confidence_score=confidence_score,
            reproducibility_score=0.8 if confidence_score >= 0.6 else 0.6,
            limitations=limitations,
            next_steps=["Increase sample size", "Test additional conditions"],
        )

    async def _update_hypothesis_status(self, hypothesis: ResearchHypothesis, result: ExperimentResult):
        """Update hypothesis status based on experimental result"""

        if result.confidence_score >= 0.7:
            hypothesis.status = HypothesisStatus.SUPPORTED
            hypothesis.confidence_level = result.confidence_score
        elif result.confidence_score <= 0.4:
            hypothesis.status = HypothesisStatus.REJECTED
            hypothesis.confidence_level = 1.0 - result.confidence_score
        else:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            hypothesis.confidence_level = result.confidence_score

        hypothesis.evidence.extend(result.conclusions)

    async def _refine_hypotheses_for_next_iteration(self, hypotheses: List[ResearchHypothesis], iteration_results: List[ExperimentResult]):
        """Refine hypotheses based on experimental results"""

        for hypothesis in hypotheses:
            if hypothesis.status == HypothesisStatus.INCONCLUSIVE:
                # Find relevant experimental results
                relevant_results = [r for r in iteration_results if r.hypothesis_id == hypothesis.id]

                if relevant_results:
                    feedback = f"Previous experiment results: {relevant_results[0].conclusions}"
                    experimental_data = relevant_results[0].analysis

                    # Refine hypothesis
                    refined_hypothesis = await self.hypothesis_generator.refine_hypothesis(
                        hypothesis,
                        feedback,
                        experimental_data
                    )

                    # Update hypothesis in place
                    hypothesis.statement = refined_hypothesis.statement
                    hypothesis.reasoning = refined_hypothesis.reasoning
                    hypothesis.testable_predictions = refined_hypothesis.testable_predictions
                    hypothesis.success_criteria = refined_hypothesis.success_criteria
                    hypothesis.variables = refined_hypothesis.variables

    async def _synthesize_research_results(self, result: ScientificResearchResult) -> Dict[str, Any]:
        """Synthesize all research results into comprehensive analysis"""

        successful_executions = [
            exec_record
            for exec_record in result.executions
            if exec_record.status == ExperimentStatus.COMPLETED
            and bool(exec_record.output_data.get("success"))
        ]
        successful_results = [
            res
            for res in result.results
            if res.status == ExperimentStatus.COMPLETED
            and res.data.get("success")
        ]

        if not successful_executions and not successful_results:
            self.logger.info(
                "No successful experiments with verifiable artifacts for research %s",
                result.research_id,
            )
            return {
                "overall_findings": "No experiments produced verifiable artifacts or successful command executions. The research question remains unanswered.",
                "evidence_strength": "None",
                "conclusions": [
                    "Insufficient evidence gathered; rerun experiments with proper instrumentation and artifact collection."
                ],
                "implications": [
                    "Collect and verify build/run artifacts before attempting synthesis.",
                    "Ensure computational steps execute successfully (rc=0) and persist outputs under the workspace.",
                ],
                "limitations": [
                    "All attempts failed or produced no artifacts",
                ],
                "future_research": [
                    "Review execution logs, fix errors, and rerun with stricter assertions.",
                ],
                "confidence_assessment": "Very low",
            }

        idea_lines = []
        for idea in result.ideas:
            evaluation = result.idea_evaluations.get(idea.id)
            score_display = evaluation.overall_score if evaluation else "N/A"
            decision_display = evaluation.decision if evaluation else "unknown"
            idea_lines.append(
                f"- {idea.title} (id: {idea.id}, overall score: {score_display}, decision: {decision_display}, confidence: {idea.confidence_score:.2f})"
            )

        idea_summary = "\n".join(idea_lines) if idea_lines else "- No alternative ideas evaluated"

        execution_lines = []
        for exec_record in result.executions:
            status_value = getattr(exec_record.status, "value", str(exec_record.status))
            success_flag = exec_record.output_data.get("success")
            error_snippet = "; ".join(exec_record.errors[:2]) if exec_record.errors else "None"
            files_created = exec_record.output_data.get("files_created")
            workspace_id = exec_record.output_data.get("workspace_id")
            execution_lines.append(
                f"- Execution {exec_record.id} (design {exec_record.design_id}): status={status_value}, success={success_flag}, errors={error_snippet}, files_created={files_created}, workspace={workspace_id}"
            )

        execution_summary = "\n".join(execution_lines) if execution_lines else "- No experiments were executed"

        result_lines = []
        for res in result.results:
            status_value = getattr(res.status, "value", str(res.status))
            conclusion_text = res.conclusions[0] if res.conclusions else "No conclusion"
            result_lines.append(
                f"- Result from execution {res.execution_id}: status={status_value}, confidence={res.confidence_score:.2f}, conclusion={conclusion_text}"
            )
        result_summary = "\n".join(result_lines) if result_lines else "- No experiment results available"

        failure_notes = []
        for exec_record in result.executions:
            if exec_record.status != ExperimentStatus.COMPLETED:
                failure_notes.append(
                    f"Execution {exec_record.id} failed with errors: {', '.join(exec_record.errors) or 'unknown error'}"
                )
        failure_summary = "\n".join(failure_notes) if failure_notes else "None"

        synthesis_prompt = f"""
        Synthesize the following scientific research results. Use only the facts provided. Do not claim that repository source code was modified or that experiments succeeded unless the execution summary explicitly confirms it.

        Research Question: {result.query}
        Hypotheses Tested: {len(result.hypotheses)}
        Experiments Conducted: {len(result.experiments)}
        Iterations Completed: {result.iteration_count}

        Selected Idea ID: {result.selected_idea_id}
        Idea Evaluations:
{idea_summary}

        Literature Review Available: {result.literature_review is not None}
        Code Analysis Available: {result.code_analysis is not None}

        Execution Summary:
{execution_summary}

        Experiment Result Summary:
{result_summary}

        Recorded Execution Failures:
{failure_summary}

        Provide comprehensive synthesis in JSON format with:
        1. "overall_findings": Summary grounded strictly in the execution and result summaries
        2. "evidence_strength": Assessment of evidence quality and strength given the actual outcomes
        3. "conclusions": Final research conclusions (state explicitly if the research question remains unanswered)
        4. "implications": Practical and theoretical implications, making clear when evidence is absent
        5. "limitations": Study limitations and encountered errors or gaps
        6. "future_research": Recommendations for future research directions to obtain real evidence
        7. "confidence_assessment": Overall confidence in findings based on executed experiments

        Respond with valid JSON only. If experiments failed or produced no data, make that explicit in the findings and conclusions.
        """

        try:
            response = await self.llm_client.generate(synthesis_prompt)

            import json, re
            parsed = None
            try:
                parsed = json.loads(response)
            except Exception:
                match = re.search(r"\{[\s\S]*\}\s*$", str(response))
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        parsed = None
            if parsed is not None:
                return parsed

            raise ValueError("LLM synthesis response was not valid JSON")

        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")

            completed_executions = [
                exec_record
                for exec_record in result.executions
                if exec_record.status == ExperimentStatus.COMPLETED
            ]
            successful_results = [
                res
                for res in result.results
                if res.status == ExperimentStatus.COMPLETED and res.data.get("success")
            ]

            fallback_overview = (
                "Automated synthesis unavailable; summarising recorded execution results."
                if successful_results or completed_executions
                else "Automated synthesis unavailable; experiments did not yield parseable results."
            )

            conclusions: List[str] = []
            for res in successful_results:
                if res.conclusions:
                    conclusions.extend(res.conclusions[:1])
            if not conclusions:
                if completed_executions:
                    conclusions.append(
                        "Experiments executed successfully, but narrative synthesis failed; review execution summary for details."
                    )
                else:
                    conclusions.append(
                        "Research question remains unanswered because experiments failed to complete."
                    )

            implications = [
                "Execution summary:\n" + execution_summary,
                "Experiment result summary:\n" + result_summary,
            ]

            limitations = failure_notes or ["Synthesis LLM failed to produce JSON output"]

            future_research = [
                "Address synthesis pipeline failure so structured results can be generated automatically",
            ]
            if not successful_results:
                future_research.append("Collect additional experimental evidence before drawing conclusions")

            confidence_label = (
                "Moderate" if successful_results else "Low"
            )

            return {
                "overall_findings": fallback_overview,
                "evidence_strength": "See execution summary",
                "conclusions": conclusions,
                "implications": implications,
                "limitations": limitations,
                "future_research": future_research,
                "confidence_assessment": confidence_label,
            }

    async def _generate_reproducibility_report(self, result: ScientificResearchResult) -> Dict[str, Any]:
        """Generate reproducibility report"""

        total_executions = len(result.executions)
        successful_executions = [
            exec_record
            for exec_record in result.executions
            if exec_record.status == ExperimentStatus.COMPLETED
            and exec_record.output_data.get("success", True)
        ]
        failed_executions = [
            exec_record for exec_record in result.executions if exec_record.status != ExperimentStatus.COMPLETED
        ]

        reproducibility_score = 0.0
        if total_executions:
            reproducibility_score = round(len(successful_executions) / total_executions, 2)

        code_available = bool(
            (result.code_analysis and result.code_analysis.repositories)
            or any(exec_record.output_data.get("files_created") for exec_record in successful_executions)
        )

        data_accessible = any(
            exec_record.output_data.get("files_created") or exec_record.output_data.get("data_points")
            for exec_record in successful_executions
        )

        statistical_methods_clear = any(
            res.analysis.get("statistical_tests")
            for res in result.results
            if res.status == ExperimentStatus.COMPLETED
        )

        recommendations = []
        if not successful_executions:
            recommendations.append("Re-run experiments after resolving execution failures")
        if failed_executions:
            recommendations.append("Investigate and document failure modes before drawing conclusions")
        if not code_available:
            recommendations.append("Publish experiment code or logs for reproducibility")
        if not data_accessible:
            recommendations.append("Persist raw datasets and link them in the report")
        if not statistical_methods_clear:
            recommendations.append("Provide concrete statistical analysis or justify its absence")

        if not recommendations:
            recommendations.append("Maintain detailed experiment logs for future replications")

        return {
            "methodology_documented": bool(result.experiments),
            "code_available": code_available,
            "data_accessible": data_accessible,
            "statistical_methods_clear": statistical_methods_clear,
            "reproducibility_score": reproducibility_score,
            "recommendations": recommendations,
        }

    async def _generate_publication_draft(self, result: ScientificResearchResult) -> str:
        """Generate academic publication draft"""

        execution_lines = []
        failed_execution_lines = []
        for exec_record in result.executions:
            status_value = getattr(exec_record.status, "value", str(exec_record.status))
            success_flag = exec_record.output_data.get("success")
            error_snippet = "; ".join(exec_record.errors[:2]) if exec_record.errors else "None"
            line = f"- Execution {exec_record.id}: status={status_value}, success={success_flag}, errors={error_snippet}"
            execution_lines.append(line)
            if exec_record.status != ExperimentStatus.COMPLETED or success_flag is False:
                failed_execution_lines.append(line)
        execution_summary = "\n".join(execution_lines) if execution_lines else "- No experiments executed"

        successful_results = [res for res in result.results if res.status == ExperimentStatus.COMPLETED]
        failure_notes = failed_execution_lines

        draft_prompt = f"""
        Generate an academic publication draft for the following research. The draft must stay faithful to the provided execution evidence. If experiments failed or produced no usable data, state that explicitly instead of fabricating results.

        Title: Research on {result.query}
        Hypotheses: {len(result.hypotheses)} tested
        Experiments Designed: {len(result.experiments)}
        Successful Executions: {len(successful_results)}
        Overall Confidence Score: {result.confidence_score}

        Synthesis Summary: {result.synthesis.get('overall_findings', 'No synthesis available')}
        Execution Evidence:
{execution_summary}

        Confirmed Failures or Issues:
{chr(10).join(failure_notes) if failure_notes else 'None recorded'}

        Structure the publication with:
        1. Abstract (150-200 words) that accurately reflects executed work and its limitations
        2. Introduction and Background
        3. Methodology (describe attempted steps; highlight missing or failed components)
        4. Results (state clearly when no empirical results were obtained)
        5. Discussion
        6. Conclusions (make clear if the research question remains open)
        7. Future Work (focus on steps required to obtain real evidence)

        Write in academic style suitable for peer review. Do not claim code modifications, data collection, or successful integrations unless they are explicitly confirmed in the execution evidence above.
        """

        try:
            publication_draft = await self.llm_client.generate(draft_prompt)
            return publication_draft
        except Exception as e:
            self.logger.error(f"Error generating publication draft: {e}")
            successful_runs = [res for res in result.results if res.status == ExperimentStatus.COMPLETED]
            return (
                f"# Research Publication Draft\n\n## {result.query}\n\n"
                f"Experiments attempted: {len(result.executions)} (successful: {len(successful_runs)}).\n"
                "Automated drafting failed; please review execution logs and rerun once empirical data is available."
            )

    async def _generate_research_recommendations(self, result: ScientificResearchResult) -> List[str]:
        """Generate actionable research recommendations"""

        recommendations = []
        successful_results = [res for res in result.results if res.status == ExperimentStatus.COMPLETED]
        if not successful_results:
            recommendations.append("Resolve execution failures and rerun experiments to gather evidence")

        # Based on confidence score
        if result.confidence_score >= 0.8:
            recommendations.append("Results are strong enough for practical application")
        elif result.confidence_score >= 0.6:
            recommendations.append("Results show promise but need additional validation")
        else:
            recommendations.append("Preliminary results require significant additional research")

        # Based on iteration count
        if result.iteration_count >= 3:
            recommendations.append("Thorough iterative research conducted")
        else:
            recommendations.append("Consider additional research iterations")

        # Based on multi-engine integration
        if result.literature_review and result.code_analysis:
            recommendations.append("Comprehensive multi-source research completed")

        return recommendations

    def _calculate_overall_confidence(self, results: List[ExperimentResult]) -> float:
        """Calculate overall confidence across all experimental results"""

        if not results:
            return 0.0

        successful_results = [res for res in results if res.status == ExperimentStatus.COMPLETED]
        if not successful_results:
            return 0.0

        total_confidence = sum(res.confidence_score for res in successful_results)
        avg_confidence = total_confidence / len(successful_results)

        consistency_bonus = 0.1 if len(successful_results) > 1 else 0.0

        return min(avg_confidence + consistency_bonus, 1.0)

if GEPAOptimizer is not None:
    import dspy  # type: ignore

    class IdeaGenerationProgram(dspy.Module):
        """DSPy module wrapping idea generation prompt for GEPA optimisation."""

        def __init__(self, llm_client: LLMClient, max_ideas: int, template: str, temperature: float = 0.6):
            super().__init__()
            self.llm_client = llm_client
            self.max_ideas = max(1, max_ideas)
            self.base_template = template
            self.temperature = temperature
            self.prompt_param = dspy.Parameter(template)

        def forward(self, sample: Any) -> Dict[str, Any]:
            question = getattr(sample, "question", None)
            if question is None and isinstance(sample, dict):
                question = sample.get("question")
            if question is None:
                question = str(sample)

            template = self.prompt_param.value or self.base_template
            prompt_text = template.format(max_ideas=self.max_ideas, research_question=question)
            response = asyncio.run(
                self.llm_client.generate(
                    prompt_text,
                    max_tokens=6000,  # Increased to prevent truncation of complex hypotheses
                    temperature=self.temperature,
                )
            )
            idea_dicts = ScientificResearchEngine._idea_response_to_dicts(response, self.max_ideas)
            return {"raw": response, "prompt": prompt_text, "ideas": idea_dicts}
from ..exec.ras import ResearchActionSpec
from ..exec.ras_executor import RASExecutor, RASExecutionError
from ..exec.ras_validator import RASValidationError, validate_research_action_spec
