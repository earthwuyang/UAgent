"""Smart Router for classifying and routing user requests to appropriate research engines"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

# Exceptions defined at end of file


class EngineType(str, Enum):
    """Research engine types"""
    DEEP_RESEARCH = "DEEP_RESEARCH"
    SCIENTIFIC_RESEARCH = "SCIENTIFIC_RESEARCH"
    CODE_RESEARCH = "CODE_RESEARCH"


@dataclass
class ClassificationRequest:
    """Request for classification and routing"""
    user_request: str
    context: Optional[Dict[str, Any]] = None
    override_engine: Optional[str] = None
    confidence_threshold: float = 0.7


@dataclass
class ClassificationResult:
    """Result of classification and routing"""
    primary_engine: str
    confidence_score: float
    sub_components: Dict[str, bool]
    reasoning: str
    workflow_plan: Dict[str, Any]


class SmartRouter:
    """Intelligent request classifier and router"""

    def __init__(self, llm_client, cache=None, config: Optional[Dict[str, Any]] = None):
        """Initialize smart router

        Args:
            llm_client: LLM client for classification
            cache: Cache client for storing results
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.cache = cache
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.cache_ttl = self.config.get("cache_ttl", 86400)  # 24 hours

        # Classification prompt
        self.classification_prompt = """
        Analyze the user request and classify it into one of these categories.
        Note: Scientific research is the most complex and may include elements
        of both deep research and code research.

        1. DEEP_RESEARCH: General information gathering, literature reviews,
           market analysis, comprehensive topic exploration, fact-finding

        2. SCIENTIFIC_RESEARCH: **MOST COMPLEX** - Hypothesis testing, experimental
           design, data analysis, research methodology. Often includes:
           - Literature review (deep research component)
           - Code implementation and testing (code research component)
           - Iterative experimentation with feedback loops
           - Statistical analysis and validation
           - Paper writing and peer review

        3. CODE_RESEARCH: Repository analysis, code understanding,
           implementation patterns, technical documentation, best practices

        Classification Priority:
        - If request involves hypothesis, experiments, or scientific methodology → SCIENTIFIC_RESEARCH
        - If request is purely about finding/understanding existing code → CODE_RESEARCH
        - If request is general information gathering without experimentation → DEEP_RESEARCH

        Examples:
        - "Research the latest trends in AI" → DEEP_RESEARCH
        - "Design and run experiments to test if attention mechanisms improve transformer performance" → SCIENTIFIC_RESEARCH
        - "Find transformer implementations and test a new attention mechanism hypothesis" → SCIENTIFIC_RESEARCH
        - "Study how different transformer architectures work" → SCIENTIFIC_RESEARCH (involves experimentation)
        - "Find and analyze open source implementations of transformers" → CODE_RESEARCH
        - "What are the best practices for implementing transformers?" → CODE_RESEARCH

        Respond with JSON format:
        {
            "engine": "DEEP_RESEARCH|SCIENTIFIC_RESEARCH|CODE_RESEARCH",
            "confidence_score": 0.0-1.0,
            "reasoning": "explanation of classification decision",
            "sub_components": {
                "deep_research": boolean,
                "code_research": boolean,
                "experimentation": boolean,
                "iteration": boolean
            }
        }
        """

    async def classify_and_route(self, request: ClassificationRequest) -> ClassificationResult:
        """Classify user request and generate routing plan

        Args:
            request: Classification request

        Returns:
            Classification result with routing plan

        Raises:
            InvalidRequestError: If request is invalid
            ThresholdError: If confidence is below threshold
            ClassificationError: If classification fails
        """
        # Validate request
        if not request.user_request or not request.user_request.strip():
            raise InvalidRequestError("User request cannot be empty")

        # Handle manual override
        if request.override_engine:
            return await self._handle_override(request)

        # Check cache
        cache_key = self._generate_cache_key(request.user_request)
        if self.cache:
            try:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    self.logger.info(f"Cache hit for request: {request.user_request[:50]}...")
                    return ClassificationResult(**cached_result)
            except Exception as e:
                self.logger.warning(f"Cache get error: {e}")
                # Continue without cache

        # Perform classification
        classification_data = await self._classify_with_retry(request)

        # Validate confidence
        if classification_data["confidence_score"] < request.confidence_threshold:
            raise ThresholdError(
                f"Classification confidence {classification_data['confidence_score']} "
                f"below threshold {request.confidence_threshold}"
            )

        # Generate workflow plan
        workflow_plan = await self._generate_workflow_plan(
            classification_data["engine"],
            request.user_request,
            classification_data["sub_components"]
        )

        # Create result
        result = ClassificationResult(
            primary_engine=classification_data["engine"],
            confidence_score=classification_data["confidence_score"],
            sub_components=classification_data["sub_components"],
            reasoning=classification_data["reasoning"],
            workflow_plan=workflow_plan
        )

        # Cache result
        if self.cache:
            try:
                await self.cache.set(cache_key, asdict(result), ttl=self.cache_ttl)
            except Exception as e:
                self.logger.warning(f"Cache set error: {e}")
                # Continue without caching

        self.logger.info(
            f"Classified request as {result.primary_engine} "
            f"with confidence {result.confidence_score}"
        )

        return result

    async def _handle_override(self, request: ClassificationRequest) -> ClassificationResult:
        """Handle manual engine override"""
        engine = request.override_engine.upper()
        if engine not in [e.value for e in EngineType]:
            raise InvalidRequestError(f"Invalid override engine: {request.override_engine}")

        # Generate basic workflow plan for override
        workflow_plan = await self._generate_workflow_plan(engine, request.user_request, {})

        return ClassificationResult(
            primary_engine=engine,
            confidence_score=1.0,  # Manual override has full confidence
            sub_components={"manual_override": True},
            reasoning=f"Manual override to {engine}",
            workflow_plan=workflow_plan
        )

    async def _classify_with_retry(self, request: ClassificationRequest) -> Dict[str, Any]:
        """Classify request with retry logic"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Classification attempt {attempt + 1}")

                # Call LLM for classification
                response = await self.llm_client.classify(
                    request.user_request,
                    self.classification_prompt
                )

                # Validate response format
                required_keys = ["engine", "confidence_score", "reasoning", "sub_components"]
                if not all(key in response for key in required_keys):
                    raise ValueError(f"Invalid LLM response format: {response}")

                return response

            except Exception as e:
                last_error = e
                self.logger.warning(f"Classification attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)

        raise ClassificationError(f"Classification failed after {self.max_retries} attempts: {last_error}")

    async def _generate_workflow_plan(
        self,
        engine: str,
        request: str,
        sub_components: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Generate workflow plan based on classification"""

        if engine == EngineType.SCIENTIFIC_RESEARCH:
            return await self._generate_scientific_workflow(request, sub_components)
        elif engine == EngineType.DEEP_RESEARCH:
            return await self._generate_deep_research_workflow(request, sub_components)
        elif engine == EngineType.CODE_RESEARCH:
            return await self._generate_code_research_workflow(request, sub_components)
        else:
            raise ValueError(f"Unknown engine type: {engine}")

    async def _generate_scientific_workflow(
        self,
        request: str,
        sub_components: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Generate workflow plan for scientific research"""
        workflow_plan = {
            "primary_engine": "scientific_research",
            "complexity_level": 0.9,  # High complexity
            "sub_workflows": [],
            "iteration_enabled": sub_components.get("iteration", True),
            "feedback_loops": True,
            "max_iterations": 5
        }

        # Add sub-workflows based on requirements
        if sub_components.get("deep_research", False):
            workflow_plan["sub_workflows"].append({
                "engine": "deep_research",
                "phase": "literature_review",
                "priority": "high",
                "description": "Comprehensive literature review and background research"
            })

        if sub_components.get("code_research", False):
            workflow_plan["sub_workflows"].append({
                "engine": "code_research",
                "phase": "implementation_analysis",
                "priority": "high",
                "description": "Analyze existing implementations and code patterns"
            })

        # Always include core scientific research phases
        core_phases = [
            {
                "engine": "scientific_research",
                "phase": "hypothesis_generation",
                "priority": "critical",
                "description": "Generate testable research hypotheses"
            },
            {
                "engine": "scientific_research",
                "phase": "experimental_design",
                "priority": "critical",
                "description": "Design rigorous experimental methodology"
            },
            {
                "engine": "scientific_research",
                "phase": "code_generation_and_execution",
                "priority": "critical",
                "includes_openhands": True,
                "description": "Generate and execute experimental code"
            },
            {
                "engine": "scientific_research",
                "phase": "analysis_and_validation",
                "priority": "critical",
                "description": "Analyze results and validate findings"
            }
        ]

        workflow_plan["sub_workflows"].extend(core_phases)
        return workflow_plan

    async def _generate_deep_research_workflow(
        self,
        request: str,
        sub_components: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Generate workflow plan for deep research"""
        return {
            "primary_engine": "deep_research",
            "complexity_level": 0.5,
            "sub_workflows": [
                {
                    "engine": "deep_research",
                    "phase": "multi_source_search",
                    "priority": "high",
                    "description": "Search across web, academic, and technical sources"
                },
                {
                    "engine": "deep_research",
                    "phase": "synthesis_and_analysis",
                    "priority": "high",
                    "description": "Synthesize and analyze gathered information"
                },
                {
                    "engine": "deep_research",
                    "phase": "report_generation",
                    "priority": "medium",
                    "description": "Generate comprehensive research report"
                }
            ],
            "iteration_enabled": False,
            "feedback_loops": False
        }

    async def _generate_code_research_workflow(
        self,
        request: str,
        sub_components: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Generate workflow plan for code research"""
        return {
            "primary_engine": "code_research",
            "complexity_level": 0.6,
            "sub_workflows": [
                {
                    "engine": "code_research",
                    "phase": "repository_discovery",
                    "priority": "high",
                    "description": "Find relevant repositories and code bases"
                },
                {
                    "engine": "code_research",
                    "phase": "code_analysis",
                    "priority": "high",
                    "description": "Analyze code patterns and implementations"
                },
                {
                    "engine": "code_research",
                    "phase": "documentation_generation",
                    "priority": "medium",
                    "description": "Generate usage examples and documentation"
                }
            ],
            "iteration_enabled": False,
            "feedback_loops": False,
            "includes_openhands": True,
            "use_repo_master": True
        }

    def _generate_cache_key(self, request: str) -> str:
        """Generate cache key for request"""
        # Normalize request and create hash
        normalized = request.strip().lower()
        return f"router:classification:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"

    async def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        # This would be implemented with actual metrics collection
        return {
            "total_classifications": 0,
            "accuracy_rate": 0.0,
            "cache_hit_rate": 0.0,
            "average_response_time": 0.0,
            "engine_distribution": {
                "DEEP_RESEARCH": 0,
                "SCIENTIFIC_RESEARCH": 0,
                "CODE_RESEARCH": 0
            }
        }


# Custom exceptions
class RouterException(Exception):
    """Base exception for router errors"""
    pass


class ClassificationError(RouterException):
    """Error during classification process"""
    pass


class InvalidRequestError(RouterException):
    """Invalid request format or content"""
    pass


class ThresholdError(RouterException):
    """Classification confidence below threshold"""
    pass
