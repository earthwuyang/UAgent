"""Unified node execution for research tree experiments.

Bridges hierarchical research nodes with the UnifiedOrchestrator so that every
node can leverage the AgentLaboratory, AI-Scientist, RepoMaster, and search
capabilities without duplicating bespoke execution logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from .unified_orchestrator import (
    UnifiedOrchestrator,
    WorkflowConfig,
    WorkflowType,
    OrchestrationStrategy,
)
from .code_executor import CodeExecutor
from .workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - to avoid circular imports at runtime
    from .research_tree import ExperimentResult, ExperimentType, ResearchNode


class NodeExecutionEngine:
    """Executes research tree nodes through the unified orchestration stack."""

    def __init__(self, orchestrator: UnifiedOrchestrator):
        self.orchestrator = orchestrator
        self.code_executor = CodeExecutor()
        self.workspace_manager = WorkspaceManager()

        # Lazily import ExperimentType to avoid circular imports
        from .research_tree import ExperimentType  # type: ignore

        self._supported_types = {
            ExperimentType.COMPUTATIONAL,
            ExperimentType.CODE_STUDY,
            ExperimentType.LITERATURE_ANALYSIS,
            ExperimentType.AI_SCIENTIST_DISCOVERY,
            ExperimentType.AGENT_LAB_EXPERIMENT,
            ExperimentType.WEB_SEARCH_ANALYSIS,
            ExperimentType.CODE_EXECUTION_TEST,
            ExperimentType.HIERARCHICAL_MULTI_AGENT,
        }

    def supports(self, experiment_type: Optional["ExperimentType"]) -> bool:
        return bool(experiment_type and experiment_type in self._supported_types)

    async def execute_node(
        self,
        goal_id: str,
        node: "ResearchNode",
        timeout: Optional[float] = None,
        telemetry_callback: Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]] = None,
    ) -> "ExperimentResult":
        """Execute a node via the unified orchestrator and adapt the result."""

        from .research_tree import ExperimentType  # type: ignore

        if node.experiment_type == ExperimentType.COMPUTATIONAL:
            return await self._execute_computational(goal_id, node, telemetry_callback)

        config, inputs = self._build_workflow_payload(goal_id, node)
        if not config:
            raise ValueError(
                f"No unified workflow mapping available for experiment type: {node.experiment_type}"
            )

        logger.info(
            "Dispatching node %s (%s) to unified orchestrator workflow %s",
            node.id,
            node.experiment_type.value if node.experiment_type else "unknown",
            config.workflow_type.value,
        )

        listener = None
        workflow_id = await self.orchestrator.execute_workflow(config, inputs)

        if telemetry_callback:
            initial_event = {
                "workflow_id": workflow_id,
                "event": "workflow_scheduled",
                "goal_id": goal_id,
                "node_id": node.id,
                "components": config.components,
                "experiment_type": node.experiment_type.value if node.experiment_type else None,
            }
            try:
                initial_result = telemetry_callback(initial_event)
                if asyncio.iscoroutine(initial_result):
                    await initial_result
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    f"Telemetry callback failed during scheduling for workflow {workflow_id}: {exc}"
                )

            async def listener(event: Dict[str, Any]):
                if event.get("workflow_id") != workflow_id:
                    return
                try:
                    result = telemetry_callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # pragma: no cover - telemetry should not break execution
                    logger.debug(f"Telemetry callback failed for workflow {workflow_id}: {exc}")

            self.orchestrator.add_event_listener(listener)

        try:
            workflow_result = await self.orchestrator.wait_for_completion(workflow_id, timeout)
        finally:
            if listener:
                self.orchestrator.remove_event_listener(listener)

        if not workflow_result:
            raise RuntimeError(f"Workflow {workflow_id} did not return a result")

        success = workflow_result.status == "completed"
        confidence = self._estimate_confidence(success, workflow_result.components_used)
        metrics = self._build_metrics(workflow_id, workflow_result)
        insights = self._summarize_insights(workflow_result.results)

        execution_time = workflow_result.execution_time or 0.0

        from .research_tree import ExperimentResult  # type: ignore

        return ExperimentResult(
            experiment_id=node.id,
            success=success,
            confidence=confidence,
            metrics=metrics,
            data=workflow_result.results or {},
            insights=insights,
            execution_time=execution_time,
            resources_used={"components": workflow_result.components_used},
        )

    async def _execute_computational(
        self,
        goal_id: str,
        node: "ResearchNode",
        telemetry_callback: Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]] = None,
    ) -> "ExperimentResult":
        from .research_tree import ExperimentResult  # type: ignore

        async def emit(event: Dict[str, Any]):
            if telemetry_callback:
                try:
                    result = telemetry_callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # pragma: no cover
                    logger.debug(f"Telemetry callback failed for computational node {node.id}: {exc}")

        await emit({
            "event": "code_execution_started",
            "goal_id": goal_id,
            "node_id": node.id,
            "description": node.description,
        })

        code_result = await self.code_executor.execute_computational_task(
            task_description=node.description or node.title,
            config=node.experiment_config or {},
            node_id=node.id
        )

        await emit({
            "event": "code_execution_completed",
            "goal_id": goal_id,
            "node_id": node.id,
            "success": code_result.success,
            "workspace_path": code_result.metrics.get("workspace_path"),
            "execution_time": code_result.execution_time,
        })

        metrics = {
            "workspace_path": code_result.metrics.get("workspace_path"),
            "execution_time": code_result.execution_time,
            "iterations": code_result.iterations,
            "debug_attempts": code_result.debug_attempts,
            "generated_files": code_result.metrics.get("generated_files"),
        }

        data = {
            "final_code": code_result.final_code,
            "code_blocks": code_result.code_blocks,
            "workspace_path": code_result.metrics.get("workspace_path"),
        }

        insights = code_result.insights or []

        return ExperimentResult(
            experiment_id=node.id,
            success=code_result.success,
            confidence=code_result.confidence,
            metrics=metrics,
            data=data,
            insights=insights,
            execution_time=code_result.execution_time,
            resources_used={"components": ["code_executor"]},
        )

    def _build_workflow_payload(
        self,
        goal_id: str,
        node: "ResearchNode",
    ) -> Tuple[Optional[WorkflowConfig], Dict[str, Any]]:
        from .research_tree import ExperimentType  # type: ignore

        exp_type = node.experiment_type
        node_context = node.context or {}

        base_inputs = {
            "goal_id": goal_id,
            "node_id": node.id,
            "title": node.title,
            "description": node.description,
            "node_context": node_context,
            "experiment_config": node.experiment_config,
        }

        if exp_type == ExperimentType.COMPUTATIONAL:
            config = WorkflowConfig(
                workflow_type=WorkflowType.AUTOMATED_RESEARCH,
                strategy=OrchestrationStrategy.ROMA_RECURSIVE,
                components=["ai_scientist", "search_engine", "agent_lab", "meta_agent"],
                parameters={
                    "collaboration_enabled": True,
                    "auto_iteration": node.experiment_config.get("auto_iteration", True),
                },
            )
            inputs = {
                **base_inputs,
                "research_questions": node.experiment_config.get(
                    "research_questions", node_context.get("research_questions", [])
                ),
                "query": node.experiment_config.get(
                    "search_query", node.description or node.title
                ),
            }
            return config, inputs

        if exp_type == ExperimentType.LITERATURE_ANALYSIS or exp_type == ExperimentType.WEB_SEARCH_ANALYSIS:
            query = node.experiment_config.get("search_query") or node.description or node.title
            if not query:
                return None, {}

            config = WorkflowConfig(
                workflow_type=WorkflowType.MULTI_MODAL_SEARCH,
                strategy=OrchestrationStrategy.ADAPTIVE,
                components=["search_engine", "ai_scientist"],
                parameters={"result_synthesis": True},
            )
            inputs = {**base_inputs, "query": query, "domain": node_context.get("domain")}
            return config, inputs

        if exp_type == ExperimentType.CODE_STUDY or exp_type == ExperimentType.CODE_EXECUTION_TEST:
            repo_path = (
                node.experiment_config.get("repository_path")
                or node_context.get("repository_path")
                or node_context.get("repo_path")
            )
            if not repo_path:
                if exp_type == ExperimentType.CODE_EXECUTION_TEST:
                    fallback_config = WorkflowConfig(
                        workflow_type=WorkflowType.AUTOMATED_RESEARCH,
                        strategy=OrchestrationStrategy.SEQUENTIAL,
                        components=["agent_lab", "meta_agent"],
                        parameters={
                            "execution_focus": True,
                            "auto_iteration": False
                        },
                    )
                    fallback_inputs = {
                        **base_inputs,
                        "execution_goal": "validate_generated_code",
                        "notes": node.experiment_config
                    }
                    return fallback_config, fallback_inputs
                return None, {}

            analysis_depth = node.experiment_config.get("analysis_depth", "semantic")
            config = WorkflowConfig(
                workflow_type=WorkflowType.CODE_ANALYSIS,
                strategy=OrchestrationStrategy.SEQUENTIAL,
                components=["repo_master", "agent_lab", "search_engine"],
                parameters={
                    "analysis_depth": analysis_depth,
                    "pattern_detection": node.experiment_config.get("pattern_detection", True),
                    "collaboration_enabled": True,
                },
            )
            inputs = {
                **base_inputs,
                "repository_path": repo_path,
            }
            return config, inputs

        if exp_type == ExperimentType.AI_SCIENTIST_DISCOVERY or exp_type == ExperimentType.HIERARCHICAL_MULTI_AGENT:
            config = WorkflowConfig(
                workflow_type=WorkflowType.AUTOMATED_RESEARCH,
                strategy=OrchestrationStrategy.ROMA_RECURSIVE,
                components=["ai_scientist", "agent_lab", "meta_agent"],
                parameters={
                    "collaboration_enabled": node.experiment_config.get("collaboration_enabled", True),
                    "auto_iteration": True,
                },
            )
            inputs = {
                **base_inputs,
                "title": node.title or "AI Scientist Investigation",
                "research_questions": node.experiment_config.get("research_questions", []),
            }
            return config, inputs

        if exp_type == ExperimentType.AGENT_LAB_EXPERIMENT:
            config = WorkflowConfig(
                workflow_type=WorkflowType.COLLABORATIVE_DEVELOPMENT,
                strategy=OrchestrationStrategy.PARALLEL,
                components=["agent_lab", "meta_agent", "repo_master"],
                parameters={
                    "team_size": node.experiment_config.get("team_size", 4),
                    "specialization_level": node.experiment_config.get("specialization_level", "high"),
                },
            )
            inputs = {
                **base_inputs,
                "project_name": node.title,
                "requirements": node.experiment_config.get("requirements", []),
            }
            return config, inputs

        return None, {}

    def _estimate_confidence(self, success: bool, components: List[str]) -> float:
        if not success:
            return 0.2

        base = 0.7
        if "ai_scientist" in components:
            base += 0.1
        if "agent_lab" in components:
            base += 0.05
        if "repo_master" in components:
            base += 0.05

        return min(base, 0.95)

    def _build_metrics(self, workflow_id: str, workflow_result) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "status": workflow_result.status,
            "components": workflow_result.components_used,
            "execution_time": workflow_result.execution_time,
        }
        if workflow_result.metadata:
            metrics["metadata"] = workflow_result.metadata
        return metrics

    def _summarize_insights(self, results: Optional[Dict[str, Any]]) -> List[str]:
        if not results:
            return []

        insights: List[str] = []

        def _summaries(prefix: str, value: Any):
            if isinstance(value, dict):
                keys = list(value.keys())[:3]
                readable = ", ".join(keys)
                insights.append(f"{prefix} produced keys: {readable}")
            elif isinstance(value, list):
                insights.append(f"{prefix} returned {len(value)} items")
            else:
                snippet = str(value)
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                insights.append(f"{prefix}: {snippet}")

        for key, value in results.items():
            _summaries(key, value)

        # Include compact JSON snapshot for debugging visibility
        try:
            compact = json.dumps(results, default=str)[:400]
            insights.append(f"Results snapshot: {compact}")
        except Exception:
            logger.debug("Failed to serialize workflow results for insights", exc_info=True)

        return insights
