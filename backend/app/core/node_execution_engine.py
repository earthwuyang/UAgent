"""Unified node execution for research tree experiments.

Bridges hierarchical research nodes with the UnifiedOrchestrator so that every
node can leverage the AgentLaboratory, AI-Scientist, RepoMaster, and search
capabilities without duplicating bespoke execution logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

from .unified_orchestrator import (
    UnifiedOrchestrator,
    WorkflowConfig,
    WorkflowType,
    OrchestrationStrategy,
)
from .code_executor import CodeExecutor
from .workspace_manager import WorkspaceManager
from .llm_client import llm_client

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
        if node.experiment_type == ExperimentType.CODE_EXECUTION_TEST:
            return await self._execute_code_validation(goal_id, node, telemetry_callback)

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

    async def _execute_code_validation(
        self,
        goal_id: str,
        node: "ResearchNode",
        telemetry_callback: Optional[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]] = None,
    ) -> "ExperimentResult":
        from .research_tree import ExperimentResult  # type: ignore
        from .research_tree import ExperimentType  # type: ignore

        workspace_path = None
        if isinstance(node.experiment_config, dict):
            workspace_path = node.experiment_config.get("workspace_path")
        if not workspace_path or not Path(workspace_path).exists():
            raise ValueError("workspace_path is required for CODE_EXECUTION_TEST nodes")

        scripts_dir = Path(workspace_path) / "scripts"

        async def emit(event: Dict[str, Any]):
            if telemetry_callback:
                try:
                    result = telemetry_callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # pragma: no cover
                    logger.debug(f"Telemetry callback failed for code validation {node.id}: {exc}")

        await emit({
            "event": "code_validation_started",
            "goal_id": goal_id,
            "node_id": node.id,
            "workspace_path": workspace_path,
        })

        commands = []
        for script_name in ["build.sh", "run.sh", "test.sh"]:
            script_path = scripts_dir / script_name
            if script_path.exists():
                commands.append((script_name, script_path))

        execution_outputs = []
        success = True

        for name, script in commands:
            attempt = 0
            while True:
                result = await self._run_script(script, Path(workspace_path))
                execution_outputs.append({
                    "script": name,
                    "attempt": attempt + 1,
                    "return_code": result["return_code"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                })
                await emit({
                    "event": "code_validation_step",
                    "goal_id": goal_id,
                    "node_id": node.id,
                    "script": name,
                    "attempt": attempt + 1,
                    "return_code": result["return_code"],
                })

                if result["return_code"] == 0:
                    break

                fix_applied = await self._attempt_auto_fix(
                    name,
                    result,
                    Path(workspace_path),
                    emit,
                    goal_id,
                    node.id,
                )
                if not fix_applied or attempt >= node.experiment_config.get("retries", 0):
                    success = False
                    break

                attempt += 1

            if not success:
                break

        confidence = 0.9 if success else 0.2
        insights = []
        if success:
            insights.append("Docker image built and container validated successfully.")
        else:
            insights.append("Validation scripts reported failures; inspect stdout/stderr for details.")

        metrics = {
            "workspace_path": workspace_path,
            "steps_executed": len(execution_outputs),
            "success": success,
        }

        if success:
            await self._cleanup_containers(Path(workspace_path))

        await emit({
            "event": "code_validation_completed",
            "goal_id": goal_id,
            "node_id": node.id,
            "success": success,
        })

        return ExperimentResult(
            experiment_id=node.id,
            success=success,
            confidence=confidence,
            metrics=metrics,
            data={"execution_outputs": execution_outputs},
            insights=insights,
            execution_time=sum(max(0.0, step.get("execution_time", 0.0)) for step in execution_outputs),
            resources_used={"components": ["code_executor", "shell"]},
        )

    async def _run_script(self, script_path: Path, cwd: Path) -> Dict[str, Any]:
        if not script_path.exists():
            return {
                "stdout": "",
                "stderr": f"Script {script_path.name} not found",
                "return_code": 1,
                "execution_time": 0.0,
            }

        start = asyncio.get_running_loop().time()
        process = await asyncio.create_subprocess_exec(
            "bash",
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await process.communicate()
        end = asyncio.get_running_loop().time()
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "return_code": process.returncode,
            "execution_time": max(0.0, end - start),
        }

    async def _attempt_auto_fix(
        self,
        script_name: str,
        result: Dict[str, Any],
        workspace: Path,
        emit: Callable[[Dict[str, Any]], Awaitable[None]],
        goal_id: str,
        node_id: str,
    ) -> bool:
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""

        workspace_listing = []
        for path in workspace.rglob('*'):
            if path.is_file():
                try:
                    rel = path.relative_to(workspace)
                    workspace_listing.append(str(rel))
                except Exception:
                    continue
            if len(workspace_listing) >= 50:
                break

        important_files = [
            workspace / "docker-compose.yml",
            workspace / "Dockerfile",
            workspace / "my.cnf",
            workspace / "init.sql",
        ]
        file_snippets = []
        for file_path in important_files:
            if file_path.exists() and file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    snippet = content if len(content) < 1500 else content[:1500] + "\n..."
                    file_snippets.append(f"```{file_path.name}\n{snippet}\n```")
                except Exception:
                    continue

        prompt = f"""You are an expert DevOps engineer. A validation script failed while deploying a Dockerized MySQL setup.

Script name: {script_name}
Exit code: {result.get('return_code')}

STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n\nWorkspace files (partial): {workspace_listing}

Key file contents:\n{''.join(file_snippets) if file_snippets else 'N/A'}

Analyze the failure and output JSON with:
  "reason": short explanation
  "files": list of {{"path": relative_path, "content": new_content}}
  "commands": optional shell commands to run after updating files (e.g., clean up containers).
  "notes": optional extra guidance for the next attempt.
Ensure that any port conflicts or missing files are fixed by modifying the files accordingly.
"""

        try:
            llm_response = await llm_client.generate_response(
                prompt=prompt,
                system_prompt="You are a precise DevOps fixer. Output ONLY JSON.",
                temperature=0.1,
                max_tokens=800,
            )
            if not llm_response.get("success"):
                return False

            import json
            import re

            content = llm_response.get("content", "")
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                return False

            fix_spec = json.loads(json_match.group())
        except Exception as exc:
            logger.debug(f"LLM fix generation failed: {exc}")
            return False

        files = fix_spec.get("files", []) or []
        commands = fix_spec.get("commands", []) or []
        reason = fix_spec.get("reason", "auto_fix")
        applied = False

        for file_entry in files:
            rel_path = file_entry.get("path")
            content = file_entry.get("content")
            if not rel_path or content is None:
                continue
            target_path = workspace / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            applied = True

        for command in commands:
            if isinstance(command, str) and command.strip():
                await self._run_command(command, workspace)
                applied = True

        if applied:
            await emit({
                "event": "code_validation_fix_applied",
                "goal_id": goal_id,
                "node_id": node_id,
                "fix": reason,
                "details": {"files": [f.get("path") for f in files], "commands": commands},
            })

        return applied

    async def _cleanup_containers(self, workspace: Path) -> None:
        compose_file = workspace / "docker-compose.yml"
        if compose_file.exists():
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker-compose",
                    "down",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
                await process.communicate()
            except Exception:
                pass

    async def _run_command(self, command: str, workspace: Path) -> None:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception as exc:
            logger.debug(f"Command '{command}' failed during auto-fix: {exc}")

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
