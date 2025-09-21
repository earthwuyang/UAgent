"""Bridges OpenHands CodeAct/CodeReact capabilities for scientific research."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.llm_client import LLMClient
from ..core.openhands import OpenHandsClient, CodeGenerationRequest, CodeGenerationResult
from ..services.codeact_runner import CodeActRunner


ProgressCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]

_WORKSPACE_CONFIG_KEYS = {"max_file_size", "max_total_size", "timeout", "python_path", "allowed_commands"}


def _safe_json_loads(content: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None


@dataclass
class GoalPlanStep:
    """Single step within a goal-oriented plan."""

    id: str
    description: str
    expected_output: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    code_result: Optional[CodeGenerationResult] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class GoalPlan:
    """Plan derived from OpenHands goal-planning prompt."""

    plan_id: str
    goal: str
    summary: str
    steps: List[GoalPlanStep] = field(default_factory=list)


class OpenHandsGoalPlanBridge:
    """Convenience helper to use OpenHands for goal planning and code execution."""

    def __init__(self, client: OpenHandsClient, llm_client: LLMClient):
        self._client = client
        self._llm_client = llm_client
        self._codeact_runner = None
        if getattr(client, 'action_runner', None) and client.action_runner.is_available:
            self._codeact_runner = CodeActRunner(llm_client, client.action_runner)
        self._codeact_max_steps = int(os.getenv("CODEACT_MAX_STEPS", "20"))
        self._codeact_action_timeout = int(os.getenv("CODEACT_ACTION_TIMEOUT", "1800"))
        self._bootstrap_enabled = os.getenv("CODEACT_BOOTSTRAP_ENABLED", "true").lower() != "false"
        self._bootstrap_goal = os.getenv(
            "CODEACT_BOOTSTRAP_GOAL",
            """
Prepare the workspace for complex software experiments by ensuring the following:
- Update package indexes and install mandatory build tools and system dependencies (git, build-essential, clang, cmake, ninja, make, libreadline-dev, zlib1g-dev, libssl-dev, python3-dev, python3-venv, pkg-config).
- Install database tooling commonly required for data-intensive research (PostgreSQL server/client utilities, DuckDB CLI/Python package).
- Initialize local runtime directories under the workspace (e.g., workspace/db/postgres, workspace/db/duckdb) and configure services to run on non-default ports (such as PostgreSQL on 55432) without requiring elevated privileges.
- Set environment exports by writing a script (e.g., workspace/env.sh) that defines PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, DUCKDB_PATH, and ensure future commands source this script before running experiments.
- Verify installations by printing version information (psql --version, duckdb --version, gcc --version) and confirming services respond to simple test queries.
Follow safe practices (no sudo if not required, avoid destructive commands) and make changes idempotent so re-running the preparation does not fail.
""",
        )

    @staticmethod
    def _sanitize_workspace_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return {key: raw[key] for key in _WORKSPACE_CONFIG_KEYS if key in raw}

    @staticmethod
    def _extract_candidate_paths(text: str) -> List[str]:
        if not text:
            return []
        candidates = set()
        # Match patterns like path/to/file.ext or file.ext
        for match in re.findall(r"[\w\-./]+\.[A-Za-z0-9_]+", text):
            if match.startswith("http"):
                continue
            candidates.add(match.strip())
        for match in re.findall(r"(?:[A-Za-z0-9_\-]+/){1,}[A-Za-z0-9_\-./]*", text):
            if match.startswith("http"):
                continue
            candidates.add(match.strip().rstrip("./"))
        return [c for c in candidates if c]

    async def generate_goal_plan(self, goal: str, context: Dict[str, Any]) -> GoalPlan:
        """Use LLM to produce a structured multi-step plan."""

        context_snippet = json.dumps(context, indent=2) if context else "{}"
        plan_prompt = f"""
You are an expert CodeAct/CodeReact planning assistant inside the OpenHands framework.
Break the high-level scientific goal into executable coding steps that an autonomous agent can follow.

Goal:
"{goal}"

Relevant context (JSON):
{context_snippet}

Respond **only** with JSON using the following schema:
{{
  "summary": "overall plan summary",
  "steps": [
    {{
      "id": "short_step_id",
      "description": "concise action to perform",
      "expected_output": "what this step should produce",
      "dependencies": ["optional_previous_step_ids"]
    }}
  ]
}}

Ensure dependencies reference earlier step ids. Provide 3-6 actionable steps.
"""

        response = await self._llm_client.generate(plan_prompt, max_tokens=700, temperature=0.3)
        parsed = _safe_json_loads(response) or {}

        steps: List[GoalPlanStep] = []
        for raw in parsed.get("steps", []) or []:
            if not isinstance(raw, dict):
                continue
            step_id = str(raw.get("id") or f"step-{len(steps)+1}")
            step = GoalPlanStep(
                id=step_id,
                description=str(raw.get("description") or "Execute plan step"),
                expected_output=str(raw.get("expected_output") or ""),
                dependencies=[str(dep) for dep in raw.get("dependencies", []) if dep],
            )
            steps.append(step)

        if not steps:
            steps = [
                GoalPlanStep(
                    id="step-1",
                    description="Implement baseline experiment script",
                    expected_output="Python script that collects and logs experimental results",
                )
            ]

        plan = GoalPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal=goal,
            summary=str(parsed.get("summary") or "Generated OpenHands plan"),
            steps=steps,
        )
        return plan

    async def execute_goal_plan(
        self,
        plan: GoalPlan,
        execution_context: Dict[str, Any],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GoalPlan:
        """Execute goal plan steps using OpenHands CodeAct code generation."""

        session_config = await self._client.create_session(
            research_type="scientific_research",
            session_id=f"plan_{plan.plan_id}",
            config=self._sanitize_workspace_config(execution_context.get("resource_requirements")),
        )

        # Optional environment bootstrap before executing plan steps
        if (
            self._bootstrap_enabled
            and execution_context.get("bootstrap_environment", True)
            and self._codeact_runner is not None
        ):
            workspace_path = self._client.workspace_manager.get_workspace_path(
                session_config.workspace_config.get("workspace_id")
            ) if session_config.workspace_config else None
            if workspace_path:
                async def _bootstrap_cb(event: str, data: Dict[str, Any]):
                    if progress_callback:
                        await progress_callback(
                            "bootstrap_event",
                            {
                                "event": event,
                                "data": data,
                            },
                        )

                if progress_callback:
                    await progress_callback(
                        "bootstrap_start",
                        {
                            "goal": plan.goal,
                            "workspace": str(workspace_path),
                        },
                    )
                bootstrap_result = await self._codeact_runner.run(
                    workspace_path=workspace_path,
                    goal=self._bootstrap_goal,
                    max_steps=self._codeact_max_steps,
                    timeout_per_action=self._codeact_action_timeout,
                    progress_cb=_bootstrap_cb,
                )
                if progress_callback:
                    await progress_callback(
                        "bootstrap_complete",
                        {
                            "success": bool(bootstrap_result.get("success")),
                        },
                    )

        for step in plan.steps:
            step.status = "running"
            if progress_callback:
                await progress_callback(
                    "step_started",
                    {
                        "step_id": step.id,
                        "description": step.description,
                    },
                )

            # Workspace path for CodeAct operations
            workspace_path = self._client.workspace_manager.get_workspace_path(
                session_config.workspace_config.get("workspace_id")
            ) if session_config.workspace_config else None

            step_stdout = ""
            step_stderr = ""
            step_success = False

            if self._codeact_runner and workspace_path:
                if progress_callback:
                    await progress_callback(
                        "codeact_start",
                        {
                            "step_id": step.id,
                            "description": step.description,
                        },
                    )

                async def _cb(event: str, data: Dict[str, Any]):
                    if progress_callback:
                        await progress_callback(
                            "codeact_event",
                            {
                                "step_id": step.id,
                                "event": event,
                                "data": data,
                            },
                        )

                goal_text = (
                    f"Complete the following plan step as part of the research goal:\n"
                    f"Goal: {plan.goal}\n"
                    f"Step description: {step.description}\n"
                    f"Expected output: {step.expected_output}\n"
                    "Do not rely on placeholders. Make concrete, runnable changes."
                )

                codeact_result = await self._codeact_runner.run(
                    workspace_path=workspace_path,
                    goal=goal_text,
                    max_steps=self._codeact_max_steps,
                    timeout_per_action=self._codeact_action_timeout,
                    progress_cb=_cb,
                )

                step_success = bool(codeact_result.get("success"))
                step.status = "completed" if step_success else "failed"
                step.notes.append(str(codeact_result))

                if isinstance(codeact_result, dict):
                    steps_log = codeact_result.get("steps") or []
                    if steps_log:
                        last_obs = steps_log[-1].get("observation") if isinstance(steps_log[-1], dict) else ""
                        step_stdout = last_obs or ""
                    if not step_success:
                        step_stderr = codeact_result.get("error", "") or ""

                if step_success and workspace_path:
                    expected_paths = self._extract_candidate_paths(step.expected_output)
                    missing_artifacts: List[str] = []
                    for rel_path in expected_paths:
                        candidate = Path(rel_path)
                        if candidate.is_absolute():
                            continue
                        try:
                            candidate_path = (workspace_path / candidate).resolve()
                            workspace_root = workspace_path.resolve()
                            candidate_path.relative_to(workspace_root)
                        except Exception:
                            continue
                        if not candidate_path.exists():
                            missing_artifacts.append(rel_path)

                    if missing_artifacts:
                        validation_message = (
                            "Expected artifacts not found after CodeAct run: "
                            + ", ".join(sorted(missing_artifacts))
                        )
                        step_success = False
                        step.status = "failed"
                        step.notes.append(validation_message)
                        step_stderr = (
                            f"{step_stderr}\n{validation_message}".strip()
                            if step_stderr
                            else validation_message
                        )
                        if progress_callback:
                            await progress_callback(
                                "codeact_validation_failed",
                                {
                                    "step_id": step.id,
                                    "missing_artifacts": missing_artifacts,
                                },
                            )

                if progress_callback:
                    await progress_callback(
                        "codeact_complete",
                        {
                            "step_id": step.id,
                            "success": step_success,
                        },
                    )

            else:
                # Enforce CodeAct-only execution for goal steps. If the runtime or workspace
                # is unavailable, mark the step as failed so the caller can provision it
                # rather than silently generating placeholder code.
                step.status = "failed"
                step_success = False
                step_stderr = "CodeAct runtime or workspace not available; refusing legacy LLM codegen"
                step.notes.append(step_stderr)

            if progress_callback:
                await progress_callback(
                    "step_completed",
                    {
                        "step_id": step.id,
                        "status": step.status,
                        "stdout": step_stdout,
                        "stderr": step_stderr,
                    },
                )

            if not step_success:
                break

        return plan
