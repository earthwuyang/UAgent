"""Bridges OpenHands CodeAct/CodeReact capabilities for scientific research."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.llm_client import LLMClient
from ..core.openhands import OpenHandsClient, CodeGenerationRequest, CodeGenerationResult


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

    @staticmethod
    def _sanitize_workspace_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return {key: raw[key] for key in _WORKSPACE_CONFIG_KEYS if key in raw}

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

            request = CodeGenerationRequest(
                session_id=session_config.session_id,
                task_description=f"{step.description}\nExpected Output: {step.expected_output}",
                context={"plan_goal": plan.goal, **execution_context},
                execute_immediately=True,
                language="python",
                timeout=execution_context.get("timeout", 300),
            )

            result = await self._client.generate_and_execute_code(
                request,
                llm_client=self._llm_client,
            )
            step.code_result = result
            step.notes.append(result.analysis or "")
            step.status = "completed" if result.success else "failed"

            if progress_callback:
                await progress_callback(
                    "step_completed",
                    {
                        "step_id": step.id,
                        "status": step.status,
                        "stdout": result.execution_result.stdout if result.execution_result else "",
                        "stderr": result.execution_result.stderr if result.execution_result else "",
                    },
                )

            if not result.success:
                break

        return plan
