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
# Legacy CodeAct runner removed; V2 bridge only
from ..utils.json_utils import JsonParseError, safe_json_loads


ProgressCallback = Callable[[str, Dict[str, Any]], Awaitable[None]]

_WORKSPACE_CONFIG_KEYS = {"max_file_size", "max_total_size", "timeout", "python_path", "allowed_commands"}


def _safe_json_loads(content: str) -> Optional[Dict[str, Any]]:
    if content is None:
        return None
    try:
        parsed = safe_json_loads(content)
    except JsonParseError:
        return None
    return parsed if isinstance(parsed, dict) else parsed


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
        # Migrate to V2 by default: prefer V2 runner for all actions
        self._codeact_runner = None
        self._codeact_max_steps = int(os.getenv("CODEACT_MAX_STEPS", "20"))
        # Use the unified OPENHANDS_ACTION_TIMEOUT (default: 120 seconds = 2 minutes)
        self._codeact_action_timeout = int(os.getenv("OPENHANDS_ACTION_TIMEOUT", "120"))
        self._bootstrap_enabled = os.getenv("CODEACT_BOOTSTRAP_ENABLED", "true").lower() != "false"
        self._bootstrap_goal = os.getenv(
            "CODEACT_BOOTSTRAP_GOAL",
            """
Prepare the workspace for software and data experiments **without** installing new system packages. Operate only within the provided directory tree.
- Do NOT run `apt-get`, `sudo`, or any system package manager. Assume required toolchains already exist.
- Do NOT clone external repositories (e.g., OpenHands). Work entirely within the current workspace.
- Inspect the current environment (e.g., git, python, compilers, databases found on PATH) and record findings in `workspace/env_report.txt`. Capture only the presence of tools you expect to use; avoid sweeping version scans or heavyweight discovery loops. If a tool is missing, note it instead of attempting installation.
- Only create directories or files that the current task explicitly requires. Avoid hard-coded technology names; prefer generic locations such as `workspace/run/` or `workspace/tmp/` as needed.
- Generate a shell script `workspace/env.sh` that exports environment variables actually used in the current workspace (leave placeholders commented when no concrete values exist). The script must be idempotent and safe to source multiple times.
- Summarise the inspection results, outstanding gaps, and created resources in `workspace/bootstrap_summary.md`.
All actions must be idempotent and keep the workspace self-contained.
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

Constraints:
- Operate strictly inside the provided workspace; prefer creating/editing files and running commands there.
- Do NOT clone unrelated repositories. In particular, do NOT clone the OpenHands repository — the runtime is already integrated.
- Avoid `sudo`/`apt-get` unless explicitly required by the user.

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

Ensure dependencies reference earlier step ids. Provide 3-6 actionable steps. Do not include actions that clone OpenHands or unrelated repositories.
"""

        # Increase max_tokens to prevent truncation of plan JSON
        response = await self._llm_client.generate(plan_prompt, max_tokens=4000, temperature=0.3)
        parsed = _safe_json_loads(response)
        if not isinstance(parsed, dict):
            # Try to extract JSON from potentially truncated response
            json_match = re.search(r'\{[\s\S]*', response)
            if json_match:
                try:
                    # json is already imported at the top of the file
                    parsed = json.loads(json_match.group(0) + ']}')  # Try to close truncated JSON
                except (json.JSONDecodeError, ValueError) as e:
                    # Log the attempt but continue with the error
                    import logging
                    logging.debug(f"Failed to repair truncated JSON: {e}")

            if not isinstance(parsed, dict):
                raise RuntimeError(f"OpenHands plan generation returned non-dict payload (response length: {len(response)}); refusing fallback")

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
            raise RuntimeError("OpenHands plan generation produced zero valid steps")

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
        """Execute GoalPlan by creating V2 step artifacts (no proxy enforcement)."""
        # Use a stable session id for the plan
        session_id = execution_context.get("session_id") or f"experiment_{plan.plan_id}"

        session_config = await self._client.ensure_session(
            research_type="scientific_research",
            session_id=session_id,
            config=self._sanitize_workspace_config(execution_context.get("resource_requirements")),
        )

        # Use V2 client/runner to generate lightweight artifacts per step.
        workspace_path = self._client.workspace_manager.get_workspace_path(
            session_config.workspace_config.get("workspace_id")
        ) if session_config.workspace_config else None

        if not workspace_path:
            for s in plan.steps:
                s.status = "failed"
                s.notes.append("No workspace path available for V2 execution")
            return plan

        try:
            from ..services.codeact_runner import CodeActRunnerV2
            from ..integrations.openhands_runtime import OpenHandsClientV2
            v2_client = OpenHandsClientV2(workspace_path=workspace_path, allowed_write_roots=[workspace_path, workspace_path/"code", workspace_path/"experiments", workspace_path/"logs", workspace_path/"output", workspace_path/"workspace"])  # type: ignore
            v2_runner = CodeActRunnerV2(v2_client, workspace_path)
            # Create a simple artifact per step with description + expected output
            for step in plan.steps:
                step.status = "running"
                if progress_callback:
                    await progress_callback("step_started", {"step_id": step.id, "description": step.description})
                rel = f"experiments/{plan.plan_id}/logs/step_{step.id}.txt"
                content = f"Description:\n{step.description}\n\nExpected output:\n{step.expected_output}\n"
                await v2_runner.create_if_absent(rel, content)
                step.status = "completed"
                if progress_callback:
                    await progress_callback("step_completed", {"step_id": step.id, "status": step.status, "stdout": rel, "stderr": ""})
        except Exception as exc:
            for s in plan.steps:
                s.status = "failed"
                s.notes.append(f"V2 execution failed: {exc}")
        return plan

        # Execute each step with CodeAct
        for step in plan.steps:
            step.status = "running"
            if progress_callback:
                await progress_callback(
                    "step_started",
                    {"step_id": step.id, "description": step.description},
                )

            workspace_path = self._client.workspace_manager.get_workspace_path(
                session_config.workspace_config.get("workspace_id")
            ) if session_config.workspace_config else None

            step_stdout = ""
            step_stderr = ""
            step_success = False

            if self._codeact_runner and workspace_path:
                async def _cb(event: str, data: Dict[str, Any]):
                    if progress_callback:
                        await progress_callback(
                            "codeact_event",
                            {"step_id": step.id, "event": event, "data": data},
                        )

                goal_text = (
                    f"Complete the following plan step as part of the research goal:\n"
                    f"Goal: {plan.goal}\n"
                    f"Step description: {step.description}\n"
                    f"Expected output: {step.expected_output}\n"
                    "Do not run `apt-get`, `sudo`, or other system package managers. "
                    "Work strictly within the existing Python environment and workspace. "
                    "Prefer relative paths under the workspace. Combine multiple shell commands in a single bash -lc if needed."
                )

                result = await self._codeact_runner.run(
                    workspace_path=workspace_path,
                    goal=goal_text,
                    max_steps=self._codeact_max_steps,
                    timeout_per_action=self._codeact_action_timeout,
                    progress_cb=_cb,
                )

                step_success = bool(result.get("success"))
                step.status = "completed" if step_success else "failed"
                step.notes.append(str(result))
                if isinstance(result, dict):
                    steps_log = result.get("steps") or []
                    if steps_log:
                        last_obs = steps_log[-1].get("observation") if isinstance(steps_log[-1], dict) else ""
                        step_stdout = last_obs or ""
                    if not step_success:
                        step_stderr = result.get("error", "") or ""
            else:
                step.status = "failed"
                step_stderr = "CodeAct runtime or workspace not available"
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


from .openhands_runtime import OpenHandsClientV2  # type: ignore
from ..services.proxy_sql_tool import ProxySQLTool


class OpenHandsBridgeV2:
    """Minimal plan executor using CodeActRunnerV2 with proxy_sql support.

    Steps schema example:
      {"steps": [
         {"kind":"create_if_absent","path":"code/a.py","content":"print('hi')"},
         {"kind":"run","cmd":"./.venv/bin/python code/a.py"},
         {"kind":"proxy_sql","engine":"duckdb","sql":"SELECT 1"}
      ]}
    """

    def __init__(self, runner, proxy_base: str | None = None):
        self.runner = runner
        self._proxy_tool: ProxySQLTool | None = None
        try:
            ws_dir = getattr(runner, "ws_dir", None) or getattr(runner, "workspace_dir", None)
            if ws_dir is None:
                from pathlib import Path
                ws_dir = Path(".").resolve()
            self._proxy_tool = ProxySQLTool(runner.client, ws_dir, proxy_base or "http://127.0.0.1:7890/sql")
        except Exception:
            self._proxy_tool = None

    async def execute_steps(self, goal_plan: dict, plan_context: dict, progress_cb=None) -> dict:
        await self.runner.ensure_bootstrap()
        steps = list(goal_plan.get("steps", []))
        results = []
        for idx, step in enumerate(steps):
            kind = step.get("kind")
            if progress_cb:
                try:
                    await progress_cb({"phase":"begin_step","index":idx,"kind":kind})
                except Exception:
                    pass
            if kind == "create_if_absent":
                r = await self.runner.create_if_absent(step["path"], step.get("content", ""))
            elif kind == "write":
                r = await self.runner.write(step["path"], step.get("content", ""), overwrite=step.get("overwrite", True))
            elif kind == "read":
                r = await self.runner.read(step["path"]) 
            elif kind == "run":
                r = await self.runner.run(step["cmd"], timeout_sec=step.get("timeout_sec", 300))
            elif kind == "proxy_sql" and self._proxy_tool is not None:
                pr = await self._proxy_tool.execute(step.get("sql", ""), engine=step.get("engine", ""), dataset=step.get("dataset"), timeout_sec=step.get("timeout_sec", 120), retries=step.get("retries", 2))
                r = {"success": pr.ok, "result": {"status": pr.status, "rows": pr.rows_count, "elapsed_ms": pr.elapsed_ms, "error": pr.error_message}}
            else:
                r = {"success": False, "error": {"type": "CLIENT_ERROR", "message": f"unknown step kind: {kind}"}}
            ok = r.success if hasattr(r, "success") else r.get("success")
            results.append({"index": idx, "ok": ok, "result": r.__dict__ if hasattr(r, "__dict__") else r})
            if progress_cb:
                try:
                    await progress_cb({"phase":"end_step","index":idx,"ok":bool(ok)})
                except Exception:
                    pass
            if not ok and step.get("required", True):
                break
        return {"results": results}

    async def execute_goal_plan(
        self,
        plan: GoalPlan,
        execution_context: Dict[str, Any],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> GoalPlan:
        """Execute goal plan steps using OpenHands CodeAct code generation."""

        # Use the session ID from execution context if available, otherwise create unified ID
        session_id = execution_context.get("session_id")
        if not session_id:
            # Create a unified session ID that will be reused for all OpenHands operations
            session_id = f"experiment_{plan.plan_id}"

        # Use ensure_session to reuse existing workspace if available
        session_config = await self._client.ensure_session(
            research_type="scientific_research",
            session_id=session_id,
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
                    "Do not run `apt-get`, `sudo`, or any other system package manager commands unless really need to update system dependencies. "
                    "Work strictly with the existing Python environment and workspace resources. Use relative paths inside the workspace (e.g., `collect_data.py` or `code/collect_data.py`) instead of absolute `/workspace` prefixes. "
                    "If you need multiple shell commands, combine them into one `bash -lc 'cmd1 ; cmd2'` invocation. "
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
