"""Minimal CodeAct-style runner using OpenHands actions and an LLM.

This runner instructs the LLM to emit tool calls in the OpenHands
function-call text format and executes them against the action server,
feeding observations back for iterative repair.
"""

from __future__ import annotations

import asyncio
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMClient
from ..integrations.openhands_runtime import OpenHandsActionServerRunner


TOOL_SPEC = """
You can call these tools; reply ONLY with a single tool call in the following format:

<function=execute_bash>
<parameter=command>
python3 code/collect_data.py
</parameter>
</function>

<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>code/collect_data.py</parameter>
<parameter=file_text>
print('hello')
</parameter>
</function>

Tools:
- execute_bash: run a shell command
  <function=execute_bash><parameter=command>echo hello</parameter></function>
- str_replace_editor: file editor for create/str_replace/insert/write/view
  - create a file:
    <function=str_replace_editor><parameter=command>create</parameter><parameter=path>code/demo.py</parameter><parameter=file_text>print('hi')</parameter></function>
  - view (read) a file:
    <function=str_replace_editor><parameter=command>view</parameter><parameter=path>code/demo.py</parameter></function>
  - write full content:
    <function=str_replace_editor><parameter=command>write</parameter><parameter=path>code/demo.py</parameter><parameter=content>print('ok')</parameter></function>
  - str_replace in a file:
    <function=str_replace_editor><parameter=command>str_replace</parameter><parameter=path>code/demo.py</parameter><parameter=old_str>foo</parameter><parameter=new_str>bar</parameter></function>
  - insert new_str after a line number:
    <function=str_replace_editor><parameter=command>insert</parameter><parameter=path>code/demo.py</parameter><parameter=insert_line>10</parameter><parameter=new_str>print('new')</parameter></function>
- file_read: alias to read a file
  <function=file_read><parameter=path>code/demo.py</parameter></function>
- ipython_run: run an IPython cell within the workspace python kernel
  <function=ipython_run><parameter=code>print('hello')</parameter></function>
- finish: conclude with a final message
  <function=finish><parameter=message>Task complete</parameter></function>
""".strip()


FINISH_NAME = "finish"


def _parse_tool_call(content: str) -> Tuple[str, Dict[str, str]]:
    content = content.strip()
    func_match = re.search(r"<function=([a-zA-Z0-9_\-]+)>(.*?)</function>", content, re.DOTALL)
    if not func_match:
        raise ValueError("No function call found in response")
    func_name = func_match.group(1).strip()
    inner = func_match.group(2)
    params = {}
    for m in re.finditer(r"<parameter=([a-zA-Z0-9_\-]+)>(.*?)</parameter>", inner, re.DOTALL):
        key = m.group(1).strip()
        val = m.group(2).strip()
        params[key] = val
    return func_name, params


@dataclass
class CodeActStep:
    thought: str
    tool: str
    params: Dict[str, str]
    observation: str
    success: bool


class CodeActRunner:
    def __init__(self, llm_client: LLMClient, action_runner: OpenHandsActionServerRunner):
        self.llm = llm_client
        self.action_runner = action_runner

    @staticmethod
    def _render_action_output(action_result) -> str:
        parts: list[str] = []

        def _append(text: Optional[str]) -> None:
            if isinstance(text, str):
                stripped = text.strip()
                if stripped and stripped not in parts:
                    parts.append(stripped)

        _append(getattr(action_result, "stdout", None))
        _append(getattr(action_result, "stderr", None))

        exec_result = getattr(action_result, "execution_result", None)
        if exec_result is not None:
            _append(getattr(exec_result, "stdout", None))
            _append(getattr(exec_result, "stderr", None))

        if not parts:
            raw = getattr(action_result, "raw_observation", None)
            if isinstance(raw, dict):
                for key in ("content", "stdout", "stderr"):
                    value = raw.get(key)
                    _append(value)
            elif raw is not None:
                _append(str(raw))

        if not parts:
            command_text = None
            exec_result = getattr(action_result, "execution_result", None)
            if exec_result is not None:
                command_text = getattr(exec_result, "command", None)
            if not command_text:
                command_text = getattr(action_result, "command", None)
            if command_text:
                parts.append(f"[command produced no output] {command_text}")
            else:
                parts.append("[command produced no observable output]")

        return "\n".join(parts).strip()

    async def run(
        self,
        workspace_path,
        goal: str,
        max_steps: int = 8,
        timeout_per_action: int = 180,
        progress_cb: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not self.action_runner.is_available:
            return {"success": False, "error": "OpenHands action runner is unavailable"}

        session = await self.action_runner.open_session(workspace_path)
        workspace_path = Path(workspace_path)
        logs_dir = workspace_path / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logs_dir = workspace_path
        transcript_file = logs_dir / "codeact_steps.jsonl"
        steps: List[CodeActStep] = []
        transcript: List[str] = []
        try:
            for step_idx in range(1, max_steps + 1):
                prompt = self._build_prompt(goal, transcript)
                if progress_cb:
                    try:
                        await progress_cb("planning", {"step": step_idx, "goal": goal})
                    except Exception:
                        pass
                raw = await self.llm.generate(prompt, max_tokens=800, temperature=0.2)
                try:
                    func, params = _parse_tool_call(str(raw))
                except Exception:
                    # Ask again with stricter instruction
                    strict_prompt = prompt + "\nRespond with exactly one tool call as specified."
                    raw = await self.llm.generate(strict_prompt, max_tokens=600, temperature=0.1)
                    try:
                        func, params = _parse_tool_call(str(raw))
                    except Exception:
                        message = await self._fallback_plain_response(goal, transcript, raw)
                        fallback_step = CodeActStep(
                            thought="fallback_final",
                            tool="fallback_plain",
                            params={},
                            observation=message,
                            success=bool(message.strip()),
                        )
                        steps.append(fallback_step)
                        transcript.append(f"FALLBACK: {message}")
                        try:
                            with transcript_file.open("a", encoding="utf-8") as fp:
                                fp.write(
                                    json.dumps(
                                        {
                                            "step_index": step_idx,
                                            "tool": "fallback_plain",
                                            "params": {},
                                            "success": bool(message.strip()),
                                            "observation": message,
                                        }
                                    )
                                    + "\n"
                                )
                        except Exception:
                            pass
                        if progress_cb:
                            try:
                                await progress_cb(
                                    "finish",
                                    {"message": message, "mode": "fallback_no_tool"},
                                )
                            except Exception:
                                pass
                        return {
                            "success": bool(message.strip()),
                            "message": message,
                            "steps": [s.__dict__ for s in steps],
                            "fallback_used": True,
                        }

                if func == FINISH_NAME:
                    message = params.get("message", "")
                    steps.append(CodeActStep(thought="finish", tool=func, params=params, observation=message, success=True))
                    transcript.append(f"FINISH: {message}")
                    if progress_cb:
                        try:
                            await progress_cb("finish", {"message": message})
                        except Exception:
                            pass
                    return {"success": True, "message": message, "steps": [s.__dict__ for s in steps]}

                if progress_cb:
                    try:
                        await progress_cb("tool_call", {"tool": func, "params": params})
                    except Exception:
                        pass
                observation_text, ok = await self._execute_tool(session, func, params, timeout_per_action)
                step_record = CodeActStep(thought="", tool=func, params=params, observation=observation_text, success=ok)
                steps.append(step_record)
                transcript.append(f"TOOL {func}: {params}\nOBS:\n{observation_text}")
                try:
                    with transcript_file.open("a", encoding="utf-8") as fp:
                        fp.write(
                            json.dumps(
                                {
                                    "step_index": step_idx,
                                    "tool": func,
                                    "params": params,
                                    "success": ok,
                                    "observation": observation_text,
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                if progress_cb:
                    try:
                        await progress_cb(
                            "tool_result",
                            {"tool": func, "success": ok, "observation_preview": observation_text[:400]},
                        )
                    except Exception:
                        pass

            return {"success": False, "error": "Max steps reached", "steps": [s.__dict__ for s in steps]}
        finally:
            await session.close()

    async def _fallback_plain_response(
        self,
        goal: str,
        transcript: List[str],
        raw: Any,
    ) -> str:
        """Produce a plain-text response when the model cannot emit tool calls."""
        message = str(raw or "").strip()
        if message and "<function=" not in message:
            return message

        fallback_prompt = self._build_fallback_prompt(goal, transcript)
        try:
            generated = await self.llm.generate(
                fallback_prompt,
                max_tokens=600,
                temperature=0.3,
            )
            message = str(generated or "").strip() or message
        except Exception:
            # Keep the best effort response even if the retry fails
            pass

        return message or "Unable to complete the request without tool calls."

    def _build_prompt(self, goal: str, transcript: List[str]) -> str:
        history = "\n\n".join(transcript[-6:]) if transcript else ""
        instructions = (
            "You are a CodeAct agent working in a Linux workspace. "
            "Use tools to create/edit files and run bash. Ensure code is runnable and executed. "
            "When executing Python, prefer `python3 ...` from bash. "
            "If an error occurs, inspect files/logs and fix by editing. "
            "No Simulation Policy: Do NOT fabricate or simulate results. Do NOT use 'simulate', 'simulation', 'mock', 'placeholder', "
            "'synthetic', 'random.uniform', 'np.random' or similar. Collect real outputs by running real commands/programs. "
            "Every step must operate on the actual workspace and produce verifiable artifacts/logs."
        )
        return (
            f"{instructions}\n\nGoal:\n{goal}\n\n"
            f"Recent transcript:\n{history}\n\n"
            f"Tools:\n{TOOL_SPEC}\n\n"
            "Respond with exactly one tool call."
        )

    def _build_fallback_prompt(self, goal: str, transcript: List[str]) -> str:
        history = "\n\n".join(transcript[-6:]) if transcript else "(no prior tool calls)"
        return (
            "Tool execution mode is unavailable. Provide the best possible final answer "
            "in plain text, summarizing the actions or guidance needed to achieve the goal. "
            "Do not include any <function> tags or tool specifications.\n\n"
            f"Goal:\n{goal}\n\n"
            f"Context:\n{history}\n\n"
            "Final answer:"
        )

    async def _execute_tool(
        self,
        session: OpenHandsActionServerRunner.Session,
        func: str,
        params: Dict[str, str],
        timeout: int,
    ) -> Tuple[str, bool]:
        # No OpenHands imports in parent; use Session helpers that build raw action dicts
        try:
            if func == "execute_bash":
                cmd = params.get("command", "")
                res = await session.run_cmd(cmd, timeout=timeout, blocking=True)
                content = self._render_action_output(res)
                return content, res.execution_result.success

            if func == "str_replace_editor":
                command = params.get("command", "").strip()
                path = params.get("path", "").strip()
                if command == "view":
                    res = await session.file_read(path, timeout=timeout)
                    content = self._render_action_output(res)
                    return content, res.execution_result.success

                res = await session.file_edit(
                    path,
                    command,
                    file_text=params.get("file_text"),
                    old_str=params.get("old_str"),
                    new_str=params.get("new_str"),
                    insert_line=int(params.get("insert_line")) if params.get("insert_line") else None,
                    timeout=timeout,
                )
                content = self._render_action_output(res)
                return content, res.execution_result.success

            if func == "file_read":
                path = params.get("path", "").strip()
                res = await session.file_read(path, timeout=timeout)
                content = self._render_action_output(res)
                return content, res.execution_result.success

            if func == "write":
                path = params.get("path", "").strip()
                content_text = params.get("content", "")
                res = await session.file_write(path, content_text, timeout=timeout)
                content = self._render_action_output(res)
                return content, res.execution_result.success

            if func == "ipython_run":
                code = params.get("code", "")
                res = await session.ipython_run(code, timeout=timeout)
                content = self._render_action_output(res)
                return content, res.execution_result.success

            # Unknown function; treat as no-op
            return f"Unknown function: {func}", False
        except Exception as exc:
            return f"Exception during tool execution: {exc}", False


__all__ = ["CodeActRunner"]
