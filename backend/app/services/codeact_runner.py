"""Minimal CodeAct-style runner using OpenHands actions and an LLM.

This runner instructs the LLM to emit tool calls in the OpenHands
function-call text format and executes them against the action server,
feeding observations back for iterative repair.
"""

from __future__ import annotations

import asyncio
import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMClient
from ..integrations.openhands_runtime import OpenHandsActionServerRunner

# Read default timeout from environment variable (default: 120 seconds = 2 minutes)
DEFAULT_ACTION_TIMEOUT = int(os.getenv("OPENHANDS_ACTION_TIMEOUT", "120"))


TOOL_SPEC = """
You can call these tools; reply ONLY with a single tool call in the following format.
Guidance:
- If the target file does not exist or the directory is empty, your next action should be to CREATE the file using str_replace_editor with command=create.
- Prefer creating/editing files under the code/ directory when implementing new functionality.
- Avoid repeatedly listing or viewing the same paths; move on to create or edit as needed.

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

    # Try to find function call pattern
    func_match = re.search(r"<function=([a-zA-Z0-9_\-]+)>(.*?)</function>", content, re.DOTALL)
    if not func_match:
        # Check if response was truncated (common cause of parse failure)
        if "<function=" in content and "</function>" not in content:
            raise ValueError(f"Function call appears truncated (missing </function> tag). Response length: {len(content)}")
        elif "<function=" in content:
            raise ValueError(f"Malformed function call syntax in response")
        else:
            raise ValueError(f"No function call found in response (expected <function=...> format)")

    func_name = func_match.group(1).strip()
    inner = func_match.group(2)
    params = {}

    # Parse parameters with better error handling
    for m in re.finditer(r"<parameter=([a-zA-Z0-9_\-]+)>(.*?)</parameter>", inner, re.DOTALL):
        key = m.group(1).strip()
        val = m.group(2).strip()
        params[key] = val

    # Log parsing success for debugging
    import logging
    logging.debug(f"Successfully parsed tool call: {func_name} with {len(params)} parameters")

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
    def _is_placeholder_text(text: Optional[str]) -> bool:
        if not isinstance(text, str):
            return False
        t = text.strip().lower()
        if not t:
            return False
        tokens = [
            "placeholder",
            "this is a placeholder",
            "todo:",
            "todo -",
            "stub implementation",
            "skeleton",
            "template only",
            "example only",
            "dummy implementation",
            "boilerplate",
        ]
        return any(tok in t for tok in tokens)

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
        timeout_per_action: int = DEFAULT_ACTION_TIMEOUT,
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
        repeat_tracker: Dict[tuple, Dict[str, int]] = {}
        steps: List[CodeActStep] = []
        transcript: List[str] = []
        # Inject a lightweight workspace snapshot to steer the model away from repeated listings
        try:
            wp = Path(workspace_path)
            code_dir = wp / "code"
            exp_dir = wp / "experiments"
            code_entries = sorted([p.name for p in code_dir.iterdir()]) if code_dir.exists() else []
            exp_entries = sorted([p.name for p in exp_dir.iterdir()]) if exp_dir.exists() else []
            snapshot = {
                "workspace": str(wp),
                "code_dir_exists": code_dir.exists(),
                "code_entries": code_entries[:20],
                "experiments_dir_exists": exp_dir.exists(),
                "experiments_entries": exp_entries[:20],
            }
            transcript.append("SYSTEM_CONTEXT: " + json.dumps(snapshot))
        except Exception:
            pass
        stagnation_hints_applied = 0
        try:
            for step_idx in range(1, max_steps + 1):
                prompt = self._build_prompt(goal, transcript)
                if progress_cb:
                    try:
                        await progress_cb("planning", {"step": step_idx, "goal": goal})
                    except Exception:
                        pass
                # Increase max_tokens significantly to avoid truncation
                # Many tool calls with file content can be quite long
                # Some responses can be 15000-20000+ characters with detailed analysis and file contents
                raw = await self.llm.generate(prompt, max_tokens=20000, temperature=0.2)
                try:
                    func, params = _parse_tool_call(str(raw))
                except Exception as e1:
                    # Log first parsing failure for debugging
                    import logging
                    logging.warning(f"First parse attempt failed: {e1}. Raw response length: {len(str(raw))}")

                    # Ask again with stricter instruction and even more tokens
                    strict_prompt = prompt + "\nRespond with ONLY a tool call, no explanations. Start with <function= and end with </function>."
                    raw = await self.llm.generate(strict_prompt, max_tokens=15000, temperature=0.1)
                    try:
                        func, params = _parse_tool_call(str(raw))
                    except Exception as e2:
                        # Log second parsing failure
                        logging.error(f"Second parse attempt also failed: {e2}. Falling back to plain response.")
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

                repeat_key = None
                if func == "execute_bash":
                    repeat_key = ("execute_bash", params.get("command", "").strip())
                elif func == "str_replace_editor" and params.get("command") == "view":
                    repeat_key = ("view", params.get("path", "").strip())

                if repeat_key and repeat_key[1]:
                    tracker = repeat_tracker.setdefault(repeat_key, {})
                    count = tracker.get(observation_text, 0) + 1
                    tracker[observation_text] = count
                    if count >= 3:
                        repeat_message = (
                            f"Command '{repeat_key[1]}' produced identical output {count} times."
                        )
                        # Instead of aborting immediately, inject corrective guidance and continue
                        steps.append(CodeActStep(
                            thought="repeat_guard",
                            tool=func,
                            params=params,
                            observation=repeat_message,
                            success=False,
                        ))
                        hint = (
                            "HINT: Stop repeatedly listing/viewing. If the target file does not exist or the directory is empty, "
                            "use str_replace_editor with command=create to create a new Python file under code/ and implement the required logic."
                        )
                        transcript.append(repeat_message)
                        transcript.append(hint)
                        stagnation_hints_applied += 1
                        try:
                            with transcript_file.open("a", encoding="utf-8") as fp:
                                fp.write(
                                    json.dumps(
                                        {
                                            "step_index": step_idx,
                                            "tool": func,
                                            "params": params,
                                            "success": False,
                                            "observation": repeat_message,
                                            "reason": "repeat_guard",
                                        }
                                    )
                                    + "\n"
                                )
                        except Exception:
                            pass
                        if progress_cb:
                            try:
                                await progress_cb(
                                    "tool_result_update",
                                    {"tool": func, "success": False, "observation_preview": repeat_message[:160]},
                                )
                            except Exception:
                                pass
                        # If we've hinted multiple times, bail out to avoid infinite loops
                        if stagnation_hints_applied >= 2:
                            return {
                                "success": False,
                                "error": repeat_message,
                                "steps": [s.__dict__ for s in steps],
                                "repetition_guard": True,
                            }
                        # Otherwise continue to next planning step
                        continue

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
                max_tokens=2000,
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
            "Do not fabricate or simulate results; collect real outputs by running actual commands and programs. "
            "Avoid inserting placeholder, template, or boilerplate content; write production-ready code with concrete logic and error handling. "
            "Avoid randomly generated values (e.g. random.uniform, np.random) unless explicitly required. "
            "Every step must operate on the actual workspace and produce verifiable artifacts/logs."
        )
        return (
            f"{instructions}\n\nGoal:\n{goal}\n\n"
            f"Recent transcript:\n{history}\n\n"
            f"Tools:\n{TOOL_SPEC}\n\n"
            "IMPORTANT: Respond with EXACTLY ONE tool call using the XML format shown above.\n"
            "Your ENTIRE response must be a single tool call, starting with <function= and ending with </function>.\n"
            "Do NOT include any explanatory text before or after the tool call."
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
                # Guard against cloning OpenHands repo – it's already integrated
                low = cmd.lower()
                if "git clone" in low and ("openhands" in low or "all-hands-ai/openhands" in low):
                    return (
                        "Cloning the OpenHands repository is not required (runtime already integrated). "
                        "Operate within the current workspace and implement the task logic instead.",
                        False,
                    )

                # Fix pip install commands to be non-interactive and show progress
                actual_timeout = timeout
                if "pip install" in cmd or "pip3 install" in cmd:
                    # Use a reasonable timeout for pip installs (3 minutes max)
                    actual_timeout = min(max(timeout, 180), 180)  # Max 3 minutes for pip installs
                    import logging
                    logging.info(f"Using pip timeout of {actual_timeout}s for pip install command")

                    # Make pip non-interactive and show progress
                    if "--no-input" not in cmd and "-q" not in cmd:
                        # Add flags to make pip non-interactive and verbose
                        cmd = cmd.replace("pip install", "pip install --no-input --progress-bar on")
                        cmd = cmd.replace("pip3 install", "pip3 install --no-input --progress-bar on")
                        # Also set environment variable to ensure non-interactive
                        cmd = f"export PIP_NO_INPUT=1 && {cmd}"
                        logging.info(f"Modified pip command for non-interactive execution: {cmd[:200]}")

                res = await session.run_cmd(cmd, timeout=actual_timeout, blocking=True)
                content = self._render_action_output(res)
                return content, res.execution_result.success

            if func == "str_replace_editor":
                command = params.get("command", "").strip()
                path = params.get("path", "").strip()
                if command == "view":
                    res = await session.file_read(path, timeout=timeout)
                    content = self._render_action_output(res)
                    return content, res.execution_result.success

                # Production-use guard: reject placeholder content before sending to runtime
                if self._is_placeholder_text(params.get("file_text")):
                    msg = (
                        "Placeholder content detected. Write production-ready code that performs real work, "
                        "with concrete logic, error handling, and verifiable outputs."
                    )
                    return msg, False

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
                if self._is_placeholder_text(content_text):
                    msg = (
                        "Placeholder content detected. Provide a complete, runnable implementation instead of a template."
                    )
                    return msg, False
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
