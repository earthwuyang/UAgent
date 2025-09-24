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
import os
import time
import shlex
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.llm_client import LLMClient
import contextlib
from ..utils.json_utils import safe_json_loads
from ..integrations.openhands_runtime import OpenHandsActionServerRunner


TOOL_SPEC = """
You can call these tools; reply ONLY with a single tool call in the following format.
Guidance:
- If the target file does not exist or the directory is empty, your next action should be to CREATE the file using str_replace_editor with command=create.
- Prefer creating/editing files under the code/ directory when implementing new functionality.
- Avoid repeatedly listing or viewing the same paths; move on to create or edit as needed.
- For any network operation (git clone / wget / curl / pip install), and only when a local proxy is required,
  prefix your shell command with `source scripts/net_proxy.sh &&`. Keep purely local commands proxy-free.

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

TOOL_SELECTION_RUBRIC = (
    "Tool selection rules (soft):\n"
    "1) If the workspace snapshot shows no code files under `code/`, your FIRST non-trivial action should be one of:\n"
    "   - str_replace_editor.create (create a new file under code/)\n"
    "   - str_replace_editor.write  (write initial content into a path under code/)\n"
    "2) Avoid repeating directory listings if no new files appear; take a constructive action (create/write/edit).\n"
    "3) Translate unsupported verbs: list/lsdir -> run 'ls -pa'; cat -> file_read; edit/write/create -> str_replace_editor.\n"
    "4) For network downloads/builds (git/wget/curl/pip), opt-in to proxy ONLY when needed by prefixing: `source scripts/net_proxy.sh && <command>`.\n"
    "5) For pip installs on slow networks, prefer the Tsinghua mirror: add `-i https://pypi.tuna.tsinghua.edu.cn/simple` (and extra wheel indexes when appropriate).\n"
    "6) Avoid large source builds (torch/dgl) when wheels exist; prefer official wheel indexes (e.g., PyTorch CPU wheels via `--extra-index-url https://download.pytorch.org/whl/cpu`, DGL via `-f https://data.dgl.ai/wheels/repo.html`).\n"
)

SUPERVISOR_PROMPT = (
    "Workspace rules:\n"
    "- Use ONLY the current workspace; do NOT create plan_plan_* directories; do NOT clone the OpenHands repo; do NOT delete openhands_session_* directories.\n"
    "- All outputs must live under this workspace. Use $WORKSPACE_ROOT (from workspace/env.sh). Never try to write to /workspace.\n\n"
    "Acceptance criteria (must pass):\n"
    "- After running code/collect_data.py (or equivalent), create non-empty artifacts under experiments/<exp_id>/results (e.g., .csv/.jsonl/.parquet).\n"
    "- Write a non-trivial log to experiments/<exp_id>/logs/data_collection.log (include errors if any).\n"
    "- Exit rc=0 and print an artifact index (paths + file sizes) or summarize where outputs are written.\n"
    "- If artifacts are missing/empty or the script prints placeholder text (e.g., 'Nothing to do yet'), treat as failure: summarize the log tail and propose the next concrete fix (missing deps, proxy, path, etc.).\n\n"
    "Anti-stagnation:\n"
    "- Do not repeat list/read when outputs are identical across two attempts. Create or write the needed file(s) under code/ and proceed to execution.\n"
)


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
    def _normalize_tool_call(func: str, params: Dict[str, str]) -> tuple[str, Dict[str, str]]:
        f = (func or "").strip().lower()
        p = dict(params or {})
        # aliases to supported tools
        if f in {"list", "lsdir", "dir"}:
            f = "execute_bash"
            p.setdefault("command", "ls -pa")
        elif f in {"cat", "show", "view_file"}:
            f = "file_read"
            # ensure path exists in params
            if not p.get("path") and p.get("command"):
                p["path"] = p.pop("command")
        elif f in {"create", "write", "edit"}:
            # unify under str_replace_editor with explicit sub-command
            sub_cmd = f
            f = "str_replace_editor"
            p.setdefault("command", sub_cmd)
        return f, p

    @staticmethod
    def _is_placeholder_text(text: Optional[str]) -> bool:
        """Heuristic to detect obviously non-production placeholder content.

        This check is intentionally conservative to avoid blocking legitimate
        files that merely contain words like "placeholder" or "todo" in
        comments (e.g., env scripts). We only flag when short texts contain
        clear placeholder phrases.
        """
        if not isinstance(text, str):
            return False
        t = text.strip()
        if not t:
            return False
        # Only flag very short files/snippets as placeholders
        if len(t) > 1000:
            return False
        tl = t.lower()
        patterns = [
            r"\bthis is a placeholder\b",
            r"\bstub implementation\b",
            r"\bskeleton\b",
            r"\btemplate only\b",
            r"\bexample only\b",
            r"\bdummy implementation\b",
            r"\bboilerplate\b",
            r"^\s*todo\b",
            r"^\s*#\s*todo\b",
        ]
        try:
            import re as _re
            return any(_re.search(pat, tl, _re.IGNORECASE | _re.MULTILINE) for pat in patterns)
        except Exception:
            # Fallback: very conservative contains check
            fallback_tokens = [
                "this is a placeholder",
                "stub implementation",
                "skeleton",
                "template only",
                "example only",
                "dummy implementation",
                "boilerplate",
            ]
            return any(tok in tl for tok in fallback_tokens)

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
                prompt_with_rubric = f"{prompt}\n\n{TOOL_SELECTION_RUBRIC}\n\n{SUPERVISOR_PROMPT}"
                # Guard LLM latency with a timeout so we don't stall after action server is ready
                llm_timeout = int(os.getenv("CODEACT_LLM_TIMEOUT", "60"))
                try:
                    raw = await asyncio.wait_for(
                        self.llm.generate(prompt_with_rubric, max_tokens=800, temperature=0.2),
                        timeout=llm_timeout,
                    )
                except asyncio.TimeoutError:
                    timeout_msg = f"LLM planning timed out after {llm_timeout}s; retrying with stricter instruction"
                    if progress_cb:
                        try:
                            await progress_cb("tool_result_update", {"tool": "planning", "success": None, "observation_preview": timeout_msg})
                        except Exception:
                            pass
                    # Continue with stricter instruction path below
                    raw = ""
                try:
                    func, params = _parse_tool_call(str(raw))
                except Exception:
                    # Ask again with stricter instruction
                    strict_prompt = prompt_with_rubric + "\nRespond with exactly one tool call as specified."
                    try:
                        raw = await asyncio.wait_for(
                            self.llm.generate(strict_prompt, max_tokens=600, temperature=0.1),
                            timeout=llm_timeout,
                        )
                    except asyncio.TimeoutError:
                        raw = ""
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

                # normalize aliases before executing
                func, params = self._normalize_tool_call(func, params)

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
                observation_text, ok = await self._execute_tool(
                    session, func, params, timeout_per_action, progress_cb, str(workspace_path)
                )
                # If a collector stub is detected, treat as failure and nudge the model to implement real logic
                if func == "execute_bash":
                    cmd_lower = params.get("command", "").lower()
                    if "code/collect_data.py" in cmd_lower and "nothing to do yet" in observation_text.lower():
                        hint = (
                            "Detected a stub collector. Replace code/collect_data.py with production-ready logic that fetches/generates real data, "
                            "writes non-empty artifacts under experiments/<exp_id>/results, logs to experiments/<exp_id>/logs/data_collection.log, and exits rc=0."
                        )
                        ok = False
                        transcript.append(hint)
                        try:
                            with transcript_file.open("a", encoding="utf-8") as fp:
                                fp.write(json.dumps({
                                    "step_index": step_idx,
                                    "tool": func,
                                    "params": params,
                                    "success": False,
                                    "observation": hint,
                                    "reason": "collector_stub"
                                }) + "\n")
                        except Exception:
                            pass
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
            "Do not fabricate or simulate results; collect real outputs by running actual commands and programs. "
            "Avoid inserting placeholder, template, or boilerplate content; write production-ready code with concrete logic and error handling. "
            "Avoid randomly generated values (e.g. random.uniform, np.random) unless explicitly required. "
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
        progress_cb: Optional[Any] = None,
        workspace_base: Optional[str] = None,
    ) -> Tuple[str, bool]:
        # No OpenHands imports in parent; use Session helpers that build raw action dicts
        try:
            # Heartbeat task to keep UI responsive on long operations
            heartbeat_task: Optional[asyncio.Task] = None
            heartbeat_running = True
            start = asyncio.get_event_loop().time()

            async def _heartbeat():
                # Send periodic tool_result_update events while waiting
                while heartbeat_running:
                    await asyncio.sleep(15)
                    if not heartbeat_running:
                        break
                    if progress_cb:
                        try:
                            elapsed = int(asyncio.get_event_loop().time() - start)
                            await progress_cb(
                                "tool_result_update",
                                {
                                    "tool": func,
                                    "success": None,
                                    "observation_preview": f"[heartbeat] running {elapsed}s...",
                                },
                            )
                        except Exception:
                            # Heartbeats are best-effort
                            pass

            # Start heartbeat for potentially long tools
            if func in {"execute_bash"}:
                heartbeat_task = asyncio.create_task(_heartbeat())

            if func == "execute_bash":
                cmd = params.get("command", "")
                # Enforce per-command timeout (default 120s). On timeout, consult LLM for next action.
                base_timeout = int(os.getenv("CODEACT_PER_COMMAND_TIMEOUT", "120"))
                eff_timeout = base_timeout
                # Guard against cloning OpenHands repo – it's already integrated
                low = cmd.lower()
                if "git clone" in low and ("openhands" in low or "all-hands-ai/openhands" in low):
                    heartbeat_running = False
                    if heartbeat_task:
                        heartbeat_task.cancel()
                        with contextlib.suppress(Exception):
                            await heartbeat_task
                    return (
                        "Cloning the OpenHands repository is not required (runtime already integrated). "
                        "Operate within the current workspace and implement the task logic instead.",
                        False,
                    )
                # Wrap command to stream output to a stable log file inside workspace
                # so the heartbeat can tail and forward chunks to the UI.
                log_rel = "logs/codeact_exec.log"
                wrapped = f"bash scripts/run_with_logs.sh {shlex.quote(cmd)} logs codeact_exec.log"

                # Tail the log file during heartbeat
                log_abs = None
                try:
                    ws_base = Path(workspace_base) if workspace_base else None
                    if ws_base:
                        log_abs = ws_base / log_rel
                except Exception:
                    log_abs = None

                last_size = 0

                async def _stream_tail() -> None:
                    nonlocal last_size
                    if not progress_cb:
                        return
                    if not log_abs or not log_abs.exists():
                        return
                    try:
                        # Read new bytes since last_size
                        with log_abs.open("rb") as fp:
                            fp.seek(last_size)
                            chunk = fp.read(4096)
                            if not chunk:
                                return
                            last_size += len(chunk)
                            text = chunk.decode(errors="ignore")
                            preview = text[-400:]
                            await progress_cb(
                                "tool_result_update",
                                {
                                    "tool": func,
                                    "success": None,
                                    "observation_preview": preview,
                                    "log_path": str(log_rel),
                                },
                            )
                    except Exception:
                        # Best-effort; ignore tailing errors
                        return

                # Upgrade heartbeat to also stream log tail
                original_hb = _heartbeat

                async def _heartbeat_with_tail():
                    while heartbeat_running:
                        try:
                            if progress_cb:
                                await progress_cb(
                                    "tool_result_update",
                                    {"tool": func, "success": None, "observation_preview": "[heartbeat] running..."},
                                )
                            await _stream_tail()
                        except Exception:
                            pass
                        await asyncio.sleep(5)

                # Start heartbeat for potentially long tools (streaming tail)
                # Cancel any previously-started generic heartbeat to avoid duplicate updates
                try:
                    if heartbeat_task:
                        heartbeat_running = False
                        heartbeat_task.cancel()
                        with contextlib.suppress(Exception):
                            await heartbeat_task
                except Exception:
                    pass
                heartbeat_running = True
                heartbeat_task = asyncio.create_task(_heartbeat_with_tail())

                # Ensure logs directory exists before running
                try:
                    if workspace_base:
                        (Path(workspace_base) / "logs").mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                try:
                    res = await session.run_cmd(wrapped, timeout=eff_timeout, blocking=True)
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.TimeoutException, asyncio.TimeoutError) as texc:
                    # Notify progress stream about timeout
                    if progress_cb:
                        try:
                            await progress_cb(
                                "tool_result_update",
                                {
                                    "tool": func,
                                    "success": None,
                                    "observation_preview": f"[timeout] command exceeded {eff_timeout}s: {cmd[:140]}",
                                },
                            )
                        except Exception:
                            pass
                    # Ask LLM how to proceed
                    advise = await self._advise_on_timeout(cmd, eff_timeout, log_rel)
                    # Try to follow LLM advice if it suggests increasing timeout
                    if advise.get("action") == "increase_timeout":
                        try_timeout = int(advise.get("new_timeout", eff_timeout * 2))
                        try_timeout = max(try_timeout, eff_timeout + 60)
                        try_timeout = min(try_timeout, int(os.getenv("CODEACT_MAX_TIMEOUT", "1800")))
                        if progress_cb:
                            try:
                                await progress_cb(
                                    "tool_result_update",
                                    {
                                        "tool": func,
                                        "success": None,
                                        "observation_preview": f"[retry] increasing timeout to {try_timeout}s per LLM advice: {advise.get('reason','')}"[:400],
                                    },
                                )
                            except Exception:
                                pass
                        res = await session.run_cmd(wrapped, timeout=try_timeout, blocking=True)
                    else:
                        # Return timeout as failure
                        return (f"[timeout] {cmd} after {eff_timeout}s. Advice: {advise}", False)
                content = self._render_action_output(res)
                return content, res.execution_result.success

            if func == "str_replace_editor":
                command = params.get("command", "").strip()
                path = params.get("path", "").strip()
                if command == "view":
                    res = await session.file_read(path, timeout=timeout)
                    content = self._render_action_output(res)
                    return content, res.execution_result.success

                # Production-use note: if content looks like a trivial placeholder, warn but allow
                if self._is_placeholder_text(params.get("file_text")):
                    warn_note = (
                        "[note] content appears minimal/placeholder-like; proceeding as requested. "
                        "Ensure follow-up steps replace scaffolding with real logic."
                    )
                    # Do not block; continue to write below

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
                # Warn but allow for short placeholder-like writes
                if self._is_placeholder_text(content_text):
                    content_text = (
                        "# NOTE: initial minimal content; will be expanded in subsequent steps\n" + content_text
                    )
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
        finally:
            # stop heartbeat
            try:
                heartbeat_running = False
                if heartbeat_task:
                    heartbeat_task.cancel()
                    with contextlib.suppress(Exception):
                        await heartbeat_task
            except Exception:
                pass


    async def _advise_on_timeout(self, command: str, timeout_s: int, log_rel_path: str) -> Dict[str, Any]:
        """Ask the LLM how to proceed after a timeout.

        Returns a dict with fields:
        - action: 'increase_timeout' | 'background' | 'abort'
        - new_timeout: int (optional)
        - reason: str
        """
        prompt = (
            "A shell command timed out. Provide a JSON with fields:\n"
            "{\n  \"action\": one of ['increase_timeout','background','abort'],\n"
            "  \"new_timeout\": integer seconds if action=='increase_timeout',\n"
            "  \"reason\": short explanation\n}\n\n"
            f"Command: {command}\n"
            f"Timeout: {timeout_s}s\n"
            f"Log file: {log_rel_path}\n"
            "Keep the advice minimal, valid JSON only."
        )
        try:
            resp = await self.llm.generate(prompt, max_tokens=200, temperature=0.1, response_format={"type":"json_object"})
        except Exception:
            resp = await self.llm.generate(prompt, max_tokens=200, temperature=0.1)
        try:
            data = safe_json_loads(resp)
            if isinstance(data, dict):
                action = str(data.get("action","increase_timeout")).strip()
                if action not in {"increase_timeout","background","abort"}:
                    action = "increase_timeout"
                nt = data.get("new_timeout")
                reason = str(data.get("reason",""))
                return {"action": action, "new_timeout": nt, "reason": reason}
        except Exception:
            pass
        return {"action": "increase_timeout", "new_timeout": timeout_s * 2, "reason": "Fallback advice"}

__all__ = ["CodeActRunner"]
