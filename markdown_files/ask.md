UAgent backend — request for deep diagnosis plan (OpenHands V2, no CodeAct)

Context (current state)
- Legacy CodeAct loop removed; all actions go through OpenHandsClientV2 + CodeActRunnerV2.
- Data collection flow:
  1) RAS plan (experiments/<design_id>/ras_spec.json) if present.
  2) Otherwise, author one Python script with LLM (single prompt), write to code/collect_data_<exec>.py via V2, run python3, and parse exactly one stdout line "Final result: {json}".
- We persist LLM prompt/output artifacts and (on parse failure) the script stdout under experiments/<id>/logs/ for debugging.

Problems observed now (with examples)
1) V2 write intermittently fails, although server claims success
- Logs:
  - [CodeAct] Executing write action … POST http://127.0.0.1:<port>/execute_action → 200 OK
  - [CodeAct] Received response for write → action=write exit_code=-1 success=False output=
  - FULL_RESPONSE: {"message": "I wrote to the file …", "observation": "write", "extras": {"path": "…/code/collect_data_exec_196b8477.py"}}
- Our client then throws "Failed to write data collection script via V2".
- Hypothesis: we are mixing endpoints (/execute_action legacy vs /actions SSE). The response is a non-terminal “message” (no exit_code), so we record exit_code=-1. We need one consistent file-ops path and success mapping.

2) Authored script frequently lacks the exact "Final result:" line
- Failure: "Data collection script did not print a 'Final result:' line".
- The cleaned code (persisted) looks plausible but lacks the final print. stdout (persisted) confirms no such line.
- Parsing is now case-insensitive; we also persist script stdout to logs for triage.

What we already implemented
- V2-only client & deterministic runner (ensure venv, write, run) with persistent session reuse per workspace.
- Better fenced-code cleaning for LLM outputs (handles missing closing ``` and stray fences).
- More tolerant final-line parsing (case-insensitive); stdout persistence on failure.
- Logs of LLM prompt/raw/cleaned and script stdout for each attempt.

What we need from you (concrete patch plan)
1) File-ops protocol unification & terminalization
- Switch all file-ops to /actions SSE using str_replace_editor.{create,replace} (or native write tool if supported) and define success semantics:
  - Treat observation+stream-close as success for file-ops; or
  - Require final_result.exit_code == 0.
- Ensure OpenHandsClientV2 never calls /execute_action for write. Provide explicit mapping for create/replace/read.
- Instrument request/response and the last event type to a single JSONL trace per workspace.

2) Write fallback (only if primary fails)
- If the primary file-op returns non-terminal or exit_code<0, fallback to a bash heredoc via V2 run (mkdir -p + cat <<'EOF' … EOF). Log primary/fallback attempts to identify which layer fails.

3) Authoring contract enforcement
- Keep LLM authoring but enforce an epilogue append on our side when the script lacks the print:
  print('Final result: ' + json.dumps(result, separators=(',',':')))
- Continue persisting prompt/raw/cleaned for diagnosis; add a diff trace if we append the epilogue.

4) Path & allowed-roots
- Normalize to relative workspace paths for file-ops (code/..., not host-absolute) to avoid policy mismatches.

5) Minimal test matrix
- File-ops SSE:
  - create → success on observation+close
  - replace → success with correct content
  - outside roots → PATH_DENIED fast
- Authoring:
  - Fenced block without closing fence → cleaned and written or rejected with logs
  - Missing final print → epilogue appended, final line printed, stdout persisted
- Run:
  - Script prints final line exactly once → parsed and normalized

Key references & logs to inspect
- Scientific engine: backend/app/core/research_engines/scientific_research.py (_collect_experimental_data)
- V2 client: backend/app/integrations/openhands_runtime.py (OpenHandsClientV2)
- Runner: backend/app/services/codeact_runner.py (CodeActRunnerV2)
- Example write failure:
  [CodeAct] Executing write action with action_time[0/9996] http_timeout=60s
  POST /execute_action → 200
  Received response for write
  action=write exit_code=-1 success=False output=
  FULL_RESPONSE: {"message": "I wrote to the file …", "observation": "write", "extras": {"path": "…/code/collect_data_exec_196b8477.py"}}

Deliverable
Please provide exact code patches and a short test plan to:
- Route all file-ops to the correct SSE endpoint and confirm terminalization logic for editor ops.
- Add the bash heredoc fallback and logging when the primary write fails.
- Append a minimal epilogue (if needed) to guarantee the single "Final result:" line, while still logging unmodified LLM output for audit.
- Include a few unit/integration checks to verify success and reproduce prior failures.

