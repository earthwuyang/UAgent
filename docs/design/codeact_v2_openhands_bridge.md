# CodeAct v2 + OpenHands Bridge (Deterministic, Idempotent, Testable)

## Goals
- Deterministic action semantics (no string-heuristics).
- Idempotent file API (no create-vs-write loops).
- Explicit terminal signaling for all tools.
- Job service for long `execute_bash` with polling, de-dup & cancellation.
- One-shot, idempotent environment bootstrap (venv/pip).
- Clear observability: typed JSONL event trace.
- Strict allowed-paths contract (deny fast, never hang).
- Backwards-compatible shim for existing prompts.

## Architecture Overview

LLM (CodeAct)
→ CodeAct Runner (state machine)
→ OpenHands Bridge (plan→steps coordination)
→ OpenHands Client V2 (typed protocol)
→ OpenHands Action Server (unchanged)

### Action Envelope & Result

```
{
  "id": "uuid",
  "tool": "str_replace_editor" | "run" | "read" | "write" | "view",
  "args": {...},
  "timeout_sec": 90,
  "cwd": "/workspace",
  "meta": {"workspace_id": "...", "correlation_id": "..."}
}
```

Result:
```
{
  "id": "uuid",
  "tool": "write",
  "success": true,
  "exit_code": 0,
  "duration_ms": 121,
  "stdout": "",
  "stderr": "",
  "files_created": [],
  "files_modified": [],
  "artifact_paths": [],
  "error": null
}
```

Terminalization:
- run/execute_bash: must have final_result with exit_code; otherwise SERVER_ERROR.
- file-ops: accept clean close + last observation as success (exit_code=0). If no obs and no final_result → SERVER_ERROR.

### File API Semantics (idempotent)
- create_if_absent(path, content) → success if newly created; if exists, no-op success.
- write(path, content, overwrite=true) → atomic write.
- append(path, content)
- read(path[, range])
- stat(path)
- exists(path)

All file ops:
- Enforce allowed roots locally; return PATH_DENIED fast.
- Pre-mkdir parents for write/append/create.

### Job Service (execute_bash)
- start_job(cmd) → job_id
- poll_job(job_id) → RUNNING|SUCCEEDED|FAILED + tail
- wait_job(job_id) → terminal ActionResult
- De-dup content hash (cmd+env+cwd) to avoid re-running bootstrap.

### Environment Bootstrap (one-shot)
- Canonical interpreter: ./.venv/bin/python and ./.venv/bin/pip.
- ensure_venv(): if stamp indicates ready, no-op; else create venv + install requirements.txt.

### Repetition control
- Per-workspace command hash counters; skip identical command after 2 runs.

### Allowed paths
- Explicit allowed_write_roots; never hang on disallowed path.

### Observability
- Single JSONL per workspace logs/openhands_trace.jsonl with request/event/result/error records.

### Error taxonomy
- TIMEOUT, PATH_DENIED, TOOL_UNSUPPORTED, SERVER_ERROR, CLIENT_ERROR, NETWORK_PROXY, INTERRUPTED.

## Compatibility
- New client/runner added alongside legacy; toggle with env.

## Tests
- File-op observation-close is success.
- PATH_DENIED on outside roots.
- run de-dup short-circuits on repetition.

