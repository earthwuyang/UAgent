## Refactor Request: CodeAct + OpenHands Bridge (complete simplification)

Audience: GPT‑5‑Pro — please deliver a complete refactor plan and patch set that removes the brittle heuristics we added and replaces them with a clean, deterministic, idempotent, and testable design for CodeAct execution and the OpenHands bridge.

### Context (what we have today)
- UAgent integrates a CodeAct loop (LLM-driven) with an OpenHands action server via our bridge and runner.
- Actions are sent as discrete calls (HTTP POST) per step. There is no persistent shell; each execute_bash runs in a fresh process.
- File operations (create/view/replace/write) frequently emit “observation” text but not a structured terminal event or exit_code. We tuned the client to infer success.
- Scientific research engine expects “Final result: {json}” and seeks measurements/statistics for hypothesis evaluation.

Key modules to refactor:
- backend/app/services/codeact_runner.py (tool loop, repetition guard, file ops, bash execution)
- backend/app/integrations/openhands_runtime.py (action server client, path remap, timeouts, file/batch actions, job service)
- backend/app/integrations/openhands_bridge.py (plan execution, coordination)

### Pain points we need to eliminate
1) File‑op terminal signal mismatch
   - Server often emits only observation text for edit/view with no exit_code/final_result.
   - Result: false negatives, retries, 120s/1800s timeouts, repeated create/view loops.

2) Create vs write semantics cause loops
   - LLM emits str_replace_editor.create for existing files; server refuses overwrite; loop repeats.

3) Repetition and non‑idempotent bootstrap
   - execute_bash repeats “python3 -m venv … && pip install …” many times because shell state is not persisted; imports aren’t checked.

4) Weak repetition guard
   - Guard compares observation text only; pip/read outputs vary so repeats slip through.

5) Error/Success inference via heuristics
   - We scan logs for “ERROR:”/“could not open file” etc. to downgrade success. This is brittle and increases complexity.

6) Allowed‑roots/paths friction
   - Writes under code/ and experiments/ sometimes stalled due to server path policy; we widened roots in env as a workaround.

7) Proxy/backend failures amplify loops
   - When external services (e.g., proxy at 7890) return 502, the agent often re‑edits/re‑checks files instead of handling the dependency failure cleanly.

### Current band‑aids (we would like to remove)
- Mark edit/write success by scanning observation strings.
- Auto‑convert repeated create → write; track created paths in a set for the run.
- Fallback to heredoc writes on file‑op timeout; pre‑mkdir to avoid ENOENT.
- View fallback via bash cat on timeout.
- Execute_bash idempotent wrapper for venv/pip with import checks.
- Repetition guard for same shell command ≥ 3 times.

These work, but are cumbersome, heuristic, and hard to reason about.

### Desired end‑state (what we want you to design and spec)
- Deterministic, typed action protocol and client:
  - Every action returns a structured ActionResult: {success: bool, exit_code: int, error: {type, message}, files_created/modified, stdout, stderr, duration, artifact_paths}.
  - For file‑ops, the server must always emit a terminal result (no guessing). If server cannot, client must implement a reliable completion rule (not time‑based only).

- Idempotent file API surface:
  - create_if_absent(path, content), write(path, content, overwrite=true), read(path, range?), append(path, content), stat(path), exists(path) — single responsibility, atomic behavior, no semantic ambiguity.

- Job service with de‑dup and polling:
  - execute_bash is submitted as a job with a job_id; status endpoint returns RUNNING|SUCCEEDED|FAILED + tail + exit_code.
  - Optional de‑dup by content hash to avoid re‑running identical bootstrap commands.
  - Background tasks are explicitly represented (no nohup magic in the client).

- Idempotent environment bootstrap per workspace:
  - Canonical interpreter path (./venv/bin/python, ./venv/bin/pip).  Import checks before installing.  One line to ensure venv exists/ready.
  - No reliance on “source venv/bin/activate” across steps.

- Repetition control and backoff baked into the state machine:
  - Count identical commands; short‑circuit on repeated no‑ops; suggest next step.

- Precise allowed‑paths contract:
  - Make writable roots explicit and return a clear error if outside. Do not hang if a path is disallowed.

- Streaming and terminal signalling:
  - Prefer SSE or chunked responses with an explicit final event for all tools. If HTTP/POST only, the server must return a terminal JSON with success + exit_code.

- Observability:
  - Single JSONL event log per run: request→event(s)→result with timestamps and durations.
  - Minimal, privacy‑safe content capture.

### Constraints
- We can refactor our runner/bridge/client freely; we prefer not to fork the upstream action server.
- Keep tool surface compatible with CodeAct prompts (finish, execute_bash, str_replace_editor view/create/replace, write, file_read).
- Avoid heavy external deps; keep code self‑contained and testable.

### Deliverables (what we want from you)
1) Design doc (Markdown) with:
   - Architecture, action protocol, state machine diagrams, error taxonomy.
   - File API semantics (create_if_absent/write/read/append/stat/exists) and examples.
   - Job service contract and client flow for execute_bash with de‑dup.
   - Idempotent bootstrap recipe and how to surface interpreter paths to the LLM (prompt/tool spec).

2) Refactor plan + patches:
   - backend/app/services/codeact_runner.py (loop, repetition control, file/batch ops, idempotent wrappers)
   - backend/app/integrations/openhands_runtime.py (typed results, job client, explicit terminal signalling)
   - backend/app/integrations/openhands_bridge.py (coordination points)
   - Update tool spec exposed to the LLM to reflect the simplified “write” and “create_if_absent” semantics.

3) Tests:
   - Unit tests for action result parsing, file API, job polling, repetition guard.
   - Integration tests that cover: create/write/read, large write, repeated commands, long‑running installs, path denial, and proxy 502 handling without loops.

4) Migration + compatibility:
   - A minimal shim layer so existing prompts still work, while new semantics are used under the hood.
   - Rollback plan and flags to toggle legacy vs new behavior.

### Evidence of issues (from our logs)
- Repetitive pip installs on every step; shell state isn’t persisted; no import checks before install.
- create on existing files triggers “Cannot overwrite with create,” then loops create/view.
- view timeouts repeat; no single reliable terminal signal; multiple 120s/1800s hangs.
- Directory listings (ls -pa) repeat with no progress.
- Proxy at 7890 returns 502; agent responds by re‑editing/reading files rather than handling the dependency failure.

### Non‑goals
- We don’t want more heuristics (log string scanning) or longer timeouts. We want a clean protocol + state machine.

### Success criteria
- No repeated create/view loops; no “hangs” on disallowed paths; no silent “success=false” for successful file‑ops.
- Bootstrap is one‑shot and idempotent; re‑runs are cheap and skip when ready.
- Every action returns a typed, terminal result; background/long tasks are tracked as jobs with status.
- Clear, small JSONL trace suitable for debugging.

### Code pointers
- CodeAct: backend/app/services/codeact_runner.py
- OpenHands bridge/runtime client: backend/app/integrations/openhands_bridge.py, backend/app/integrations/openhands_runtime.py
- Scientific research exec integration: backend/app/core/research_engines/scientific_research.py (experiment execution)

### Final ask
Please produce:
1) A concise design doc aligning to the above end‑state.
2) A patch set that implements the new protocol, job client, and file API semantics, with tests.
3) An updated tool spec (what the LLM sees) that removes ambiguity for file ops and execute_bash.
4) Clear migration notes and toggles so we can roll this out without breaking current runs.

We are happy to iterate quickly — prioritize correctness, determinism, and simplicity over heuristics.
- Using DashScope Qwen for LLM
- Docker container for execution isolation
- WebSocket for real-time communication
- Python 3.11+ environment
- Ubuntu Linux host system

Please provide a detailed implementation plan that addresses these issues systematically, with specific code examples and architecture decisions.
