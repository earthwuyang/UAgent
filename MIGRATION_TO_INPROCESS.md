# Migration Guide: OpenHands API → In‑Process Integration

This guide describes how to migrate an existing UAgent/OpenHands integration that calls the OpenHands HTTP/API server to a fully in‑process integration (import and run OpenHands directly in the same Python process).

The end‑state is:
- No HTTP Action Server dependency for core flows.
- In‑process execution via `ActionExecutor` with streaming logs to the workspace.
- Jobs API replaced with blocking `run_cmd` wrappers.
- Scientific research engine executes comprehensive experiments in one in‑process session.


## 1) Prerequisites

- Ensure the OpenHands source repository is available locally and importable:
  - Recommended layout: place `OpenHands/` at the project root so imports like `from openhands.runtime.action_execution_server import ActionExecutor` work.
  - If the repo is not installed as a package, add it to `sys.path`. The runtime helper already does this automatically.
    - See: `backend/app/integrations/openhands_runtime.py:38` and `backend/app/integrations/openhands_inprocess.py:45`.
- Python 3.10+ environment with the same dependencies used by your previous OpenHands setup.
- Network access is not required for core in‑process execution (unless your experiment scripts explicitly fetch resources).


## 2) Remove/Bypass HTTP Action Server usage

- Stop calling the HTTP Action Server runner in your code paths. Prefer the in‑process runner.
- Files to adopt:
  - `backend/app/integrations/openhands_inprocess.py` — primary in‑process runner. It imports `ActionExecutor` and exposes a `Session` with:
    - `send_action(payload, timeout=...)`
    - `run_cmd(command, timeout=..., blocking=True, cwd=...)`
    - `file_read(...)`, `file_write(...)`, `file_edit(...)`
    - Append‑only live logs under `logs/openhands_inprocess` and per‑command logs under `logs/commands`.
- Files where HTTP fallback was removed or discouraged:
  - `backend/app/core/openhands/client.py:70` — `OpenHandsClient` now requires in‑process OpenHands. If imports fail, it raises.
  - `backend/app/integrations/openhands_runtime.py:1268` — V2 client shim removed.


## 3) Replace “jobs” queue with blocking wrappers

- Replace any call sites like `start_job/poll_job/wait_job` with direct `run_cmd` calls on the in‑process session:
  - Before (V2/jobs):
    - `job_id = await client.start_job("python script.py", timeout_sec=...)`
    - `result = await client.wait_job(job_id, ...)`
  - After (in‑process):
    - `session = await OpenHandsInProcessRunner().open_session(workspace_path)`
    - `result = await session.run_cmd("python script.py", timeout=..., blocking=True)`
- Removed components:
  - `_InProcessJobService` and `jobs_*` methods on the in‑process session are gone.
  - If you need background processes, create an `asyncio.create_task` wrapper and keep writing to per‑command logs (see `stream_monitor`).
- Example refactors in this repo:
  - `backend/app/services/codeact_runner.py:20` — new `CodeActRunner` using in‑process `run_cmd`, `file_write/read`.
  - `backend/app/services/proxy_sql_tool.py:9` — uses `OpenHandsInProcessRunner` and `run_cmd` directly to post to the local SQL proxy.


## 4) Acquire and use a long‑lived in‑process session

- Pattern:
  - `runner = OpenHandsInProcessRunner()`
  - `session = await runner.open_session(workspace_path: Path)`
  - Call `session.run_cmd(...)`, `session.file_read(...)`, `session.file_write(...)` as needed.
  - Close when finished: `await session.close()`
- If you want to keep a stable session per workspace, retain the `session` in your calling component and reuse it.


## 5) Workspace, sessions, and paths

- `WorkspaceManager` still provisions workspaces under `uagent_workspaces/`.
- OpenHands in‑process logs are written inside the workspace:
  - `logs/openhands_inprocess/{live_stdout,live_stderr,live_combined}.log`
  - `logs/commands/*.log` for each action with full STDOUT/STDERR snapshots.
- Research engine (backend) writes its own append‑only JSONL log:
  - `{workspace_base}/uagent_workspaces/{session_id}/logs/research/backend_live.log`


## 6) Streaming logs (append‑only)

- Live OpenHands logs are append‑only and never rewritten.
- Each `run`/`edit`/`read` action also creates a per‑command file in `logs/commands/` with an execution header, context, and full response sections.
- Backend research phases log to `backend_live.log` without truncation.


## 7) Scientific Research: comprehensive in one session

- Experiments are sent as a single comprehensive prompt, executed in one in‑process OpenHands session. This allows installations and artifacts to be reused across steps.
- Final artifact expectations:
  - `experiments/{plan_id}/results/final.json` with `success: true` and key fields (data, analysis, conclusions, measurements).
  - `experiments/{plan_id}/README.md` with exact commands and reproduction steps.
- If final.json is incomplete or limitations contradict success, the LLM audit requests a resume attempt with targeted guidance.
- Relevant code:
  - `backend/app/core/research_engines/scientific_research.py:1967` — comprehensive execution + LLM success audit + resume.
  - `backend/app/core/research_engines/scientific_research.py:2624` — stricter codegen prompt that pins instruction and output paths.


## 8) Environment variables and timeouts

- Runner timeouts:
  - `OPENHANDS_ACTION_TIMEOUT` (default: 120)
  - `OPENHANDS_MAX_ACTION_TIMEOUT` (cap for adaptive retries)
  - `OPENHANDS_RUN_MAX_ATTEMPTS` (retry count for adaptive “add -y” on package managers)
  - `OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT` (min timeout for package installs)
- Scientific research engine:
  - `inprocess_max_attempts` (config key) — LLM codegen/run attempts for a single experiment in process
  - `max_resume_attempts` (config key) — comprehensive resume rounds
  - `EXPERIMENTS_PER_HYPOTHESIS` — number of experiments per (combined) hypothesis set


## 9) Common pitfalls and how to avoid them

- Import errors for OpenHands:
  - Ensure `OpenHands/` exists at project root or that the path is added to `sys.path` before importing `openhands.*`.
- `psycopg2` missing in generated scripts:
  - Either bootstrap a venv and `pip install psycopg2-binary` within workspace, or avoid driver dependencies (use `psql` CLI and parse output). Your prompt can nudge the latter.
- `final.json` not found or `success != true`:
  - The codegen prompt now mandates the exact final.json path; ensure your generated script writes there and sets success true only when the experiment actually completes.
- Path mismatches (e.g., writing under `experiments/seqplan_*` vs. `experiments/exp_*`):
  - Use the injected `{plan_id}` paths only; do not hardcode other IDs.
- pg_duckdb vs PostgreSQL 19 incompatibility:
  - If your logs show compile/compat errors, instruct retry attempts to use a known supported Postgres version (e.g., 15) or skip pg_duckdb parts for the minimal viable path.
- `live_combined.log` is empty:
  - This can happen; inspect `logs/commands/*.log` for full per‑action STDOUT/STDERR, which is authoritative.


## 10) Removal checklist (API/V2)

- Remove or ignore HTTP Action Server paths (only in‑process is supported for core flows).
- Remove `OpenHandsClientV2` usages.
- Replace jobs queue patterns with direct `run_cmd` calls as described above.


## 11) Verification steps

1) Ensure OpenHands is importable (one‑time):
   - `python -c 'from openhands.runtime.action_execution_server import ActionExecutor; print("OK")'`
2) Create a workspace via the app and confirm:
   - `uagent_workspaces/<session>/logs/openhands_inprocess/` exists.
3) Trigger a small experiment and tail logs:
   - `tail -f uagent_workspaces/<session>/logs/research/backend_live.log`
   - `tail -f uagent_workspaces/<experiment_session>/logs/commands/*run_python3*.log`
4) Confirm final.json and README.md artifacts under `experiments/{plan_id}/`.


## 12) File pointers in this repo

- In‑process runner:
  - backend/app/integrations/openhands_inprocess.py:67
- Client (in‑process only):
  - backend/app/core/openhands/client.py:70
- Runtime helpers / path setup:
  - backend/app/integrations/openhands_runtime.py:34
- Research engine (comprehensive execution + audit/resume):
  - backend/app/core/research_engines/scientific_research.py:1967
  - backend/app/core/research_engines/scientific_research.py:2624
- Logging (OpenHands):
  - backend/app/integrations/stream_monitor.py:19
- Backend research append‑only log:
  - backend/app/core/research_engines/scientific_research.py:3778


## 13) Stashing and re‑do instructions

If you are redoing this migration from scratch with another model (e.g., glm‑4.6):
- Save current work: `git stash -u`
- Provide this guide to the model.
- Ask it to:
  1) Ensure OpenHands is importable and switch all call sites to `OpenHandsInProcessRunner`.
  2) Remove jobs queue calls and replace with `run_cmd`.
  3) Keep append‑only logging in `logs/openhands_inprocess` and `logs/commands`.
  4) Verify comprehensive scientific flow artifacts are written under `experiments/{plan_id}/`.

That’s it—once these steps are applied, your project runs entirely in‑process with OpenHands.

