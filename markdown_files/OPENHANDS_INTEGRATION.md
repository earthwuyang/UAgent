OpenHands V3 Headless Handoff — Integration Plan for UAgent

Summary

- Goal: Hand off each experiment to OpenHands’ headless CodeAct loop (one subprocess per experiment) and collect deterministic artifacts. No UAgent-side CodeAct loop; no Docker; no `pip install openhands-ai`.
- Approach: Use the vendored OpenHands source already present at `OpenHands/`, run `python -m openhands.core.main` with `RUNTIME=local` on the host, set a fixed workspace for each run, and require the agent to write a single `final.json` artifact that UAgent reads after process exit.

Repo Reality Check (current state)

- Upstream source checkout: we will use the official OpenHands repository at `OpenHands/` (as a git submodule or plain clone). The old vendored copy is moved to `OpenHands_bk/` for reference.
- Current integration model (V1/V2):
  - Uses an action execution server (`OpenHands/openhands/runtime/action_execution_server.py`) driven via `OpenHandsActionServerRunner` in `backend/app/integrations/openhands_runtime.py` and related helpers.
  - `ScientificResearchEngine` coordinates planning and code execution through `OpenHandsGoalPlanBridge` and `CodeActRunner` (HTTP/Action API style), still “inside” UAgent.
- Workspace management: `backend/app/core/openhands/workspace_manager.py` handles per-session workspaces; `UAGENT_WORKSPACE_DIR` is used as base in `backend/app/main.py`.
- LLM: UAgent initializes its own LLM client (LiteLLM/OpenAI/Anthropic/etc.). OpenHands (when run headless) will read its own LLM config from environment or config.toml.

Why switch to headless

- Deterministic runs: one process per experiment with explicit exit code and stable artifacts on disk.
- Simpler boundary: UAgent assembles a goal + workspace; OpenHands owns the full CodeAct loop; UAgent only parses artifacts/logs.
- Avoids bespoke `/execute_action` orchestration and brittle parsing of “Final result” lines.

Target Architecture (V3 Bridge)

- `OpenHandsCodeActBridgeV3` (new):
  - Seeds an experiment workspace (`…/experiments/<id>/`), including a short README with the artifact contract.
  - Spawns a subprocess: `python -m openhands.core.main -t <goal> -n <sid> -i <max_steps>` using the vendored source.
  - Environment: `PYTHONPATH` includes `OpenHands/`, `RUNTIME=local`, `WORKSPACE_BASE=<workspace>`, `ENABLE_BROWSER=false` (unless needed), and `LLM_*` env mirrored from UAgent env.
  - Streams stdout/stderr to `logs/openhands_stdout.log` and `logs/openhands_stderr.log`.
  - Enforces an overall wall-clock timeout; on timeout, terminates the process and returns a structured failure.
  - After exit, deterministically inspects artifacts: prefer `experiments/<id>/results/final.json`.

What stays and what moves

- Stays (for now, behind a flag): existing V1/V2 action-server-based classes and tests (`openhands_runtime.py`, `CodeActRunner`, `OpenHandsGoalPlanBridge`) remain for one release.
- Moves: ScientificResearchEngine’s experiment execution path will, when enabled, fully delegate to `OpenHandsCodeActBridgeV3` and consume artifacts only. No UAgent-side tool/action loop.

Detailed Change List

1) Bridge: add `backend/app/integrations/openhands_codeact_bridge_v3.py`

- Core responsibilities:
  - Define `CodeActRunConfig` (goal, workspace path, max steps, max minutes, browser disabled flag, session name) and `CodeActRunSummary` (success, exit code, duration, artifact paths, reason).
  - Build the headless command:
    - Interpreter: `${PYTHON:-python}`
    - Module: `-m openhands.core.main`
    - Args: `-t <goal> -n <session_name> -i <max_steps> --no-auto-continue`
  - Env bridging:
    - `PYTHONPATH=<repo_root>/OpenHands[:$PYTHONPATH]`
    - `RUNTIME=local`
    - `WORKSPACE_BASE=<cfg.workspace>` (forces local runtime to use this directory instead of a temp dir)
    - `ENABLE_BROWSER=false` (optional)
    - LLM wiring: set `LLM_MODEL`, `LLM_API_KEY`, and `LLM_BASE_URL` if available from UAgent env (fallback to `LITELLM_*` when present).
  - Logging and events:
    - Write stdout/stderr to `workspace/logs/openhands_{stdout,stderr}.log`.
    - Optionally capture OpenHands event JSONL if/when upstream exposes a headless flag; otherwise rely on logs.
  - Artifact contract:
    - Require the agent to write a single JSON object to `experiments/<id>/results/final.json`.
    - On success, return the parsed path; on failure, return a structured error with stdout tail for debugging.

2) ScientificResearchEngine: switch to handoff (feature-flagged)

- Location: `backend/app/core/research_engines/scientific_research.py`.
- Add a gated code path (env `UAGENT_OPENHANDS_V3=1` or config) to:
  - Construct a concise goal string from `design.json` and experiment context.
  - Derive a deterministic workspace under the session (reusing `UAGENT_WORKSPACE_DIR` as base).
  - Call `OpenHandsCodeActBridgeV3.run()` to execute the experiment.
  - Parse `final.json` when present; otherwise return a structured failure with logs.
- Keep existing V1/V2 paths intact when the flag is off.

3) Prompt and artifact contract

- Goal template (example):
  - Summarize the experiment name, inputs, data locations, and expected measurements.
  - Hard requirements:
    - Reuse existing data if present; do not fabricate outputs.
    - Write the single file `experiments/<id>/results/final.json` containing fields required by UAgent (we will document these per-task).
    - Only read/write within `{WORKSPACE_BASE}/code`, `{WORKSPACE_BASE}/data`, `{WORKSPACE_BASE}/experiments`, and `{WORKSPACE_BASE}/workspace`.
- The bridge seeds `README_UAGENT.md` under the workspace to restate this contract.

4) LLM provider wiring (headless)

- OpenHands headless loads the default LLM from env (`LLM_*`) or config.toml.
- Bridge maps UAgent env to OpenHands:
  - If `LLM_MODEL`/`LLM_API_KEY`/`LLM_BASE_URL` are set, pass through.
  - Else, if `LITELLM_MODEL`/`LITELLM_API_KEY`/`LITELLM_API_BASE` exist, copy them to `LLM_*`.
- This keeps a single source of truth for LLM credentials and endpoints without editing OpenHands’ config files.

5) Runtime selection and workspace control

- Use OpenHands’ built‑in `local` runtime (no Docker). No custom HostRuntime is needed in this repo because upstream ships `impl/local/local_runtime.py`.
- To ensure reproducibility and idempotency across retries:
  - Set `RUNTIME=local`.
  - Set `WORKSPACE_BASE=<absolute path>` for each run so the runtime operates inside our pre‑created workspace directory, not a temp dir.
  - Set `ENABLE_BROWSER=false` unless the task requires browsing (consistent with serverless/headless use).

6) Observability and logs

- Logs written to `workspace/logs/`:
  - `openhands_stdout.log`
  - `openhands_stderr.log`
- Optional: If we patch/stabilize an events JSONL path upstream, store as `workspace/logs/openhands_events.jsonl`.
- UAgent returns the tail of stdout on failure to aid error diagnosis in the UI/API.

7) Dependencies (host-only; no `pip install openhands-ai`)

- OpenHands runs from source; ensure runtime deps are available in our venv:
  - Already present in `backend/requirements.txt`: `httpx`, `pydantic`, `tenacity`, `libtmux`, FastAPI/uvicorn, etc.
  - Additions if missing in the environment: `jupyter` (required by `LocalRuntime` dependency checks). If this is not installed in your environment, we will include it in a focused `requirements-openhands.txt` and note it in setup docs.
- No Docker required.

8) Configuration and toggles

- New envs (read by UAgent):
  - `UAGENT_OPENHANDS_V3=1` — enable headless handoff path.
  - `UAGENT_OPENHANDS_MAX_STEPS` (default 80), `UAGENT_OPENHANDS_MAX_MINUTES` (default 30), `UAGENT_OPENHANDS_DISABLE_BROWSER=true`.
- OpenHands env set by bridge per run:
  - `RUNTIME=local`, `WORKSPACE_BASE=<workspace>`, `ENABLE_BROWSER=false`, `LLM_*` as above.

9) File changes (planned)

- Add
  - `backend/app/integrations/openhands_codeact_bridge_v3.py` — subprocess wrapper.
  - `requirements-openhands.txt` — only if we need to explicitly add `jupyter` or similar host deps not already present.
- Modify
  - `backend/app/core/research_engines/scientific_research.py` — add V3 code path behind `UAGENT_OPENHANDS_V3`.
  - (Optional) `backend/app/main.py` — wire feature flag defaults from env into the engine config (read-only, no behavior change when flag is off).
- Keep (unchanged for now)
  - `backend/app/integrations/openhands_runtime.py`, `openhands_runtime_improved.py`, `services/codeact_runner.py`, `integrations/openhands_bridge.py` — retained for fallback during migration.

10) Test plan

- Smoke: Minimal goal that writes `experiments/exp_x/results/final.json` and exits < 2 min. Assert: process exit code == 0, `final.json` exists, logs present.
- Timeout: Cap `max_minutes=1` for a longer goal; assert: bridge returns `success=false`, reason=timeout, stdout tail present.
- Path denial: Instruct the agent to write outside workspace; verify OpenHands runtime rejects or fails deterministically; assert nonzero exit + log includes error.
- Idempotency: Re-run same goal in same workspace; ensure `final.json` shape identical and no thrashing.
- Failure reporting: Induce a failing sub-step; assert the agent records failure details in `final.json` rather than fabricating results.

11) Rollout

- Ship behind `UAGENT_OPENHANDS_V3=1` (default OFF initially).
- Keep V1/V2 for one release, then remove once stable.
- Rollback is trivial: unset the flag and the engine uses the legacy path.

Implementation Notes (grounded in this repo)

- Headless entrypoint exists at `OpenHands/openhands/core/main.py` with argparse support (`-t/--task`, `-n/--name`, `-i/--max-iterations`, `--no-auto-continue`).
- Runtime selection is by config/env; we’ll set `RUNTIME=local` via env — no code changes needed in OpenHands.
- Local runtime uses `WORKSPACE_BASE` if set; otherwise it creates temp dirs. We will always set it for deterministic paths.
- Existing code already handles putting `OpenHands/` on `sys.path`; the bridge will replicate this for subprocess via `PYTHONPATH`.
- LLM config: headless reads from env; we map UAgent’s provider/env to `LLM_*` at launch.

Using Official OpenHands (replace vendored copy)

- Rationale: Track official updates with minimal friction, while keeping our integration paths unchanged (`python -m openhands.core.main` and imports via `OpenHands/`).
- One-time migration steps:
  - Back up the current vendored directory:
    - `mv OpenHands OpenHands_bk`
  - Add the official repo at the same path so imports remain stable. Recommended: add as a submodule pinned to `main`:
    - `git submodule add -b main https://github.com/All-Hands-AI/OpenHands.git OpenHands`
    - `git submodule update --init --recursive`
  - Alternatively, if you prefer a plain clone (not a submodule):
    - `git clone https://github.com/All-Hands-AI/OpenHands.git OpenHands`
    - `cd OpenHands && git checkout main`
- Keeping up-to-date (choose one):
  - Submodule workflow:
    - Pull latest upstream and update the pointer in this repo:
      - `git submodule update --remote --merge --recursive`
      - `git add OpenHands && git commit -m "chore: bump OpenHands submodule"`
  - Plain clone workflow:
    - `cd OpenHands && git fetch origin && git checkout main && git pull --ff-only`
- Local patches (if any):
  - Do not modify files directly inside `OpenHands/`. Port local changes as patches maintained in this repo or upstream PRs. The previous vendored copy remains in `OpenHands_bk/` for reference.
- Import/runtime expectations:
  - Path and module names remain the same (`OpenHands/openhands/...`, `python -m openhands.core.main`).
  - `backend/app/integrations/openhands_runtime.py` will continue inserting `OpenHands/` onto `sys.path` for subprocesses and in-process imports.

Pseudocode sketch (bridge run)

```
cfg.workspace.mkdir(parents=True)
logs = workspace / 'logs'; logs.mkdir(exist_ok=True)
env = os.environ.copy()
env['PYTHONPATH'] = f"{repo_root/'OpenHands'}:{env.get('PYTHONPATH','')}"
env['RUNTIME'] = 'local'
env['WORKSPACE_BASE'] = str(cfg.workspace)
if cfg.disable_browser: env['ENABLE_BROWSER'] = 'false'
# LLM mapping
for (src, dst) in [("LLM_MODEL","LLM_MODEL"),("LITELLM_MODEL","LLM_MODEL"),
                   ("LLM_API_KEY","LLM_API_KEY"),("LITELLM_API_KEY","LLM_API_KEY"),
                   ("LLM_BASE_URL","LLM_BASE_URL"),("LITELLM_API_BASE","LLM_BASE_URL")]:
    if src in os.environ and dst not in env: env[dst] = os.environ[src]

cmd = [python, '-m', 'openhands.core.main',
       '-t', cfg.goal, '-n', session_name, '-i', str(cfg.max_steps), '--no-auto-continue']
proc = subprocess.Popen(cmd, cwd=cfg.workspace, env=env, stdout=stdout_log, stderr=stderr_log)
wait with timeout (cfg.max_minutes*60), else terminate + return timeout summary
on exit: discover experiments/*/results/final.json → success path
```

Risks and mitigations

- LocalRuntime dependency checks: requires `jupyter` and (on Unix) a working `tmux` via `libtmux`. We will:
  - Document the requirement and include `jupyter` in `requirements-openhands.txt` if not present.
  - Reuse existing `libtmux` Python dep already in `backend/requirements.txt`, but a system `tmux` binary may be needed — call out in README for setup.
- LLM config drift between UAgent and OpenHands: the env mapping in the bridge ensures a consistent provider and model.
- Long‑running processes: enforce a hard wall time; return logs on failure; allow engine to retry with adjusted budgets if needed.

What will not change

- No Docker dependency will be introduced.
- No pip install of `openhands-ai`; source usage continues via the upstream checkout at `OpenHands/`.
- Existing `/api/openhands` routes and V1/V2 infra remain intact during migration.

Next steps

1) Implement `backend/app/integrations/openhands_codeact_bridge_v3.py`.
2) Gate ScientificResearchEngine to call the bridge when `UAGENT_OPENHANDS_V3=1`.
3) Add a minimal `requirements-openhands.txt` only if we detect missing host deps (e.g., `jupyter`).
4) Add smoke tests that assert `final.json` creation and proper timeout handling.
5) Flip the default after validation, then retire V1/V2.

Appendix — Quick Commands

- Backup vendored copy and add upstream as submodule:
  - `mv OpenHands OpenHands_bk`
  - `git submodule add -b main https://github.com/All-Hands-AI/OpenHands.git OpenHands`
  - `git submodule update --init --recursive`
- Update to latest upstream (submodule):
  - `git submodule update --remote --merge --recursive`
  - `git add OpenHands && git commit -m "chore: bump OpenHands submodule"`
- Update to latest upstream (plain clone):
  - `cd OpenHands && git fetch origin && git checkout main && git pull --ff-only`
