# UAgent + OpenHands: Help Request for GPT‑5‑Pro

This document summarizes the project, the symptoms we’re seeing, what we’ve already changed, and the specific help we’d like from you. The goal is to get UAgent reliably executing real experiments (no simulations), with resilient LLM parsing, good session/workspace hygiene, and a strict-but-usable research tree view.

## Project Summary
- UAgent is a scientific research agent that plans → edits code → runs experiments → gathers evidence. It integrates:
  - OpenHands (local runtime, headless action server) for CodeAct/CodeReact style code execution.
  - LiteLLM for model routing and a unified LLM API (OpenAI, OpenRouter, DashScope/Qwen, Kimi, GLM, etc.).
  - Backend (FastAPI) + React frontend (Vite) with a research tree visualization.
- We aim for typed plans, evidence‑gated results, and reproducibility (artifacts/logs/models), while supporting multiple LLMs.

## Current Architecture (short)
- Backend
  - Scientific Research Engine orchestrates ideas → hypotheses → experiment design → CodeAct execution.
  - OpenHands integration spawns an action execution server and exposes actions (run/read/write/edit/etc.).
  - LiteLLM client centralizes model calls and sanitizes JSON.
- Frontend
  - Research session view + tree visualization + HMR dev server.
- Workspaces
  - Per‑session workspace under `UAGENT_WORKSPACE_DIR` (OpenHands runtime writes here).

## What Works
- Sessions start; OpenHands action server launches; workspaces are created.
- LiteLLM calls multiple providers (Kimi, GLM, OpenRouter) and streams logs.
- JSON sanitation exists (with a sanitizer before safe_json_loads), retry on invalid JSON.
- We log CodeAct steps to `logs/codeact_steps.jsonl` per workspace.

## What’s Broken (key symptoms)
1) CodeAct stagnation: runtime keeps listing/viewing, rarely creates/edits files
- Repeated actions like `find . -type f -name "*.py" | head -20` return no output; model repeats them.
- We saw server rejects non‑existent actions (LLMMalformedActionError: action='list'), and agent falls back to lists/views.
- We need the agent to choose create/edit when directories are empty.

2) LLM JSON parsing brittle
- Kimi/GLM frequently return fenced blocks (```json …```), or include unescaped newlines; sometimes truncated.
- Scientific engine and experiment design steps fail with “Expecting , delimiter” and similar.
- Some providers embed hidden reasoning or return empty responses.

3) Session/workspace UX
- Refreshing frontend sometimes starts over (new workspace/sessions) instead of resuming.
- Too many tree nodes; not a strict tree; duplicate IDs (e.g., `rootId` re‑declared) and noisy subnodes.
- Want parallel experiments, but need strict tree structure and saved success workspaces.

4) Env/config issues
- .env variable expansion questions; missing port defaulted to 80 (we want explicit error).
- Frontend WebSocket tried 127.0.0.1:80; should honor BACKEND_PORT / Vite env.
- Dependency conflicts (anyio between fastapi/httpx/anthropic/openhands-ai) and mismatched pins.

5) OpenHands subtle runtime quirks
- Directory reads sometimes return success with exit_code=-1; we map to success when content exists.
- File path remapping (/workspace → host path) works, but repetitive “view/list” loops persist.
- Occasionally HTTP 500 from action server when LLM asks for unsupported actions (e.g., `list`).

## What We Already Changed (minimal, targeted)
- Relaxed “simulation” guardrails to warnings by default; kept `strict_simulation_guard` switch if needed.
  - backend/app/core/research_engines/scientific_research.py:620, 645
- CodeAct failure diagnostics now summarize the last rounds (instead of generic “flow error”).
  - backend/app/core/research_engines/scientific_research.py:1037–1123
- CodeAct steering improvements (no hard forcing):
  - Added TOOL_SPEC guidance: if dir empty or file missing, prefer `str_replace_editor` with `command=create` under `code/`.
  - Injected a lightweight workspace snapshot into the initial transcript so the LLM sees that `code/` is empty.
  - Repeat‑guard now injects a corrective hint and gives the model another chance; only then aborts.
  - backend/app/services/codeact_runner.py (TOOL_SPEC + run() stagnation guard)
- Softer prompt language across CodeAct/OpenHands client to avoid over‑blocking legitimate content.
  - backend/app/services/codeact_runner.py:350–357
  - backend/app/core/openhands/client.py:240–243

## Representative Logs
- CodeAct loop (listing forever):
  - Steps repeatedly: `find . -type f -name "*.py" | head -20` → `[command produced no observable output]` → repeat_guard.
- LLMMalformedActionError (OpenHands): action ‘list’ is not defined; supported actions exclude ‘list’.
- JSON parse failures: “Expecting ‘,’ delimiter”, “LLM response was empty”, or code fences with invalid strings.

## Questions For GPT‑5‑Pro
1) Tool‑selection prompt design
- Given the workspace snapshot and our TOOL_SPEC guidance, what’s the best minimal prompt/update to get models (Qwen/Kimi/GLM/OpenAI) to prefer `create`/`write` after empty listings without over‑constraining them?
- Would adding a soft “first non‑trivial step must be either create or edit” rubric help, or does that reduce flexibility on repos with pre‑existing code?

2) Robust action schema across models
- Some models propose unsupported actions (`list`). We currently map read‑dir to `run "ls -pa"` and reject `list` with a server 500. Should we pre‑translate common aliases (list → run ls; lsdir → run ls; cat → read) before posting to the action server? Provide a recommended alias map.

3) JSON output hardening
- Propose a universal sanitizer that handles:
  - fenced blocks (```json …```), stray preambles, unescaped newline clusters in strings,
  - trailing commas, missing commas after strings, unterminated objects due to truncation.
- We already “collapse unescaped newline clusters inside quoted strings.” Should we additionally run a JSON5‑like tolerant parser or implement a small repair FSM, and only then hard fail?
- What max_tokens and stop sequences should we use for GLM/Kimi to prevent mid‑object truncation?

4) LiteLLM configuration and routing
- For mixed providers (dashscope Qwen, Kimi, GLM, OpenRouter), what provider‑specific knobs (api_base, top_p, frequency penalties) best reduce malformed JSON while keeping reasoning quality?
- Suggest a liteLLM “classification” model and a “tool‑friendly” model pairing with clear fallbacks (no tool call → regenerate plain text).

5) Session persistence + parallel experiments
- Recommend a simple session model to ensure refresh resumes existing workspace instead of starting over. We have `session_id`, `workspace_id` per run; suggest a locking and resume protocol.
- What’s a clean approach to kick off multiple parallel experiments (N≥3) without the tree becoming noisy, while keeping a strict tree and clear parent→child links?

6) Research tree view cleanup
- We need a strict tree with labelled nodes like `level-<n>-node-<k>` and deduped roots, showing only important nodes.
- Suggest a small algorithm to:
  - infer a single root,
  - attach nodes with missing parents under a consistent fallback root,
  - filter “non‑important” nodes (e.g., intermediate tool chatter),
  - avoid TS/Scope redeclaration issues (we hit “Identifier 'rootId' has already been declared”).

7) Env and frontend integration
- Best practice to reference env vars inside .env (we saw `unbound variable` for `OPENROUTER_BASE_URL`). Should we avoid ${…} indirection within .env and instead resolve at process start?
- Vite dev server: ideal way to set `VITE_BACKEND_URL` and `VITE_DEV_SERVER_PORT`, and ensure HMR connects to the right backend port (avoid 127.0.0.1:80 fallback)?
- If a port env var is missing, should we fail fast at startup vs. default any value?

8) Dependency conflicts
- anyio clashes (fastapi<4.0.0, httpx, anthropic, openhands‑ai pins). Recommend a compatible matrix or strategy (pin Starlette/FastAPI/anyio versions or isolate vendor deps) that avoids fighting pip’s resolver.

9) Timeouts and streaming
- Recommend sane per‑action timeout defaults for OpenHands actions (run/read/edit). We currently use 180s; should we adaptively extend for pip install / git clone and report periodic progress?
- Best practice to stream long pip/git outputs to backend logs without overwhelming the UI.

10) Parallel experiment acceptance gates
- Propose assertions that guarantee “real experiments” (e.g., git_diff_nonempty for code edits, rc==0 for builds, non‑zero artifact sizes) while remaining flexible across varied domains.

## Constraints & Preferences
- Do not hard‑require `duckdb` or any domain‑specific dependency in repository‑wide requirements. Experiments bring their own deps.
- Keep OpenHands running from source in `OpenHands/`; UAgent launches the action server internally (no Docker required by default).
- Preserve ability to run many different LLMs via LiteLLM.

## Acceptance Criteria We’d Like GPT‑5‑Pro To Optimize For
- CodeAct no longer loops on list/view; it reliably chooses create/edit when needed.
- Hypothesis/design JSON consistently parsed (sanitizer/repair + model settings) across GLM/Kimi/Qwen/OpenAI.
- Refresh resumes sessions; successful experiment workspaces are preserved for inspection.
- The research tree is a strict tree, labelled, and shows only important nodes; no duplicate identifier errors.
- Frontend connects to correct backend port; missing ports fail fast.
- A minimal conflict‑free dependency strategy.

## Repo Pointers (for your review)
- CodeAct runner and guidance: backend/app/services/codeact_runner.py
- Scientific engine orchestration (JSON parse, retries, evidence gates): backend/app/core/research_engines/scientific_research.py
- OpenHands action runtime wrapper: backend/app/integrations/openhands_runtime.py
- LiteLLM client integration: backend/app/core/llm_client.py
- Frontend tree viz (duplicate identifier bug): frontend/src/components/research/ResearchTreeVisualization.tsx
- Env example: .env.example; scripts/start_* load env

## What We Want From You
- A staged change plan (small PR‑sized chunks) addressing: prompt/tool‑spec fixes; JSON sanitizer/repair; session/workspace persistence; strict tree labeling/filtering; env/frontend alignment; conflict‑free dep pins; timeouts/progress streaming.
- Provide concrete code snippets or patches for each stage, with clear rollback/recovery if a step degrades behavior.

Thanks! We’ll run your plan immediately and report back with logs/artifacts.

