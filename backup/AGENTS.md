# Repository Guidelines

These guidelines help contributors work effectively on uagent — a ROMA‑style full‑stack agent augmented with RepoMaster‑grade search and AI‑Scientist/AgentLaboratory parallel experimentation.

## Project Structure & Module Organization
- `backend/` FastAPI service and core logic
  - `app/main.py` app entry, routes wiring
  - `app/routers/` `search.py`, `experiments.py` API endpoints
  - `app/core/` experiment orchestration, aggregation utilities
  - `app/utils/` search engines (SearxNG), helpers
- `frontend/` Vite + React + TS UI (ROMA‑style panels + RepoMaster‑style repo view)
- `tests/` Backend unit tests (pytest)

## Build, Test, and Development Commands
- Backend
  - `cd backend && uvicorn app.main:app --reload` run API at `localhost:8000`
  - `cd backend && pytest -q` run tests
- Frontend
  - `cd frontend && npm install && npm run dev` run UI at `localhost:5173`
- Example API calls
  - Search GitHub (Playwright): `GET /api/search?q=site:github.com+vector+database`
  - GitHub-only search: `GET /api/github/search?q=agent+framework`
  - Start AI-Scientist: `POST /api/experiments/ai-scientist/start` → `{ job_id }`; poll `GET /api/jobs/{job_id}`
  - Start AgentLab: `POST /api/experiments/agent-lab/start` → `{ job_id }`; poll `GET /api/jobs/{job_id}`
  - Submit experiment tree: `POST /api/experiments/run` (JSON plan)

## Coding Style & Naming Conventions
- Python: 4‑space indent, type hints, snake_case for modules/functions, PascalCase for classes. Format with `black` and `ruff` (lint/fix before PR).
- TypeScript/React: Prettier + ESLint defaults; components in `PascalCase.tsx`; hooks in `use*.ts`.
- Paths/names: keep feature‑oriented folders under `app/core/*` and `frontend/src/*`.

## Testing Guidelines
- Backend: `pytest` with `tests/test_*.py`; prefer fast, isolated unit tests; mock network.
- Coverage: target ≥80% for core orchestration (`app/core/*`).
- Frontend: add component tests with Vitest when UI stabilizes.

## Commit & Pull Request Guidelines
- Commits: concise, imperative; prefer Conventional Commits (`feat:`, `fix:`, `docs:`).
- PRs: include purpose, scope, screenshots (UI), sample API payloads, and “How to test”. Link issues and note breaking changes.

## Agent‑Specific Notes
- Search: Playwright‑based “human‑like” search (Bing → DuckDuckGo → Google) with optional Xvfb when no `DISPLAY`. Install browsers first: `cd backend && python -m playwright install chromium`. Use `site:github.com` in queries for repo‑focused results.
- Experiments: tree‑of‑experiments with parallel branches; failed nodes auto‑rollback; results aggregated upward (ROMA‑style summaries).
- Config: use env vars in `backend/.env` (tokens, endpoints). Keep secrets out of commits.
