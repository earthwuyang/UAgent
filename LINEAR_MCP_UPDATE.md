# Research Tree Sync Status – October 12, 2025

## Actions Completed
- Restarted backend in tmux session `uagent-backend` via `.venv/bin/activate && ./start_openhands_research.sh`.
- Launched UI at `http://localhost:2999`, opened conversation `db47f586de6041f18056ddac85e7a016`, and kicked off research goal “hybrid OLAP query planners”.
- Exercised UI tabs (especially **Research Tree**) using Playwright MCP automation, including `fetch` requests to research API endpoints for tree/status snapshots.

## Observed Results
- Backend startup succeeded; research extension routes and DB initialization logged. Experiment registration log emitted (`✅ Registered experiment … Active experiments: 1`).
- Missing expected singleton confirmation logs (no “✅ …using global ResearchSessionManager singleton” for API, middleware, or WebSocket paths).
- UI Research Tree panel remains disconnected (`Connection: Disconnected`, `0/0` nodes, empty tree) even while MCP fetch calls execute.
- Direct API probes return empty payloads:
  - `/api/research/experiments/<exp_id>/tree` → `version: 0`, empty node list.
  - `/api/research/experiments/<exp_id>/status` → status `idle`, zeroed stats.
- Browser console repeatedly reports `net::ERR_CONNECTION_RESET` for WebSocket/SSE endpoints on ephemeral localhost ports (e.g., 50487, 57713).

## Current Assessment
- Backend and orchestrator create and register experiments, but the API/UI still operate against a session manager instance with no state (or fail before logging). The absence of singleton logs suggests import/init path divergence for API/WebSocket layers.
- Connection reset errors imply WebSocket endpoints may be failing early, preventing progress events from reaching the UI.

## Recommended Next Steps
1. Add temporary debug logging (stdout/warn level) at the top of `get_session_manager()` in API and WebSocket modules to confirm execution and ensure `get_global_session_manager()` is invoked post-deploy.
2. Trace WebSocket handshake failures (capture server-side stack traces or enable uvicorn access logs) to determine why connections reset immediately.
3. Validate event bus wiring: confirm the singleton instance registered with experiments receives node updates (e.g., log `len(self.experiments)` inside `register()`/`_handle_event`).
4. After instrumentation, rerun the Playwright flow to verify singleton logs appear and that `/status` endpoint reports non-zero progress before retesting UI sync.
