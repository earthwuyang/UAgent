# RepoMaster Integration Plan (Detailed)

## 1. Objectives
- Replace the placeholder code-research responses with **real repository discovery, analysis, execution and reporting** driven by the upstream RepoMaster pipeline.
- Run RepoMaster **directly on the host** while reusing UAgent’s DashScope `qwen3-max-preview` credentials so every LLM call is consistent with existing engines.
- Surface RepoMaster’s work inside a UAgent session: progress tree nodes, live LLM logs, cloned-repo artefacts, and the final markdown report must all stream into the Live Progress, Tree View, Conversation, and Results panels without mixing sessions.

## 2. Architecture Recon (what we inspected)
| RepoMaster file | Role (confirmed from source) | Integration hook |
|-----------------|------------------------------|------------------|
| `configs/mode_config.py` | Builds runtime configs (`ModeConfigManager`, `RunConfig` subclasses) and materialises `work_dir` + timeouts. | Use it to create a `repository_agent` config bound to our session workspace so RepoMaster honours our directories/timeouts. |
| `configs/oai_config.py` | Central LLM provider registry (OpenAI/Claude/DeepSeek). Picks provider via `DEFAULT_API_PROVIDER`. | Add a `dashscope` provider and make the bridge set `DEFAULT_API_PROVIDER=dashscope` before constructing agents. |
| `src/utils/utils_config.py` | `AppConfig` stores per-user session state (`work_dir`, message queue). | Call `AppConfig.get_instance().create_session(session_id)` with a UAgent-managed path so downstream helpers (e.g. `TaskManager.get_work_dir`) reuse it instead of generating random folders. |
| `src/core/agent_scheduler.py` | Houses `RepoMasterAgent`. Creates `ExtendedAssistantAgent`/`ExtendedUserProxyAgent`, registers tools, and exposes `solve_task_with_repo` and `run_repository_agent`. | Inject callbacks here to stream milestones + LLM traffic. When `solve_task_with_repo` finishes `_extract_final_answer`, we can gather the summary for our report payload. |
| `src/core/git_task.py` | `TaskManager` + `AgentRunner` orchestrate repo cloning, dataset copy, task prompt assembly, and `CodeExplorer` execution. | Hook progress callbacks during repo setup, dataset linking, execution start/finish, and attach stable identifiers for the research tree. |
| `src/services/agents/deep_search_agent.py` | `AutogenDeepSearchAgent` handles GitHub discovery and browsing via `WebBrowser`. | Surface search hits through callbacks so UAgent can visualise candidate repos. Reuse our DashScope config when instantiating its `ExtendedAssistantAgent`. |
| `src/services/autogen_upgrade/base_agent.py` & `src/services/agents/agent_client.py` | Provide `ExtendedAssistantAgent`, `ExtendedUserProxyAgent`, `Trackable*` classes, message interception, and file-change detection. | Tie our LLM streaming adapter into these trackable agents instead of rewriting Autogen. They already wrap `self._process_received_message`; we extend that wrapper to call UAgent’s `StreamingLLMClient`. |
| `src/core/conversation_manager.py`, `src/core/repo_summary.py`, `src/core/tree_code.py` | Manage persistent conversation summaries, repo analysis and tree structures. | Use these for future enhancements (detailed reporting and tree generation) but no direct changes required for the initial bridge. |

## 3. Integration Blueprint
1. **Vendor layout**: copy RepoMaster into `backend/vendor/repomaster` (plain copy so we can patch files; document upstream commit SHA).
2. **Bridge layer** (`backend/app/integrations/repomaster_bridge.py`): wraps RepoMaster calls, owns callback plumbing, and converts results into UAgent models.
3. **Engine wiring**: update `CodeResearchEngine` so workflow plans that request repository analysis delegate to the bridge. On failure, retain current fallback behaviour.
4. **Streaming**: route progress events to `progress_tracker`, LLM logs to `StreamingLLMClient`, and final markdown to the session store so the frontend tabs render full content.

## 4. Detailed Steps

### 4.1 Vendor RepoMaster + dependencies
- Copy the upstream repo into `backend/vendor/repomaster` (keep git metadata or store SHA in `THIRD_PARTY.md`).
- Extend `backend/pyproject.toml`/`requirements.txt` with the minimal packages the backend actually uses (Autogen, playwright deps, jina-client, etc.). Avoid Streamlit/UI-only deps so the backend stay lightweight.
- Update `backend/.env.example` and configuration docs to request `SERPER_API_KEY`, `JINA_API_KEY`, and reuse existing `DASHSCOPE_API_KEY` plus optional Git creds for cloning.

### 4.2 DashScope alignment
- Modify `backend/vendor/repomaster/configs/oai_config.py` to add:
  ```python
  'dashscope': {
      "config_list": [{
          "model": os.getenv("DASHSCOPE_MODEL", "qwen3-max-preview"),
          "api_key": os.getenv("DASHSCOPE_API_KEY"),
          "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
          "api_type": "dashscope"
      }]
  }
  ```
- Ensure `DEFAULT_PROVIDER_PRIORITY` includes `'dashscope'` ahead of OpenAI so our env variables take over when set.
- In the bridge, set `os.environ["DEFAULT_API_PROVIDER"] = "dashscope"` and feed the resulting `llm_config` into every agent constructor.

### 4.3 Bridge (`backend/app/integrations/repomaster_bridge.py`)
- **Context bootstrap**: accept `session_id` from the router, compute `workspace = session_manager.ensure_workspace(session_id) / "repomaster"`, and register it via `AppConfig.get_instance().create_session(session_id)` before any RepoMaster class is created.
- **Config**: call `ModeConfigManager().create_config(mode="backend", backend_mode="repository_agent", work_dir=str(workspace), timeout=workflow.max_timeout)` and derive `llm_config = manager.get_llm_config(api_type="dashscope")`. Pass `manager.get_execution_config()` through to `RepoMasterAgent`.
- **Callbacks**:
  - `progress_callback(event)`: translate RepoMaster phases (deep search, clone, code_execution, report_compile) into our `ProgressEvent` schema (`node_id`, `parent_id`, `status`, `payload`). Provide helper utilities to flatten structured data (e.g., top repos => tree children).
  - `llm_callback(role, message, metadata)`: push into `StreamingLLMClient` so the frontend can expand truncated entries on demand. Preserve `turn_id` using RepoMaster’s `Trackable*` message counters if available.
  - `artifact_callback(path, description)`: optional for surfacing generated files; map into our workspace event bus if the user wants downloads.
- **Execution**: run RepoMaster synchronously via `asyncio.to_thread(repo_master.solve_task_with_repo, task_prompt)` to keep FastAPI endpoints async-friendly. Accept optional `RepositoryContext` (preselected repo) to call `run_repository_agent` instead.
- **Result shaping**: gather the markdown summary from `_extract_final_answer`, combine with any structured repo list returned by callbacks (`repo_candidates`, `analysis_notes`) and map to our `CodeResearchResult`. Persist the raw markdown for the Results tab.

### 4.4 Required RepoMaster patches
- `src/core/agent_scheduler.py`:
  - Extend `RepoMasterAgent.__init__` to accept `progress_callback`, `llm_callback`, `artifact_callback` (defaulting to no-ops).
  - Invoke `progress_callback` when entering/leaving major phases (`solve_task_with_repo` start/end, tool invocation, `_extract_final_answer`). Include metadata such as selected repo, mode switches, and Autogen decision making.
  - Use the existing `ExtendedAssistantAgent` / `ExtendedUserProxyAgent` wrappers to tap into `_process_received_message`. Add hooks that call `llm_callback` **before** the Streamlit-only logic runs (guarded so running without Streamlit works headlessly).
- `src/services/agents/agent_client.py`:
  - The `Trackable*` classes already intercept messages; add lightweight observers (e.g., `self.external_llm_callback`) invoked inside `_process_received_message` and tool-response handlers so we emit every token/turn to UAgent.
- `src/core/git_task.py`:
  - After `DataProcessor.setup_task_environment` starts copying/cloning, call `progress_callback({"event": "repo_clone", ...})` with deterministic node ids (`clone::<repo_name>`). Similar events for dataset linking, `CodeExplorer` start, completion, and failure.
  - Return structured data (paths, repo info) from `AgentRunner.run_agent` so the bridge can embed it in the final report.
- `src/services/agents/deep_search_agent.py`:
  - Emit callbacks with the JSON repo list returned by `github_repo_search` and each browsing pass. Capture enough metadata (`rank`, `readme_summary`) so the frontend tree can display candidates.
- Document all patches in `backend/vendor/repomaster/PATCHES.md` for traceability.

### 4.5 UAgent backend changes
- `backend/app/core/research_engines/code_research.py`: add a `RepoMasterExecutor` branch that instantiates the bridge with the current session id, forwards workflow directives (`max_repos`, `timeout`, `repo_hint`), and consumes the structured response.
- `backend/app/routers/smart_router.py`: set `workflow_plan.use_repo_master=True` when classification tags a request as repository-level. After execution, stash the returned markdown, repo list, and artefacts into the session store.
- `backend/app/core/session_manager.py`: extend the stored session schema to keep `repo_master` specific payloads (`report_markdown`, `repositories`, `artifacts`) so the Results tab can render a full markdown preview and download links.
- `backend/app/core/streaming_llm_client.py`: no code change, but add a helper to register external bridges so we can push long-form messages and expose an API that the frontend can query for "expand on double-click".

### 4.6 Frontend alignment
- `frontend/src/components/research/ResearchTreeVisualization.tsx`: add mapping for RepoMaster phases (`repo_search`, `tool_invocation`, `code_execution`, `report_compose`) so progress, completion, and child nodes render as a true tree (ROMA-style). Use event metadata to keep nodes scoped to the active session.
- `frontend/src/components/research/LLMConversationLogs.tsx`: ensure logs reference `messageId` supplied by the bridge so double-click fetches the full text from the backend.
- `frontend/src/components/ResearchResult.tsx`: render the new markdown (`report_markdown`) plus structured repo list (name, url, summary, match score). Support "Copy" and "Export" buttons using the existing UI affordances.

## 5. Runtime Flow After Integration
1. Smart router marks a request as `code_research` → workflow sets `use_repo_master`.
2. Code research engine invokes `RepoMasterBridge.run_task(session_id, prompt, options)`.
3. Bridge initialises RepoMaster with DashScope + workspace context, registers callbacks, and runs the agent in a worker thread.
4. RepoMaster operations emit callbacks → bridge translates them into UAgent `progress_tracker`/`websocket_manager` events, keeping tree + live progress in sync.
5. When `_extract_final_answer` returns, bridge packages markdown + structured insights → session store → frontend Results tab renders the complete report while the conversation log retains expandable LLM messages.

## 6. Validation Plan
- **Unit tests**: mock RepoMaster callbacks to verify bridge → session manager plumbing (progress + llm logs) works and that markdown persists.
- **Integration tests**: execute a real RepoMaster run against a small public repo (e.g., `tiangolo/full-stack-fastapi-postgresql`) inside a temporary workspace; assert a markdown report, repo list, and completed tree nodes exist.
- **E2E smoke**: drive the frontend via Playwright (existing setup) to ensure Live Progress, Tree View, LLM Conversation, and Results stay consistent while switching tabs or reloading.

## 7. Deployment Notes
- Running RepoMaster requires git, playwright browsers, and outbound network access—document prerequisites in `USAGE.md` and provide helper scripts to install Playwright + headless browsers.
- Long-running repo analyses can be heavy; expose workflow knobs (`max_turns`, `clone_timeout`, `max_repo_size_mb`) through router overrides so operators can tune resource usage.
- Maintain a `PATCHES.md` detailing every upstream modification to ease future syncing with RepoMaster releases.

✅ Once this plan is approved we can start copying the vendor code, add the DashScope provider, implement the bridge, patch callbacks, and wire the backend/frontend pieces as outlined.
