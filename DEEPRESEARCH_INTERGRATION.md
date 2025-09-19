# Deep Research Integration Plan

## 1. Context and Goals

- **DeepResearch repository** (./DeepResearch) implements a ReAct-style multi-agent loop (`inference/react_agent.py`) that orchestrates tool calls (`tool_search.py`, `tool_visit.py`, `tool_scholar.py`, `tool_python.py`, `tool_file.py`) under the guidance of the `MultiTurnReactAgent` prompt (`inference/prompt.py`). It expects a planning backend (sglang/vLLM server) and produces iterative `<think>`, `<tool_call>`, `<tool_response>`, `<answer>` transcripts.
- **UAgent deep research engine** (`backend/app/core/research_engines/deep_research.py`) currently simulates search results and only performs a single synthesis call. The UI tree relies on `progress_tracker.log_research_progress` with `metadata.parent_id` to render nodes.
- Goal: lift UAgent’s deep research into a true multi-turn tree-search workflow inspired by DeepResearch, while retaining our WebSocket streaming, CLI monitor, and workspace orchestration.

## 2. Integration Objectives

1. **Agentic Loop** – Replace simulated search with an iterative planner → tool executor → observer loop. Each loop iteration produces tree nodes for plan steps, tool calls, observations, and analysis updates.
2. **Tree Synchronisation** – Guarantee every plan branch maps cleanly onto the UI tree (`TREE_NODE_ADDED`/`UPDATED`) and CLI renderer. Nodes must persist per session and be navigable across live progress, tree view, and results.
3. **Evidence Management** – Cache raw tool outputs (search snippets, visited page summaries, code results) to support double-click expansion in the UI and full-report assembly.
4. **Final Report Pipeline** – Build the final markdown report from structured tree data (plan → evidence → synthesis) and stream the result to both UI and CLI monitors.
5. **Configurability** – Allow toggling heavy features (e.g., headless browser search vs. API search, scholar vs. web tools) via the existing engine config block.

## 3. Proposed Architecture Changes

### 3.1 Engine Orchestrator
- Introduce a `ResearchPlanner` inside `deep_research.py` that issues an initial plan using the Qwen/DashScope client (`StreamingLLMClient.generate`).
- Represent the plan as a list of `ResearchStep` dataclasses (step id, goal, parent id, status). Persist them in a session-scoped tree map.
- Create a `ToolExecutor` adapter that wraps existing UAgent search utilities (Playwright + Bing fallback, academic search, etc.) and new DeepResearch tools when available.
- Implement an asynchronous loop:
  1. `PLAN` → add child nodes under the session root for each sub-goal.
  2. `EXECUTE` → for each active node, call the tool executor, log a child node per tool invocation, attach observation payloads.
  3. `REFLECT` → use LLM to summarise findings; either close the branch or schedule follow-up sub-nodes.
  4. Repeat until termination criteria (confidence threshold, MAX_DEPTH, MAX_ITERATIONS) or planner emits `<answer>`.

### 3.2 Tree & Progress Mapping
- Extend `_log_progress` to support explicit `node_id`/`parent_id` from planner steps so siblings preserve order. Node metadata should include `node_type` (`plan`, `tool_call`, `observation`, `analysis`, `summary`).
- Store full LLM/tool payloads in a session-level cache (`self._session_evidence[session_id][node_id]`) so the UI can fetch complete text on demand.
- Broadcast a `TREE_NODE_UPDATED` event when a node transitions from `running` → `completed` or gains new evidence.

### 3.3 Evidence & Report Construction
- As each branch completes, accumulate structured evidence objects (`title`, `source`, `content`, `confidence`).
- Adjust `_synthesize_results` to accept the structured tree and craft a markdown report with sections: Executive Summary, Plan Breakdown, Supporting Evidence (with citations), Recommendations, Next Actions.
- Persist the final report per session and surface via `GET /api/research/sessions/{id}/full` so the frontend “Results” tab and CLI output display the full markdown preview.

### 3.4 Tooling Bridge
- Implement adapter classes in `backend/app/core/research_engines/deep_research_tools.py` (new) that mimic the DeepResearch `BaseTool` interface but call our existing services:
  - `WebSearchTool` → uses Playwright + Bing (already required after Serper removal) and returns concatenated snippets.
  - `ScholarSearchTool` → stub hooking into future academic search provider.
  - `BrowserVisitTool` → reuses our browser worker to fetch and summarise pages; optionally call DeepResearch’s `EXTRACTOR_PROMPT` for better summaries.
- Keep the door open for copying advanced tooling from `DeepResearch/inference/` later by matching signatures (`call(params)` returning raw text).

## 4. Implementation Plan

1. **Refactor Engine Skeleton**
   - Create helper dataclasses (`ResearchStep`, `ToolExecution`, `EvidenceBlock`).
   - Maintain session-scoped dicts for nodes, evidence, and plan status.

2. **Planner Integration**
   - Add `_generate_plan(query)` using `StreamingLLMClient` so plan tokens stream to the UI/CLI.
   - Parse bullet list or JSON into `ResearchStep`s; log tree nodes immediately.

3. **Tool Execution Loop**
   - For each step, run `_execute_step(step)`:
     - Compose queries (LLM assisted) based on step goal.
     - Call tool adapters asynchronously (allow fan-out across sources).
     - Log tool call node (`node_type="tool_call"`) and child observation nodes.

4. **Reflection & Expansion**
   - After observations, call `_reflect_and_update(step, evidence)` to decide whether to:
     - mark step complete,
     - spawn follow-up sub-steps (child nodes), or
     - escalate to synthesis.

5. **Final Synthesis**
   - Pass structured evidence into `_synthesize_results` (rewritten to prefer JSON output but with fallback parser).
   - Broadcast final summary node and call `progress_tracker.log_research_completed`.

6. **API & Storage Adjustments**
   - Update existing session storage (if necessary) to include `deep_research` evidence blocks so `/full` responses include plan, steps, evidence, and final report.

7. **Testing & Validation**
   - Update `test/unit/test_deep_research.py` to cover:
     - Plan generation returns ordered steps and emits tree nodes.
     - Step execution logs tool and observation nodes with parent linkage.
     - Synthesis builds markdown report referencing gathered evidence.
   - Add integration test stub to ensure CLI monitor receives tree updates when running mocked tool executors.

## 5. Risks & Mitigations

- **Tool Latency** – Headless browser searches can be slow. Mitigate via async gather, configurable timeouts, and progressive progress updates so front-end shows “waiting on search”.
- **LLM Token Limits** – Plan/reflect loops may produce long context. Reset per branch and summarise observations before feeding back to the model.
- **Parsing Robustness** – Provide tolerant parsers for plan/evidence JSON blocks to avoid crashes when model returns text; add regex fallbacks similar to existing `_parse_synthesis_response`.
- **UI Consistency** – Ensure node IDs are deterministic (`step-{idx}`, `tool-{step}-{k}`) so reconnects rebuild the same tree structure.

## 6. Deliverables

- Enhanced deep research engine module implementing the iterative tree-search workflow.
- New tooling adapter module (if required) with Playwright-backed search.
- Updated API responses carrying step-by-step evidence and full markdown reports.
- Tests covering planning, execution logging, and synthesis reporting.

