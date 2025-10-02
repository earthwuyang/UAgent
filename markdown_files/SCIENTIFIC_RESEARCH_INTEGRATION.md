# Scientific Research Engine Integration Plan

## 1. Context Recap

- **Current UAgent engine** (`backend/app/core/research_engines/scientific_research.py`) already orchestrates literature review, hypothesis generation, experiment execution, and synthesis. Execution is mostly sequential and stops after meeting a confidence threshold.
- **Agent Laboratory** (`backup/backend/agent_laboratory_local/ai_lab_repo.py`, `agents.py`): emphasizes stage-based workflows with specialized roles (PhD/Postdoc/Professor/Engineers), iterative plan refinement, and reviewer-style scoring via `get_score`. Supports notes, cost supervision, and optional parallelization flags.
- **AI Scientist v2** (`backup/backend/ai_scientist_local`): introduces idea ideation with semantic-scholar tooling (`perform_ideation_temp_free.py`) and best-first/parallel tree search (`bfts_config.yaml`) to explore multiple research branches concurrently, scoring each branch before selecting the best.

## 2. Integration Goals

1. **Idea Front-End** – Generate multiple candidate research ideas/branches before committing to experimentation (borrowing AI Scientist’s ideation loop with tool usage and reflection).
2. **Information Fusion** – For each idea, automatically trigger deep research and code research to gather background context and implementation leads (Agent Laboratory’s literature+plan phases).
3. **Parallel Experiment Trees** – Execute hypotheses/experiments for different ideas concurrently, tracking them as independent nodes in the research tree (AI Scientist parallel workers + Agent Lab stages).
4. **Automated Evaluation & Selection** – Score each idea’s outcome (Agent Lab reviewer scoring + AI Scientist best-first selection), pick the highest-performing idea, and surface justification.
5. **Tree & Journal Alignment** – Every phase (idea generation, branch exploration, scoring, selection) must broadcast structured nodes/events so the UI tree, log journal, CLI monitor, and REST `/full` session output stay synchronized.

## 3. High-Level Architecture Changes

### 3.1 New Data Structures & Bridges
- `ResearchIdea`: captures idea id, title, summary, initial prompt, associated literature/code pointers, linked hypotheses/experiments, and evaluation score.
- `IdeaEvaluation`: stores reviewer-style metrics (overall score, strengths/weaknesses, decision) akin to Agent Laboratory’s `get_score` output.
- Extend `ScientificResearchResult` with `ideas` (list), `selected_idea_id`, and `idea_evaluations` for final reporting.
- `OpenHandsGoalPlanBridge`: wraps OpenHands `CodeAct`/`CodeReact` functionality to emit executable goal plans and run the generated code through `OpenHandsClient` for each experimental branch. Exposes structured `GoalPlan`/`GoalPlanStep` objects so tree nodes can mirror CodeReact’s planning/execution loop.

### 3.2 Pipeline Phases
1. **Ideation Phase**
   - Adapt AI Scientist’s prompt structure: iterative reflections with tool calls (Semantic Scholar-style search stubbed via DeepResearch `search_specific_source`).
   - Generate N ideas (configurable, default 3). Each idea becomes a `ResearchIdea` node with its own tree branch (`node_type="idea"`).
2. **Context Gathering Phase (per idea)**
   - Trigger DeepResearch (`DeepResearchEngine.research`) and CodeResearch for each idea concurrently (`asyncio.gather`), storing summarized key findings/integration guides back into the idea.
   - Emit tree nodes under the idea for literature/code results.
3. **Experiment Planning & Execution**
   - For every idea, run hypothesis generation + experiment design/execution using existing mechanisms, but isolate per idea. Launch in parallel limited by `max_parallel_ideas` (config; default matches AI Scientist `num_workers`).
   - Feed each experiment design into `OpenHandsGoalPlanBridge` to obtain a CodeReact-style goal plan. Execute every plan step through `OpenHandsClient.generate_and_execute_code`, logging progress/results back to the tree.
   - Track plan steps, code execution, and experiment outcomes per idea branch (progress events include idea id in metadata).
4. **Evaluation & Selection**
   - Aggregate experiment outputs (including OpenHands plan execution metadata) and feed into a reviewer prompt modeled after Agent Laboratory’s `get_score` (overall, originality, significance, etc.).
   - Record `IdeaEvaluation`; update tree with reviewer result node.
   - Select idea with best overall score (tie-break with confidence and novelty). Mark winning idea branch as `node_type="best_choice"`.
5. **Synthesis & Report**
   - Run synthesis on winning idea but include summary table of other ideas and their scores, clearly noting why the winner was chosen.

### 3.3 Configuration Additions
- `scientific_research` config keys:
  - `max_ideas` (default 3)
  - `max_parallel_ideas` (default 2)
  - `idea_reflection_rounds` (default 2-3)
  - `idea_score_model_temperature`, `idea_score_weights` for reviewer scoring adjustments
  - `enable_agent_notes` to inject user-specified notes similar to Agent Lab `task-notes`

### 3.4 Tree/Progress Logging Enhancements
- `ScientificResearchEngine._log_progress` should accept optional `idea_id` and include it in metadata so frontend can filter per idea.
- New helper: `_log_idea_node(session_id, idea)` to standardize creation of idea branches.
- Broadcast idea evaluation and final selection via `progress_tracker.log_research_progress` and final `log_research_completed` metadata (include winning idea title and score breakdown).

## 4. Implementation Plan

1. **Foundational Refactor**
   - Add new dataclasses (`ResearchIdea`, `IdeaEvaluation`) and update `ScientificResearchResult`.
   - Extend session state dictionaries to track idea nodes similar to existing `_session_phase_nodes`.

2. **Ideation Module**
   - Implement `_generate_research_ideas` inspired by `perform_ideation_temp_free.py`:
     - Build prompts using config-defined reflections.
     - Reuse DeepResearch `search_specific_source` for literature queries (fallback to standard search if tool missing).
     - Ensure outputs are JSON parsed with safe fallbacks (mirroring `_safe_parse_json`).

3. **Parallel Branch Execution**
   - Introduce `_execute_idea_pipeline(idea, session_id)` which:
     1. Calls deep/code research concurrently.
     2. Generates hypotheses and experiments restricted to that idea.
     3. Runs experiments (existing methods) but names nodes with `idea.id` prefix.
   - Use `asyncio.Semaphore(max_parallel_ideas)` to throttle concurrency and `asyncio.gather` to execute idea pipelines.

4. **Evaluation & Selection**
   - Create `_evaluate_idea(idea, experiments)` leveraging Agent Lab `get_score` style prompt; store metrics.
   - Add `_select_best_idea(ideas)` returning winning idea id + rationale (score matrix).
   - Update final synthesis to highlight winning idea plus comparison table for others.

5. **Reporting & APIs**
   - Ensure `/api/research/sessions/{id}/full` serialization includes ideas, evaluations, and winning idea summary.
   - Update CLI renderer (if applicable) to display idea branches and highlight winner.

6. **Testing Strategy**
   - Unit: mock ideation to return deterministic ideas; assert selection logic picks highest score.
   - Integration: run scientific research with `max_ideas=2`, verifying concurrent calls to deep/code research (mocked to speed) and final metadata includes `selected_idea_id`.
   - Consider smoke test for tree: ensure `TREE_NODE_ADDED` events include idea nodes.

## 5. Risks & Mitigations
- **LLM JSON drift** – Use regex fallback (as AI Scientist and DeepResearch do) and guard rails on missing fields.
- **Concurrency races** – Protect shared structures (e.g., idea tracking dicts) and ensure logging remains order-consistent via idea-specific keys.
- **Long runtimes** – Provide config to limit `max_ideas` or disable parallelization; log progress increments per major step so UI doesn’t appear stalled.
- **Cost tracking** – Optionally future work: integrate Agent Lab supervisor concept by logging token usage per idea.

## 6. Deliverables
- Updated scientific research engine with idea-based parallel exploration, deep/code integration per idea, reviewer scoring, and best-idea selection.
- New configuration knobs and documentation sections in `USAGE.md`/`PLAN.md`/`CLAUDE.md` as needed.
- Tests or compile checks verifying new code paths.
