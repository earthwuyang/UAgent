# Integration Summary - DeepResearch + RepoMaster into OpenHands

## 📊 What I've Done

### 1. Source Code Exploration ✅
- **DeepResearch** (`/home/wuy/AI/UAgent/DeepResearch/`)
  - Multi-agent web research system
  - Components: WebWalker (planning), WebResummer (aggregation), WebSailor (browsing)
  - Tools: Serper API (search), Jina Reader (web reading), Python execution
  - Framework: ReAct loops with LLM

- **RepoMaster** (`/home/wuy/AI/UAgent/RepoMaster/`)
  - Multi-agent code repository exploration
  - Components: agent_scheduler (orchestration), deep_search_agent, code_explorer
  - Tools: GitHub search, local repo analysis, code execution
  - Framework: Autogen group chat

### 2. Codex Consultation ✅
Conducted comprehensive architecture review with Codex AI, receiving detailed recommendations on:
- Extension structure (single `uagent_research` vs separate)
- Adapter pattern for unifying different agent frameworks
- Tool abstraction and sharing
- Dependency management (optional extras)
- Database schema (no changes needed)
- Frontend integration (SSE/WebSocket)

### 3. Integration Plan Created ✅
Created comprehensive plan document: **`DEEPRESEARCH_REPOMASTER_INTEGRATION_PLAN.md`**

Key architectural decisions:
- **Single Extension**: Both frameworks integrated into `extensions/uagent_research`
- **Adapter Pattern**: Unified interface for DeepResearch, RepoMaster, and CodeActAgent
- **Shared Tools**: Common tool registry (Serper, Jina, GitHub) accessible by all
- **Unified Tree**: Single research tree showing all branches and results
- **Parallel Execution**: AsyncIO TaskGroup with concurrency limits
- **No Core Changes**: Extension-only integration preserving OpenHands core

---

## 🏗️ Proposed Architecture

### High-Level Flow

```
User Request → SkillRouter → Agent Adapter → Tools → Results → Research Tree
                    ↓
            ┌───────┴────────┬──────────────┐
            ↓                ↓              ↓
    DeepResearchAdapter  RepoMasterAdapter  CodeActAdapter
    (Web Research)       (GitHub/Code)      (Execution)
            ↓                ↓              ↓
         Serper           GitHub         OpenHands
         Jina             Analysis       Sandbox
```

### Key Components

**1. Agent Adapters** (3 types)
- `DeepResearchAdapter`: Wraps WebWalker/WebResummer for web research
- `RepoMasterAdapter`: Wraps autogen scheduler for repo tasks
- `CodeActAgentAdapter`: Wraps OpenHands CodeActAgent for experiments

**2. Shared Tools** (standardized interface)
- `SerperSearchTool`: Web search ($0.001/query)
- `JinaReadTool`: Web page reading ($0.0002/page)
- `GitHubSearchTool`: Repository discovery
- `RepoAnalysisTool`: Code structure analysis
- `PythonExecutionTool`: Sandbox execution

**3. TreeSearchOrchestrator**
- Manages parallel branches
- Routes tasks to adapters
- Enforces budgets and concurrency
- Streams events to frontend

**4. SkillRouter**
- Intelligent task routing
- Heuristic-based (upgradeable to learned)
- Routes: web → DeepResearch, code → RepoMaster, execution → CodeAct

---

## 📁 File Structure (New Files)

```
extensions/uagent_research/
├── adapters/
│   ├── base.py                    # AgentAdapter interface
│   ├── deepresearch/adapter.py    # DeepResearch wrapper
│   ├── repomaster/adapter.py      # RepoMaster wrapper
│   └── codeact/adapter.py         # CodeAct wrapper
│
├── tools/
│   ├── search/serper_tool.py      # Web search
│   ├── browse/jina_tool.py        # Web reading
│   └── code/github_search_tool.py # GitHub search
│
├── orchestrator/
│   ├── tree_orchestrator.py       # Main orchestrator
│   └── event_bus.py               # Event streaming
│
├── router/
│   └── skill_router.py            # Intelligent routing
│
├── models/
│   ├── events.py                  # Event schema
│   └── research_tree.py           # Tree models
│
├── vendor/
│   ├── deepresearch/              # Copied DeepResearch source
│   └── repomaster/                # Copied RepoMaster source
│
└── api/
    └── routes.py                  # FastAPI endpoints

frontend/src/extensions/uagent_research/
├── components/
│   ├── ResearchTreeView.tsx      # Tree visualization
│   └── NodeInspector.tsx         # Node details
└── hooks/
    └── useResearchTree.ts         # SSE/WebSocket
```

---

## 🔄 Scientific Research Workflow Example

**User**: "Compare ML algorithms for time series forecasting"

**Flow**:
1. **Idea Generation** (TreeOrchestrator)
   - Generates 3 ideas: NAS, Ensemble, Transfer Learning

2. **Parallel Research Branches**:
   - **Branch A (DeepResearch)**:
     - Search web for "neural architecture search time series"
     - Read top 5 papers from arXiv
     - Summarize: "NAS shows 15% improvement over baselines"

   - **Branch B (RepoMaster)**:
     - Search GitHub for "time series forecasting pytorch"
     - Find `pytorch/forecasting` repo
     - Analyze code structure
     - Find example: `examples/autoformer.py`

   - **Branch C (CodeActAgent)**:
     - Clone repo
     - Run example: `python examples/autoformer.py`
     - Collect results: "Accuracy: 89.2%"

3. **Hypothesis Formation**:
   - Combine results: "NAS+Autoformer should achieve >90% accuracy"

4. **Experiment Execution** (CodeActAgent):
   - Implement NAS search for Autoformer
   - Run experiments
   - Final result: "Accuracy: 91.5% ✓"

5. **Research Tree**:
```
Root: "Compare ML algorithms"
├─ Idea 1: "Neural Architecture Search" (DeepResearch)
│  ├─ WebSearch: "NAS time series" (5 papers)
│  └─ Summary: "NAS shows promise..."
├─ Idea 2: "Find implementations" (RepoMaster)
│  ├─ GitHubSearch: "time series pytorch"
│  ├─ RepoAnalysis: "pytorch/forecasting"
│  └─ CodeRun: "Autoformer 89.2%"
└─ Hypothesis: "NAS+Autoformer >90%" (CodeActAgent)
   └─ Experiment: "Result: 91.5% ✓"
```

---

## 🔧 Implementation Phases (6 Weeks)

### Week 1: Foundation
- Extension structure
- AgentAdapter interface
- Event models
- SkillRouter (basic)
- EventBus

### Week 2: Tools
- SerperSearchTool
- JinaReadTool
- GitHubSearchTool
- Caching + rate limiting

### Week 3: Adapters
- Copy DeepResearch source
- DeepResearchAdapter
- Copy RepoMaster source
- RepoMasterAdapter
- CodeActAgentAdapter

### Week 4: Orchestration
- TreeSearchOrchestrator
- Branch executor
- PUCT scoring
- Budget tracking

### Week 5: Frontend
- ResearchTreeView
- NodeInspector
- SSE streaming
- Controls

### Week 6: Testing
- Integration tests
- E2E tests
- Performance tuning
- Documentation

---

## 💡 Key Design Decisions (Based on Codex)

### ✅ Unified Extension
**Decision**: Single `uagent_research` extension
**Rationale**: Shared tree, consistent storage, unified API, no core changes

### ✅ Adapter Pattern
**Decision**: Wrap each framework with AgentAdapter interface
**Rationale**: Clean abstraction, parallel evolution, easy testing

### ✅ Optional Dependencies
**Decision**: `pip install openhands[research_all]`
**Rationale**: Keeps core lightweight, optional features

### ✅ No DB Schema Changes
**Decision**: Use existing `research_tree` JSON field
**Rationale**: Non-breaking, flexible, easy rollout

### ✅ Skill Router
**Decision**: RepoMaster's scheduler for code tasks, top-level router delegates
**Rationale**: Preserves RepoMaster's strength, clean separation

### ✅ Event Streaming
**Decision**: SSE for server→client, events: Plan/Step/ToolCall/Observation/Summary/Complete
**Rationale**: Real-time UI updates, simple protocol

---

## 📋 Configuration

**Environment Variables**:
```bash
SERPER_API_KEY=xxx        # Web search
JINA_API_KEY=xxx          # Web reading (optional with public proxy)
GITHUB_TOKEN=xxx          # Higher rate limits
RESEARCH_MAX_CONCURRENCY=8
RESEARCH_MAX_COST=10.0
RESEARCH_TIMEOUT=600
```

**Optional Extras**:
```bash
pip install openhands[deepresearch]    # DeepResearch only
pip install openhands[repomaster]      # RepoMaster only
pip install openhands[research_all]    # Both frameworks
```

---

## 🎯 Next Steps

### For You to Decide:

1. **Approve Architecture?**
   - Single extension vs separate extensions?
   - Adapter pattern acceptable?
   - File structure makes sense?

2. **Priority?**
   - Start with Phase 1 (Foundation)?
   - Or prototype frontend first to validate UX?
   - Or integrate tools first to test APIs?

3. **Dependencies?**
   - Copy entire DeepResearch/RepoMaster source?
   - Or cherry-pick only needed modules?
   - Vendor vs pip install from local paths?

4. **Frontend UX?**
   - Tree visualization library preference? (d3, react-flow, custom)
   - Node details: modal vs side panel?
   - Real-time updates: SSE vs WebSocket?

### Ready to Implement:

Once you approve the architecture, I can:
1. **Create extension structure** (Phase 1)
2. **Implement AgentAdapter interface**
3. **Add first tool** (SerperSearchTool)
4. **Prototype DeepResearchAdapter**
5. **Create basic frontend tree view**

---

## 📚 Documents Created

1. **`RESEARCH_TREE_IMPLEMENTATION_PLAN.md`** - Original tree visualization plan
2. **`RESEARCH_TREE_ARCHITECTURE.md`** - Detailed architecture diagrams
3. **`UPDATED_IMPLEMENTATION_PLAN.md`** - v2 plan with Codex improvements
4. **`DEEPRESEARCH_REPOMASTER_INTEGRATION_PLAN.md`** - Complete integration plan ⭐
5. **`INTEGRATION_SUMMARY_FOR_REVIEW.md`** - This document

---

## ❓ Questions for You

1. Should I proceed with copying DeepResearch and RepoMaster source code into OpenHands?
2. Do you want me to start with Phase 1 (foundation) implementation?
3. Any changes to the proposed architecture?
4. Do you have API keys for Serper and Jina to test tools?

**Please review the architecture and let me know how to proceed!**
