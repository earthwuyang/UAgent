# Research Tree Implementation - Summary for Review

## 📋 What We're Building

A **research tree visualization** with **parallel search** capabilities that:

1. Shows research progress in real-time (top-right icon)
2. Automatically routes scientific research requests through: **Ideas → Hypotheses → Experiments**
3. Executes multiple branches in parallel (8 concurrent tasks)
4. Integrates with **Tongyi DeepResearch** and **RepoMaster** MCP servers
5. Updates UI live via WebSocket

## 🎯 Key Architecture Decisions

### 1. **TreeSearchOrchestrator (TSO)**
- **What**: Core parallel execution engine
- **How**: Uses `asyncio` with semaphore for concurrency control
- **Algorithm**: Beam search with UCB1 scoring (exploration vs exploitation)
- **Why**: Clean separation from OpenHands core, extensible, scalable

### 2. **ResearchEventBus**
- **What**: In-memory pub/sub for real-time events
- **Features**: Event coalescing (batch updates every 100ms), backpressure handling
- **Why**: Prevents WebSocket flood, smooth UI updates

### 3. **MCP Tool Adapters**
- **What**: Pluggable providers (Tongyi, RepoMaster, LLM fallback)
- **Features**: Rate limiting, circuit breaker, health checks
- **Why**: Graceful degradation, provider independence

### 4. **Intelligent Router**
- **What**: Classifies user requests as research vs regular coding
- **How**: Pattern matching + optional LLM classification
- **Why**: Automatic scientific workflow activation

### 5. **Frontend**
- **Components**:
  - `ResearchTreeToggle`: Icon in top-right navigation
  - `ResearchTreePanel`: Slide-out panel with tree visualization
  - `TreeNode`: Recursive node component with status/score
- **Tech**: React, TanStack Query (caching), WebSocket (real-time)
- **Visualization**: Simple hierarchical list (can upgrade to d3 later)

## 📊 Data Flow Example

```
User: "Compare ML algorithms for time series"
  ↓
IntelligentRouter: Detects "compare" + "algorithms" → scientific_research
  ↓
ResearchService: Creates research_xyz
  ↓
TSO: Spawns parallel tasks:
  ├─ Task1: Tongyi → Idea 1: "Neural architecture search"
  ├─ Task2: LLM → Idea 2: "Ensemble methods"
  ├─ Task3: Tongyi → Idea 3: "Transfer learning"
  └─ Task4: LLM → Idea 4: "Reinforcement learning"
  ↓ (scores with UCB1)
  ├─ Idea 1 (0.85) → Generate 3 hypotheses (parallel)
  └─ Idea 2 (0.72) → Generate 3 hypotheses (parallel)
  ↓ (selects top hypotheses)
  ├─ Hypothesis 1.1 → Run experiment (RepoMaster)
  ├─ Hypothesis 1.2 → Run experiment (OpenHands)
  └─ Hypothesis 2.1 → Run experiment (RepoMaster)
  ↓
Complete! Shows best results in tree
```

Each step emits events → WebSocket → Frontend updates in real-time

## 🔧 Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Backend | FastAPI + SQLAlchemy async | Existing stack, async-native |
| Database | SQLite JSON field | Simple, no schema changes |
| Concurrency | asyncio + Semaphore | Python-native, easy to debug |
| Real-time | WebSocket | Bidirectional, low latency |
| Frontend | React + TypeScript | Existing stack |
| State | TanStack Query | Caching, optimistic updates |
| MCP | Model Context Protocol | Standard, extensible |

## 📁 New Files to Create

```
Backend (Python):
├── extensions/uagent_research/uagent_research/
│   ├── core/
│   │   ├── tree_search_orchestrator.py    ← Main parallel engine
│   │   ├── event_bus.py                    ← Pub/sub system
│   │   ├── intelligent_router.py           ← Request classifier
│   │   └── mcp_adapters/
│   │       ├── tongyi_adapter.py           ← Tongyi MCP client
│   │       └── repomaster_adapter.py       ← RepoMaster MCP client
│   ├── models/
│   │   └── tree_node.py                    ← Node data model
│   └── services/
│       └── research_service.py              ← Enhanced service

Frontend (TypeScript):
├── frontend/src/
│   ├── components/research/
│   │   ├── ResearchTreeToggle.tsx          ← Top-right icon
│   │   ├── ResearchTreePanel.tsx           ← Main panel
│   │   └── TreeNode.tsx                    ← Node component
│   └── hooks/
│       ├── use-research-tree.ts            ← Data fetching
│       └── use-research-websocket.ts       ← WS connection
```

## 🚀 Implementation Phases

### Phase 1: Backend Core (Week 1)
- [x] **Day 1-2**: Models, EventBus
- [x] **Day 3-5**: TreeSearchOrchestrator
- [x] **Day 6-7**: Basic testing

### Phase 2: MCP Integration (Week 2)
- [ ] **Day 1-3**: MCP adapter framework
- [ ] **Day 4-5**: Tongyi + RepoMaster adapters
- [ ] **Day 6-7**: Integration tests

### Phase 3: API Layer (Week 3)
- [ ] **Day 1-3**: ResearchService, WebSocket
- [ ] **Day 4-5**: REST endpoints
- [ ] **Day 6-7**: End-to-end testing

### Phase 4: Frontend (Week 4)
- [ ] **Day 1-2**: Toggle icon, basic panel
- [ ] **Day 3-4**: Tree visualization
- [ ] **Day 5-6**: WebSocket integration
- [ ] **Day 7**: Polish + testing

## ⚠️ Potential Challenges

1. **SQLite JSON Performance**
   - Problem: Frequent writes to large JSON
   - Solution: Batch writes, write-through cache, periodic snapshots

2. **WebSocket Event Flood**
   - Problem: 100s of events/sec could freeze UI
   - Solution: Event coalescing (100ms batches), queue limits, throttling

3. **MCP Provider Failures**
   - Problem: Tongyi/RepoMaster might be down
   - Solution: Circuit breaker, fallback to LLM, retry with backoff

4. **Request Misclassification**
   - Problem: Regular coding task marked as research
   - Solution: Manual override button, improve classifier over time

## 🤔 Questions for You

### 1. Architecture
- ✅ Does the TSO + EventBus + MCP adapter separation make sense?
- ✅ Any concerns about using asyncio for parallelism?
- ❓ Should we support distributed execution (multiple workers)?

### 2. Algorithms
- ✅ Beam search + UCB1 for scoring - good choice?
- ❓ Should we support other algorithms (MCTS, A*)?
- ❓ How should we weight novelty vs feasibility vs impact?

### 3. UI/UX
- ✅ Top-right icon for tree toggle - good placement?
- ❓ Should tree be collapsible/expandable per node?
- ❓ What info should be shown on hover?
- ❓ How to show experiment results (inline vs modal)?

### 4. MCP Integration
- ❓ Do you have Tongyi DeepResearch MCP server running?
- ❓ Do you have RepoMaster MCP server running?
- ❓ What endpoints/methods do they expose?
- ❓ Are there rate limits we should know about?

### 5. Priority
- ❓ Which phase should we tackle first?
- ❓ Can we start with simplified version (no MCP, just LLM)?
- ❓ Should we prototype frontend first to validate UX?

### 6. Configuration
```python
RESEARCH_CONFIG = {
    "beam_width": 4,        # How many top nodes to expand? (2-8)
    "max_parallel": 8,      # Max concurrent tasks? (4-16)
    "max_nodes": 300,       # Budget limit? (100-1000)
    "timeout_s": 600,       # Max duration? (300-1200)
}
```
Are these defaults reasonable?

## 🎬 Next Steps

**Option A: Start with Backend Core**
1. Implement TreeSearchOrchestrator
2. Create EventBus
3. Add mock MCP adapters for testing
4. Test parallel execution

**Option B: Start with Frontend Prototype**
1. Create ResearchTreePanel with static data
2. Validate UX and visualization
3. Add WebSocket subscription (mock backend)
4. Polish UI before building backend

**Option C: Start with MCP Integration**
1. Connect to actual Tongyi/RepoMaster servers
2. Test their capabilities
3. Design adapter interface
4. Build TSO around adapters

**My Recommendation**: Option A (Backend Core) → Option C (MCP) → Option B (Frontend)

This allows us to:
- Validate parallel execution works
- Test real MCP integration early
- Build frontend with real data

---

## 📝 Decision Log

Please review and let me know:

1. **Approve architecture?** (TSO, EventBus, MCP adapters)
2. **Algorithm choice?** (Beam search + UCB1)
3. **MCP server status?** (Are they running? Endpoints?)
4. **Configuration?** (Beam width, parallelism, timeouts)
5. **Starting point?** (Which phase first?)

I can start implementation once you approve the plan!
