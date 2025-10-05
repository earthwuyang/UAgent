# Research Tree Architecture - Detailed Design

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Top Navigation Bar                                 [🔬 Research]    │  │
│  │                                                          ↓ (click)    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Main Content              │  Research Tree Panel (Slide-out)        │  │
│  │  ┌──────────────────┐      │  ┌────────────────────────────────────┐│  │
│  │  │  Chat Interface  │      │  │  Root: "Compare ML algorithms"     ││  │
│  │  │                  │      │  │  ├─ Idea 1 (score: 0.85) ✓        ││  │
│  │  │  User: Compare   │      │  │  │  ├─ Hypothesis 1.1 (0.92) ⚡   ││  │
│  │  │  ML algorithms   │      │  │  │  │  └─ Experiment ✓           ││  │
│  │  │                  │      │  │  │  └─ Hypothesis 1.2 (0.78)      ││  │
│  │  │  Agent: Starting │      │  │  ├─ Idea 2 (score: 0.72) ⚡        ││  │
│  │  │  research...     │      │  │  │  └─ Hypothesis 2.1 (0.88)      ││  │
│  │  │                  │      │  │  └─ Idea 3 (score: 0.65)          ││  │
│  │  └──────────────────┘      │  │                                     ││  │
│  │                             │  │  Legend: ✓=done ⚡=running         ││  │
│  │                             │  └────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↕ WebSocket + REST
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BACKEND SERVICES                                  │
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ User Request     │                                                        │
│  │ "Compare ML algos"│                                                       │
│  └────────┬─────────┘                                                        │
│           ↓                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │  IntelligentRouter (Request Classifier)                          │      │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐│      │
│  │  │ Pattern Match    │  │ LLM Classification│  │ Keyword Detection││      │
│  │  │ - "compare"      │  │ - Research intent?│  │ - "experiment"   ││      │
│  │  │ - "experiment"   │  │ - Code task?      │  │ - "hypothesis"   ││      │
│  │  └──────────────────┘  └──────────────────┘  └─────────────────┘│      │
│  │                                                                   │      │
│  │  Result: intent = "scientific_research"                          │      │
│  └────────────────────────────┬─────────────────────────────────────┘      │
│                                ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ResearchService (FastAPI)                                          │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ POST /v1/research/start                                        ││   │
│  │  │   → Create research_id                                         ││   │
│  │  │   → Initialize ResearchTree in DB                             ││   │
│  │  │   → Launch TreeSearchOrchestrator(research_id)                ││   │
│  │  │   → Return research_id to client                              ││   │
│  │  │                                                                 ││   │
│  │  │ GET /v1/research/{id}/tree                                     ││   │
│  │  │   → Fetch current tree state from DB                          ││   │
│  │  │                                                                 ││   │
│  │  │ WS /ws/research/{id}                                           ││   │
│  │  │   → Subscribe to EventBus for research_id                     ││   │
│  │  │   → Stream events: node_added, node_updated, complete         ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  └────────────────────────────────┬────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TreeSearchOrchestrator (TSO) - Parallel Execution Engine          │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ async run(prompt):                                             ││   │
│  │  │   1. Create ROOT node                                          ││   │
│  │  │   2. Generate IDEAS (parallel, beam_width=4)                   ││   │
│  │  │      └─> spawn 4 asyncio tasks concurrently                    ││   │
│  │  │   3. Score IDEAS using UCB1                                    ││   │
│  │  │   4. Generate HYPOTHESES for top ideas (parallel)              ││   │
│  │  │      └─> spawn N tasks with semaphore(max_parallel=8)          ││   │
│  │  │   5. Score HYPOTHESES                                          ││   │
│  │  │   6. Run EXPERIMENTS for top hypotheses (parallel)             ││   │
│  │  │   7. Collect results, emit complete event                      ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  │                                                                      │   │
│  │  Parallel Execution Model:                                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐ │   │
│  │  │ asyncio.Semaphore(8) ─── Controls max concurrent tasks       │ │   │
│  │  │                                                                │ │   │
│  │  │ Task Pool:                                                     │ │   │
│  │  │   [Task1: Generate Idea 1] ─> MCP Tongyi                      │ │   │
│  │  │   [Task2: Generate Idea 2] ─> LLM                             │ │   │
│  │  │   [Task3: Generate Idea 3] ─> MCP Tongyi                      │ │   │
│  │  │   [Task4: Generate Idea 4] ─> LLM                             │ │   │
│  │  │   ... (up to 8 concurrent)                                    │ │   │
│  │  │                                                                │ │   │
│  │  │ Per-Provider Rate Limiting:                                   │ │   │
│  │  │   Tongyi: TokenBucket(rate=10/s, burst=20)                    │ │   │
│  │  │   RepoMaster: TokenBucket(rate=5/s, burst=10)                 │ │   │
│  │  └──────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    ↓ For each node expansion                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  MCP Tool Adapters (Pluggable Providers)                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │   │
│  │  │ Tongyi DeepRes   │  │ RepoMaster Deep  │  │ Standard LLM    │  │   │
│  │  │ - Idea gen       │  │ - Code analysis  │  │ - Fallback      │  │   │
│  │  │ - Hypothesis gen │  │ - Experiment run │  │ - General tasks │  │   │
│  │  │ - Rate limited   │  │ - Rate limited   │  │                 │  │   │
│  │  └──────────────────┘  └──────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ResearchEventBus (In-Memory Pub/Sub)                              │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ Subscribers: Map<research_id, Set<Queue>>                      ││   │
│  │  │                                                                 ││   │
│  │  │ Event Coalescing:                                              ││   │
│  │  │   - Batch node_updated events (flush every 100ms)             ││   │
│  │  │   - Immediate publish for node_added, complete, error         ││   │
│  │  │                                                                 ││   │
│  │  │ Backpressure Handling:                                         ││   │
│  │  │   - Queue size limit (100)                                     ││   │
│  │  │   - Drop events if queue full                                  ││   │
│  │  │   - Remove dead subscribers                                    ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WebSocketGateway                                                   │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ On connect:                                                     ││   │
│  │  │   - Authenticate user                                           ││   │
│  │  │   - Verify access to research_id                               ││   │
│  │  │   - Subscribe to EventBus                                       ││   │
│  │  │   - Start streaming events                                      ││   │
│  │  │                                                                 ││   │
│  │  │ Event Types:                                                    ││   │
│  │  │   {type: "node_added", node: {...}}                            ││   │
│  │  │   {type: "node_updated", node: {id, status, score, version}}  ││   │
│  │  │   {type: "progress", progress: {expanded: 15, total: 50}}     ││   │
│  │  │   {type: "complete", summary: {best_nodes: [...], duration}}  ││   │
│  │  │   {type: "error", error: {message: "..."}}                    ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATABASE (SQLite + JSON)                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  research_sessions table                                            │   │
│  │  ┌────────────────────────────────────────────────────────────────┐│   │
│  │  │ id: "research_123"                                             ││   │
│  │  │ user_id: "user_456"                                            ││   │
│  │  │ research_tree: {  ← JSON field                                ││   │
│  │  │   nodes: {                                                     ││   │
│  │  │     "root_0": {                                                ││   │
│  │  │       id: "root_0",                                            ││   │
│  │  │       type: "root",                                            ││   │
│  │  │       summary: "Compare ML algorithms",                        ││   │
│  │  │       status: "done",                                          ││   │
│  │  │       version: 1                                               ││   │
│  │  │     },                                                          ││   │
│  │  │     "idea_1": {                                                ││   │
│  │  │       id: "idea_1",                                            ││   │
│  │  │       parent_id: "root_0",                                     ││   │
│  │  │       type: "idea",                                            ││   │
│  │  │       summary: "Use neural architecture search",              ││   │
│  │  │       score: 0.85,                                             ││   │
│  │  │       status: "done",                                          ││   │
│  │  │       metadata: {provider: "tongyi", cost: 0.02},             ││   │
│  │  │       version: 2                                               ││   │
│  │  │     },                                                          ││   │
│  │  │     "hypothesis_1_1": {...},                                   ││   │
│  │  │     ...                                                         ││   │
│  │  │   },                                                            ││   │
│  │  │   edges: [                                                      ││   │
│  │  │     {from: "root_0", to: "idea_1"},                            ││   │
│  │  │     {from: "idea_1", to: "hypothesis_1_1"},                    ││   │
│  │  │     ...                                                         ││   │
│  │  │   ],                                                            ││   │
│  │  │   stats: {created: 15, expanded: 12, complete: false}          ││   │
│  │  │ }                                                               ││   │
│  │  │ version: 42  ← Incremented on each update                     ││   │
│  │  └────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

```

## Sequence Diagram: Scientific Research Flow

```
User → Frontend → Router → ResearchService → TSO → MCP → EventBus → Frontend

1. User submits: "Compare ML algorithms for time series forecasting"

2. Frontend → Router: POST /api/conversation/message
   Router classifies: intent = "scientific_research"

3. Router → ResearchService: POST /v1/research/start
   {prompt: "Compare ML algorithms...", type: "scientific"}

4. ResearchService:
   - Creates research_id = "research_xyz"
   - Initializes DB: ResearchTree with ROOT node
   - Spawns: asyncio.create_task(TSO.run(research_id))
   - Returns: {research_id: "research_xyz", status: "started"}

5. Frontend:
   - Opens WebSocket: ws://api/ws/research/research_xyz
   - Displays research tree panel
   - Shows root node

6. TSO (in background):
   Step 1: Generate 4 ideas in parallel
   ┌────────────────────────────────────┐
   │ Task1 → Tongyi: Generate idea 1    │
   │ Task2 → LLM: Generate idea 2       │ ← All run concurrently
   │ Task3 → Tongyi: Generate idea 3    │   (asyncio.gather)
   │ Task4 → LLM: Generate idea 4       │
   └────────────────────────────────────┘

   For each idea created:
     TSO → DB: Insert node
     TSO → EventBus: publish({type: "node_added", node: idea_1})
     EventBus → WebSocket → Frontend
     Frontend: Tree updates in real-time ✨

7. TSO:
   Step 2: Score ideas using UCB1
   idea_1: score=0.85 (neural architecture search)
   idea_2: score=0.72 (ensemble methods)
   idea_3: score=0.65 (transfer learning)
   idea_4: score=0.55 (reinforcement learning)

8. TSO:
   Step 3: Generate 3 hypotheses for each top-2 ideas (parallel)
   ┌────────────────────────────────────────────────┐
   │ Task1 → Tongyi: Hypothesis 1.1 for idea_1     │
   │ Task2 → Tongyi: Hypothesis 1.2 for idea_1     │
   │ Task3 → Tongyi: Hypothesis 1.3 for idea_1     │
   │ Task4 → LLM: Hypothesis 2.1 for idea_2        │ ← 6 tasks
   │ Task5 → LLM: Hypothesis 2.2 for idea_2        │   concurrent
   │ Task6 → LLM: Hypothesis 2.3 for idea_2        │
   └────────────────────────────────────────────────┘

   Events stream to frontend for each hypothesis created

9. TSO:
   Step 4: Score hypotheses, select top-4
   hyp_1_1: 0.92 ← Best
   hyp_1_2: 0.88
   hyp_2_1: 0.85
   hyp_1_3: 0.78

10. TSO:
    Step 5: Run experiments for top-4 hypotheses (parallel)
    ┌─────────────────────────────────────────────────────┐
    │ Task1 → RepoMaster: Run experiment for hyp_1_1     │
    │ Task2 → OpenHands: Run experiment for hyp_1_2      │
    │ Task3 → RepoMaster: Run experiment for hyp_2_1     │
    │ Task4 → OpenHands: Run experiment for hyp_1_3      │
    └─────────────────────────────────────────────────────┘

    Each experiment:
      - Generates test code
      - Executes in sandbox
      - Collects metrics
      - Updates node with results

    Events stream to frontend:
      {type: "node_updated", node: {id: "exp_1", status: "running"}}
      ... (later)
      {type: "node_updated", node: {id: "exp_1", status: "done", score: 0.95}}

11. TSO:
    Step 6: All experiments complete
    TSO → EventBus: publish({
      type: "complete",
      summary: {
        best_nodes: ["exp_1", "exp_3"],
        total_nodes: 15,
        duration_s: 245
      }
    })

12. Frontend:
    - Receives complete event
    - Shows ✅ Complete badge
    - Highlights best nodes
    - Shows summary panel with results
```

## Data Models

### Node Structure

```json
{
  "id": "idea_1_1234567890",
  "parent_id": "root_0",
  "type": "idea",
  "prompt": "Original prompt sent to generator",
  "summary": "Use neural architecture search to discover optimal LSTM variants for time series",
  "score": 0.85,
  "status": "done",
  "metadata": {
    "provider": "tongyi",
    "cost": 0.023,
    "latency_ms": 1245,
    "ts": "2025-10-04T11:30:00Z",
    "version": 3,
    "visits": 5,
    "avg_child_score": 0.88
  },
  "version": 3,
  "created_at": "2025-10-04T11:28:15Z",
  "updated_at": "2025-10-04T11:30:00Z"
}
```

### Research Tree Structure

```json
{
  "research_id": "research_xyz",
  "nodes": {
    "root_0": {...},
    "idea_1": {...},
    "idea_2": {...},
    "hypothesis_1_1": {...},
    "experiment_1_1_1": {...}
  },
  "edges": [
    {"from": "root_0", "to": "idea_1"},
    {"from": "root_0", "to": "idea_2"},
    {"from": "idea_1", "to": "hypothesis_1_1"},
    {"from": "hypothesis_1_1", "to": "experiment_1_1_1"}
  ],
  "stats": {
    "created": 15,
    "expanded": 12,
    "complete": true,
    "best_score": 0.95,
    "total_cost": 0.45
  },
  "version": 42
}
```

## Scoring Algorithm: UCB1

Upper Confidence Bound (UCB1) balances exploration vs exploitation:

```python
def ucb1_score(node: TreeNode, total_expanded: int, c: float = 1.414) -> float:
    """
    UCB1 = avg_score + c * sqrt(ln(N) / n)

    where:
    - avg_score: average score of this node and its children
    - N: total number of expansions so far
    - n: number of times this node has been visited
    - c: exploration constant (default sqrt(2))
    """
    avg_score = node.metadata.get("avg_child_score", node.score or 0.5)
    visits = node.metadata.get("visits", 1)

    exploration_bonus = c * math.sqrt(math.log(total_expanded) / visits)

    return avg_score + exploration_bonus
```

**Why UCB1?**
- High avg_score → Exploit known good paths
- Low visits → Explore under-explored paths
- Balances breadth and depth naturally

## Feature Flags

```python
# Config file or environment
RESEARCH_CONFIG = {
    "enabled": True,
    "beam_width": 4,              # Top K nodes to expand at each level
    "max_parallel": 8,            # Max concurrent async tasks
    "max_nodes": 300,             # Budget limit
    "timeout_s": 600,             # 10 minutes
    "ucb_c": 1.414,               # Exploration constant
    "providers": {
        "tongyi": {
            "enabled": True,
            "rate_limit": 10,     # requests/second
            "timeout": 30         # seconds
        },
        "repomaster": {
            "enabled": True,
            "rate_limit": 5,
            "timeout": 60
        }
    },
    "websocket": {
        "enabled": True,
        "queue_size": 100,
        "coalesce_interval_ms": 100
    }
}
```

## Frontend Component Tree

```
App
└── ConversationPage
    ├── ChatPanel (existing)
    └── ResearchTreeToggle (NEW)
        onClick → show/hide panel
        └── ResearchTreePanel (NEW)
            ├── Header
            │   └── "Research Tree: {title}"
            ├── TreeView
            │   └── VirtualizedTree
            │       └── TreeNode[] (recursive)
            │           ├── NodeIcon (💡/🔬/⚗️)
            │           ├── Summary
            │           ├── ScoreBadge
            │           ├── StatusIndicator (⚡/✅/❌)
            │           └── Children (collapsed/expanded)
            └── Footer
                └── Stats: {created} nodes, {expanded} expanded
```

## Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| SQLite JSON performance degrades | High | Medium | Use write-through cache, batch writes, consider PostgreSQL for production |
| WebSocket flood causes UI jank | Medium | High | Event coalescing, queue limits, throttling, virtualization |
| MCP provider failures | Medium | Medium | Circuit breaker, fallback to LLM, retry with backoff |
| Race conditions in parallel expansion | High | Low | Optimistic locking with versioning, DB-level uniqueness constraints |
| Request misclassification (false research) | Low | Medium | Manual override button, user feedback loop, improve classifier |
| Budget overrun (too many nodes) | Medium | Low | Hard limits, early termination, cost tracking |

## Success Metrics

**Performance:**
- Tree expansion: < 2s per node
- WebSocket latency: < 100ms
- UI responsiveness: 60 FPS even with 300 nodes

**Quality:**
- Idea novelty: > 0.7 avg score
- Hypothesis testability: > 0.8 avg score
- Experiment success rate: > 60%

**User Experience:**
- Time to first idea: < 10s
- Time to complete research: < 5 minutes
- User satisfaction: > 4/5 rating
