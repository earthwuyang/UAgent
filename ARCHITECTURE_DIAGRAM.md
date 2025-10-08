# UAgent Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                              │
│                         (React Frontend @ :3000)                         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ HTTP/WebSocket
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI SERVER                                  │
│                      (openhands/server/app.py)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    RESEARCH MIDDLEWARE                          │   │
│  │         (extensions/uagent_research/middleware/)                │   │
│  │                                                                  │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │           TASK CLASSIFIER                                 │ │   │
│  │  │  (classifier/task_classifier.py)                         │ │   │
│  │  │                                                           │ │   │
│  │  │  • Analyzes user query                                   │ │   │
│  │  │  • Calculates confidence score                           │ │   │
│  │  │  • Decides: trigger research?                            │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  │                          ↓                                      │   │
│  │                   If confidence >= 0.7                          │   │
│  │                          ↓                                      │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │      TREE SEARCH ORCHESTRATOR                            │ │   │
│  │  │  (orchestrator/tree_orchestrator.py)                     │ │   │
│  │  │                                                           │ │   │
│  │  │  ┌─────────────────────────────────────────────────┐    │ │   │
│  │  │  │         PUCT ALGORITHM                          │    │ │   │
│  │  │  │                                                  │    │ │   │
│  │  │  │  Loop:                                           │    │ │   │
│  │  │  │    1. SELECT best node (PUCT score)            │    │ │   │
│  │  │  │    2. EXPAND node (generate children)          │    │ │   │
│  │  │  │    3. EXECUTE children (parallel)              │    │ │   │
│  │  │  │    4. BACKPROPAGATE results                    │    │ │   │
│  │  │  │    5. PUBLISH tree to API                      │    │ │   │
│  │  │  │                                                  │    │ │   │
│  │  │  │  Until: budget exhausted or goal reached       │    │ │   │
│  │  │  └─────────────────────────────────────────────────┘    │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    RESEARCH API                                 │   │
│  │         (api/research_routes.py)                               │   │
│  │                                                                  │   │
│  │  GET  /api/research/experiments/{id}/tree                      │   │
│  │  GET  /api/research/experiments/{id}                           │   │
│  │  POST /api/research/experiments/{id}/pause                     │   │
│  │  POST /api/research/experiments/{id}/resume                    │   │
│  │  POST /api/research/experiments/{id}/stop                      │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          SKILL ROUTER                                    │
│                    (router/skill_router.py)                             │
│                                                                          │
│  Analyzes task → Routes to appropriate adapter                          │
└────────────────┬────────────────┬────────────────┬───────────────────────┘
                 │                │                │
        ┌────────┘                │                └────────┐
        │                         │                         │
        ↓                         ↓                         ↓
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  DEEPRESEARCH    │    │   REPOMASTER     │    │    CODEACT       │
│    ADAPTER       │    │    ADAPTER       │    │    ADAPTER       │
│                  │    │                  │    │                  │
│  Web Research    │    │  Code Research   │    │  Code Execution  │
│  • Bing Search   │    │  • GitHub Search │    │  • Shell Cmds    │
│  • Web Browse    │    │  • AST Analysis  │    │  • File Ops      │
│  • Extract Info  │    │  • Repo Analysis │    │  • Testing       │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ↓
                    ┌────────────────────────┐
                    │    RESEARCH TOOLS      │
                    │  (tools/)              │
                    │                        │
                    │  • Bing Search         │
                    │  • GitHub Search       │
                    │  • Web Browse          │
                    │  • Code Analysis       │
                    │  • File Operations     │
                    └────────────────────────┘
                                 │
                                 ↓
                    ┌────────────────────────┐
                    │    EXTERNAL APIS       │
                    │                        │
                    │  • Bing Search API     │
                    │  • GitHub API          │
                    │  • Anthropic API       │
                    │  • OpenAI API          │
                    └────────────────────────┘
```

---

## Data Flow: User Query → Research Results

```
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 1: USER SENDS QUERY                                             │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ "Modify postgres and pg_duckdb 
                                 │  to support vector search"
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 2: TASK CLASSIFICATION                                          │
│                                                                       │
│  Task Classifier analyzes:                                           │
│  ✓ Keywords: "modify", "support"                                     │
│  ✓ Complexity: Multiple codebases                                    │
│  ✓ Multi-stage: "and" connector                                      │
│                                                                       │
│  Result: TaskType.COMPLEX_RESEARCH, Confidence: 0.95                 │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ confidence >= 0.7 ✓
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 3: RESEARCH INITIALIZATION                                      │
│                                                                       │
│  • Generate experiment_id: "exp_6411470e..."                         │
│  • Create TreeSearchOrchestrator                                     │
│  • Initialize budget: 50 iterations, $10 max                         │
│  • Start research (background task)                                  │
│  • Return to user: "Research mode activated"                         │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ async background
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 4: TREE INITIALIZATION                                          │
│                                                                       │
│  Create ROOT node:                                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ id: "root"                                                      │ │
│  │ type: ROOT                                                      │ │
│  │ content: "Modify postgres and pg_duckdb..."                     │ │
│  │ status: COMPLETE                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 5: ITERATION 1 - EXPAND ROOT                                   │
│                                                                       │
│  Generate 3 IDEAS:                                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ IDEA-0: "Web research: postgres vector search"                 │ │
│  │ IDEA-1: "Web research: pg_duckdb vector"                        │ │
│  │ IDEA-2: "Code research: GitHub implementations"                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Tree structure:                                                     │
│  ROOT                                                                │
│  ├── IDEA-0 (PENDING)                                                │
│  ├── IDEA-1 (PENDING)                                                │
│  └── IDEA-2 (PENDING)                                                │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 6: PARALLEL EXECUTION                                           │
│                                                                       │
│  For each IDEA (parallel):                                           │
│                                                                       │
│  IDEA-0 → SkillRouter → DeepResearchAdapter                          │
│           ↓                                                           │
│           Bing Search: "postgres vector search"                      │
│           ↓                                                           │
│           Browse top 3 results                                       │
│           ↓                                                           │
│           Extract: "Found pgvector extension"                        │
│           ↓                                                           │
│           Update node: status=COMPLETE, value=0.8                    │
│                                                                       │
│  IDEA-1 → DeepResearchAdapter → "Found DuckDB vector docs"           │
│  IDEA-2 → RepoMasterAdapter → "Found 5 GitHub repos"                 │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 7: PUBLISH TREE                                                 │
│                                                                       │
│  Serialize tree to JSON:                                             │
│  {                                                                    │
│    "version": 1,                                                     │
│    "experiment_id": "exp_6411470e...",                               │
│    "data": {                                                         │
│      "nodes": [                                                      │
│        {"id": "root", "type": "ROOT", ...},                          │
│        {"id": "idea-0", "type": "IDEA", "status": "COMPLETE", ...},  │
│        {"id": "idea-1", "type": "IDEA", "status": "COMPLETE", ...},  │
│        {"id": "idea-2", "type": "IDEA", "status": "COMPLETE", ...}   │
│      ],                                                              │
│      "edges": [                                                      │
│        {"source": "root", "target": "idea-0"},                       │
│        {"source": "root", "target": "idea-1"},                       │
│        {"source": "root", "target": "idea-2"}                        │
│      ],                                                              │
│      "stats": {                                                      │
│        "total_nodes": 4,                                             │
│        "total_cost": 0.15,                                           │
│        "max_depth": 1                                                │
│      }                                                               │
│    }                                                                 │
│  }                                                                    │
│                                                                       │
│  POST to: /api/research/experiments/{id}/tree                        │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 8: ITERATION 2 - SELECT BEST IDEA                              │
│                                                                       │
│  Calculate PUCT scores:                                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ IDEA-0: Q=0.8, P=0.8, N=1 → PUCT = 1.366                       │ │
│  │ IDEA-1: Q=0.8, P=0.8, N=1 → PUCT = 1.366                       │ │
│  │ IDEA-2: Q=0.8, P=0.7, N=1 → PUCT = 1.295                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Select: IDEA-0 (highest PUCT)                                       │
│                                                                       │
│  Expand IDEA-0 → Generate 2 HYPOTHESES:                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ HYP-0: "Use pgvector extension approach"                        │ │
│  │ HYP-1: "Implement custom vector type"                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Tree structure:                                                     │
│  ROOT                                                                │
│  ├── IDEA-0 (COMPLETE)                                               │
│  │   ├── HYP-0 (PENDING)                                             │
│  │   └── HYP-1 (PENDING)                                             │
│  ├── IDEA-1 (COMPLETE)                                               │
│  └── IDEA-2 (COMPLETE)                                               │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 9: CONTINUE ITERATIONS                                          │
│                                                                       │
│  Repeat: SELECT → EXPAND → EXECUTE → BACKPROPAGATE → PUBLISH        │
│                                                                       │
│  Until:                                                              │
│  • Max iterations reached (50)                                       │
│  • Max cost exceeded ($10)                                           │
│  • All paths exhausted                                               │
│  • User stops research                                               │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 10: FINAL RESULTS                                               │
│                                                                       │
│  Final tree (example):                                               │
│  ROOT                                                                │
│  ├── IDEA-0 (COMPLETE, value=0.85)                                   │
│  │   ├── HYP-0 (COMPLETE, value=0.9)                                 │
│  │   │   ├── EXP-0 (COMPLETE, value=1.0) ← Best result!             │
│  │   │   └── EXP-1 (COMPLETE, value=0.8)                             │
│  │   └── HYP-1 (COMPLETE, value=0.7)                                 │
│  ├── IDEA-1 (COMPLETE, value=0.75)                                   │
│  │   └── HYP-2 (COMPLETE, value=0.75)                                │
│  └── IDEA-2 (COMPLETE, value=0.8)                                    │
│      └── HYP-3 (COMPLETE, value=0.8)                                 │
│                                                                       │
│  Stats:                                                              │
│  • Total nodes: 15                                                   │
│  • Total cost: $4.50                                                 │
│  • Best path: ROOT → IDEA-0 → HYP-0 → EXP-0                          │
│  • Success rate: 87%                                                 │
│                                                                       │
│  Synthesize findings → Return to user                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT INTERACTIONS                           │
└─────────────────────────────────────────────────────────────────────────┘

User Query
    │
    ↓
┌─────────────────┐
│ FastAPI Server  │
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│ Research Middleware     │
│                         │
│ 1. Intercept request    │
│ 2. Call classifier      │────────→ ┌──────────────────┐
│ 3. Check confidence     │          │ Task Classifier  │
│ 4. Trigger research     │←─────────│                  │
└────────┬────────────────┘          │ • Analyze query  │
         │                           │ • Return score   │
         │                           └──────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Tree Search Orchestrator                                    │
│                                                              │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│ │   SELECT     │→ │   EXPAND     │→ │   EXECUTE    │      │
│ │              │  │              │  │              │      │
│ │ • Calculate  │  │ • Generate   │  │ • Route task │      │
│ │   PUCT       │  │   children   │  │ • Run adapter│      │
│ │ • Pick best  │  │ • Add to tree│  │ • Collect    │      │
│ │   node       │  │              │  │   results    │      │
│ └──────────────┘  └──────────────┘  └──────┬───────┘      │
│                                             │              │
│ ┌──────────────┐  ┌──────────────┐         │              │
│ │   PUBLISH    │← │ BACKPROPAGATE│←────────┘              │
│ │              │  │              │                         │
│ │ • Serialize  │  │ • Update     │                         │
│ │   tree       │  │   values     │                         │
│ │ • POST API   │  │ • Propagate  │                         │
│ └──────┬───────┘  └──────────────┘                         │
└────────┼────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────┐
│ Skill Router                                                │
│                                                              │
│ Analyze task → Determine adapter                            │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         ↓              ↓              ↓              ↓
┌────────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐
│ DeepResearch   │ │ RepoMaster │ │  CodeAct   │ │  Custom  │
│   Adapter      │ │  Adapter   │ │  Adapter   │ │  Adapter │
└────────┬───────┘ └──────┬─────┘ └──────┬─────┘ └────┬─────┘
         │                │               │            │
         └────────────────┴───────────────┴────────────┘
                          │
                          ↓
                 ┌────────────────┐
                 │ Research Tools │
                 │                │
                 │ • Search       │
                 │ • Browse       │
                 │ • Code Analyze │
                 └────────┬───────┘
                          │
                          ↓
                 ┌────────────────┐
                 │ External APIs  │
                 │                │
                 │ • Bing         │
                 │ • GitHub       │
                 │ • LLMs         │
                 └────────────────┘
```

---

## PUCT Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PUCT ALGORITHM FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

Start
  │
  ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Initialize Tree                                                          │
│                                                                          │
│ ROOT ← Create root node with user query                                 │
│ tree ← ResearchTree(root)                                                │
│ budget ← Budget(max_iterations=50, max_cost=10.0)                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ↓
                        ┌────────────────┐
                        │ Budget OK?     │
                        │ (iterations <  │
                        │  max, cost <   │
                        │  max)          │
                        └────┬───────────┘
                             │
                    ┌────────┴────────┐
                    │ Yes             │ No → End
                    ↓                 ↓
         ┌──────────────────┐    ┌──────────────┐
         │ SELECT           │    │ Synthesize   │
         │                  │    │ Results      │
         │ For each leaf:   │    └──────────────┘
         │   Calculate PUCT │
         │   = Q + c*P*√N/n │
         │                  │
         │ Pick max PUCT    │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ EXPAND           │
         │                  │
         │ Generate children│
         │ based on type:   │
         │                  │
         │ ROOT → IDEAS     │
         │ IDEA → HYPS      │
         │ HYP  → EXPS      │
         │                  │
         │ Add to tree      │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ EXECUTE          │
         │                  │
         │ For each child:  │
         │   1. Route task  │
         │   2. Get adapter │
         │   3. Run async   │
         │                  │
         │ Wait all done    │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ BACKPROPAGATE    │
         │                  │
         │ For each child:  │
         │   Update:        │
         │   • visits += 1  │
         │   • avg_value    │
         │   • cost         │
         │                  │
         │ Propagate up     │
         └────────┬─────────┘
                  │
                  ↓
         ┌──────────────────┐
         │ PUBLISH          │
         │                  │
         │ Serialize tree   │
         │ POST to API      │
         │ Notify frontend  │
         └────────┬─────────┘
                  │
                  ↓
                  └──────────→ Loop back to "Budget OK?"
```

---

## Adapter Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ADAPTER EXECUTION FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

Task from Orchestrator
         │
         ↓
┌────────────────────┐
│ Skill Router       │
│                    │
│ Analyze task:      │
│ • Keywords         │
│ • Context          │
│ • Node type        │
│                    │
│ Determine adapter  │
└─────────┬──────────┘
          │
          ├─────────────────┬─────────────────┬─────────────────┐
          ↓                 ↓                 ↓                 ↓
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ DeepResearch     │ │ RepoMaster   │ │  CodeAct     │ │   Custom     │
└────────┬─────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                  │                │                │
         ↓                  ↓                ↓                ↓
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 1. Initialize    │ │ 1. Initialize│ │ 1. Initialize│ │ 1. Initialize│
│ 2. Use tools:    │ │ 2. Use tools:│ │ 2. Use tools:│ │ 2. Use tools:│
│    • Bing Search │ │    • GitHub  │ │    • Shell   │ │    • Custom  │
│    • Web Browse  │ │    • AST     │ │    • Files   │ │              │
│ 3. Extract info  │ │ 3. Analyze   │ │ 3. Execute   │ │ 3. Process   │
│ 4. Yield events  │ │ 4. Yield     │ │ 4. Yield     │ │ 4. Yield     │
└────────┬─────────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                  │                │                │
         └──────────────────┴────────────────┴────────────────┘
                            │
                            ↓
                   ┌────────────────┐
                   │ Event Stream   │
                   │                │
                   │ • SEARCH       │
                   │ • BROWSE       │
                   │ • ANALYZE      │
                   │ • EXECUTE      │
                   │ • COMPLETE     │
                   └────────┬───────┘
                            │
                            ↓
                   ┌────────────────┐
                   │ Update Node    │
                   │                │
                   │ • status       │
                   │ • value        │
                   │ • cost         │
                   │ • results      │
                   └────────┬───────┘
                            │
                            ↓
                   Return to Orchestrator
```

---

*Architecture Diagrams v1.0 - Last Updated: 2025-01-06*