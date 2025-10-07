# UAgent Research Extension - Complete Mechanism Explained

## Overview

The UAgent Research Extension adds **autonomous parallel research capabilities** to OpenHands. When users ask complex questions, the system automatically triggers a tree-based research process that explores multiple approaches simultaneously.

---

## 🎯 What Happens When You Send a Message

### Simple Query (No Research Triggered)
```
User: "What is the capital of France?"
↓
OpenHands Agent responds directly
→ "Paris"
```

### Complex Research Query (Research Triggered)
```
User: "Please modify postgres and pg_duckdb source code to support vector search"
↓
Task Classifier analyzes the message
↓
Confidence: 0.95 (> 0.5 threshold)
↓
Research Mode Activated!
↓
Tree Search Orchestrator starts parallel exploration
```

---

## 🔍 Step-by-Step: What Happens Inside

### **Phase 1: Message Classification**

**Location**: `extensions/uagent_research/classifier/task_classifier.py`

When you send a message, the classifier checks:

1. **Research Keywords**: "research", "analyze", "investigate", "explore", "compare"
2. **Complexity Indicators**: "multiple", "various", "comprehensive", "systematic"
3. **Multi-stage Tasks**: "modify ... and ...", "first ... then ..."

**Example Classification**:
```python
Message: "Please modify postgres and pg_duckdb source code to support vector search"

Analysis:
- Has "modify" (action verb) ✓
- Has "and" (multi-part task) ✓
- Mentions multiple codebases ✓
- Complex technical work ✓

Result: TaskType.COMPLEX_RESEARCH, Confidence: 0.95
```

---

### **Phase 2: Research Middleware Intercepts**

**Location**: `extensions/uagent_research/middleware/research_middleware.py`

**Hook Points**:
1. **First message**: `openhands/server/services/conversation_service.py:165-206`
2. **Subsequent messages**: `openhands/server/session/session.py:367-398`

**What Happens**:
```python
# 1. Middleware receives user message
user_message = "Please modify postgres and pg_duckdb..."

# 2. Calls task classifier
should_trigger, confidence, reasoning = classifier.should_trigger_research(user_message)

# 3. If confidence >= 0.5 (configurable threshold)
if should_trigger:
    # 4. Generate experiment ID
    experiment_id = f"exp_{conversation_id}_{timestamp}_{random}"
    # Example: "exp_6411470ef5854f71bfecdfd7b6689330_1759761226_4991e0f1"

    # 5. Create TreeSearchOrchestrator
    orchestrator = TreeSearchOrchestrator(
        max_parallel=3,  # Run 3 branches simultaneously
        budget=Budget(max_iterations=50, max_cost=10.0)
    )

    # 6. Start research in background (non-blocking!)
    asyncio.create_task(orchestrator.run(goal=user_message))

    # 7. Return metadata to user
    return {
        'should_trigger_research': True,
        'experiment_id': experiment_id,
        'confidence': 0.95
    }
```

**System Message Shown to User**:
```
[System: Research mode activated - Experiment ID: exp_..., Confidence: 0.95.
Check the Research Tree tab for progress.]
```

---

### **Phase 3: Tree Search Orchestrator (PUCT Algorithm)**

**Location**: `extensions/uagent_research/orchestrator/tree_orchestrator.py`

This is the **core of the research system**. It uses PUCT (Predictor + Upper Confidence bounds applied to Trees), the same algorithm used in AlphaZero.

#### **3.1 Initialize Tree**

```python
# Create root node
root_node = ResearchNode(
    id="root",
    type=NodeType.ROOT,
    title="Research Root",
    content="Please modify postgres and pg_duckdb source code...",
    status=NodeStatus.COMPLETE
)

# Create tree
tree = ResearchTree(research_id=experiment_id)
tree.add_node(root_node)
```

**Tree Structure**:
```
root (COMPLETE)
├── (will expand to multiple ideas)
```

#### **3.2 PUCT Main Loop**

The orchestrator runs iterations, each iteration:

1. **Select** best node using PUCT score
2. **Expand** that node (generate children)
3. **Execute** children in parallel
4. **Backpropagate** results

**PUCT Formula**:
```
PUCT(node) = Q(node) + c * P(node) * sqrt(N(parent)) / (1 + N(node))

Where:
- Q(node) = Average value (0.0 to 1.0, success rate)
- P(node) = Prior probability (confidence, 0.0 to 1.0)
- N(parent) = Number of times parent was visited
- N(node) = Number of times node was visited
- c = Exploration constant (1.414 = sqrt(2))
```

This balances:
- **Exploitation**: Focus on high-value paths (Q term)
- **Exploration**: Try unexplored paths (U term)

---

#### **3.3 Iteration 1: Expand Root**

**Code**: `tree_orchestrator.py:274-349`

```python
# Select best node → Root (only option)
node = root

# Expand root → Generate 3 ideas
children = [
    ResearchNode(
        id="idea-0",
        type=NodeType.IDEA,
        title="Idea 1: Web Research",
        content="Search web for: postgres vector search implementation",
        status=NodeStatus.PENDING,
        prior=0.8  # High confidence
    ),
    ResearchNode(
        id="idea-1",
        type=NodeType.IDEA,
        title="Idea 2: Web Research",
        content="Search web for: pg_duckdb vector extension",
        status=NodeStatus.PENDING,
        prior=0.8
    ),
    ResearchNode(
        id="idea-2",
        type=NodeType.IDEA,
        title="Idea 3: Code Research",
        content="Search GitHub for: postgres pg_duckdb vector implementations",
        status=NodeStatus.PENDING,
        prior=0.7  # Slightly lower confidence
    )
]

# Add to tree
for child in children:
    tree.add_node(child, parent_id="root")
```

**Tree Now**:
```
root (COMPLETE)
├── idea-0 (PENDING) "Web Research: postgres vector search"
├── idea-1 (PENDING) "Web Research: pg_duckdb vector"
└── idea-2 (PENDING) "Code Research: GitHub implementations"
```

---

#### **3.4 Execute Children in Parallel**

**Code**: `tree_orchestrator.py:351-371`

```python
async def _execute_children_parallel(children):
    tasks = []
    for child in children:
        task = asyncio.create_task(_execute_node(child))
        tasks.append(task)

    # Wait for all 3 to complete
    await asyncio.gather(*tasks)
```

**For Each Child**:

1. **Route to Adapter** (`tree_orchestrator.py:372-440`)

```python
# Create task
task = Task(
    id="idea-0",
    goal="Search web for: postgres vector search implementation",
    context="Web Research"
)

# Route using SkillRouter
adapter_name = router.route(task, context)
# Returns: "deepresearch" (for web searches)

adapter = adapter_registry.get("deepresearch")
# Gets: DeepResearchAdapter
```

2. **Adapter Executes Task**

**DeepResearchAdapter** (`extensions/uagent_research/adapters/deepresearch/adapter.py`):

```python
async def run(task, context):
    # 1. Use Bing Search Tool
    search_results = await bing_search(
        query="postgres vector search implementation",
        count=10
    )

    # 2. Browse top results
    for result in search_results[:3]:
        content = await web_browse(result.url)
        # Extract relevant information

    # 3. Emit events
    yield ResearchEvent(
        type=EventType.SEARCH,
        data={"results": search_results}
    )

    yield ResearchEvent(
        type=EventType.BROWSE,
        data={"url": result.url, "content": content}
    )

    # 4. Complete
    yield CompleteEvent(
        type=EventType.COMPLETE,
        data={"summary": "Found 3 implementations..."}
    )
```

3. **Update Node**

```python
# On completion
node.status = NodeStatus.COMPLETE
node.visits += 1
node.avg_value = 0.8  # Success!
node.cost = 0.05  # API costs

stats["completed_nodes"] += 1
stats["total_cost"] += 0.05
```

**Tree After Execution**:
```
root (COMPLETE)
├── idea-0 (COMPLETE, visits=1, value=0.8) "Found pgvector extension"
├── idea-1 (COMPLETE, visits=1, value=0.8) "Found DuckDB vector docs"
└── idea-2 (COMPLETE, visits=1, value=0.8) "Found 5 GitHub repos"
```

---

#### **3.5 Publish Tree to API**

**Code**: `tree_orchestrator.py:456-501`

After each iteration:

```python
def _publish_tree_to_api():
    tree_snapshot = {
        "version": 1,  # Iteration number
        "timestamp": "2025-10-06T14:39:40",
        "experiment_id": "exp_...",
        "data": {
            "nodes": [
                {
                    "id": "root",
                    "type": "ROOT",
                    "title": "Research Root",
                    "status": "COMPLETE",
                    "visits": 1,
                    "avg_value": 0.0,
                    ...
                },
                {
                    "id": "idea-0",
                    "type": "IDEA",
                    "title": "Web Research: postgres",
                    "status": "COMPLETE",
                    "visits": 1,
                    "avg_value": 0.8,
                    ...
                }
            ],
            "edges": [
                {"source": "root", "target": "idea-0"},
                {"source": "root", "target": "idea-1"},
                {"source": "root", "target": "idea-2"}
            ],
            "stats": {
                "total_nodes": 4,
                "total_edges": 3,
                "total_cost": 0.15,
                "max_depth": 1
            }
        }
    }

    # Store in API endpoint
    update_tree_state(experiment_id, tree_snapshot)
```

Now frontend can fetch this data!

---

#### **3.6 Iteration 2: Select and Expand Best Idea**

**PUCT Calculation**:

```python
For idea-0:
    Q = 0.8 (avg_value)
    P = 0.8 (prior)
    N(parent) = 1 (root visits)
    N(node) = 1 (idea-0 visits)

    U = 1.414 * 0.8 * sqrt(1) / (1 + 1) = 0.566
    PUCT = 0.8 + 0.566 = 1.366

For idea-1:
    PUCT = 0.8 + 0.566 = 1.366

For idea-2:
    Q = 0.8
    P = 0.7 (lower prior)
    U = 1.414 * 0.7 * sqrt(1) / 2 = 0.495
    PUCT = 0.8 + 0.495 = 1.295
```

**Winner**: idea-0 or idea-1 (tie, randomly pick idea-0)

**Expand idea-0** → Generate 2 hypotheses:

```python
children = [
    ResearchNode(
        id="idea-0-hyp-0",
        type=NodeType.HYPOTHESIS,
        title="Hypothesis 1",
        content="Test using pgvector extension approach",
        status=NodeStatus.PENDING,
        prior=0.6
    ),
    ResearchNode(
        id="idea-0-hyp-1",
        type=NodeType.HYPOTHESIS,
        title="Hypothesis 2",
        content="Test custom vector implementation",
        status=NodeStatus.PENDING,
        prior=0.6
    )
]
```

**Tree Now**:
```
root (COMPLETE)
├── idea-0 (COMPLETE)
│   ├── idea-0-hyp-0 (PENDING) "Use pgvector"
│   └── idea-0-hyp-1 (PENDING) "Custom implementation"
├── idea-1 (COMPLETE)
└── idea-2 (COMPLETE)
```

Execute hypotheses in parallel, repeat...

---

#### **3.7 Iteration 3+: Deeper Exploration**

Each hypothesis can spawn an **experiment**:

```python
ResearchNode(
    id="idea-0-hyp-0-exp",
    type=NodeType.EXPERIMENT,
    title="Run Experiment",
    content="Execute: Modify postgres to use pgvector",
    status=NodeStatus.PENDING,
    prior=0.5
)
```

**Routed to**: CodeActAdapter (for code execution)

**CodeActAdapter** executes:
1. Clone postgres repo
2. Analyze code structure
3. Make modifications
4. Run tests
5. Report results

**Final Tree** (after 5 iterations):
```
root
├── idea-0 "Web: postgres vector"
│   ├── hyp-0 "Use pgvector"
│   │   └── exp "Modify postgres" → SUCCESS (0.9)
│   └── hyp-1 "Custom impl"
│       └── exp "Build from scratch" → FAILED (0.1)
├── idea-1 "Web: pg_duckdb"
│   ├── hyp-0 "Integrate DuckDB"
│   │   └── exp "Modify pg_duckdb" → SUCCESS (0.85)
│   └── hyp-1 "Fork and extend"
│       └── exp "Create fork" → RUNNING
└── idea-2 "GitHub: implementations"
    └── hyp-0 "Use existing lib"
        └── exp "Test lib XYZ" → SUCCESS (0.95) ⭐ BEST
```

PUCT will explore the most promising paths more!

---

## 🔌 Integration Points with OpenHands

### **1. Message Interception**

**When**: User sends any message to OpenHands

**Where**:
- First message: `openhands/server/services/conversation_service.py:165-206`
- Follow-up messages: `openhands/server/session/session.py:367-398`

**How**:
```python
# In conversation_service.py
if RESEARCH_MIDDLEWARE_AVAILABLE and initial_user_msg:
    result = await research_middleware.process_message(
        user_message=initial_user_msg,
        session_id=conversation_id
    )

    if result.get('should_trigger_research'):
        # Append system message to conversation
        system_msg = f"[System: Research mode activated - Experiment ID: {result['experiment_id']}]"
```

The regular OpenHands agent **continues to run normally**. Research happens **in parallel** in the background!

---

### **2. Research Tools Available to Adapters**

Adapters can use these tools:

#### **BingSearchTool** (`extensions/uagent_research/tools/search/bing_search_tool.py`)
```python
results = await bing_search(
    query="postgres vector search",
    count=10,
    market="en-US"
)
# Returns: List of search results with titles, URLs, snippets
```

#### **WebBrowseTool** (`extensions/uagent_research/tools/browse/web_browse_tool.py`)
```python
content = await web_browse(
    url="https://github.com/pgvector/pgvector",
    max_chars=50000
)
# Returns: Cleaned markdown content from page
```

These tools are **separate from OpenHands' native tools** and designed for research tasks.

---

### **3. Adapter Types**

Three specialized adapters handle different research tasks:

#### **DeepResearchAdapter** (`extensions/uagent_research/adapters/deepresearch/adapter.py`)
- **Purpose**: Web research, information gathering
- **Uses**: BingSearchTool, WebBrowseTool
- **Routed for**: Keywords like "search", "find", "research", "investigate"

#### **RepoMasterAdapter** (`extensions/uagent_research/adapters/repomaster/adapter.py`)
- **Purpose**: Code repository analysis
- **Uses**: GitHub API, code parsing tools
- **Routed for**: Keywords like "github", "repository", "codebase", "source code"

#### **CodeActAdapter** (`extensions/uagent_research/adapters/codeact/adapter.py`)
- **Purpose**: Code execution, experiments
- **Uses**: OpenHands sandbox, code execution
- **Routed for**: Keywords like "execute", "run", "test", "implement"

---

### **4. Skill Router**

**Location**: `extensions/uagent_research/router/skill_router.py`

Routes tasks to appropriate adapters:

```python
def route(task: Task, context: Context) -> str:
    # Analyze task content
    if "search" in task.goal.lower() or "find" in task.goal.lower():
        return "deepresearch"

    elif "github" in task.goal.lower() or "repository" in task.goal.lower():
        return "repomaster"

    elif "execute" in task.goal.lower() or "run" in task.goal.lower():
        return "codeact"

    else:
        # Default to deepresearch for general research
        return "deepresearch"
```

---

### **5. Event Bus (Real-time Updates)**

**Location**: `extensions/uagent_research/orchestrator/event_bus.py`

All research events flow through the event bus:

```python
# Orchestrator publishes events
await event_bus.publish(ResearchEvent(
    type=EventType.SEARCH,
    node_id="idea-0",
    data={"query": "postgres vector", "results_count": 10}
))

# Frontend subscribes via WebSocket
# File: extensions/uagent_research/orchestrator/ws_publisher.py
await websocket.send_json({
    "type": "SEARCH",
    "node_id": "idea-0",
    "data": {...}
})
```

**Frontend receives**:
- Node creation events
- Status updates (PENDING → RUNNING → COMPLETE)
- Search results
- Error messages
- Completion summaries

---

## 🎨 Frontend Visualization

### **Research Tree Panel**

**Location**: `frontend/src/routes/research-tab.tsx`

1. **Polls API every 3 seconds**:
```typescript
const fetchTree = async () => {
    const response = await fetch(
        `/api/research/experiments/${experimentId}/tree`
    );
    const data = await response.json();
    setTreeData(data.data);
};

useEffect(() => {
    const interval = setInterval(fetchTree, 3000);
    return () => clearInterval(interval);
}, []);
```

2. **Displays Tree**:
```tsx
<ResearchTree
    nodes={treeData.nodes}
    edges={treeData.edges}
    stats={treeData.stats}
/>
```

Renders as interactive graph:
- **Nodes**: Colored by status (gray=pending, blue=running, green=complete, red=failed)
- **Edges**: Show parent-child relationships
- **Tooltips**: Show node details on hover
- **Click**: Expand node to see content

---

## 📊 Configuration

### **Environment Variables** (`.env`)

```bash
# Enable/disable auto-triggering
ENABLE_AUTO_RESEARCH_TRIGGER=true

# Confidence threshold (0.0 to 1.0)
# Higher = more selective, Lower = more aggressive
RESEARCH_CONFIDENCE_THRESHOLD=0.5

# Budget limits
RESEARCH_MAX_ITERATIONS=50      # Max PUCT iterations
RESEARCH_MAX_COST=10.0          # Max $ to spend
RESEARCH_MAX_PARALLEL=3         # Max parallel branches

# API keys for tools
BING_API_KEY=your_bing_key
GITHUB_TOKEN=your_github_token
```

---

## 🔄 Complete Flow Example

Let's trace a complete example:

### **User Message**:
```
"Research neural architecture search and implement the best approach"
```

### **Flow**:

```
1. User sends message
   ↓
2. Task Classifier analyzes
   → "research" keyword ✓
   → "implement" (complex) ✓
   → Confidence: 0.85
   ↓
3. Middleware triggers research
   → Creates experiment_id
   → Starts TreeSearchOrchestrator
   → Shows system message to user
   ↓
4. Iteration 1: Expand root
   → Generate 3 ideas
   → Execute in parallel:
      - idea-0: Web search "neural architecture search"
      - idea-1: Web search "NAS implementations"
      - idea-2: GitHub search "neural architecture search"
   ↓
5. All complete successfully
   → idea-0: value=0.8
   → idea-1: value=0.85
   → idea-2: value=0.9 ⭐
   → Publish tree (version 1)
   ↓
6. Iteration 2: Select idea-2 (highest PUCT)
   → Expand to 2 hypotheses:
      - hyp-0: "Use DARTS implementation"
      - hyp-1: "Use ProxylessNAS"
   → Execute in parallel (RepoMasterAdapter)
      - Clone and analyze repos
   ↓
7. Iteration 3: Select hyp-0 (DARTS scored higher)
   → Generate experiment:
      - exp: "Implement DARTS in sandbox"
   → Execute (CodeActAdapter):
      - Clone repo
      - Set up environment
      - Run tests
      - SUCCESS! value=0.95
   → Publish tree (version 3)
   ↓
8. Budget check: 3 iterations done, continue?
   → Cost: $0.45 (< $10 limit) ✓
   → Continue...
   ↓
9. Iteration 4: Explore alternative (hyp-1)
   → Generate experiment for ProxylessNAS
   → Execute...
   → SUCCESS! value=0.88 (lower than DARTS)
   ↓
10. Iteration 5: DARTS path has highest value
    → Backpropagate success
    → All nodes updated
    ↓
11. Max iterations reached (or budget exhausted)
    → Research complete
    → Final tree published
    ↓
12. User sees in frontend:
    → Tree visualization with all paths explored
    → Best path highlighted (DARTS: 0.95)
    → Can click nodes to see details
```

---

## 🧠 Key Insights

### **Why PUCT?**

PUCT balances:
1. **Exploitation**: Focus on what works (high Q value)
2. **Exploration**: Try new things (high U term for unvisited nodes)

This means:
- Early iterations: Broad exploration (try many ideas)
- Later iterations: Deep exploitation (focus on best path)

### **Why Parallel Execution?**

Instead of:
```
Search web → wait → analyze results → wait → browse pages → wait...
(Sequential, slow)
```

We do:
```
Search web for A | Search web for B | Search GitHub for C
    ↓                  ↓                      ↓
   Browse A1       Browse B1              Clone C1
    ↓                  ↓                      ↓
   Browse A2       Browse B2              Analyze C1
    ↓                  ↓                      ↓
   Done (2.3s)    Done (1.8s)           Done (3.1s)

Total: 3.1s (not 7.2s!)
```

### **Why Separate from OpenHands Agent?**

- **OpenHands Agent**: Interactive, user-guided, single-threaded
- **Research System**: Autonomous, parallel, exploratory

They complement each other:
- Agent handles direct user interaction
- Research explores in background
- User gets best of both worlds

---

## 🛠️ For Developers: Extending the System

### **Add a New Adapter**

1. Create adapter:
```python
# extensions/uagent_research/adapters/myadapter/adapter.py
class MyAdapter(AgentAdapter):
    def __init__(self):
        super().__init__(
            name="myadapter",
            description="Does cool stuff",
            cost_per_step=0.01
        )

    async def run(self, task: Task, context: Context):
        # Do work
        yield ResearchEvent(...)
        yield CompleteEvent(...)
```

2. Register in orchestrator:
```python
# tree_orchestrator.py
from ..adapters.myadapter.adapter import MyAdapter

adapter_registry.register(MyAdapter())
```

3. Update router:
```python
# router/skill_router.py
if "my_keyword" in task.goal.lower():
    return "myadapter"
```

### **Add a New Tool**

```python
# extensions/uagent_research/tools/mytool/my_tool.py
async def my_tool(param: str) -> dict:
    """Does something useful"""
    # Implementation
    return {"result": "..."}
```

Use in adapter:
```python
from ...tools.mytool.my_tool import my_tool

result = await my_tool("input")
```

---

## 🎯 Current Limitations & Future Work

### **Current Limitations**:

1. **Fixed Tree Structure**: ROOT → IDEA → HYPOTHESIS → EXPERIMENT
   - Should be dynamic based on task type

2. **Hardcoded Node Generation**: Creates fixed number of children
   - Should use LLM to generate ideas

3. **No Result Aggregation**: Each branch works independently
   - Should synthesize results across branches

4. **No Persistent Storage**: Tree lost when server restarts
   - Should save to database

5. **Limited Adapter Intelligence**: Simple keyword routing
   - Should use LLM for smarter routing

### **Future Enhancements**:

1. **Dynamic Tree Generation**:
```python
# Use LLM to generate child nodes
ideas = await llm.generate_ideas(
    goal=goal,
    parent_results=parent.results,
    count=3
)
```

2. **Cross-Branch Synthesis**:
```python
# Combine results from multiple branches
summary = await llm.synthesize(
    branch_results=[branch1.result, branch2.result, branch3.result]
)
```

3. **Persistent Trees**:
```python
# Save to PostgreSQL
await db.save_tree(tree)

# Resume later
tree = await db.load_tree(experiment_id)
await orchestrator.resume(tree)
```

4. **Smart Routing**:
```python
# Use LLM to select adapter
adapter_name = await llm.route(
    task=task,
    available_adapters=["deepresearch", "repomaster", "codeact"]
)
```

5. **Reward Modeling**:
```python
# Learn from user feedback
if user.feedback == "helpful":
    tree.update_rewards(path=best_path, reward=1.0)
    model.learn(tree)
```

---

## 📚 Summary

**UAgent Research Extension** = **Autonomous Parallel Research** for OpenHands

**Key Components**:
1. **Task Classifier**: Decides when to trigger research
2. **Research Middleware**: Intercepts messages, starts orchestrator
3. **Tree Orchestrator**: PUCT-based parallel exploration
4. **Adapters**: Execute different types of tasks (web, code, experiments)
5. **Tools**: Search, browse, analyze (separate from OpenHands tools)
6. **Event Bus**: Real-time updates to frontend
7. **API Endpoints**: Serve tree data to visualization

**How It Works**:
- Complex query → Classified → Research triggered (background)
- PUCT explores multiple paths in parallel
- Each path uses specialized adapters + tools
- Tree published to API after each iteration
- Frontend visualizes progress in real-time
- Best path emerges through exploration + exploitation

**Integration with OpenHands**:
- Hooks into message flow (non-invasive)
- Runs in background (non-blocking)
- Uses separate tools (no conflicts)
- Complements regular agent (not replacing)

The result: **Smarter, faster, autonomous research** while OpenHands agent handles user interaction! 🚀
