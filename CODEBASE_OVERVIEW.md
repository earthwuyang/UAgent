# UAgent Codebase Overview

## Project Summary

**UAgent** is a research-enhanced fork of **OpenHands** (formerly OpenDevin), an AI-powered software development platform. This fork adds autonomous parallel research capabilities through the **UAgent Research Extension**, enabling the system to automatically trigger tree-based research processes for complex queries.

---

## 🏗️ Architecture Overview

### High-Level Structure

```
/home/wuy/AI/UAgent/
├── OpenHands/                          # Modified OpenHands fork (main codebase)
│   ├── openhands/                      # Core OpenHands platform
│   │   ├── server/                     # FastAPI backend server
│   │   ├── agenthub/                   # Agent implementations
│   │   ├── controller/                 # Agent control logic
│   │   ├── runtime/                    # Execution environments (Docker, etc.)
│   │   ├── llm/                        # LLM integration layer
│   │   ├── events/                     # Event system
│   │   └── storage/                    # Data persistence
│   │
│   ├── extensions/                     # Extension system
│   │   └── uagent_research/            # ⭐ UAgent Research Extension
│   │       ├── classifier/             # Task classification (triggers research)
│   │       ├── middleware/             # Request interception
│   │       ├── orchestrator/           # Tree search orchestrator (PUCT algorithm)
│   │       ├── adapters/               # Execution adapters (DeepResearch, RepoMaster, CodeAct)
│   │       ├── router/                 # Skill routing
│   │       ├── tools/                  # Research tools (search, browse, code analysis)
│   │       ├── api/                    # Research API endpoints
│   │       └── models/                 # Data models
│   │
│   ├── frontend/                       # React frontend
│   └── workspace/                      # User workspaces
│
├── original_openhands/                 # Original OpenHands (reference)
└── requirements.txt                    # Python dependencies
```

---

## 🎯 Core Components

### 1. OpenHands Platform (Base)

**Location**: `/home/wuy/AI/UAgent/OpenHands/openhands/`

OpenHands is an AI software engineer that can:
- Execute code in sandboxed environments
- Browse the web
- Interact with APIs
- Modify codebases
- Run commands

**Key Modules**:

- **Server** (`openhands/server/`): FastAPI application
  - `app.py`: Main FastAPI app
  - `session/session.py`: Manages agent sessions
  - `services/conversation_service.py`: Handles conversations
  - `routes/`: API endpoints

- **Agents** (`openhands/agenthub/`): Different agent implementations
  - CodeActAgent: Primary coding agent
  - BrowsingAgent: Web browsing
  - PlannerAgent: Task planning

- **Runtime** (`openhands/runtime/`): Execution environments
  - Docker-based sandboxes
  - File system access
  - Command execution

- **LLM** (`openhands/llm/`): Language model integration
  - Supports multiple providers (OpenAI, Anthropic, etc.)
  - Function calling
  - Streaming responses

---

### 2. UAgent Research Extension ⭐

**Location**: `/home/wuy/AI/UAgent/OpenHands/extensions/uagent_research/`

This is the **core innovation** of UAgent - it adds autonomous research capabilities.

#### 2.1 Task Classifier

**File**: `classifier/task_classifier.py`

**Purpose**: Determines if a user query should trigger research mode

**How it works**:
```python
# Analyzes user messages for:
- Research keywords: "research", "analyze", "investigate", "explore"
- Complexity indicators: "multiple", "various", "comprehensive"
- Multi-stage tasks: "modify ... and ...", "first ... then ..."

# Returns:
(should_trigger: bool, task_type: TaskType, confidence: float, reasoning: str)
```

**Example**:
```python
query = "Modify postgres and pg_duckdb source code to support vector search"

# Classification:
# - Has "modify" (action verb) ✓
# - Has "and" (multi-part task) ✓
# - Mentions multiple codebases ✓
# - Complex technical work ✓
# Result: TaskType.COMPLEX_RESEARCH, confidence=0.95
```

**Configuration**:
- Threshold: 0.7 (configurable via `RESEARCH_CONFIDENCE_THRESHOLD`)
- Can be enabled/disabled via `ENABLE_AUTO_RESEARCH_TRIGGER`

---

#### 2.2 Research Middleware

**File**: `middleware/research_middleware.py`

**Purpose**: Intercepts user messages and triggers research when appropriate

**Integration Points**:
1. **First message**: `openhands/server/services/conversation_service.py:165-206`
2. **Subsequent messages**: `openhands/server/session/session.py:367-398`

**Flow**:
```python
User sends message
    ↓
Middleware intercepts
    ↓
Calls task_classifier.should_trigger_research()
    ↓
If confidence >= threshold:
    ↓
    Generate experiment_id
    ↓
    Create TreeSearchOrchestrator
    ↓
    Start research in background (non-blocking)
    ↓
    Return metadata to user
    ↓
    Show system message: "Research mode activated"
```

**Key Features**:
- Non-blocking: Research runs in background
- Error handling: Falls back to normal conversation if research fails
- Metadata tracking: Experiment ID, confidence, task type

---

#### 2.3 Tree Search Orchestrator

**File**: `orchestrator/tree_orchestrator.py`

**Purpose**: Core research engine using PUCT algorithm (same as AlphaZero)

**Algorithm**: PUCT (Predictor + Upper Confidence bounds applied to Trees)

**PUCT Formula**:
```
PUCT(node) = Q(node) + c * P(node) * sqrt(N(parent)) / (1 + N(node))

Where:
- Q(node) = Average value (success rate, 0.0 to 1.0)
- P(node) = Prior probability (confidence, 0.0 to 1.0)
- N(parent) = Number of times parent was visited
- N(node) = Number of times node was visited
- c = Exploration constant (1.414 = sqrt(2))
```

**Main Loop**:
```python
while not budget_exceeded():
    # 1. SELECT: Choose best node using PUCT
    node = select_best_node(tree)
    
    # 2. EXPAND: Generate children
    children = expand_node(node)
    
    # 3. EXECUTE: Run children in parallel
    await execute_children_parallel(children)
    
    # 4. BACKPROPAGATE: Update values
    backpropagate_results(node, children)
    
    # 5. PUBLISH: Update API with tree state
    publish_tree_to_api(tree)
```

**Tree Structure**:
```
ROOT (user query)
├── IDEA 1 (approach 1)
│   ├── HYPOTHESIS 1.1
│   │   └── EXPERIMENT 1.1.1
│   └── HYPOTHESIS 1.2
├── IDEA 2 (approach 2)
│   └── HYPOTHESIS 2.1
└── IDEA 3 (approach 3)
```

**Node Types**:
- `ROOT`: User's original query
- `IDEA`: High-level approach
- `HYPOTHESIS`: Specific strategy
- `EXPERIMENT`: Concrete test/implementation

**Budget Control**:
- Max iterations: 50 (configurable)
- Max cost: $10.00 (configurable)
- Max parallel branches: 3 (configurable)

---

#### 2.4 Adapters

**Location**: `adapters/`

Adapters execute different types of research tasks.

**Available Adapters**:

1. **DeepResearchAdapter** (`adapters/deepresearch/`)
   - Web search and browsing
   - Uses Bing Search API
   - Extracts and summarizes content
   - Best for: Literature review, finding implementations

2. **RepoMasterAdapter** (`adapters/repomaster/`)
   - Code repository analysis
   - GitHub search
   - Code comprehension
   - Best for: Finding similar projects, understanding codebases

3. **CodeActAdapter** (`adapters/codeact/`)
   - Code execution
   - File manipulation
   - Command running
   - Best for: Implementation, testing, experiments

**Adapter Interface**:
```python
class BaseAdapter:
    async def run(self, task: Task, context: Context) -> AsyncIterator[Event]:
        """Execute task and yield events"""
        pass
```

---

#### 2.5 Skill Router

**File**: `router/skill_router.py`

**Purpose**: Routes tasks to appropriate adapters

**Routing Logic**:
```python
def route(task: Task, context: Context) -> str:
    # Analyze task content
    if "search web" in task.goal.lower():
        return "deepresearch"
    elif "analyze code" in task.goal.lower():
        return "repomaster"
    elif "implement" in task.goal.lower():
        return "codeact"
    else:
        return "deepresearch"  # default
```

---

#### 2.6 Research API

**File**: `api/research_routes.py`

**Endpoints**:

```python
# Get research tree state
GET /api/research/experiments/{experiment_id}/tree

# Get experiment metadata
GET /api/research/experiments/{experiment_id}

# List all experiments
GET /api/research/experiments

# Control research
POST /api/research/experiments/{experiment_id}/pause
POST /api/research/experiments/{experiment_id}/resume
POST /api/research/experiments/{experiment_id}/stop
```

**Tree State Format**:
```json
{
  "version": 1,
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
        "prior": 1.0,
        "cost": 0.0
      }
    ],
    "edges": [
      {"source": "root", "target": "idea-0"}
    ],
    "stats": {
      "total_nodes": 4,
      "total_edges": 3,
      "total_cost": 0.15,
      "max_depth": 1
    }
  }
}
```

---

#### 2.7 Tools

**Location**: `tools/`

Research tools used by adapters:

- **Search** (`tools/search/`):
  - Bing Search
  - Google Search
  - GitHub Search

- **Browse** (`tools/browse/`):
  - Web page fetching
  - Content extraction
  - Screenshot capture

- **Code** (`tools/code/`):
  - Code analysis
  - AST parsing
  - Dependency extraction

---

## 🔄 Complete Flow Example

### User Query: "Modify postgres and pg_duckdb to support vector search"

**Step 1: Classification**
```
User message → Task Classifier
→ Confidence: 0.95 (COMPLEX_RESEARCH)
→ Trigger research: YES
```

**Step 2: Middleware Intercepts**
```
Research Middleware
→ Generate experiment_id: "exp_6411470ef5854f71bfecdfd7b6689330_1759761226_4991e0f1"
→ Create TreeSearchOrchestrator
→ Start research (background)
→ Return to user: "Research mode activated"
```

**Step 3: Tree Initialization**
```
TreeSearchOrchestrator
→ Create ROOT node with user query
→ Tree: [ROOT]
```

**Step 4: Iteration 1 - Expand Root**
```
Select: ROOT (only option)
Expand: Generate 3 ideas
→ IDEA-0: "Web research: postgres vector search"
→ IDEA-1: "Web research: pg_duckdb vector"
→ IDEA-2: "Code research: GitHub implementations"

Tree: [ROOT] → [IDEA-0, IDEA-1, IDEA-2]
```

**Step 5: Execute Ideas in Parallel**
```
For each IDEA:
  1. Route to adapter (SkillRouter)
     → IDEA-0 → DeepResearchAdapter
     → IDEA-1 → DeepResearchAdapter
     → IDEA-2 → RepoMasterAdapter
  
  2. Adapter executes
     → DeepResearch: Bing search + browse top results
     → RepoMaster: GitHub search + analyze repos
  
  3. Update node
     → status: COMPLETE
     → avg_value: 0.8 (success)
     → cost: 0.05
```

**Step 6: Publish Tree**
```
TreeSearchOrchestrator
→ Serialize tree to JSON
→ POST to /api/research/experiments/{id}/tree
→ Frontend can now fetch and display
```

**Step 7: Iteration 2 - Select Best Idea**
```
Calculate PUCT scores:
→ IDEA-0: PUCT = 1.366
→ IDEA-1: PUCT = 1.366
→ IDEA-2: PUCT = 1.295

Select: IDEA-0 (highest, or random tie-break)
Expand: Generate 2 hypotheses
→ HYP-0: "Use pgvector extension"
→ HYP-1: "Implement custom vector type"

Tree: [ROOT] → [IDEA-0] → [HYP-0, HYP-1]
```

**Step 8: Continue Until Budget Exhausted**
```
Repeat:
  Select → Expand → Execute → Backpropagate → Publish

Stop when:
  - Max iterations reached (50)
  - Max cost exceeded ($10)
  - User stops research
  - All paths exhausted
```

**Step 9: Final Results**
```
TreeSearchOrchestrator
→ Collect all successful experiments
→ Synthesize findings
→ Return summary to user
```

---

## 📊 Data Models

### Research Tree Node

```python
class ResearchNode:
    id: str                    # Unique identifier
    type: NodeType             # ROOT, IDEA, HYPOTHESIS, EXPERIMENT
    title: str                 # Human-readable title
    content: str               # Detailed description
    status: NodeStatus         # PENDING, RUNNING, COMPLETE, FAILED
    parent_id: Optional[str]   # Parent node ID
    
    # PUCT algorithm fields
    visits: int                # Number of times visited
    avg_value: float           # Average success rate (0.0 to 1.0)
    prior: float               # Prior probability (0.0 to 1.0)
    
    # Metadata
    cost: float                # API costs incurred
    created_at: datetime
    updated_at: datetime
```

### Research Tree

```python
class ResearchTree:
    research_id: str           # Experiment ID
    nodes: Dict[str, ResearchNode]
    edges: List[Tuple[str, str]]  # (parent_id, child_id)
    stats: TreeStats
```

### Tree Stats

```python
class TreeStats:
    total_nodes: int
    total_edges: int
    total_cost: float
    max_depth: int
    completed_nodes: int
    failed_nodes: int
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Research auto-trigger
ENABLE_AUTO_RESEARCH_TRIGGER=true    # Enable/disable research
RESEARCH_CONFIDENCE_THRESHOLD=0.7    # Trigger threshold (0.0 to 1.0)

# Budget limits
RESEARCH_MAX_ITERATIONS=50           # Max research iterations
RESEARCH_MAX_COST=10.0              # Max cost in dollars
RESEARCH_MAX_PARALLEL=3             # Max concurrent branches

# LLM configuration
ANTHROPIC_API_KEY=sk-...            # For Claude
OPENAI_API_KEY=sk-...               # For GPT
```

### Config File

**Location**: `extensions/uagent_research/config.py`

```python
ENABLE_AUTO_RESEARCH_TRIGGER = os.getenv('ENABLE_AUTO_RESEARCH_TRIGGER', 'true').lower() == 'true'
RESEARCH_CONFIDENCE_THRESHOLD = float(os.getenv('RESEARCH_CONFIDENCE_THRESHOLD', '0.7'))
RESEARCH_MAX_ITERATIONS = int(os.getenv('RESEARCH_MAX_ITERATIONS', '50'))
RESEARCH_MAX_COST = float(os.getenv('RESEARCH_MAX_COST', '10.0'))
RESEARCH_MAX_PARALLEL = int(os.getenv('RESEARCH_MAX_PARALLEL', '3'))
```

---

## 🧪 Testing

### Test Structure

```
extensions/uagent_research/tests/
├── test_extension_loading.py       # Extension loading tests
├── test_integration_workflow.py    # End-to-end workflow tests
├── test_models.py                  # Data model tests
├── test_openhands_startup.py       # Server startup tests
├── test_server_integration.py      # API integration tests
└── test_websocket.py               # WebSocket tests
```

### Running Tests

```bash
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
pytest -v
```

### Coverage

```bash
pytest --cov=. --cov-report=html
# View: htmlcov/index.html
```

---

## 🚀 Deployment

### Local Development

```bash
cd /home/wuy/AI/UAgent/OpenHands

# Install dependencies
pip install -r requirements.txt
pip install -e extensions/uagent_research

# Start server
./start_openhands_research.sh
```

### Production

```bash
# Set environment variables
export ENABLE_AUTO_RESEARCH_TRIGGER=true
export RESEARCH_CONFIDENCE_THRESHOLD=0.7

# Start with Docker
docker-compose up -d
```

---

## 📝 Key Files Reference

### Core OpenHands

| File | Purpose |
|------|---------|
| `openhands/server/app.py` | Main FastAPI application |
| `openhands/server/session/session.py` | Agent session management |
| `openhands/server/services/conversation_service.py` | Conversation handling |
| `openhands/agenthub/codeact_agent/codeact_agent.py` | Primary coding agent |
| `openhands/llm/llm.py` | LLM integration |

### UAgent Research Extension

| File | Purpose |
|------|---------|
| `extensions/uagent_research/classifier/task_classifier.py` | Task classification |
| `extensions/uagent_research/middleware/research_middleware.py` | Request interception |
| `extensions/uagent_research/orchestrator/tree_orchestrator.py` | PUCT tree search |
| `extensions/uagent_research/router/skill_router.py` | Task routing |
| `extensions/uagent_research/adapters/deepresearch/adapter.py` | Web research |
| `extensions/uagent_research/adapters/repomaster/adapter.py` | Code research |
| `extensions/uagent_research/adapters/codeact/adapter.py` | Code execution |
| `extensions/uagent_research/api/research_routes.py` | Research API |

---

## 🔍 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Research Trigger

```bash
# Test classifier
python -c "
from extensions.uagent_research.classifier.task_classifier import task_classifier
result = task_classifier.should_trigger_research('your query here')
print(f'Should trigger: {result[0]}, Confidence: {result[2]:.2f}')
"
```

### Monitor Research Progress

```bash
# Watch tree updates
curl http://localhost:3000/api/research/experiments/{experiment_id}/tree

# Check logs
tail -f logs/openhands.log | grep -i research
```

---

## 📚 Documentation Files

Key documentation in the repository:

- `UAGENT_MECHANISM_EXPLAINED.md`: Detailed mechanism explanation
- `ENABLE_RESEARCH_GUIDE.md`: How to enable research
- `RESEARCH_CONFIG_SUMMARY.md`: Configuration guide
- `UAGENT_INTEGRATION_FLOW.md`: Integration flow
- `FIXES_APPLIED.md`: Bug fixes applied
- `QUICK_START.txt`: Quick start guide

---

## 🎓 Learning Path

To understand the codebase:

1. **Start with**: `UAGENT_MECHANISM_EXPLAINED.md` - High-level overview
2. **Read**: `extensions/uagent_research/README.md` - Extension overview
3. **Study**: `classifier/task_classifier.py` - How research is triggered
4. **Explore**: `orchestrator/tree_orchestrator.py` - Core algorithm
5. **Understand**: `adapters/` - How tasks are executed
6. **Review**: `api/research_routes.py` - API interface

---

## 🤝 Contributing

### Adding a New Adapter

1. Create adapter directory: `extensions/uagent_research/adapters/my_adapter/`
2. Implement `BaseAdapter` interface
3. Register in `adapters/__init__.py`
4. Update `router/skill_router.py` routing logic
5. Add tests

### Modifying PUCT Algorithm

Edit `orchestrator/tree_orchestrator.py`:
- `_calculate_puct_score()`: PUCT formula
- `_select_best_node()`: Node selection
- `_expand_node()`: Child generation
- `_execute_children_parallel()`: Parallel execution

---

## 📊 Performance Metrics

### Typical Research Session

- **Nodes created**: 10-50
- **Iterations**: 5-20
- **Duration**: 2-10 minutes
- **Cost**: $0.50-$5.00
- **Parallel branches**: 3

### Optimization Tips

1. **Reduce max_parallel**: Lower concurrent branches (saves cost)
2. **Increase threshold**: Higher confidence required (fewer triggers)
3. **Limit iterations**: Cap research depth
4. **Cache results**: Reuse previous research

---

## 🐛 Common Issues

### Research Not Triggering

**Check**:
1. `ENABLE_AUTO_RESEARCH_TRIGGER=true`
2. Query confidence >= threshold
3. Server restarted after config change
4. Middleware loaded successfully

### Frontend Stuck

**Solutions**:
1. Check server logs for errors
2. Verify research API responding
3. Disable research temporarily
4. Clear browser cache

### High Costs

**Solutions**:
1. Lower `RESEARCH_MAX_COST`
2. Reduce `RESEARCH_MAX_ITERATIONS`
3. Increase `RESEARCH_CONFIDENCE_THRESHOLD`
4. Use cheaper LLM models

---

## 🔗 External Dependencies

### Python Packages

- **FastAPI**: Web framework
- **SQLAlchemy**: Database ORM
- **LiteLLM**: Multi-provider LLM interface
- **Docker**: Container runtime
- **Playwright**: Browser automation
- **BeautifulSoup**: HTML parsing

### APIs

- **Bing Search API**: Web search
- **GitHub API**: Code search
- **Anthropic API**: Claude LLM
- **OpenAI API**: GPT LLM

---

## 📈 Future Enhancements

Planned features:

1. **Frontend Visualization**: React tree viewer
2. **Real-time Updates**: WebSocket streaming
3. **Result Synthesis**: Automatic summary generation
4. **Cost Optimization**: Smarter budget allocation
5. **Multi-modal Research**: Image/video analysis
6. **Collaborative Research**: Multi-agent coordination

---

## 📞 Support

- **Documentation**: See markdown files in repository
- **Issues**: Check `error.log` and server logs
- **Configuration**: Review `config.py` and environment variables
- **Testing**: Run test suite for validation

---

## 🎯 Summary

**UAgent** = **OpenHands** + **Autonomous Research Extension**

**Key Innovation**: Automatic tree-based research using PUCT algorithm

**Main Components**:
1. Task Classifier (triggers research)
2. Research Middleware (intercepts requests)
3. Tree Orchestrator (PUCT algorithm)
4. Adapters (execute tasks)
5. Research API (expose results)

**Use Cases**:
- Complex multi-step tasks
- Research and analysis
- Code exploration
- Systematic problem-solving

**Benefits**:
- Parallel exploration
- Automatic approach discovery
- Budget-controlled execution
- Real-time progress tracking

---

*Last Updated: 2025-01-06*
*Codebase Version: OpenHands 0.56.0 + UAgent Research Extension*