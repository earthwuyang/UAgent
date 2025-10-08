# UAgent Quick Reference Guide

## 🚀 Quick Start

### Start the Server
```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_openhands_research.sh
```

### Access the UI
```
http://localhost:3000
```

### Test Research Trigger
```bash
# Send a complex query like:
"Research neural architecture search methods and implement the best approach"
```

---

## 📁 Directory Structure (Quick Map)

```
/home/wuy/AI/UAgent/
├── OpenHands/                                    # Main codebase
│   ├── openhands/                                # Core platform
│   │   ├── server/                               # Backend (FastAPI)
│   │   │   ├── app.py                            # Main app
│   │   │   ├── session/session.py                # Session management
│   │   │   └── services/conversation_service.py  # Conversation handling
│   │   ├── agenthub/                             # Agents
│   │   ├── llm/                                  # LLM integration
│   │   └── runtime/                              # Execution environments
│   │
│   └── extensions/uagent_research/               # ⭐ Research extension
│       ├── classifier/task_classifier.py         # Triggers research
│       ├── middleware/research_middleware.py     # Intercepts requests
│       ├── orchestrator/tree_orchestrator.py     # PUCT algorithm
│       ├── adapters/                             # Task executors
│       │   ├── deepresearch/                     # Web research
│       │   ├── repomaster/                       # Code research
│       │   └── codeact/                          # Code execution
│       ├── router/skill_router.py                # Routes tasks
│       ├── api/research_routes.py                # Research API
│       └── tools/                                # Research tools
```

---

## 🔑 Key Concepts

### 1. Research Triggering
```
User Query → Task Classifier → Confidence Score
                                    ↓
                            If >= 0.7 (threshold)
                                    ↓
                            Research Mode Activated
```

### 2. PUCT Algorithm
```
PUCT(node) = Q(node) + c * P(node) * sqrt(N(parent)) / (1 + N(node))

Q = Quality (success rate)
P = Prior (confidence)
N = Visit count
c = Exploration constant (1.414)
```

### 3. Tree Structure
```
ROOT (user query)
├── IDEA (approach)
│   ├── HYPOTHESIS (strategy)
│   │   └── EXPERIMENT (test)
│   └── HYPOTHESIS
└── IDEA
```

### 4. Node Status Flow
```
PENDING → RUNNING → COMPLETE
                  ↘ FAILED
```

---

## 🛠️ Configuration

### Environment Variables
```bash
# Enable/disable research
export ENABLE_AUTO_RESEARCH_TRIGGER=true

# Confidence threshold (0.0 to 1.0)
export RESEARCH_CONFIDENCE_THRESHOLD=0.7

# Budget limits
export RESEARCH_MAX_ITERATIONS=50
export RESEARCH_MAX_COST=10.0
export RESEARCH_MAX_PARALLEL=3

# LLM keys
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
```

### Config File
```python
# Location: extensions/uagent_research/config.py

ENABLE_AUTO_RESEARCH_TRIGGER = True
RESEARCH_CONFIDENCE_THRESHOLD = 0.7
RESEARCH_MAX_ITERATIONS = 50
RESEARCH_MAX_COST = 10.0
RESEARCH_MAX_PARALLEL = 3
```

---

## 📡 API Endpoints

### Research Tree
```bash
# Get tree state
GET /api/research/experiments/{experiment_id}/tree

# Response:
{
  "version": 1,
  "experiment_id": "exp_...",
  "data": {
    "nodes": [...],
    "edges": [...],
    "stats": {...}
  }
}
```

### Experiment Control
```bash
# Pause research
POST /api/research/experiments/{experiment_id}/pause

# Resume research
POST /api/research/experiments/{experiment_id}/resume

# Stop research
POST /api/research/experiments/{experiment_id}/stop
```

### List Experiments
```bash
# Get all experiments
GET /api/research/experiments

# Response:
{
  "experiments": [
    {
      "id": "exp_...",
      "status": "running",
      "created_at": "2025-01-06T10:00:00"
    }
  ]
}
```

---

## 🔍 Debugging Commands

### Test Classifier
```bash
python -c "
from extensions.uagent_research.classifier.task_classifier import task_classifier
result = task_classifier.should_trigger_research('your query here')
print(f'Trigger: {result[0]}, Confidence: {result[2]:.2f}')
"
```

### Check Research Status
```bash
# View tree
curl http://localhost:3000/api/research/experiments/{exp_id}/tree | jq

# Watch logs
tail -f logs/openhands.log | grep -i research
```

### Toggle Research
```bash
# Enable
./toggle_research.sh enable

# Disable
./toggle_research.sh disable

# Check status
./toggle_research.sh status
```

---

## 🧪 Testing

### Run All Tests
```bash
cd extensions/uagent_research
pytest -v
```

### Run Specific Test
```bash
pytest tests/test_integration_workflow.py -v
```

### Coverage Report
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

---

## 📊 Monitoring

### Key Metrics
```python
# Tree stats
{
  "total_nodes": 15,
  "total_edges": 14,
  "total_cost": 2.50,
  "max_depth": 3,
  "completed_nodes": 12,
  "failed_nodes": 1
}
```

### Performance Indicators
- **Nodes/iteration**: 2-5 (normal)
- **Cost/node**: $0.05-$0.50 (typical)
- **Success rate**: 70-90% (good)
- **Depth**: 2-4 levels (optimal)

---

## 🐛 Troubleshooting

### Research Not Triggering

**Check 1**: Config enabled?
```bash
grep ENABLE_AUTO_RESEARCH_TRIGGER extensions/uagent_research/config.py
```

**Check 2**: Server restarted?
```bash
# Must restart after config changes
pkill -f openhands
./start_openhands_research.sh
```

**Check 3**: Query complex enough?
```bash
# Test with known complex query
"Research and implement multiple approaches to solve X"
```

### High Costs

**Solution 1**: Lower budget
```bash
export RESEARCH_MAX_COST=5.0
```

**Solution 2**: Reduce iterations
```bash
export RESEARCH_MAX_ITERATIONS=20
```

**Solution 3**: Increase threshold
```bash
export RESEARCH_CONFIDENCE_THRESHOLD=0.8
```

### Frontend Stuck

**Solution 1**: Check API
```bash
curl http://localhost:3000/api/research/experiments
```

**Solution 2**: Check logs
```bash
tail -f logs/openhands.log
```

**Solution 3**: Disable research
```bash
export ENABLE_AUTO_RESEARCH_TRIGGER=false
# Restart server
```

---

## 🎯 Common Tasks

### Add New Adapter

1. **Create directory**
```bash
mkdir -p extensions/uagent_research/adapters/my_adapter
```

2. **Implement adapter**
```python
# adapters/my_adapter/adapter.py
from ..base.adapter import BaseAdapter

class MyAdapter(BaseAdapter):
    async def run(self, task, context):
        # Your implementation
        yield Event(...)
```

3. **Register adapter**
```python
# adapters/__init__.py
from .my_adapter.adapter import MyAdapter

ADAPTER_REGISTRY = {
    'my_adapter': MyAdapter,
    # ...
}
```

4. **Update router**
```python
# router/skill_router.py
def route(task, context):
    if "my_task" in task.goal:
        return "my_adapter"
    # ...
```

### Modify PUCT Parameters

```python
# orchestrator/tree_orchestrator.py

# Change exploration constant
EXPLORATION_CONSTANT = 1.414  # Default (sqrt(2))
# Higher = more exploration
# Lower = more exploitation

# Change selection strategy
def _calculate_puct_score(self, node, parent):
    q = node.avg_value
    p = node.prior
    n_parent = parent.visits
    n_node = node.visits
    
    # Modify formula here
    u = EXPLORATION_CONSTANT * p * math.sqrt(n_parent) / (1 + n_node)
    return q + u
```

### Add Custom Tool

1. **Create tool**
```python
# tools/my_tool/my_tool.py
async def my_tool(param1, param2):
    # Tool implementation
    return result
```

2. **Register tool**
```python
# tools/__init__.py
from .my_tool.my_tool import my_tool

TOOL_REGISTRY = {
    'my_tool': my_tool,
    # ...
}
```

3. **Use in adapter**
```python
# adapters/my_adapter/adapter.py
from tools import my_tool

async def run(self, task, context):
    result = await my_tool(param1, param2)
    yield Event(data=result)
```

---

## 📝 Code Snippets

### Trigger Research Manually
```python
from extensions.uagent_research.orchestrator.tree_orchestrator import TreeSearchOrchestrator
from extensions.uagent_research.models import Budget

orchestrator = TreeSearchOrchestrator(
    max_parallel=3,
    budget=Budget(max_iterations=50, max_cost=10.0)
)

await orchestrator.run(
    goal="Your research goal",
    experiment_id="exp_123"
)
```

### Access Research Tree
```python
from extensions.uagent_research.api.research_routes import get_tree_state

tree_state = await get_tree_state(experiment_id="exp_123")
print(f"Nodes: {len(tree_state['data']['nodes'])}")
print(f"Cost: ${tree_state['data']['stats']['total_cost']:.2f}")
```

### Monitor Progress
```python
import asyncio
from extensions.uagent_research.orchestrator.event_bus import event_bus

async def monitor():
    async for event in event_bus.subscribe("research.*"):
        print(f"Event: {event.type}, Node: {event.node_id}")

asyncio.create_task(monitor())
```

---

## 🔗 Important Links

### Documentation
- Main overview: `CODEBASE_OVERVIEW.md`
- Mechanism explained: `UAGENT_MECHANISM_EXPLAINED.md`
- Enable guide: `ENABLE_RESEARCH_GUIDE.md`
- Integration flow: `UAGENT_INTEGRATION_FLOW.md`

### Key Files
- Task classifier: `extensions/uagent_research/classifier/task_classifier.py`
- Tree orchestrator: `extensions/uagent_research/orchestrator/tree_orchestrator.py`
- Research middleware: `extensions/uagent_research/middleware/research_middleware.py`
- Research API: `extensions/uagent_research/api/research_routes.py`

### External Resources
- OpenHands docs: https://docs.all-hands.dev
- PUCT algorithm: AlphaZero paper
- FastAPI docs: https://fastapi.tiangolo.com

---

## 💡 Tips & Tricks

### Optimize Performance
```bash
# Use faster LLM for ideas
export RESEARCH_IDEA_MODEL=gpt-4o-mini

# Use powerful LLM for experiments
export RESEARCH_EXPERIMENT_MODEL=claude-sonnet-4-5

# Cache results
export RESEARCH_ENABLE_CACHE=true
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG

# Enable research debug
export RESEARCH_DEBUG=true

# Save all trees
export RESEARCH_SAVE_TREES=true
```

### Cost Optimization
```bash
# Use cheaper models
export RESEARCH_DEFAULT_MODEL=gpt-4o-mini

# Limit parallel branches
export RESEARCH_MAX_PARALLEL=2

# Shorter iterations
export RESEARCH_MAX_ITERATIONS=30
```

---

## 📊 Cheat Sheet

### Node Types
| Type | Purpose | Parent | Children |
|------|---------|--------|----------|
| ROOT | User query | None | IDEAS |
| IDEA | Approach | ROOT | HYPOTHESES |
| HYPOTHESIS | Strategy | IDEA | EXPERIMENTS |
| EXPERIMENT | Test | HYPOTHESIS | None |

### Node Status
| Status | Meaning | Next |
|--------|---------|------|
| PENDING | Not started | RUNNING |
| RUNNING | In progress | COMPLETE/FAILED |
| COMPLETE | Success | - |
| FAILED | Error | - |

### Adapters
| Adapter | Use Case | Tools |
|---------|----------|-------|
| DeepResearch | Web search | Bing, Browse |
| RepoMaster | Code analysis | GitHub, AST |
| CodeAct | Execution | Shell, Files |

### API Status Codes
| Code | Meaning |
|------|---------|
| 200 | Success |
| 404 | Experiment not found |
| 500 | Server error |

---

## 🎓 Learning Resources

### Beginner
1. Read `UAGENT_MECHANISM_EXPLAINED.md`
2. Try simple queries
3. Watch tree grow in UI
4. Check API responses

### Intermediate
1. Study `tree_orchestrator.py`
2. Understand PUCT algorithm
3. Modify adapter behavior
4. Add custom tools

### Advanced
1. Implement new adapter
2. Modify PUCT formula
3. Add ML-based routing
4. Optimize performance

---

## 🚨 Emergency Commands

### Stop All Research
```bash
# Kill all research processes
pkill -f tree_orchestrator

# Disable auto-trigger
export ENABLE_AUTO_RESEARCH_TRIGGER=false

# Restart server
./start_openhands_research.sh
```

### Reset Database
```bash
# Backup first
cp openhands_research.db openhands_research.db.backup

# Delete database
rm openhands_research.db

# Restart (will recreate)
./start_openhands_research.sh
```

### Clear Cache
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Clear logs
rm -rf logs/*

# Restart
./start_openhands_research.sh
```

---

*Quick Reference v1.0 - Last Updated: 2025-01-06*