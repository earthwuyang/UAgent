# 🌳 Parallel Tree Search: Complete Flow from Frontend to Execution

## Overview

When you send a research-oriented message as the **first message** in a new conversation, the system automatically triggers parallel tree search. Here's the complete flow.

---

## The Files & Flow

```
Frontend (Browser)
    ↓ HTTP POST: /api/conversations (with initial_user_message)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 1. ENTRY POINT                                                  │
│ File: openhands/server/services/conversation_service.py         │
│ Function: create_conversation()                                 │
│ Line: 141                                                       │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CHECK FOR RESEARCH MODE                                      │
│ File: openhands/server/services/conversation_service.py         │
│ Lines: 168-170                                                  │
│                                                                 │
│ if RESEARCH_MIDDLEWARE_AVAILABLE and initial_user_msg:          │
│     result = await research_middleware.process_message(...)     │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. RESEARCH MIDDLEWARE - CLASSIFICATION                         │
│ File: extensions/uagent_research/middleware/research_middleware.py│
│ Function: process_message()                                     │
│ Lines: 459-570                                                  │
│                                                                 │
│ Key Logic:                                                      │
│   • Calls task_classifier.should_trigger_research()             │
│   • If should_trigger == True: starts research                  │
│   • Returns experiment_id                                       │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. LLM-BASED TASK CLASSIFIER                                    │
│ File: extensions/uagent_research/classifier/task_classifier.py  │
│ Function: should_trigger_research()                             │
│ Lines: 195-218                                                  │
│                                                                 │
│ What it does:                                                   │
│   • Sends message to LLM with classification prompt             │
│   • LLM analyzes: research indicators, complexity, multi-stage  │
│   • Returns: (should_trigger, task_type, confidence, reasoning) │
│                                                                 │
│ Your prompt result:                                             │
│   should_trigger = True                                         │
│   task_type = complex_research                                  │
│   confidence = 0.95                                             │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. START RESEARCH EXPERIMENT                                    │
│ File: extensions/uagent_research/middleware/research_middleware.py│
│ Function: start_research()                                      │
│ Lines: 278-350                                                  │
│                                                                 │
│ Creates:                                                        │
│   • Unique experiment_id                                        │
│   • TreeSearchOrchestrator instance                             │
│   • Background asyncio task                                     │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. TREE SEARCH ORCHESTRATOR INITIALIZATION                      │
│ File: extensions/uagent_research/orchestrator/tree_orchestrator.py│
│ Class: TreeSearchOrchestrator                                   │
│ Init: Lines ~120-250                                            │
│                                                                 │
│ Sets up:                                                        │
│   • Research tree structure                                     │
│   • Root node with goal                                         │
│   • Event bus for communication                                 │
│   • Parallel expansion workers                                  │
│   • UCB (Upper Confidence Bound) selection                      │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. BACKGROUND TASK RUNNER                                       │
│ File: extensions/uagent_research/middleware/research_middleware.py│
│ Function: _run_research_experiment()                            │
│ Lines: 352-403                                                  │
│                                                                 │
│ asyncio.create_task():                                          │
│   • Runs orchestrator.run() in background                       │
│   • Doesn't block main conversation thread                      │
│   • Publishes updates via WebSocket                             │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. PARALLEL TREE SEARCH EXECUTION (THE CORE!)                  │
│ File: extensions/uagent_research/orchestrator/tree_orchestrator.py│
│ Function: run()                                                 │
│ Lines: ~470-540                                                 │
│                                                                 │
│ Main Loop:                                                      │
│   while not done and iteration < max_iterations:                │
│     1. SELECT: Choose best node via UCB                         │
│     2. EXPAND: Generate multiple hypotheses in parallel         │
│     3. SIMULATE: Execute hypotheses via agent adapters          │
│     4. BACKPROPAGATE: Update tree with results                  │
│     5. PRUNE: Remove low-value branches                         │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. PARALLEL HYPOTHESIS GENERATION                               │
│ File: extensions/uagent_research/orchestrator/tree_orchestrator.py│
│ Function: _parallel_expand()                                    │
│ Lines: ~665-750                                                 │
│                                                                 │
│ Creates multiple hypotheses simultaneously:                     │
│   • Parallel LLM calls for idea generation                      │
│   • Each hypothesis = potential solution path                   │
│   • Creates child nodes in tree                                 │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 10. AGENT EXECUTION                                             │
│ File: extensions/uagent_research/adapters/codeact/adapter.py    │
│ Function: execute()                                             │
│                                                                 │
│ Each hypothesis is tested:                                      │
│   • Creates agent session (CodeActAgent)                        │
│   • Executes tasks in sandbox                                   │
│   • Collects results/observations                               │
│   • Returns success/failure + quality score                     │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ 11. REAL-TIME UPDATES TO FRONTEND                               │
│ File: extensions/uagent_research/orchestrator/ws_publisher.py   │
│ Function: publish_tree_update()                                 │
│                                                                 │
│ WebSocket messages sent to frontend:                            │
│   • Tree structure updates                                      │
│   • Node status changes (pending → running → complete)          │
│   • Cost tracking                                               │
│   • Results and observations                                    │
└─────────────────────────────────────────────────────────────────┘
    ↓
Frontend (Browser) - Research Tree UI updates in real-time
```

---

## Key Files Summary

### 1. Entry & Classification
- **`openhands/server/services/conversation_service.py`**
  - Entry point for new conversations
  - Calls research middleware if available

- **`extensions/uagent_research/middleware/research_middleware.py`**
  - Central orchestration logic
  - Manages experiment lifecycle
  - Lines 278-350: `start_research()` - THE TRIGGER POINT

- **`extensions/uagent_research/classifier/task_classifier.py`**
  - LLM-based intelligent classification
  - Determines if research mode should activate

### 2. Core Parallel Tree Search Engine
- **`extensions/uagent_research/orchestrator/tree_orchestrator.py`** ⭐ MOST IMPORTANT
  - **Line ~470**: `run()` - Main tree search loop
  - **Line ~541**: `_select_and_expand()` - UCB selection
  - **Line ~665**: `_parallel_expand()` - Parallel hypothesis generation
  - **Line ~610**: `_expand_node_with_simulation()` - Execute & simulate
  - **Line ~830**: `_backpropagate()` - Update tree with results

### 3. Agent Execution
- **`extensions/uagent_research/adapters/codeact/adapter.py`**
  - Executes individual hypotheses
  - Manages agent sessions
  - Collects results

- **`extensions/uagent_research/adapters/codeact/session_runner.py`**
  - Runs CodeActAgent in sandbox
  - Handles async agent execution

### 4. Communication
- **`extensions/uagent_research/orchestrator/event_bus.py`**
  - Event-driven communication
  - Coordinates between components

- **`extensions/uagent_research/orchestrator/ws_publisher.py`**
  - WebSocket updates to frontend
  - Real-time tree visualization

---

## The Most Important File

**`extensions/uagent_research/orchestrator/tree_orchestrator.py`**

This is THE CORE parallel tree search implementation. The key method is:

```python
async def run(self) -> Dict[str, Any]:
    """
    Main tree search loop.
    
    Implements:
    - UCB-based node selection
    - Parallel hypothesis expansion
    - Agent-based simulation
    - Result backpropagation
    - Dynamic pruning
    """
```

Located around **line 470**.

---

## How to Trace Execution

### 1. Enable Debug Logging
Add to your environment:
```bash
export LOG_LEVEL=DEBUG
```

### 2. Watch Server Logs
```bash
tmux attach -t uagent-backend
```

### 3. Look for These Log Messages

**Classification**:
```
TaskClassifier initialized with model: dashscope/qwen3-coder-plus
LLM classified task as complex_research (confidence: 0.95)
```

**Research Start**:
```
🔬 Research mode triggered
Research mode triggered for conversation <id>
[COORDINATOR] Middleware returned experiment_id: <exp_id>
```

**Tree Search Execution**:
```
TreeSearchOrchestrator initialized for experiment <exp_id>
Starting tree search loop, max_iterations=50
[TREE] Iteration 1: Selecting best node for expansion
[TREE] Selected node <node_id> with UCB score 0.85
[TREE] Expanding node <node_id> with 3 parallel hypotheses
[TREE] Hypothesis 1: <description>
[AGENT] Executing hypothesis <hyp_id>
[TREE] Backpropagating results: success=True, quality=0.92
```

---

## Testing the Flow

### Option 1: UI Test
1. Start NEW conversation
2. Send research prompt as first message
3. Watch logs in tmux
4. See Research Tree in UI

### Option 2: API Test
```bash
curl -X POST http://localhost:3000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "CodeActAgent",
    "initial_user_message": "research goal: your complex task here"
  }'
```

### Option 3: Direct Orchestrator Test
```bash
source /home/wuy/AI/UAgent/.venv/bin/activate
cd /home/wuy/AI/UAgent/OpenHands

python << 'PYEOF'
import asyncio
import sys
sys.path.insert(0, 'extensions/uagent_research')

from orchestrator.tree_orchestrator import TreeSearchOrchestrator

async def test():
    orchestrator = TreeSearchOrchestrator(
        goal="Research and implement ML query routing",
        research_id="test_001",
        session_id="test_session",
        max_iterations=5
    )
    
    result = await orchestrator.run()
    print(f"Result: {result}")

asyncio.run(test())
PYEOF
```

---

## Summary

**THE KEY FILE**: `extensions/uagent_research/orchestrator/tree_orchestrator.py`

**THE TRIGGER**: `extensions/uagent_research/middleware/research_middleware.py:278` (`start_research()`)

**THE FLOW**:
1. User sends message → `conversation_service.py`
2. Middleware checks → `research_middleware.py:process_message()`
3. LLM classifies → `task_classifier.py:should_trigger_research()`
4. If research → `research_middleware.py:start_research()`
5. Creates orchestrator → `tree_orchestrator.py:__init__()`
6. Runs in background → `tree_orchestrator.py:run()`
7. Parallel expansion → `tree_orchestrator.py:_parallel_expand()`
8. Agent execution → `codeact/adapter.py:execute()`
9. Updates frontend → WebSocket via `ws_publisher.py`

The entire system is event-driven and asynchronous, with the tree search running in parallel to the main conversation thread! 🚀
