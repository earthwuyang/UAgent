# Quick Reference: Parallel Tree Search Files

## 🎯 The Most Important Files

### 1. **THE CORE ENGINE** ⭐
```
extensions/uagent_research/orchestrator/tree_orchestrator.py
```
- **What**: Parallel tree search implementation (MCTS-style)
- **Key Method**: `run()` at line ~470
- **Does**: UCB selection, parallel expansion, agent simulation, backpropagation

### 2. **THE TRIGGER**
```
extensions/uagent_research/middleware/research_middleware.py
```
- **What**: Middleware that intercepts first messages
- **Key Method**: `start_research()` at line 278
- **Does**: Creates orchestrator, starts background task

### 3. **THE CLASSIFIER**
```
extensions/uagent_research/classifier/task_classifier.py
```
- **What**: LLM-based intelligent task classification
- **Key Method**: `should_trigger_research()` at line 195
- **Does**: Determines if message is research-worthy (your prompt: 0.95 confidence)

### 4. **THE ENTRY POINT**
```
openhands/server/services/conversation_service.py
```
- **What**: Handles new conversation creation
- **Key Function**: `create_conversation()` at line 141
- **Does**: Calls middleware to check for research mode

## 📊 Quick Flow Diagram

```
User Message (Frontend)
    ↓
conversation_service.py:141
    ↓
research_middleware.py:459 (process_message)
    ↓
task_classifier.py:195 (LLM classification)
    ↓  [if should_trigger == True]
    ↓
research_middleware.py:278 (start_research)
    ↓
tree_orchestrator.py:__init__ (create orchestrator)
    ↓
research_middleware.py:352 (_run_research_experiment)
    ↓
tree_orchestrator.py:470 (run - THE MAIN LOOP!)
    ↓
[Parallel expansion, agent execution, backpropagation]
    ↓
WebSocket updates to frontend
```

## 🔧 Key Line Numbers

| File | Function | Line | Purpose |
|------|----------|------|---------|
| `conversation_service.py` | `create_conversation` | 141 | Entry point |
| `research_middleware.py` | `process_message` | 459 | Check if research |
| `research_middleware.py` | `start_research` | 278 | Trigger research |
| `research_middleware.py` | `_run_research_experiment` | 352 | Run in background |
| `task_classifier.py` | `should_trigger_research` | 195 | LLM classification |
| `tree_orchestrator.py` | `__init__` | ~120 | Initialize tree |
| `tree_orchestrator.py` | `run` | ~470 | **MAIN LOOP** |
| `tree_orchestrator.py` | `_select_and_expand` | ~541 | UCB selection |
| `tree_orchestrator.py` | `_parallel_expand` | ~665 | Parallel hypotheses |
| `tree_orchestrator.py` | `_expand_node_with_simulation` | ~610 | Execute hypothesis |
| `tree_orchestrator.py` | `_backpropagate` | ~830 | Update tree |

## 🚀 Testing Commands

### Watch Logs
```bash
tmux attach -t uagent-backend
```

### Test Classifier
```bash
source /home/wuy/AI/UAgent/.venv/bin/activate
cd /home/wuy/AI/UAgent/OpenHands
python extensions/uagent_research/classifier/task_classifier.py
```

### Create New Conversation via API
```bash
curl -X POST http://localhost:3000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"agent": "CodeActAgent", "initial_user_message": "your research prompt"}'
```

## 📝 Log Messages to Watch For

```
# Classification
TaskClassifier initialized with model: dashscope/qwen3-coder-plus
LLM classified task as complex_research (confidence: 0.95)

# Trigger
🔬 Research mode triggered
Research mode triggered for conversation <id>

# Orchestrator
TreeSearchOrchestrator initialized for experiment <exp_id>
Starting tree search loop, max_iterations=50

# Execution
[TREE] Iteration 1: Selecting best node for expansion
[TREE] Selected node <node_id> with UCB score 0.85
[TREE] Expanding node <node_id> with 3 parallel hypotheses
[AGENT] Executing hypothesis <hyp_id>
[TREE] Backpropagating results: success=True, quality=0.92
```

## 📚 Documentation Files

- `PARALLEL_TREE_SEARCH_FLOW.md` - Complete detailed flow
- `LLM_CLASSIFIER_UPGRADE.md` - LLM classifier docs
- `RESEARCH_AUTO_TRIGGER_GUIDE.md` - Testing guide
- `HOW_TO_TEST_NEW_CONVERSATION.md` - New conversation instructions

---

**TL;DR**: Send your research prompt as the **first message** in a **NEW conversation**, and watch `tree_orchestrator.py:470` spring into action! 🌳
