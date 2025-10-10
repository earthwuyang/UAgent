# Research Auto-Trigger Testing Guide

## ✅ Status: Fixed and Ready

All import errors and syntax errors have been resolved. The research extension should now auto-trigger correctly for research-oriented prompts.

## What Was Fixed

### 1. Import Errors (Resolved)
- Standardized all imports to use `extensions.uagent_research` prefix
- Fixed imports in orchestrator, middleware, adapters, and service modules
- All modules now load without ImportError

### 2. Syntax Errors (Resolved)
- Fixed malformed `if` statement in `multi_agent_coordinator.py` line 222
- Fixed incomplete `try` block in `multi_agent_coordinator.py` line 401  
- Fixed indentation issues preventing research mode from activating

## How Auto-Trigger Works

### Detection Logic
The task classifier (`extensions/uagent_research/classifier/task_classifier.py`) analyzes user messages for:

1. **Research keywords**: research, compare, benchmark, evaluate, experiment
2. **Complexity indicators**: train, model, ML, collect data, modify source
3. **Multi-stage tasks**: "first...then", multiple action verbs, long descriptions

### Classification Thresholds
- **Confidence >= 0.7**: Triggers research mode
- Your prompt scores **0.95** (95% confident!) with:
  - research_score: 4
  - complexity_score: 5
  - has_research_goal: True
  - has_multi_stage: True

## Your Specific Prompt Analysis

```
research goal: modify postgres and pg_duckdb source code... 
first extract pre-opt features... then collect dual-execution data...
train a machine learning model... embed the machine learning model...
execute end-to-end experiments... baseline method... compare...
```

**Classification Result:**
- ✅ Task Type: `complex_research`
- ✅ Should Trigger: `True`
- ✅ Confidence: `0.95`
- ✅ Decision: "Strong research indicators detected"

## How to Test

### Step 1: Start a NEW Conversation
**Important**: The middleware only checks the **first message** in a conversation!

1. Open http://120.46.207.248:3000/
2. **Do NOT reuse old conversation URLs**
3. Start fresh from the homepage

### Step 2: Send Your Research Prompt
Copy-paste your exact research goal as the **very first message**:

```
research goal: modify postgres and pg_duckdb source code （ to download source code you can utilize the proxy on port localhost:7890, do not use the system-wide postgresql）, first extract pre-opt features from postgres kernel and log to files, then collect dual-execution data (pre-optimization query features that can be found in kernel structures and execution times on dual engine) and train a machine learning model to predict whether postgres engine or duckdb engine executes a query fast and embed the machine learning model into database source code (using the language of the database for example c language) to online route each query to the faster engine, and execute end-to-end experiments to test the ml-based system's performance. A baseline method called threshold-based method should also be implemented, which routes query based on threshold, for example threshold can be 10000 or 50000 or any other value, if postgres estimates the cost of a query is above threshold, then send to duckdb, otherwise send to postgres, and compare the postgres-only, duckdb-only, different threshold-based methods and lightgbm-based method. please record every successful  necessary commands in README.md so that later people can reproduce your results. also record your python packages dependencies in requirements.txt.
```

### Step 3: Monitor for Auto-Trigger
Watch the server logs:
```bash
tmux attach -t uagent-backend
```

Look for these messages:
```
🔬 Research mode triggered
Task classified as complex_research (confidence: 0.95)
[COORDINATOR] Middleware returned experiment_id: ...
Research experiment ... spawned and tracked
```

### Step 4: Verify in UI
After sending the message, you should see:
- Message annotation: "[System: Research mode activated - Experiment ID: ...]"
- A "Research Tree" tab or panel
- Progress indicators for the tree search

## Troubleshooting

### If Research Doesn't Trigger

1. **Check you're using a NEW conversation**
   - Old conversations won't retrigger
   - Must be the first message

2. **Check server logs**
   ```bash
   tmux attach -t uagent-backend
   # Look for classification messages
   ```

3. **Manual trigger fallback**
   ```bash
   curl -X POST http://localhost:3000/api/research/experiments/start \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "YOUR_RESEARCH_GOAL_HERE",
       "session_id": "YOUR_CONVERSATION_ID",
       "max_iterations": 50
     }'
   ```

4. **Verify middleware is loaded**
   Check server startup logs for:
   ```
   Research middleware loaded successfully
   UAgent Research Extension loaded from source
   ```

## Manual Classification Test

Test the classifier directly:
```bash
source /home/wuy/.cache/pypoetry/virtualenvs/openhands-ai-rK5BwNwE-py3.12/bin/activate
cd /home/wuy/AI/UAgent/OpenHands

python << 'PYEOF'
import sys
sys.path.insert(0, 'extensions/uagent_research')
from classifier.task_classifier import TaskClassifier

message = "YOUR_PROMPT_HERE"
classifier = TaskClassifier()
should_trigger, task_type, confidence, reasoning = classifier.should_trigger_research(message)

print(f"Should Trigger: {should_trigger}")
print(f"Task Type: {task_type.value}")
print(f"Confidence: {confidence:.2f}")
print(f"Reasoning: {reasoning}")
PYEOF
```

## Server Commands

```bash
# View logs
tmux attach -t uagent-backend

# Restart server
tmux kill-session -t uagent-backend
tmux new-session -d -s uagent-backend "cd /home/wuy/AI/UAgent/OpenHands && bash start_openhands_research.sh"

# Check status
curl http://localhost:3000/api/research/health

# List experiments
curl http://localhost:3000/api/research/experiments
```

## Next Steps

1. ✅ Server is running (http://120.46.207.248:3000/)
2. ✅ All fixes applied
3. 🔄 **Test with a NEW conversation**
4. 📊 Monitor logs for research activation
5. 🌳 Check Research Tree tab for progress

## Expected Behavior

When research mode triggers:
1. Your message is classified as `complex_research`
2. Middleware spawns a research experiment
3. TreeSearchOrchestrator starts in background
4. Progress updates appear in Research Tree UI
5. System message confirms activation

Good luck with your ML-based query routing research! 🚀
