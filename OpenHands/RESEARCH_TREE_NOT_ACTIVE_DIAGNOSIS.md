# Research Tree Panel Not Active - Diagnosis

## Date: 2025-10-08 20:48 UTC
## Conversation: 78915da6a06647a2a06b9a58b1be10f0
## URL: http://120.46.207.248:3000/conversations/78915da6a06647a2a06b9a58b1be10f0

## Status: ⚠️ ISSUE IDENTIFIED

### Summary
The Research Tree panel exists in the UI but shows "No Active Research" because:
1. ✅ Research tab is present and accessible
2. ✅ Research API endpoints are working
3. ✅ Research experiment can be created
4. ❌ **Research orchestrator is NOT executing** (CRITICAL)

### Investigation Results

#### 1. API Check ✅
```bash
curl "http://120.46.207.248:3000/api/research/experiments/78915da6a06647a2a06b9a58b1be10f0/tree"
```
Response:
```json
{
    "version": 0,
    "timestamp": "2025-10-08T12:47:23.661352",
    "experiment_id": "78915da6a06647a2a06b9a58b1be10f0",
    "data": {
        "nodes": [],
        "edges": [],
        "stats": {}
    }
}
```
**Result**: API works but tree is empty

#### 2. Conversation Metadata ✅
```json
{
    "research_experiment_id": null,
    "research_goal": null,
    "research_locked": false
}
```
**Result**: No research experiment was linked to this conversation

#### 3. Experiment Creation ✅
```bash
curl -X POST "http://120.46.207.248:3000/api/research/experiments/start" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Research ML-based query routing for Postgres-DuckDB hybrid system",
    "session_id": "78915da6a06647a2a06b9a58b1be10f0",
    "research_type": "code"
  }'
```
Response:
```json
{
    "id": "exp_78915da6a06647a2a06b9a58b1be10f0_1759927722_3aca95b5",
    "session_id": "78915da6a06647a2a06b9a58b1be10f0",
    "experiment_type": "code",
    "goal": "Research ML-based query routing for Postgres-DuckDB hybrid system",
    "status": "pending",
    "progress_percentage": 0.0
}
```
**Result**: Experiment created successfully

#### 4. Backend Logs ❌ CRITICAL ISSUE
```
[DEBUG] Orchestrator not available, experiment exp_78915da6a06647a2a06b9a58b1be10f0_1759927722_3aca95b5 NOT executing
```

**ROOT CAUSE**: The research orchestrator is not being initialized or started!

### Why the Orchestrator Isn't Running

The experiment creation endpoint in `research_routes.py` creates the experiment but doesn't start execution. Let me check the code:

```python
@router.post("/experiments/start", response_model=ExperimentResponse)
async def start_experiment(
    request: StartResearchRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    # Creates experiment record
    experiment = ResearchSession(...)
    session.add(experiment)
    await session.commit()
    
    # Should trigger orchestrator but likely not working
    background_tasks.add_task(...)  # ← This might be failing
```

### Possible Causes

1. **BackgroundTasks not executing**
   - FastAPI BackgroundTasks might not be set up correctly
   - Background task might be failing silently

2. **Orchestrator not initialized**
   - TreeOrchestrator needs to be created and stored
   - `_active_orchestrators` dict might not be working

3. **Missing dependencies**
   - LLM configuration might be missing
   - Database connections might be failing

4. **Research middleware disabled**
   - The auto-trigger mechanism might not be enabled
   - Configuration flags might be off

### Next Steps to Fix

#### Option 1: Check Configuration
```bash
# Check if research auto-trigger is enabled
grep -r "ENABLE_AUTO_RESEARCH" /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research/.env
```

#### Option 2: Manually Trigger Orchestrator
The experiment was created but not started. Need to:
1. Find the actual execution trigger
2. Check if there's a separate "start" action needed
3. Verify background task execution

#### Option 3: Check Backend Logs for Errors
```bash
tmux capture-pane -t uagent-backend -p -S -1000 | grep -i "error\|exception\|fail"
```

### Frontend Behavior

The frontend research tab correctly:
1. ✅ Checks for `research_experiment_id` in conversation metadata
2. ✅ Falls back to using `conversation_id` as experiment ID
3. ✅ Polls `/api/research/experiments/{id}/tree` every 5 seconds
4. ✅ Shows "No Active Research" when tree is empty

The issue is **backend-side** - the orchestrator isn't populating the tree.

### Files Involved

**Backend:**
- `/extensions/uagent_research/uagent_research/api/research_routes.py` - API endpoints
- `/extensions/uagent_research/uagent_research/orchestrator/tree_orchestrator.py` - Orchestrator
- `/extensions/uagent_research/middleware/research_middleware.py` - Auto-trigger

**Frontend:**
- `/frontend/src/routes/research-tab.tsx` - Research tab UI
- `/frontend/src/state/research-tree-store.ts` - State management
- `/frontend/src/components/research/ResearchTreePanel.tsx` - Tree visualization

### Recommended Fix

The research orchestrator needs to be properly started. This likely requires:

1. **Check the start_experiment function** - verify background task execution
2. **Verify orchestrator initialization** - check if TreeOrchestrator is created
3. **Enable research middleware** - ensure auto-trigger is working
4. **Check LLM configuration** - orchestrator needs LLM to generate research plan

### Testing After Fix

1. Refresh the frontend page
2. Check Research Tree tab - should show nodes
3. Monitor backend logs for orchestrator activity
4. Verify tree API returns nodes

---
Generated: 2025-10-08 20:48 UTC
Issue: Research tree panel not active/empty
Status: Orchestrator not executing
