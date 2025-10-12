# Research Tree Not Showing - Root Cause Analysis

## Issue Description

When the research auto-trigger fires, the experiment is created and registered with the session manager, but the Research Tree tab in the UI shows:
- Connection: Disconnected
- Nodes: 0  
- Edges: 0
- Status: Idle
- "No research data available"

## Root Cause

**The experiment is NOT being saved to the database when auto-triggered from the title generation flow.**

### What Happens Currently

1. User sends "research goal: ..." message
2. Title generation code calls `research_middleware.process_message()`
3. Middleware creates experiment ID: `exp_{session_id}_{timestamp}_{uuid}`
4. Middleware creates `TreeSearchOrchestrator` 
5. Middleware registers with `ResearchSessionManager` (in-memory singleton)
6. Background task `_run_research()` is created
7. **PROBLEM**: Experiment is NOT saved to database

### Why the UI Shows "Disconnected"

The UI's Research Tree tab calls:
```
GET /api/research/experiments/{exp_id}/status
```

This endpoint (line 1199-1315 in research_routes.py):
1. **First** checks if experiment exists in DATABASE (lines 1265-1279)
2. If not found in DB → returns HTTP 404
3. Only checks session manager if DB record exists

Since auto-triggered experiments aren't in the database, the endpoint returns 404.

## Code Locations

### Where Auto-Trigger Happens
```
OpenHands/openhands/utils/conversation_summary.py:120-154
```

### Where Middleware Creates Experiment  
```
OpenHands/extensions/uagent_research/middleware/research_middleware.py:599-766
```

Key code at lines 634-756:
- Creates experiment_id
- Creates orchestrator
- Registers with session_manager
- **Does NOT create database record**

### Where API Checks Status
```
OpenHands/extensions/uagent_research/uagent_research/api/research_routes.py:1199-1315
```

Key code at lines 1265-1279:
```python
# Check experiment exists - try by ID first, then by session_id
result = await session.execute(
    sql_select(Experiment).where(Experiment.id == experiment_id)
)
experiment = result.scalar_one_or_none()

# If not found by ID, try by session_id  
if not experiment:
    result = await session.execute(
        sql_select(Experiment).where(Experiment.session_id == experiment_id)
        .order_by(Experiment.created_at.desc())
    )
    experiment = result.scalar_one_or_none()

if not experiment:
    raise HTTPException(404, f"Experiment {experiment_id} not found")
```

**The experiment doesn't exist in DB, so 404 is returned.**

## Solution Options

### Option 1: Make Middleware Save to Database (Recommended)

Modify `research_middleware.start_research()` to create database record:

```python
async def start_research(...):
    experiment_id = f"exp_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    
    # CREATE DATABASE RECORD HERE
    from ..uagent_research.models import Experiment, ExperimentStatus, ExperimentType
    from ..uagent_research.models.base import get_session as get_db_session
    
    async for db_session in get_db_session():
        new_experiment = Experiment(
            id=experiment_id,
            session_id=session_id,
            experiment_type=ExperimentType.SCIENTIFIC,
            goal=goal,
            status=ExperimentStatus.RUNNING,
            created_at=datetime.utcnow()
        )
        db_session.add(new_experiment)
        await db_session.commit()
        break
    
    # Then continue with orchestrator creation...
    orchestrator = TreeSearchOrchestrator(...)
    ...
```

### Option 2: Change API Endpoint Logic

Modify `/experiments/{experiment_id}/status` endpoint to check session manager FIRST:

```python
@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: str, session: AsyncSession = Depends(get_session)):
    # TRY SESSION MANAGER FIRST
    session_mgr = get_session_manager()
    if session_mgr:
        try:
            status = session_mgr.get_status(experiment_id)
            return status  # Return immediately if found
        except KeyError:
            pass  # Not in session manager, check database
    
    # THEN check database as fallback
    result = await session.execute(...)
    ...
```

## Why This Wasn't Caught Earlier

The normal research flow (via explicit API call to `/api/research/start`) DOES create database records (see research_routes.py:283-327). The auto-trigger path bypasses this.

## Recommendation

**Implement Option 1** - Make the middleware create database records.

This is the cleaner solution because:
1. All experiments should be persisted, regardless of how they're triggered
2. Maintains consistency between manual and auto-triggered experiments
3. Allows UI to query experiment status even if session manager crashes/restarts
4. Historical record keeping

The database record can be minimal initially and updated by the background task as the research progresses.

## Files to Modify

1. `OpenHands/extensions/uagent_research/middleware/research_middleware.py` (line ~635)
   - Add database record creation after experiment_id generation
   - Import necessary models (Experiment, ExperimentStatus, ExperimentType)
   - Use `get_session()` to get database session

2. Verify imports work across module boundaries
   - The middleware is in `extensions/uagent_research/middleware/`
   - The models are in `extensions/uagent_research/uagent_research/models/`
   - May need to adjust import paths

## Testing After Fix

1. Start OpenHands server
2. Send "research goal: test message"
3. Check database for experiment record:
   ```sql
   SELECT id, session_id, status FROM experiments WHERE session_id = '{conversation_id}';
   ```
4. Call status endpoint:
   ```bash
   curl http://localhost:2999/api/research/experiments/{exp_id}/status
   ```
5. Verify Research Tree tab shows "Connected" with nodes

## Current Workaround

None available. The research IS running in the background, but the UI cannot display it because the status endpoint returns 404.
