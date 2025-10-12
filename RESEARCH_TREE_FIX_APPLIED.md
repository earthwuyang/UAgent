# Research Tree Visibility Fix - APPLIED ✅

## Issue Summary
Research experiments created via auto-trigger were not visible in the UI's Research Tree tab because they were only registered in-memory with the ResearchSessionManager but not saved to the database. The UI status endpoint checks the database first and returns 404 if no record exists.

## Root Cause
The middleware's `start_research()` method at line 639 created experiment IDs and registered them with the session manager, but skipped database persistence. This caused:
- GET `/api/research/experiments/{exp_id}/status` → 404 Not Found
- UI Research Tree showing "Connection: Disconnected"
- Research actually running but invisible to users

## Solution Applied

### Changes Made to `/Users/wuy/Desktop/code/UAgent/OpenHands/extensions/uagent_research/middleware/research_middleware.py`

#### 1. Added Database Model Imports (after line 60)
```python
# Import database models for experiment persistence
try:
    from ..uagent_research.models import (
        Experiment as DBExperiment,
        ExperimentStatus as DBExperimentStatus,
        ExperimentType as DBExperimentType,
    )
    from ..uagent_research.models.base import get_session as get_db_session
    from datetime import datetime
    DATABASE_AVAILABLE = True
    logger.info("✅ Database models loaded for experiment persistence")
except ImportError as e:
    logger.warning(f"Database models not available: {e}")
    DATABASE_AVAILABLE = False
    DBExperiment = None
    DBExperimentStatus = None
    DBExperimentType = None
    get_db_session = None
```

#### 2. Added Database Record Creation (after line 639)
```python
# Create database record for UI visibility and persistence
if DATABASE_AVAILABLE:
    try:
        logger.info(f"💾 Creating database record for experiment {experiment_id}")
        async for db_session in get_db_session():
            # Map research_type string to ExperimentType enum
            exp_type_map = {
                'scientific': DBExperimentType.SCIENTIFIC,
                'code': DBExperimentType.CODE,
                'roma': DBExperimentType.ROMA,
            }
            exp_type = exp_type_map.get(research_type, DBExperimentType.SCIENTIFIC)
            
            # Create experiment record
            new_experiment = DBExperiment(
                id=experiment_id,
                session_id=session_id,
                experiment_type=exp_type,
                goal=goal,
                status=DBExperimentStatus.RUNNING,
                created_at=datetime.utcnow(),
            )
            db_session.add(new_experiment)
            await db_session.commit()
            logger.info(f"✅ Database record created successfully for {experiment_id}")
            break  # Only need one iteration
    except Exception as e:
        # Don't fail research if database creation fails
        logger.error(f"❌ Failed to create database record for {experiment_id}: {e}")
        logger.info("ℹ️  Research will continue with in-memory tracking only")
else:
    logger.debug(f"Database not available - experiment {experiment_id} uses in-memory tracking")
```

## Key Design Decisions

### 1. Graceful Degradation
- If database imports fail → Set `DATABASE_AVAILABLE = False`
- If database creation fails → Log error but continue research
- Research works even without database (in-memory only mode)

### 2. Timing
- Database record created IMMEDIATELY after `experiment_id` generation
- Created BEFORE orchestrator initialization
- Ensures UI can see experiment as soon as it's created

### 3. Error Handling
- Try-except around database operations
- Comprehensive logging with emoji indicators:
  - 💾 = Starting database operation
  - ✅ = Success
  - ❌ = Failure
  - ℹ️ = Informational message

### 4. Type Mapping
Maps research_type strings to database enum:
- 'scientific' → DBExperimentType.SCIENTIFIC
- 'code' → DBExperimentType.CODE
- 'roma' → DBExperimentType.ROMA
- Default → DBExperimentType.SCIENTIFIC

## Expected Behavior After Fix

### Before Fix ❌
1. User sends "research goal: test message"
2. Middleware creates experiment in-memory only
3. UI calls `/api/research/experiments/{exp_id}/status`
4. Endpoint returns 404 (not in database)
5. Research Tree shows "Disconnected"

### After Fix ✅
1. User sends "research goal: test message"
2. Middleware creates experiment in database AND memory
3. UI calls `/api/research/experiments/{exp_id}/status`
4. Endpoint returns 200 OK with status data
5. Research Tree shows "Connected" with live updates

## Testing Checklist

- [ ] Start OpenHands server
- [ ] Send "research goal: test message" in chat
- [ ] Check server logs for "✅ Database record created successfully"
- [ ] Query database: `SELECT * FROM experiments WHERE session_id = '{conversation_id}'`
- [ ] Call API: `curl http://localhost:2999/api/research/experiments/{exp_id}/status`
- [ ] Verify Research Tree tab shows "Connected"
- [ ] Verify nodes appear in the tree visualization
- [ ] Check that research actually executes in background

## Verification Steps

### 1. Check Server Logs
Look for these messages:
```
✅ Database models loaded for experiment persistence
💾 Creating database record for experiment exp_...
✅ Database record created successfully for exp_...
```

### 2. Query Database
```sql
SELECT id, session_id, experiment_type, status, created_at 
FROM experiments 
WHERE session_id = 'eae55e0cc19f4e4a90300177621c09be'
ORDER BY created_at DESC;
```

### 3. Test API Endpoint
```bash
curl -s http://localhost:2999/api/research/experiments/exp_eae55e0cc19f4e4a90300177621c09be_1760254224_0f56cc/status | jq
```

Should return:
```json
{
  "experiment_id": "exp_...",
  "status": "running",
  "stats": {
    "total_nodes": ...,
    "completed": ...,
    ...
  }
}
```

### 4. Check UI
1. Navigate to conversation with research goal
2. Click "Research Tree" tab
3. Verify shows:
   - Connection: **Connected** (not Disconnected)
   - Nodes: > 0
   - Live tree visualization

## Rollback Plan

If issues arise, revert the changes:

```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands
git diff extensions/uagent_research/middleware/research_middleware.py
git checkout extensions/uagent_research/middleware/research_middleware.py
```

Then restart the server.

## Related Files

- **Modified**: `extensions/uagent_research/middleware/research_middleware.py`
- **Referenced**: `extensions/uagent_research/uagent_research/models/__init__.py`
- **Referenced**: `extensions/uagent_research/uagent_research/models/base.py`
- **Referenced**: `extensions/uagent_research/uagent_research/api/research_routes.py`

## Related Documentation

- `RESEARCH_AUTO_TRIGGER_SUCCESS.md` - Auto-trigger implementation details
- `RESEARCH_TREE_NOT_SHOWING_ROOT_CAUSE.md` - Original problem analysis

## Next Steps

1. Restart OpenHands server to load the changes
2. Test with a new conversation
3. Monitor logs for any database errors
4. Verify Research Tree shows connected status
5. If successful, commit the changes to version control

---

**Fix Applied**: 2025-01-12
**Modified By**: Automated analysis and implementation
**Status**: Ready for Testing
