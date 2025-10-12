# Research Tree Singleton Fix - Implementation Documentation

## Issue Summary
GitHub Issue #1: Research tree not displaying due to synchronization failure between session manager and research middleware.

**Root Cause**: Multiple ResearchSessionManager instances were being created instead of using a single global singleton, causing experiments registered in one instance to be invisible to other components (API, WebSocket, middleware).

## Solution Implemented

Added comprehensive logging throughout the singleton pattern implementation to:
1. Track singleton instance creation
2. Verify all components use the same singleton instance
3. Monitor experiment registration and visibility
4. Debug WebSocket connection issues

## Files Modified

### 1. `/extensions/uagent_research/services/research_session_manager.py`

**Changes**:
- Added instance ID logging in `__init__()` method
- Added debug logging in `get_global_session_manager()` fast path
- Enhanced singleton creation logging with instance ID and component availability
- Added detailed experiment tracking in `register()` method

**Expected Log Output**:
```
INFO: ResearchSessionManager initialized (instance ID: 140123456789)
INFO: ✅ Global ResearchSessionManager singleton created (instance ID: 140123456789)
INFO:    Event bus: available
INFO:    Control bus: available
INFO:    LLM: unavailable
```

### 2. `/extensions/uagent_research/middleware/research_middleware.py`

**Changes**:
- Enhanced `get_session_manager()` with instance ID and experiment count logging
- Added debug logging for cached session manager returns
- Enhanced registration verification with instance ID logging
- Added experiment status logging after successful registration

**Expected Log Output**:
```
INFO: ✅ Middleware using global ResearchSessionManager singleton (instance ID: 140123456789)
INFO:    Experiments in singleton: 0
INFO: ✅ Registered experiment: exp_abc123_1760245260_92577f
INFO:    Session manager instance ID: 140123456789
INFO:    Total experiments in session manager: 1
INFO:    Experiment status: running
INFO: ✅ Registration verified for exp_abc123_1760245260_92577f
INFO:    Session manager instance ID: 140123456789
INFO:    Total experiments in session manager: 1
INFO:    Experiment status: running
```

### 3. `/extensions/uagent_research/uagent_research/api/research_routes.py`

**Changes**:
- Added warning when control bus not available
- Enhanced `get_session_manager()` with instance ID and experiment list logging
- Added stack trace logging for errors

**Expected Log Output**:
```
INFO: ✅ API using global ResearchSessionManager singleton (instance ID: 140123456789)
INFO:    Total experiments in singleton: 1
INFO:    Active experiment IDs: ['exp_abc123_1760245260_92577f']
```

### 4. `/extensions/uagent_research/uagent_research/api/websocket_routes.py`

**Changes**:
- Added warning when control components not available
- Enhanced `get_session_manager()` with instance ID and experiment list logging
- Added debug logging for cached session manager
- Added stack trace logging for errors

**Expected Log Output**:
```
INFO: ✅ WebSocket using global ResearchSessionManager singleton (instance ID: 140123456789)
INFO:    Total experiments in singleton: 1
INFO:    Active experiment IDs: ['exp_abc123_1760245260_92577f']
```

## Verification Checklist

When the singleton is working correctly, you should see:

### ✅ On Server Startup
1. Single singleton creation log with instance ID
2. Component availability status (event bus, control bus, LLM)

### ✅ On Research Start
1. Middleware logs using global singleton with specific instance ID
2. Experiment registration with same instance ID
3. Registration verification confirming experiment in session manager
4. Total experiment count increases

### ✅ On API/WebSocket Access
1. API routes log using global singleton with SAME instance ID
2. WebSocket routes log using global singleton with SAME instance ID
3. All components show same experiment count
4. All components list same experiment IDs

### ❌ Signs of Problem (What to Look For)
1. Multiple "singleton created" messages with different instance IDs
2. Different instance IDs across components
3. API/WebSocket showing 0 experiments while middleware shows >0
4. Registration verification failures
5. Missing singleton confirmation logs

## Testing Instructions

### 1. Clear Python Cache
```bash
cd ~/Desktop/code/UAgent-claude/OpenHands
find . -type f -name '*.pyc' -delete
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
```

### 2. Start Server
```bash
source .venv/bin/activate
./start_openhands_research.sh
```

### 3. Monitor Logs
```bash
# In another terminal
tmux attach -t uagent-backend
# Press Ctrl+B then [ to enter scroll mode
# Search for "singleton" or "instance ID"
```

### 4. Start Research Experiment
Via UI or API:
```bash
curl -X POST http://localhost:2999/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Test singleton functionality",
    "session_id": "test-session-123",
    "research_type": "scientific"
  }'
```

### 5. Verify Singleton Usage
Check logs for:
- Same instance ID across all components
- Experiment count increases after registration
- API endpoints return experiment data
- WebSocket connections succeed

## Expected Success Criteria

1. **Single Instance**: Only ONE "singleton created" message appears
2. **Consistent ID**: All components log the SAME instance ID (e.g., 140123456789)
3. **Experiment Visibility**: After registration:
   - Middleware shows experiment count = 1
   - API shows same experiment in active list
   - WebSocket shows same experiment in active list
4. **No Errors**: No "registration verification failed" errors
5. **API Returns Data**: `/api/research/experiments/{id}/status` returns non-empty data
6. **WebSocket Connects**: No `ERR_CONNECTION_RESET` errors

## Debugging Tips

### If Singleton Logs Don't Appear
1. Check if files were properly saved and Python cache was cleared
2. Verify imports use `get_global_session_manager()` not direct instantiation
3. Check for circular import issues

### If Multiple Instance IDs Appear
1. Search codebase for direct `ResearchSessionManager()` calls:
   ```bash
   grep -rn "ResearchSessionManager()" extensions/uagent_research/ \
     | grep -v "def __init__" \
     | grep -v "class ResearchSessionManager"
   ```
2. Ensure all components import from correct module path

### If Registration Verification Fails
1. Check the instance ID in error message
2. Compare with singleton creation instance ID
3. If different, there's a second instance being created somewhere

## Next Steps

If this logging reveals the singleton is NOT being used:
1. The logs will show which component is creating a new instance
2. Track down where that component is instantiating directly
3. Update it to use `get_global_session_manager()`

If the singleton IS being used but WebSocket still fails:
1. Check WebSocket handler initialization errors
2. Review event bus wiring
3. Examine network/firewall issues

## Related Files

- Issue: https://github.com/earthwuyang/UAgent/issues/1
- Linear: UAG-18
- Modified files in this fix:
  - `services/research_session_manager.py`
  - `middleware/research_middleware.py`
  - `uagent_research/api/research_routes.py`
  - `uagent_research/api/websocket_routes.py`
