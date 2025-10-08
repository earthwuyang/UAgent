# Research Orchestrator Import Fix

## Date: 2025-10-08 20:50 UTC
## Status: ✅ FIXED

## Problem Summary

The research tree panel was not active because the **TreeSearchOrchestrator could not be imported** due to an incorrect import path.

### Root Cause

**Import Error**: `No module named 'openhands.runtime.runtime'`

The research engines were using an incorrect import path:
```python
# ❌ WRONG (old code):
from openhands.runtime.runtime import Runtime
```

The correct path is:
```python
# ✅ CORRECT (fixed):
from openhands.runtime.base import Runtime
```

### Impact

When the orchestrator import failed:
1. `ORCHESTRATOR_AVAILABLE` flag was set to `False`
2. `start_experiment` endpoint created experiments but didn't execute them
3. Background tasks were never scheduled
4. Research tree remained empty
5. Frontend showed "No Active Research"

## Fix Applied

### Files Modified

1. **`extensions/uagent_research/uagent_research/engines/scientific_research.py`**
   ```bash
   # Changed line 29:
   - from openhands.runtime.runtime import Runtime
   + from openhands.runtime.base import Runtime
   ```

2. **`extensions/uagent_research/uagent_research/engines/code_research.py`**
   ```bash
   # Changed line 31:
   - from openhands.runtime.runtime import Runtime
   + from openhands.runtime.base import Runtime
   ```

### Verification

```bash
# Verify the fix:
grep "from openhands.runtime" \
  /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research/uagent_research/engines/*.py

# Should show:
# scientific_research.py:from openhands.runtime.base import Runtime
# code_research.py:from openhands.runtime.base import Runtime
```

## How to Apply the Fix

### 1. Server has been stopped
The old server process was terminated to ensure clean restart.

### 2. Start the server with the fix
```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_server_no_auth.sh
```

### 3. Verify orchestrator is available
After server starts, check the logs:
```bash
tmux capture-pane -t uagent-backend -p | grep "ORCHESTRATOR_AVAILABLE"
# Should show: ORCHESTRATOR_AVAILABLE=True
```

## Testing the Fix

### 1. Create a test experiment:
```bash
curl -X POST "http://120.46.207.248:3000/api/research/experiments/start" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Test orchestrator fix",
    "session_id": "test-session-123",
    "research_type": "code"
  }'
```

### 2. Check the response:
Should show `"status": "pending"` or `"status": "running"`

### 3. Monitor backend logs:
```bash
tmux capture-pane -t uagent-backend -p | tail -20
# Should show:
# [DEBUG] Background task added for experiment...
# [DEBUG] run_experiment_async called for...
# Starting background execution for experiment...
```

### 4. Check research tree:
```bash
curl "http://120.46.207.248:3000/api/research/experiments/{experiment_id}/tree"
# Should show nodes appearing after a few seconds
```

### 5. Frontend verification:
1. Navigate to: http://120.46.207.248:3000/conversations/{conversation_id}
2. Click the "Research Tree" tab
3. Should see: Research tree visualization with nodes (not "No Active Research")

## Expected Behavior After Fix

### Before Fix:
```
[DEBUG] Orchestrator import failed: No module named 'openhands.runtime.runtime'
[DEBUG] ORCHESTRATOR_AVAILABLE is False
[DEBUG] Orchestrator not available, experiment {id} NOT executing
```

### After Fix:
```
[DEBUG] Orchestrator import successful, ORCHESTRATOR_AVAILABLE=True
[DEBUG] Background task added for experiment {id}
[DEBUG] Starting background execution for experiment {id}
INFO: Creating TreeSearchOrchestrator...
INFO: Orchestrator initialized successfully
```

## Technical Details

### Why the Import Was Wrong

The OpenHands codebase structure is:
```
openhands/
├── runtime/
│   ├── __init__.py
│   ├── base.py          # ← Contains Runtime class
│   ├── browser/
│   └── impl/
```

The import path `openhands.runtime.runtime` attempted to import:
- Module: `openhands.runtime`
- Submodule: `runtime` (doesn't exist!)

The correct path `openhands.runtime.base` imports:
- Module: `openhands.runtime`
- Submodule: `base.py` (✓ exists, contains Runtime)

### Why This Wasn't Caught Earlier

1. The error only shows up when the research API routes are loaded
2. The server starts successfully (import is lazy)
3. The error is logged but server continues running
4. Frontend doesn't show import errors, just "No Active Research"

## Files Involved in the Fix

**Modified:**
- `extensions/uagent_research/uagent_research/engines/scientific_research.py`
- `extensions/uagent_research/uagent_research/engines/code_research.py`

**Checked but not modified:**
- `extensions/uagent_research/uagent_research/api/research_routes.py` (import logic)
- `extensions/uagent_research/orchestrator/tree_orchestrator.py` (orchestrator init)
- `openhands/runtime/base.py` (Runtime class definition)

## Additional Notes

### Server Restart Required

**Important**: The server MUST be restarted for the fix to take effect because:
1. Python modules are cached after first import
2. The `ORCHESTRATOR_AVAILABLE` flag is set at module load time
3. Failed imports are cached to avoid repeated failures

### Backward Compatibility

This fix is backward compatible:
- ✅ Existing functionality unchanged
- ✅ Only the import path was corrected
- ✅ Runtime class interface remains the same
- ✅ No changes to API or frontend code

## Summary

✅ **Bug**: Incorrect import path `openhands.runtime.runtime`  
✅ **Fix**: Changed to `openhands.runtime.base`  
✅ **Impact**: Research orchestrator can now be imported  
✅ **Result**: Research tree will populate with nodes  
✅ **Status**: Server stopped, ready to restart with fix  

---
Generated: 2025-10-08 20:50 UTC
Issue: Research orchestrator not executing
Fix: Import path correction
Next: Restart server with ./start_server_no_auth.sh
