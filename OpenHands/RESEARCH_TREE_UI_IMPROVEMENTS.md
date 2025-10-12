# Research Tree UI Improvements

**Date:** 2025-10-12  
**Status:** ✅ Controls Removed | 🔄 Agent Status Investigation

---

## Fix #1: Removed Control Buttons from Research Tree Panel ✅

### Problem
The Research Tree tab had duplicate control buttons (Start, Pause, Cancel) that should only appear in the main OpenHands chat UI.

### Solution
Simplified the Research Tree panel to show only the tree visualization and stats.

### Files Modified
- `frontend/src/routes/research-tab.tsx`

### Changes Made

1. **Removed Components:**
   - `ExperimentStatus` component
   - `ExperimentControls` component  
   - `ExperimentProgress` component
   - `handleControlAction` function

2. **Simplified Layout:**
   - Replaced multiple grid sections with single stats bar
   - Shows: Research Tree title, connection status, and key metrics (Nodes, Edges, Cost, Tokens)
   - Maintains clean, minimal interface

3. **Removed Imports:**
   ```typescript
   // Removed:
   ExperimentControls,
   ExperimentProgress,
   ExperimentStatus,
   ```

### Result
- ✅ Clean UI with only tree visualization and essential stats
- ✅ Control buttons remain only in main chat UI (as intended)
- ✅ Reduced clutter and improved focus on tree view

---

## Issue #2: Agent Shows "STOPPED" Instead of "Awaiting User Input" 🔄

### Problem
After initialization completes, the agent shows status "STOPPED" instead of "Awaiting User Input".

### Investigation

#### Current Behavior
File: `openhands/server/conversation_manager/standalone_conversation_manager.py` (line 755-761)

```python
# If agent is in a terminal state (finished/stopped/error), mark conversation as stopped
if agent_state in [
    AgentState.FINISHED,
    AgentState.STOPPED,
    AgentState.REJECTED,
    AgentState.ERROR,
]:
    return ConversationStatus.STOPPED
```

#### Root Cause
The agent controller sets state to `AgentState.STOPPED` at the end of initialization instead of `AgentState.AWAITING_USER_INPUT`.

#### Expected Behavior
1. User creates new conversation → Agent initializes → Agent state should be `AWAITING_USER_INPUT`
2. User sends message → Agent state becomes `RUNNING`
3. Agent completes task → Agent state becomes `AWAITING_USER_INPUT` (if using `wait_for_response`)
4. User stops agent → Agent state becomes `STOPPED`

#### Key Code References

**When agent sets AWAITING_USER_INPUT:** (line 616 in agent_controller.py)
```python
# If the agent is waiting for a response, set the appropriate state
if action.wait_for_response:
    await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
```

**When agent sets STOPPED:** (line 276 in agent_controller.py)
```python
async def close(self, set_stop_state: bool = True) -> None:
    if set_stop_state:
        await self.set_agent_state_to(AgentState.STOPPED)
```

### Recommended Fix

The agent should be set to `AWAITING_USER_INPUT` after successful initialization, not `STOPPED`.

Possible locations for fix:
1. In initialization completion handler
2. When runtime becomes ready
3. After first system message is added

This requires deeper investigation of the initialization flow to find the exact point where the state should transition from `INIT`/`STARTING` to `AWAITING_USER_INPUT`.

---

## Testing

### Test Controls Removal ✅
```bash
# 1. Start server
cd /Users/wuy/Desktop/code/UAgent
./start_openhands_research.sh

# 2. Open browser
open http://localhost:2999/conversations/{conversation_id}

# 3. Click Research Tree tab
# Expected: See clean UI with only stats bar and tree view
# No Start/Pause/Cancel buttons should be visible
```

### Test Agent Status 🔄
```bash
# 1. Create new conversation
# 2. Wait for initialization to complete
# 3. Check status in UI
# Current: Shows "STOPPED"
# Expected: Should show "Awaiting User Input"
```

---

## Summary

✅ **Fixed:** Research Tree panel now has clean UI without control buttons  
🔄 **Investigating:** Agent status showing "STOPPED" instead of "Awaiting User Input" after initialization

The control buttons fix is complete and ready. The agent status issue requires further investigation into the initialization flow to determine the proper state transition point.
