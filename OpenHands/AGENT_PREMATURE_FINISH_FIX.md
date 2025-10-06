# Agent Premature Finish Bug - Fix Documentation

**Date**: 2025-10-06
**Issue**: Agent enters FINISHED/STOPPED state after completing intermediate work instead of AWAITING_USER_INPUT
**Status**: ✅ **FIXED**

---

## 🐛 Problem Description

### User Report

> "After OpenHands finishes its first part of the job (e.g., implementing an ML-based query routing system), it comes into a stopped state, and I can no longer send further messages to it. However, it should be in 'waiting user input' state."

### Root Cause Analysis

The agent was **incorrectly interpreting "completed intermediate work" as "task complete"** and calling the `finish` tool, which:

1. Creates `AgentFinishAction`
2. Sets agent state to `FINISHED` (not `AWAITING_USER_INPUT`)
3. Prevents further user messages
4. Conversation appears "stuck" or "stopped"

---

## 🔍 Technical Investigation

### State Transition Flow (Before Fix)

```
Agent completes first part of work
  ↓
Agent thinks "task completed successfully"
  ↓
Agent calls FinishTool
  ↓
AgentFinishAction created
  ↓
agent_controller.py:514 → set_agent_state_to(AgentState.FINISHED)
  ↓
State = FINISHED ❌
  ↓
User cannot send more messages
```

### Key Files Involved

1. **`openhands/agenthub/codeact_agent/tools/finish.py`**
   - Defines `FinishTool` with description
   - Agent reads this to decide when to finish
   - **Original description was too permissive**

2. **`openhands/agenthub/codeact_agent/function_calling.py:157-160`**
   - Handles FinishTool call
   - Creates AgentFinishAction

3. **`openhands/controller/agent_controller.py:512-514`**
   - Handles AgentFinishAction
   - Sets state to FINISHED
   - **No check for interactive mode**

---

## ✅ Fixes Applied

### Fix #1: Updated FinishTool Description

**File**: `openhands/agenthub/codeact_agent/tools/finish.py`

**Before**:
```python
_FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information
```

**Problem**: Agent interprets "completed the user's requested task" as completing ANY part of the work.

**After**:
```python
_FINISH_DESCRIPTION = """Signals the completion of the ENTIRE task or conversation.

IMPORTANT: DO NOT use this tool after completing intermediate steps or partial work!

ONLY use this tool when:
- The user has EXPLICITLY confirmed they are satisfied and want to end the conversation
- The user says "done", "exit", "quit", or similar farewell
- You have completed ALL parts of a multi-stage task AND the user has no follow-up questions
- You cannot proceed further due to CRITICAL technical limitations (not just waiting for user input)

DO NOT use this tool when:
- You've just completed the first part of a multi-step task
- The user might want to continue, extend, or iterate on the work
- You're waiting for user confirmation or feedback
- You're presenting intermediate results

Instead, use MessageAction to:
- Present your results and ask if the user wants to continue
- Suggest next steps or improvements
- Wait for user confirmation before finishing
```

**Effect**: Agent will be much more conservative about finishing and will use MessageAction instead.

---

### Fix #2: Interactive Mode Safety Check

**File**: `openhands/controller/agent_controller.py:512-542`

**Before**:
```python
elif isinstance(action, AgentFinishAction):
    self.state.outputs = action.outputs
    await self.set_agent_state_to(AgentState.FINISHED)
```

**Problem**: Always sets state to FINISHED, even in interactive mode where user might want to continue.

**After**:
```python
elif isinstance(action, AgentFinishAction):
    self.state.outputs = action.outputs

    # In interactive mode, don't actually finish - wait for user confirmation
    if not self.headless_mode:
        self.log(
            'info',
            'Agent attempted to finish, but in interactive mode - awaiting user confirmation instead',
            extra={'msg_type': 'FINISH_INTERCEPTED'},
        )
        # Set state to awaiting user input instead of finished
        await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)

        # Add a message observation to inform the agent
        self.event_stream.add_event(
            AgentStateChangedObservation(
                content=(
                    "State changed to AWAITING_USER_INPUT. "
                    "The conversation will continue when the user sends their next message. "
                    "Do not finish the task unless the user explicitly confirms they are done."
                ),
                agent_state=AgentState.AWAITING_USER_INPUT,
            ),
            EventSource.AGENT,
        )
    else:
        # In headless mode, respect the finish action
        await self.set_agent_state_to(AgentState.FINISHED)
```

**Effect**:
- **Interactive mode**: Intercepts AgentFinishAction and sets state to AWAITING_USER_INPUT
- **Headless mode**: Respects the finish action (maintains backward compatibility)
- Agent receives feedback that state changed to AWAITING_USER_INPUT

---

## 🎯 Expected Behavior After Fix

### State Transition Flow (After Fix)

```
Agent completes first part of work
  ↓
Agent reads new FinishTool description
  ↓
Agent decides: "This is intermediate work, NOT complete task"
  ↓
Agent uses MessageAction instead of FinishTool
  ↓
Presents results: "I've completed X. Would you like me to continue with Y?"
  ↓
State = AWAITING_USER_INPUT ✅
  ↓
User can send more messages
```

### If Agent Still Calls Finish (Safety Net)

```
Agent calls FinishTool (rare)
  ↓
AgentFinishAction created
  ↓
agent_controller.py:517 checks headless_mode
  ↓
headless_mode = False (interactive)
  ↓
State = AWAITING_USER_INPUT (intercepted) ✅
  ↓
Agent receives AgentStateChangedObservation
  ↓
Agent learns: "Don't finish unless user confirms"
  ↓
User can continue conversation
```

---

## 📊 Impact

### Before Fix

| Scenario | Agent Behavior | User Experience |
|----------|----------------|-----------------|
| Complete intermediate work | Calls finish tool | ❌ Stuck, can't continue |
| Multi-step task | Finishes after step 1 | ❌ Can't do step 2 |
| Waiting for feedback | Calls finish tool | ❌ Conversation ends |

### After Fix

| Scenario | Agent Behavior | User Experience |
|----------|----------------|-----------------|
| Complete intermediate work | Uses MessageAction | ✅ Can continue |
| Multi-step task | Awaits user input | ✅ Can request step 2 |
| Waiting for feedback | Stays AWAITING_USER_INPUT | ✅ Can provide feedback |
| User says "done" | Calls finish tool | ✅ Properly ends |

---

## 🧪 Testing

### Test Case 1: Multi-Step Task

**User**: "First, implement feature X. Then, add tests. Finally, deploy."

**Expected Behavior**:
1. Agent implements feature X
2. Agent sends MessageAction: "Feature X implemented. Ready for tests?"
3. State = AWAITING_USER_INPUT ✅
4. User: "Yes, continue"
5. Agent implements tests
6. Agent sends MessageAction: "Tests added. Ready to deploy?"
7. State = AWAITING_USER_INPUT ✅
8. User: "Yes"
9. Agent deploys
10. Agent asks: "All done. Anything else?"
11. User: "No, we're done"
12. Agent calls finish tool (or user types /exit)
13. State = FINISHED ✅

### Test Case 2: Intermediate Results

**User**: "Research ML routing and implement it"

**Expected Behavior**:
1. Agent implements ML routing (first part)
2. Agent sends MessageAction: "Implemented core ML routing. I can also add baseline methods and run experiments. Continue?"
3. State = AWAITING_USER_INPUT ✅
4. User can respond with more requests

### Test Case 3: User Satisfaction Check

**User**: "Fix the bug"

**Expected Behavior**:
1. Agent fixes bug
2. Agent sends MessageAction: "Bug fixed. Would you like me to add tests or make any other changes?"
3. State = AWAITING_USER_INPUT ✅
4. User: "No, looks good"
5. Agent: "Great! Let me know if you need anything else."
6. State = AWAITING_USER_INPUT ✅ (stays awaiting unless user says "exit" or agent calls finish)

---

## 🔧 Configuration

### Headless vs Interactive Mode

The fix respects the execution mode:

- **Interactive Mode** (`headless_mode=False`):
  - Default for web UI
  - AgentFinishAction intercepted → AWAITING_USER_INPUT
  - Prevents premature conversation ending

- **Headless Mode** (`headless_mode=True`):
  - Used for batch processing, CI/CD
  - AgentFinishAction respected → FINISHED
  - Agent can finish when task complete

### Mode Detection

```python
# In agent_controller.py
if not self.headless_mode:
    # Interactive: intercept finish
    await self.set_agent_state_to(AgentState.AWAITING_USER_INPUT)
else:
    # Headless: allow finish
    await self.set_agent_state_to(AgentState.FINISHED)
```

---

## 🎓 Agent Learning

The fix includes an `AgentStateChangedObservation` that teaches the agent:

```
"State changed to AWAITING_USER_INPUT. The conversation will continue when
the user sends their next message. Do not finish the task unless the user
explicitly confirms they are done."
```

This observation:
- Goes into the agent's message history
- Helps the agent learn not to call finish prematurely
- Provides context for future decisions

---

## 📁 Files Modified

1. **`openhands/agenthub/codeact_agent/tools/finish.py`** (lines 5-30)
   - Updated `_FINISH_DESCRIPTION` with stricter criteria
   - Added explicit DO NOT conditions
   - Added alternatives (use MessageAction instead)

2. **`openhands/controller/agent_controller.py`** (lines 512-542)
   - Added headless_mode check
   - Intercepts AgentFinishAction in interactive mode
   - Sets state to AWAITING_USER_INPUT
   - Adds AgentStateChangedObservation for agent feedback

---

## ✅ Verification

### Check 1: Tool Description Updated

```bash
grep -A 20 "_FINISH_DESCRIPTION" ./openhands/agenthub/codeact_agent/tools/finish.py
```

Should show updated description with "DO NOT use this tool after completing intermediate steps"

### Check 2: Interactive Mode Check Added

```bash
grep -A 20 "headless_mode" ./openhands/controller/agent_controller.py | grep -A 10 "AgentFinishAction"
```

Should show the interceptor logic

### Check 3: Test in UI

1. Send a multi-step task
2. After agent completes first step, check state
3. Should be `AWAITING_USER_INPUT` not `FINISHED`
4. Should be able to send follow-up messages

---

## 🚀 Summary

**Problem**: Agent prematurely finished after intermediate work

**Root Cause**:
1. FinishTool description too permissive
2. No safety check for interactive mode

**Solution**:
1. ✅ Updated FinishTool description with strict criteria
2. ✅ Added interactive mode interceptor
3. ✅ Agent stays in AWAITING_USER_INPUT for intermediate work
4. ✅ User can continue conversation naturally

**Impact**:
- ✅ Prevents premature conversation ending
- ✅ Maintains backward compatibility (headless mode unchanged)
- ✅ Improves user experience in web UI
- ✅ Agent learns to use MessageAction for intermediate results

**Restart Required**: Yes - changes to tool descriptions and controller logic require server restart

---

**The agent will now stay responsive and await user input after completing intermediate work!** 🎉
