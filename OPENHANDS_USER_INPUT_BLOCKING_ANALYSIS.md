# OpenHands User Input Blocking - Root Cause Analysis & Solution

## Executive Summary

OpenHands is pausing execution and waiting for user input, causing scientific experiments to hang indefinitely. The root cause is that OpenHands agent creates `MessageAction` with `wait_for_response=True`, causing state transition from `RUNNING` → `AWAITING_USER_INPUT`, which then tries to call `input('>> ')` in a non-interactive environment, resulting in EOFError.

## Evidence from Logs

### Critical Log Sequence

```
[2025-10-02 06:37:27] Setting agent(CodeActAgent) state from AgentState.RUNNING to AgentState.AWAITING_USER_INPUT

[2025-10-02 06:37:27] Error in event callback: EOF when reading a line
  File "/openhands/code/openhands/core/main.py", line 199, in on_event
    message = read_input(config.cli_multiline_input)
  File "/openhands/code/openhands/io/io.py", line 17, in read_input
    return input('>> ').rstrip()
EOFError: EOF when reading a line
```

### Trigger Pattern

The agent creates a MessageAction with:
```python
MessageAction(
    content='...<function=task_tracker>...',
    wait_for_response=True,  # ← THIS IS THE PROBLEM
    action=<ActionType.MESSAGE: 'message'>,
)
```

This happens when:
1. Agent uses certain function calls (like `task_tracker`)
2. Agent asks questions or seeks confirmation
3. Agent provides informational messages expecting acknowledgment

## Root Causes (Multi-Level)

### 1. **Agent Behavior Level**
- OpenHands CodeActAgent is designed for interactive CLI mode
- When agent outputs certain types of messages (especially with function calls), it sets `wait_for_response=True`
- This is hardcoded behavior in the agent's response parser

### 2. **Event Handling Level**
- When `AWAITING_USER_INPUT` state is set, event stream triggers callback
- Callback in `/openhands/code/openhands/core/main.py:199` calls `read_input()`
- `read_input()` uses Python's `input()` which blocks on stdin

### 3. **Environment Mismatch**
- UAgent runs OpenHands in Docker containers via subprocess
- No interactive terminal (stdin is closed/EOF)
- When `input()` is called, it immediately raises `EOFError`
- This crashes the event callback thread, but agent remains stuck in `AWAITING_USER_INPUT` state

### 4. **Configuration Gap**
We already set many auto-confirm flags:
```python
"OPENHANDS_SECURITY_CONFIRMATION": "false",
"OPENHANDS_AUTO_CONFIRM": "true",
"CONFIRM_MODE": "auto",
"AUTO_CONFIRM_ACTIONS": "true",
"OPENHANDS_DISABLE_SECURITY": "true",
```

**BUT** these only disable security confirmation prompts, NOT the agent's `wait_for_response` behavior on MessageActions.

## Why This is Critical

1. **Experiments Hang Indefinitely**: Once agent enters `AWAITING_USER_INPUT`, it never recovers
2. **Silent Failure**: UAgent thinks experiment is still running, but it's actually stuck
3. **Resource Waste**: Docker containers remain running, consuming resources
4. **Incomplete Results**: Experiments never generate final.json or README.md

## Solution Approaches (Ranked by Robustness)

### ✅ **Solution 1: Auto-Reply to User Input Requests (RECOMMENDED)**

**Strategy**: Intercept `AWAITING_USER_INPUT` state and automatically send confirmation messages.

**Implementation**:
1. Monitor OpenHands event stream for `AWAITING_USER_INPUT` state
2. When detected, automatically send a message action to continue execution
3. Use simple confirmations: "yes", "continue", "proceed", or empty message

**Pros**:
- Works with any future agent behavior changes
- Minimal code changes to UAgent
- Maintains compatibility with OpenHands updates
- Graceful fallback behavior

**Cons**:
- Requires event stream monitoring
- Small latency overhead

### ⚠️ **Solution 2: Modify Agent Configuration**

**Strategy**: Set configuration to run in non-interactive/headless mode.

**Implementation**:
Add environment variables:
```python
"OPENHANDS_HEADLESS": "true",
"OPENHANDS_NON_INTERACTIVE": "true",
"OPENHANDS_AUTO_CONTINUE": "true",
"OPENHANDS_SKIP_USER_INPUT": "true",
```

**Pros**:
- Simple configuration change
- No event monitoring needed

**Cons**:
- These flags may not exist (need to verify in OpenHands source)
- May not work if agent hardcodes `wait_for_response=True`

### ⚠️ **Solution 3: Message Interception & Modification**

**Strategy**: Intercept agent's MessageActions and force `wait_for_response=False`.

**Implementation**:
1. Hook into OpenHands event stream
2. When MessageAction is created, modify its `wait_for_response` attribute
3. Re-emit modified action

**Pros**:
- Guaranteed to work
- Prevents state transition entirely

**Cons**:
- Requires deep integration with OpenHands internals
- May break on OpenHands updates
- More complex implementation

### ❌ **Solution 4: Patch OpenHands Source Code** (NOT RECOMMENDED)

**Strategy**: Directly modify OpenHands agent code to disable `wait_for_response`.

**Pros**:
- Complete control

**Cons**:
- Requires maintaining fork
- Breaks on updates
- Not sustainable

## Recommended Implementation Plan

### Phase 1: Quick Fix (Solution 1 - Auto-Reply)

**File**: `backend/app/integrations/openhands_single_container.py`

**Add automatic state monitoring and response**:

```python
class OpenHandsV3Client:
    def __init__(self):
        # ... existing code ...
        self._auto_continue_enabled = True
        self._event_monitor_task = None

    async def _monitor_agent_state(self, session_id: str):
        """Monitor agent state and auto-continue when waiting for user input"""
        while self._auto_continue_enabled:
            try:
                # Check agent state via event stream
                state = await self._get_agent_state(session_id)

                if state == "awaiting_user_input":
                    logger.warning(
                        f"Agent {session_id} waiting for user input - auto-continuing"
                    )

                    # Send auto-continue message
                    await self._send_continue_message(session_id)

                await asyncio.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"Error in state monitor: {e}")
                await asyncio.sleep(5)

    async def _send_continue_message(self, session_id: str):
        """Send automatic continuation message to agent"""
        continue_action = {
            "action": "message",
            "args": {
                "content": "continue",  # Simple confirmation
                "wait_for_response": False
            }
        }

        # Send via OpenHands API
        await self._execute_action(session_id, continue_action)
```

**Start monitor when session begins**:
```python
async def ensure_session(self, ...):
    # ... create session ...

    # Start state monitor
    self._event_monitor_task = asyncio.create_task(
        self._monitor_agent_state(session_id)
    )

    return session_config
```

### Phase 2: Enhanced Solution (Add Configuration Flags)

Try additional environment variables (may or may not work):

```python
env = {
    # Existing flags...

    # NEW: Attempt to disable wait_for_response
    "OPENHANDS_HEADLESS": "true",
    "OPENHANDS_NON_INTERACTIVE": "true",
    "OPENHANDS_AUTO_CONTINUE": "true",
    "OPENHANDS_SKIP_USER_INPUT": "true",
    "OPENHANDS_BATCH_MODE": "true",
    "CLI_MODE": "false",  # Disable CLI-specific behaviors
}
```

### Phase 3: Monitoring & Metrics

Add tracking:
```python
# Count how often auto-continue is triggered
self.auto_continue_count = 0

# Log patterns that trigger waiting
def _log_wait_trigger(self, last_action):
    logger.info(f"Wait triggered by: {last_action.action_type}")
    # Identify patterns to potentially avoid
```

## Testing Strategy

### 1. Unit Test
```python
async def test_auto_continue_on_user_input():
    client = OpenHandsV3Client()
    session_id = await client.ensure_session(...)

    # Simulate agent entering AWAITING_USER_INPUT
    await client._inject_state_change(session_id, "awaiting_user_input")

    # Verify auto-continue is sent within 2 seconds
    await asyncio.sleep(2)
    assert client.auto_continue_count == 1
```

### 2. Integration Test
```python
async def test_experiment_completes_without_hanging():
    # Run a full experiment that previously hung
    result = await scientific_engine.conduct_research(
        "Train ML model for PostgreSQL/DuckDB routing"
    )

    # Verify completion
    assert result.status == "completed"
    assert "final.json" in result.output_files
```

### 3. Real-World Test
- Run the exact query that previously hung
- Monitor for `AWAITING_USER_INPUT` states
- Verify automatic continuation
- Verify experiment completes successfully

## Expected Outcomes

### After Fix:
1. ✅ Experiments continue automatically when agent waits for input
2. ✅ No more hanging/stuck experiments
3. ✅ Complete final.json and README.md generation
4. ✅ Proper experiment completion with all results

### Metrics to Track:
- Number of auto-continues per experiment session
- Time spent in AWAITING_USER_INPUT state (should be <2 seconds)
- Experiment completion rate (should increase to ~100%)
- Average experiment duration (should decrease)

## Alternative: Message Streaming Integration

If auto-reply doesn't work, we can intercept at the streaming level:

```python
# In streaming integration
def _process_event(self, event):
    if event.type == "agent_state_changed":
        if event.data.get("state") == "awaiting_user_input":
            # Force continue
            self._force_continue()

    # Normal processing
    return event
```

## Conclusion

**Root Cause**: OpenHands agent's `wait_for_response=True` on MessageActions causes `AWAITING_USER_INPUT` state, which tries to read stdin in non-interactive environment, causing EOFError.

**Recommended Fix**: Implement automatic state monitoring and send continuation messages when `AWAITING_USER_INPUT` is detected.

**Priority**: **CRITICAL** - Blocks all scientific experiments from completing.

**Estimated Implementation Time**: 2-3 hours for Phase 1 (auto-reply solution)

**Risk Level**: Low - solution is isolated and reversible
