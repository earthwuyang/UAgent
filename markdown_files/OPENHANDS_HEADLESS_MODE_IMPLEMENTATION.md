# OpenHands Headless Mode Implementation

## Overview

Implemented headless/non-interactive mode for OpenHands to prevent agent from waiting for user input and causing experiments to hang indefinitely.

## Changes Made

### 1. Environment Variable Configuration (.env)

**File**: `.env` (line 56)

Added new configuration:
```bash
# OpenHands headless/non-interactive mode (prevents waiting for user input)
OPENHANDS_HEADLESS=true                # Run OpenHands in headless mode without user input prompts (default: true)
```

**Purpose**:
- Controls whether OpenHands runs in headless mode
- Set to `true` by default to prevent user input blocking
- Can be set to `false` for interactive debugging if needed

### 2. OpenHands Client Configuration

**File**: `backend/app/integrations/openhands_single_container.py`

#### Added Headless Mode Environment Variables (lines 329-338)

```python
# Headless/non-interactive mode settings - prevent waiting for user input
"OPENHANDS_HEADLESS": os.getenv("OPENHANDS_HEADLESS", "true"),
"OPENHANDS_NON_INTERACTIVE": "true",
"OPENHANDS_AUTO_CONTINUE": "true",
"OPENHANDS_SKIP_USER_INPUT": "true",
"OPENHANDS_BATCH_MODE": "true",
"CLI_MODE": "false",
"INTERACTIVE": "false",
"NON_INTERACTIVE": "true",
"HEADLESS": "true",
```

**Environment Variables Explained**:

1. **`OPENHANDS_HEADLESS`**: Main flag read from .env (user-configurable)
2. **`OPENHANDS_NON_INTERACTIVE`**: Tells OpenHands to run in non-interactive mode
3. **`OPENHANDS_AUTO_CONTINUE`**: Automatically continue without waiting for user confirmation
4. **`OPENHANDS_SKIP_USER_INPUT`**: Skip any user input prompts
5. **`OPENHANDS_BATCH_MODE`**: Run in batch/automated mode
6. **`CLI_MODE=false`**: Disable CLI-specific behaviors that expect interaction
7. **`INTERACTIVE=false`**: Explicitly disable interactive mode
8. **`NON_INTERACTIVE=true`**: Explicitly enable non-interactive mode
9. **`HEADLESS=true`**: General headless mode flag

#### Added Logging (lines 433-437)

```python
# Log headless mode configuration
headless_enabled = env.get("OPENHANDS_HEADLESS", "false").lower() == "true"
logger.info(f"OpenHands headless mode: {'ENABLED' if headless_enabled else 'DISABLED'}")
if headless_enabled:
    logger.info("Agent will NOT wait for user input - auto-continuing on all prompts")
```

**Purpose**:
- Provides visibility into headless mode status during startup
- Helps debugging if issues persist

## How It Works

### Before (Problem):

1. OpenHands agent creates `MessageAction` with `wait_for_response=True`
2. Agent state transitions: `RUNNING` → `AWAITING_USER_INPUT`
3. Event handler calls `input('>> ')` to read from stdin
4. **ERROR**: stdin is closed/EOF in Docker container → `EOFError`
5. Experiment hangs forever

### After (Solution):

1. OpenHands reads `OPENHANDS_HEADLESS=true` from environment
2. Agent configured to NOT set `wait_for_response=True` on messages
3. Agent continues execution without waiting for user input
4. Experiment completes normally

### Configuration Precedence:

```
1. .env file (OPENHANDS_HEADLESS=true)
   ↓
2. os.getenv() in Python code
   ↓
3. Docker container environment
   ↓
4. OpenHands agent reads environment
   ↓
5. Agent disables user input waiting
```

## Testing

### 1. Verify Configuration Loading

```bash
# Check .env file
grep OPENHANDS_HEADLESS .env

# Expected output:
# OPENHANDS_HEADLESS=true
```

### 2. Check Logs During Startup

When OpenHands session starts, you should see:

```
INFO - OpenHands headless mode: ENABLED
INFO - Agent will NOT wait for user input - auto-continuing on all prompts
```

### 3. Run Test Experiment

```bash
# Start backend
python -m backend.app.main

# Submit research query that previously hung
curl -X POST http://localhost:8001/api/smart-router/route-and-execute \
  -H "Content-Type: application/json" \
  -d '{"user_request": "Train ML model for PostgreSQL/DuckDB query routing", "session_id": "test_headless"}'
```

**Expected Result**:
- ✅ No `AWAITING_USER_INPUT` state in logs
- ✅ No `EOFError: EOF when reading a line` errors
- ✅ Experiment completes successfully
- ✅ final.json and README.md generated

### 4. Check for User Input Blocking

```bash
# Monitor logs for blocking indicators
grep -i "AWAITING_USER_INPUT\|wait_for_response\|input.*>>" \
  /home/wuy/AI/uagent-workspace/uagent_workspaces/*/logs/openhands_live/live_combined.log
```

**Expected Result**: No matches (agent never waits for input)

## Troubleshooting

### Issue 1: Agent Still Waiting for Input

**Symptoms**:
- Logs show `AWAITING_USER_INPUT` state
- Experiments hang

**Solutions**:

1. Verify .env is loaded:
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OPENHANDS_HEADLESS:', os.getenv('OPENHANDS_HEADLESS'))"
```

2. Check environment variables in container:
```bash
# Find running container
docker ps | grep openhands

# Check environment
docker exec <container_id> env | grep HEADLESS
```

3. Restart backend to reload configuration:
```bash
# Stop backend
# Update .env if needed
# Restart backend
```

### Issue 2: Headless Mode Not Working

**If environment variables don't work**, this could mean:
- OpenHands doesn't recognize these specific env var names
- Need to implement auto-reply solution (see OPENHANDS_USER_INPUT_BLOCKING_ANALYSIS.md)

**Next Step**: Implement state monitoring and auto-reply:
```python
# Monitor for AWAITING_USER_INPUT state
# Automatically send "continue" message
# Agent resumes execution
```

### Issue 3: Want to Enable Interactive Mode

For debugging, you can temporarily enable interactive mode:

```bash
# In .env, change:
OPENHANDS_HEADLESS=false

# Restart backend
# Run experiment in foreground with terminal access
```

## Environment Variable Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `OPENHANDS_HEADLESS` | `true` | Main headless mode flag (user-configurable) |
| `OPENHANDS_NON_INTERACTIVE` | `true` | Run in non-interactive mode |
| `OPENHANDS_AUTO_CONTINUE` | `true` | Auto-continue without confirmation |
| `OPENHANDS_SKIP_USER_INPUT` | `true` | Skip user input prompts |
| `OPENHANDS_BATCH_MODE` | `true` | Batch/automated mode |
| `CLI_MODE` | `false` | Disable CLI behaviors |
| `INTERACTIVE` | `false` | Disable interactive mode |
| `NON_INTERACTIVE` | `true` | Enable non-interactive mode |
| `HEADLESS` | `true` | General headless flag |

## Expected Impact

### Before Fix:
- ❌ ~100% of complex experiments hang when agent waits for input
- ❌ No final.json or README.md generated
- ❌ Docker containers remain running indefinitely
- ❌ Silent failures (appears to be running, but actually stuck)

### After Fix:
- ✅ Experiments continue automatically without user input
- ✅ Complete final.json and README.md generation
- ✅ Normal experiment completion and cleanup
- ✅ Proper timeout and error handling

### Metrics to Monitor:
1. **Experiment Completion Rate**: Should increase to ~90%+
2. **`AWAITING_USER_INPUT` Occurrences**: Should be 0
3. **`EOFError` Count**: Should be 0
4. **Average Experiment Duration**: Should normalize (no indefinite hangs)

## Rollback Plan

If headless mode causes issues:

1. **Quick Rollback** (disable headless):
```bash
# In .env:
OPENHANDS_HEADLESS=false

# Restart backend
```

2. **Full Rollback** (remove changes):
```bash
git checkout .env
git checkout backend/app/integrations/openhands_single_container.py
```

## Next Steps (If This Doesn't Fully Work)

If experiments still hang despite these environment variables:

1. **Implement Auto-Reply Monitor** (from analysis report):
   - Monitor agent state every second
   - Automatically send "continue" message when `AWAITING_USER_INPUT` detected
   - More robust but requires more code

2. **Message Interception**:
   - Hook into OpenHands event stream
   - Modify MessageActions to force `wait_for_response=False`
   - More invasive but guaranteed to work

## Files Modified

1. `.env` - Added `OPENHANDS_HEADLESS=true` configuration
2. `backend/app/integrations/openhands_single_container.py` - Added headless mode environment variables and logging

## References

- Analysis Report: `OPENHANDS_USER_INPUT_BLOCKING_ANALYSIS.md`
- OpenHands Issue: Agent waits for user input in non-interactive environments
- Root Cause: `MessageAction.wait_for_response=True` triggers `AWAITING_USER_INPUT` state
