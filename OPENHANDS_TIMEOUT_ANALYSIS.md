# OpenHands Timeout and Retry Analysis

## Problem Statement

Many OpenHands experiments fail "mysteriously" without proper retry. Analysis of failed experiments shows:

**Issue**: Per-action timeouts (300s) cause commands to fail, but OpenHands doesn't properly retry these operations, leading to incomplete experiments.

## Root Cause Analysis

### 1. Per-Action Timeout (300 seconds)

**File**: `backend/app/integrations/openhands_single_container.py` (Line 448)
```toml
[sandbox]
timeout = 300
```

**Behavior**:
- Each bash command times out after 300 seconds (5 minutes)
- Git clone of large repositories (PostgreSQL, DuckDB) often exceeds this
- Compilation tasks can exceed this limit

**Example from logs**:
```
[2025-10-02 15:48:45] Command "git clone https://github.com/postgres/postgres.git" timed out after 300.0 seconds
[2025-10-02 15:48:45] exit_code: -1
[2025-10-02 15:48:52] Agent tries to send C-c (interrupt)
[2025-10-02 15:48:52] ERROR: CLIRuntime does not support interactive input
```

### 2. OpenHands Retry Behavior

When a command times out:
1. **Timeout observation returned** with exit_code=-1
2. **Agent tries to recover** (e.g., send C-c, try alternative approach)
3. **CLIRuntime limitation**: Cannot send interrupts or signals to running processes
4. **Agent continues** but work is lost

**The agent DOES try alternatives** (e.g., wget instead of git clone), but this wastes time and may not work.

### 3. Total Experiment Timeout

**Configuration**:
- `max_minutes` = 9999999 minutes (essentially infinite)
- `max_iterations` = 999999999 (essentially infinite)

**Not the problem**: The total timeout is high enough.

### 4. Why Experiments Fail "Mysteriously"

**Symptom**: Exit code 0, but no final.json

**Analysis from logs**:
```json
{
  "status": "failed",
  "error_message": "Exit code 0, no final.json found",
  "duration_seconds": 486.909003
}
```

**Root Cause**:
1. Multiple actions timeout (git clone, compilation, etc.)
2. Agent tries alternatives but they also timeout or fail
3. **Agent gives up** after too many failures
4. OpenHands exits cleanly (code 0) without completing the task
5. **No final.json created** → Marked as failed
6. **UAgent retry doesn't trigger** because exit code is 0 (treated as "success")

## Solutions

### Option 1: Increase Per-Action Timeout (NOT RECOMMENDED)

**Rejected**: 300 seconds is reasonable for individual actions. If an action takes longer, it may indicate network issues or the task is too complex.

### Option 2: Better Prompt Engineering (RECOMMENDED)

**Strategy**: Modify experiment prompts to handle timeouts gracefully:

1. **Break down large downloads**:
   - Use shallow clones: `git clone --depth 1`
   - Download specific branches: `git clone --single-branch --branch stable`
   - Use tarballs instead of git clone for large repos

2. **Add timeout handling instructions**:
   ```
   If a command times out:
   1. Try a shallow clone: git clone --depth 1
   2. If that fails, download a tarball release instead
   3. If downloads fail, document the issue in final.json with success=false
   ```

3. **Add progress checkpoints**:
   ```
   After each major step, save intermediate results to final.json
   Even if the experiment fails, report what was completed
   ```

### Option 3: Implement Proper Retry in UAgent (RECOMMENDED)

**Current Issue**: Exit code 0 + no final.json is treated as "success"

**Fix**: Treat "no final.json" as failure regardless of exit code

**File**: `backend/app/integrations/openhands_single_container.py` (Line 827)

**Before**:
```python
error_message = None if success else f"Exit code {exit_code}" + ("" if final_json else ", no final.json found")
```

**Issue**: `success` is determined by exit code, not by final.json existence

**Solution**: Change success criteria

**Implementation needed**:
```python
# Success requires BOTH exit code 0 AND final.json with success=true
has_final_json = final_json is not None
final_json_success = final_json.get("success", False) if has_final_json else False
success = (exit_code == 0) and has_final_json and final_json_success

error_message = None if success else f"Exit code {exit_code}" + ("" if has_final_json else ", no final.json found")
```

This way, experiments without final.json will be marked as failed and trigger retries.

### Option 4: Use Proxy Configuration Proactively

**Issue**: Network timeouts due to slow connections

**Solution**: Ensure proxy is always configured if available

**Already implemented** in prompts:
```
PROXY CONFIGURATION:
- ALWAYS check if a proxy is available on localhost:7890
- Configure for ALL network operations (git, wget, curl, pip)
```

## Recommendations

### Immediate Actions

1. **Fix success detection** (Option 3)
   - Require final.json for success
   - This will trigger UAgent retries for incomplete experiments

2. **Improve prompts** (Option 2)
   - Add shallow clone instructions
   - Add timeout recovery strategies
   - Add progress checkpoint requirements

### Long-term Actions

1. **OpenHands Runtime Improvements**
   - Add support for sending signals to running processes (C-c, SIGTERM)
   - Better timeout handling with automatic retries
   - Configurable per-action timeouts based on action type

2. **Monitoring and Metrics**
   - Track timeout frequency per action type
   - Identify patterns in failed experiments
   - Adjust timeouts dynamically based on historical data

## Test Plan

1. **Baseline**: Run existing scientific research experiment
   - Expect some timeouts on git clone
   - Check if final.json is created

2. **After success detection fix**:
   - Verify failed experiments trigger retries
   - Verify final.json is required for success

3. **After prompt improvements**:
   - Verify shallow clones are used
   - Verify timeout recovery works
   - Verify progress checkpoints are saved

## Related Files

- `backend/app/integrations/openhands_single_container.py` - Container management
- `backend/app/core/research_engines/scientific_research.py` - Experiment execution
- `backend/app/core/experiment_manager.py` - Experiment lifecycle management

## Current Configuration

**Per-Action Timeout**: 300 seconds (5 minutes)
**Total Experiment Timeout**: 9999999 minutes (essentially infinite)
**Max Iterations**: 999999999 (essentially infinite)
**Retry Attempts** (UAgent level): 3 attempts per experiment

## Implementation Priority

High - Affects experiment success rate and requires minimal code changes
