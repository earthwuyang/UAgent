# Frontend Success Count Mismatch Analysis

## Problem Statement

**Observed Issue**: Frontend shows **(0/1) succeeded** even though:
- OpenHands execution completed successfully
- final.json was created with `"success": true`
- Experiment workspace is in `/arxiv/successful/` directory
- Logs show no agent errors or failures

## Evidence from Logs

### ✅ Experiment Completed Successfully

1. **final.json Created** (Line 6622-6623 in log):
```
[2025-10-02 14:16:40] File created successfully at: /workspace/experiments/exp_434c646a/results/final.json
```

2. **Content Shows Success**:
```json
{
    "success": true,
    "data": {...},
    "conclusions": ["Task completed successfully", "Files created as requested"],
    "errors": []
}
```

3. **Agent Never Entered FINISHED State**:
- Log shows: `AgentState.LOADING → AgentState.RUNNING`
- **MISSING**: `AgentState.RUNNING → AgentState.FINISHED`
- Agent remained in RUNNING state until timeout/session closure

4. **No Explicit Finish Action**:
```bash
grep -i "finish.*action\|AgentState.FINISHED" live_combined.log
# Result: No matches found
```

## Root Cause Analysis

### Problem: Agent Never Calls `finish` Action

**OpenHands Agent Behavior**:
1. Agent tool is `finish` - Available but **never called**
2. Agent continued executing even after creating final.json
3. Agent ran additional commands (creating more Python scripts)
4. No termination signal sent to UAgent backend

### Why Agent Didn't Call `finish`:

1. **Prompt Doesn't Explicitly Require It**:
   - Experiment prompt says: "Save to `final.json`"
   - Does NOT say: "Call `finish` action when done"
   - Agent assumes more work needed

2. **Agent Tried to Do More**:
   - After creating final.json, agent attempted to create additional implementation scripts
   - Continued working on "simplified" versions of the experiment
   - Never explicitly told to stop

3. **No Completion Criteria in Prompt**:
   - Missing: "After creating final.json, call the `finish` action with a summary"
   - Missing: "Your task is complete when final.json exists"

### UAgent Backend Parsing Logic

**How UAgent Determines Success**:

```python
# In backend/app/core/research_engines/scientific_research.py

# Method 1: Check if agent called finish action
if agent_finished:
    status = "completed"
else:
    status = "running" or "timeout"

# Method 2: Check exit code from OpenHands process
if exit_code == 0:
    status = "completed"
else:
    status = "failed"

# Method 3: Parse final.json (if exists)
if os.path.exists("final.json"):
    with open("final.json") as f:
        result = json.load(f)
        if result.get("success") == True:
            status = "completed"
```

**Current Behavior**:
- Agent never called `finish` → Method 1 fails
- Session timeout/closed without finish → exit code likely non-zero → Method 2 fails
- final.json exists and has `"success": true` → **Method 3 should work BUT may not be checked**

## Why Frontend Shows (0/1) Succeeded

### Hypothesis 1: UAgent Only Checks `finish` Action ❌ LIKELY

**Code Location**: `backend/app/core/research_engines/scientific_research.py`

```python
# Pseudo-code for current logic
def _parse_experiment_results(execution):
    if execution.agent_finished:  # Checks if agent called finish
        return ExperimentStatus.COMPLETED
    else:
        return ExperimentStatus.FAILED  # ← THIS IS HAPPENING
```

**Evidence**:
- Frontend counter shows 0/1 succeeded
- Experiment is archived to `successful/` directory (contradictory!)
- This suggests two different success checks:
  1. Archival logic checks final.json → sees success → moves to `/successful/`
  2. Frontend counter checks agent finish status → no finish call → counts as failed

### Hypothesis 2: final.json Path Mismatch ⚠️ POSSIBLE

**Expected Path**: `/workspace/results/final.json`
**Actual Path**: `/workspace/experiments/exp_434c646a/results/final.json`

UAgent parser may be looking in wrong directory:
```python
# Expected
final_json_path = f"/workspace/results/final.json"

# Actual location
final_json_path = f"/workspace/experiments/{exp_id}/results/final.json"
```

**Evidence**:
- Experiment prompt specifies nested path
- Agent created it in nested path
- Parser may not account for this

### Hypothesis 3: Parsing Happens Before final.json Created ⚠️ UNLIKELY

**Timeline**:
1. Agent creates final.json (14:16:40)
2. Agent continues working (14:17:17 - still running)
3. Session timeout/closed
4. UAgent parses results ← May have snapshot before final.json

**Evidence Against**: Logs show final.json was created, workspace archived after completion

## Solution Approaches

### ✅ Solution 1: Update Experiment Prompt to Require `finish` Action (RECOMMENDED)

**File**: `backend/app/core/research_engines/scientific_research.py`

**In `_build_comprehensive_experiment_prompt()` method, add**:

```python
# After the final.json requirement section

═══════════════════════════════════════════════════════════════
COMPLETION REQUIREMENT (MANDATORY)
═══════════════════════════════════════════════════════════════

After you have saved final.json and README.md, you MUST call the finish action:

<function=finish>
<parameter=outputs>
{
  "final.json": "Path to final.json file",
  "README.md": "Path to README.md file",
  "success": true/false,
  "summary": "Brief summary of experiment results"
}
</parameter>
</function>

IMPORTANT: Calling the finish action is REQUIRED to mark the experiment as complete.
Without calling finish, the experiment will be marked as FAILED regardless of final.json content.
```

**Expected Impact**:
- Agent will call `finish` after creating final.json
- UAgent backend will receive finish signal
- Frontend counter will correctly show (1/1) succeeded

### ✅ Solution 2: Update Result Parsing to Check final.json (FALLBACK)

**File**: `backend/app/core/research_engines/scientific_research.py`

**In result parsing logic**:

```python
def _parse_experiment_results(self, execution, workspace_path):
    """Parse experiment results from execution"""

    # Method 1: Check if agent called finish (current behavior)
    if execution.agent_finished:
        return ExperimentStatus.COMPLETED

    # Method 2: Check final.json even if agent didn't finish (NEW)
    final_json_paths = [
        f"{workspace_path}/results/final.json",
        f"{workspace_path}/experiments/*/results/final.json",  # Glob pattern
    ]

    for pattern in final_json_paths:
        matching_files = glob.glob(pattern)
        if matching_files:
            final_json_path = matching_files[0]
            try:
                with open(final_json_path) as f:
                    result = json.load(f)
                    if result.get("success") == True:
                        logger.info(f"Found final.json with success=true at {final_json_path}")
                        return ExperimentStatus.COMPLETED
            except Exception as e:
                logger.warning(f"Failed to parse final.json: {e}")

    # Method 3: If no finish and no valid final.json, mark as failed
    return ExperimentStatus.FAILED
```

**Expected Impact**:
- Even if agent doesn't call finish, success is detected from final.json
- More robust to agent behavior variations
- Frontend counter will correctly show successes

### ⚠️ Solution 3: Fix Path Mismatch in Parser

**If parser expects specific path**, update to check multiple locations:

```python
# OLD
final_json_path = f"{workspace}/results/final.json"

# NEW
final_json_paths = [
    f"{workspace}/results/final.json",
    f"{workspace}/experiments/*/results/final.json",
    f"{workspace}/**/final.json",  # Recursive search
]

for path_pattern in final_json_paths:
    files = glob.glob(path_pattern, recursive=True)
    if files:
        final_json_path = files[0]
        break
```

## Testing Strategy

### 1. Verify Current Parsing Logic

```bash
# Check what method is used to determine success
grep -n "ExperimentStatus.COMPLETED\|agent_finished\|final.json" \
  backend/app/core/research_engines/scientific_research.py
```

### 2. Add Debug Logging

```python
logger.info(f"Checking experiment success:")
logger.info(f"  - Agent finished: {execution.agent_finished}")
logger.info(f"  - final.json exists: {os.path.exists(final_json_path)}")
logger.info(f"  - final.json content: {final_json_content}")
logger.info(f"  - Determined status: {status}")
```

### 3. Test with Simple Experiment

```python
# Submit minimal experiment that creates final.json and calls finish
query = """
Create a file /workspace/results/final.json with {"success": true},
then call the finish action.
"""

# Verify frontend shows (1/1) succeeded
```

## Expected Behavior After Fix

### Before Fix:
- ❌ Agent creates final.json but doesn't call finish
- ❌ UAgent only checks agent_finished → False → marks as failed
- ❌ Frontend shows (0/1) succeeded
- ✅ Workspace archived to `/successful/` (uses different check)

### After Fix (Solution 1):
- ✅ Prompt explicitly requires calling finish
- ✅ Agent creates final.json AND calls finish
- ✅ UAgent checks agent_finished → True → marks as completed
- ✅ Frontend shows (1/1) succeeded

### After Fix (Solution 2):
- ⚠️ Agent creates final.json (may or may not call finish)
- ✅ UAgent checks final.json first, then agent_finished
- ✅ final.json has success=true → marks as completed
- ✅ Frontend shows (1/1) succeeded

## Recommended Implementation

**Use BOTH Solution 1 and Solution 2**:

1. **Solution 1** (prompt update):
   - Best practice - agent should signal completion
   - Makes behavior explicit and predictable
   - Works well when agent follows instructions

2. **Solution 2** (parsing fallback):
   - Defensive programming - handles cases where agent doesn't finish
   - More robust to agent behavior variations
   - Ensures success detection even if prompt is ignored

**Priority**: High - Affects user perception of experiment success rate

**Estimated Time**: 30-45 minutes for both solutions

## Files to Modify

1. `backend/app/core/research_engines/scientific_research.py`:
   - Update `_build_comprehensive_experiment_prompt()` to require finish action
   - Update result parsing logic to check final.json as fallback

2. Optional - Add integration test:
   - `test/integration/test_experiment_success_detection.py`
   - Test both finish action and final.json fallback paths

## References

- OpenHands agent actions: https://docs.all-hands.dev/modules/usage/agents
- finish action documentation: https://docs.all-hands.dev/modules/usage/agents#finish-action
- Experiment logs: `/uagent-workspace/arxiv/successful/20251002_134856_.../logs/`
