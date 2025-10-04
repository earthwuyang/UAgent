# UAgent Experiment Completion Improvements

## Problem Statement

The previous experiment (`20251004_011810`) for ML-based query routing between PostgreSQL and DuckDB was marked as "successful" but actually:
- Only collected **simulated/fake data** using `random.uniform()`
- **Never trained** an actual ML model
- **Never embedded** the ML model into C/C++ source code
- **Never implemented** the online routing mechanism
- **Never conducted** real end-to-end performance testing

The OpenHands agent finished prematurely with placeholder results instead of completing all required steps.

## Root Cause

1. **Weak validation criteria** - The system accepted experiments with "simulated" data
2. **Insufficient retry attempts** - Only 2-3 attempts before giving up
3. **Lack of keyword detection** - No programmatic checks for simulation indicators
4. **No ML-specific validation** - Didn't verify model training/deployment occurred

## Changes Made

### 1. Enhanced Anti-Simulation Requirements (Lines 2678-2708)

**File**: `backend/app/core/research_engines/scientific_research.py`

**Changes**:
- Added **CRITICAL** warning header emphasizing failure consequences
- Explicit ⛔ **FORBIDDEN** list including:
  - Simulated/mock/fake/placeholder data
  - `random.uniform()` or synthetic data generation
  - Finishing before all steps are complete
- Mandatory ✅ **REQUIREMENTS** checklist:
  - Real data collection from actual execution
  - Actual ML model training (not just planning)
  - Model embedding in C/C++ source code
  - Online routing implementation
  - End-to-end performance testing
- Added guidance: **REPORT blockers instead of simulating**
- Listed 5 completion criteria that MUST be met

### 2. Stricter LLM Success Validator (Lines 3886-3955)

**File**: `backend/app/core/research_engines/scientific_research.py`

**Changes**:
- Added **programmatic keyword detection** before LLM check:
  - Detects: "simulated", "simulation", "sample data", "demonstration purposes", "placeholder", "mock", "fake", "random.uniform", "not performed", etc.
  - Returns `False` immediately if found
- Added **limitations section scanner**:
  - Checks for red flags like "data was simulated"
  - Rejects if limitations indicate incomplete work
- Added **conclusions validator**:
  - Detects "not performed" or "not trained" in conclusions
  - Fails experiment if ML training wasn't done
- Added **artifact verification**:
  - Checks `modifications_made` for ML model embedding evidence
  - Checks for routing mechanism implementation
  - For ML experiments, **requires** both model embedding AND routing
- Enhanced **LLM prompt** with:
  - ❌ RED FLAGS section (12 specific rejection criteria)
  - ✅ ACCEPTANCE criteria (6 mandatory requirements)
  - Instruction to be "VERY strict - when in doubt, mark false"
  - Increased reason length to 80 words for detailed feedback

### 3. Increased Retry Attempts (Lines 1972, 3147)

**File**: `backend/app/core/research_engines/scientific_research.py`

**Before**:
- `max_resume_attempts`: 3
- `max_attempts_per_experiment`: 2

**After**:
- `max_resume_attempts`: **5** (67% increase)
- `max_attempts_per_experiment`: **5** (150% increase)

**Rationale**: Complex experiments like ML model deployment need more attempts to:
- Download and build source code
- Collect real performance data
- Train ML models
- Embed models and implement routing
- Run comprehensive end-to-end tests

### 4. Context-Aware Validation (Lines 2153-2155, 3889)

**File**: `backend/app/core/research_engines/scientific_research.py`

**Changes**:
- Pass `experiment_goal` to `_assess_final_json_success()`
- Enables goal-aware validation (future enhancement point)
- Validates results match original requirements

## How the System Now Works

### Experiment Execution Flow

```
1. Agent receives experiment with ML deployment requirements
2. Agent attempts to complete work
3. Agent generates final.json
   ↓
4. PROGRAMMATIC VALIDATION (NEW!)
   - Scan for simulation keywords → REJECT if found
   - Check limitations for red flags → REJECT if found
   - Verify ML model embedding → REJECT if missing
   - Verify routing implementation → REJECT if missing
   ↓
5. LLM VALIDATION (ENHANCED!)
   - Strict 12-point RED FLAG check
   - Require ALL 6 acceptance criteria
   - When in doubt → REJECT
   ↓
6. IF REJECTED:
   - Extract specific blockers from limitations/errors
   - Create resume feedback
   - Add to prior_errors list
   - RETRY (up to 5 attempts now)
   - Pass feedback to agent: "Resolve these blockers before finishing"
   ↓
7. IF ACCEPTED:
   - Mark experiment as successful
   - Archive results
```

### Retry Mechanism

When an experiment fails validation:

```python
resume_feedback = self._summarize_resume_feedback(
    data_result,  # Contains the rejected results
    llm_reason     # Specific reason from LLM validator
)
# e.g., "LLM: ML model was not actually trained - limitations mention 'simulation'"

updated_prior_errors.append(resume_feedback)

# Next attempt receives:
container_goal += f"\n\nPrevious errors to fix:\n{json.dumps(prior_errors[-3:], indent=2)}"
```

The agent sees **exactly what was wrong** and gets multiple chances (5 instead of 2) to fix it.

## Expected Impact

### Before (Old Behavior)
```
Attempt 1: Agent builds source, creates simulation → Accepted ✅
Result: Incomplete experiment marked as successful
```

### After (New Behavior)
```
Attempt 1: Agent builds source, creates simulation → REJECTED ❌
           Reason: "Simulation detected: found keywords ['simulated', 'demonstration']"

Attempt 2: Agent tries again with feedback → Still uses mock data → REJECTED ❌
           Reason: "Limitation indicates incomplete work: Timing data was simulated"

Attempt 3: Agent reports blocker "need initdb" → Gets help → Proceeds

Attempt 4: Agent collects real data, trains model → Missing routing → REJECTED ❌
           Reason: "Routing experiment but no evidence of routing mechanism implementation"

Attempt 5: Agent implements routing, tests end-to-end → ACCEPTED ✅
Result: Complete experiment with real ML deployment
```

## Testing the Changes

To verify the improvements work:

1. **Re-run the same experiment** that previously succeeded with simulation
2. **Expected outcome**: Should now REJECT the simulated results
3. **Agent should retry** with error feedback about simulation
4. **Eventually complete** all steps or report specific blocker

### Test Command
```bash
# Run the ML routing experiment again
# It should now reject simulation and demand real implementation
```

## Key Files Modified

1. **`backend/app/core/research_engines/scientific_research.py`**
   - Lines 2678-2708: Enhanced anti-simulation requirements
   - Lines 3886-3955: Stricter validation with keyword detection
   - Line 1972: Increased `max_resume_attempts` to 5
   - Line 3147: Increased `max_attempts_per_experiment` to 5

## Validation Keywords Detected

The system now automatically rejects experiments containing:
- "simulated", "simulation"
- "sample data", "synthetic data"
- "demonstration purposes", "for demonstration"
- "placeholder", "mock", "fake", "dummy data"
- "random.uniform", "random" (in data generation context)
- "not performed", "not trained"
- "design completed" (without implementation)
- "design only", "planned but not"

## Summary

These changes transform UAgent from accepting **simulated results** to **demanding real implementation**:

✅ **Programmatic simulation detection**
✅ **ML-specific artifact validation**
✅ **Stricter LLM validation with detailed criteria**
✅ **150% more retry attempts for complex work**
✅ **Clear feedback loop for agent improvement**

The system will now **reject** the previous "successful" experiment and **force** the agent to:
1. Actually collect real performance data
2. Train an actual ML model
3. Embed the model in C/C++ code
4. Implement the routing mechanism
5. Run real end-to-end tests

**Result**: No more fake experiments marked as successful.
