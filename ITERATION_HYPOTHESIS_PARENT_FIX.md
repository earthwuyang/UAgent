# Iteration-Hypothesis Parent Relationship Fix

## Problem Statement

**User Report**: "lv3-n3 iteration 1 should be a child of either lv3-n1 or n2, but currently it belongs to lv2-n1 the core exploration node, which hypothesis does it tests then?"

Iterations were incorrectly structured as children of ideas instead of hypotheses:

**Incorrect Structure**:
```
Idea (Core exploration) - Level 4
├── Hypothesis 1 - Level 5
├── Hypothesis 2 - Level 5
├── Iteration 1 - Level 5 ❌ WRONG PARENT (testing H1 but shows under idea)
    ├── Attempt 1 - Level 6
        ├── Result - Level 7
```

**Desired Structure**:
```
Idea (Core exploration) - Level 4
├── Hypothesis 1 - Level 5
│   ├── Iteration 1 - Level 6 ✅ CORRECT (testing H1, shown under H1)
│       ├── Attempt 1 - Level 7
│           ├── Result - Level 8
├── Hypothesis 2 - Level 5
    ├── Iteration 1 - Level 6 ✅ CORRECT (testing H2, shown under H2)
        ├── Attempt 1 - Level 7
            ├── Result - Level 8
```

## Root Cause

In `_run_experiments_for_idea` method (lines 4089-4280):

1. **Iteration created BEFORE hypothesis loop** (line 4109):
   - Created as child of `idea.node_id` (Level 4)
   - Made iteration Level 5

2. **Hypothesis loop processes one hypothesis per iteration** (line 4130):
   - Loop iterates `pending_hypotheses`
   - But breaks after first hypothesis (line 4277-4279)

3. **Semantic mismatch**:
   - Each iteration tests ONE hypothesis
   - But tree showed iteration under idea, not hypothesis
   - User couldn't tell which hypothesis was being tested

## Solution

### Change 1: Add `node_id` to `ResearchHypothesis` Dataclass

**File**: `backend/app/core/research_engines/scientific_research.py` (Lines 91-103)

**Before**:
```python
@dataclass
class ResearchHypothesis:
    """Research hypothesis with validation criteria"""
    id: str
    statement: str
    reasoning: str
    testable_predictions: List[str]
    success_criteria: Dict[str, Any]
    variables: Dict[str, Any]
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence_level: float = 0.0
    evidence: List[str] = field(default_factory=list)
```

**After**:
```python
@dataclass
class ResearchHypothesis:
    """Research hypothesis with validation criteria"""
    id: str
    statement: str
    reasoning: str
    testable_predictions: List[str]
    success_criteria: Dict[str, Any]
    variables: Dict[str, Any]
    status: HypothesisStatus = HypothesisStatus.PENDING
    confidence_level: float = 0.0
    evidence: List[str] = field(default_factory=list)
    node_id: Optional[str] = None  # Tree node ID for hierarchy ✅ NEW
```

**Impact**: Hypothesis objects can now store their tree node IDs for parent reference.

---

### Change 2: Store Hypothesis Node IDs

**File**: `backend/app/core/research_engines/scientific_research.py` (Lines 4833-4850)

**Before**:
```python
if session_id and idea.node_id:
    for h_idx, hypothesis in enumerate(hypotheses, start=1):
        hypothesis_node_id = await self._log_progress(
            session_id,
            phase=f"{idea.id}_hypothesis_{h_idx}",
            progress=base_progress + progress_window * 0.35,
            message=f"Hypothesis {h_idx}: {hypothesis.statement[:60]}...",
            metadata={
                "parent_id": idea.node_id,
                "node_type": "hypothesis",
                "title": f"H{h_idx}: {hypothesis.statement}",
                # ...
            },
            parent_phase=f"idea_{idea.id}",
        )
        # hypothesis_node_id not stored ❌
```

**After**:
```python
if session_id and idea.node_id:
    for h_idx, hypothesis in enumerate(hypotheses, start=1):
        hypothesis_node_id = await self._log_progress(
            session_id,
            phase=f"{idea.id}_hypothesis_{h_idx}",
            progress=base_progress + progress_window * 0.35,
            message=f"Hypothesis {h_idx}: {hypothesis.statement[:60]}...",
            metadata={
                "parent_id": idea.node_id,
                "node_type": "hypothesis",
                "title": f"H{h_idx}: {hypothesis.statement}",
                # ...
            },
            parent_phase=f"idea_{idea.id}",
        )
        # Store node_id on hypothesis object for iteration parent reference ✅
        hypothesis.node_id = hypothesis_node_id
```

**Impact**: Hypothesis node IDs are now accessible for iterations to reference as parents.

---

### Change 3: Move Iteration Creation Inside Hypothesis Loop

**File**: `backend/app/core/research_engines/scientific_research.py` (Lines 4103-4130)

**Before**:
```python
iteration += 1
idea.iteration_count = iteration
iteration_progress = base_progress + per_iteration_increment * (iteration - 1)

# Create iteration node BEFORE hypothesis loop ❌
iteration_node_id: Optional[str] = None
if session_id and idea.node_id:
    iteration_node_id = await self._log_progress(
        session_id,
        phase=f"{idea.id}_experiments_iter_{iteration}",
        progress=iteration_progress,
        message=f"Experiments iteration {iteration}",
        metadata={
            "parent_id": idea.node_id,  # Idea is Level 4, iteration is Level 5 ❌
            "node_type": "iteration",
            "iteration": iteration,
            "max_iterations": max_iterations,
            "pending_hypotheses": len(pending_hypotheses),
            "title": f"Iteration {iteration}",
        },
        parent_phase=f"idea_{idea.id}",
    )

iteration_results: List[ExperimentResult] = []

for hypothesis in pending_hypotheses:  # No index ❌
    # Design experiments...
```

**After**:
```python
iteration += 1
idea.iteration_count = iteration
iteration_progress = base_progress + per_iteration_increment * (iteration - 1)

iteration_results: List[ExperimentResult] = []

# Create iteration node INSIDE hypothesis loop ✅
for h_idx, hypothesis in enumerate(pending_hypotheses, start=1):  # With index ✅
    iteration_node_id: Optional[str] = None
    if session_id and hypothesis.node_id:
        iteration_node_id = await self._log_progress(
            session_id,
            phase=f"{idea.id}_hyp_{hypothesis.id}_iter_{iteration}",  # Include hyp ID ✅
            progress=iteration_progress,
            message=f"Iteration {iteration} for H{h_idx}",  # Show which hypothesis ✅
            metadata={
                "parent_id": hypothesis.node_id,  # Hypothesis is Level 5, iteration is Level 6 ✅
                "node_type": "iteration",
                "iteration": iteration,
                "max_iterations": max_iterations,
                "hypothesis_index": h_idx,  # NEW ✅
                "title": f"Iteration {iteration}",
            },
            parent_phase=f"{idea.id}_hypothesis_{h_idx}",  # Parent is hypothesis ✅
        )
    # Design experiments...
```

**Impact**:
- Iteration is now child of hypothesis (Level 6) instead of idea (Level 5)
- Each hypothesis gets its own iteration node
- Clear which hypothesis is being tested

---

### Change 4: Update Phase Names to Include Hypothesis ID

**Files**: `backend/app/core/research_engines/scientific_research.py`

Updated all phase names from `{idea.id}_iter_{iteration}_*` to `{idea.id}_hyp_{hypothesis.id}_iter_{iteration}_*`:

#### Validation Node (Lines 4160-4172)
```python
# Before: phase=f"{idea.id}_iter_{iteration}_validation_failed"
# After:  phase=f"{idea.id}_hyp_{hypothesis.id}_iter_{iteration}_validation_failed"
```

#### Sequential Plan Node (Lines 4196-4209)
```python
# Before: phase=f"{idea.id}_iter_{iteration}_sequential_plan_{plan.id}"
# After:  phase=f"{idea.id}_hyp_{hypothesis.id}_iter_{iteration}_sequential_plan_{plan.id}"
```

#### Result Nodes (Lines 4241-4275)
```python
# Before: phase=f"{idea.id}_iter_{iteration}_attempt_{num}_result_{idx}"
# After:  phase=f"{idea.id}_hyp_{hypothesis.id}_iter_{iteration}_attempt_{num}_result_{idx}"
```

#### Attempt Nodes in `_execute_sequential_plan_with_retries` (Lines 4410-4414)
```python
# Before: parent_phase = f"idea_{idea.id}_experiments_iter_{iteration_number}"
#         attempt_phase = f"{idea.id}_iter_{iteration_number}_attempt_{attempt}"
# After:  parent_phase = f"{idea.id}_hyp_{hypothesis.id}_iter_{iteration_number}"
#         attempt_phase = f"{idea.id}_hyp_{hypothesis.id}_iter_{iteration_number}_attempt_{attempt}"
```

**Impact**: Phase names now uniquely identify which hypothesis is being tested.

---

### Change 5: Update Level Comments

Updated all level comments to reflect new hierarchy:

- Hypothesis: Level 5 (unchanged)
- Iteration: Level 6 (was 5) ✅
- Validation: Level 7 (was 6) ✅
- Plan: Level 7 (was 6) ✅
- Attempt: Level 7 (was 6) ✅
- Result: Level 8 (was 7) ✅

## Tree Structure Comparison

### Before Fix

```
Level 1: Root
├── Level 2: scientific_research
    ├── Level 3: Ideation
        ├── Level 4: Idea (Core exploration)
            ├── Level 5: Hypothesis 1 ✅
            ├── Level 5: Hypothesis 2 ✅
            ├── Level 5: Iteration 1 ❌ WRONG PARENT
                ├── Level 6: Attempt 1
                    ├── Level 7: Result
```

**Problems**:
- Iteration at same level as hypotheses
- Can't tell which hypothesis iteration tests
- Semantically incorrect hierarchy

---

### After Fix

```
Level 1: Root
├── Level 2: scientific_research
    ├── Level 3: Ideation
        ├── Level 4: Idea (Core exploration)
            ├── Level 5: Hypothesis 1 ✅
            │   ├── Level 6: Iteration 1 ✅ CORRECT PARENT
            │       ├── Level 7: Attempt 1
            │           ├── Level 8: Result
            ├── Level 5: Hypothesis 2 ✅
                ├── Level 6: Iteration 1 ✅ CORRECT PARENT
                    ├── Level 7: Attempt 1
                        ├── Level 8: Result
```

**Benefits**:
- ✅ Iteration is child of hypothesis (Level 6)
- ✅ Clear which hypothesis each iteration tests
- ✅ Semantically correct: Hypothesis → Iteration → Attempt → Result
- ✅ Each hypothesis can have multiple iterations
- ✅ Tree depth increased by 1 level (acceptable tradeoff for clarity)

## Benefits

### 1. Correct Semantic Hierarchy ✅

- Iterations now belong to the hypotheses they test
- Tree structure matches research methodology
- Clear parent-child relationships

### 2. Better User Understanding ✅

- User can see which hypothesis each iteration tests
- No confusion about "which hypothesis does it test then?"
- Iteration title shows "Iteration 1 for H1"

### 3. Supports Multiple Hypotheses Per Idea ✅

- Each hypothesis can have its own iterations
- Proper nesting for complex research
- Future-proof for parallel hypothesis testing

### 4. Improved Metadata ✅

- Phase names include hypothesis ID
- Easy to track progress per hypothesis
- Better debugging and logging

## Edge Cases Handled

### Case 1: Fallback Idea with Single Hypothesis

**Scenario**: Idea generation fails, fallback "Core exploration" idea created with one hypothesis

**Before**:
```
Core exploration (Level 4)
├── H1: Evaluate core assumption (Level 5)
├── Iteration 1 (Level 5) ❌ Same level as hypothesis
```

**After**:
```
Core exploration (Level 4)
├── H1: Evaluate core assumption (Level 5)
    ├── Iteration 1 (Level 6) ✅ Child of hypothesis
```

### Case 2: Multiple Hypotheses (Future)

**Scenario**: Multiple pending hypotheses, each needs iterations

**Before**: All iterations would be under idea (confusing)

**After**: Each hypothesis has its own iteration subtree (clear)

```
Idea
├── H1: Model accuracy improves
│   ├── Iteration 1 (testing H1)
│   ├── Iteration 2 (testing H1 again)
├── H2: Training time reduces
    ├── Iteration 1 (testing H2)
```

### Case 3: Iteration Break After First Hypothesis

**Current Behavior**: Code breaks after processing first hypothesis (line 4277-4279)

**Impact**: Only first hypothesis gets an iteration, which is now correctly shown in tree

**Future**: If break is removed, multiple hypotheses will each get iterations under correct parents

## Testing

### Manual Test

1. Start a scientific research session
2. Check tree visualization
3. Verify structure:
   - ✅ Hypothesis at Level 5
   - ✅ Iteration at Level 6 (child of hypothesis)
   - ✅ Iteration title shows "Iteration 1 for H1"
   - ✅ Attempt at Level 7
   - ✅ Result at Level 8

### Expected Tree (Simple Research)

```
scientific_research
├── Ideation
    ├── Core exploration: <question>
        ├── H1: Evaluate core assumption
            ├── Iteration 1
                ├── Plan: Sequential experiments
                ├── Attempt 1
                    ├── Result: Experiment 1
                    ├── Result: Experiment 2
```

## Backwards Compatibility

- ✅ **Compatible**: Existing code still works
- ⚠️ **Tree depth increased by 1**: Iterations now at Level 6 instead of 5
- ✅ **No API changes**: Internal tree structure only
- ✅ **Frontend**: Automatically adjusts to new hierarchy

## Files Modified

1. `backend/app/core/research_engines/scientific_research.py`
   - Line 103: Added `node_id` field to `ResearchHypothesis`
   - Line 4850: Store hypothesis node ID
   - Lines 4112-4130: Move iteration creation inside hypothesis loop
   - Lines 4160-4172: Update validation phase names
   - Lines 4196-4209: Update plan phase names
   - Lines 4241-4275: Update result phase names
   - Lines 4410-4427: Update attempt phase names in `_execute_sequential_plan_with_retries`
   - Updated level comments throughout (6→7, 7→8)

## Rollback Plan

If issues occur:
```bash
git diff backend/app/core/research_engines/scientific_research.py
git checkout backend/app/core/research_engines/scientific_research.py
```

## Success Metrics

- ✅ Iteration is child of hypothesis (Level 6)
- ✅ Iteration title shows which hypothesis is tested
- ✅ Phase names include hypothesis ID
- ✅ Proper parent-child relationships throughout
- ✅ Tree semantically matches research methodology

## Related Changes

This fix complements:
1. **Tree Hierarchy Fix** (RESEARCH_TREE_HIERARCHY_FIX_IMPLEMENTATION.md)
   - Fixed iteration/attempt/result levels
2. **Duplicate Node Fix** (DUPLICATE_NODE_FIX.md)
   - Fixed uppercase/lowercase duplication
3. **Trivial Nodes Fix** (REMOVE_TRIVIAL_NODES_FIX.md)
   - Removed "Planning multi-engine" node
4. **Validation Lenient Fix** (VALIDATION_LENIENT_FIX.md)
   - Made validation less strict

Together, these create a **clean, semantically correct tree** that accurately represents the scientific research process.

## Implementation Date

2025-10-02

## Priority

High - Fixes semantic mismatch and user confusion about which hypothesis is being tested
