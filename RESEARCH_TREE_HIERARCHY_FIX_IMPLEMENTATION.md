# Research Tree Hierarchy Fix - Implementation Summary

## Implementation Date
2025-10-02

## Problem Solved
Fixed incorrect tree node hierarchy where validation, attempt, and result nodes were appearing at Level 4 instead of their correct levels (6, 6, and 7 respectively).

## Changes Made

### File Modified
`backend/app/core/research_engines/scientific_research.py`

### Change Summary

| Change # | Lines | Description | Level Fixed |
|----------|-------|-------------|-------------|
| 1 | 4085-4106 | Store `iteration_node_id` when creating iteration nodes | Level 5 ✅ |
| 2 | 4142-4155 | Fix validation node to be child of iteration | Level 6 ✅ |
| 3 | 4176-4192 | Fix sequential plan node to be child of iteration | Level 6 ✅ |
| 4 | 4363-4382 | Add `iteration_node_id` parameter to `_execute_sequential_plan_with_retries` | N/A |
| 5 | 4389-4413 | Fix attempt nodes to be children of iteration | Level 6 ✅ |
| 6 | 4430-4451 | Update attempt status node with correct parent | Level 6 ✅ |
| 7 | 4415-4438 | Store `attempt_node_id` in execution metadata | N/A |
| 8 | 4209-4258 | Fix result nodes to be children of attempts | Level 7 ✅ |
| 9 | 4815-4830 | Fix hypothesis nodes with correct node_type | Level 5 ✅ |
| 10 | 4852-4871 | Add evaluation node logging | Level 5 ✅ |

## Detailed Changes

### Change 1: Store Iteration Node ID (Lines 4085-4106)

**Before**:
```python
if session_id and idea.node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_experiments_iter_{iteration}",
        message=f"Experiments iteration {iteration}",
        metadata={
            "parent_id": idea.node_id,
            "node_type": "step",  # ❌ Generic type
        },
    )
```

**After**:
```python
iteration_node_id: Optional[str] = None
if session_id and idea.node_id:
    iteration_node_id = await self._log_progress(  # ✅ Store node ID
        session_id,
        phase=f"{idea.id}_experiments_iter_{iteration}",
        message=f"Experiments iteration {iteration}",
        metadata={
            "parent_id": idea.node_id,  # Idea is Level 4, iteration is Level 5
            "node_type": "iteration",  # ✅ Specific type
            "title": f"Iteration {iteration}",
            "max_iterations": max_iterations,
        },
    )
```

**Impact**: Iteration nodes now properly stored at Level 5 with semantic node type.

---

### Change 2: Fix Validation Node (Lines 4142-4155)

**Before**:
```python
if session_id and idea.node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_validation_failed",
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea (Level 4)
            "node_type": "error",
        },
        parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
    )
```

**After**:
```python
if session_id and iteration_node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_iter_{iteration}_validation_failed",
        metadata={
            "parent_id": iteration_node_id,  # ✅ Points to iteration (Level 5)
            "node_type": "validation_error",
            "title": "Validation Failed",
        },
        parent_phase=f"{idea.id}_experiments_iter_{iteration}",
    )
```

**Impact**: Validation nodes now appear at Level 6 under iterations.

---

### Change 3: Fix Sequential Plan Node (Lines 4176-4192)

**Before**:
```python
if session_id and idea.node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_sequential_plan_{sequential_plan.id}",
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea (Level 4)
            "node_type": "step",
        },
    )
```

**After**:
```python
sequential_plan_node_id: Optional[str] = None
if session_id and iteration_node_id:
    sequential_plan_node_id = await self._log_progress(  # ✅ Store node ID
        session_id,
        phase=f"{idea.id}_iter_{iteration}_sequential_plan_{sequential_plan.id}",
        metadata={
            "parent_id": iteration_node_id,  # ✅ Points to iteration (Level 5)
            "node_type": "plan",
            "title": f"Plan: {sequential_plan.overall_objective}",
        },
    )
```

**Impact**: Sequential plan nodes now appear at Level 6 under iterations.

---

### Change 4: Add iteration_node_id Parameter (Lines 4363-4382)

**Before**:
```python
async def _execute_sequential_plan_with_retries(
    self,
    idea: ResearchIdea,
    # ... other params
    iteration_number: int,
) -> List[ExperimentExecution]:
```

**After**:
```python
async def _execute_sequential_plan_with_retries(
    self,
    idea: ResearchIdea,
    # ... other params
    iteration_number: int,
    iteration_node_id: Optional[str] = None,  # ✅ New parameter
) -> List[ExperimentExecution]:
    """
    Args:
        iteration_node_id: Node ID of the iteration (Level 5) to use as parent for attempts
    """
```

**Impact**: Allows attempts to reference iteration as parent.

---

### Change 5: Fix Attempt Nodes (Lines 4389-4413)

**Before**:
```python
if session_id and idea.node_id:
    attempt_node_id = await self._log_progress(
        session_id,
        phase=attempt_phase,
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea (Level 4)
            "node_type": "step",
        },
    )
```

**After**:
```python
parent_id_for_attempt = iteration_node_id if iteration_node_id else idea.node_id

if session_id and parent_id_for_attempt:
    attempt_node_id = await self._log_progress(
        session_id,
        phase=f"{idea.id}_iter_{iteration_number}_attempt_{attempt}",
        metadata={
            "parent_id": parent_id_for_attempt,  # ✅ Points to iteration (Level 5)
            "node_type": "attempt",
            "title": f"Attempt {attempt}/{self.max_attempts_per_experiment}",
            "max_attempts": self.max_attempts_per_experiment,
        },
    )
```

**Impact**: Attempt nodes now appear at Level 6 under iterations.

---

### Change 7: Store Attempt Node ID in Execution (Lines 4415-4438)

**New Code**:
```python
# Store attempt_node_id in each execution for later use when creating result nodes
for execution in executions:
    if not hasattr(execution, 'intermediate_results'):
        execution.intermediate_results = {}
    execution.intermediate_results['attempt_node_id'] = attempt_node_id
    execution.intermediate_results['attempt_number'] = attempt
    execution.intermediate_results['iteration_number'] = iteration_number
```

**Impact**: Result nodes can now reference their parent attempt.

---

### Change 8: Fix Result Nodes (Lines 4209-4258)

**Before**:
```python
if session_id and idea.node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_result_{experiment_result.execution_id}",
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea (Level 4)
            "node_type": "result",
        },
    )
```

**After**:
```python
# Get attempt_node_id from execution metadata
attempt_node_id_for_result = execution.intermediate_results.get('attempt_node_id')
attempt_num = execution.intermediate_results.get('attempt_number', 1)

if session_id and attempt_node_id_for_result:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_iter_{iteration}_attempt_{attempt_num}_result_{exp_index + 1}",
        metadata={
            "parent_id": attempt_node_id_for_result,  # ✅ Points to attempt (Level 6)
            "node_type": "result",
            "title": f"Result: {design.name}",
        },
        parent_phase=f"{idea.id}_iter_{iteration}_attempt_{attempt_num}",
    )
```

**Impact**: Result nodes now appear at Level 7 under attempts.

---

### Change 9: Fix Hypothesis Node Type (Lines 4815-4830)

**Before**:
```python
await self._log_progress(
    session_id,
    phase=f"{idea.id}_hypothesis_{h_idx}",
    metadata={
        "parent_id": idea.node_id,
        "node_type": "result",  # ❌ Wrong type
        "title": hypothesis.statement,
    },
)
```

**After**:
```python
hypothesis_node_id = await self._log_progress(
    session_id,
    phase=f"{idea.id}_hypothesis_{h_idx}",
    message=f"Hypothesis {h_idx}: {hypothesis.statement[:60]}...",
    metadata={
        "parent_id": idea.node_id,  # Idea is Level 4, hypothesis is Level 5
        "node_type": "hypothesis",  # ✅ Correct type
        "title": f"H{h_idx}: {hypothesis.statement}",
        "testable_predictions": hypothesis.testable_predictions[:3],
    },
)
```

**Impact**: Hypothesis nodes have semantic type and appear correctly at Level 5.

---

### Change 10: Add Evaluation Node (Lines 4852-4871)

**New Code**:
```python
# Log evaluation node to tree
if session_id and idea.node_id and idea.evaluation:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_evaluation",
        message=f"Idea Evaluation: Score {idea.evaluation.overall_score:.1f}/10",
        metadata={
            "parent_id": idea.node_id,  # Idea is Level 4, evaluation is Level 5
            "node_type": "evaluation",
            "title": "Idea Evaluation",
            "overall_score": idea.evaluation.overall_score,
            "novelty_score": idea.evaluation.novelty_score,
            "feasibility_score": idea.evaluation.feasibility_score,
            "impact_score": idea.evaluation.impact_score,
            "confidence": idea.confidence_score,
        },
    )
```

**Impact**: Evaluation nodes now appear at Level 5 (previously missing from tree).

---

## Tree Hierarchy After Fix

```
Level 1: Root
├── Level 2: Scientific Research Session
    ├── Level 3: Ideation Phase
    │   ├── Level 4: Idea #1
    │   │   ├── Level 5: Hypothesis #1 ✅
    │   │   ├── Level 5: Iteration #1 ✅
    │   │   │   ├── Level 6: Validation (if fails) ✅
    │   │   │   ├── Level 6: Sequential Plan ✅
    │   │   │   ├── Level 6: Attempt #1 ✅
    │   │   │   │   ├── Level 7: Data Collection ✅
    │   │   │   │   ├── Level 7: Result for Exp 1 ✅
    │   │   │   │   ├── Level 7: Result for Exp 2 ✅
    │   │   │   ├── Level 6: Attempt #2 (if retry) ✅
    │   │   ├── Level 5: Evaluation ✅
    │   ├── Level 4: Idea #2
    ├── Level 3: Synthesis Phase
```

## Node Types Updated

| Node Type | Old Value | New Value | Purpose |
|-----------|-----------|-----------|---------|
| Iteration | `"step"` | `"iteration"` | Semantic clarity |
| Validation | `"error"` | `"validation_error"` | Specific error type |
| Plan | `"step"` | `"plan"` | Semantic clarity |
| Attempt | `"step"` | `"attempt"` | Semantic clarity |
| Hypothesis | `"result"` | `"hypothesis"` | Correct semantic type |
| Evaluation | N/A (missing) | `"evaluation"` | New node type |
| Result | `"result"` | `"result"` | No change (correct) |

## Testing Recommendations

### Manual Testing
1. Run a scientific research query
2. Open frontend tree visualization
3. Verify hierarchy levels:
   - Ideas at Level 4
   - Hypotheses, Iterations, Evaluation at Level 5
   - Validation, Plan, Attempts at Level 6
   - Results, Data Collection at Level 7

### Verification Checklist
- [ ] Tree expands/collapses correctly at each level
- [ ] No orphaned nodes
- [ ] Node icons match node types
- [ ] Progress percentages increase monotonically
- [ ] Validation nodes appear under iterations
- [ ] Attempt nodes appear under iterations
- [ ] Result nodes appear under attempts
- [ ] Evaluation nodes appear under ideas

## Expected User Experience Improvements

### Before Fix
- Confusing flat hierarchy at Level 4
- Hard to distinguish between different execution stages
- Attempt and validation nodes mixed with ideas
- No visual separation between iterations and attempts

### After Fix
- Clear hierarchical structure (7-8 levels deep)
- Easy to understand research flow
- Proper nesting: Idea → Iteration → Attempt → Result
- Visual clarity in experiment progression
- Semantic node types for better styling and icons

## Rollback Plan

If issues occur:
```bash
git diff backend/app/core/research_engines/scientific_research.py
git checkout backend/app/core/research_engines/scientific_research.py
```

Changes are:
- ✅ Non-breaking: All changes are to metadata, not data
- ✅ Backwards compatible: Old sessions won't be affected
- ✅ Self-contained: Only one file modified
- ✅ Syntax-validated: Code compiles without errors

## Performance Impact

- **Negligible**: Only adds node ID storage in local variables
- **No database changes**: Tree structure is ephemeral (WebSocket only)
- **No API changes**: All changes internal to scientific_research.py

## Success Metrics

After deployment:
1. Tree depth should be 7-8 levels (was 4 before)
2. Validation nodes at Level 6 (was Level 4)
3. Attempt nodes at Level 6 (was Level 4)
4. Result nodes at Level 7 (was Level 4)
5. Evaluation nodes visible at Level 5 (was missing)

## Related Documents

- Planning document: `RESEARCH_TREE_HIERARCHY_FIX_PLAN.md`
- Issue analysis: Validation and attempts appearing at wrong level
- Frontend impact: Tree visualization will show correct hierarchy

## Implementation Notes

- Used `Optional[str]` for node IDs to handle cases where logging is disabled
- Stored node IDs in execution metadata for cross-method access
- Used fallback logic (`iteration_node_id if iteration_node_id else idea.node_id`) for robustness
- All phase names updated to include iteration/attempt numbers for uniqueness
- Node titles added for better frontend display
