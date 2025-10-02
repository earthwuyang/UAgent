# Research Tree Hierarchy Fix Plan

## Problem Statement

The current research tree in UAgent has incorrect node hierarchy levels. Specifically:

**Current Behavior** (INCORRECT):
- Experiment validation nodes appear at **Level 4** (should be Level 5)
- Attempt nodes appear at **Level 4** (should be Level 5)
- Sequential plan nodes and their children don't have proper nesting

**Root Cause**:
- Nodes are created with `parent_id` pointing to the idea node directly
- `parent_phase` doesn't properly establish intermediate parent relationships
- Frontend calculates tree levels based on parent relationships, resulting in incorrect nesting

## Current Tree Hierarchy (As Implemented)

```
Level 1: Root
├── Level 2: Scientific Research Session (session_id)
    ├── Level 3: Ideation Phase
    │   ├── Level 4: Idea #1
    │   │   ├── Level 4: Hypothesis #1 ❌ WRONG (should be Level 5)
    │   │   ├── Level 4: Iteration #1 ❌ WRONG (should be Level 5)
    │   │   │   ├── Level 4: Validation (failed) ❌ WRONG (should be Level 6)
    │   │   │   ├── Level 4: Sequential Plan ❌ WRONG (should be Level 6)
    │   │   │   ├── Level 4: Attempt #1 ❌ WRONG (should be Level 7)
    │   │   │   │   ├── Level 4: Data Collection ❌ WRONG (should be Level 8)
    │   │   │   │   ├── Level 4: Container execution ❌ WRONG (should be Level 8)
    │   │   │   ├── Level 4: Attempt #2 ❌ WRONG (should be Level 7)
    │   │   ├── Level 4: Result #1 ❌ WRONG (should be Level 7 or 8)
    │   │   ├── Level 4: Evaluation ❌ WRONG (should be Level 5)
    │   ├── Level 4: Idea #2
    ├── Level 3: Synthesis Phase
```

## Desired Tree Hierarchy (CORRECT)

```
Level 1: Root
├── Level 2: Scientific Research Session (session_id)
    ├── Level 3: Ideation Phase
    │   ├── Level 4: Idea #1 (ResearchIdea)
    │   │   ├── Level 5: Hypothesis #1
    │   │   ├── Level 5: Iteration #1 (Experiments Iteration)
    │   │   │   ├── Level 6: Validation (pre-execution)
    │   │   │   │   ├── Level 7: Validation Failed (if fails)
    │   │   │   │   ├── Level 7: Validation Passed (if passes)
    │   │   │   ├── Level 6: Sequential Plan Design
    │   │   │   │   ├── Level 7: Plan Details (num_experiments, objective)
    │   │   │   ├── Level 6: Plan Execution
    │   │   │       ├── Level 7: Attempt #1
    │   │   │       │   ├── Level 8: Data Collection Start
    │   │   │       │   ├── Level 8: Container Execution
    │   │   │       │   ├── Level 8: Result for Experiment 1
    │   │   │       │   ├── Level 8: Result for Experiment 2
    │   │   │       │   ├── Level 8: Post-execution Validation
    │   │   │       ├── Level 7: Attempt #2 (if first fails)
    │   │   │       │   ├── Level 8: Data Collection Start
    │   │   │       │   ├── Level 8: Container Execution
    │   │   │       │   ├── Level 8: Results...
    │   │   ├── Level 5: Iteration #2 (if needed)
    │   │   │   ├── Level 6: ...
    │   │   ├── Level 5: Idea Evaluation
    │   ├── Level 4: Idea #2
    │   │   ├── Level 5: ...
    ├── Level 3: Idea Evaluation Summary
    ├── Level 3: Synthesis Phase
```

## Node Level Definitions

| Level | Node Type | Description | Parent |
|-------|-----------|-------------|--------|
| 1 | Root | Top-level research session root | None |
| 2 | Research Session | Main research session container | Root |
| 3 | Phase | High-level phases (Ideation, Synthesis) | Research Session |
| 4 | Idea | Individual research idea being explored | Ideation Phase |
| 5 | Idea Components | Hypotheses, Iterations, Evaluation | Idea |
| 6 | Iteration Steps | Validation, Plan Design, Plan Execution | Iteration |
| 7 | Execution Attempts | Individual attempt at executing plan | Plan Execution |
| 8 | Attempt Details | Data collection, results, validation | Attempt |

## Code Changes Required

### File: `backend/app/core/research_engines/scientific_research.py`

#### Change 1: Create Iteration Node Properly (Lines ~4089-4102)

**Current Code**:
```python
if session_id and idea.node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_experiments_iter_{iteration}",
        progress=iteration_progress,
        message=f"Experiments iteration {iteration}",
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea - makes this Level 4
            "node_type": "step",
            "iteration": iteration,
            "pending_hypotheses": len(pending_hypotheses),
        },
        parent_phase=f"idea_{idea.id}",  # ❌ Points to idea
    )
```

**Fixed Code**:
```python
if session_id and idea.node_id:
    iteration_node_id = await self._log_progress(
        session_id,
        phase=f"{idea.id}_experiments_iter_{iteration}",
        progress=iteration_progress,
        message=f"Experiments iteration {iteration}",
        metadata={
            "parent_id": idea.node_id,  # ✅ Idea is Level 4, so this is Level 5
            "node_type": "iteration",  # ✅ Changed from "step" to "iteration"
            "iteration": iteration,
            "pending_hypotheses": len(pending_hypotheses),
            "title": f"Iteration {iteration}",
        },
        parent_phase=f"idea_{idea.id}",  # ✅ Correctly points to idea (Level 4)
    )
    # Store iteration_node_id for children to use
```

#### Change 2: Validation Node Should Be Child of Iteration (Lines ~4137-4150)

**Current Code**:
```python
await self._log_progress(
    session_id,
    phase=f"{idea.id}_validation_failed",
    progress=iteration_progress + per_iteration_increment * 0.15,
    message=f"Experiment plan validation failed - redesigning with stricter constraints",
    metadata={
        "parent_id": idea.node_id,  # ❌ Points to idea (Level 4) - makes validation Level 4
        "node_type": "error",
        "validation_errors": validation_errors[:5],
    },
    parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",  # ❌ Confusing
)
```

**Fixed Code**:
```python
await self._log_progress(
    session_id,
    phase=f"{idea.id}_iter_{iteration}_validation_failed",
    progress=iteration_progress + per_iteration_increment * 0.15,
    message=f"Experiment plan validation failed - redesigning with stricter constraints",
    metadata={
        "parent_id": iteration_node_id,  # ✅ Points to iteration (Level 5) - makes validation Level 6
        "node_type": "validation_error",
        "validation_errors": validation_errors[:5],
        "title": "Validation Failed",
    },
    parent_phase=f"{idea.id}_experiments_iter_{iteration}",  # ✅ Points to iteration phase
)
```

#### Change 3: Sequential Plan Node Should Be Child of Iteration (Lines ~4171-4185)

**Current Code**:
```python
await self._log_progress(
    session_id,
    phase=f"{idea.id}_sequential_plan_{sequential_plan.id}",
    progress=iteration_progress + per_iteration_increment * 0.2,
    message=f"Designed sequential plan: {sequential_plan.overall_objective}",
    metadata={
        "parent_id": idea.node_id,  # ❌ Points to idea - makes plan Level 4
        "node_type": "step",
        "title": sequential_plan.overall_objective,
        "num_experiments": sequential_plan.num_experiments,
        "experiment_names": [exp.name for exp in sequential_plan.experiments],
    },
    parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
)
```

**Fixed Code**:
```python
sequential_plan_node_id = await self._log_progress(
    session_id,
    phase=f"{idea.id}_iter_{iteration}_sequential_plan_{sequential_plan.id}",
    progress=iteration_progress + per_iteration_increment * 0.2,
    message=f"Designed sequential plan: {sequential_plan.overall_objective}",
    metadata={
        "parent_id": iteration_node_id,  # ✅ Points to iteration (Level 5) - makes plan Level 6
        "node_type": "plan",
        "title": f"Plan: {sequential_plan.overall_objective}",
        "num_experiments": sequential_plan.num_experiments,
        "experiment_names": [exp.name for exp in sequential_plan.experiments],
    },
    parent_phase=f"{idea.id}_experiments_iter_{iteration}",  # ✅ Points to iteration
)
# Store sequential_plan_node_id for execution attempts to use
```

#### Change 4: Create Plan Execution Container Node (New - Lines after ~4186)

**New Code** (insert after sequential plan node creation):
```python
# Create "Plan Execution" container node for attempts
plan_execution_node_id = await self._log_progress(
    session_id,
    phase=f"{idea.id}_iter_{iteration}_plan_{sequential_plan.id}_execution",
    progress=iteration_progress + per_iteration_increment * 0.25,
    message=f"Executing plan: {sequential_plan.overall_objective}",
    metadata={
        "parent_id": sequential_plan_node_id,  # ✅ Child of plan (Level 6) - makes this Level 7... wait no
        # Actually, let's make it child of iteration for better organization
        "parent_id": iteration_node_id,  # ✅ Child of iteration (Level 5) - makes execution Level 6
        "node_type": "execution",
        "title": "Plan Execution",
        "num_experiments": sequential_plan.num_experiments,
    },
    parent_phase=f"{idea.id}_iter_{iteration}_sequential_plan_{sequential_plan.id}",
)
```

**Actually, better structure**: Let's make attempts direct children of iteration for cleaner hierarchy:
- Iteration (Level 5)
  - Validation (Level 6)
  - Sequential Plan (Level 6)
  - Attempt #1 (Level 6) ← simpler
  - Attempt #2 (Level 6)

#### Change 5: Attempt Nodes Should Be Children of Iteration (Lines ~4382-4396)

**Current Code**:
```python
if session_id and idea.node_id:
    attempt_node_id = await self._log_progress(
        session_id,
        phase=attempt_phase,  # f"{idea.id}_sequential_execution_attempt_{attempt}"
        progress=attempt_progress,
        message=f"Attempt {attempt} for sequential plan (starting): {plan.overall_objective}",
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea (Level 4) - makes attempt Level 4
            "node_type": "step",
            "attempt": attempt,
            "num_experiments": plan.num_experiments,
            "status": "starting",
        },
        parent_phase=parent_phase,  # f"idea_{idea.id}_experiments_iter_{iteration_number}"
    ) or attempt_node_id
```

**Fixed Code**:
```python
if session_id and idea.node_id:
    attempt_node_id = await self._log_progress(
        session_id,
        phase=f"{idea.id}_iter_{iteration_number}_attempt_{attempt}",  # ✅ More specific phase
        progress=attempt_progress,
        message=f"Attempt {attempt} for sequential plan: {plan.overall_objective}",
        metadata={
            "parent_id": iteration_node_id,  # ✅ Points to iteration (Level 5) - makes attempt Level 6
            "node_type": "attempt",  # ✅ Changed from "step" to "attempt"
            "attempt": attempt,
            "max_attempts": self.max_attempts_per_experiment,
            "num_experiments": plan.num_experiments,
            "status": "starting",
            "title": f"Attempt {attempt}/{self.max_attempts_per_experiment}",
        },
        parent_phase=f"{idea.id}_experiments_iter_{iteration_number}",  # ✅ Points to iteration
    ) or attempt_node_id
```

#### Change 6: Data Collection Nodes Should Be Children of Attempt (Lines ~2676-2717)

**Current Code** (in `_collect_experimental_data`):
```python
parent_node_id = attempt_context.get("parent_node_id")  # This is attempt_node_id
parent_phase = attempt_context.get("parent_phase")  # This is attempt phase
# ...
async def _log_collection_event(...):
    # ...
    metadata.setdefault("parent_id", parent_node_id)  # ✅ Already correct - points to attempt
    # ...
```

**No change needed** - data collection already uses `parent_node_id` from `attempt_context`, which is the attempt node ID. This is correct.

#### Change 7: Result Nodes Should Be Children of Attempt (Lines ~4211-4226, 4229-4244)

**Current Code**:
```python
if session_id and idea.node_id:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_result_{experiment_result.execution_id}",
        progress=iteration_progress + per_iteration_increment * 0.65,
        message=f"Result for {design.name} (Exp {exp_index + 1}/{sequential_plan.num_experiments})",
        metadata={
            "parent_id": idea.node_id,  # ❌ Points to idea - makes result Level 4
            "node_type": "result",
            "conclusions": experiment_result.conclusions[:2],
            "confidence_score": experiment_result.confidence_score,
            "experiment_index": exp_index + 1,
            "total_experiments": sequential_plan.num_experiments,
        },
        parent_phase=f"idea_{idea.id}_experiments_iter_{iteration}",
    )
```

**Fixed Code**:
```python
if session_id and attempt_node_id:  # ✅ Use attempt_node_id instead of idea.node_id
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_iter_{iteration}_attempt_{attempt}_result_{exp_index + 1}",
        progress=iteration_progress + per_iteration_increment * 0.65,
        message=f"Result: {design.name} (Exp {exp_index + 1}/{sequential_plan.num_experiments})",
        metadata={
            "parent_id": attempt_node_id,  # ✅ Points to attempt (Level 6) - makes result Level 7
            "node_type": "result",
            "title": f"Result: {design.name}",
            "conclusions": experiment_result.conclusions[:2],
            "confidence_score": experiment_result.confidence_score,
            "experiment_index": exp_index + 1,
            "total_experiments": sequential_plan.num_experiments,
        },
        parent_phase=f"{idea.id}_iter_{iteration_number}_attempt_{attempt}",  # ✅ Points to attempt
    )
```

**Same fix for failed experiment results** (lines 4229-4244).

#### Change 8: Hypothesis Nodes Should Be Level 5 (Lines ~4780-4794)

**Current Code**:
```python
if session_id and idea.node_id:
    for h_idx, hypothesis in enumerate(hypotheses, start=1):
        await self._log_progress(
            session_id,
            phase=f"{idea.id}_hypothesis_{h_idx}",
            progress=base_progress + progress_window * 0.35,
            message=f"Hypothesis {h_idx}",
            metadata={
                "parent_id": idea.node_id,  # ✅ Actually correct - Level 5
                "node_type": "result",  # ⚠️ Should be "hypothesis"
                "title": hypothesis.statement,
                "reasoning": hypothesis.reasoning,
            },
            parent_phase=f"idea_{idea.id}",  # ✅ Correct
        )
```

**Fixed Code**:
```python
if session_id and idea.node_id:
    for h_idx, hypothesis in enumerate(hypotheses, start=1):
        hypothesis_node_id = await self._log_progress(
            session_id,
            phase=f"{idea.id}_hypothesis_{h_idx}",
            progress=base_progress + progress_window * 0.35,
            message=f"Hypothesis {h_idx}: {hypothesis.statement[:60]}...",
            metadata={
                "parent_id": idea.node_id,  # ✅ Correct - points to idea (Level 4), makes this Level 5
                "node_type": "hypothesis",  # ✅ Changed from "result" to "hypothesis"
                "title": f"H{h_idx}: {hypothesis.statement}",
                "reasoning": hypothesis.reasoning,
                "testable_predictions": hypothesis.testable_predictions,
            },
            parent_phase=f"idea_{idea.id}",  # ✅ Correct
        )
        # Store hypothesis node ID if needed later
```

#### Change 9: Evaluation Node Should Be Level 5 (needs to be added)

**Currently missing** - evaluation happens but isn't logged to tree. Should add:

```python
# In _evaluate_idea or after it's called (around line 4810-4814)
if session_id and idea.node_id and idea.evaluation:
    await self._log_progress(
        session_id,
        phase=f"{idea.id}_evaluation",
        progress=base_progress + progress_window * 0.95,
        message=f"Idea Evaluation: Score {idea.evaluation.overall_score:.1f}/10",
        metadata={
            "parent_id": idea.node_id,  # ✅ Points to idea (Level 4) - makes evaluation Level 5
            "node_type": "evaluation",
            "title": "Idea Evaluation",
            "overall_score": idea.evaluation.overall_score,
            "novelty_score": idea.evaluation.novelty_score,
            "feasibility_score": idea.evaluation.feasibility_score,
            "impact_score": idea.evaluation.impact_score,
        },
        parent_phase=f"idea_{idea.id}",  # ✅ Correct
    )
```

## Implementation Strategy

### Phase 1: Store Intermediate Node IDs ✅
1. Update `_run_experiments_for_idea` to store `iteration_node_id`
2. Update `_run_experiments_for_idea` to store `sequential_plan_node_id`
3. Pass `iteration_node_id` to `_execute_sequential_plan_with_retries`
4. Pass `iteration_node_id` and `attempt_number` down the call chain

### Phase 2: Fix Parent References ✅
1. Update validation node creation (Change 2)
2. Update sequential plan node creation (Change 3)
3. Update attempt node creation (Change 5)
4. Update result node creation (Change 7)
5. Update hypothesis node type (Change 8)

### Phase 3: Add Missing Nodes ✅
1. Add evaluation node logging (Change 9)

### Phase 4: Update Node Types ✅
1. Change "step" → "iteration" for iteration nodes
2. Change "step" → "attempt" for attempt nodes
3. Change "result" → "hypothesis" for hypothesis nodes
4. Add new types: "validation_error", "plan", "evaluation"

### Phase 5: Test and Verify ✅
1. Run a simple scientific research query
2. Verify tree structure in frontend
3. Check that all nodes are at correct levels
4. Verify parent-child relationships are correct

## Testing Checklist

After implementation, verify:

- [ ] Level 4: Ideas appear directly under Ideation phase
- [ ] Level 5: Hypotheses appear under Ideas
- [ ] Level 5: Iterations appear under Ideas
- [ ] Level 6: Validation nodes appear under Iterations
- [ ] Level 6: Sequential Plan nodes appear under Iterations
- [ ] Level 6: Attempt nodes appear under Iterations
- [ ] Level 7: Data collection nodes appear under Attempts
- [ ] Level 7: Result nodes appear under Attempts
- [ ] Level 5: Evaluation nodes appear under Ideas
- [ ] Node types are semantically correct (iteration, attempt, hypothesis, etc.)
- [ ] Tree can expand/collapse properly in frontend
- [ ] No orphaned nodes
- [ ] Progress percentages are monotonically increasing down the tree

## Rollback Plan

If issues arise:
1. Changes are isolated to `scientific_research.py`
2. Can revert by using git: `git checkout backend/app/core/research_engines/scientific_research.py`
3. No database schema changes required
4. No API contract changes required

## Success Metrics

- Tree depth should be 7-8 levels for a typical research query
- Validation and Attempt nodes should be at Level 6 (not Level 4)
- Result nodes should be at Level 7 (children of attempts)
- Frontend visualization should show clear hierarchical structure
- Users can easily understand the research flow by looking at the tree

## Notes

- The frontend calculates tree levels based on parent_id relationships
- The `parent_phase` parameter is used for logical grouping but doesn't affect tree structure
- Node IDs must be stored and passed down to ensure proper parent-child relationships
- Consistent node_type values help frontend apply appropriate styling and icons
