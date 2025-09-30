# Sequential Experiment Execution Specification

## Overview

Fix the critical architectural bug where multiple experiments for a scientific research hypothesis are executed independently in separate OpenHands sessions/containers, preventing later experiments from building on previous results. This specification defines the changes needed to execute all experiments for a hypothesis sequentially in a single OpenHands session.

## Problem Statement

### Current Behavior (BROKEN)
```
Scientific Goal → Split into N experiments
↓
Experiment 1 → OpenHands Session 1 (Docker Container 1) → Result 1
Experiment 2 → OpenHands Session 2 (Docker Container 2) → Result 2 ❌ Cannot access Result 1
Experiment 3 → OpenHands Session 3 (Docker Container 3) → Result 3 ❌ Cannot access Result 1 or 2
```

**Issues:**
1. Each experiment starts from a clean slate in a new container
2. Later experiments cannot access code, data, or artifacts from previous experiments
3. The experimental progression is broken - no iterative refinement possible
4. Violates scientific research principles where experiments build upon each other

### Desired Behavior (FIXED)
```
Scientific Goal → Design all N experiments as a sequential research plan
↓
All Experiments → Single OpenHands Session (Same Docker Container)
  ├─ Experiment 1 → Result 1
  ├─ Experiment 2 (builds on Result 1) → Result 2
  └─ Experiment 3 (builds on Result 1 & 2) → Result 3
```

**Benefits:**
1. Shared workspace and environment across all experiments
2. Later experiments can analyze/refine previous results
3. True iterative scientific research workflow
4. Efficient resource usage (single container vs multiple)

## Current Architecture Analysis

### File: `backend/app/core/research_engines/scientific_research.py`

#### Current Flow
1. **Line 2391-2424**: `_run_experiments_for_idea()` loops through hypotheses and experiment rounds
2. **Line 2394**: `design_experiment()` designs ONE experiment at a time
3. **Line 2414**: `_execute_experiment_with_retries()` executes ONE experiment at a time
4. **Line 2520**: `execute_experiment()` executes in potentially separate OpenHands sessions

#### Code Location (Lines 2391-2424)
```python
for hypothesis in pending_hypotheses:
    for exp_round in range(1, self.experiments_per_hypothesis + 1):
        # ❌ BUG: Design one experiment at a time
        design = await self.experiment_designer.design_experiment(hypothesis)

        # ❌ BUG: Execute one experiment at a time in separate session
        final_execution, execution_history = await self._execute_experiment_with_retries(
            idea, hypothesis, design, session_id, ...
        )
```

## Requirements

### Functional Requirements

#### FR1: Consolidated Experiment Design
- **Input**: Research hypothesis + number of sequential experiments needed
- **Output**: Complete sequential experimental plan containing all experiment designs
- **Behavior**: LLM designs ALL experiments together as a cohesive research plan
- **Context**: Each experiment design includes awareness of previous experiment objectives

#### FR2: Single-Session Sequential Execution
- **Input**: Sequential experimental plan + OpenHands session context
- **Output**: List of all experiment executions with shared artifacts
- **Behavior**: Execute all experiments in the SAME OpenHands session/container
- **Persistence**: Workspace state persists between experiments

#### FR3: Inter-Experiment Context Propagation
- **Input**: Results from previous experiments
- **Output**: Enhanced context for subsequent experiments
- **Behavior**: Each experiment receives:
  - Previous experiment results
  - Previous experiment artifacts/code
  - Previous experiment data files
  - Previous experiment analysis

#### FR4: Progress Tracking Updates
- **Input**: Sequential execution progress
- **Output**: WebSocket progress events for all experiments
- **Behavior**: Track progress for entire sequential execution, not individual experiments

### Non-Functional Requirements

#### NFR1: Performance
- **Single Container**: Reuse the same Docker container for all experiments
- **No Overhead**: Eliminate container startup/teardown overhead between experiments
- **Resource Efficiency**: Share computational resources across experiments

#### NFR2: Reliability
- **Atomic Failure**: If one experiment fails, preserve results from previous experiments
- **Error Propagation**: Pass error context to subsequent experiment attempts
- **Rollback Safety**: Ability to inspect workspace state at any experiment boundary

#### NFR3: Observability
- **Clear Logging**: Log which experiment is executing in the sequential plan
- **Progress Granularity**: Track progress within and across experiments
- **Artifact Tracing**: Track which artifacts were created by which experiment

## Interface

### New Method: `design_sequential_experiments()`

```python
async def design_sequential_experiments(
    self,
    hypothesis: ResearchHypothesis,
    num_experiments: int,
    resources: Optional[Dict[str, Any]] = None,
) -> SequentialExperimentPlan:
    """
    Design a complete sequential experimental plan for hypothesis testing.

    Args:
        hypothesis: Research hypothesis to test
        num_experiments: Number of sequential experiments to design
        resources: Available computational resources

    Returns:
        SequentialExperimentPlan containing all experiment designs
    """
```

#### Input Schema
```python
@dataclass
class SequentialExperimentPlan:
    """Complete plan for sequential experiments"""
    id: str
    hypothesis_id: str
    num_experiments: int
    experiments: List[ExperimentDesign]  # Ordered list of experiments
    overall_objective: str
    experiment_dependencies: Dict[str, List[str]]  # Which experiments depend on which
    shared_setup: str  # Common setup code for all experiments
    expected_total_duration: str
```

### Modified Method: `execute_sequential_experiments()`

```python
async def execute_sequential_experiments(
    self,
    plan: SequentialExperimentPlan,
    session_context: OpenHandsSessionContext,
    prior_errors: Optional[List[str]] = None,
    attempt_number: int = 1,
    attempt_context: Optional[Dict[str, Any]] = None,
) -> List[ExperimentExecution]:
    """
    Execute all experiments in the plan sequentially in a single OpenHands session.

    Args:
        plan: Sequential experimental plan
        session_context: OpenHands session to use for ALL experiments
        prior_errors: Errors from previous attempt (if retrying)
        attempt_number: Current attempt number
        attempt_context: Progress tracking context

    Returns:
        List of all experiment executions in order
    """
```

#### Execution Flow
```
1. Shared Setup Phase (once)
   ├─ Create workspace directory structure
   ├─ Install common dependencies
   └─ Setup data directories

2. For each experiment in plan:
   ├─ Phase A: Prepare experiment-specific environment
   │   └─ Load results from previous experiments
   ├─ Phase B: Execute experiment code
   │   └─ Generate and run experiment script
   ├─ Phase C: Collect and save results
   │   └─ Store results for next experiment
   └─ Phase D: Update progress tracking

3. Final Analysis Phase
   ├─ Aggregate results from all experiments
   └─ Generate comprehensive analysis
```

## Behavior

### Normal Operation Flow

#### Phase 1: Consolidated Design (Lines 2391-2412)
```python
# NEW: Design all experiments together
sequential_plan = await self.experiment_designer.design_sequential_experiments(
    hypothesis=hypothesis,
    num_experiments=self.experiments_per_hypothesis,
    resources=resources,
)
idea.experimental_plans.append(sequential_plan)

# Log consolidated design
await self._log_progress(
    session_id,
    phase=f"{idea.id}_sequential_plan_{sequential_plan.id}",
    message=f"Designed sequential plan: {sequential_plan.overall_objective}",
    metadata={
        "num_experiments": len(sequential_plan.experiments),
        "experiment_names": [exp.name for exp in sequential_plan.experiments],
    },
)
```

#### Phase 2: Sequential Execution (Lines 2414-2534)
```python
# NEW: Execute all experiments in one session
all_executions = await self._execute_sequential_plan_with_retries(
    idea=idea,
    hypothesis=hypothesis,
    plan=sequential_plan,
    session_id=session_id,
    session_context=openhands_context,
    iteration_number=iteration,
)

# Process results from all experiments
for execution in all_executions:
    if execution.status == ExperimentStatus.COMPLETED:
        experiment_result = await self._analyze_experiment_result(
            hypothesis,
            execution.design,  # Design from the plan
            execution,
        )
        idea.results.append(experiment_result)
```

### Edge Cases

#### EC1: Mid-Sequence Failure
**Scenario**: Experiment 2 fails in a 3-experiment sequence

**Handling**:
```python
# Preserve results from Experiment 1
successful_executions = executions[:1]  # Experiment 1 succeeded
failed_execution = executions[1]  # Experiment 2 failed
remaining_executions = executions[2:]  # Experiment 3 not attempted

# On retry, skip successful experiments
retry_plan = SequentialExperimentPlan(
    experiments=plan.experiments[1:],  # Start from failed experiment
    prior_results=successful_executions,  # Pass previous results
)
```

#### EC2: Container Crash
**Scenario**: Docker container crashes mid-execution

**Handling**:
```python
# Detect container crash
if execution_error.type == "container_crash":
    # Restart container with same workspace
    new_session = await openhands_client.restore_session(
        workspace_id=session_context.workspace_id,
        checkpoint=last_successful_experiment,
    )

    # Resume from last checkpoint
    remaining_plan = create_remaining_plan(plan, checkpoint_index)
```

#### EC3: Resource Exhaustion
**Scenario**: Experiments consume too much disk/memory

**Handling**:
```python
# Monitor resource usage
if workspace_size > MAX_WORKSPACE_SIZE:
    # Cleanup between experiments
    await cleanup_intermediate_artifacts(experiment_index)

# Add resource limits to plan
plan.resource_limits = {
    "max_disk_per_experiment": "10GB",
    "max_memory": "16GB",
}
```

### Error Handling

#### Error Propagation
```python
# When experiment fails, pass error to next attempt
prior_errors = [
    f"Experiment {i+1} ({exp.name}): {execution.errors}"
    for i, (exp, execution) in enumerate(zip(plan.experiments, executions))
    if execution.status == ExperimentStatus.FAILED
]

# Retry with error context
await execute_sequential_experiments(
    plan=plan,
    session_context=session_context,
    prior_errors=prior_errors,  # LLM can learn from these
    attempt_number=attempt + 1,
)
```

## Testing

### Test Scenarios

#### TS1: Sequential Design Coherence
```python
async def test_sequential_design_coherence():
    """Verify that designed experiments form a coherent sequence"""
    hypothesis = create_test_hypothesis()

    plan = await designer.design_sequential_experiments(
        hypothesis=hypothesis,
        num_experiments=3,
    )

    # Assert experiments reference each other
    assert "previous experiment" in plan.experiments[1].description.lower()
    assert "building on" in plan.experiments[2].description.lower()

    # Assert dependency tracking
    assert plan.experiment_dependencies["exp_2"] == ["exp_1"]
    assert plan.experiment_dependencies["exp_3"] == ["exp_1", "exp_2"]
```

#### TS2: Single Container Execution
```python
async def test_single_container_execution():
    """Verify all experiments run in same container"""
    plan = create_test_sequential_plan(num_experiments=3)
    session_context = create_test_session()

    executions = await executor.execute_sequential_experiments(
        plan=plan,
        session_context=session_context,
    )

    # Assert same workspace for all
    workspace_ids = [exec.workspace_id for exec in executions]
    assert len(set(workspace_ids)) == 1, "All experiments must share workspace"

    # Assert same session
    session_ids = [exec.session_id for exec in executions]
    assert len(set(session_ids)) == 1, "All experiments must share session"
```

#### TS3: Context Propagation
```python
async def test_context_propagation():
    """Verify later experiments can access previous results"""
    plan = SequentialExperimentPlan(
        experiments=[
            ExperimentDesign(name="Create dataset", ...),
            ExperimentDesign(name="Analyze dataset", ...),  # Needs dataset from exp 1
        ]
    )

    executions = await executor.execute_sequential_experiments(plan, session)

    # Verify experiment 2 accessed experiment 1's output
    exp2_logs = executions[1].logs
    assert any("dataset" in log.lower() for log in exp2_logs)
    assert executions[1].output_data.get("dataset_source") == "experiment_1"
```

#### TS4: Mid-Sequence Recovery
```python
async def test_mid_sequence_recovery():
    """Verify recovery from mid-sequence failures"""
    plan = create_test_sequential_plan(num_experiments=3)

    # Inject failure in experiment 2
    with mock_failure(experiment_index=1):
        executions = await executor.execute_sequential_experiments(plan, session)

    # Assert experiment 1 succeeded
    assert executions[0].status == ExperimentStatus.COMPLETED

    # Assert experiment 2 failed
    assert executions[1].status == ExperimentStatus.FAILED

    # Assert experiment 3 was not attempted
    assert len(executions) == 2  # Only 2 executions, not 3
```

### Success Criteria

#### SC1: Design Quality
- [ ] Sequential plan contains correct number of experiments
- [ ] Each experiment builds logically on previous ones
- [ ] Dependencies are correctly tracked
- [ ] Shared setup code is present

#### SC2: Execution Correctness
- [ ] All experiments execute in same OpenHands session
- [ ] All experiments share the same workspace directory
- [ ] Later experiments can access previous experiment artifacts
- [ ] Execution order matches design order

#### SC3: Progress Tracking
- [ ] WebSocket events emitted for entire sequential execution
- [ ] Progress increases monotonically through all experiments
- [ ] Final completion event marks all experiments complete
- [ ] Individual experiment boundaries are visible in UI

#### SC4: Error Recovery
- [ ] Failed experiments don't affect previous successful results
- [ ] Error context is passed to retry attempts
- [ ] Container crashes are handled gracefully
- [ ] Resource exhaustion is detected and handled

### Performance Benchmarks

| Metric | Current (Broken) | Target (Fixed) | Measurement |
|--------|------------------|----------------|-------------|
| Container startups per 3-exp sequence | 3 | 1 | Docker stats |
| Total execution time (3 experiments) | ~90 min | ~45 min | End-to-end timer |
| Disk I/O overhead | High | Low | iostat during execution |
| Context sharing success rate | 0% | 100% | Artifact accessibility test |
| Resource utilization efficiency | ~33% | ~90% | Docker resource monitoring |

## Implementation Plan

### Phase 1: Add Sequential Design Method
**File**: `backend/app/core/research_engines/scientific_research.py`

1. Add `SequentialExperimentPlan` dataclass (after line 300)
2. Implement `ExperimentDesigner.design_sequential_experiments()` (after line 634)
3. Update LLM prompt to design sequential plans (modify `design_experiment` prompt)

### Phase 2: Add Sequential Execution Method
**File**: `backend/app/core/research_engines/scientific_research.py`

1. Implement `ExperimentExecutor.execute_sequential_experiments()` (after line 969)
2. Add shared setup phase
3. Add inter-experiment context propagation
4. Add progress tracking for sequential execution

### Phase 3: Update Main Research Loop
**File**: `backend/app/core/research_engines/scientific_research.py` (Lines 2391-2534)

1. Replace individual `design_experiment()` with `design_sequential_experiments()`
2. Replace individual `execute_experiment()` with `execute_sequential_experiments()`
3. Update progress tracking to handle sequential plans
4. Update result processing to handle sequential executions

### Phase 4: Update Progress Tracking
**File**: `backend/app/core/websocket_manager.py`

1. Add events for sequential plan design
2. Add events for sequential execution progress
3. Add events for inter-experiment transitions

### Phase 5: Testing
**Files**: `backend/test/unit/test_scientific_research.py`, `backend/test/integration/test_workflows.py`

1. Add unit tests for sequential design
2. Add unit tests for sequential execution
3. Add integration tests for full workflow
4. Add end-to-end tests with real OpenHands

## Migration Strategy

### Backward Compatibility
- Keep existing `design_experiment()` and `execute_experiment()` methods for single experiments
- Add feature flag: `UAGENT_USE_SEQUENTIAL_EXPERIMENTS=1` (default enabled)
- Allow fallback to old behavior if flag is disabled

### Rollout Plan
1. **Development**: Test with `EXPERIMENTS_PER_HYPOTHESIS=2` (small sequences)
2. **Staging**: Test with `EXPERIMENTS_PER_HYPOTHESIS=3` (medium sequences)
3. **Production**: Enable for all users after validation

### Monitoring
- Log successful sequential executions vs failures
- Track container reuse rate (should be ~100%)
- Monitor workspace size growth
- Alert on sequential execution failures

## Open Questions

1. **Workspace Cleanup**: When should we cleanup intermediate artifacts between experiments?
   - **Proposal**: Add optional cleanup phase between experiments if disk usage exceeds threshold

2. **Checkpoint Granularity**: Should we checkpoint after each experiment for crash recovery?
   - **Proposal**: Save workspace snapshot after each successful experiment

3. **Parallel Experiments**: Should we support parallel execution of independent experiments?
   - **Proposal**: Not in Phase 1; add in Phase 2 if dependency graph allows

4. **Maximum Sequence Length**: What's the reasonable limit for sequential experiments?
   - **Proposal**: Cap at 5 experiments per sequence to avoid excessive runtime

## References

- Current implementation: `backend/app/core/research_engines/scientific_research.py:2391-2534`
- OpenHands integration: `backend/app/integrations/openhands_single_container.py`
- Progress tracking: `backend/app/core/websocket_manager.py`
- Experiment manager: `backend/app/core/experiment_manager.py`
