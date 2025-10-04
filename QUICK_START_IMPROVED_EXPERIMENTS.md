# Quick Start: Running Improved Experiments

## What Changed?

UAgent now **rejects simulated experiments** and demands **real implementation**. The ML routing experiment that previously succeeded with fake data will now be rejected and retried until properly completed.

## How to Run the ML Routing Experiment Again

### Option 1: Run the Same Experiment (Will Now Be Stricter)

The system will automatically detect the previous simulation and demand real work:

```bash
# The exact same experiment prompt will now:
# 1. Detect simulation keywords in results
# 2. Reject fake data
# 3. Retry up to 5 times (instead of 2)
# 4. Demand actual ML model training and deployment
```

### Option 2: Verify the Validation Works

Test that simulation detection works:

```python
# The validator will now catch:
result = {
    "success": True,
    "analysis": {
        "limitations": ["Timing data was simulated for demonstration purposes"]
    }
}
# Result: REJECTED - "Simulation detected: found keywords ['simulated', 'demonstration']"
```

## Key Improvements

### 1. Simulation Detection (Automatic)
```
✅ REJECTS if final.json contains:
- "simulated", "simulation"
- "sample data", "demonstration"
- "random.uniform", "placeholder"
- "not performed", "mock", "fake"
```

### 2. ML-Specific Validation (Automatic)
```
✅ For ML experiments, REQUIRES evidence of:
- Model embedding in C/C++ code
- Routing mechanism implementation
- Real data collection
- Actual model training
```

### 3. More Retry Attempts
```
Before: 2-3 attempts → Often gives up too early
After:  5 attempts   → Sufficient for complex deployments
```

### 4. Better Feedback Loop
```
Attempt 1: Simulated → Rejected
Feedback: "Simulation detected: found keywords ['simulated']"

Attempt 2: Agent knows to avoid simulation
Feedback: "ML model was not actually trained - only designed"

Attempt 3: Agent trains model
Feedback: "Routing mechanism not implemented"

Attempt 4: Agent implements routing
Feedback: "Missing end-to-end tests"

Attempt 5: Complete ✅
```

## Expected Behavior

### Old System (Before Changes)
```
🤖 Agent: "I've simulated the experiment with random.uniform()"
📊 Result: final.json with success=true, simulated data
✅ System: "Experiment successful!" ← WRONG
```

### New System (After Changes)
```
🤖 Agent: "I've simulated the experiment with random.uniform()"
📊 Result: final.json with success=true, simulated data
❌ System: "REJECTED - Simulation detected: ['simulated', 'random.uniform']"
📝 Feedback: "Limitation indicates incomplete work: Timing data was simulated"

🔄 Retry Attempt 2/5...
🤖 Agent: "I'll report the blocker instead of simulating"
💬 Agent: "Cannot run initdb due to container restrictions"
📋 System: "Acknowledged - please propose solution or continue with workaround"

🔄 Retry Attempt 3/5...
🤖 Agent: Collects real data, trains model
📊 Result: final.json with real measurements
✅ System: "Experiment successful!" ← CORRECT
```

## Validation Checklist

The system now validates:

- [ ] No simulation keywords in final.json
- [ ] No "limitations" mentioning simulation/demonstration
- [ ] For ML experiments: Model embedding evidence in modifications_made
- [ ] For routing experiments: Routing mechanism implementation
- [ ] Real data collection (not random/synthetic)
- [ ] Actual ML model training (not just designed)
- [ ] End-to-end system deployment
- [ ] Measurable performance results

## Files Modified

✅ `backend/app/core/research_engines/scientific_research.py`
- Enhanced anti-simulation requirements
- Added keyword detection
- Stricter LLM validation
- Increased retry attempts

## Next Steps

1. **Run any ML/deployment experiment** - The system will now be much stricter
2. **Observe rejections** - Simulated results will be caught
3. **Watch retries** - Agent gets up to 5 attempts with specific feedback
4. **Verify completion** - Only real implementations are accepted

## Troubleshooting

### If Agent Still Simulates After 5 Attempts

Check that the agent is receiving the feedback:
```python
# In logs, look for:
"Previous errors to fix: [...'Simulation detected...']"
```

### If Valid Work is Rejected

The validator might be too strict. Check:
```python
# Review the rejection reason:
"LLM assessment rejection: [specific reason]"
```

If legitimate work is rejected, the keyword list may need refinement.

## Summary

**Before**: Agents could finish with simulated data → Marked as successful
**After**: Simulated data is detected and rejected → Agent must do real work

The system now ensures experiments are **actually completed** rather than **appearing complete**.
