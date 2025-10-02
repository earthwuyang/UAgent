# Research Parallelism Configuration

## Overview
The UAgent system now supports environment variable configuration to control research parallelism, making it easier to debug by limiting parallel operations to sequential execution.

## Environment Variables

### Configuration in `.env`
```bash
# Research execution configurations
MAX_RESEARCH_IDEAS=1                   # Max number of research ideas to generate (debug: 1, production: 3-5)
MAX_PARALLEL_IDEAS=1                   # Max number of ideas to process in parallel (debug: 1, production: 2-4)
EXPERIMENTS_PER_HYPOTHESIS=1           # Number of experiments per hypothesis (debug: 1, production: 2-3)
```

## Implementation Details

### Files Modified

1. **`.env`** (lines 17-20)
   - Added three new environment variables for parallelism control
   - All set to 1 for sequential debugging

2. **`backend/app/core/research_engines/scientific_research.py`**
   - Line 1458: Load `EXPERIMENTS_PER_HYPOTHESIS` from environment
   - Lines 1462-1468: Log parallelism settings on initialization
   - Lines 1950-1953: Load `MAX_PARALLEL_IDEAS` with logging
   - Lines 1959-1960: Load `MAX_RESEARCH_IDEAS` with logging
   - Line 2258: Log experiments per hypothesis during execution

### How It Works

1. **Idea Generation**:
   - Limited to 1 idea with `MAX_RESEARCH_IDEAS=1`
   - Logged when ideas are generated

2. **Parallel Processing**:
   - Limited to sequential with `MAX_PARALLEL_IDEAS=1`
   - Affects asyncio Semaphore for idea processing
   - Logged when processing begins

3. **Experiment Rounds**:
   - Limited to 1 experiment per hypothesis with `EXPERIMENTS_PER_HYPOTHESIS=1`
   - Reduces total experiment count
   - Logged for each hypothesis

### Verification

Run the verification script to confirm settings:
```bash
python verify_env_vars.py
```

Expected output:
```
✅ MAX_RESEARCH_IDEAS = 1 (correct)
✅ MAX_PARALLEL_IDEAS = 1 (correct)
✅ EXPERIMENTS_PER_HYPOTHESIS = 1 (correct)
🎉 All parallelism settings are correctly set to 1 for debugging!
```

## Benefits for Debugging

1. **Sequential Execution**: All operations run one at a time
2. **Cleaner Logs**: No interleaved output from parallel operations
3. **Easier Tracing**: Can follow execution path linearly
4. **Reduced Complexity**: Eliminates concurrency issues
5. **Faster Debugging**: Only one idea/experiment to track

## Production Settings

For production deployment, update `.env`:
```bash
MAX_RESEARCH_IDEAS=5                   # Generate more diverse ideas
MAX_PARALLEL_IDEAS=4                   # Process ideas in parallel
EXPERIMENTS_PER_HYPOTHESIS=3           # More thorough testing
```

## Monitoring

The backend logs will show:
```
Research parallelism settings: experiments_per_hypothesis=1, MAX_PARALLEL_IDEAS=1, MAX_RESEARCH_IDEAS=1
Generating up to 1 research ideas (MAX_RESEARCH_IDEAS)
Processing 1 ideas with parallelism=1 (MAX_PARALLEL_IDEAS=1)
Running 1 experiments per hypothesis (EXPERIMENTS_PER_HYPOTHESIS)
```

## Testing

1. Start the backend:
```bash
python -m backend.app.main
```

2. Monitor logs for parallelism messages:
```bash
grep -i "parallelism\|MAX_" backend.log
```

3. Submit a research query and verify only 1 idea is generated and processed sequentially