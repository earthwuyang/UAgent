# Quick Summary: Experiment Resume Fix

## The Problem
Experiments restart from scratch on every retry, wasting hours rebuilding source code.

## The Root Cause
Each retry creates a **new Docker container**, and files weren't persisting to the mounted `/workspace` directory.

## The Fix

### 1. Added Container Labels
Track containers by session name to find previous attempts.

### 2. Clean Up Old Container Before New One
Find and remove the old container, but **keep the workspace files** on the host.

### 3. Mount Same Workspace to New Container
The new container sees all files from the previous attempt because they're in the mounted host directory.

### 4. Tell Agent About Persistence
Added explicit warning:
```
⚠️ WORK IN /workspace FOR PERSISTENCE
Files persist across retries - check if work exists before redoing!
```

## Files Changed

**`backend/app/integrations/openhands_single_container.py`**:
- Added `_find_existing_container()` method (line 59)
- Added container labels (line 593)
- Added cleanup logic before creating new container (line 604)
- Added workspace persistence warning to agent prompt (line 254)

## Result

**Before**: Attempt 1 (40 min timeout), Attempt 2 (40 min timeout), Attempt 3 (40 min timeout)...
**After**: Attempt 1 (40 min partial), Attempt 2 (30 min to complete) = ✅ SUCCESS in 70 min total

## Test It

1. Start an experiment that builds source code
2. Let it timeout mid-build
3. Check host directory - source files should be there
4. Retry should see existing files and continue

## Documentation

- **Full technical details**: `EXPERIMENT_RESUME_FIX_SUMMARY.md`
- **Workspace persistence issue**: `WORKSPACE_PERSISTENCE_FIX.md`
- **Validation improvements**: `EXPERIMENT_COMPLETION_IMPROVEMENTS.md`
