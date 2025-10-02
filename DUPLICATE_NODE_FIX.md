# Duplicate Node Fix - SCIENTIFIC_RESEARCH vs scientific_research

## Problem Statement

At the beginning of a scientific research session, two duplicate nodes were being created in the tree:
1. `SCIENTIFIC_RESEARCH` (uppercase)
2. `scientific_research` (lowercase)

This caused visual clutter and confusion in the frontend tree visualization.

## Root Cause

**Inconsistent engine name casing** in `smart_router.py`:

- Line 158: `log_research_started` was called with `engine=classification_result.primary_engine` (uppercase "SCIENTIFIC_RESEARCH")
- Line 165: `log_research_progress` was called with `engine=classification_result.primary_engine.lower()` (lowercase "scientific_research")

Both calls created tree nodes with different engine identifiers, resulting in duplicate nodes.

## Solution

**File**: `backend/app/routers/smart_router.py` (Lines 155-162)

### Before
```python
await progress_tracker.log_research_started(
    session_id=session_id,
    request=request.user_request,
    engine=classification_result.primary_engine  # ❌ UPPERCASE "SCIENTIFIC_RESEARCH"
)

# Seed a minimal root node so late-joining clients can build a tree immediately
try:
    await progress_tracker.log_research_progress(
        session_id=session_id,
        engine=classification_result.primary_engine.lower(),  # ❌ lowercase "scientific_research"
        phase="session_initialized",
        ...
    )
```

### After
```python
# Use lowercase engine name consistently to avoid duplicate nodes
engine_name_lowercase = classification_result.primary_engine.lower()

await progress_tracker.log_research_started(
    session_id=session_id,
    request=request.user_request,
    engine=engine_name_lowercase  # ✅ Consistent lowercase
)

# Seed a minimal root node so late-joining clients can build a tree immediately
try:
    await progress_tracker.log_research_progress(
        session_id=session_id,
        engine=engine_name_lowercase,  # ✅ Consistent lowercase
        phase="session_initialized",
        ...
    )
```

## Changes Made

1. **Added variable**: `engine_name_lowercase = classification_result.primary_engine.lower()`
2. **Updated log_research_started**: Changed `engine=classification_result.primary_engine` to `engine=engine_name_lowercase`
3. **Updated log_research_progress**: Already used lowercase, now uses the same variable for consistency
4. **Added comment**: Explains why lowercase is used

## Impact

### Before Fix
```
Tree:
├── SCIENTIFIC_RESEARCH (uppercase) ❌
├── scientific_research (lowercase) ❌
    ├── Ideation
    ├── ...
```

### After Fix
```
Tree:
├── scientific_research (lowercase) ✅
    ├── Ideation
    ├── ...
```

## Consistency Check

Verified that all other calls to progress tracker in smart_router.py already use lowercase:
- Line 210: `engine="deep_research"` ✅
- Line 260: `engine="code_research"` ✅
- Line 366: `engine="scientific_research"` ✅

## Testing

### Manual Test
1. Start a new scientific research session
2. Check the tree visualization in frontend
3. Verify only ONE root node appears: `scientific_research`
4. No duplicate `SCIENTIFIC_RESEARCH` node

### Expected Behavior
- ✅ Single root node named `scientific_research`
- ✅ No duplicate uppercase node
- ✅ Tree structure is clean and hierarchical

## Files Modified

- `backend/app/routers/smart_router.py` (Lines 155-162)

## Rollback Plan

If issues occur:
```bash
git diff backend/app/routers/smart_router.py
git checkout backend/app/routers/smart_router.py
```

## Related Issues

- This fix complements the tree hierarchy fix (RESEARCH_TREE_HIERARCHY_FIX_IMPLEMENTATION.md)
- Both fixes together ensure a clean, properly structured tree visualization

## Success Metrics

- ✅ No duplicate nodes at tree root
- ✅ Consistent lowercase naming throughout
- ✅ Clean tree visualization
- ✅ No visual clutter

## Implementation Date

2025-10-02

## Priority

High - Affects user experience and tree visualization clarity
