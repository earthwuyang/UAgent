# JSON Import Error Fix

## Problem
Error message: `cannot access local variable 'json' where it is not associated with a value`

## Root Cause
In `backend/app/integrations/openhands_bridge.py`, there was a redundant `import json` statement inside a try block (line 150), while `json` was already imported at the top of the file (line 5). This caused a scope issue where the local import could fail but still be referenced.

## Solution

### Before (Incorrect)
```python
if not isinstance(parsed, dict):
    import re
    json_match = re.search(r'\{[\s\S]*', response)
    if json_match:
        try:
            import json  # PROBLEMATIC: Redundant local import
            parsed = json.loads(json_match.group(0) + ']}')
        except:
            pass
```

### After (Fixed)
```python
if not isinstance(parsed, dict):
    json_match = re.search(r'\{[\s\S]*', response)
    if json_match:
        try:
            # json is already imported at the top of the file
            parsed = json.loads(json_match.group(0) + ']}')
        except (json.JSONDecodeError, ValueError) as e:
            # Log the attempt but continue with the error
            import logging
            logging.debug(f"Failed to repair truncated JSON: {e}")
```

## Changes Made

1. **Removed redundant import**: Deleted `import json` from line 150
2. **Used module-level import**: Referenced the `json` module imported at line 5
3. **Improved error handling**: Added specific exception types and logging
4. **Added comment**: Clarified that json is imported at the top

## File Modified
- `/backend/app/integrations/openhands_bridge.py` (lines 144-157)

## Impact
✅ Fixes the "cannot access local variable 'json'" error
✅ Cleaner code without redundant imports
✅ Better error handling with specific exceptions
✅ Debugging support with logging

## Testing
```bash
# Verify syntax
python -m py_compile backend/app/integrations/openhands_bridge.py

# Run the backend to test
python -m backend.app.main
```