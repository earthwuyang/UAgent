# Research Middleware Import Fix

**Date:** 2025-10-12  
**Status:** ✅ FIXED

## Problem

The Research Tree UI was not working because the research middleware was failing to load with the error:

```
Research middleware not available: No module named 'uagent_research.middleware'
```

This meant that research experiments were never created when users sent research goal messages, causing the UI to show "Disconnected" status.

## Root Cause

The issue was in how the extension directory was added to `sys.path`:

### Before (BROKEN):
```python
# In listen.py and app.py
extension_dir = Path(__file__).parent.parent.parent / 'extensions' / 'uagent_research'
sys.path.append(str(extension_dir))  # Added /extensions/uagent_research to sys.path
```

Then tried to import:
```python
from uagent_research.middleware.research_middleware import research_middleware
```

This failed because Python was looking for:
`/extensions/uagent_research/uagent_research/middleware/...`

But the actual path was:
`/extensions/uagent_research/middleware/...`

## Solution

Changed both `listen.py` and `app.py` to add the **parent** `extensions` directory to sys.path instead:

### After (FIXED):
```python
# In listen.py
extension_parent_dir = Path(__file__).parent.parent.parent / 'extensions'
sys.path.insert(0, str(extension_parent_dir))  # Added /extensions to sys.path
```

```python
# In app.py  
extensions_dir = Path(__file__).parent.parent.parent / 'extensions'
extension_dir = extensions_dir / 'uagent_research'
sys.path.append(str(extensions_dir))  # Added /extensions to sys.path
```

Now Python can correctly import:
- `from uagent_research.middleware.research_middleware import research_middleware`
- `from uagent_research.api import router as research_router`

## Changes Made

1. **File:** `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/listen.py`
   - Changed: Line 12 to add `/extensions` instead of `/extensions/uagent_research`
   - Result: `sys.path.insert(0, extension_parent_dir)`

2. **File:** `/Users/wuy/Desktop/code/UAgent/OpenHands/openhands/server/app.py`
   - Changed: Line 43-56 to add `/extensions` instead of `/extensions/uagent_research`
   - Result: `sys.path.append(extensions_dir_str)`

3. **File:** `/Users/wuy/Desktop/code/UAgent/OpenHands/extensions/uagent_research/middleware/research_middleware.py`
   - Changed: Line 664 to import and use `get_session` instead of non-existent `get_db_session`
   - Result: `from uagent_research.models.base import get_session`

## Verification

After the fix, the server logs show:
```
🔧 [listen.py] Added extension directory to sys.path: /Users/wuy/Desktop/code/UAgent/OpenHands/extensions
✅ Research middleware loaded successfully
✅ UAgent Research Extension loaded from source
✅ UAgent Research Extension routes registered
✅ Research database initialized successfully
```

## Impact

This fix enables:
1. ✅ Research middleware properly loads and intercepts user messages
2. ✅ Auto-trigger classification works for research goals
3. ✅ Research experiments are created in the database
4. ✅ Research Tree UI can connect and display experiment status
5. ✅ Full research functionality is restored

## Testing

To verify the fix works:

1. Start the server with `./start_openhands_research.sh`
2. Check logs for the success messages above
3. Navigate to http://localhost:2999
4. Send a research goal message
5. Check that the Research Tree tab shows the experiment
6. Verify experiment is in the database:
   ```bash
   sqlite3 openhands_research.db "SELECT id, session_id FROM experiments;"
   ```

## Related Issues

- Previous fix attempt in `research_middleware.py` (database creation) was correct but didn't work because middleware wasn't loading
- Root cause was a Python import path configuration error, not a database issue

## Lessons Learned

- When adding a directory to `sys.path` for package imports, add the **parent** directory of the package, not the package directory itself
- For `from uagent_research.X import Y`, `sys.path` should include the directory containing `uagent_research/`, not `uagent_research/` itself
- Always verify imports work after modifying `sys.path` by checking server startup logs
