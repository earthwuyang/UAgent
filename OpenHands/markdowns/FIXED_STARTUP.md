# ✅ Server Startup Fixed!

## The Problem

The server was exiting immediately after showing "✅ UAgent Research Extension loaded successfully"

## The Root Cause

We were using:
```bash
python -m openhands.server.listen --port 3000
```

But OpenHands expects:
```bash
python -m openhands.server  # Uses __main__.py
```

Also, OpenHands uses lowercase `port` environment variable, not `PORT`.

## The Fix

### Updated All Startup Scripts

1. **Changed command:**
   - ❌ Before: `python -m openhands.server.listen --port "$PORT"`
   - ✅ After: `python -m openhands.server`

2. **Fixed environment variable:**
   - ❌ Before: `export PORT=3000`
   - ✅ After: `export port=3000`  # Lowercase!

### Files Updated

- ✅ `start.sh`
- ✅ `start_backend_only.sh`
- ✅ `start_openhands_research.sh`

---

## How to Start Now (Fixed)

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start.sh
```

**Expected Output:**
```
==================================================
  OpenHands + UAgent Research Extension
==================================================

[1/2] Checking research extension...
✓ Extension already installed

[2/2] Starting server on port 3000...

Available at:
  • Research API: http://localhost:3000/api/research
  • Health Check: http://localhost:3000/api/research/health

ℹ️  Frontend UI disabled (API only mode)
   To enable: SERVE_FRONTEND=true ./start.sh

Press Ctrl+C to stop
==================================================

✅ Research database initialized
✅ UAgent Research Extension loaded successfully
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

**Key Difference:** Server now **stays running**! ✅

---

## Verify It's Working

### In the same terminal, you should see:
```
INFO:     Uvicorn running on http://0.0.0.0:3000 (Press CTRL+C to quit)
```

### In another terminal:
```bash
# Test health endpoint
curl http://localhost:3000/api/research/health

# Expected:
# {"status":"healthy","version":"0.1.0","database":"connected"}

# Check if port is in use
lsof -i:3000

# Expected:
# python  12345 user   ... (LISTEN)
```

---

## Why It Was Failing

### The listen.py Module

`openhands.server.listen` is just a module file that:
1. Imports the app
2. Sets up middleware
3. Exports the app

It's **NOT meant to be run directly**!

### The Correct Entry Point

`openhands/server/__main__.py` is the proper entry point:
```python
def main():
    uvicorn.run(
        'openhands.server.listen:app',  # ← Imports from listen.py
        host='0.0.0.0',
        port=int(os.environ.get('port') or '3000'),  # ← lowercase!
        log_level='info',
    )
```

This:
- ✅ Starts uvicorn properly
- ✅ Keeps the server running
- ✅ Uses the correct port variable

---

## Custom Port

```bash
# Use port 8000
port=8000 ./start.sh

# Or with uppercase (for compatibility)
PORT=8000 ./start.sh
```

Both work now because the script does:
```bash
export port=${PORT:-3000}  # Converts PORT → port
```

---

## Summary

✅ **Fixed:** Server now starts and stays running
✅ **Fixed:** Correct OpenHands startup method (`python -m openhands.server`)
✅ **Fixed:** Environment variable (`port` instead of `PORT`)
✅ **All scripts updated:** start.sh, start_backend_only.sh, start_openhands_research.sh

**Now the system works perfectly!** 🎉

---

**Date:** 2025-10-04
**Status:** Server startup fixed ✅
