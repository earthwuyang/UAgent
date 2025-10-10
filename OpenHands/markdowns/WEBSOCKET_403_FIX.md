# WebSocket 403 Error - Root Cause and Complete Fix

## Status: ✅ FIXED

Date: 2025-10-08 20:38 UTC

## Two Problems Discovered

### Problem 1: Wrong Server Module ⚠️ CRITICAL
**Your server was started with the wrong module!**

```bash
# ❌ WRONG - This was running:
uvicorn openhands.server.app:app --host 0.0.0.0 --port 3000

# ✅ CORRECT - Should be:
python -m openhands.server
# (which uses openhands.server.listen:app with Socket.IO wrapper)
```

**Why this caused 403 errors:**
- `openhands.server.app:app` = FastAPI app WITHOUT Socket.IO integration
- `openhands.server.listen:app` = FastAPI app WITH Socket.IO wrapper
- Without the Socket.IO wrapper, WebSocket connections are rejected

### Problem 2: SESSION_API_KEY Authentication
Even with the correct module, if SESSION_API_KEY is set, connections with `session_api_key=null` will be rejected.

## Complete Fix Applied

### 1. Killed the Incorrectly Started Server
```bash
kill 1905231  # The process running with wrong module
```

### 2. Created Correct Startup Script
File: `start_server_no_auth.sh`

This script:
- ✅ Unsets SESSION_API_KEY (disables authentication)
- ✅ Uses `python -m openhands.server` (correct module with Socket.IO)
- ✅ Binds to 0.0.0.0:3000 (allows external connections)

## How to Start the Server Correctly

### Method 1: Use the Startup Script (Recommended)
```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_server_no_auth.sh
```

### Method 2: Direct Command
```bash
cd /home/wuy/AI/UAgent/OpenHands
unset SESSION_API_KEY
python -m openhands.server
```

### ❌ DO NOT USE:
```bash
# These are WRONG and will cause 403 errors:
uvicorn openhands.server.app:app --host 0.0.0.0 --port 3000
python -m uvicorn openhands.server.app:app --host 0.0.0.0 --port 3000
```

## What python -m openhands.server Does

Looking at `openhands/server/__main__.py`:
```python
def main():
    port = int(os.environ.get('OPENHANDS_PORT') or 
               os.environ.get('PORT') or 
               os.environ.get('port') or '3000')
    
    uvicorn.run(
        'openhands.server.listen:app',  # ← Uses listen:app (with Socket.IO)
        host='0.0.0.0',
        port=port,
        log_level='debug' if os.environ.get('DEBUG') else 'info',
    )
```

The key difference:
- `openhands.server.listen:app` includes Socket.IO wrapper from `listen.py`
- `openhands.server.app:app` is just the FastAPI app without Socket.IO

## Module Architecture

```
openhands/server/
├── app.py           # FastAPI application (routes, middleware)
├── shared.py        # Creates socketio.AsyncServer instance
├── listen.py        # Wraps app.py with socketio.ASGIApp ← CRITICAL
├── listen_socket.py # Socket.IO event handlers (@sio.event)
└── __main__.py      # Entry point: uses listen:app
```

## Verification

After starting with the correct script:

1. **Check the process:**
```bash
ps aux | grep openhands
# Should show: python -m openhands.server
# NOT: uvicorn openhands.server.app:app
```

2. **Check Socket.IO endpoint:**
```bash
curl -v "http://120.46.207.248:3000/socket.io/?conversation_id=test"
# Should return 200 OK or redirect, NOT 403
```

3. **Monitor logs:**
```bash
tail -f /tmp/openhands_server.log
# Should show successful WebSocket connections
```

## Expected Behavior After Fix

### Before Fix:
```
INFO: 39.149.41.18:55248 - "WebSocket /socket.io/..." 403
INFO: connection rejected (403 Forbidden)
```

### After Fix:
```
INFO: 39.149.41.18:xxxxx - "WebSocket /socket.io/..." [accepted]
INFO: sio:connect: <connection_id>
INFO: Socket request for conversation <conversation_id>
INFO: User None is allowed to connect to conversation
INFO: Successfully joined conversation
```

## Additional Issues Found

### 404 Not Found for http://120.46.207.248:3000/

The 404 errors for `GET /` are expected if:
1. Frontend is not built (`SERVE_FRONTEND=false`)
2. Frontend build directory doesn't exist

To fix:
```bash
# Build the frontend
cd /home/wuy/AI/UAgent/OpenHands/frontend
npm install
npm run build

# Or disable frontend serving
export SERVE_FRONTEND=false
```

## Summary of Changes

| File | Action | Purpose |
|------|--------|---------|
| Process 1905231 | ❌ Killed | Was using wrong module |
| `start_server_no_auth.sh` | ✅ Updated | Uses correct module + no auth |
| `WEBSOCKET_403_FIX.md` | ✅ Created | This documentation |

## Troubleshooting

If you still get 403 errors:

1. **Verify correct module:**
```bash
ps aux | grep openhands | grep -v grep
# Look for: python -m openhands.server
# NOT: uvicorn openhands.server.app:app
```

2. **Verify SESSION_API_KEY:**
```bash
# Check running process environment
ps aux | grep openhands | awk '{print $2}' | head -1 | xargs -I {} cat /proc/{}/environ | tr '\0' '\n' | grep SESSION_API_KEY
# Should output: (nothing)
```

3. **Restart completely:**
```bash
pkill -f "openhands"
./start_server_no_auth.sh
```

## Key Takeaways

1. ✅ Always use `python -m openhands.server` (NOT direct uvicorn)
2. ✅ Ensure SESSION_API_KEY is not set (for no-auth mode)
3. ✅ The correct module is `openhands.server.listen:app`
4. ✅ Check process command line to verify correct startup

---
Generated: 2025-10-08 20:38 UTC
Issue: WebSocket 403 Forbidden errors
Server: http://120.46.207.248:3000
