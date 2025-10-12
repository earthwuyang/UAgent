# Research tree not displaying: Synchronization failure between session manager and research middleware

**Labels:** bug, critical, research-tree

---

## Problem Summary

The research tree visualization panel shows no data even when research experiments are running with nodes. Investigation reveals a critical synchronization issue where the session manager and research middleware use **different instances** of ResearchSessionManager.

### Latest Status (2025-10-12 @ 03:21 UTC)

**✅ Backend Working:**

* Experiments register successfully: "✅ Registered experiment … Active experiments: 1"
* Progress reports generated: "4 nodes"
* Orchestrator layer has experiment state

**❌ API/UI Broken:**

* API endpoints return empty data: `{"version": 0, "nodes": [], "edges": []}`
* Status endpoint shows: `{"status": "idle", "nodes": 0, "edges": 0}`
* UI displays: "Disconnected", "No research data available", "0/0 nodes"
* WebSocket connections fail: `net::ERR_CONNECTION_RESET`

**❌ Singleton Logs Missing:**

* No "✅ API using global ResearchSessionManager singleton"
* No "✅ Middleware using global ResearchSessionManager singleton"
* No "✅ WebSocket using global ResearchSessionManager singleton"

**Root Cause:** API/WebSocket layers are NOT calling `get_global_session_manager()`, using separate instances with no state.

---

## Root Cause Confirmed

**Multiple ResearchSessionManager instances** - middleware uses one instance for experiment registration, API uses different instance for queries. Result: experiments registered in middleware instance not visible to API.

## Solution Implemented ✅

Implemented thread-safe singleton pattern to ensure all components share same ResearchSessionManager instance.

### Changes Made:

1. **ResearchSessionManager Singleton** (`services/research_session_manager.py`)
   * Added module-level singleton variables with threading.Lock
   * Implemented `get_global_session_manager()` with double-check locking
   * Added `reset_global_session_manager()` for testing
   * Maintains backward compatibility
2. **Middleware Integration** (`middleware/research_middleware.py`)
   * Updated `get_session_manager()` to call `get_global_session_manager()`
   * Removed local instance creation
   * Logs: '✅ Middleware using global ResearchSessionManager singleton'
3. **API Integration** (`uagent_research/api/research_routes.py`)
   * Updated `get_session_manager()` to use singleton
   * All endpoints now query same instance
   * Logs: '✅ API using global ResearchSessionManager singleton'
4. **Registration Verification** (`middleware/research_middleware.py`)
   * Added verification after `session_mgr.register()` call
   * Checks if experiment_id in session_mgr.experiments
   * Raises RuntimeError on failure with detailed diagnostics
   * Cleans up partial state on registration failure
5. **Progress Sync** (`openhands/server/session/session.py`)
   * Added singleton access in progress reporting
   * Graceful handling if research extension not loaded

---

## ⚠️ CRITICAL ISSUE - Singleton Not Being Used

### WebSocket Connection Failures

**Browser console errors** (repeated):

```
net::ERR_CONNECTION_RESET on localhost ports: 50487, 57713
```

**Diagnosis:**

* WebSocket endpoints failing early (before handshake)
* Prevents progress events from reaching UI
* Likely caused by missing singleton init or dependency errors

### API Layer Isolation

**Direct API probes confirm empty state:**

```bash
GET /api/research/experiments/<exp_id>/tree
→ {"version": 0, "nodes": [], "edges": []}

GET /api/research/experiments/<exp_id>/status  
→ {"status": "idle", "nodes": 0, "edges": 0, "cost": 0.0}
```

**Analysis:**

* API queries an instance with **zero experiments**
* Backend registers experiments in a **different instance**
* Node updates never reach UI

---

## Reproduction History

### Test 1: 2025-10-12 @ 03:00 UTC

**Test Environment:**

* Server: Running on port 2999 in tmux session `uagent-backend`
* Started: Fresh server start with `.venv/bin/activate` and `start_openhands_research.sh`
* Conversation: d993ffe4cd0e4b14a6322d3b8f3d5f2c ("ML-Based Query Routing for Postgres and DuckDB")
* Experiment ID: `exp_d993ffe4cd0e4b14a6322d3b8f3d5f2c_1760237899_03f8d8`

**Evidence of Dual Instance Problem:**

**1. Backend Session Manager (HAS data):**

```
10:59:19 - openhands:INFO: session.py:730 - Research progress reported for experiment exp_d993ffe4cd0e4b14a6322d3b8f3d5f2c_1760237899_03f8d8: 4 nodes
```

✅ Experiment is successfully registered and reporting progress
✅ Session runtime can see and update experiment state
✅ Tree has 4 nodes of research data

**2. Frontend/API (NO data visible):**

* UI Shows: "Disconnected" status with orange indicator
* UI Shows: "No research data available"
* UI Shows: "0/0 nodes" in Experiment Progress
* Status: Idle (despite active experiment)

**3. Missing Singleton Logs:**

```bash
# Expected but NOT found in logs:
❌ "✅ Global ResearchSessionManager singleton created"
❌ "✅ Middleware using global ResearchSessionManager singleton" 
❌ "✅ API using global ResearchSessionManager singleton"
```

### Test 2: 2025-10-12 @ 03:21 UTC

**Test Environment:**

* Server: Port 2999, tmux session `uagent-backend`
* Conversation: `db47f586de6041f18056ddac85e7a016`
* Research goal: "hybrid OLAP query planners"
* Tool: Playwright MCP automation

**Results:**

* ✅ Backend startup succeeded
* ✅ Experiment registration logged: "Active experiments: 1"
* ❌ Singleton confirmation logs still missing
* ❌ UI Research Tree disconnected
* ❌ API returns empty payloads
* ❌ WebSocket ERR_CONNECTION_RESET errors

---

## Technical Analysis

### Why the Singleton Isn't Working

1. **Module Import Caching Issue:**
   * Python may have cached old bytecode before singleton was added
   * Even after server restart, if `.pyc` files exist, old code runs
   * Need to verify `.pyc` files were cleared from ALL relevant directories
2. **Multiple Import Paths:**
   * Session runtime imports from one path
   * Middleware imports from another path
   * If Python treats these as different modules, singleton won't work
   * Need to verify import paths are consistent
3. **Initialization Order:**
   * Singleton may be bypassed if instances created before singleton init
   * Need to ensure singleton is initialized early in app lifecycle
   * Check if any code creates ResearchSessionManager() directly
4. **Threading Issues:**
   * Session runtime runs in different thread than middleware
   * If singleton lock not working, might create multiple instances
   * Need to verify threading.Lock is functioning
5. **WebSocket Layer Not Instrumented:**
   * WebSocket handlers may not call `get_global_session_manager()`
   * May be creating instances directly
   * Connection failures suggest early initialization errors

### Code Locations to Verify

**Files that MUST use singleton:**

```python
# middleware/research_middleware.py
from ..services.research_session_manager import get_global_session_manager

def get_session_manager(...):
    return get_global_session_manager()  # ← MUST call this

# uagent_research/api/research_routes.py  
from ..services.research_session_manager import get_global_session_manager

def get_session_manager(...):
    return get_global_session_manager()  # ← MUST call this

# WebSocket handler (wherever Research Tree WS is handled)
from ..services.research_session_manager import get_global_session_manager

# When handling connections:
session_mgr = get_global_session_manager()  # ← MUST call this

# openhands/server/session/session.py
from extensions.uagent_research.services.research_session_manager import get_global_session_manager

# When reporting progress:
session_mgr = get_global_session_manager()  # ← MUST call this
```

---

## Debugging Action Plan

### 1. Add Debug Logging to Confirm Singleton Usage

**Add at module import time:**

```python
# At TOP of uagent_research/api/research_routes.py
import sys
print(f"[API DEBUG] Loading research_routes.py", file=sys.stderr, flush=True)

# In get_session_manager():
print(f"[API DEBUG] get_session_manager() called", file=sys.stderr, flush=True)
session_mgr = get_global_session_manager()
print(f"[API DEBUG] Instance: {id(session_mgr)}, experiments: {len(session_mgr.experiments)}", file=sys.stderr, flush=True)
```

**Repeat for WebSocket module:**

```python
# In WebSocket handler
print(f"[WS DEBUG] WebSocket handler initializing", file=sys.stderr, flush=True)
session_mgr = get_global_session_manager()
print(f"[WS DEBUG] Instance: {id(session_mgr)}, experiments: {len(session_mgr.experiments)}", file=sys.stderr, flush=True)
```

### 2. Trace WebSocket Handshake Failures

**Enable detailed logging:**

```python
import logging
logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
logging.getLogger("websockets").setLevel(logging.DEBUG)
```

**Add try/except blocks:**

* Wrap WebSocket endpoint handlers
* Log full stack traces on failures
* Check for missing dependencies

### 3. Validate Event Bus Wiring

```python
# In research_session_manager.py
def register(self, experiment_id, ...):
    logger.info(f"✅ After registration: {len(self.experiments)} total")
    logger.info(f"   Instance ID: {id(self)}")
    logger.info(f"   Experiments: {list(self.experiments.keys())}")

def _handle_event(self, event):
    logger.debug(f"Event: {event.type}, exp: {event.experiment_id}")
    logger.debug(f"   Instance has {len(self.experiments)} experiments")
```

### 4. Check for Direct Instantiation

```bash
cd OpenHands
grep -rn "ResearchSessionManager()" extensions/uagent_research/ \
  | grep -v "def __init__" \
  | grep -v "class ResearchSessionManager"
```

### 5. Verify Import Paths

```bash
grep -rn "from.*research_session_manager import" \
  extensions/uagent_research/ openhands/
```

### 6. Force Complete Cache Clear

```bash
find OpenHands -type f -name '*.pyc' -delete
find OpenHands -type d -name '__pycache__' -exec rm -rf {} +
find OpenHands/.venv -type f -name '*.pyc' -delete 2>/dev/null || true
```

### 7. Rerun Test with Full Logging

**After instrumentation:**

1. Restart server
2. Capture startup logs: `tmux capture-pane -t uagent-backend -p -S -1000`
3. Run Playwright test
4. **Verify:**
   * ✅ Singleton creation appears
   * ✅ API logs "using global singleton"
   * ✅ WebSocket logs "using global singleton"
   * ✅ All show **same instance ID**
   * ✅ `/status` returns data
   * ✅ WebSocket connects (no ERR_CONNECTION_RESET)

---

## Expected Behavior After Fix

✅ Singleton creation log appears once on server start
✅ All components log "using global singleton" with **same instance ID**
✅ Only ONE instance exists across entire application
✅ Experiments registered in middleware visible to API immediately
✅ Research Tree shows real-time data from active experiments
✅ UI connection status shows "Connected" (green)
✅ Progress updates flow from backend to frontend
✅ Diagnostics endpoint shows accurate experiment count
✅ WebSocket connections establish successfully
✅ API endpoints return populated tree data

---

## Files Modified

* `extensions/uagent_research/services/research_session_manager.py`
* `extensions/uagent_research/middleware/research_middleware.py`
* `extensions/uagent_research/uagent_research/api/research_routes.py`
* `openhands/server/session/session.py`
* **TODO:** WebSocket handler module (needs investigation)

---

## Environment

* **OS**: MacOS
* **Shell**: Zsh 5.9
* **Working Directory**: `/Users/wuy/Desktop/code/UAgent`
* **Python**: 3.12 (from .venv)
* **Server Port**: 2999
* **Tmux Session**: `uagent-backend`
* **Test Tools**: Puppeteer MCP, Playwright MCP

---

## Priority

**CRITICAL** - This blocks all research tree functionality. WebSocket failures prevent any real-time updates to UI.

---

## Related

* Linear Issue: [UAG-18](https://linear.app/uagent-ai/issue/UAG-18/research-tree-not-displaying-synchronization-failure-between-session)
* Project: OpenHands-UAgent
* Status: In Progress
* Created: 2025-10-11T14:59:26.909Z
* Updated: 2025-10-12T03:23:58.003Z
