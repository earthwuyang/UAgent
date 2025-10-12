# Research Tree WebSocket and API Connection Fix Summary

**Date:** 2025-10-12  
**Status:** ✅ FIXED

## Problem Summary

The Research Tree visualization in the OpenHands frontend was not displaying any data and showing "Disconnected" status due to two critical issues:

1. **WebSocket Connection Issue**: Frontend WebSocket was connecting to wrong port (localhost:3001 instead of localhost:2999)
2. **Tree API Endpoint Issue**: Frontend was fetching tree data from frontend port instead of backend port

## Root Causes

### Issue 1: WebSocket Connecting to Wrong Port

**Location**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/hooks/useResearchWS.ts` (line 115)

**Problem**: The WebSocket hook was using `window.location.host` as fallback, which resolved to `localhost:3001` (frontend) instead of `localhost:2999` (backend).

```typescript
// Before (WRONG):
const backendHost = import.meta.env.VITE_BACKEND_BASE_URL || window.location.host;
const wsUrl = `${protocol}//${backendHost}/api/research/ws/experiment/${experimentId}`;
// Result: ws://localhost:3001/api/research/ws/experiment/... ❌
```

**Root Cause**: The `VITE_BACKEND_BASE_URL` environment variable was not being set during frontend build, so it always fell back to `window.location.host`.

### Issue 2: Tree API Fetching from Wrong URL

**Location**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes/research-tab.tsx` (line 91-92)

**Problem**: The tree fetch was using a relative URL, which resolved to the frontend host instead of the backend.

```typescript
// Before (WRONG):
const response = await fetch(`/api/research/experiments/${experimentId}/tree`);
// Result: http://localhost:3001/api/research/experiments/.../tree ❌
```

## Solutions Implemented

### Fix 1: Configure Backend URL Environment Variable

**File Created**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/.env`

```bash
VITE_BACKEND_BASE_URL=localhost:2999
```

**Effect**: Vite now injects this value at build time, making it available to the frontend code via `import.meta.env.VITE_BACKEND_BASE_URL`.

### Fix 2: Update Tree Fetch to Use Backend URL

**File Modified**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes/research-tab.tsx`

**Change**:
```typescript
// After (CORRECT):
const backendHost = import.meta.env.VITE_BACKEND_BASE_URL || window.location.host;
const response = await fetch(
  `${window.location.protocol}//${backendHost}/api/research/experiments/${experimentId}/tree`,
);
// Result: http://localhost:2999/api/research/experiments/.../tree ✅
```

## Verification Steps

### 1. WebSocket Connection Test
```bash
# Frontend console should show:
[Research WS] Connecting to ws://localhost:2999/api/research/ws/experiment/...
[Research WS] Connected
```

### 2. Backend Routes Verification
```bash
curl http://localhost:2999/api/research/health
```

Expected backend routes include:
- `/api/research/experiments/{experiment_id}/tree` ✅
- `/api/research/ws/experiment/{experiment_id}` ✅
- `/api/research/experiments/{experiment_id}/status` ✅

### 3. Research Tree UI Status
- Status should show: **"Connected"** (green indicator)
- WebSocket establishes connection to port 2999
- Tree data fetches from correct backend endpoint

## Why No Tree Data Appears (Expected Behavior)

Even after fixing the connection issues, the Research Tree may show "No research data available" because:

1. **No Active Research Experiment**: The conversation hasn't started a research experiment yet
2. **Research Mode Not Triggered**: Research mode requires sending a message starting with "research goal:"
3. **Empty Tree is Valid**: Until an experiment runs, an empty tree is the correct state

### How to Start a Research Experiment

Send a message in the chat interface:
```
research goal: [your research objective]
```

Example:
```
research goal: Investigate ML-based query routing between PostgreSQL and DuckDB engines
```

## Files Modified

1. **Created**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/.env`
   - Added `VITE_BACKEND_BASE_URL=localhost:2999`

2. **Modified**: `/Users/wuy/Desktop/code/UAgent/OpenHands/frontend/src/routes/research-tab.tsx`
   - Line 91-93: Added backend host resolution for tree fetch
   - Changed relative URL to absolute URL with backend host

## Backend Configuration

The backend correctly registers research routes on startup:

```
✅ UAgent Research Extension loaded from source
✅ UAgent Research Extension routes registered
REST API prefix: /api/research
REST API routes: [.../experiments/{experiment_id}/tree...]
WebSocket prefix: /api/research/ws
WebSocket routes: [/api/research/ws/experiment/{experiment_id}]
```

## Testing Checklist

- [x] Frontend `.env` file created with correct backend URL
- [x] Frontend code updated to use backend URL for tree API
- [x] WebSocket connects to `localhost:2999` (verified in browser console)
- [x] Tree API fetches from `localhost:2999` (verified in browser network tab)
- [x] Backend routes properly registered (verified in startup logs)
- [x] Research Tree UI shows "Connected" status
- [ ] Research experiment started (user action required)
- [ ] Tree visualization displays nodes and edges (requires active experiment)

## Common Issues and Solutions

### Issue: "ERR_NETWORK_CHANGED" or "Failed to fetch"
**Solution**: Ensure `.env` file exists and frontend was restarted after creating it.

### Issue: WebSocket still connects to port 3001
**Solution**: Hard refresh the browser (Cmd+Shift+R) to clear cached JavaScript.

### Issue: "No research data available"
**Solution**: This is expected - start a research experiment by sending a "research goal:" message.

### Issue: PostHog client key error
**Solution**: This is a non-critical analytics error and can be safely ignored.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (localhost:3001)                 │
│                                                              │
│  ┌─────────────────┐         ┌──────────────────────┐      │
│  │ Research Tree   │────────▶│ useResearchWS Hook   │      │
│  │ UI Component    │         │ (WebSocket Client)   │      │
│  └─────────────────┘         └──────────────────────┘      │
│         │                              │                    │
│         │ HTTP GET /tree               │ WS Connect         │
│         ▼                              ▼                    │
└─────────┼──────────────────────────────┼───────────────────┘
          │                              │
          │ ✅ localhost:2999            │ ✅ localhost:2999
          │                              │
┌─────────┼──────────────────────────────┼───────────────────┐
│         ▼                              ▼                    │
│  ┌──────────────────────┐  ┌────────────────────────┐     │
│  │ /api/research/       │  │ /api/research/ws/      │     │
│  │ experiments/.../tree │  │ experiment/{id}        │     │
│  │ (REST Endpoint)      │  │ (WebSocket Endpoint)   │     │
│  └──────────────────────┘  └────────────────────────┘     │
│                                                             │
│              Backend (localhost:2999)                       │
│           UAgent Research Extension                         │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Start a Research Experiment**: Send a message with "research goal:" to trigger research mode
2. **Monitor Progress**: Watch the Research Tree tab for live updates
3. **Verify WebSocket Messages**: Check browser console for incoming tree updates
4. **Test Tree Interaction**: Click nodes, zoom, pan to verify full functionality

## Conclusion

The Research Tree WebSocket and API connection issues have been successfully resolved. The system is now properly configured with:

- ✅ Frontend environment variables pointing to correct backend
- ✅ WebSocket connections established to backend (port 2999)
- ✅ Tree API requests directed to backend (port 2999)
- ✅ Backend routes properly registered and accessible

The empty tree state is **expected behavior** until a research experiment is actively running.
