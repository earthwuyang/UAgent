# UAgent Research System Diagnostic Checklist

This document provides a comprehensive guide for diagnosing issues in the UAgent Research system, including tree visualization, WebSocket communication, and API integration.

## Quick Start Diagnostic Endpoints

Before diving into detailed diagnostics, check these endpoints first:

1. **Health Check**: `GET /health`
   - Verifies server is running
   - Basic sanity check

2. **Research API Diagnostics**: `GET /api/research/diagnostics`
   - Active experiments
   - Tree state information
   - API initialization status

3. **WebSocket Diagnostics**: `GET /api/research/ws/diagnostics`
   - Active connections
   - Connection metadata
   - Message statistics

4. **API Module Diagnostics**: Available via `get_module_diagnostics()` function
   - Module loading status
   - Router availability
   - Import errors

## Diagnostic Flow by Symptom

### Symptom 1: Tree Not Appearing in Frontend

**Step 1: Verify Server is Running**
```bash
curl http://localhost:3000/health
```
Expected: `{"status": "ok", "timestamp": "..."}`

**Step 2: Check Research API Status**
```bash
curl http://localhost:3000/api/research/diagnostics
```
Look for:
- `api_initialized: true`
- Experiment ID in `active_experiments`
- Non-zero `nodes_count` in experiment details

**Step 3: Verify Tree State**
```bash
curl http://localhost:3000/api/research/experiments/{experiment_id}/tree
```
Expected: Tree data with nodes and edges

**Step 4: Check WebSocket Connections**
```bash
curl http://localhost:3000/api/research/ws/diagnostics
```
Look for:
- `total_connections > 0`
- Your experiment ID in `experiments` list
- No errors in connection metadata

**Step 5: Check Server Logs**

Search for these patterns:
```
✅ Tree state updated for {experiment_id}
📡 Broadcasting tree update to X client(s)
✅ Broadcast completed for {experiment_id}
```

If missing, search for errors:
```
❌ Failed to update tree state
❌ Broadcast failed
❌ Error fetching tree
```

### Symptom 2: WebSocket Connection Failing

**Step 1: Verify WebSocket Endpoint**
- Frontend should connect to: `ws://localhost:3000/api/research/ws/experiment/{experiment_id}`
- Check browser console for connection attempts

**Step 2: Check Connection Manager**
```bash
curl http://localhost:3000/api/research/ws/diagnostics
```
Look for initialization and connection tracking

**Step 3: Check Server Logs**

Connection logs:
```
🔌 New WebSocket connection request for experiment {id}
✅ WebSocket connection established for {id}
```

Disconnection logs:
```
🔌 Client initiated disconnect for experiment {id}
✅ Client cleanly disconnected
```

Error patterns:
```
❌ Failed to establish WebSocket connection
❌ WebSocket error for experiment
```

### Symptom 3: Tree Updates Not Broadcasting

**Step 1: Verify Orchestrator is Publishing**

Check logs for:
```
🔍 Attempting to publish tree to API
✅ Tree published to API successfully
```

If missing, check for:
```
❌ Failed to publish tree to API
⚠️ API broadcast module not available
```

**Step 2: Check API is Receiving Updates**

Look for:
```
🔄 Updating tree state for {experiment_id}
✅ Tree state updated for {experiment_id}
```

**Step 3: Verify Broadcast is Triggered**

Look for:
```
📡 Broadcasting tree update to X client(s)
✅ Broadcast completed
```

**Step 4: Check Individual Client Send**

Look for:
```
✅ Sent message to connection {conn_id}
```

Or errors:
```
❌ Error sending to client {conn_id}
```

### Symptom 4: Experiment Not Starting

**Step 1: Check Middleware Initialization**

Look for:
```
🔬 Starting research for session {session_id}
📋 Experiment ID: {experiment_id}
```

**Step 2: Verify Orchestrator Creation**

Look for:
```
✅ TreeSearchOrchestrator created for {experiment_id}
📦 Orchestrator stored in active_orchestrators
```

**Step 3: Check Background Task Start**

Look for:
```
🚀 Background research task created
🏃 Research execution started
🌳 Starting tree search orchestrator
```

**Step 4: Check Session Registration**

Look for:
```
🔬 Registering experiment {experiment_id} for session {session_id}
✅ Registered experiment
```

### Symptom 5: Module Import Errors

**Step 1: Check API Module Loading**

Look for:
```
📦 Loading UAgent Research API module
✅ research_routes module loaded successfully
✅ websocket_routes module loaded successfully
```

**Step 2: Check for Import Errors**

Search for:
```
❌ Failed to import research_routes
❌ Failed to import websocket_routes
```

**Step 3: Verify Router Registration**

In `app.py` startup logs, look for:
```
🔄 Registering research API routes
✅ Research API routes registered
```

## Log Patterns Reference

### Success Patterns

| Pattern | Meaning |
|---------|---------|
| `✅ Tree state updated` | API received tree update |
| `📡 Broadcasting tree update` | WebSocket broadcast started |
| `✅ Broadcast completed` | All clients notified |
| `✅ WebSocket connection established` | Client connected |
| `✅ Sent message to connection` | Individual message sent |
| `🔬 Starting research` | Research task initiated |
| `🌳 Starting tree search` | Orchestrator running |

### Warning Patterns

| Pattern | Meaning |
|---------|---------|
| `⚠️ No active connections` | No clients listening |
| `⚠️ Experiment not found` | Invalid experiment ID |
| `⚠️ Attempted to disconnect unknown` | Connection already closed |
| `ℹ️ No active tree found` | Tree not yet created |

### Error Patterns

| Pattern | Meaning | Common Causes |
|---------|---------|---------------|
| `❌ Failed to publish tree to API` | Orchestrator can't reach API | Import failure, API not initialized |
| `❌ Error sending to client` | WebSocket send failed | Client disconnected, network issue |
| `❌ Failed to update tree state` | API update failed | Invalid data, exception in handler |
| `❌ Broadcast failed` | WebSocket broadcast error | No connections, exception in send |
| `❌ WebSocket error` | General WS failure | Connection dropped, protocol error |

## Common Issues and Solutions

### Issue: "No tree appearing but experiment is running"

**Diagnosis:**
1. Check if orchestrator is publishing: `grep "publish tree to API" logs`
2. Check if API is receiving: `grep "Tree state updated" logs`
3. Check if broadcast is working: `grep "Broadcasting tree update" logs`

**Solutions:**
- If orchestrator not publishing: Check `broadcast_tree_update` import in orchestrator
- If API not receiving: Check `update_tree_state` function and route registration
- If not broadcasting: Check WebSocket connections via diagnostics endpoint

### Issue: "WebSocket connects but no updates"

**Diagnosis:**
1. Verify client is in active connections: `curl /api/research/ws/diagnostics`
2. Check if broadcast is being called: `grep "Broadcasting tree update" logs`
3. Check for send errors: `grep "Error sending to client" logs`

**Solutions:**
- If client not in list: Check experiment ID match
- If broadcast not called: Check orchestrator publishing
- If send errors: Check client implementation, connection state

### Issue: "Multiple experiments showing wrong data"

**Diagnosis:**
1. Check active experiments: `curl /api/research/diagnostics`
2. Verify experiment IDs in logs
3. Check connection manager grouping: `curl /api/research/ws/diagnostics`

**Solutions:**
- Ensure experiment IDs are unique and properly tracked
- Verify frontend is using correct experiment ID
- Check session/experiment mapping in coordinator

## Testing Checklist

Use this checklist to verify the diagnostic system is working:

- [ ] Health endpoint responds
- [ ] Research diagnostics endpoint returns data
- [ ] WebSocket diagnostics endpoint returns data
- [ ] Module diagnostics function is accessible
- [ ] Logs contain emoji indicators (✅, ❌, 📡, etc.)
- [ ] Logs include experiment IDs
- [ ] Logs include connection IDs for WebSocket
- [ ] Error logs include stack traces
- [ ] Success logs include relevant metrics
- [ ] Broadcast logs include client counts

## Advanced Diagnostics

### Enable Debug Logging

Set environment variable or modify logger:
```python
import logging
logging.getLogger('extensions.uagent_research').setLevel(logging.DEBUG)
logging.getLogger('openhands.server').setLevel(logging.DEBUG)
```

### Track Specific Experiment

Search logs by experiment ID:
```bash
grep "{experiment_id}" logs/*.log
```

### Monitor Real-Time Updates

Tail logs with filtering:
```bash
tail -f logs/app.log | grep -E "(📡|✅|❌|🔌)"
```

### Inspect WebSocket Messages

Use browser DevTools or Wireshark to inspect WebSocket frames

### Verify API State

Query diagnostic endpoints in a loop:
```bash
watch -n 2 'curl -s http://localhost:3000/api/research/diagnostics | jq'
```

## Diagnostic Code Locations

For reference, here's where diagnostic code is implemented:

| Component | File | Key Functions |
|-----------|------|---------------|
| Health Check | `openhands/server/app.py` | `health_check()` |
| Research API Diagnostics | `extensions/uagent_research/api/research_routes.py` | `get_research_diagnostics()` |
| WebSocket Diagnostics | `extensions/uagent_research/api/websocket_routes.py` | `get_websocket_diagnostics()`, `ConnectionManager.get_diagnostics()` |
| Module Diagnostics | `extensions/uagent_research/api/__init__.py` | `get_module_diagnostics()` |
| Session Diagnostics | `openhands/server/session/session.py` | `get_research_diagnostics()` |
| Tree Publishing | `extensions/uagent_research/orchestrator/tree_orchestrator.py` | `_publish_tree_to_api()` |

## Logging Standards

All diagnostic logs should follow these conventions:

**Format:**
```
[LEVEL] {emoji} {message} {details}
```

**Emojis:**
- ✅ Success
- ❌ Error
- ⚠️ Warning
- ℹ️ Info
- 📡 Broadcasting/Communication
- 🔌 Connection
- 🔬 Research/Experiment
- 🌳 Tree/Orchestrator
- 📊 Statistics/Metrics
- 🔄 Processing/Update
- 📦 Module/Package
- 🚀 Initialization/Start
- 🧹 Cleanup

**Details:**
- Always include experiment_id when relevant
- Include connection_id for WebSocket operations
- Include counts (nodes, clients, etc.) when available
- Use try-except with exc_info=True for errors

## Contact and Support

For issues not covered by this checklist:

1. Check the main README for system architecture
2. Review component-specific documentation
3. Examine recent commits for related changes
4. Enable debug logging for detailed traces
5. Use diagnostic endpoints to gather state information

## Appendix: Diagnostic Script

```bash
#!/bin/bash
# diagnostic_check.sh - Quick diagnostic script

echo "=== UAgent Research System Diagnostics ==="
echo

echo "1. Health Check:"
curl -s http://localhost:3000/health | jq
echo

echo "2. Research API Status:"
curl -s http://localhost:3000/api/research/diagnostics | jq
echo

echo "3. WebSocket Status:"
curl -s http://localhost:3000/api/research/ws/diagnostics | jq
echo

echo "4. Recent Errors in Logs:"
grep -E "❌|ERROR" logs/app.log | tail -n 10
echo

echo "5. Recent Success Messages:"
grep "✅" logs/app.log | tail -n 10
```

Save this as `diagnostic_check.sh`, make executable, and run for a quick diagnostic overview.
