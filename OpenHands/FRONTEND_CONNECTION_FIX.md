# Frontend Connection Issue - Research Tree Not Showing Data

## Problem
The research tree panel shows "Disconnected" and no data because the frontend WebSocket cannot connect to the backend.

## Root Cause
When running `execute_ml_routing_research.py` directly, it bypasses the OpenHands FastAPI web server that handles WebSocket connections. The frontend expects:

1. **HTTP API** at `/api/research/*` for starting/controlling experiments
2. **WebSocket** at `/api/research/ws/{experiment_id}` for real-time updates

## Solution Options

### Option 1: Use OpenHands Web Server (RECOMMENDED)

Instead of running the standalone script, start research through the OpenHands web interface or API:

#### Step 1: Start OpenHands Server
```bash
cd /home/wuy/AI/UAgent/OpenHands
# Start OpenHands with research extension enabled
poetry run python openhands/server/listen.py
```

#### Step 2: Use the Web UI or API
- **Via Web UI**: Open browser to `http://localhost:3000` and use the research interface
- **Via API**: POST to `/api/research/start` endpoint

```bash
curl -X POST http://localhost:3000/api/research/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Develop ML-based query routing for PostgreSQL + DuckDB",
    "session_id": "ml_routing_001",
    "config": {
      "max_iterations": 50,
      "max_cost": 20.0,
      "max_parallel": 3
    }
  }'
```

### Option 2: Add WebSocket Server to Standalone Script

Modify `execute_ml_routing_research.py` to include a minimal FastAPI server:

```python
import uvicorn
from fastapi import FastAPI
from extensions.uagent_research.api import websocket_routes, research_routes

app = FastAPI()
app.include_router(websocket_routes.ws_router)
app.include_router(research_routes.router)

# Start server in background thread
import threading
server_thread = threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=3000),
    daemon=True
)
server_thread.start()

# Then start research as normal
middleware = ResearchMiddleware()
experiment_id = await middleware.start_research(...)
```

### Option 3: Use Research API Routes

The proper way is to integrate with OpenHands' existing server:

1. **Check if API routes are registered**:
```bash
cd /home/wuy/AI/UAgent/OpenHands
grep -r "research_routes\|websocket_routes" openhands/server/
```

2. **Ensure routes are imported in server setup**:
```python
# In openhands/server/listen.py or similar
from extensions.uagent_research.api import research_routes, websocket_routes

app.include_router(research_routes.router)
app.include_router(websocket_routes.ws_router)
```

## Verification Steps

### 1. Check if WebSocket endpoint is available
```bash
# With OpenHands server running
curl http://localhost:3000/api/research/diagnostics
```

Expected response:
```json
{
  "status": "ok",
  "api_initialized": true,
  "active_experiments": [],
  "total_experiments": 0
}
```

### 2. Test WebSocket connection
```javascript
// In browser console
const ws = new WebSocket('ws://localhost:3000/api/research/ws/test_experiment');
ws.onopen = () => console.log('✅ Connected');
ws.onmessage = (e) => console.log('📨 Message:', e.data);
ws.onerror = (e) => console.log('❌ Error:', e);
```

### 3. Check frontend configuration
Look for WebSocket connection code in frontend:
```bash
find frontend -name "*.tsx" -o -name "*.ts" | xargs grep -l "WebSocket\|useWebSocket"
```

## Quick Fix for Testing

If you want to keep using the standalone script but see the research progress:

```python
# Add this to execute_ml_routing_research.py after starting research

print("\n📊 Polling research status...")
while middleware.is_experiment_running(experiment_id):
    await asyncio.sleep(5)
    
    # Get orchestrator and print stats
    orch_data = middleware.get_orchestrator_for_tracking(experiment_id)
    if orch_data:
        orch = orch_data['orchestrator']
        if hasattr(orch, 'tree') and orch.tree:
            stats = orch.tree.stats
            print(f"  Nodes: {stats.get('total', 0)}, "
                  f"Completed: {stats.get('completed', 0)}, "
                  f"Running: {stats.get('running', 0)}")
```

## Recommended Action

**Start OpenHands with the web server** instead of using the standalone script:

```bash
cd /home/wuy/AI/UAgent/OpenHands

# Method 1: Via poetry
poetry run python openhands/server/listen.py

# Method 2: Direct python
python openhands/server/listen.py

# Then access UI at http://localhost:3000
```

The frontend will automatically connect to the WebSocket endpoint and display real-time research progress.

## Files to Check

1. `openhands/server/listen.py` - Main server entry point
2. `openhands/server/app.py` - FastAPI app configuration  
3. `extensions/uagent_research/api/__init__.py` - API initialization
4. `frontend/src/api/research.ts` (or similar) - Frontend API client

## Summary

The frontend requires a running FastAPI server with WebSocket support. The standalone `execute_ml_routing_research.py` script runs independently and doesn't provide this infrastructure. To see the research tree visualization, you must either:

1. Start the OpenHands web server (recommended)
2. Modify the standalone script to include a FastAPI server
3. Use the API endpoints to start research programmatically

