# UAgent Research Extension Diagnostic Checklist

This document provides step-by-step diagnostics for troubleshooting the UAgent Research Extension.

## Quick Start Verification

### 1. Server Startup Check
```bash
# Check server logs for research extension loading
grep -i "research" /tmp/openhands_server.log | tail -20

# Expected output:
# ✅ UAgent Research Extension loaded from source
# ✅ Research database initialized
# RESEARCH EXTENSION STARTUP DIAGNOSTICS
# REST API prefix: /api/research
# WebSocket prefix: /api/research/ws
```

### 2. Health Endpoint Check
```bash
curl -s http://localhost:3000/api/research/health | python3 -m json.tool
```

Expected response:
```json
{
  "status": "healthy",
  "extension": "uagent_research",
  "version": "0.1.0",
  "timestamp": "2025-10-08T...",
  "routes_registered": true,
  "api_prefix": "/api/research",
  "ws_prefix": "/api/research/ws"
}
```

### 3. Diagnostics Endpoint Check
```bash
curl -s http://localhost:3000/api/research/diagnostics | python3 -m json.tool
```

Expected: Detailed diagnostics including middleware status, active orchestrators, database stats.

### 4. WebSocket Connection Check
```bash
# Test WebSocket upgrade
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: test123" \
  http://localhost:3000/api/research/ws/experiment/test-001

# Expected: HTTP/1.1 101 Switching Protocols
```

## Detailed Diagnostics

### Orchestrator Diagnostics

Check orchestrator logs for:
- **PUCT Loop**: Tree search iterations
- **Expansion**: New node creation
- **Parallel Execution**: Concurrent agent tasks
- **Broadcasting**: UI updates

```bash
# Filter orchestrator logs
grep -E "PUCT|Expansion|Parallel|Broadcasting" /tmp/openhands_server.log
```

### Database Diagnostics

```bash
# Check database schema
sqlite3 /path/to/openhands_research.db ".tables"

# Count experiments
sqlite3 /path/to/openhands_research.db "SELECT COUNT(*) FROM experiments;"

# Check recent experiments
sqlite3 /path/to/openhands_research.db \
  "SELECT id, status, created_at FROM experiments ORDER BY created_at DESC LIMIT 5;"
```

### Frontend Console Diagnostics

Open browser developer console (F12) and check for:

1. **Network Tab**:
   - WebSocket connection to `/api/research/ws/experiment/{id}`
   - Status should be "101 Switching Protocols" 
   - Messages tab should show incoming tree updates

2. **Console Tab**:
   - No errors related to research tree
   - Look for messages like: `[Research WS] Connected to ws://...`

3. **React DevTools**:
   - Check `useResearchWS` hook state
   - Verify `useResearchTreeStore` has nodes and edges

## Python Test Script

Save as `test_research_extension.py` and run with `python3 test_research_extension.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive test script for UAgent Research Extension.
Tests HTTP endpoints and WebSocket connections.
"""

import asyncio
import aiohttp
import json
from datetime import datetime

BASE_URL = "http://localhost:3000"

async def test_health():
    """Test health endpoint."""
    print("\n1. Testing Health Endpoint...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/research/health") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"   ✅ Health: {data['status']}")
                print(f"   📦 Version: {data['version']}")
                return True
            else:
                print(f"   ❌ Health check failed: {resp.status}")
                return False

async def test_diagnostics():
    """Test diagnostics endpoint."""
    print("\n2. Testing Diagnostics Endpoint...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/research/diagnostics") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"   ✅ Middleware available: {data.get('middleware_available')}")
                print(f"   🔬 Active orchestrators: {data.get('active_orchestrators', 0)}")
                return True
            else:
                print(f"   ⚠️ Diagnostics endpoint returned: {resp.status}")
                return False

async def test_websocket():
    """Test WebSocket connection."""
    print("\n3. Testing WebSocket Connection...")
    experiment_id = "test_diagnostic_" + datetime.now().strftime("%Y%m%d%H%M%S")
    ws_url = f"ws://localhost:3000/api/research/ws/experiment/{experiment_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print(f"   ✅ WebSocket connected")
                
                # Send a ping
                await ws.send_json({"type": "ping"})
                
                # Wait for response with timeout
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        print(f"   📨 Received: {msg.data[:100]}...")
                        return True
                except asyncio.TimeoutError:
                    print(f"   ⚠️ No response received (this may be normal)")
                    return True
    except Exception as e:
        print(f"   ❌ WebSocket error: {e}")
        return False

async def test_tree_endpoint():
    """Test tree endpoint."""
    print("\n4. Testing Tree Endpoint...")
    # Use a known experiment ID or create one
    experiment_id = "test_diagnostic"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/research/experiments/{experiment_id}/tree") as resp:
            if resp.status == 200:
                data = await resp.json()
                node_count = len(data.get('data', {}).get('nodes', []))
                print(f"   ✅ Tree endpoint accessible")
                print(f"   🌳 Nodes: {node_count}")
                return True
            else:
                print(f"   ⚠️ Tree endpoint returned: {resp.status}")
                return False

async def main():
    """Run all tests."""
    print("="*60)
    print("UAgent Research Extension Diagnostic Tests")
    print("="*60)
    
    results = []
    results.append(("Health", await test_health()))
    results.append(("Diagnostics", await test_diagnostics()))
    results.append(("WebSocket", await test_websocket()))
    results.append(("Tree Endpoint", await test_tree_endpoint()))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
```

## Success Criteria Checklist

- [ ] Server starts without errors
- [ ] `/api/research/health` returns healthy status
- [ ] `/api/research/diagnostics` shows middleware available
- [ ] WebSocket connection upgrades successfully (101 response)
- [ ] Tree endpoint returns valid JSON structure
- [ ] Frontend console shows no errors
- [ ] WebSocket messages appear in browser Network tab
- [ ] Research tree tab displays nodes and edges
- [ ] Orchestrator logs show PUCT iterations
- [ ] Database contains experiment records

## Common Issues and Solutions

### Issue: "Research extension not found"
**Solution**: Check extension_dir path in app.py, ensure directory exists

### Issue: "Failed to load research middleware"
**Solution**: Check imports, ensure all dependencies installed

### Issue: "WebSocket connection refused"
**Solution**: Verify WS routes registered, check firewall rules

### Issue: "Empty tree displayed"
**Solution**: 
1. Check database has data
2. Verify API builds tree from DB
3. Check frontend is polling correct experiment ID

### Issue: "Orchestrator not running"
**Solution**: Verify middleware.start_research() was called, check background task logs

## Advanced Diagnostics

### Enable Debug Logging
```python
# In app.py or middleware
import logging
logging.getLogger('uagent_research').setLevel(logging.DEBUG)
```

### Monitor Real-time Updates
```bash
# Watch server logs
tail -f /tmp/openhands_server.log | grep -i research

# Watch WebSocket traffic (in browser DevTools > Network > WS)
# Look for messages with type: tree_update, node_expanded, etc.
```

### Database Query Examples
```sql
-- Most recent experiments
SELECT id, session_id, status, progress_percentage, created_at 
FROM experiments 
ORDER BY created_at DESC 
LIMIT 10;

-- Ideas for an experiment
SELECT i.id, i.title, i.status 
FROM ideas i 
WHERE i.session_id = 'your-session-id';

-- Tree structure
SELECT 
  e.id as experiment,
  i.id as idea,
  h.id as hypothesis
FROM experiments e
LEFT JOIN ideas i ON i.session_id = e.session_id
LEFT JOIN hypotheses h ON h.idea_id = i.id
WHERE e.id = 'your-experiment-id';
```

## Contact and Support

For issues not covered in this checklist:
1. Check GitHub issues
2. Review middleware logs for detailed error traces
3. Test with diagnostic script above
4. Collect relevant logs and create detailed issue report
