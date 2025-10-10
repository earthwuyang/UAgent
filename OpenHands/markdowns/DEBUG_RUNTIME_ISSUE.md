# Runtime Initialization Issue

## Problem
Agent shows "Stopped" when initialization should be ready.

## Error Analysis

### Primary Error
```
AgentRuntimeDisconnectedError: Failed to connect to runtime at http://localhost:42174/execute_action: [Errno 111] Connection refused
```

**Meaning**: The OpenHands runtime server (action execution server) failed to start or isn't listening on the expected port.

### Secondary Error
```
McpError: Timed out while waiting for response to ClientRequest. Waited 30.0 seconds.
```

**Meaning**: MCP (Model Context Protocol) client couldn't connect to MCP server within timeout.

## Root Causes (Likely)

### 1. Runtime Container/Process Not Starting
The runtime (Docker container or local process) that executes agent actions isn't initializing.

**Check:**
```bash
# Is Docker running?
docker ps

# Check Docker logs
docker logs $(docker ps -a | grep openhands | awk '{print $1}' | head -1)

# Check if port is in use
netstat -tuln | grep 42174
lsof -i :42174
```

### 2. MCP Server Not Starting
The MCP server (browser automation) isn't responding.

**Check:**
```bash
# Check MCP processes
ps aux | grep mcp
ps aux | grep playwright

# Check if browser automation is working
python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); print('OK')"
```

### 3. Timeout Too Short
30 second timeout might be too short if initializing for the first time.

## Solutions

### Solution 1: Check Docker Status
```bash
# Restart Docker if needed
sudo systemctl restart docker

# Clean up old containers
docker ps -a | grep openhands | awk '{print $1}' | xargs docker rm -f
```

### Solution 2: Increase Timeouts
Edit these files:

**openhands/mcp/utils.py (line ~149)**
```python
# Change from:
with anyio.fail_after(timeout):  # Default 30s

# To:
with anyio.fail_after(timeout or 60):  # Give more time
```

**openhands/mcp/client.py**
```python
# Increase MCP timeout
timeout = 60  # Instead of 30
```

### Solution 3: Check Runtime Configuration
```bash
# Check if runtime type is set correctly
grep -r "runtime" openhands/core/config.py

# Try using eventstream runtime instead of action execution
export RUNTIME_TYPE=eventstream
```

### Solution 4: Disable MCP Temporarily
If MCP is not critical for your use case:

```python
# In openhands/server/app.py or session.py
# Comment out MCP client initialization
# await create_mcp_clients(...)
```

### Solution 5: Check Port Availability
```bash
# If port 42174 is in use, kill the process
lsof -ti:42174 | xargs kill -9

# Or change the port in config
export RUNTIME_PORT=42175
```

## Quick Fix Attempts

### Attempt 1: Restart Everything
```bash
# Kill all related processes
pkill -f openhands
pkill -f playwright
docker rm -f $(docker ps -aq)

# Start fresh
poetry run python openhands/server/listen.py
```

### Attempt 2: Use Different Runtime
```bash
# Try CLI runtime instead of Docker
export SANDBOX_TYPE=local
poetry run python openhands/server/listen.py
```

### Attempt 3: Skip MCP
```bash
# Disable MCP entirely
export DISABLE_MCP=true
poetry run python openhands/server/listen.py
```

## Diagnostic Commands

```bash
# 1. Check what's listening on ports
netstat -tuln | grep -E '(42174|3000|8000)'

# 2. Check Docker status
docker info
docker ps -a

# 3. Check Python processes
ps aux | grep python | grep openhands

# 4. Check logs
tail -f logs/openhands.log  # If logging to file
journalctl -f  # System logs

# 5. Test MCP manually
python -c "import asyncio; from openhands.mcp.utils import create_mcp_clients; asyncio.run(create_mcp_clients({}))"
```

## Is This Related to Import Fixes?

**No** - The import standardization we just completed fixes the **research middleware loading**.

This runtime issue is about the **agent execution environment** (Docker/local runtime + MCP servers).

These are separate systems:
- ✅ Research middleware: Now loads correctly
- ❌ Runtime environment: Having initialization issues

## Recommended Actions

1. **Check Docker is running**: `docker ps`
2. **Increase timeouts**: Edit `openhands/mcp/utils.py` and `openhands/mcp/client.py`
3. **Try without Docker**: `export SANDBOX_TYPE=local`
4. **Disable MCP temporarily**: `export DISABLE_MCP=true`
5. **Check logs**: Look for more detailed error messages

## Related Files to Check

- `openhands/runtime/impl/action_execution/action_execution_client.py`
- `openhands/mcp/utils.py`
- `openhands/mcp/client.py`
- `openhands/server/session/session.py`

