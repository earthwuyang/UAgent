# Simple Start Guide - UAgent + OpenHands

## The Simplest Way to Start

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start.sh
```

That's it! The server will start and you'll see:

```
==================================================
  OpenHands + UAgent Research Extension
==================================================

[1/2] Installing research extension...
✓ Extension installed

[2/2] Starting server on port 3000...

Available at:
  • Research API: http://localhost:3000/api/research
  • Health Check: http://localhost:3000/api/research/health

Press Ctrl+C to stop
==================================================

✅ Research database initialized
✅ UAgent Research Extension loaded successfully
INFO: Uvicorn running on http://0.0.0.0:3000
```

## What If It Doesn't Work?

### Issue 1: Frontend Build Error

**Error Message:**
```
RuntimeError: Directory './frontend/build' does not exist
```

**Solution - Build Frontend Once:**
```bash
cd /home/wuy/AI/UAgent/OpenHands/frontend
npm install
npm run build
cd ..
./start.sh
```

**Or Skip Frontend (API Only):**
```bash
# Just use the API, no UI needed
cd /home/wuy/AI/UAgent/OpenHands
./start_backend_only.sh
```

### Issue 2: Extension Not Installing

**Error Message:**
```
Extension not built
WARNING: Extension installation had warnings
```

**Solution:**
```bash
# Manually install the extension
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
pip install -e .

# Then start
cd /home/wuy/AI/UAgent/OpenHands
./start.sh
```

### Issue 3: Port Already in Use

**Error Message:**
```
Address already in use: port 3000
```

**Solution - Use Different Port:**
```bash
PORT=8000 ./start.sh
```

### Issue 4: Script Exits Without Starting Server

**Possible Causes:**
1. Build errors (npm build failed)
2. Python dependency issues
3. Port conflict

**Debug Steps:**
```bash
# Check what's wrong
cd /home/wuy/AI/UAgent/OpenHands

# 1. Test extension install
cd extensions/uagent_research
pip install -e .
python -c "from uagent_research.models.base import init_database; print('OK')"

# 2. If that works, try starting server directly
cd /home/wuy/AI/UAgent/OpenHands
python -m openhands.server.listen --port 3000
```

## Testing the API

Once the server is running:

```bash
# In another terminal
curl http://localhost:3000/api/research/health

# Expected response:
# {"status":"healthy","version":"0.1.0","database":"connected"}
```

## What's Available

| Endpoint | Description | URL |
|----------|-------------|-----|
| Health Check | Check if system is running | http://localhost:3000/api/research/health |
| List Experiments | Get all experiments | http://localhost:3000/api/research/experiments |
| Create Experiment | Start new experiment | POST http://localhost:3000/api/research/experiments/start |
| Get Experiment | Get experiment details | http://localhost:3000/api/research/experiments/{id} |
| WebSocket | Real-time updates | ws://localhost:3000/api/research/ws/experiment/{id} |

## Quick Test

```bash
# Create an experiment
curl -X POST http://localhost:3000/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Test the system",
    "session_id": "test_1",
    "research_type": "scientific"
  }'

# List experiments
curl http://localhost:3000/api/research/experiments
```

## Configuration

### Environment Variables

```bash
# Database (default: SQLite)
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"

# Custom port
export PORT=8000

# Then start
./start.sh
```

### One-Line Config

```bash
RESEARCH_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/db" PORT=8000 ./start.sh
```

## Startup Scripts Comparison

| Script | What It Does | When to Use |
|--------|--------------|-------------|
| `./start.sh` | Simple, minimal output | Quick start, testing |
| `./start_backend_only.sh` | API only, no UI | Fast development, API testing |
| `./start_openhands_research.sh` | Full checks, builds frontend | First time, production |

## Common Workflows

### Development (No UI Needed)
```bash
cd /home/wuy/AI/UAgent/OpenHands
./start.sh
```

### Production (With UI)
```bash
cd /home/wuy/AI/UAgent/OpenHands
# First time: build frontend
cd frontend && npm install && npm run build && cd ..
# Then start
./start.sh
```

### Quick API Testing
```bash
cd /home/wuy/AI/UAgent/OpenHands
PORT=8000 ./start.sh &
sleep 5
curl http://localhost:8000/api/research/health
kill %1
```

## Summary

**Recommended command:**
```bash
cd /home/wuy/AI/UAgent/OpenHands && ./start.sh
```

**If that fails:**
1. Try `./start_backend_only.sh` (skips frontend)
2. Check error messages
3. Build frontend manually if needed: `cd frontend && npm run build`
4. Ensure extension installed: `cd extensions/uagent_research && pip install -e .`

**Test it works:**
```bash
curl http://localhost:3000/api/research/health
```

---

**Created**: 2025-10-04
**Status**: Simplified startup process ✅
