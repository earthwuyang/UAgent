# How to Start UAgent-Embedded OpenHands

## Quick Answer

You have **two options** for starting the system:

### Option 1: Full System (with Frontend UI) - **RECOMMENDED**

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_openhands_research.sh
```

**What you get:**
- ✅ OpenHands Web UI (http://localhost:3000)
- ✅ Research API (http://localhost:3000/api/research)
- ✅ WebSocket streaming (ws://localhost:3000/api/research/ws)

**First time setup:** Will automatically build the frontend (takes ~5 minutes)

### Option 2: Backend Only (API only) - **FASTER START**

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_backend_only.sh
```

**What you get:**
- ✅ Research API (http://localhost:3000/api/research)
- ✅ WebSocket streaming (ws://localhost:3000/api/research/ws)
- ❌ No Web UI

**Startup time:** ~5 seconds (no frontend build needed)

---

## Detailed Comparison

| Feature | Full System | Backend Only |
|---------|-------------|--------------|
| **Startup Time** | ~5 min (first time)<br>~10 sec (subsequent) | ~5 seconds |
| **OpenHands UI** | ✅ Yes | ❌ No |
| **Research API** | ✅ Yes | ✅ Yes |
| **WebSocket** | ✅ Yes | ✅ Yes |
| **Use Case** | Interactive development | API testing, automation |

---

## Option 1: Full System (Detailed)

### First Time Startup

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_openhands_research.sh
```

**What happens:**
```
[1/6] Checking installation...
✓ Extension found
✓ Python found

[2/6] Checking dependencies...
✓ Extension dependencies OK

[3/6] Checking frontend...
⚠ Frontend not built, building now...
  This may take a few minutes...
  Installing Node.js dependencies...
  Building frontend...
✓ Frontend built successfully

[4/6] Configuring environment...
✓ Using default SQLite database
✓ Server will run on port: 3000

[5/6] Verifying installation...
✓ Installation verified

[6/6] Starting OpenHands server...

Server will be available at:
  • Main UI:       http://localhost:3000
  • Research API:  http://localhost:3000/api/research
  • WebSocket:     ws://localhost:3000/api/research/ws

✅ Research database initialized
✅ UAgent Research Extension loaded successfully
INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:3000
```

### Subsequent Startups

After the first build, frontend is cached and startup is fast (~10 seconds).

### What You Can Access

1. **OpenHands Web UI**: http://localhost:3000
   - Standard OpenHands interface
   - Chat with agents
   - File operations
   - Git operations

2. **Research API**: http://localhost:3000/api/research
   - 8 HTTP endpoints for experiments
   - Create, list, get, delete experiments
   - Generate ideas and hypotheses

3. **WebSocket Streaming**: ws://localhost:3000/api/research/ws
   - Real-time experiment progress
   - Session updates

---

## Option 2: Backend Only (Detailed)

### Startup

```bash
cd /home/wuy/AI/UAgent/OpenHands
./start_backend_only.sh
```

**What happens:**
```
[1/3] Checking extension...
✓ Extension found

[2/3] Installing extension...
✓ Extension ready

[3/3] Starting backend...

Backend API will be available at:
  • Research API:  http://localhost:3000/api/research
  • WebSocket:     ws://localhost:3000/api/research/ws
  • Health:        http://localhost:3000/api/research/health

Note: Frontend UI is not available in this mode

✅ UAgent Research Extension loaded
Starting server...

INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:3000
```

### What You Can Access

1. **Research API**: http://localhost:3000/api/research
   - All HTTP endpoints work
   - Full CRUD operations

2. **WebSocket Streaming**: ws://localhost:3000/api/research/ws
   - Real-time updates work

3. **No Web UI**: http://localhost:3000 → **404 Not Found**
   - Use curl/Postman/Python instead

### When to Use Backend Only

- ✅ Testing the API
- ✅ Running automated scripts
- ✅ CI/CD pipelines
- ✅ Quick development iteration
- ✅ Don't need the Web UI

---

## Testing the System

### Test the API (Both Options)

```bash
# In another terminal
/home/wuy/AI/UAgent/test_research_api.sh
```

**Expected Output:**
```
[1/5] Testing server health...
✓ OpenHands server is running

[2/5] Testing research extension health...
✓ Research extension is healthy

[3/5] Testing list experiments...
✓ Can list experiments

[4/5] Testing create experiment...
✓ Experiment created successfully
Experiment ID: exp_abc123

[5/5] Testing get experiment...
✓ Can retrieve experiment

✅ All API tests passed!
```

### Test WebSocket Connection

**Using wscat:**
```bash
# Install wscat if needed
npm install -g wscat

# Connect
wscat -c ws://localhost:3000/api/research/ws/experiment/test_exp
```

**Using Python:**
```python
import asyncio
import websockets

async def test():
    uri = "ws://localhost:3000/api/research/ws/experiment/test_exp"
    async with websockets.connect(uri) as ws:
        message = await ws.recv()
        print(message)

asyncio.run(test())
```

---

## Troubleshooting

### Frontend Build Fails

**Error:** `npm install` or `npm run build` fails

**Solution 1 - Use backend only:**
```bash
./start_backend_only.sh
```

**Solution 2 - Fix Node.js:**
```bash
# Check Node.js version
node --version  # Should be v18 or higher

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Port Already in Use

**Error:** `Address already in use: port 3000`

**Solution - Use different port:**
```bash
PORT=8000 ./start_openhands_research.sh
# or
PORT=8000 ./start_backend_only.sh
```

### Extension Not Loading

**Error:** No "✅ UAgent Research Extension loaded successfully" message

**Solution:**
```bash
# Verify installation
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
python verify_installation.py

# If that fails, reinstall
pip install -e .
```

### Database Errors

**Error:** Database connection or initialization fails

**Solution - Reset database:**
```bash
# For SQLite
rm /home/wuy/AI/UAgent/OpenHands/openhands_research.db

# Then restart
./start_backend_only.sh
```

---

## Configuration

### Environment Variables

```bash
# Database (defaults to SQLite)
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"

# Or PostgreSQL
export RESEARCH_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/openhands_research"

# Custom port
export PORT=8000

# Then start
./start_openhands_research.sh
```

### Config in One Command

```bash
RESEARCH_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/db" PORT=8000 ./start_backend_only.sh
```

---

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:3000/api/research/health

# Create experiment
curl -X POST http://localhost:3000/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Test sorting algorithms",
    "session_id": "session_1",
    "research_type": "scientific"
  }'

# List experiments
curl http://localhost:3000/api/research/experiments

# Get specific experiment
curl http://localhost:3000/api/research/experiments/exp_123
```

### Using Python

```python
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        # Create experiment
        response = await client.post(
            "http://localhost:3000/api/research/experiments/start",
            json={
                "goal": "Compare algorithms",
                "session_id": "session_1",
                "research_type": "scientific"
            }
        )
        print(response.json())

asyncio.run(main())
```

### Using WebSocket (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:3000/api/research/ws/experiment/exp_123');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};

// Keep alive
setInterval(() => ws.send('ping'), 30000);
```

---

## Summary

### Quick Start Commands

**For full system (with UI):**
```bash
cd /home/wuy/AI/UAgent/OpenHands && ./start_openhands_research.sh
```

**For API only (faster):**
```bash
cd /home/wuy/AI/UAgent/OpenHands && ./start_backend_only.sh
```

**Test the API:**
```bash
/home/wuy/AI/UAgent/test_research_api.sh
```

### What's Available

| Component | Full System | Backend Only | URL |
|-----------|-------------|--------------|-----|
| OpenHands UI | ✅ | ❌ | http://localhost:3000 |
| Research API | ✅ | ✅ | http://localhost:3000/api/research |
| WebSocket | ✅ | ✅ | ws://localhost:3000/api/research/ws |
| Health Check | ✅ | ✅ | http://localhost:3000/api/research/health |

### Recommendation

- **Development/Interactive**: Use `./start_openhands_research.sh`
- **API Testing/Automation**: Use `./start_backend_only.sh`
- **First Time**: Try backend only first to test quickly

---

**Last Updated**: 2025-10-04
**Status**: Phase 2 Complete - Both startup modes working ✅
