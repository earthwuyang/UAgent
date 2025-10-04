# UAgent-Embedded OpenHands - Startup Guide

## Quick Start (Development Mode)

### 1. Start the Backend

```bash
# Navigate to OpenHands directory
cd /home/wuy/AI/UAgent/OpenHands

# Optional: Set database URL (defaults to SQLite)
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"

# Start the OpenHands server (includes UAgent Research Extension)
python -m openhands.server.listen
```

**Expected Output**:
```
[INFO] Research database initialized: sqlite+aiosqlite:///./openhands_research.db
✅ UAgent Research Extension loaded successfully
[INFO] Application startup complete
[INFO] Uvicorn running on http://0.0.0.0:3000
```

The backend will be available at:
- **Main UI**: http://localhost:3000
- **API**: http://localhost:3000/api
- **Research API**: http://localhost:3000/api/research
- **WebSocket**: ws://localhost:3000/api/research/ws

### 2. Start the Frontend

OpenHands has its own frontend that will automatically load when you access http://localhost:3000.

**The UAgent Research Extension will be accessible via**:
- REST API endpoints
- WebSocket connections
- (Frontend UI coming in Phase 3)

---

## Detailed Startup Instructions

### Backend Setup

#### Option 1: Development Mode (Recommended)

```bash
# 1. Navigate to OpenHands
cd /home/wuy/AI/UAgent/OpenHands

# 2. Activate virtual environment (if using one)
# source venv/bin/activate

# 3. Ensure extension is installed
cd extensions/uagent_research
pip install -e .
cd ../..

# 4. Set environment variables (optional)
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"
export RESEARCH_DEBUG=false

# 5. Start the server
python -m openhands.server.listen

# Or with custom port:
# python -m openhands.server.listen --port 8000
```

#### Option 2: Production Mode

```bash
# Use PostgreSQL for production
export RESEARCH_DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/openhands_research"

# Start with production settings
python -m openhands.server.listen --host 0.0.0.0 --port 3000
```

#### Option 3: Using Docker (if available)

```bash
# Build Docker image with extension
cd /home/wuy/AI/UAgent/OpenHands
docker build -t openhands-research .

# Run container
docker run -p 3000:3000 \
  -e RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db" \
  openhands-research
```

---

## Verifying the System is Running

### 1. Check Backend Health

```bash
# Check OpenHands is running
curl http://localhost:3000/api/health

# Check Research Extension is loaded
curl http://localhost:3000/api/research/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "database": "connected"
}
```

### 2. Check Database

```bash
# For SQLite
ls -la openhands_research.db

# For PostgreSQL
psql $RESEARCH_DATABASE_URL -c "\dt"
```

### 3. Test API Endpoints

```bash
# List experiments
curl http://localhost:3000/api/research/experiments

# Expected response:
# {"experiments": [], "total": 0}
```

### 4. Test WebSocket Connection

**Using wscat** (install with `npm install -g wscat`):
```bash
# Connect to experiment WebSocket
wscat -c ws://localhost:3000/api/research/ws/experiment/test_exp

# You should see:
# Connected
# < {"type":"connected","experiment_id":"test_exp",...}
```

**Using JavaScript** (browser console):
```javascript
const ws = new WebSocket('ws://localhost:3000/api/research/ws/experiment/test_exp');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// Should log: {type: "connected", experiment_id: "test_exp", ...}
```

---

## Using the System

### Via REST API

#### 1. Create a Research Session

```bash
curl -X POST http://localhost:3000/api/research/experiments/start \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Compare quicksort vs mergesort performance",
    "session_id": "session_1",
    "research_type": "scientific"
  }'
```

Response:
```json
{
  "id": "exp_abc123",
  "status": "pending",
  "session_id": "session_1",
  "experiment_type": "scientific",
  "goal": "Compare quicksort vs mergesort performance",
  "created_at": "2025-10-04T16:00:00Z"
}
```

#### 2. Check Experiment Status

```bash
curl http://localhost:3000/api/research/experiments/exp_abc123
```

#### 3. Watch Progress in Real-Time

**Terminal 1** - Start WebSocket listener:
```bash
wscat -c ws://localhost:3000/api/research/ws/experiment/exp_abc123
```

**Terminal 2** - Start experiment (when implemented):
```bash
# This will trigger progress updates via WebSocket
curl -X POST http://localhost:3000/api/research/experiments/exp_abc123/execute
```

### Via Python SDK

```python
import asyncio
import httpx

async def run_experiment():
    async with httpx.AsyncClient() as client:
        # Start experiment
        response = await client.post(
            "http://localhost:3000/api/research/experiments/start",
            json={
                "goal": "Test hypothesis about sorting algorithms",
                "session_id": "session_1",
                "research_type": "scientific"
            }
        )
        experiment_id = response.json()["id"]
        print(f"Started experiment: {experiment_id}")

        # Poll for status
        while True:
            response = await client.get(
                f"http://localhost:3000/api/research/experiments/{experiment_id}"
            )
            data = response.json()
            print(f"Status: {data['status']}, Progress: {data['progress']['percentage']}%")

            if data["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(5)

asyncio.run(run_experiment())
```

### Via WebSocket (Real-Time)

**Python**:
```python
import asyncio
import websockets
import json

async def watch_experiment(experiment_id):
    uri = f"ws://localhost:3000/api/research/ws/experiment/{experiment_id}"

    async with websockets.connect(uri) as websocket:
        print(f"Watching experiment: {experiment_id}")

        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "progress":
                print(f"Progress: {data['data']['percentage']}% - {data['data']['current_step']}")
            elif data["type"] == "result":
                print("Experiment completed!")
                print(json.dumps(data["data"], indent=2))
                break

asyncio.run(watch_experiment("exp_abc123"))
```

**JavaScript** (browser):
```javascript
const experimentId = 'exp_abc123';
const ws = new WebSocket(`ws://localhost:3000/api/research/ws/experiment/${experimentId}`);

ws.onopen = () => console.log('Connected');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'connected':
            console.log('WebSocket connected:', data.message);
            break;
        case 'progress':
            console.log(`${data.data.percentage}% - ${data.data.current_step}`);
            // Update UI progress bar
            updateProgressBar(data.data.percentage);
            break;
        case 'result':
            console.log('Completed:', data.data);
            break;
    }
};

// Keep alive
setInterval(() => ws.send('ping'), 30000);
```

---

## Accessing the OpenHands UI

### Main UI

Open your browser and go to:
```
http://localhost:3000
```

This is the **standard OpenHands UI**. The UAgent Research Extension runs in the background and can be accessed via:

1. **API calls** (as shown above)
2. **WebSocket connections**
3. **Custom frontend** (coming in Phase 3)

### OpenHands Features Still Available

All standard OpenHands features work normally:
- Chat interface
- File operations
- Git operations
- Agent conversations
- Code execution

The research extension adds additional capabilities on top.

---

## Configuration

### Environment Variables

```bash
# Research Extension Database
export RESEARCH_DATABASE_URL="sqlite+aiosqlite:///./openhands_research.db"
# Or PostgreSQL:
# export RESEARCH_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/research"

# Debug mode
export RESEARCH_DEBUG=true

# OpenHands Configuration (optional)
export WORKSPACE_BASE="/path/to/workspace"
export LLM_MODEL="anthropic/claude-3-5-sonnet-20241022"
export LLM_API_KEY="your-api-key"
```

### Configuration File

Create `config.toml` in the OpenHands root:

```toml
# OpenHands Configuration
[core]
workspace_base = "/workspace"

[llm]
model = "anthropic/claude-3-5-sonnet-20241022"
api_key = "your-api-key"

# Research Extension Configuration
[research]
enabled = true
database_url = "sqlite+aiosqlite:///./openhands_research.db"
max_concurrent_experiments = 5
default_timeout_seconds = 3600
```

---

## Troubleshooting

### Backend Won't Start

**Issue**: Server fails to start

**Check**:
1. Port is not in use:
   ```bash
   lsof -i :3000
   # If something is using port 3000, kill it or use different port
   python -m openhands.server.listen --port 8000
   ```

2. Dependencies installed:
   ```bash
   pip install -r requirements.txt
   cd extensions/uagent_research
   pip install -e .
   ```

3. Check logs:
   ```bash
   python -m openhands.server.listen 2>&1 | tee server.log
   ```

### Extension Not Loading

**Issue**: "✅ UAgent Research Extension loaded successfully" not shown

**Solutions**:

1. Verify extension directory exists:
   ```bash
   ls -la /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
   ```

2. Check extension can be imported:
   ```bash
   cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research
   python verify_installation.py
   ```

3. Check OpenHands app.py has integration code:
   ```bash
   grep -n "RESEARCH_EXTENSION_AVAILABLE" /home/wuy/AI/UAgent/OpenHands/openhands/server/app.py
   ```

### Database Errors

**Issue**: "Database not initialized" or connection errors

**Solutions**:

1. For SQLite - check file permissions:
   ```bash
   touch openhands_research.db
   chmod 666 openhands_research.db
   ```

2. For PostgreSQL - verify connection:
   ```bash
   psql $RESEARCH_DATABASE_URL -c "SELECT 1;"
   ```

3. Reset database:
   ```bash
   # For SQLite
   rm openhands_research.db

   # For PostgreSQL
   psql $RESEARCH_DATABASE_URL -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   ```

### WebSocket Connection Fails

**Issue**: WebSocket connections immediately disconnect

**Solutions**:

1. Check server is running:
   ```bash
   curl http://localhost:3000/api/research/health
   ```

2. Check firewall:
   ```bash
   sudo ufw status
   # Allow port if needed:
   sudo ufw allow 3000
   ```

3. Use correct WebSocket URL:
   ```
   ws://localhost:3000/api/research/ws/experiment/{id}
   NOT wss:// (unless you have SSL)
   ```

---

## Development Workflow

### Running Tests

```bash
cd /home/wuy/AI/UAgent/OpenHands/extensions/uagent_research

# Run all tests
pytest -v

# Run specific test file
pytest tests/test_integration_workflow.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Watching Logs

```bash
# In one terminal - start server with debug logging
export RESEARCH_DEBUG=true
python -m openhands.server.listen

# In another terminal - watch database
watch -n 1 "sqlite3 openhands_research.db 'SELECT id, status, progress_percentage FROM experiments;'"

# In another terminal - test API
while true; do
  curl -s http://localhost:3000/api/research/experiments | jq
  sleep 5
done
```

### Hot Reload (Development)

For development, use uvicorn directly with reload:

```bash
cd /home/wuy/AI/UAgent/OpenHands

# Install uvicorn with reload support
pip install "uvicorn[standard]"

# Start with auto-reload
uvicorn openhands.server.app:app --reload --host 0.0.0.0 --port 3000
```

Now any changes to Python files will automatically reload the server.

---

## Production Deployment

### 1. Use PostgreSQL

```bash
# Create database
createdb openhands_research

# Set environment
export RESEARCH_DATABASE_URL="postgresql+asyncpg://user:password@localhost/openhands_research"
```

### 2. Use Process Manager

**Using systemd**:

Create `/etc/systemd/system/openhands.service`:
```ini
[Unit]
Description=OpenHands with UAgent Research
After=network.target

[Service]
Type=simple
User=openhands
WorkingDirectory=/home/wuy/AI/UAgent/OpenHands
Environment="RESEARCH_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/openhands_research"
ExecStart=/usr/bin/python3 -m openhands.server.listen --host 0.0.0.0 --port 3000
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl enable openhands
sudo systemctl start openhands
sudo systemctl status openhands
```

**Using PM2** (Node.js):
```bash
npm install -g pm2

# Create ecosystem.config.js
cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [{
    name: 'openhands',
    script: 'python3',
    args: '-m openhands.server.listen',
    cwd: '/home/wuy/AI/UAgent/OpenHands',
    env: {
      RESEARCH_DATABASE_URL: 'postgresql+asyncpg://user:pass@localhost/openhands_research'
    }
  }]
};
EOF

pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### 3. Use Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket support
    location /api/research/ws {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

---

## Summary

### Quick Start Commands

```bash
# 1. Navigate to OpenHands
cd /home/wuy/AI/UAgent/OpenHands

# 2. Start backend
python -m openhands.server.listen

# 3. Open browser
# Go to http://localhost:3000

# 4. Test research API
curl http://localhost:3000/api/research/health
```

### Available Endpoints

- **Main UI**: http://localhost:3000
- **Research API**: http://localhost:3000/api/research/*
- **WebSocket**: ws://localhost:3000/api/research/ws/*
- **Health Check**: http://localhost:3000/api/research/health

### Next Steps

1. ✅ Backend is running
2. ⏳ Frontend UI (coming in Phase 3)
3. ⏳ Advanced features (coming in Phase 4)

The system is now **ready for use via API and WebSocket**. The frontend dashboard will be built in Phase 3.

---

**Last Updated**: 2025-10-04
**Status**: Phase 2 Complete - Backend Operational ✅
