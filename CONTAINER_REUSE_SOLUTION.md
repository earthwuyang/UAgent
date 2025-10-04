# Container Reuse & Workspace Persistence - Complete Solution

## Current Implementation (v1: Workspace Persistence)

### What It Does
✅ **Preserves workspace files** across retries
✅ **Reuses workspace data** without restarting builds from scratch
✅ **Tells agent** to check for existing work before redoing it

### How It Works

**Attempt 1:**
```bash
# Create container with label: openhands_session=exp_abc123
# Mount: /host/workspace → /workspace (in container)
# Agent clones repos to /workspace/duckdb_source
# Build starts, timeout after 30 min
# Container exits
# Files persist on host: /host/workspace/duckdb_source/
```

**Attempt 2:**
```bash
# Find container with label: openhands_session=exp_abc123
# Stop and remove old container
# Create NEW container with SAME label
# Mount: /host/workspace → /workspace (SAME host directory!)
# NEW container sees: /workspace/duckdb_source/ ✅ (files from attempt 1!)
# Agent checks existing work, continues build
# Success!
```

### Limitations
❌ **Creates new container** each retry (not truly reusing)
❌ **OpenHands CLI exits** after task completion
❌ **Can't send new task** to running CLI process

### Why This Approach?

**OpenHands CLI Mode:**
```bash
openhands.core.main -t "task" --no-auto-continue
# Runs once, outputs result, exits
# Cannot receive new messages while running
```

**Workspace Mount Saves Us:**
```python
volumes = {
    str(workspace_dir): {"bind": "/workspace", "mode": "rw"}
}
# Same host directory mounted to each new container
# Files persist! ✅
```

## Future Implementation (v2: True Container Reuse)

### Goal
Keep the SAME container running and send new tasks to it without restart.

### Required Changes

#### 1. Switch from CLI to Server/Session Mode

**Current (CLI):**
```bash
# One-shot execution
openhands.core.main -t "build duckdb" --no-auto-continue
# Exits when done
```

**Target (Server):**
```bash
# Long-running session
openhands-server --port 3000 &
# Stays running, accepts HTTP/WebSocket requests
```

#### 2. Use OpenHandsAppSession for Communication

Already exists in `openhands_adapter.py`:

```python
from openhands_adapter import OpenHandsAppSession

# Create session
session = OpenHandsAppSession(
    base_url="http://localhost:3000",
    session_id=cfg.session_name,
    conversation_id=conv_id
)

# Send initial task
await session.send_user_message("Build PostgreSQL and DuckDB")

# Later, send continuation message
await session.send_user_message("""
Previous task timed out. Continue from where you left off.
Check /workspace for existing work.
""")
```

#### 3. Keep Container Running Between Tasks

```python
# In openhands_single_container.py

if existing_container and existing_container.status == 'running':
    # Send new task via HTTP API to container
    container_ip = existing_container.attrs['NetworkSettings']['IPAddress']
    url = f"http://{container_ip}:3000/api/send_message"

    requests.post(url, json={
        "session_id": cfg.session_name,
        "message": cfg.goal
    })

    # Monitor the same container (don't create new one!)
    result = self._monitor_existing_container(existing_container, cfg)
    return result
```

## Implementation Plan for v2

### Step 1: Test OpenHands Server Mode

```bash
# Start OpenHands in server mode manually
docker run -d \
  -p 3000:3000 \
  -v /workspace:/workspace \
  docker.all-hands.dev/all-hands-ai/runtime:0.57 \
  openhands-server --port 3000

# Test sending messages
curl -X POST http://localhost:3000/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Get conversation_id from response, then:
curl -X POST http://localhost:3000/api/messages \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_123",
    "message": "Build DuckDB from source"
  }'
```

### Step 2: Create OpenHandsServerContainer Class

```python
# New file: openhands_server_container.py

class OpenHandsServerContainer:
    """Runs OpenHands in server mode for persistent sessions"""

    def _start_server_container(self, cfg):
        """Start container with OpenHands server"""
        cmd = [
            "openhands-server",
            "--port", "3000",
            "--workspace", "/workspace"
        ]

        container = self.docker_client.containers.run(
            image=...,
            command=cmd,
            ports={'3000/tcp': ('127.0.0.1', 3000)},
            volumes={...},
            detach=True,
            labels={"openhands_session": cfg.session_name}
        )

        # Wait for server to be ready
        self._wait_for_server_ready(container)
        return container

    async def send_task_to_running_container(self, container, task):
        """Send new task to already-running server"""
        port = container.attrs['NetworkSettings']['Ports']['3000/tcp'][0]['HostPort']
        url = f"http://localhost:{port}/api/messages"

        response = requests.post(url, json={
            "conversation_id": self.conversation_id,
            "message": task
        })

        # Stream responses via WebSocket
        await self._monitor_via_websocket(port)
```

### Step 3: Update Research Engine to Use Server Mode

```python
# In scientific_research.py

# Instead of:
container_bridge = OpenHandsSingleContainer()
result = await container_bridge.run_async(config)

# Use:
container_bridge = OpenHandsServerContainer()
session = await container_bridge.get_or_create_session(config.session_name)
result = await session.send_task(config.goal)
```

## Comparison

| Feature | v1 (Current) | v2 (Future) |
|---------|--------------|-------------|
| Workspace Persistence | ✅ Via volume mount | ✅ Same container |
| Container Reuse | ❌ New container | ✅ Same container |
| Build Progress | ✅ Files persist | ✅ Process persists |
| Resume Speed | ~10s (container start) | ~0s (instant) |
| Memory/State | ❌ Lost | ✅ Preserved |
| Implementation | ✅ Complete | ⏳ Planned |

## Why v1 Works Well Enough

Even without true container reuse, v1 provides **90%+ of the benefit**:

**Example: DuckDB Build Resume**

```bash
# Attempt 1 (timeout after 30 min)
- PostgreSQL built: 10 min
- DuckDB cloned: 2 min
- DuckDB build started: 18 min (produced 1500 .o files)
- Timeout
- Files saved to host: /workspace/duckdb_source/build/*.o

# Attempt 2 (with v1)
- Container starts: 10 sec
- Agent checks /workspace/duckdb_source: EXISTS! ✅
- Agent runs: cd /workspace/duckdb_source && make -j16
- Make sees existing .o files ✅
- Only compiles remaining files: 25 min
- SUCCESS in 25 min (not 40 min!)
```

**Time Saved: 15+ minutes per retry**

The only thing lost is **in-memory state** (running processes, environment variables set at runtime). But for compilation tasks, **filesystem state** is what matters most.

## Current Status

✅ **v1 is IMPLEMENTED and READY**
- Workspace files persist via volume mount
- Agent explicitly told to check for existing work
- Container labels enable tracking across retries
- Old containers cleaned up before creating new ones

⏳ **v2 is PLANNED for future enhancement**
- Requires switching to OpenHands server mode
- Needs API integration for sending messages
- More complex but enables instant task continuation

## Recommendation

**Use v1 now, migrate to v2 later** when the benefits justify the complexity:

**v1 is sufficient if:**
- Tasks are primarily filesystem-based (builds, clones, file operations)
- Retry delays of 10-15 seconds are acceptable
- OpenHands CLI mode is simpler to maintain

**v2 is needed if:**
- Tasks maintain critical in-memory state
- Sub-second resume times are required
- Long-running interactive sessions are common
- Need to send multiple follow-up questions without reset
