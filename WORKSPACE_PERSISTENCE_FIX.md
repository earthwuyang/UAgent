# Workspace Persistence Issue & Fix

## Problem Statement

When experiments retry after failure, they **start from scratch** instead of continuing from the previous attempt's progress. Specifically:

- **Attempt 1**: Builds PostgreSQL ✅, starts building DuckDB (30 minutes), times out ❌
- **Attempt 2**: Starts fresh, clones repos again, builds PostgreSQL again ❌ (should continue DuckDB build)

This wastes enormous amounts of time re-doing work that was already completed.

## Root Cause

### Issue 1: New Container Per Retry

File: `backend/app/integrations/openhands_single_container.py:585`

```python
container = self.docker_client.containers.run(**container_config)
```

**Every retry creates a BRAND NEW Docker container**:
- New container = fresh filesystem
- Previous build artifacts in `/workspace` inside the old container are **LOST**

### Issue 2: Workspace Mount is Correct But Unused

The workspace directory **IS** mounted as a volume:

```python
# Line 193
volumes = {
    str(cfg.workspace.resolve()): {"bind": "/workspace", "mode": "rw"},
}
```

**But there are TWO workspace directories:**

1. **`session_1759559338623_kr8m6m8yj`** - Base session workspace (empty, unused)
2. **`experiment_session_1759559338623_kr8m6m8yj_ml-query-router`** - Experiment-specific workspace

### Issue 3: Each Retry Uses Same Workspace But Agent Ignores It

Evidence from file inspection:
```bash
$ ls /home/wuy/AI/uagent-workspace/uagent_workspaces/experiment_session_1759559338623_kr8m6m8yj_ml-query-router/workspace/
# EMPTY!
```

The agent works in `/workspace` inside the container, but files aren't persisting between retries because:
1. The previous container is removed/stopped
2. The new container mounts the same **host directory** to `/workspace`
3. BUT the host directory is **EMPTY** - files were only inside the previous container's filesystem, not the mount!

## Why Files Don't Persist

When the agent runs `git clone` inside the container:

```bash
# Inside container at /workspace
git clone https://github.com/duckdb/duckdb.git duckdb_source
```

This creates `/workspace/duckdb_source` **inside the container**.

**Expected**: Files should appear in the host-mounted directory
**Actual**: Files are **NOT** appearing in the host directory

### Possible Causes

1. **Agent is NOT using `/workspace`** - files go to a different location
2. **Container volume mount is failing** - permission issues
3. **Files are created in a subdirectory** that's not mounted

## Investigation Results

Checking inside the running container:

```bash
$ docker exec bda82fcdbc4c ls -la /workspace
drwxrwxrwx 8 openhands openhands  4096 Oct  4 07:11 .
drwxrwxrwx 2 openhands openhands  4096 Oct  4 07:11 workspace  # subdirectory!
```

**AH HA!** There's a `workspace/` **subdirectory** inside `/workspace`. The mount structure is:

```
Host: /home/wuy/AI/uagent-workspace/uagent_workspaces/experiment.../
  ├── workspace/           # Empty subdirectory
  ├── experiments/
  ├── logs/
  └── code/

Container: /workspace/
  ├── workspace/           # Subdirectory (mounted from host)
  ├── experiments/         # (mounted from host)
  ├── logs/                # (mounted from host)
  └── code/                # (mounted from host)
```

**The agent is probably cloning into `/workspace/` directly, but we need to verify where files actually end up!**

## Solution Options

### Option 1: Reuse Same Container Across Retries (Recommended)

Instead of creating a new container, **reuse the existing one**:

```python
# backend/app/integrations/openhands_single_container.py

def run(self, cfg: SingleContainerConfig) -> SingleContainerResult:
    # Check if container for this session already exists
    existing_container = self._find_existing_container(cfg.session_name)

    if existing_container and existing_container.status == 'running':
        # Reuse the container!
        logger.info(f"Reusing existing container {existing_container.id} for session {cfg.session_name}")
        container = existing_container
    else:
        # Create new container only if none exists
        container = self.docker_client.containers.run(**container_config)
```

**Benefits:**
- All files in `/workspace` persist automatically
- Partial build progress (compiled .o files, cloned repos) is preserved
- DuckDB can continue from where it left off

**Implementation:**
```python
def _find_existing_container(self, session_name: str):
    """Find existing container for this session"""
    try:
        containers = self.docker_client.containers.list(
            all=True,
            filters={"label": f"openhands_session={session_name}"}
        )
        return containers[0] if containers else None
    except Exception as e:
        logger.warning(f"Failed to find existing container: {e}")
        return None
```

### Option 2: Ensure Volume Mount Actually Works

Make sure files go to the mounted directory:

```python
# backend/app/integrations/openhands_single_container.py:238

# Add working directory enforcement to the goal
enhanced_goal = f"""IMPORTANT: Work in /workspace directory.
ALL files (source code, builds, data) MUST be in /workspace so they persist across retries.

{cfg.goal}
"""
```

### Option 3: Increase Timeout to Avoid Retries

The DuckDB build needs 60+ minutes:

```bash
export OPENHANDS_MAX_ACTION_TIMEOUT=7200  # 2 hours
```

### Option 4: Use Pre-compiled DuckDB

Avoid building from source:

```bash
# In the experiment prompt, suggest using pre-built binaries
wget https://github.com/duckdb/duckdb/releases/download/v1.1.3/duckdb_cli-linux-amd64.zip
```

## Recommended Fix

**Implement Option 1 (Container Reuse)** + **Option 2 (Enforce /workspace)**:

1. Add container labeling for session tracking
2. Find and reuse existing container if it exists
3. Only create new container if none exists or old one exited
4. Ensure agent is explicitly told to use `/workspace`

### Implementation Steps

1. **Add label to container config** (line 560):
```python
container_config = {
    "image": ...,
    "labels": {
        "openhands_session": cfg.session_name,
        "uagent_experiment": "true"
    },
    ...
}
```

2. **Check for existing container** (before line 585):
```python
existing = self._find_existing_container(cfg.session_name)
if existing and existing.status in ['running', 'exited']:
    if existing.status == 'exited':
        logger.info(f"Restarting existing container {existing.id}")
        existing.restart()
    container = existing
else:
    container = self.docker_client.containers.run(**container_config)
```

3. **Send new command to existing container**:
```python
if existing:
    # Send new task to the already-running OpenHands agent
    # This requires the agent to support receiving new tasks
    result = self._send_new_task_to_container(container, cfg.goal)
```

## Expected Outcome

After fix:

```
Attempt 1:
  - Create container
  - Clone PostgreSQL ✅
  - Clone DuckDB ✅
  - Build PostgreSQL ✅
  - Start building DuckDB (30 min elapsed, timeout)

Attempt 2:
  - Reuse same container
  - PostgreSQL source: ✅ Still there
  - DuckDB source: ✅ Still there
  - PostgreSQL binary: ✅ Already built
  - Continue building DuckDB from where it left off ✅
  - Complete after 30 more minutes ✅

Total time: 60 minutes (instead of 60+ minutes PER ATTEMPT)
```

## Testing

After implementing the fix:

1. Start an experiment that takes >30 minutes
2. Let it timeout
3. Check if Attempt 2 sees the files from Attempt 1:
   ```bash
   # In Attempt 2, the agent should see:
   ls /workspace/duckdb_source  # Should NOT be empty
   ```

4. Verify build continues:
   ```bash
   # Should show partial .o files from Attempt 1
   find /workspace/duckdb_source/build -name "*.o" | wc -l
   ```
