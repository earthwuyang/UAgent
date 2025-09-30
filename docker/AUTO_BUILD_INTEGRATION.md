# Automatic Docker Image Build Integration

## Overview

UAgent now **automatically checks and builds** the required Docker image if it's not found locally or on Docker Hub. No manual intervention needed!

## How It Works

### Integrated Check in `openhands_single_container.py`

When scientific research experiments start, UAgent automatically:

1. **Checks locally**: Is the image already built on this machine?
   - ✅ If found → Use it immediately
   - ❌ If not → Continue to step 2

2. **Checks Docker Hub**: Is the image available remotely?
   - ✅ If found → Pull it automatically
   - ❌ If not → Continue to step 3

3. **Builds automatically**: Build from source locally
   - Finds `docker/research-runtime.Dockerfile`
   - Runs `docker build` automatically
   - Streams build logs to application logger
   - Takes ~10-15 minutes

### Implementation Details

**File**: `backend/app/integrations/openhands_single_container.py`

**Key Function**: `ensure_docker_image_exists(image_name: str) -> bool`

**Called From**: `OpenHandsSingleContainer.run_async()` method

**Behavior**:
- **Runs once per application lifetime** (uses global flag with lock)
- **Non-blocking**: Uses async/await
- **Logging**: All progress logged at INFO level
- **Error handling**: Returns False if build fails, prevents experiment from running

## User Experience

### First Time Setup (No Image)

```bash
# Start UAgent backend
uvicorn app.main:app --reload --port 8001 --app-dir backend

# Submit research query
curl -X POST http://localhost:8001/api/research/scientific \
  -H "Content-Type: application/json" \
  -d '{"query": "Test experiment", "research_id": "test"}'
```

**What Happens**:
```
INFO: Checking Docker image: earthwuyang/uagent-research:v1.0
WARNING: ⚠️  Image not found locally: earthwuyang/uagent-research:v1.0
INFO: Checking Docker Hub for: earthwuyang/uagent-research:v1.0
WARNING: ⚠️  Image not found on Docker Hub: earthwuyang/uagent-research:v1.0
INFO: Building image locally from source (this may take 10-15 minutes)...
INFO: Building from: /Users/wuy/Desktop/code/UAgent/docker/research-runtime.Dockerfile
INFO: ======================================================================
INFO: Docker build starting - this will take ~10-15 minutes
INFO: ======================================================================
INFO: [docker build] #1 [internal] load build definition...
INFO: [docker build] #2 [internal] load metadata...
... (build logs stream in real-time) ...
INFO: ✅ Successfully built: earthwuyang/uagent-research:v1.0
INFO: Starting experiment...
```

### Subsequent Runs (Image Exists)

```
INFO: Checking Docker image: earthwuyang/uagent-research:v1.0
INFO: ✅ Image found locally: earthwuyang/uagent-research:v1.0
INFO: Starting experiment...
```

**Fast**: No delay, uses existing image immediately.

## Configuration

The image name comes from `.env`:

```bash
UAGENT_OPENHANDS_IMAGE=earthwuyang/uagent-research:v1.0
```

## Benefits

### 1. Zero Manual Setup
- No need to run `docker build` manually
- No need to remember complex build commands
- Works out of the box

### 2. Smart Caching
- Only checks/builds once per application lifetime
- Uses async lock to prevent duplicate builds
- Reuses existing images when available

### 3. Automatic Fallback
- Tries Docker Hub first (faster)
- Falls back to local build automatically
- Clear error messages if build fails

### 4. Production Ready
- Logs all steps for debugging
- Handles timeouts and errors gracefully
- Prevents experiments from running with wrong image

## Developer Experience

### Scenario 1: New Developer Onboarding

```bash
# Clone repo
git clone https://github.com/yourorg/UAgent.git
cd UAgent

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start backend (NO docker build command needed!)
uvicorn app.main:app --reload --port 8001 --app-dir backend

# Submit research query - image builds automatically on first run
```

### Scenario 2: Dockerfile Changes

```bash
# Edit Dockerfile
vim docker/research-runtime.Dockerfile

# Remove old image to force rebuild
docker rmi earthwuyang/uagent-research:v1.0

# Restart backend - new image builds automatically
uvicorn app.main:app --reload --port 8001 --app-dir backend
```

### Scenario 3: CI/CD Pipeline

```yaml
# .github/workflows/test.yml
- name: Run UAgent Tests
  run: |
    # No docker build step needed!
    # Tests will build image automatically if not cached
    pytest test/
```

## Troubleshooting

### Build Fails

**Symptom**: Error message: "Docker image X not available and build failed"

**Solution**:
1. Check logs for specific build error
2. Ensure Docker daemon is running: `docker ps`
3. Check Dockerfile syntax: `docker/research-runtime.Dockerfile`
4. Ensure sufficient disk space: `docker system df`

### Build Too Slow

**Symptom**: Build takes > 20 minutes

**Solution**:
1. Check internet connection (downloads packages)
2. Check Docker resources: Settings → Resources → increase CPU/Memory
3. Pre-pull base image: `docker pull ghcr.io/all-hands-ai/runtime:0.57-nikolaik`

### Wrong Image Used

**Symptom**: Experiments fail with "command not found"

**Solution**:
1. Check .env has correct image: `grep UAGENT_OPENHANDS_IMAGE .env`
2. Remove old image: `docker rmi <old_image>`
3. Restart backend to rebuild

## Comparison: Before vs After

### Before (Manual)

```bash
# Developer workflow
cd UAgent

# 1. Read documentation to find build command
# 2. Copy/paste complex docker build command
docker build -t earthwuyang/uagent-research:v1.0 -f docker/research-runtime.Dockerfile .

# 3. Wait 10-15 minutes
# 4. If error, debug Dockerfile
# 5. Rebuild

# 6. THEN start backend
uvicorn app.main:app --reload --port 8001 --app-dir backend

# 7. Hope you built the right image
```

### After (Automatic)

```bash
# Developer workflow
cd UAgent

# 1. Start backend
uvicorn app.main:app --reload --port 8001 --app-dir backend

# 2. Submit research query
# 3. System builds image automatically if needed
# Done!
```

## Technical Details

### Thread Safety

```python
# Global flag with async lock prevents race conditions
_IMAGE_CHECK_PERFORMED = False
_IMAGE_CHECK_LOCK = asyncio.Lock()

async def ensure_docker_image_exists(image_name: str) -> bool:
    global _IMAGE_CHECK_PERFORMED

    async with _IMAGE_CHECK_LOCK:  # Only one check at a time
        if _IMAGE_CHECK_PERFORMED:
            return True
        # ... check and build ...
        _IMAGE_CHECK_PERFORMED = True
```

### Process Management

```python
# Stream build output to logger in real-time
process = subprocess.Popen(
    build_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in iter(process.stdout.readline, ''):
    logger.info(f"[docker build] {line.rstrip()}")

process.wait()
```

### Error Recovery

- **Timeout**: Docker Hub check times out after 30 seconds
- **Build failure**: Returns False, prevents experiment from starting
- **Missing Dockerfile**: Clear error message with path
- **Permission errors**: Logged but don't crash app

## Files Modified

1. `backend/app/integrations/openhands_single_container.py`
   - Added `ensure_docker_image_exists()` function (lines 33-151)
   - Modified `run_async()` to call check (lines 1089-1100)
   - Added global flags and lock (lines 28-30)

## Testing

### Manual Test

```bash
# 1. Remove image if exists
docker rmi earthwuyang/uagent-research:v1.0

# 2. Start backend
uvicorn app.main:app --reload --port 8001 --app-dir backend

# 3. Submit test query
curl -X POST http://localhost:8001/api/research/scientific \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a hello world program", "research_id": "test"}'

# 4. Watch logs - should see automatic build
```

### Expected Log Output

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     127.0.0.1:54321 - "POST /api/research/scientific HTTP/1.1" 200 OK
INFO:app.integrations.openhands_single_container - Checking Docker image: earthwuyang/uagent-research:v1.0
WARNING:app.integrations.openhands_single_container - ⚠️  Image not found locally: earthwuyang/uagent-research:v1.0
INFO:app.integrations.openhands_single_container - Checking Docker Hub for: earthwuyang/uagent-research:v1.0
WARNING:app.integrations.openhands_single_container - ⚠️  Image not found on Docker Hub: earthwuyang/uagent-research:v1.0
INFO:app.integrations.openhands_single_container - Building image locally from source (this may take 10-15 minutes)...
INFO:app.integrations.openhands_single_container - Building from: /Users/wuy/Desktop/code/UAgent/docker/research-runtime.Dockerfile
INFO:app.integrations.openhands_single_container - ======================================================================
INFO:app.integrations.openhands_single_container - Docker build starting - this will take ~10-15 minutes
INFO:app.integrations.openhands_single_container - ======================================================================
INFO:app.integrations.openhands_single_container - [docker build] #1 [internal] load build definition from research-runtime.Dockerfile
... (many build logs) ...
INFO:app.integrations.openhands_single_container - ✅ Successfully built: earthwuyang/uagent-research:v1.0
```

## Future Enhancements

Potential improvements:

1. **Build Progress Bar**: Show % completion for Docker build
2. **Build Caching**: Cache intermediate layers more aggressively
3. **Multi-platform**: Automatically detect architecture and build correct platform
4. **Pre-warming**: Build image during app startup in background
5. **Version Check**: Rebuild if Dockerfile changed since last build

## See Also

- Docker image specification: `docker/research-runtime.Dockerfile`
- Build instructions: `docker/README.md`
- Multi-platform build: `docker/BUILD_MULTIPLATFORM.md`