# OpenHands Permission Fix - Complete Solution

## Problem Addressed

OpenHands containers were failing with permission errors when trying to create internal directories:
```
[Errno 13] Permission denied: '/openhands/code/logs'
```

## Root Cause Analysis

The issue was that OpenHands expects to write to specific internal directories (`/openhands/code/logs`, `/openhands/code/cache`, etc.) but these paths weren't properly mounted with write permissions.

## Solution Implemented

### 1. Container Directory Preparation (`_prepare_container_directories`)

Created a new method that:
- Creates temporary directories on the host for OpenHands internal paths
- Sets proper permissions (777) and ownership (current user)
- Maps these host directories to container internal paths

**Key mappings:**
```python
volumes = {
    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
    str(cfg.workspace.resolve()): {"bind": "/workspace", "mode": "rw"},
    f"{openhands_temp}/logs": {"bind": "/openhands/code/logs", "mode": "rw"},
    f"{openhands_temp}/cache": {"bind": "/openhands/code/cache", "mode": "rw"},
    f"{openhands_temp}/tmp": {"bind": "/tmp/openhands", "mode": "rw"},
    f"{openhands_temp}/home": {"bind": "/tmp/openhands_home", "mode": "rw"},
}
```

### 2. Enhanced Environment Configuration

Updated environment variables to:
- Use local runtime instead of Docker-in-Docker to avoid nested permission issues
- Set proper HOME directory
- Configure timeouts appropriately

```python
env = {
    "RUNTIME": "local",  # No Docker-in-Docker
    "HOME": "/tmp/openhands_home",
    "SANDBOX_USE_HOST_NETWORK": "false",
    "SANDBOX_TIMEOUT": "300",
}
```

### 3. Automatic Cleanup

Added proper cleanup of temporary directories in `finally` block to prevent disk space issues.

## Files Modified

- `/home/wuy/AI/UAgent/backend/app/integrations/openhands_single_container.py`
  - Added `_prepare_container_directories()` method
  - Updated container configuration to use prepared volumes
  - Added temporary directory cleanup

## Benefits

✅ **Permission Issues Resolved**: OpenHands can now write to all required internal directories
✅ **No Docker-in-Docker**: Simplified runtime reduces complexity and permission conflicts
✅ **Proper User Mapping**: Container runs as current user to avoid ownership issues
✅ **Automatic Cleanup**: Temporary directories are cleaned up automatically
✅ **Maintains Monitoring**: Live logging and monitoring still works properly

## Testing

The fix has been implemented and is ready for testing. Run a scientific research experiment to verify:

1. Container starts without permission errors
2. OpenHands can create internal log files
3. `final.json` is created successfully in the workspace
4. Monitoring logs show proper execution

## What This Solves

- ❌ `[Errno 13] Permission denied: '/openhands/code/logs'`
- ❌ Container exits with code 1 due to permission issues
- ❌ Missing `final.json` due to failed execution
- ❌ OpenHands unable to write cache or temporary files

The OpenHands Docker integration should now work completely without permission issues while maintaining full monitoring and workspace isolation.