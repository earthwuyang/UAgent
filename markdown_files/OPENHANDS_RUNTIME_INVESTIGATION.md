# OpenHands Runtime Investigation - Final Analysis

## Problem Summary

Despite extensive configuration attempts, OpenHands continues to default to Docker runtime even when explicitly configured otherwise. This causes "Docker client failed" errors because OpenHands tries to access Docker-in-Docker within our container.

## What We've Tried

### 1. Environment Variables Approach ❌
```python
env = {
    "RUNTIME": "exec",
    "DISABLE_DOCKER": "true",
    "DOCKER_HOST": "",
    "RUNTIME_DISABLE_DOCKER": "true",
}
```
**Result**: OpenHands ignores these variables

### 2. CLI Arguments Approach ❌
```bash
--runtime exec  # Causes "unrecognized arguments" error
```
**Result**: CLI doesn't accept runtime arguments

### 3. Configuration File Approach ❌
```toml
runtime: local
workspace_base: /workspace
```
**Result**: Config file ignored or wrong format

### 4. Multiple Environment Variable Names ❌
```bash
export RUNTIME=local
export OPENHANDS_RUNTIME=local
export OH_RUNTIME=local
```
**Result**: Still defaults to DockerRuntime

## Key Findings

### OpenHands Runtime Selection Logic
Based on our testing, OpenHands has hardcoded runtime selection that:
1. **Defaults to Docker** when Docker socket is available
2. **Ignores environment variables** for runtime selection
3. **Has no CLI arguments** for runtime configuration
4. **May require code-level changes** to override

### Available Runtimes (Theoretical)
- `docker` - Default (what it keeps using)
- `local` - Local execution with server
- `eventstream` - Stream-based execution
- `exec` - Direct execution
- `ssh` - Remote execution

### Actual Behavior
OpenHands always initializes `DockerRuntime` regardless of our configuration.

## Root Cause Analysis

The issue appears to be in OpenHands' `setup.py:73`:
```
13:08:50 - openhands:DEBUG: setup.py:73 - Initializing runtime: DockerRuntime
```

This suggests OpenHands has built-in logic that:
1. Detects Docker socket availability
2. Automatically chooses Docker runtime
3. Ignores user configuration

## Alternative Solutions

### Option 1: Accept Docker Runtime ✅ **RECOMMENDED**
Instead of fighting OpenHands' runtime selection, work with it:

```python
# Provide Docker socket with proper permissions
volumes = {
    "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
    # ... other volumes
}

# Use group-based permissions (not 777)
"group_add": [docker_gid]
```

**Benefits**:
- Works with OpenHands' default behavior
- Maintains security through group permissions
- Still provides hardware isolation via outer container

### Option 2: OpenHands Source Modification ❌ **COMPLEX**
Modify OpenHands source code to respect runtime environment variables.

**Issues**:
- Requires maintaining custom OpenHands fork
- Complex integration and updates
- Not sustainable

### Option 3: Alternative Tools ✅ **FUTURE CONSIDERATION**
Consider other AI coding tools that support:
- True headless execution
- Configurable runtimes
- No Docker dependencies

## Current Status

### What Works ✅
- OpenHands container starts successfully
- Poetry environment is properly configured
- All permissions and volumes are set correctly
- LLM configuration is working
- Playwright installation succeeds

### What Doesn't Work ❌
- Runtime selection override
- Forcing non-Docker execution
- Avoiding Docker-in-Docker architecture

## Recommended Path Forward

### Immediate Solution: Docker-in-Docker with Security
```python
# Accept OpenHands' Docker runtime choice
# Configure it securely with proper permissions
container_config = {
    "volumes": {
        "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
        # ... workspace volumes
    },
    "group_add": [docker_gid],  # Secure Docker access
    "user": f"{current_uid}:{current_gid}",
}

env = {
    "RUNTIME": "docker",  # Accept the default
    "SANDBOX_BASE_CONTAINER_IMAGE": "docker.all-hands.dev/all-hands-ai/runtime:0.57-nikolaik",
    "SANDBOX_USE_HOST_NETWORK": "true",
    # ... other config
}
```

### Architecture Result
```
Host System (Protected)
└── UAgent Container (Hardware isolation)
    └── OpenHands with Docker Runtime
        └── Nested Docker Container (Experiment isolation)
            └── Scientific experiment execution
```

**Benefits**:
- ✅ Hardware isolation via UAgent container
- ✅ Experiment isolation via nested containers
- ✅ Works with OpenHands' default behavior
- ✅ Secure Docker socket access via groups
- ✅ Complete monitoring and transparency

**Trade-offs**:
- More complex architecture (Docker-in-Docker)
- Higher resource usage
- Docker socket permission management

## Conclusion

OpenHands appears to have hardcoded Docker runtime preference that cannot be easily overridden through configuration. The most practical solution is to work with this behavior by providing secure Docker socket access rather than fighting the runtime selection.

This still achieves our goals:
- **Hardware isolation** through the outer container
- **Experiment safety** through nested containers
- **Complete monitoring** through log streaming
- **Secure execution** through proper permission management

**Final Recommendation**: Implement the Docker-in-Docker solution with proper security controls for production scientific research workflows.