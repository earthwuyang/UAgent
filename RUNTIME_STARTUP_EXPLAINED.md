# OpenHands Runtime Startup Process

## What Happens During "Starting runtime... (this may take 1-2 minutes)"

When you see this message in the UI, the backend is initializing the execution environment where your agent will run commands and interact with code.

---

## Translation Key
- **Frontend displays:** `STATUS$STARTING_RUNTIME`
- **English:** "Starting runtime... (this may take 1-2 minutes)"
- **Translation file:** `frontend/src/i18n/translation.json` (line 6962-6977)

---

## Backend Status Flow

The runtime goes through these states (defined in `openhands/runtime/runtime_status.py`):

1. **STOPPED** - Runtime not started
2. **BUILDING_RUNTIME** - Building Docker/K8s images (if needed)
3. **STARTING_RUNTIME** ← *"Starting runtime..." message*
4. **RUNTIME_STARTED** - Container/pod is running
5. **SETTING_UP_WORKSPACE** - Configuring workspace
6. **SETTING_UP_GIT_HOOKS** - Installing git hooks
7. **READY** - Ready for agent tasks

---

## What Happens During STARTING_RUNTIME Phase

### Docker Runtime (`openhands/runtime/impl/docker/docker_runtime.py`)

#### 1. **Set Status** (line 171)
```python
self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
```

#### 2. **Check for Existing Container** (lines 173-180)
- Try to attach to existing container
- If not found and `attach_to_existing=True`, raise error
- Otherwise, proceed to build/create

#### 3. **Build Runtime Image** (lines 181, 245-290)
```python
self.maybe_build_runtime_container_image()
```

**For UAgent custom image (`openhands-uagent:v0.1`):**
- Check if image exists locally
- If not, try to pull from `earthwuyang/openhands-uagent:v0.1`
- If pull fails, build from `Dockerfile.runtime-fixed`
- This is the **most time-consuming step** (can take 1-2 minutes)

#### 4. **Start Container** (line 185)
```python
await call_sync_from_async(self.init_container)
```
- Creates and starts Docker container
- Mounts volumes
- Configures networking
- Sets environment variables

#### 5. **Wait for Server Ready** (lines 196-212)
```python
# Set status again before waiting
self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

# Poll runtime server for readiness
await self._poll_runtime_ready(timeout=60)
await call_sync_from_async(self.wait_until_alive)
```
- Waits for the action execution server inside container to become responsive
- Timeout: 60 seconds
- If fails, continues anyway (graceful degradation)

#### 6. **Setup Initial Environment** (lines 214-218)
```python
await call_sync_from_async(self.setup_initial_env)
```
- Install plugins
- Configure workspace
- Setup git hooks

#### 7. **Connect to Networks** (lines 228-243)
- Connect to additional Docker networks if configured

#### 8. **Mark as READY** (line 225)
```python
self.set_runtime_status(RuntimeStatus.READY)
```

---

### Kubernetes Runtime (`openhands/runtime/impl/kubernetes/kubernetes_runtime.py`)

Similar flow but for Kubernetes pods:

#### 1. **Set Status** (line 223)
```python
self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
```

#### 2. **Try Attach to Existing Pod** (line 227)
```python
await call_sync_from_async(self._attach_to_pod)
```

#### 3. **Initialize K8s Resources** (line 239)
```python
await call_sync_from_async(self._init_k8s_resources)
```
- Creates Pod, Service, PVC
- Creates VSCode ingress if configured
- Configures node selectors and tolerations

#### 4. **Wait for Pod Ready** (lines 250-263)
```python
self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
await call_sync_from_async(self._wait_until_ready)
```
- Polls Kubernetes API for pod status
- Checks `Ready` condition
- Timeout: 300 seconds (5 minutes) with 2-second intervals

#### 5. **Setup Environment & Mark Ready** (lines 265-276)
```python
await call_sync_from_async(self.setup_initial_env)
self.set_runtime_status(RuntimeStatus.READY)
```

---

## Why It Takes 1-2 Minutes

### Time Breakdown (Docker):

1. **Image Check/Pull/Build** - **60-90 seconds** (if not cached)
   - Pull from Docker Hub: 30-60s
   - Build from Dockerfile: 60-90s
   
2. **Container Creation** - **5-10 seconds**
   - Create container
   - Mount volumes
   - Setup networking

3. **Server Startup** - **10-20 seconds**
   - Start action execution server inside container
   - Wait for HTTP endpoints to respond

4. **Environment Setup** - **5-10 seconds**
   - Install plugins
   - Configure workspace
   - Setup git hooks

**Total: 80-130 seconds (1.3-2.2 minutes)**

### Time Breakdown (Kubernetes):

1. **Pod Scheduling** - **10-30 seconds**
   - K8s scheduler assigns pod to node
   - Pull image to node (if not cached)

2. **Container Init** - **20-40 seconds**
   - Container runtime starts pod
   - Volume mounts configured

3. **Ready Check** - **10-30 seconds**
   - Health checks pass
   - Readiness probe succeeds

4. **Environment Setup** - **5-10 seconds**

**Total: 45-110 seconds (0.75-1.8 minutes)**

---

## Optimization Opportunities

### To Speed Up Runtime Startup:

1. **Pre-pull Images**
   ```bash
   # On Docker host
   docker pull earthwuyang/openhands-uagent:v0.1
   
   # Or build locally
   docker build -t openhands-uagent:v0.1 -f Dockerfile.runtime-fixed .
   ```

2. **Use Persistent Containers**
   - Set `attach_to_existing=True` in config
   - Reuse existing containers instead of creating new ones

3. **Cache Dependencies**
   - Mount pip/npm caches as volumes
   - Pre-install common dependencies in base image

4. **Kubernetes: Use PriorityClass**
   - Give runtime pods higher priority for faster scheduling

5. **Disable Unnecessary Features**
   - Skip VSCode server if not needed
   - Reduce plugin count

---

## Troubleshooting

### If Runtime Startup Takes Longer Than 2 Minutes:

1. **Check Docker/K8s Resources**
   ```bash
   # Docker
   docker ps -a
   docker logs <container-name>
   
   # Kubernetes
   kubectl get pods -n <namespace>
   kubectl describe pod <pod-name> -n <namespace>
   kubectl logs <pod-name> -n <namespace>
   ```

2. **Check Network Connectivity**
   - Verify proxy settings
   - Check firewall rules
   - Ensure Docker Hub is accessible

3. **Check Disk Space**
   ```bash
   # Docker
   docker system df
   
   # Kubernetes node
   kubectl top nodes
   ```

4. **Review Logs**
   - Backend logs in console
   - Runtime container/pod logs
   - Check for error messages during build/pull

### Common Issues:

- **Image pull timeout**: Docker Hub rate limits, network issues
- **Build failures**: Missing Dockerfile, disk space
- **Port conflicts**: Another service using same port
- **Resource limits**: Insufficient CPU/memory

---

## Related Files

### Backend Runtime Implementations:
- `openhands/runtime/runtime_status.py` - Status enum definitions
- `openhands/runtime/impl/docker/docker_runtime.py` - Docker runtime (lines 171-226)
- `openhands/runtime/impl/kubernetes/kubernetes_runtime.py` - K8s runtime (lines 219-277)

### Frontend:
- `frontend/src/i18n/translation.json` - UI messages (line 6962)
- Status displayed in conversation manager UI

### Configuration:
- `openhands/core/config/sandbox_config.py` - Runtime settings
- `Dockerfile.runtime-fixed` - Custom UAgent runtime image
