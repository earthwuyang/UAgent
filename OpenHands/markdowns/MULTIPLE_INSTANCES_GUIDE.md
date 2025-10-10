# Running Multiple OpenHands Instances

**Guide for running multiple OpenHands instances on different ports**

---

## 🎯 Port Configuration

### Environment Variable

The server port is controlled by the `OPENHANDS_PORT` environment variable in your `.env` file.

**File**: `/home/wuy/AI/UAgent/.env`

```bash
# Server Configuration
OPENHANDS_PORT=3000  # Change this for different instances
```

### Priority Order

The code checks environment variables in this order:
1. `OPENHANDS_PORT` (recommended, most specific)
2. `PORT` (standard convention)
3. `port` (legacy, lowercase)
4. Default: `3000`

**Code Location**: `openhands/server/__main__.py:11-18`

---

## 📋 Running Multiple Instances

### Method 1: Multiple .env Files (Recommended)

Create separate `.env` files for each instance:

#### Instance 1: Port 3000
**File**: `/home/wuy/AI/UAgent/.env.instance1`
```bash
# Server Configuration
OPENHANDS_PORT=3000

# LLM Configuration
LLM_API_KEY=${DASHSCOPE_API_KEY}
LLM_BASE_URL=${DASHSCOPE_BASE_URL}
LLM_MODEL=openai/qwen3-coder-plus
# ... rest of config
```

#### Instance 2: Port 3001
**File**: `/home/wuy/AI/UAgent/.env.instance2`
```bash
# Server Configuration
OPENHANDS_PORT=3001

# LLM Configuration
LLM_API_KEY=${DASHSCOPE_API_KEY}
LLM_BASE_URL=${DASHSCOPE_BASE_URL}
LLM_MODEL=openai/qwen3-coder-plus
# ... rest of config
```

#### Instance 3: Port 3002
**File**: `/home/wuy/AI/UAgent/.env.instance3`
```bash
# Server Configuration
OPENHANDS_PORT=3002

# LLM Configuration
LLM_API_KEY=${DASHSCOPE_API_KEY}
LLM_BASE_URL=${DASHSCOPE_BASE_URL}
LLM_MODEL=openai/qwen3-coder-plus
# ... rest of config
```

#### Start Instances

```bash
# Terminal 1: Instance on port 3000
cd /home/wuy/AI/UAgent/OpenHands
cp ../.env.instance1 .env
python -m openhands.server

# Terminal 2: Instance on port 3001
cd /home/wuy/AI/UAgent/OpenHands
cp ../.env.instance2 .env
python -m openhands.server

# Terminal 3: Instance on port 3002
cd /home/wuy/AI/UAgent/OpenHands
cp ../.env.instance3 .env
python -m openhands.server
```

---

### Method 2: Environment Variable Override

Override the port directly when starting:

```bash
# Instance 1 on port 3000
OPENHANDS_PORT=3000 python -m openhands.server

# Instance 2 on port 3001
OPENHANDS_PORT=3001 python -m openhands.server

# Instance 3 on port 3002
OPENHANDS_PORT=3002 python -m openhands.server
```

**Note**: Other `.env` settings will still be loaded from the file.

---

### Method 3: Separate Working Directories

Create separate working directories for each instance:

```bash
# Setup
mkdir -p ~/openhands-instances/{instance1,instance2,instance3}

# Copy OpenHands to each
cp -r /home/wuy/AI/UAgent/OpenHands ~/openhands-instances/instance1/
cp -r /home/wuy/AI/UAgent/OpenHands ~/openhands-instances/instance2/
cp -r /home/wuy/AI/UAgent/OpenHands ~/openhands-instances/instance3/

# Configure each instance
echo "OPENHANDS_PORT=3000" > ~/openhands-instances/instance1/OpenHands/.env
echo "OPENHANDS_PORT=3001" > ~/openhands-instances/instance2/OpenHands/.env
echo "OPENHANDS_PORT=3002" > ~/openhands-instances/instance3/OpenHands/.env

# Start each instance
cd ~/openhands-instances/instance1/OpenHands && python -m openhands.server &
cd ~/openhands-instances/instance2/OpenHands && python -m openhands.server &
cd ~/openhands-instances/instance3/OpenHands && python -m openhands.server &
```

---

## 🚀 Startup Scripts

### Create Startup Scripts

#### `start-instance-1.sh`
```bash
#!/bin/bash
cd /home/wuy/AI/UAgent/OpenHands
export OPENHANDS_PORT=3000
export WORKSPACE_BASE=/home/wuy/openhands-workspace-1
python -m openhands.server
```

#### `start-instance-2.sh`
```bash
#!/bin/bash
cd /home/wuy/AI/UAgent/OpenHands
export OPENHANDS_PORT=3001
export WORKSPACE_BASE=/home/wuy/openhands-workspace-2
python -m openhands.server
```

#### `start-instance-3.sh`
```bash
#!/bin/bash
cd /home/wuy/AI/UAgent/OpenHands
export OPENHANDS_PORT=3002
export WORKSPACE_BASE=/home/wuy/openhands-workspace-3
python -m openhands.server
```

Make executable:
```bash
chmod +x start-instance-*.sh
```

---

## 🔧 Additional Configuration for Multiple Instances

When running multiple instances, you should also configure:

### 1. Separate Workspaces

```bash
# In .env.instance1
WORKSPACE_BASE=/home/wuy/openhands-workspace-1

# In .env.instance2
WORKSPACE_BASE=/home/wuy/openhands-workspace-2

# In .env.instance3
WORKSPACE_BASE=/home/wuy/openhands-workspace-3
```

### 2. Separate File Stores (Optional)

```bash
# In .env.instance1
FILE_STORE_PATH=/home/wuy/openhands-filestore-1

# In .env.instance2
FILE_STORE_PATH=/home/wuy/openhands-filestore-2

# In .env.instance3
FILE_STORE_PATH=/home/wuy/openhands-filestore-3
```

### 3. Separate Cache Directories (Optional)

```bash
# In .env.instance1
CACHE_DIR=/tmp/openhands-cache-1

# In .env.instance2
CACHE_DIR=/tmp/openhands-cache-2

# In .env.instance3
CACHE_DIR=/tmp/openhands-cache-3
```

---

## 🌐 Accessing Multiple Instances

### URLs

- Instance 1: `http://120.46.207.248:3000`
- Instance 2: `http://120.46.207.248:3001`
- Instance 3: `http://120.46.207.248:3002`

### Check Running Instances

```bash
# List all running OpenHands instances
ps aux | grep "openhands.server" | grep -v grep

# Check ports in use
netstat -tlnp | grep python
# or
ss -tlnp | grep python
```

---

## 🛑 Managing Multiple Instances

### Stop All Instances

```bash
pkill -f "openhands.server"
```

### Stop Specific Instance

```bash
# Find the PID for specific port
lsof -ti:3001 | xargs kill -9

# Or by process
ps aux | grep "openhands.server" | grep -v grep
kill <PID>
```

### Restart Specific Instance

```bash
# Stop instance on port 3001
lsof -ti:3001 | xargs kill -9

# Start it again
OPENHANDS_PORT=3001 python -m openhands.server &
```

---

## 🔍 Troubleshooting

### Issue: Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find what's using the port
lsof -i:3000

# Kill the process
lsof -ti:3000 | xargs kill -9

# Or use a different port
OPENHANDS_PORT=3010 python -m openhands.server
```

### Issue: Can't Connect to Instance

**Check**:
1. Is the instance running?
   ```bash
   ps aux | grep openhands
   ```

2. Is the port accessible?
   ```bash
   curl http://localhost:3000
   ```

3. Firewall blocking?
   ```bash
   sudo ufw status
   sudo ufw allow 3000/tcp
   sudo ufw allow 3001/tcp
   sudo ufw allow 3002/tcp
   ```

---

## 📊 Example: Complete Multi-Instance Setup

### Complete `.env.instance1` (Port 3000)
```bash
# OpenHands Instance 1 Configuration

# Server Configuration
OPENHANDS_PORT=3000

# LLM Configuration
LLM_API_KEY=${DASHSCOPE_API_KEY}
LLM_BASE_URL=${DASHSCOPE_BASE_URL}
LLM_MODEL=openai/qwen3-coder-plus
LLM_NUM_RETRIES=8
LLM_RETRY_MIN_WAIT=15
LLM_RETRY_MAX_WAIT=120
LLM_TIMEOUT=600

# Sandbox Configuration
SANDBOX_USE_HOST_NETWORK=true
SANDBOX_TIMEOUT=120

# Agent Configuration
AGENT_MEMORY_ENABLED=true
AGENT_MEMORY_MAX_THREADS=2

# Workspace
WORKSPACE_BASE=/home/wuy/openhands-workspace-1

# UAgent Research Configuration
ENABLE_AUTO_RESEARCH_TRIGGER=true
RESEARCH_CONFIDENCE_THRESHOLD=0.5
RESEARCH_MAX_ITERATIONS=9999999999
RESEARCH_MAX_COST=9999999999
RESEARCH_MAX_PARALLEL=3
```

### Complete `.env.instance2` (Port 3001)
```bash
# OpenHands Instance 2 Configuration

# Server Configuration
OPENHANDS_PORT=3001

# LLM Configuration
LLM_API_KEY=${DASHSCOPE_API_KEY}
LLM_BASE_URL=${DASHSCOPE_BASE_URL}
LLM_MODEL=openai/qwen3-coder-plus
LLM_NUM_RETRIES=8
LLM_RETRY_MIN_WAIT=15
LLM_RETRY_MAX_WAIT=120
LLM_TIMEOUT=600

# Sandbox Configuration
SANDBOX_USE_HOST_NETWORK=true
SANDBOX_TIMEOUT=120

# Agent Configuration
AGENT_MEMORY_ENABLED=true
AGENT_MEMORY_MAX_THREADS=2

# Workspace
WORKSPACE_BASE=/home/wuy/openhands-workspace-2

# UAgent Research Configuration
ENABLE_AUTO_RESEARCH_TRIGGER=true
RESEARCH_CONFIDENCE_THRESHOLD=0.5
RESEARCH_MAX_ITERATIONS=9999999999
RESEARCH_MAX_COST=9999999999
RESEARCH_MAX_PARALLEL=3
```

---

## ✅ Summary

**Port Configuration**: ✅ Added to `.env`

**Environment Variable**: `OPENHANDS_PORT` (also supports `PORT` or `port`)

**Default Port**: `3000`

**To Run Multiple Instances**:
1. Set different `OPENHANDS_PORT` for each instance
2. Use separate workspaces (`WORKSPACE_BASE`)
3. Start each instance in a separate terminal or background process

**Example**:
```bash
# Instance 1
OPENHANDS_PORT=3000 python -m openhands.server &

# Instance 2
OPENHANDS_PORT=3001 python -m openhands.server &

# Instance 3
OPENHANDS_PORT=3002 python -m openhands.server &
```

**Access at**:
- http://120.46.207.248:3000
- http://120.46.207.248:3001
- http://120.46.207.248:3002

🎉 **You can now run multiple OpenHands instances on different ports!**
