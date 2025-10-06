# UAgent Research Configuration - Complete Setup

## ✅ Configuration Added to .env

Your `.env` file at `/home/wuy/AI/UAgent/.env` now includes:

```bash
# UAgent Research Configuration
ENABLE_AUTO_RESEARCH_TRIGGER=true      # Auto-trigger research for complex queries
RESEARCH_CONFIDENCE_THRESHOLD=0.7      # Trigger if confidence >= 70%
RESEARCH_MAX_ITERATIONS=50             # Max research tree iterations
RESEARCH_MAX_COST=10.0                 # Max cost in dollars
RESEARCH_MAX_PARALLEL=3                # Max concurrent research branches
```

## 🔧 How It Works

### 1. Environment Variables → Config
- `.env` file is loaded when server starts
- `extensions/uagent_research/config.py` reads these env vars
- Config is used by `research_middleware.py`

### 2. Config Flow
```
.env file
  ↓
Environment Variables (via os.getenv)
  ↓
config.py (ENABLE_AUTO_RESEARCH_TRIGGER, etc.)
  ↓
research_middleware.py (uses config)
  ↓
conversation_service.py (calls middleware)
```

## 📝 Configuration Options

### ENABLE_AUTO_RESEARCH_TRIGGER
- **Type**: Boolean (`true` or `false`)
- **Default**: `true`
- **Effect**: Enable/disable automatic research triggering
- **Example**:
  ```bash
  ENABLE_AUTO_RESEARCH_TRIGGER=true   # Research auto-triggers
  ENABLE_AUTO_RESEARCH_TRIGGER=false  # Research disabled
  ```

### RESEARCH_CONFIDENCE_THRESHOLD
- **Type**: Float (0.0 to 1.0)
- **Default**: `0.7` (70%)
- **Effect**: Minimum confidence score to trigger research
- **Example**:
  ```bash
  RESEARCH_CONFIDENCE_THRESHOLD=0.9   # Only trigger on high confidence (90%+)
  RESEARCH_CONFIDENCE_THRESHOLD=0.5   # Trigger more easily (50%+)
  RESEARCH_CONFIDENCE_THRESHOLD=0.7   # Balanced (70%+)
  ```

### RESEARCH_MAX_ITERATIONS
- **Type**: Integer
- **Default**: `50`
- **Effect**: Maximum number of research tree iterations (PUCT loops)
- **Example**:
  ```bash
  RESEARCH_MAX_ITERATIONS=100  # More thorough research
  RESEARCH_MAX_ITERATIONS=20   # Faster but less thorough
  ```

### RESEARCH_MAX_COST
- **Type**: Float (dollars)
- **Default**: `10.0`
- **Effect**: Maximum LLM cost for research (stops when exceeded)
- **Example**:
  ```bash
  RESEARCH_MAX_COST=50.0   # Allow up to $50 per research task
  RESEARCH_MAX_COST=5.0    # Budget-conscious
  ```

### RESEARCH_MAX_PARALLEL
- **Type**: Integer
- **Default**: `3`
- **Effect**: Maximum concurrent research branches
- **Example**:
  ```bash
  RESEARCH_MAX_PARALLEL=5   # More parallelism (faster, more resources)
  RESEARCH_MAX_PARALLEL=1   # Sequential only (slower, less resources)
  ```

## 🚀 Quick Start

### 1. Restart Server (REQUIRED)
```bash
# Stop current server
pkill -f "openhands"

# Start server (it will load .env automatically)
cd /home/wuy/AI/UAgent/OpenHands
python openhands/server/app.py
# Or your normal startup command
```

### 2. Send a Complex Query
```
Research neural architecture search methods and implement the best approach,
comparing DARTS vs ENAS with benchmarks and performance metrics
```

Or your postgres query:
```
Modify postgres and pg_duckdb source code, extract pre-opt features,
train ML model, embed model, and run experiments
```

### 3. Watch Research Progress
- Check server logs for: `Research mode triggered for conversation...`
- Open "Research Tree" tab in UI
- See real-time node expansion

## 🎯 Example: Your Postgres Query

**Query**: "Modify postgres and pg_duckdb source code, extract features, train model..."

**Classification**:
- **Should Trigger**: ✅ YES
- **Confidence**: 0.90 (90%)
- **Reason**: High complexity score (5), multi-stage task, many action verbs

**What Happens**:
1. Query sent to conversation
2. Middleware classifies: `complex_research` (confidence: 0.90 ≥ 0.70 threshold)
3. Research triggered automatically
4. TreeSearchOrchestrator starts
5. Parallel agents execute:
   - **DeepResearch**: Search for postgres optimization papers, pg_duckdb info
   - **RepoMaster**: Find postgres/pg_duckdb source, ML routing examples
   - **CodeAct**: Implement feature extraction, train model, run experiments
6. Research Tree shows progress in UI
7. Results aggregated and presented

## 🔍 Verification

### Check 1: Verify .env is loaded
```bash
# Start server and check logs
# Should see: "Research middleware loaded successfully"
```

### Check 2: Test classification
```bash
cd /home/wuy/AI/UAgent/OpenHands
python << 'EOF'
from extensions.uagent_research.config import ENABLE_AUTO_RESEARCH_TRIGGER
print(f"Auto-trigger enabled: {ENABLE_AUTO_RESEARCH_TRIGGER}")
EOF
```

Should output: `Auto-trigger enabled: True`

### Check 3: Send test query
Send a complex query and check server logs for:
```
INFO - Research mode triggered for conversation {id}
INFO - Task type: complex_research, confidence: 0.XX
INFO - Research started successfully: experiment_id=exp_...
```

## 🎛️ Adjusting Settings

### Quick Toggle On/Off
```bash
# Edit .env file
nano /home/wuy/AI/UAgent/.env

# Change line:
ENABLE_AUTO_RESEARCH_TRIGGER=false  # Disable
ENABLE_AUTO_RESEARCH_TRIGGER=true   # Enable

# Restart server for changes to apply
```

### Lower Threshold (Trigger More Easily)
```bash
# Edit .env
RESEARCH_CONFIDENCE_THRESHOLD=0.5  # Trigger at 50% confidence

# Restart server
```

### Increase Budget
```bash
# Edit .env
RESEARCH_MAX_COST=20.0        # Allow $20 per research
RESEARCH_MAX_ITERATIONS=100   # Allow 100 iterations

# Restart server
```

## 📊 Current Configuration Summary

| Setting | Value | Effect |
|---------|-------|--------|
| **Auto-trigger** | `true` | ✅ Research auto-triggers |
| **Confidence** | `0.7` | Trigger if ≥ 70% confidence |
| **Max Iterations** | `50` | Up to 50 PUCT iterations |
| **Max Cost** | `$10` | Stop after $10 LLM cost |
| **Parallelism** | `3` | 3 concurrent branches |

## 🐛 Troubleshooting

### Issue: Research Not Triggering

**Solution 1: Check .env is loaded**
```bash
echo $ENABLE_AUTO_RESEARCH_TRIGGER
# Should output: true
```

**Solution 2: Check server logs**
Look for:
- "Research middleware loaded successfully" ✅
- "Research middleware not available" ❌

**Solution 3: Lower confidence threshold**
```bash
# In .env
RESEARCH_CONFIDENCE_THRESHOLD=0.5
```

### Issue: Server Won't Start

**Solution: Check syntax**
```bash
# Verify .env file has valid syntax
cat /home/wuy/AI/UAgent/.env

# No spaces around =
ENABLE_AUTO_RESEARCH_TRIGGER=true  ✅
ENABLE_AUTO_RESEARCH_TRIGGER = true  ❌
```

## 📚 Related Files

1. `/home/wuy/AI/UAgent/.env` - Main configuration (environment variables)
2. `extensions/uagent_research/config.py` - Reads .env and sets defaults
3. `extensions/uagent_research/middleware/research_middleware.py` - Uses config
4. `extensions/uagent_research/classifier/task_classifier.py` - Classifies queries
5. `openhands/server/services/conversation_service.py` - Calls middleware

## ✅ Next Steps

1. ✅ Configuration added to `.env`
2. 🔄 **Restart OpenHands server** (REQUIRED)
3. 📝 Send a complex query
4. 🌳 Watch Research Tree tab
5. 🎉 Enjoy automatic research!

---

**All configuration is now centralized in `/home/wuy/AI/UAgent/.env`** - just restart the server and research auto-triggering will work! 🚀
