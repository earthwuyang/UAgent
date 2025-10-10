# OpenHands Research Automation Guide

This guide explains how to automatically start a research task in OpenHands using Puppeteer automation.

## Prerequisites

1. **Python 3.12+** - Required for OpenHands
2. **Node.js 18+** - Required for Puppeteer automation
3. **Poetry 1.8+** - Required for Python dependency management
4. **Docker** - Required for OpenHands runtime

## Quick Start

### Step 1: Install Dependencies

First, ensure OpenHands is properly set up:

```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands

# Install Python dependencies
make build

# Or if already built, just install
poetry install --with dev,test,runtime
```

### Step 2: Start OpenHands Server

You have several options to start the server:

**Option A: Using the provided startup script (Recommended)**
```bash
./start_openhands_research.sh
```

**Option B: Using Make**
```bash
# Full stack (backend + frontend)
make run

# Or separately in two terminals:
make start-backend  # Terminal 1
make start-frontend # Terminal 2
```

**Option C: Manual startup with Poetry**
```bash
poetry run python -m openhands.server
```

Wait for the server to start and confirm it's accessible at http://localhost:3000

### Step 3: Run the Research Automation

In a new terminal:

```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands

# Make the script executable
chmod +x run_research_automation.sh

# Run the automation
./run_research_automation.sh
```

This will:
1. Check prerequisites (Node.js, OpenHands server)
2. Install Puppeteer if needed
3. Open a browser window
4. Navigate to localhost:3000
5. Send the research goal message
6. Monitor progress for 5 minutes with screenshots

## The Research Goal

The automation sends this research goal to OpenHands:

```
research goal: modify postgres and pg_duckdb source code （ to download source 
code you can utilize the proxy on port localhost:7890, do not use the system-wide 
postgresql）, first extract pre-opt features from postgres kernel and log to files, 
then collect dual-execution data (pre-optimization query features that can be found 
in kernel structures and execution times on dual engine) and train a machine learning 
model to predict whether postgres engine or duckdb engine executes a query fast and 
embed the machine learning model into database source code (using the language of 
the database for example c language) to online route each query to the faster engine, 
and execute end-to-end experiments to test the ml-based system's performance. 

A baseline method called threshold-based method should also be implemented, which 
routes query based on threshold, for example threshold can be 10000 or 50000 or any 
other value, if postgres estimates the cost of a query is above threshold, then send 
to duckdb, otherwise send to postgres, and compare the postgres-only, duckdb-only, 
different threshold-based methods and lightgbm-based method. 

please record every successful necessary commands in README.md so that later people 
can reproduce your results. also record your python packages dependencies in 
requirements.txt.
```

## Files Created

The automation creates several files in the OpenHands directory:

- **`research_automation.log`** - Detailed log of the automation process
- **`screenshot_initial_page_*.png`** - Screenshot when page first loads
- **`screenshot_message_typed_*.png`** - Screenshot after typing the message
- **`screenshot_message_sent_*.png`** - Screenshot after sending the message
- **`screenshot_progress_*s.png`** - Progress screenshots every 30 seconds

## Monitoring Research Progress

### Via Browser

The browser window will remain open during monitoring. You can:
- Watch the chat interface for responses
- Check the research tree visualization at http://localhost:3000/research/tree
- Observe the parallel tree-structured research in action

### Via Logs

Monitor the OpenHands server logs:

```bash
# If running in background
tail -f openhands_research.log

# If running with make
tail -f logs/backend.log
```

### Via Research API

You can also query the research API directly:

```bash
# List active experiments
curl http://localhost:3000/api/research/experiments

# Get experiment status
curl http://localhost:3000/api/research/experiments/<experiment_id>/status

# Get tree structure
curl http://localhost:3000/api/research/experiments/<experiment_id>/tree
```

## Manual Alternative

If you prefer to manually interact with OpenHands:

1. Open browser to http://localhost:3000
2. Copy the research goal from above
3. Paste it into the chat interface
4. Press Enter or click Send

## Troubleshooting

### Server Not Starting

If the OpenHands server fails to start:

```bash
# Check Python version
python3 --version  # Should be 3.12+

# Check Poetry environment
poetry env info

# Reset Poetry environment if needed
poetry env remove python3.12
poetry env use python3.12
poetry install
```

### Puppeteer Issues

If Puppeteer fails to install or run:

```bash
# Install Puppeteer manually
npm install puppeteer

# On macOS, you may need to install Chromium
npx puppeteer browsers install chrome
```

### Port 3000 Already in Use

```bash
# Find and kill the process using port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
export port=3001
./start_openhands_research.sh
```

### Chat Input Not Found

If the automation can't find the chat input:

1. Check the screenshot files created in the OpenHands directory
2. Manually inspect the page at http://localhost:3000
3. Update the selectors in `start_research_task.js` if needed

## Advanced Usage

### Modify the Research Goal

Edit `start_research_task.js` and update the `RESEARCH_GOAL` constant:

```javascript
const RESEARCH_GOAL = `your custom research goal here`;
```

### Change Monitoring Duration

Edit `start_research_task.js`:

```javascript
// Change from 5 minutes to your desired duration
const monitorDuration = 10 * 60 * 1000; // 10 minutes
```

### Run in Headless Mode

Edit `start_research_task.js`:

```javascript
browser = await puppeteer.launch({
    headless: true,  // Change from false to true
    // ...
});
```

## Understanding the Research System

OpenHands-UAgent uses a tree-based research approach:

1. **TreeSearchOrchestrator** breaks down the research goal into sub-tasks
2. **PUCT Algorithm** selects promising research branches: `Q + c * P * sqrt(N) / (1 + n)`
3. **Research Adapters** execute tasks:
   - DeepResearchAdapter: Web search and content extraction
   - RepoMasterAdapter: GitHub repository analysis
   - CodeActAdapter: Code execution and validation
4. **EventBus** streams updates to the frontend in real-time
5. **Parallel Execution** runs up to 3 tasks concurrently

The research continues even after the monitoring phase ends. Check the workspace directory for results:

```bash
ls -la workspace/*/
```

## Next Steps

After the research completes:

1. Check the workspace directory for generated code and results
2. Review the README.md created by the research system
3. Check requirements.txt for Python dependencies
4. Follow the reproduction steps documented by the system

## Support

For issues with:
- **OpenHands**: Check https://github.com/All-Hands-AI/OpenHands
- **This automation**: Check the logs in `research_automation.log`
- **Research system**: Check `extensions/uagent_research/`

## Files in This Directory

- **`start_openhands_research.sh`** - Starts OpenHands with research extension
- **`start_research_task.js`** - Puppeteer automation script
- **`run_research_automation.sh`** - Main automation runner (this script)
- **`RESEARCH_AUTOMATION_README.md`** - This file
