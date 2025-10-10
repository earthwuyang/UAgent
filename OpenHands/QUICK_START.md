# Quick Start: PostgreSQL + pg_duckdb ML Routing Research

This guide gets you started with the automated research task in under 5 minutes.

## Prerequisites Check

```bash
# You need:
python3 --version  # Should be 3.12+ (CRITICAL!)
node --version     # Should be 18+
docker --version   # Any recent version
poetry --version   # Should be 1.8+
```

⚠️ **IMPORTANT**: Your system has Python 3.9, but OpenHands requires Python 3.12+. You'll need to install Python 3.12 first.

## Installation Options for Python 3.12

### Option 1: Using Homebrew (Recommended for macOS)
```bash
brew install python@3.12
```

### Option 2: Using pyenv
```bash
brew install pyenv
pyenv install 3.12.0
pyenv global 3.12.0
```

### Option 3: Download from python.org
Visit https://www.python.org/downloads/ and install Python 3.12.

## Once Python 3.12 is Installed

### Step 1: Build OpenHands (One-time setup)
```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands
make build
```

This will:
- Set up Poetry environment with Python 3.12
- Install all Python dependencies
- Install frontend dependencies
- Set up pre-commit hooks
- Build the frontend

### Step 2: Configure OpenHands (One-time setup)
```bash
make setup-config
```

This will prompt you for:
- Workspace directory (press Enter for default: `./workspace`)
- LLM model (press Enter for default: `gpt-4o`)
- LLM API key (enter your OpenAI/DashScope API key)
- LLM base URL (press Enter to skip if using OpenAI)

### Step 3: Start OpenHands in Background
```bash
# Option A: Using the research startup script
nohup ./start_openhands_research.sh > server.log 2>&1 &

# Option B: Using Make
nohup make run > server.log 2>&1 &
```

Wait 30-60 seconds for the server to start, then verify:
```bash
curl http://localhost:3000
# Should return HTML or redirect
```

### Step 4: Run the Automation
```bash
./run_research_automation.sh
```

This will:
1. Check that Node.js and OpenHands are ready
2. Install Puppeteer if needed
3. Open a browser to http://localhost:3000
4. Automatically send your research goal
5. Monitor progress for 5 minutes with screenshots

## What Happens Next?

The research system will:

1. **Parse the Goal**: Break down the PostgreSQL + pg_duckdb ML routing task
2. **Generate Ideas**: Create a tree of research approaches
3. **Parallel Execution**: Run up to 3 research branches simultaneously
4. **Adaptive Exploration**: Use PUCT algorithm to select promising paths
5. **Code Generation**: Download source code, extract features, train ML model
6. **Documentation**: Create README.md and requirements.txt

## Monitoring Progress

### In the Browser
Watch the tree visualization at: http://localhost:3000/research/tree

### In the Terminal
```bash
# Watch server logs
tail -f server.log

# Or automation logs
tail -f research_automation.log
```

### Via API
```bash
# List experiments
curl http://localhost:3000/api/research/experiments

# Get status (replace <id> with actual experiment ID)
curl http://localhost:3000/api/research/experiments/<id>/status
```

## Expected Timeline

- **Setup**: 5-10 minutes (one-time)
- **Research Initialization**: 1-2 minutes
- **Active Research**: 30-60 minutes (continues in background)
- **Complete Task**: 1-3 hours (depends on complexity)

## Where to Find Results

Results will be in:
```bash
ls -la workspace/
```

Look for:
- `README.md` - Reproduction steps
- `requirements.txt` - Python dependencies
- Source code directories for PostgreSQL and pg_duckdb
- Feature extraction logs
- Trained ML models
- Experiment results

## Troubleshooting

### "Python 3.12 not found"
Install Python 3.12 (see Installation Options above), then:
```bash
poetry env use python3.12
poetry install
```

### "Port 3000 already in use"
```bash
lsof -ti:3000 | xargs kill -9
```

### "Server not responding"
Check the logs:
```bash
tail -100 server.log
```

### "Puppeteer fails"
Install manually:
```bash
npm install puppeteer
npx puppeteer browsers install chrome
```

## Manual Alternative

If automation fails, you can do it manually:

1. Open http://localhost:3000 in your browser
2. Copy the research goal from `RESEARCH_AUTOMATION_README.md`
3. Paste into the chat interface
4. Press Enter

## Need Help?

- **Full documentation**: See `RESEARCH_AUTOMATION_README.md`
- **OpenHands docs**: See `WARP.md`
- **Server logs**: `tail -f server.log`
- **Automation logs**: `tail -f research_automation.log`

## One-Liner (After Python 3.12 is installed)

```bash
cd /Users/wuy/Desktop/code/UAgent/OpenHands && \
make build && \
make setup-config && \
nohup make run > server.log 2>&1 & \
sleep 60 && \
./run_research_automation.sh
```

---

**Note**: This is a complex research task that may take 1-3 hours to complete. The system will work autonomously, exploring different approaches and documenting everything along the way.
