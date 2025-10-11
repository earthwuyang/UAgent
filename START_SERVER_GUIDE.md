# OpenHands Research Server - Start Guide

## Script Location

The startup script has been moved to the **UAgent root directory**:
- **New location**: `/Users/wuy/Desktop/code/UAgent/start_openhands_research.sh`
- **Old location**: `~/Desktop/code/UAgent/OpenHands/start_openhands_research.sh` (can be removed)

## Key Changes

### 1. Environment Variable Loading
The script now loads `.env` from the **UAgent directory** (same directory as the script):
```bash
# Loads: /Users/wuy/Desktop/code/UAgent/.env
source .env
```

### 2. Port Configuration
The server port is now controlled by `OPENHANDS_PORT` in `.env`:
```bash
# In .env file:
OPENHANDS_PORT=3001

# Server will run on port 3001 (not the default 3000)
```

### 3. Path Resolution
The script properly handles all paths relative to the UAgent directory:
- `.env` → `./env` (UAgent/.env)
- `.venv` → `./.venv` (UAgent/.venv)
- Changes to `OpenHands/` directory before starting the server

## Usage

### Start the Server
```bash
cd /Users/wuy/Desktop/code/UAgent
./start_openhands_research.sh
```

Or from any directory:
```bash
/Users/wuy/Desktop/code/UAgent/start_openhands_research.sh
```

### Access the UI
With `OPENHANDS_PORT=3001` in your `.env`:
- Main UI: http://localhost:3001
- Research API: http://localhost:3001/api/research
- WebSocket: ws://localhost:3001/api/research/ws

### Environment Variables
All environment variables from `UAgent/.env` are automatically loaded:
- `OPENHANDS_PORT` - Server port (default: 3000)
- `LLM_API_KEY` - API key for LLM
- `LLM_MODEL` - Model to use
- `WORKSPACE_BASE` - Workspace directory
- `RESEARCH_MAX_ITERATIONS` - Max research iterations
- And all other variables in `.env`

## Dependencies Handled

The script correctly handles:
1. **Virtual environment**: Activates `UAgent/.venv`
2. **Environment file**: Loads `UAgent/.env`
3. **Working directory**: Changes to `UAgent/OpenHands` before starting server
4. **Python path**: Uses the activated venv's Python with proper module resolution

## Troubleshooting

### Port Already in Use
If you get "Address already in use" error:
```bash
# Check what's using the port
lsof -i :3001

# Kill the process
pkill -9 -f "openhands.server"
```

### Script Not Found
Make sure you're in the correct directory:
```bash
cd /Users/wuy/Desktop/code/UAgent
ls -la start_openhands_research.sh
```

### Permission Denied
Make the script executable:
```bash
chmod +x /Users/wuy/Desktop/code/UAgent/start_openhands_research.sh
```

## Old Script
The old script in `OpenHands/start_openhands_research.sh` can now be removed as it's no longer needed:
```bash
rm /Users/wuy/Desktop/code/UAgent/OpenHands/start_openhands_research.sh
```
