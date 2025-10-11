# Script Migration Notes

## Summary
Moved `start_openhands_research.sh` from `OpenHands/` to `UAgent/` root directory.

## What Changed

### File Paths
| Item | Before | After |
|------|--------|-------|
| Script location | `UAgent/OpenHands/start_openhands_research.sh` | `UAgent/start_openhands_research.sh` |
| .env file | `../.env` (relative) | `.env` (relative) |
| .venv directory | `../.venv` | `.venv` |
| Working directory | Script runs from `OpenHands/` | Starts in `UAgent/`, then `cd` to `OpenHands/` |

### Key Updates

1. **SCRIPT_DIR variable added**:
   ```bash
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   cd "$SCRIPT_DIR"
   ```
   This ensures the script works from any directory.

2. **Environment loading simplified**:
   ```bash
   # Before: source ../.env
   # After:  source .env
   ```

3. **Virtual environment path**:
   ```bash
   # Before: ../.venv/bin/activate
   # After:  .venv/bin/activate
   ```

4. **Server startup**:
   ```bash
   # Added at end:
   cd "$SCRIPT_DIR/OpenHands"
   exec python -m openhands.server
   ```

### Port Configuration
The script now properly reads `OPENHANDS_PORT` from `.env`:
- Default: 3000
- Your `.env` setting: 3001
- Server will run on: http://localhost:3001

## Testing
To verify the new script works:
```bash
cd /Users/wuy/Desktop/code/UAgent
./start_openhands_research.sh
```

Expected output:
```
✓ Loading environment from .env
✓ Using .venv environment
Configuration:
  • Port:          3001
  ...
```

## Cleanup
After confirming the new script works, remove the old one:
```bash
rm /Users/wuy/Desktop/code/UAgent/OpenHands/start_openhands_research.sh
```
