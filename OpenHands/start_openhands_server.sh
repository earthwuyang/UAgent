#!/bin/bash

# Kill any existing server process
pkill -9 -f "uvicorn openhands.server.listen:app"

# Start the OpenHands server
cd /Users/wuy/Desktop/code/UAgent/OpenHands
nohup /Users/wuy/Desktop/code/UAgent/.venv/bin/python -m uvicorn openhands.server.listen:app --host 0.0.0.0 --port 3000 > openhands_server.log 2>&1 &

echo "OpenHands server starting..."
echo "PID: $!"
echo "Check logs at: openhands_server.log"
