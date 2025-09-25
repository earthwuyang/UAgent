#!/bin/bash

# Simple script to list and monitor OpenHands output logs

WORKSPACE_DIR="${1:-/home/wuy/AI/uagent-workspace}"

echo "=================================="
echo "OpenHands Command Logs"
echo "=================================="
echo ""

# Find all log files created today
echo "Today's command logs:"
find $WORKSPACE_DIR -name "*.log" -o -name "*.realtime" -type f -path "*/logs/commands/*" -mtime 0 2>/dev/null | \
    xargs ls -lt 2>/dev/null | head -20

echo ""
echo "Real-time streaming logs (*.realtime):"
find $WORKSPACE_DIR -name "*.realtime" -type f -mtime 0 2>/dev/null | \
    xargs ls -lt 2>/dev/null | head -10

echo ""
echo "To monitor a log in real-time, use:"
echo "  tail -f <log_file>"
echo ""
echo "To see the latest pip install log:"
latest_pip=$(find $WORKSPACE_DIR -name "*pip*.realtime" -type f -mtime 0 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
if [ -n "$latest_pip" ]; then
    echo "  tail -f $latest_pip"
else
    echo "  No recent pip install logs found"
fi

echo ""
echo "To search for errors:"
echo "  grep -r 'ERROR\|FAIL' $WORKSPACE_DIR/*/logs/commands/*.log"