#!/bin/bash

# Simple script to list and monitor OpenHands unified command log

WORKSPACE_DIR="${1:-/home/wuy/AI/uagent-workspace}"

echo "=================================="
echo "OpenHands Unified Command Log"
echo "=================================="
echo ""

# Find the unified commands.log file
echo "Unified command logs:"
find $WORKSPACE_DIR -name "commands.log" -type f -path "*/logs/*" 2>/dev/null | \
    xargs ls -lh 2>/dev/null

echo ""
echo "Background process PIDs:"
find $WORKSPACE_DIR -name "background_pids.txt" -type f -path "*/logs/*" 2>/dev/null | \
    xargs ls -lh 2>/dev/null

echo ""
echo "To monitor the unified log in real-time:"
unified_log=$(find $WORKSPACE_DIR -name "commands.log" -type f -path "*/logs/*" 2>/dev/null | head -1)
if [ -n "$unified_log" ]; then
    echo "  tail -f $unified_log"
    echo ""
    echo "Last 20 lines of the log:"
    echo "------------------------"
    tail -n 20 "$unified_log"
else
    echo "  No unified command log found"
fi

echo ""
echo "To search for errors in the log:"
echo "  grep -E 'ERROR|FAIL|error|failed' $unified_log"
echo ""
echo "To see only pip install commands:"
echo "  grep 'pip install' $unified_log"