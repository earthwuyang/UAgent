#!/bin/bash
# Monitor UAgent backend for OpenHands integration status

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         UAgent Backend Monitor - $(date +%H:%M:%S)              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"

    echo -e "\n📊 RECENT ACTIVITY (Last 10 lines):"
    echo "─────────────────────────────────────"
    tmux capture-pane -t uagent-backend -p 2>/dev/null | tail -10 | head -10

    echo -e "\n🔧 OPENHANDS INTEGRATION STATUS:"
    echo "─────────────────────────────────────"

    # Check which runner is active
    if tmux capture-pane -t uagent-backend -p 2>/dev/null | grep -q "OpenHandsCompleteRunner"; then
        echo "✅ NEW Integration Active: OpenHandsCompleteRunner (Complete Goal Delegation)"
    elif tmux capture-pane -t uagent-backend -p 2>/dev/null | grep -q "CodeActRunner"; then
        echo "⚠️  OLD Integration Active: CodeActRunner (Step-by-Step - May cause repetitive files)"
    else
        echo "⏸️  No OpenHands activity detected yet"
    fi

    echo -e "\n📁 FILE OPERATIONS (collect_data files):"
    echo "─────────────────────────────────────"
    tmux capture-pane -t uagent-backend -p 2>/dev/null | grep -i "collect_data" | tail -3 || echo "No collect_data operations"

    echo -e "\n⚠️  ERRORS/WARNINGS:"
    echo "─────────────────────────────────────"
    tmux capture-pane -t uagent-backend -p 2>/dev/null | grep -E "ERROR|WARNING|Failed" | tail -3 || echo "No errors detected"

    echo -e "\n🌐 ACTIVE SESSIONS:"
    echo "─────────────────────────────────────"
    tmux capture-pane -t uagent-backend -p 2>/dev/null | grep -E "session_[0-9]+|WebSocket connected" | tail -3 || echo "No active sessions"

    echo -e "\n────────────────────────────────────────────────────────────"
    echo "Press Ctrl+C to stop monitoring | Refreshing every 5 seconds..."

    sleep 5
done