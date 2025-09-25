#!/bin/bash

# OpenHands Command Output Monitor
# This script helps monitor real-time output from OpenHands commands

WORKSPACE_DIR="${1:-/home/wuy/AI/uagent-workspace}"
LOG_DIR="$WORKSPACE_DIR/uagent_workspaces/*/logs/commands"

echo "=================================="
echo "OpenHands Real-Time Output Monitor"
echo "=================================="
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo ""

# Function to show latest logs
show_latest_logs() {
    echo "Latest command logs:"
    echo "-------------------"
    find $WORKSPACE_DIR -name "*.log" -type f -path "*/logs/commands/*" -mtime -1 2>/dev/null | \
        xargs ls -lt 2>/dev/null | head -10
    echo ""
}

# Function to monitor specific log
monitor_log() {
    local log_file="$1"
    if [ -f "$log_file" ]; then
        echo "Monitoring: $log_file"
        echo "Press Ctrl+C to stop"
        echo ""
        tail -f "$log_file" | while read LINE; do
            # Color code the output
            case "$LINE" in
                *ERROR*|*FAIL*|*Exception*)
                    echo -e "\033[31m$LINE\033[0m"  # Red
                    ;;
                *SUCCESS*|*COMPLETE*|*successfully*)
                    echo -e "\033[32m$LINE\033[0m"  # Green
                    ;;
                *WARNING*|*timeout*)
                    echo -e "\033[33m$LINE\033[0m"  # Yellow
                    ;;
                *"pip install"*|*"npm install"*|*downloading*|*installing*)
                    echo -e "\033[36m$LINE\033[0m"  # Cyan
                    ;;
                *"--- "*"---"*)
                    echo -e "\033[35m$LINE\033[0m"  # Magenta for sections
                    ;;
                *)
                    echo "$LINE"
                    ;;
            esac
        done
    else
        echo "Log file not found: $log_file"
    fi
}

# Function to monitor all pip/npm installs
monitor_installs() {
    echo "Monitoring all package installations..."
    echo "--------------------------------------"
    find $WORKSPACE_DIR -name "*pip_install*.log" -o -name "*npm_install*.log" -type f -mtime -1 2>/dev/null | while read log; do
        echo "Found: $log"
        tail -n 5 "$log"
        echo "---"
    done
}

# Main menu
while true; do
    echo "Options:"
    echo "  1. Show latest logs"
    echo "  2. Monitor latest log"
    echo "  3. Monitor specific log file"
    echo "  4. Monitor all installations"
    echo "  5. Search for errors in logs"
    echo "  6. Exit"
    echo ""
    read -p "Select option (1-6): " choice

    case $choice in
        1)
            show_latest_logs
            ;;
        2)
            LATEST_LOG=$(find $WORKSPACE_DIR -name "*.log" -type f -path "*/logs/commands/*" -mtime -1 2>/dev/null | \
                xargs ls -t 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                monitor_log "$LATEST_LOG"
            else
                echo "No recent logs found"
            fi
            ;;
        3)
            read -p "Enter log file path: " log_path
            monitor_log "$log_path"
            ;;
        4)
            monitor_installs
            ;;
        5)
            echo "Searching for errors..."
            find $WORKSPACE_DIR -name "*.log" -type f -path "*/logs/commands/*" -mtime -1 2>/dev/null | \
                xargs grep -l "ERROR\|FAIL\|Exception\|timeout" 2>/dev/null | while read log; do
                echo ""
                echo "Errors in: $log"
                grep -E "ERROR|FAIL|Exception|timeout" "$log" | tail -5
            done
            ;;
        6)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
    echo ""
    echo "Press Enter to continue..."
    read
    clear
done