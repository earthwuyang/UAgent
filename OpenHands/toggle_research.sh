#!/bin/bash
# Toggle Research Auto-Triggering

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/extensions/uagent_research/config.py"

if [ "$1" == "enable" ]; then
    echo "Enabling research auto-triggering..."
    sed -i "s/ENABLE_AUTO_RESEARCH_TRIGGER = False/ENABLE_AUTO_RESEARCH_TRIGGER = True/g" "$CONFIG_FILE"
    echo "✅ Research auto-trigger ENABLED"
    echo "Please restart the OpenHands server for changes to take effect."
    
elif [ "$1" == "disable" ]; then
    echo "Disabling research auto-triggering..."
    sed -i "s/ENABLE_AUTO_RESEARCH_TRIGGER = True/ENABLE_AUTO_RESEARCH_TRIGGER = False/g" "$CONFIG_FILE"
    echo "✅ Research auto-trigger DISABLED"
    echo "Please restart the OpenHands server for changes to take effect."
    
elif [ "$1" == "status" ]; then
    if grep -q "ENABLE_AUTO_RESEARCH_TRIGGER = True" "$CONFIG_FILE"; then
        echo "Research auto-trigger is: ENABLED ✅"
    else
        echo "Research auto-trigger is: DISABLED ⛔"
    fi
    
else
    echo "Usage: $0 {enable|disable|status}"
    echo ""
    echo "Commands:"
    echo "  enable  - Enable automatic research triggering"
    echo "  disable - Disable automatic research triggering"  
    echo "  status  - Check current status"
    exit 1
fi
