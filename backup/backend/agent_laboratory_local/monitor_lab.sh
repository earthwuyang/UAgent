#!/bin/bash
# Monitor Agent Laboratory execution to prevent wasteful dummy implementations

LAB_DIR=${1:-"MATH_research_dir"}
MAX_COST=${2:-25.0}
CHECK_INTERVAL=${3:-180}

echo "🔍 Starting Agent Laboratory Monitor"
echo "📁 Lab Directory: $LAB_DIR"
echo "💰 Max Cost: \$$MAX_COST"
echo "⏰ Check Interval: ${CHECK_INTERVAL}s"
echo "=================================="

# Make sure the supervision script exists
if [ ! -f "supervision.py" ]; then
    echo "❌ supervision.py not found!"
    exit 1
fi

# Find the actual research directory
RESEARCH_DIR=$(find "$LAB_DIR" -name "research_dir_*" -type d | head -1)

if [ -z "$RESEARCH_DIR" ]; then
    echo "❌ No research directory found in $LAB_DIR"
    echo "Waiting for Agent Laboratory to create directory..."

    # Wait for directory to be created
    while [ -z "$RESEARCH_DIR" ]; do
        sleep 10
        RESEARCH_DIR=$(find "$LAB_DIR" -name "research_dir_*" -type d | head -1)
        echo "⏳ Still waiting for research directory..."
    done
fi

echo "✅ Found research directory: $RESEARCH_DIR"

# Start monitoring
python supervision.py "$RESEARCH_DIR" "$MAX_COST" "$CHECK_INTERVAL"