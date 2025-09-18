#!/bin/bash
# Restart Agent Laboratory with improved supervision

echo "🔄 Restarting Agent Laboratory with improved supervision..."

# Check if DASHSCOPE_API_KEY is set
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "❌ DASHSCOPE_API_KEY environment variable is not set!"
    echo "Please set it with: export DASHSCOPE_API_KEY='your-api-key'"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_agent_lab/bin/activate

# Start monitoring in background (optional)
if [ "$1" == "--with-monitor" ]; then
    echo "🔍 Starting monitoring in background..."
    ./monitor_lab.sh MATH_research_dir 20.0 120 > monitor.log 2>&1 &
    MONITOR_PID=$!
    echo "📊 Monitor started with PID: $MONITOR_PID"
    echo "📋 Monitor log: monitor.log"
fi

echo "🚀 Starting Agent Laboratory..."
echo "📁 Config: experiment_configs/MATH_agentlab.yaml"
echo "🤖 Model: qwen3-max-preview"
echo "💰 Cost monitoring: Enabled (threshold: $50)"
echo "=================================="

# Run the Agent Laboratory
python ai_lab_repo.py --yaml-location "experiment_configs/MATH_agentlab.yaml"

# If monitoring was started, stop it
if [ ! -z "$MONITOR_PID" ]; then
    echo "🛑 Stopping monitor..."
    kill $MONITOR_PID 2>/dev/null || true
fi

echo "✅ Agent Laboratory completed."