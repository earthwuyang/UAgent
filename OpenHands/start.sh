#!/bin/bash
# Simple startup script for OpenHands with UAgent Research Extension

cd /home/wuy/AI/UAgent/OpenHands

echo "=================================================="
echo "  OpenHands + UAgent Research Extension"
echo "=================================================="
echo ""

# Check if extension exists
echo "[1/2] Checking research extension..."
if [ -d "extensions/uagent_research/uagent_research" ]; then
    echo "✓ Extension found (loaded from source)"
else
    echo "✗ Extension not found at extensions/uagent_research/"
    exit 1
fi

# Set environment
export RESEARCH_DATABASE_URL="${RESEARCH_DATABASE_URL:-sqlite+aiosqlite:///./openhands_research.db}"
export SERVE_FRONTEND="${SERVE_FRONTEND:-true}"  # Enable frontend by default
export port="${PORT:-3000}"  # Note: OpenHands uses lowercase 'port'

echo ""
echo "[2/2] Starting server on port $port..."
echo ""
echo "Available at:"
echo "  • Research API: http://localhost:$port/api/research"
echo "  • Health Check: http://localhost:$port/api/research/health"
echo ""
if [ "$SERVE_FRONTEND" = "true" ]; then
    echo "  • Frontend UI:  http://localhost:$port"
    echo ""
    if [ ! -d "frontend/build" ]; then
        echo "⚠️  WARNING: Frontend not built!"
        echo "   Run: cd frontend && npm run build"
        echo ""
    fi
else
    echo "ℹ️  Frontend UI disabled (API only mode)"
    echo "   To enable: SERVE_FRONTEND=true ./start.sh"
    echo ""
fi
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

# Start server using the proper method
python -m openhands.server
