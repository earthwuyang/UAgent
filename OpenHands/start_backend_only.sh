#!/bin/bash
# OpenHands with UAgent Research Extension - Backend Only (No Frontend UI)

set +e  # Don't exit on error for checks

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  OpenHands Research Backend (API Only)                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}[1/3]${NC} Checking extension..."
if [ ! -d "extensions/uagent_research" ]; then
    echo -e "${RED}✗${NC} UAgent Research Extension not found!"
    exit 1
fi
echo -e "${GREEN}✓${NC} Extension found"

echo ""
echo -e "${BLUE}[2/3]${NC} Checking installation..."
if python -c "import uagent_research" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Extension already installed"
else
    echo "Installing extension (first time only)..."
    cd extensions/uagent_research
    pip install -e . -q
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Extension installed"
    else
        echo -e "${YELLOW}⚠${NC} Extension installation completed with warnings"
    fi
    cd "$SCRIPT_DIR"
fi

echo ""
echo -e "${BLUE}[3/3]${NC} Starting backend..."

# Set database URL and disable frontend
export RESEARCH_DATABASE_URL="${RESEARCH_DATABASE_URL:-sqlite+aiosqlite:///./openhands_research.db}"
export SERVE_FRONTEND=false
export port=${PORT:-3000}  # OpenHands uses lowercase 'port'
PORT=$port  # Keep for display

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}Backend API starting on port $PORT...${NC}"
echo ""
echo -e "${GREEN}Available at:${NC}"
echo -e "  • Research API:  ${BLUE}http://localhost:$PORT/api/research${NC}"
echo -e "  • WebSocket:     ${BLUE}ws://localhost:$PORT/api/research/ws${NC}"
echo -e "  • Health:        ${BLUE}http://localhost:$PORT/api/research/health${NC}"
echo ""
echo -e "${YELLOW}Note: Frontend UI disabled (set SERVE_FRONTEND=true to enable)${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Re-enable exit on error
set -e

# Start server
exec python -m openhands.server
