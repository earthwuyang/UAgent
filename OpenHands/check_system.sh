#!/bin/bash
# System Check Script - Verify all dependencies before starting

set +e  # Don't exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  OpenHands System Check                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0

# Check Python
echo -e "${BLUE}[1/5]${NC} Checking Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found!"
    ERRORS=$((ERRORS + 1))
fi

# Check Node.js (optional, only if frontend needed)
echo ""
echo -e "${BLUE}[2/5]${NC} Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} Node.js: $NODE_VERSION"
else
    echo -e "${YELLOW}⚠${NC} Node.js not found (optional, needed only for frontend)"
fi

if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} npm: $NPM_VERSION"
else
    echo -e "${YELLOW}⚠${NC} npm not found (optional, needed only for frontend)"
fi

# Check Extension
echo ""
echo -e "${BLUE}[3/5]${NC} Checking UAgent Research Extension..."
if [ -d "extensions/uagent_research/uagent_research" ]; then
    echo -e "${GREEN}✓${NC} Extension directory exists"

    # Try to import it
    if python -c "import sys; sys.path.insert(0, 'extensions/uagent_research'); from uagent_research.api import router" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Extension can be imported"
    else
        echo -e "${RED}✗${NC} Extension cannot be imported"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗${NC} Extension not found at extensions/uagent_research/"
    ERRORS=$((ERRORS + 1))
fi

# Check Frontend
echo ""
echo -e "${BLUE}[4/5]${NC} Checking Frontend..."
if [ -d "frontend/build" ]; then
    echo -e "${GREEN}✓${NC} Frontend build exists"
else
    echo -e "${YELLOW}⚠${NC} Frontend not built (run: cd frontend && npm run build)"
fi

if [ -d "frontend/node_modules" ]; then
    echo -e "${GREEN}✓${NC} Frontend dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} Frontend dependencies not installed (run: cd frontend && npm install)"
fi

# Check Port
echo ""
echo -e "${BLUE}[5/5]${NC} Checking Port..."
PORT=${PORT:-3000}
if lsof -i:$PORT &> /dev/null; then
    echo -e "${YELLOW}⚠${NC} Port $PORT is already in use"
    echo "    Process using port:"
    lsof -i:$PORT | grep LISTEN
else
    echo -e "${GREEN}✓${NC} Port $PORT is available"
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    echo ""
    echo "You can now start the server with:"
    echo "  ./start_openhands_research.sh"
else
    echo -e "${RED}✗ Found $ERRORS critical error(s)${NC}"
    echo ""
    echo "Please fix the errors above before starting."
    exit 1
fi
echo "════════════════════════════════════════════════════════════"
