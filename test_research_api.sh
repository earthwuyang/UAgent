#!/bin/bash
# Test UAgent Research API

set -e

BASE_URL="http://localhost:3000"
API_URL="$BASE_URL/api/research"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Testing UAgent Research API                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}[1/5]${NC} Testing server health..."
if curl -s "$BASE_URL/api/health" > /dev/null; then
    echo -e "${GREEN}✓${NC} OpenHands server is running"
else
    echo -e "${RED}✗${NC} OpenHands server is not running!"
    echo "Start it with: cd /home/wuy/AI/UAgent/OpenHands && ./start_openhands_research.sh"
    exit 1
fi

echo ""
echo -e "${BLUE}[2/5]${NC} Testing research extension health..."
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✓${NC} Research extension is healthy"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗${NC} Research extension health check failed!"
    echo "Response: $HEALTH_RESPONSE"
    exit 1
fi

echo ""
echo -e "${BLUE}[3/5]${NC} Testing list experiments..."
EXPERIMENTS_RESPONSE=$(curl -s "$API_URL/experiments")
echo -e "${GREEN}✓${NC} Can list experiments"
echo "Response: $EXPERIMENTS_RESPONSE"

echo ""
echo -e "${BLUE}[4/5]${NC} Testing create experiment..."
CREATE_RESPONSE=$(curl -s -X POST "$API_URL/experiments/start" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "API Test - Compare sorting algorithms",
    "session_id": "test_session_'$(date +%s)'",
    "research_type": "scientific"
  }')

if echo "$CREATE_RESPONSE" | grep -q "id"; then
    echo -e "${GREEN}✓${NC} Experiment created successfully"
    EXPERIMENT_ID=$(echo "$CREATE_RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo "Experiment ID: $EXPERIMENT_ID"
else
    echo -e "${RED}✗${NC} Failed to create experiment"
    echo "Response: $CREATE_RESPONSE"
    exit 1
fi

echo ""
echo -e "${BLUE}[5/5]${NC} Testing get experiment..."
GET_RESPONSE=$(curl -s "$API_URL/experiments/$EXPERIMENT_ID")
if echo "$GET_RESPONSE" | grep -q "$EXPERIMENT_ID"; then
    echo -e "${GREEN}✓${NC} Can retrieve experiment"
    echo "Status: $(echo "$GET_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)"
else
    echo -e "${RED}✗${NC} Failed to retrieve experiment"
    echo "Response: $GET_RESPONSE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✅ All API tests passed!${NC}"
echo ""
echo "Available endpoints:"
echo "  • Health:          $API_URL/health"
echo "  • List:            $API_URL/experiments"
echo "  • Create:          $API_URL/experiments/start"
echo "  • Get:             $API_URL/experiments/{id}"
echo "  • WebSocket:       ws://localhost:3000/api/research/ws/experiment/{id}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
