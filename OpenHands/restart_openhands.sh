#!/bin/bash
echo "═══════════════════════════════════════════════════════════════════════"
echo "                 🔄 RESTARTING OPENHANDS SERVER"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

echo "1️⃣  Stopping OpenHands server..."
pkill -f "python -m openhands.server" && echo "   ✅ Server stopped" || echo "   ℹ️  No server was running"
sleep 2

echo ""
echo "2️⃣  Cleaning up all runtime containers..."
docker stop $(docker ps -q --filter ancestor=openhands-uagent:v0.1) 2>/dev/null && echo "   ✅ Containers stopped" || echo "   ℹ️  No containers to stop"
docker rm $(docker ps -aq --filter ancestor=openhands-uagent:v0.1) 2>/dev/null && echo "   ✅ Containers removed" || echo "   ℹ️  No containers to remove"

echo ""
echo "3️⃣  Starting OpenHands server..."
cd /home/wuy/AI/UAgent/OpenHands
nohup poetry run python -m openhands.server > /tmp/openhands_server.log 2>&1 &
SERVER_PID=$!
echo "   ✅ Server started (PID: $SERVER_PID)"

echo ""
echo "4️⃣  Waiting for server to be ready..."
sleep 5

echo ""
echo "5️⃣  Checking status..."
if curl -s http://localhost:3000/ > /dev/null 2>&1; then
    echo "   ✅ Server is responding!"
else
    echo "   ⚠️  Server not responding yet, check logs:"
    echo "      tail -f /tmp/openhands_server.log"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "                           ✅ COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "🌐 Open your browser to: http://120.46.207.248:3000/"
echo ""
echo "📋 To view logs:"
echo "   tail -f /tmp/openhands_server.log"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
