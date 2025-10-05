#!/bin/bash
# Test startup script
cd /home/wuy/AI/UAgent/OpenHands

echo "Testing backend startup..."
echo ""

# Start backend in background
./start_backend_only.sh &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to start..."
sleep 10

# Test if server is running
if curl -s http://localhost:3000/api/research/health > /dev/null 2>&1; then
    echo "✅ Server started successfully!"
    curl -s http://localhost:3000/api/research/health | head -5
else
    echo "❌ Server failed to start"
fi

# Kill server
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "Done!"
