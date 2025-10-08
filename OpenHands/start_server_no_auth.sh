#!/bin/bash
# Server startup script with SESSION_API_KEY explicitly unset
# This ensures WebSocket connections work without authentication

echo "Starting OpenHands server without SESSION_API_KEY authentication..."
echo ""

# Explicitly unset SESSION_API_KEY
unset SESSION_API_KEY

# Change to project directory
cd /home/wuy/AI/UAgent/OpenHands

# IMPORTANT: Must use openhands.server.listen:app (not openhands.server.app:app)
# The listen module wraps the FastAPI app with Socket.IO support
echo "Using: openhands.server.listen:app (with Socket.IO wrapper)"
echo "Listening on: 0.0.0.0:3000"
echo ""

# Start the server with the correct module
python -m openhands.server

