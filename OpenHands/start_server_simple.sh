#!/bin/bash
# Simple OpenHands Server Starter

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting OpenHands Server (Simple Mode)"
echo "=========================================="
echo ""

# Activate venv
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    echo "✓ Activated venv: $(which python)"
    echo "✓ Python version: $(python --version)"
else
    echo "❌ Virtual environment not found at ../.venv"
    exit 1
fi

# Set PYTHONPATH to include OpenHands directory
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "✓ PYTHONPATH: $PYTHONPATH"
echo ""

# Install any missing critical dependencies
echo "Checking dependencies..."
pip install -q docker toml termcolor pexpect tenacity browsergym-core >/dev/null 2>&1 || echo "Some dependencies may be missing"

# Test import
python -c "import sys; sys.path.insert(0, '$(pwd)'); from openhands.server import listen; print('✓ Import test successful')" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Failed to import openhands module"
    echo "Trying to run server anyway..."
fi

echo ""
echo "Starting uvicorn server..."
echo "Server will be available at: http://localhost:3000"
echo ""

# Run uvicorn
exec uvicorn openhands.server.listen:app \
    --host 0.0.0.0 \
    --port 3000 \
    --log-level info
