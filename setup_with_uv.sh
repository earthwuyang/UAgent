#!/bin/bash
# Quick setup script for UAgent with uv
set -e

echo "════════════════════════════════════════════════════"
echo "  UAgent/OpenHands - Setup with uv"
echo "════════════════════════════════════════════════════"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# Check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
    echo "✗ uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "✓ uv installed successfully"
else
    echo "✓ uv is already installed ($(uv --version))"
fi

# Create virtual environment
if [ -d ".venv" ]; then
    echo "→ Removing old .venv..."
    rm -rf .venv
fi

echo "→ Creating virtual environment with uv..."
uv venv --python 3.12

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment created and activated"

# Install dependencies
echo ""
echo "→ Installing dependencies from requirements.txt..."
echo "  (This should be MUCH faster than pip or Poetry!)"
uv pip install -r requirements.txt

echo ""
echo "════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo "════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Create .env.local with your API keys:"
echo "     cp .env.local.template .env.local"
echo "     # Edit .env.local and add your DASHSCOPE_API_KEY"
echo ""
echo "  2. Start the backend:"
echo "     ./start_openhands_research.sh"
echo ""
echo "  3. Open browser at: http://localhost:2999"
echo ""
