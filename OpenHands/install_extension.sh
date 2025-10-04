#!/bin/bash
# One-time installation of UAgent Research Extension

cd /home/wuy/AI/UAgent/OpenHands

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Installing UAgent Research Extension                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if already installed
if python -c "import uagent_research" 2>/dev/null; then
    echo "✓ Extension already installed!"
    echo ""
    echo "To reinstall:"
    echo "  cd extensions/uagent_research"
    echo "  pip install -e ."
    echo ""
    exit 0
fi

# Install extension
echo "Installing extension..."
cd extensions/uagent_research
pip install -e .

if [ $? -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✅ Installation Complete!                                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "You can now start the server with:"
    echo "  ./start.sh"
    echo ""
    echo "The extension is installed and won't need reinstalling."
    echo ""
else
    echo ""
    echo "❌ Installation failed!"
    echo "Check the error messages above."
    echo ""
    exit 1
fi
