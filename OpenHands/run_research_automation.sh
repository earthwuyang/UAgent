#!/bin/bash
# Run OpenHands Research Automation
# This script sets up and runs the Puppeteer automation for the research task

set -e

cd "$(dirname "$0")"

echo "================================================================================"
echo "  OpenHands Research Automation Setup"
echo "================================================================================"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/v//' | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version 18 or higher is required. Current version: $(node --version)"
    exit 1
fi

echo "✓ Node.js version: $(node --version)"
echo ""

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed."
    exit 1
fi

echo "✓ npm version: $(npm --version)"
echo ""

# Install Puppeteer if not already installed
if [ ! -d "node_modules/puppeteer" ]; then
    echo "📦 Installing Puppeteer..."
    npm install puppeteer
    echo ""
fi

echo "✓ Puppeteer is installed"
echo ""

# Check if OpenHands server is running
echo "🔍 Checking if OpenHands server is running on port 3000..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200\|302\|404"; then
    echo "✓ OpenHands server is running"
else
    echo ""
    echo "⚠️  OpenHands server is NOT running on port 3000"
    echo ""
    echo "Please start the OpenHands server first:"
    echo ""
    echo "  Option 1: Using make (recommended)"
    echo "    cd $(pwd)"
    echo "    make run"
    echo ""
    echo "  Option 2: Using the startup script"
    echo "    cd $(pwd)"
    echo "    ./start_openhands_research.sh"
    echo ""
    echo "  Option 3: Manual startup"
    echo "    cd $(pwd)"
    echo "    make start-backend  # In one terminal"
    echo "    make start-frontend # In another terminal"
    echo ""
    read -p "Would you like to wait and retry? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    echo "Waiting 30 seconds for server to start..."
    sleep 30
    
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200\|302\|404"; then
        echo "✓ OpenHands server is now running"
    else
        echo "❌ Server is still not running. Please start it manually."
        exit 1
    fi
fi

echo ""
echo "================================================================================"
echo "  Starting Research Automation"
echo "================================================================================"
echo ""
echo "The automation will:"
echo "  1. Open a browser window"
echo "  2. Navigate to http://localhost:3000"
echo "  3. Send the research goal about PostgreSQL + pg_duckdb ML-based routing"
echo "  4. Monitor progress for 5 minutes with periodic screenshots"
echo ""
echo "Logs will be saved to: $(pwd)/research_automation.log"
echo "Screenshots will be saved to: $(pwd)/screenshot_*.png"
echo ""
echo "Press Ctrl+C to stop monitoring (research will continue in background)"
echo ""
echo "================================================================================"
echo ""

# Make the script executable
chmod +x start_research_task.js

# Run the automation
node start_research_task.js

echo ""
echo "================================================================================"
echo "  Automation Complete"
echo "================================================================================"
echo ""
echo "Check the logs and screenshots in: $(pwd)"
echo ""
