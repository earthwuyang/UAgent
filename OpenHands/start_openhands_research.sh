#!/bin/bash
# OpenHands with UAgent Research Extension - Startup Script

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "════════════════════════════════════════════════════"
echo "  OpenHands + UAgent Research Extension"
echo "════════════════════════════════════════════════════"
echo ""

# ============================================================================
# Environment Variables Configuration
# ============================================================================

# Server Configuration
export SERVE_FRONTEND="${SERVE_FRONTEND:-true}"
export port="${port:-3000}"

# Research Extension Database
export RESEARCH_DATABASE_URL="${RESEARCH_DATABASE_URL:-sqlite+aiosqlite:///./openhands_research.db}"

# Workspace Configuration
export WORKSPACE_BASE="${WORKSPACE_BASE:-./workspace}"

# Sandbox/Runtime Configuration
# Using openhands-uagent:v0.1 - will auto-pull from earthwuyang/openhands-uagent:v0.1 or build from Dockerfile.runtime-fixed if needed
export SANDBOX_RUNTIME_CONTAINER_IMAGE="${SANDBOX_RUNTIME_CONTAINER_IMAGE:-openhands-uagent:v0.1}"
export SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-120}"
export SANDBOX_USER_ID="${SANDBOX_USER_ID:-$(id -u 2>/dev/null || echo 1000)}"

# LLM Configuration (default to DashScope/Qwen)
export LLM_MODEL="${LLM_MODEL:-dashscope/qwen3-coder-plus}"
export LLM_API_KEY="${LLM_API_KEY:-${DASHSCOPE_API_KEY}}"
export LLM_BASE_URL="${LLM_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

# Agent Configuration
# export AGENT_MEMORY_ENABLED="${AGENT_MEMORY_ENABLED:-true}"

# Other Settings
export MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
export ENABLE_AUTO_LINT="${ENABLE_AUTO_LINT:-false}"

# ============================================================================

echo "Configuration:"
echo "  • Port:          $port"
echo "  • Workspace:     $WORKSPACE_BASE"
echo "  • Runtime Image: $SANDBOX_RUNTIME_CONTAINER_IMAGE"
echo "  • Research DB:   $RESEARCH_DATABASE_URL"
echo ""
echo "Available at:"
echo "  • Main UI:       http://localhost:$port"
echo "  • Research API:  http://localhost:$port/api/research"
echo "  • WebSocket:     ws://localhost:$port/api/research/ws"
echo ""
echo "Press Ctrl+C to stop"
echo "════════════════════════════════════════════════════"
echo ""

# Start server
exec python -m openhands.server
