#!/bin/bash
# OpenHands with UAgent Research Extension - Startup Script

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "════════════════════════════════════════════════════"
echo "  OpenHands + UAgent Research Extension"
echo "════════════════════════════════════════════════════"
echo ""

# Activate .venv
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
    echo "✓ Using .venv environment"
else
    echo "✗ .venv not found, using system Python"
fi

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

# If SANDBOX_VOLUMES is not provided by the user, auto-create a
# unique workspace under WORKSPACE_BASE and mount it at /workspace:rw
if [ -z "${SANDBOX_VOLUMES:-}" ]; then
    base_dir="${WORKSPACE_BASE:-./workspace}"
    mkdir -p "$base_dir" 2>/dev/null || true

    # Prefer mktemp for safe unique directory creation; fall back to timestamp
    if ws_dir="$(mktemp -d -p "$base_dir" openhands_ws_XXXXXX 2>/dev/null)"; then
        : # ws_dir set by mktemp
    else
        ts="$(date +%Y%m%d_%H%M%S)"
        ws_dir="$base_dir/ws_${ts}_${RANDOM}"
        mkdir -p "$ws_dir"
    fi

    # Ensure workspace is writable to all (container may run with different UID)
    chmod g+rwX,o+rwX "$ws_dir" 2>/dev/null || true

    export SANDBOX_VOLUMES="${ws_dir}:/workspace:rw"
    # Ensure UAgent research extension uses the same host workspace
    export UAGENT_WORKSPACE_DIR="${ws_dir}"
    echo "→ SANDBOX_VOLUMES not set; auto-created workspace: $ws_dir"
    echo "  Mounting into container as: /workspace (rw)"
    echo "  UAGENT_WORKSPACE_DIR: $UAGENT_WORKSPACE_DIR"
else
    echo "→ Using user-provided SANDBOX_VOLUMES: $SANDBOX_VOLUMES"
    # If UAGENT_WORKSPACE_DIR is not set, try to infer it from the
    # first mount that targets /workspace with an absolute host path.
    if [ -z "${UAGENT_WORKSPACE_DIR:-}" ]; then
        inferred_ws=""
        IFS=',' read -ra __mounts <<< "$SANDBOX_VOLUMES"
        for __m in "${__mounts[@]}"; do
            IFS=':' read -r __host __container __mode <<< "$__m"
            if [ "${__container}" = "/workspace" ] && [ -n "${__host}" ] && [[ "${__host}" = /* ]]; then
                # Resolve to absolute path if possible
                if command -v realpath >/dev/null 2>&1; then
                    inferred_ws="$(realpath -m "${__host}")"
                elif command -v readlink >/dev/null 2>&1; then
                    inferred_ws="$(readlink -f "${__host}" 2>/dev/null || echo "${__host}")"
                else
                    inferred_ws="${__host}"
                fi
                break
            fi
        done
        if [ -n "$inferred_ws" ]; then
            export UAGENT_WORKSPACE_DIR="$inferred_ws"
            echo "  Inferred UAGENT_WORKSPACE_DIR from SANDBOX_VOLUMES: $UAGENT_WORKSPACE_DIR"
        else
            echo "  Note: Could not infer UAGENT_WORKSPACE_DIR from SANDBOX_VOLUMES."
            echo "        If you want UAgent research artifacts co-located, set UAGENT_WORKSPACE_DIR to your host workspace."
        fi
    fi
fi

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
echo "  • Workspace base: $WORKSPACE_BASE"
echo "  • Host workspace: ${UAGENT_WORKSPACE_DIR:-<auto/inferred or not set>}"
echo "  • Workspace Mount: ${SANDBOX_VOLUMES:-<none>}"
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
