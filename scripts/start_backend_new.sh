#!/usr/bin/env bash
# Enhanced UAgent backend launcher with OpenHands integration control
# This version uses the new complete goal delegation to prevent repetitive file creation

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ENV_FILE="$ROOT_DIR/.env"
LEGACY_ENV_FILE="$ROOT_DIR/backend/.env"
CUSTOM_ENV_FILE="${1:-}"

load_env_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    echo "[start_backend] Loading environment from $file"
    set -a
    set +u
    # shellcheck disable=SC1090
    source "$file"
    set -u
    set +a
  fi
}

load_env_file "$PROJECT_ENV_FILE"
if [[ -n "$CUSTOM_ENV_FILE" ]]; then
  load_env_file "$CUSTOM_ENV_FILE"
else
  load_env_file "$LEGACY_ENV_FILE"
fi

# Source the venv - Use UAgent's own venv
VENV_PATH="$ROOT_DIR/.venv"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
  echo "[start_backend] Activating venv at $VENV_PATH"
  source "$VENV_PATH/bin/activate"
else
  # Fallback to AgentLaboratory venv if UAgent venv doesn't exist
  VENV_PATH="/home/wuy/AI/AgentLaboratory/venv_agent_lab"
  if [[ -f "$VENV_PATH/bin/activate" ]]; then
    echo "[start_backend] Fallback: Activating venv at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
  fi
fi

# Set MOONSHOT credentials directly if not already set
if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then
  # Extract from .bashrc since sourcing doesn't work in non-interactive shell
  MOONSHOT_API_KEY=$(grep 'export MOONSHOT_API_KEY=' ~/.bashrc | sed 's/^export MOONSHOT_API_KEY="//' | sed 's/"$//')
  MOONSHOT_BASE_URL=$(grep 'export MOONSHOT_BASE_URL=' ~/.bashrc | sed 's/^export MOONSHOT_BASE_URL="//' | sed 's/"$//')

  if [[ -n "$MOONSHOT_API_KEY" ]]; then
    echo "[start_backend] Extracted MOONSHOT credentials from .bashrc"
  fi
fi

# Export MOONSHOT credentials for LiteLLM
export MOONSHOT_API_KEY="${MOONSHOT_API_KEY:-}"
export MOONSHOT_BASE_URL="${MOONSHOT_BASE_URL:-https://api.moonshot.cn/v1}"

# Debug: Show MOONSHOT_API_KEY status
echo "[start_backend] MOONSHOT_API_KEY is: ${MOONSHOT_API_KEY:+set (hidden)}"
echo "[start_backend] MOONSHOT_API_KEY empty?: $([ -z "$MOONSHOT_API_KEY" ] && echo yes || echo no)"

# Set LITELLM_API_KEY and LITELLM_API_BASE for Moonshot
if [[ -n "$MOONSHOT_API_KEY" ]]; then
  export LITELLM_API_KEY="$MOONSHOT_API_KEY"
  export LITELLM_API_BASE="$MOONSHOT_BASE_URL"
  echo "[start_backend] Set LITELLM_API_KEY and LITELLM_API_BASE from MOONSHOT"
  echo "[start_backend] LITELLM_API_BASE: $LITELLM_API_BASE"
else
  echo "[start_backend] WARNING: MOONSHOT_API_KEY is empty!"
fi

BACKEND_HOST="${BACKEND_HOST:-${UAGENT_BACKEND_HOST:-127.0.0.1}}"
BACKEND_PORT="${BACKEND_PORT:-${UAGENT_BACKEND_PORT:-}}"
BACKEND_RELOAD="${BACKEND_RELOAD:-${UAGENT_BACKEND_RELOAD:-true}}"

# NEW: Control which OpenHands integration to use
# Set OPENHANDS_MODE=complete for new complete delegation (prevents repetitive files)
# Set OPENHANDS_MODE=codeact for old step-by-step approach
export OPENHANDS_MODE="${OPENHANDS_MODE:-complete}"

if [[ -z "$BACKEND_PORT" ]]; then
  echo "[start_backend] ERROR: BACKEND_PORT (or UAGENT_BACKEND_PORT) must be set in the environment." >&2
  echo "[start_backend] Hint: add BACKEND_PORT to your .env file." >&2
  exit 1
fi

echo "[start_backend] ============================================"
echo "[start_backend] Starting UAgent Backend"
echo "[start_backend] OpenHands Mode: $OPENHANDS_MODE"
if [[ "$OPENHANDS_MODE" == "complete" ]]; then
  echo "[start_backend] ✅ Using NEW Complete Goal Delegation"
  echo "[start_backend]    (Prevents repetitive file creation)"
else
  echo "[start_backend] ⚠️  Using OLD CodeAct Step-by-Step"
  echo "[start_backend]    (May create duplicate files like collect_data_**)"
fi
echo "[start_backend] ============================================"

cd "$ROOT_DIR/backend"

UVICORN_ARGS=(app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT")
if [[ "$BACKEND_RELOAD" == "true" ]]; then
  UVICORN_ARGS+=(--reload)
fi

echo "[start_backend] Running uvicorn ${UVICORN_ARGS[*]}"
exec uvicorn "${UVICORN_ARGS[@]}"