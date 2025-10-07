#!/usr/bin/env bash
set -euo pipefail

# Resolve project root as the directory of this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env (export all keys)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Ensure a sane default if not set in .env
export OPENHANDS_HEADLESS_RUNTIME="${OPENHANDS_HEADLESS_RUNTIME:-local}"
export OPENHANDS_PORT="${OPENHANDS_PORT:-3000}"

echo "[start_openhands_research] Using OPENHANDS_HEADLESS_RUNTIME=${OPENHANDS_HEADLESS_RUNTIME}"
echo "[start_openhands_research] Starting OpenHands on port ${OPENHANDS_PORT}"

# Pick Python interpreter (prefer local venv)
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON="python"
fi

# Start the OpenHands server (FastAPI)
exec "$PYTHON" -m uvicorn openhands.server.app:app --host 0.0.0.0 --port "${OPENHANDS_PORT}"

