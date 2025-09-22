#!/usr/bin/env bash
# Helper to launch the UAgent backend with environment overrides.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/backend/.env"

ENV_FILE="${1:-$DEFAULT_ENV_FILE}"

if [[ -f "$ENV_FILE" ]]; then
  echo "[start_backend] Loading environment from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

UAGENT_BACKEND_HOST="${UAGENT_BACKEND_HOST:-127.0.0.1}"
UAGENT_BACKEND_PORT="${UAGENT_BACKEND_PORT:-8001}"
UAGENT_BACKEND_RELOAD="${UAGENT_BACKEND_RELOAD:-true}"

cd "$ROOT_DIR/backend"

UVICORN_ARGS=(app.main:app --host "$UAGENT_BACKEND_HOST" --port "$UAGENT_BACKEND_PORT")
if [[ "$UAGENT_BACKEND_RELOAD" == "true" ]]; then
  UVICORN_ARGS+=(--reload)
fi

echo "[start_backend] Running uvicorn ${UVICORN_ARGS[*]}"
exec uvicorn "${UVICORN_ARGS[@]}"
