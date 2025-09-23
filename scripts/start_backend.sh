#!/usr/bin/env bash
# Helper to launch the UAgent backend with environment overrides.

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

BACKEND_HOST="${BACKEND_HOST:-${UAGENT_BACKEND_HOST:-127.0.0.1}}"
BACKEND_PORT="${BACKEND_PORT:-${UAGENT_BACKEND_PORT:-}}"
BACKEND_RELOAD="${BACKEND_RELOAD:-${UAGENT_BACKEND_RELOAD:-true}}"

if [[ -z "$BACKEND_PORT" ]]; then
  echo "[start_backend] ERROR: BACKEND_PORT (or UAGENT_BACKEND_PORT) must be set in the environment." >&2
  echo "[start_backend] Hint: add BACKEND_PORT to your .env file." >&2
  exit 1
fi

cd "$ROOT_DIR/backend"

UVICORN_ARGS=(app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT")
if [[ "$BACKEND_RELOAD" == "true" ]]; then
  UVICORN_ARGS+=(--reload)
fi

echo "[start_backend] Running uvicorn ${UVICORN_ARGS[*]}"
exec uvicorn "${UVICORN_ARGS[@]}"
