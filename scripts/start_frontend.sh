#!/usr/bin/env bash
# Helper to launch the UAgent frontend with environment overrides.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ENV_FILE="$ROOT_DIR/.env"
LEGACY_ENV_FILE="$ROOT_DIR/frontend/.env.local"
CUSTOM_ENV_FILE="${1:-}"

load_env_file() {
  local file="$1"
if [[ -f "$file" ]]; then
    echo "[start_frontend] Loading environment from $file"
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

VITE_DEV_HOST="${VITE_DEV_HOST:-${DEV_HOST:-0.0.0.0}}"
VITE_DEV_PORT="${VITE_DEV_PORT:-${VITE_DEV_SERVER_PORT:-${PORT:-3000}}}"

cd "$ROOT_DIR/frontend"

echo "[start_frontend] Running npm run dev -- --host $VITE_DEV_HOST --port $VITE_DEV_PORT"
exec npm run dev -- --host "$VITE_DEV_HOST" --port "$VITE_DEV_PORT"
