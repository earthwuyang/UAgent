#!/usr/bin/env bash
# Helper to launch the UAgent frontend with environment overrides.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ENV_FILE="$ROOT_DIR/frontend/.env.local"

ENV_FILE="${1:-$DEFAULT_ENV_FILE}"

if [[ -f "$ENV_FILE" ]]; then
  echo "[start_frontend] Loading environment from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

VITE_DEV_HOST="${VITE_DEV_HOST:-0.0.0.0}"
VITE_DEV_PORT="${VITE_DEV_PORT:-3000}"

cd "$ROOT_DIR/frontend"

echo "[start_frontend] Running npm run dev -- --host $VITE_DEV_HOST --port $VITE_DEV_PORT"
exec npm run dev -- --host "$VITE_DEV_HOST" --port "$VITE_DEV_PORT"
