#!/usr/bin/env bash
set -euo pipefail

# Simple smoke test to verify the OpenHands server is reachable before running experiments.

HOST="${OPENHANDS_HOST:-127.0.0.1}"
PORT="${OPENHANDS_PORT:-3000}"
BASE_URL="http://${HOST}:${PORT}"

echo "Checking OpenHands health at ${BASE_URL}/api/health ..."
if ! curl -sf "${BASE_URL}/api/health" >/dev/null; then
  echo "OpenHands health check failed." >&2
  exit 1
fi

echo "Health check passed."
# Extend here with headless/CLI smoke actions as needed.
