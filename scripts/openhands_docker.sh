#!/usr/bin/env bash
set -euo pipefail

# Launch the OpenHands server in Docker with access to the host Docker daemon so it
# can spawn the sandbox runtime container. Intended for Linux/macOS hosts.

if [ -z "${CODE_PATH:-}" ]; then
  echo "Set CODE_PATH to an absolute path containing your workspace (e.g. export CODE_PATH=\"$PWD\")." >&2
  exit 1
fi

if [ ! -d "$CODE_PATH" ]; then
  echo "CODE_PATH '$CODE_PATH' does not exist or is not a directory." >&2
  exit 1
fi

RUNTIME_IMAGE="docker.all-hands.dev/all-hands-ai/runtime:latest"
SERVER_IMAGE="docker.all-hands.dev/all-hands-ai/openhands:latest"

docker run -it --rm \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE="$RUNTIME_IMAGE" \
  -e SANDBOX_VOLUMES="${CODE_PATH}:/workspace:rw" \
  -e SANDBOX_USER_ID="$(id -u)" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --add-host host.docker.internal:host-gateway \
  -p 3000:3000 \
  --name openhands-app \
  "$SERVER_IMAGE" "$@"
