#!/usr/bin/env bash
set -euo pipefail

# Launch OpenHands locally using uv (Python 3.12) to avoid Docker-in-Docker issues.
# Requires uv (https://docs.astral.sh/uv/) and Python 3.12 on the host.

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx is required (see https://docs.astral.sh/uv/); aborting." >&2
  exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Warning: no LLM API key detected (OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY)." >&2
fi

uvx --python 3.12 --from openhands-ai openhands "$@"
