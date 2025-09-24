#!/usr/bin/env bash
set -euo pipefail

# Connectivity / dependency smoke test to run inside the OpenHands runtime container.
# Requires Python packages duckdb + psycopg to be installed (see sandbox_env/openhands-runtime.Dockerfile).

if ! command -v python >/dev/null 2>&1; then
  echo "python executable not found on PATH" >&2
  exit 1
fi

: "${PG_DSN:?export PG_DSN=postgresql://user:pass@host:5432/dbname before running this script}"
export DUCKDB_DB="${DUCKDB_DB:-}"

python - <<'PY'
import importlib
import os
import socket
import sys
import traceback
from urllib.parse import urlparse

required = ["duckdb", "psycopg"]
missing = []
for module in required:
    try:
        importlib.import_module(module)
        print(f"[ok] module available: {module}")
    except Exception as exc:  # pragma: no cover
        print(f"[fail] missing module: {module} -> {exc}")
        missing.append(module)

if missing:
    print("Dependency check failed; install missing packages in the runtime image.")
    sys.exit(1)

dsn = os.environ["PG_DSN"]
parsed = urlparse(dsn if dsn.startswith("postgresql://") else "postgresql://" + dsn)
host = parsed.hostname or "localhost"
port = parsed.port or 5432

print(f"[check] attempting TCP connection to PostgreSQL {host}:{port}")
try:
    sock = socket.create_connection((host, port), timeout=5)
    sock.close()
    print("[ok] TCP connectivity to PostgreSQL verified")
except Exception as exc:
    print(f"[fail] could not reach PostgreSQL at {host}:{port} -> {exc}")
    sys.exit(2)

print("[ok] smoke test succeeded")
PY
