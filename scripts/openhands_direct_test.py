#!/usr/bin/env python3
"""Simple tester for the OpenHands direct-run backend endpoint.

Features:
- Posts a goal to `/api/openhands/direct-run` and waits for completion.
- Optionally tails live logs from `live_combined.log` to monitor progress.
- Prints final result summary (success, exit code, final.json presence) and paths.

Usage examples:
  python scripts/openhands_direct_test.py \
    --goal "Write hello.py under /workspace and save final.json with success=true"

  python scripts/openhands_direct_test.py \
    --goal-file goal.txt --session exp_demo --tail

Environment variables:
  API_URL                Default: http://localhost:8000
  UAGENT_WORKSPACE_DIR   Used to locate logs if --workspace-root not provided
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import requests


DEFAULT_API = os.getenv("API_URL", "http://localhost:8001")


def read_goal(goal: Optional[str], goal_file: Optional[str]) -> str:
    if goal and goal.strip():
        return goal.strip()
    if goal_file:
        p = Path(goal_file).expanduser()
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --goal or --goal-file")


def tail_file(path: Path, stop_event: threading.Event, label: str = "LOG") -> None:
    try:
        # Wait until file exists (up to ~10 minutes)
        deadline = time.time() + 600
        while not path.exists() and time.time() < deadline and not stop_event.is_set():
            time.sleep(0.5)

        if not path.exists():
            print(f"[tail:{label}] {path} not found; skipping tail")
            return

        with path.open("r", encoding="utf-8", errors="replace") as f:
            # Start from current end so we show only live progress
            f.seek(0, os.SEEK_END)
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                print(f"[tail:{label}] {line.rstrip()}")
    except Exception as exc:
        print(f"[tail:{label}] error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenHands direct-run with optional live log tailing")
    parser.add_argument("--api", default=DEFAULT_API, help="Backend base URL (default: %(default)s)")
    parser.add_argument("--goal", default=None, help="Goal text (mutually exclusive with --goal-file)")
    parser.add_argument("--goal-file", default='goal.txt', help="Path to goal text file")
    parser.add_argument("--session", default=None, help="Session name (default: exp_<random>)")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps (default: %(default)s)")
    parser.add_argument("--max-minutes", type=int, default=60, help="Max minutes (default: %(default)s)")
    parser.add_argument("--workspace-root", default=None, help="Override workspace root used by backend")
    parser.add_argument("--tail", action="store_true", default=True, help="Tail live_combined.log while run executes")

    args = parser.parse_args()

    goal_text = read_goal(args.goal, args.goal_file)

    # Predict log path: the backend uses workspace = (workspace_root or UAGENT_WORKSPACE_DIR or /tmp/uagent-workspace)/direct_runs
    workspace_root = args.workspace_root or os.getenv("UAGENT_WORKSPACE_DIR", "/tmp/uagent-workspace")
    workspace_dir = Path(workspace_root).expanduser().resolve() / "direct_runs"
    logs_dir = workspace_dir / "logs" / "openhands_live"
    combined_log = logs_dir / "live_combined.log"
    stdout_log = logs_dir / "live_stdout.log"
    stderr_log = logs_dir / "live_stderr.log"

    if args.tail:
        stop_event = threading.Event()
        t = threading.Thread(target=tail_file, args=(combined_log, stop_event, "combined"), daemon=True)
        t.start()
    else:
        stop_event = None  # type: ignore
        t = None  # type: ignore

    url = args.api.rstrip("/") + "/api/openhands/direct-run"
    payload = {
        "goal": goal_text,
        "session_name": args.session,
        "max_steps": args.max_steps,
        "max_minutes": args.max_minutes,
        "workspace_root": str(Path(workspace_root).expanduser().resolve()),
    }

    print(f"POST {url}")
    print(f"workspace_root={workspace_root}")
    try:
        resp = requests.post(url, json=payload, timeout=None)
    except KeyboardInterrupt:
        print("Interrupted while waiting for direct-run response")
        if stop_event:
            stop_event.set()
        raise

    if stop_event:
        stop_event.set()
        if t:
            t.join(timeout=2)

    try:
        data = resp.json()
    except Exception:
        print(f"HTTP {resp.status_code}: {resp.text[:1000]}")
        raise SystemExit(1)

    print("\n=== Direct-Run Result ===")
    print(json.dumps(data, indent=2)[:4000])

    print("\nArtifacts:")
    print(f"  workspace:     {workspace_dir}")
    print(f"  session_name:  {data.get('session_name')}")
    print(f"  logs dir:      {logs_dir}")
    print(f"  combined log:  {combined_log}")
    print(f"  stdout log:    {stdout_log}")
    print(f"  stderr log:    {stderr_log}")

    if data.get("final_json"):
        # Save a copy alongside logs for convenience
        try:
            out = logs_dir / f"final_{data.get('session_name','session')}.json"
            logs_dir.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data["final_json"], indent=2), encoding="utf-8")
            print(f"  final copy:    {out}")
        except Exception as exc:
            print(f"  final copy:    failed to write ({exc})")


if __name__ == "__main__":
    main()

