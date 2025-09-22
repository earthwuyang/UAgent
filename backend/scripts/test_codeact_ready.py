import asyncio
import datetime
from pathlib import Path

from app.integrations.openhands_runtime import OpenHandsActionServerRunner

async def main() -> None:
    print(f"[{datetime.datetime.utcnow().isoformat()}Z] Starting CodeAct runtime readiness test", flush=True)
    runner = OpenHandsActionServerRunner()
    print(f"[{datetime.datetime.utcnow().isoformat()}Z] Runner available: {runner.is_available}", flush=True)
    if not runner.is_available:
        return
    workspace = Path('/home/wuy/AI/UAgent/backend/artifacts/codeact_ready_workspace').resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        session = await runner.open_session(workspace)
    except Exception as exc:
        print(f"[{datetime.datetime.utcnow().isoformat()}Z] Failed to start session: {exc}", flush=True)
        raise
    else:
        print(f"[{datetime.datetime.utcnow().isoformat()}Z] Session established: {session.is_running}", flush=True)
        await session.close()
        print(f"[{datetime.datetime.utcnow().isoformat()}Z] Session closed", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
