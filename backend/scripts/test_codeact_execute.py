import asyncio
from pathlib import Path
import datetime

from app.integrations.openhands_runtime import OpenHandsActionServerRunner

async def main():
    runner = OpenHandsActionServerRunner()
    print(f"[{datetime.datetime.utcnow().isoformat()}Z] runner available: {runner.is_available}")
    if not runner.is_available:
        return
    workspace = Path('/home/wuy/AI/UAgent/backend/artifacts/codeact_exec_workspace').resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    session = await runner.open_session(workspace)
    print(f"[{datetime.datetime.utcnow().isoformat()}Z] session running: {session.is_running}")
    try:
        result = await session.run_cmd("echo hello-from-codeact", timeout=30, blocking=True)
        print("observation:", result.raw_observation)
        print("execution success:", result.execution_result.success)
    finally:
        await session.close()

if __name__ == '__main__':
    asyncio.run(main())
