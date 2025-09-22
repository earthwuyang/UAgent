import asyncio
from pathlib import Path

import pytest

from app.integrations.openhands_runtime import OpenHandsActionServerRunner

pytestmark = pytest.mark.asyncio

async def test_codeact_runtime_run_cmd(tmp_path):
    runner = OpenHandsActionServerRunner()
    assert runner.is_available, "OpenHands runtime script not found"

    workspace = tmp_path
    workspace.mkdir(parents=True, exist_ok=True)

    session = await runner.open_session(workspace)
    try:
        result = await session.run_cmd("python --version", timeout=120)
        assert result.execution_result.success, f"Command failed: {result.execution_result.command}\nstdout: {result.execution_result.stdout}"
        assert "python" in (result.execution_result.stdout or "").lower()
    finally:
        await session.close()

async def test_codeact_runtime_ipython(tmp_path):
    runner = OpenHandsActionServerRunner()
    assert runner.is_available

    workspace = tmp_path / "ipython"
    workspace.mkdir(parents=True, exist_ok=True)

    session = await runner.open_session(workspace)
    try:
        result = await session.ipython_run("print('hello codeact')")
        assert result.execution_result.success
        assert "hello codeact" in (result.execution_result.stdout or "")
    finally:
        await session.close()
