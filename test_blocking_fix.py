#!/usr/bin/env python3
"""Test script to verify non-blocking execution of long-running commands."""

import asyncio
import sys
import os
sys.path.insert(0, '/home/wuy/AI/UAgent')

from backend.app.integrations.openhands_runtime import OpenHandsActionServerRunner


async def test_non_blocking():
    """Test that long-running commands don't block other operations."""

    runner = OpenHandsActionServerRunner()

    # Create a test workspace
    workspace = "/tmp/test_workspace"
    os.makedirs(workspace, exist_ok=True)

    print("Starting OpenHands session...")
    session = await runner.open_session(workspace)

    try:
        print("\n1. Starting a long-running command (should be non-blocking)...")
        # This should be automatically set to non-blocking
        result1 = await session.run_cmd("pip install --no-cache-dir pandas", timeout=30)
        print(f"   Result: {result1.stdout[:200]}")

        print("\n2. Running a quick read operation (should work immediately)...")
        # Create a test file first
        await session.file_write("/tmp/test.txt", "Hello World")
        # This should execute immediately even if pip is still running
        result2 = await session.file_read("/tmp/test.txt", timeout=10)
        print(f"   Read result: {result2.stdout}")

        print("\n3. Running a quick write operation (should work immediately)...")
        result3 = await session.file_write("/tmp/test2.txt", "Quick write", timeout=10)
        print(f"   Write result: Success={result3.execution_result.success}")

        print("\n4. Check if pip is still running in background...")
        result4 = await session.run_cmd("ps aux | grep 'pip install'", timeout=10)
        print(f"   Process check: {result4.stdout[:200]}")

        print("\nTest completed successfully!")

    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(test_non_blocking())