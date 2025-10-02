#!/usr/bin/env python3
"""Simple test to verify OpenHands V3 is working"""

import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.app.integrations.openhands_codeact_bridge_v3 import (
    OpenHandsCodeActBridgeV3,
    CodeActRunConfig,
)


async def test_simple_v3():
    """Test V3 with a very simple task"""
    workspace = Path("/tmp/test_v3_simple")
    workspace.mkdir(parents=True, exist_ok=True)

    # Very simple goal that should work
    goal = """
    Create a file at experiments/test/results/final.json with the following content:
    {
        "success": true,
        "measurements": [1.0, 2.0, 3.0],
        "data": {"test": "passed"},
        "analysis": {"mean": 2.0},
        "conclusions": ["Test completed successfully"]
    }
    """

    config = CodeActRunConfig(
        goal=goal,
        workspace=workspace,
        session_name="test",
        max_steps=10,
        max_minutes=5,
        disable_browser=True
    )

    print("Starting V3 test...")
    bridge = OpenHandsCodeActBridgeV3()
    result = await bridge.run_async(config)

    print(f"Success: {result.success}")
    print(f"Exit code: {result.exit_code}")
    print(f"Duration: {result.duration_seconds:.1f}s")

    if result.final_json:
        print(f"Final JSON found: {json.dumps(result.final_json, indent=2)}")
    else:
        print("No final.json found")

    if result.stderr_tail:
        print(f"Stderr (last 500 chars): ...{result.stderr_tail[-500:]}")

    return result.success


if __name__ == "__main__":
    success = asyncio.run(test_simple_v3())
    sys.exit(0 if success else 1)