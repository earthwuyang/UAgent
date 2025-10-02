#!/usr/bin/env python3
"""Test script for adaptive retry with pip install."""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.app.core.openhands import OpenHandsClient
from backend.app.core.llm_client import LLMClient
from backend.app.integrations.openhands_bridge import OpenHandsGoalPlanBridge


async def test_pip_install_with_retry():
    """Test pip install with adaptive retry."""

    print("Testing Adaptive Retry for pip install")
    print("=" * 50)

    # Check environment variables
    print("\nEnvironment Configuration:")
    print(f"OPENHANDS_ACTION_TIMEOUT = {os.getenv('OPENHANDS_ACTION_TIMEOUT', '120')} seconds")
    print(f"OPENHANDS_MAX_ACTION_TIMEOUT = {os.getenv('OPENHANDS_MAX_ACTION_TIMEOUT', '900')} seconds")
    print(f"OPENHANDS_RUN_ADAPTIVE_MULTIPLIER = {os.getenv('OPENHANDS_RUN_ADAPTIVE_MULTIPLIER', '1.75')}")
    print(f"OPENHANDS_RUN_MAX_ATTEMPTS = {os.getenv('OPENHANDS_RUN_MAX_ATTEMPTS', '3')}")
    print(f"OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT = {os.getenv('OPENHANDS_PACKAGE_CMD_MIN_TIMEOUT', '600')} seconds")

    # Initialize clients
    llm_client = LLMClient()
    openhands_client = OpenHandsClient()
    bridge = OpenHandsGoalPlanBridge(openhands_client, llm_client)

    # Create a test plan with pip install
    test_goal = """
    Install data science packages:
    1. Create a virtual environment
    2. Install pandas, numpy, and scikit-learn
    3. Verify installation by importing packages
    """

    print(f"\nTest Goal: {test_goal}")
    print("\nGenerating plan...")

    plan = await bridge.generate_goal_plan(
        goal=test_goal,
        context={"test_mode": True}
    )

    print(f"Plan generated: {plan.summary}")
    print(f"Number of steps: {len(plan.steps)}")

    # Execute with monitoring
    execution_context = {
        "session_id": "test_adaptive_retry",
        "bootstrap_environment": False,
        "resource_requirements": {
            "timeout": 120  # Start with 2 minute timeout
        }
    }

    async def progress_callback(event: str, data: dict):
        """Monitor progress with focus on timeouts and retries."""
        if event == "codeact_event":
            action_data = data.get("data", {})
            if isinstance(action_data, dict):
                # Check for timeout events
                if "timeout" in str(action_data).lower():
                    print(f"\n>>> TIMEOUT DETECTED: {action_data}")
                # Check for retry events
                elif "retry" in str(action_data).lower():
                    print(f"\n>>> RETRY DETECTED: {action_data}")
                # Check for pip install
                elif "pip install" in str(action_data):
                    print(f"\n>>> PIP INSTALL: Monitoring execution...")
                # Check for backend fallback
                elif "backend_fallback" in str(action_data):
                    print(f"\n>>> BACKEND FALLBACK: {action_data}")
        elif event in ["step_started", "step_completed"]:
            print(f"\n[{event}] Step {data.get('step_id')}: {data.get('description', '')}")
            if event == "step_completed":
                print(f"  Status: {data.get('status')}")

    print("\nExecuting plan with adaptive retry...")
    try:
        result = await bridge.execute_goal_plan(
            plan=plan,
            execution_context=execution_context,
            progress_callback=progress_callback
        )

        print("\n" + "=" * 50)
        print("EXECUTION RESULTS")
        print("=" * 50)

        for step in result.steps:
            print(f"\nStep {step.id}: {step.status}")
            print(f"  Description: {step.description}")
            if step.notes:
                print(f"  Notes: {step.notes[:200]}")

        # Check for successful pip install
        success_count = sum(1 for s in result.steps if s.status == "completed")
        failed_count = sum(1 for s in result.steps if s.status == "failed")

        print(f"\n Summary:")
        print(f"  Successful steps: {success_count}")
        print(f"  Failed steps: {failed_count}")

        if failed_count == 0:
            print("\n✓ SUCCESS: All steps completed successfully with adaptive retry!")
        else:
            print("\n⚠ WARNING: Some steps failed despite retry")

    except Exception as e:
        print(f"\n✗ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()


async def test_backend_fallback():
    """Test backend fallback execution."""

    print("\n" + "=" * 50)
    print("Testing Backend Fallback Execution")
    print("=" * 50)

    from backend.app.integrations.openhands_runtime import OpenHandsActionServerRunner

    # Create a test workspace
    workspace = Path("/tmp/test_backend_fallback")
    workspace.mkdir(exist_ok=True)

    print(f"Workspace: {workspace}")

    # Test commands that might timeout
    test_commands = [
        "echo 'Quick test'",
        "sleep 5 && echo 'Slow command'",
        "pip install --no-cache-dir requests",
    ]

    for cmd in test_commands:
        print(f"\nTesting: {cmd}")
        # This would normally be called internally when timeout occurs
        # Here we're just showing the structure


if __name__ == "__main__":
    print("OpenHands Adaptive Retry Test Suite")
    print("=" * 50)

    # Test adaptive retry with pip install
    asyncio.run(test_pip_install_with_retry())

    # Test backend fallback
    asyncio.run(test_backend_fallback())

    print("\n" + "=" * 50)
    print("Test suite complete!")