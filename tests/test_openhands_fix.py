#!/usr/bin/env python3
"""Test script to verify OpenHands command execution fixes."""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.app.core.openhands import OpenHandsClient
from backend.app.core.llm_client import LLMClient
from backend.app.integrations.openhands_bridge import OpenHandsGoalPlanBridge


async def test_command_execution():
    """Test that commands execute properly with our fixes."""

    # Initialize clients
    llm_client = LLMClient()
    openhands_client = OpenHandsClient()
    bridge = OpenHandsGoalPlanBridge(openhands_client, llm_client)

    # Create a simple test plan
    test_goal = """
    Test pip install and file creation:
    1. Create a Python script that prints 'Hello World'
    2. Install requests library using pip
    3. Create a script that uses requests to fetch a URL
    """

    print("Generating test plan...")
    plan = await bridge.generate_goal_plan(
        goal=test_goal,
        context={"test_mode": True}
    )

    print(f"Plan generated: {plan.summary}")
    print(f"Steps: {len(plan.steps)}")

    # Execute the plan
    execution_context = {
        "session_id": "test_fix_session",
        "bootstrap_environment": False,  # Skip bootstrap for test
        "resource_requirements": {
            "max_file_size": 10000,
            "timeout": 120  # 2 minute timeout per action
        }
    }

    async def progress_callback(event: str, data: dict):
        """Log progress events."""
        print(f"[{event}] {data}")
        if event == "codeact_event":
            # Check for pip install commands
            if "pip install" in str(data.get("data", {})):
                print(">>> PIP INSTALL DETECTED - Monitoring execution...")

    print("\nExecuting plan...")
    try:
        result = await bridge.execute_goal_plan(
            plan=plan,
            execution_context=execution_context,
            progress_callback=progress_callback
        )

        print("\n=== EXECUTION RESULTS ===")
        for step in result.steps:
            print(f"Step {step.id}: {step.status}")
            if step.notes:
                print(f"  Notes: {step.notes}")

        # Check if any steps failed
        failed_steps = [s for s in result.steps if s.status == "failed"]
        if failed_steps:
            print(f"\nWARNING: {len(failed_steps)} steps failed")
            for step in failed_steps:
                print(f"  - {step.id}: {step.description}")
        else:
            print("\nSUCCESS: All steps completed successfully!")

    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()


async def test_timeout_handling():
    """Test that timeouts are handled properly."""

    print("\n=== TESTING TIMEOUT HANDLING ===")

    # Check environment variable
    timeout = os.getenv("OPENHANDS_ACTION_TIMEOUT", "120")
    print(f"OPENHANDS_ACTION_TIMEOUT = {timeout} seconds")

    # Verify it's being used
    from backend.app.integrations.openhands_bridge import OpenHandsGoalPlanBridge
    from backend.app.core.openhands import OpenHandsClient
    from backend.app.core.llm_client import LLMClient

    client = OpenHandsClient()
    llm = LLMClient()
    bridge = OpenHandsGoalPlanBridge(client, llm)

    print(f"Bridge codeact_action_timeout = {bridge._codeact_action_timeout} seconds")

    if bridge._codeact_action_timeout == int(timeout):
        print("✓ Timeout configuration is correct")
    else:
        print("✗ Timeout mismatch!")


if __name__ == "__main__":
    print("OpenHands Command Execution Fix Test")
    print("=" * 50)

    # Test timeout configuration
    asyncio.run(test_timeout_handling())

    # Test command execution
    print("\n" + "=" * 50)
    response = input("Run command execution test? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(test_command_execution())
    else:
        print("Skipping command execution test")

    print("\nTest complete!")