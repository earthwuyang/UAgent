#!/usr/bin/env python3
"""
Test script to verify OpenHands command streaming integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.core.openhands.streaming_integration import (
    enable_openhands_streaming,
    stream_openhands_execution_start,
    stream_openhands_execution_progress,
    stream_openhands_execution_result,
    stream_code_generation,
    stream_experiment_setup
)
from backend.app.core.openhands.command_streamer import OpenHandsCommandStreamer

# Mock OpenHands client for testing
class MockOpenHandsClient:
    def __init__(self):
        self.workspace_manager = True
        self.name = "mock_client"

    async def execute_bash_command(self, command: str):
        """Mock bash execution"""
        print(f"Executing: {command}")
        # Simulate command result
        class Result:
            def __init__(self):
                self.stdout = f"Mock output for: {command}"
                self.stderr = ""
                self.exit_code = 0
        return Result()

async def test_streaming_integration():
    """Test the OpenHands streaming integration"""
    session_id = "test_session_123"

    print("🧪 Testing OpenHands Command Streaming Integration")
    print(f"Session ID: {session_id}")
    print("-" * 50)

    # Test 1: Enable streaming for a mock client
    print("1️⃣ Testing streaming enablement...")
    mock_client = MockOpenHandsClient()

    # Enable streaming
    streaming_client = enable_openhands_streaming(mock_client, session_id)
    print(f"✅ Streaming client created: {type(streaming_client)}")

    # Test 2: Test command streamer directly
    print("\n2️⃣ Testing command streamer...")
    workspace_path = Path("/tmp/test_workspace")
    streamer = OpenHandsCommandStreamer(session_id, workspace_path)

    # Stream command execution
    stream_id = await streamer.stream_command_execution(
        command="echo 'Hello OpenHands'",
        command_type="bash"
    )
    print(f"✅ Command execution streamed with ID: {stream_id}")

    # Stream command output
    await streamer.stream_command_output(
        stream_id=stream_id,
        output="Hello OpenHands",
        output_type="stdout",
        exit_code=0
    )
    print("✅ Command output streamed")

    # Test 3: Test streaming helper functions
    print("\n3️⃣ Testing streaming helper functions...")

    await stream_openhands_execution_start(
        session_id=session_id,
        command="test command",
        context={"test": True}
    )
    print("✅ Execution start event streamed")

    await stream_openhands_execution_progress(
        session_id=session_id,
        step="testing",
        progress=50.0,
        details={"phase": "unit_test"}
    )
    print("✅ Execution progress event streamed")

    await stream_openhands_execution_result(
        session_id=session_id,
        result={"status": "success", "output": "test complete"},
        success=True
    )
    print("✅ Execution result event streamed")

    await stream_code_generation(
        session_id=session_id,
        code="print('Generated code')",
        language="python",
        filename="test.py"
    )
    print("✅ Code generation event streamed")

    await stream_experiment_setup(
        session_id=session_id,
        experiment_name="Test Experiment",
        setup_details={"environment": "test", "dependencies": ["pytest"]}
    )
    print("✅ Experiment setup event streamed")

    # Test 4: Test streaming wrapper execution
    print("\n4️⃣ Testing streaming wrapper execution...")
    if hasattr(streaming_client, 'execute_with_streaming'):
        result = await streaming_client.execute_with_streaming(
            'execute_bash_command',
            command='echo "test"'
        )
        print(f"✅ Streaming wrapper execution completed: {result.stdout}")
    else:
        print("ℹ️ Streaming wrapper doesn't have execute_with_streaming method")

    print("\n🎉 All OpenHands streaming tests completed successfully!")

async def main():
    """Main test function"""
    try:
        await test_streaming_integration()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)