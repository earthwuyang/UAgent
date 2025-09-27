#!/usr/bin/env python3
"""Test script for OpenHands V3 headless bridge integration"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.app.integrations.openhands_codeact_bridge_v3 import (
    OpenHandsCodeActBridgeV3,
    CodeActRunConfig,
    CodeActRunSummary,
)


def test_simple_hello_world():
    """Test a simple hello world execution"""
    print("\n=== Test 1: Simple Hello World ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_hello"

        cfg = CodeActRunConfig(
            goal="Write a Python script that prints 'Hello from OpenHands V3' and saves success status to experiments/hello/results/final.json",
            workspace=workspace,
            session_name="hello",
            max_steps=10,
            max_minutes=2,
            disable_browser=True
        )

        bridge = OpenHandsCodeActBridgeV3()
        result = bridge.run(cfg)

        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Artifact: {result.artifact_path}")

        if result.final_json:
            print(f"Final JSON: {json.dumps(result.final_json, indent=2)}")

        assert result.success, f"Test failed: {result.reason}"
        assert result.final_json, "No final.json found"
        print("✅ Test 1 passed!")


def test_data_collection():
    """Test experimental data collection"""
    print("\n=== Test 2: Data Collection ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_experiment"

        cfg = CodeActRunConfig(
            goal="""Run a simple performance experiment:
1. Measure the time to compute factorial of numbers 10, 100, 1000
2. Calculate mean and std deviation
3. Save results to experiments/perf_test/results/final.json with:
   - success: true
   - measurements: list of timing values
   - analysis: dict with mean and std
   - conclusions: list with one conclusion about performance""",
            workspace=workspace,
            session_name="perf_test",
            max_steps=20,
            max_minutes=3,
            disable_browser=True
        )

        bridge = OpenHandsCodeActBridgeV3()
        result = bridge.run(cfg)

        print(f"Success: {result.success}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        if result.final_json:
            print(f"Measurements: {result.final_json.get('measurements', [])}")
            print(f"Analysis: {result.final_json.get('analysis', {})}")
            print(f"Conclusions: {result.final_json.get('conclusions', [])}")

        assert result.success, f"Test failed: {result.reason}"
        assert result.final_json, "No final.json found"
        assert result.final_json.get("measurements"), "No measurements in result"
        print("✅ Test 2 passed!")


def test_timeout():
    """Test timeout handling"""
    print("\n=== Test 3: Timeout Handling ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_timeout"

        cfg = CodeActRunConfig(
            goal="Run an infinite loop to test timeout (while True: pass)",
            workspace=workspace,
            session_name="timeout",
            max_steps=100,
            max_minutes=0.1,  # 6 seconds timeout
            disable_browser=True
        )

        bridge = OpenHandsCodeActBridgeV3()
        result = bridge.run(cfg)

        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")
        print(f"Reason: {result.reason}")

        assert not result.success, "Should have failed due to timeout"
        assert "timeout" in result.reason.lower(), f"Expected timeout in reason, got: {result.reason}"
        print("✅ Test 3 passed!")


async def test_async_execution():
    """Test async execution"""
    print("\n=== Test 4: Async Execution ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_async"

        cfg = CodeActRunConfig(
            goal="Calculate 2+2 and save the result to experiments/calc/results/final.json with success=true and result=4",
            workspace=workspace,
            session_name="calc",
            max_steps=10,
            max_minutes=2,
            disable_browser=True
        )

        bridge = OpenHandsCodeActBridgeV3()
        result = await bridge.run_async(cfg)

        print(f"Success: {result.success}")
        print(f"Duration: {result.duration_seconds:.1f}s")

        if result.final_json:
            print(f"Result: {result.final_json}")

        assert result.success, f"Test failed: {result.reason}"
        print("✅ Test 4 passed!")


def test_error_handling():
    """Test error handling and logging"""
    print("\n=== Test 5: Error Handling ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "test_error"

        cfg = CodeActRunConfig(
            goal="Intentionally raise an exception: raise ValueError('Test error') and still write experiments/error/results/final.json with success=false and error message",
            workspace=workspace,
            session_name="error",
            max_steps=10,
            max_minutes=2,
            disable_browser=True
        )

        bridge = OpenHandsCodeActBridgeV3()
        result = bridge.run(cfg)

        print(f"Success: {result.success}")
        print(f"Exit code: {result.exit_code}")

        # Check if logs were captured
        if result.stdout_tail:
            print(f"Stdout tail (last 200 chars): ...{result.stdout_tail[-200:]}")
        if result.stderr_tail:
            print(f"Stderr tail (last 200 chars): ...{result.stderr_tail[-200:]}")

        # May succeed if agent handles the error gracefully
        # or may fail - both are acceptable for this test
        print("✅ Test 5 completed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("OpenHands V3 Bridge Test Suite")
    print("=" * 60)

    # Check if OpenHands directory exists
    repo_root = Path(__file__).parent
    openhands_dir = repo_root / "OpenHands"

    if not openhands_dir.exists():
        print(f"❌ OpenHands directory not found at {openhands_dir}")
        print("Please ensure OpenHands is cloned or linked at the expected location")
        return 1

    print(f"✅ OpenHands directory found: {openhands_dir}")

    # Run synchronous tests
    try:
        test_simple_hello_world()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return 1

    try:
        test_data_collection()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return 1

    try:
        test_timeout()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return 1

    # Run async test
    try:
        asyncio.run(test_async_execution())
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return 1

    try:
        test_error_handling()
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())