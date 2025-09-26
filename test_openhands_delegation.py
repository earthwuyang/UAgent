#!/usr/bin/env python3
"""Test script for OpenHands goal delegation approach.

This tests the new approach that delegates entire research goals to OpenHands
instead of step-by-step CodeAct interactions.
"""

import asyncio
import sys
from pathlib import Path
import json
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.app.services.openhands_goal_runner import OpenHandsGoalRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_simple_goal_delegation():
    """Test delegating a simple Python task to OpenHands."""

    workspace = Path("/tmp/test_openhands_simple")
    workspace.mkdir(parents=True, exist_ok=True)

    # Simple goal that shouldn't create multiple files
    goal = """
    Create a Python script that:
    1. Generates 10 random numbers between 1 and 100
    2. Calculates their mean, median, and standard deviation
    3. Saves the results to a JSON file
    4. Prints a summary

    Name the main script 'statistics_calculator.py' in the code/ directory.
    Execute it and show the results.
    """

    runner = OpenHandsGoalRunner(
        openhands_binary="uvx --python 3.12 --from openhands-ai openhands",
        max_iterations=20,
        model="gpt-4o-mini"  # Use a smaller model for testing
    )

    # Progress callback to track execution
    async def progress_callback(event_type: str, data: dict):
        logger.info(f"Progress: {event_type} - {data.get('line', data)[:200] if isinstance(data.get('line', data), str) else data}")

    logger.info("Starting simple goal delegation test...")
    result = await runner.run_goal(
        workspace_path=workspace,
        goal=goal,
        progress_cb=progress_callback
    )

    logger.info(f"Test completed: Success={result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Generated files: {result.artifacts.get('generated_files', [])}")

    # Check if only one main file was created (not multiple variations)
    code_dir = workspace / "code"
    if code_dir.exists():
        py_files = list(code_dir.glob("*.py"))
        logger.info(f"Python files created: {[f.name for f in py_files]}")

        if len(py_files) == 1 and py_files[0].name == "statistics_calculator.py":
            logger.info("✅ SUCCESS: Only one file created as expected")
            return True
        else:
            logger.warning(f"❌ ISSUE: Expected 1 file, found {len(py_files)}")
            return False

    return result.success


async def test_research_goal_without_repetition():
    """Test a research goal similar to the one that created multiple collect_data files."""

    workspace = Path("/tmp/test_openhands_research")
    workspace.mkdir(parents=True, exist_ok=True)

    # Research goal similar to the problematic one
    goal = """
    Create a data collection script for TPC-H queries:
    1. Generate 100 sample queries with different parameters
    2. Save them to a parquet file

    Important: Create exactly ONE script named 'collect_tpc_data.py' in the code/ directory.
    Do not create multiple versions or variations of this file.
    Execute the script and verify it works.
    """

    runner = OpenHandsGoalRunner(
        openhands_binary="uvx --python 3.12 --from openhands-ai openhands",
        max_iterations=25,
        model="gpt-4o-mini"
    )

    logger.info("Starting research goal test (checking for file repetition)...")

    async def progress_callback(event_type: str, data: dict):
        # Track file creation events
        if "collect" in str(data).lower() and "create" in str(data).lower():
            logger.warning(f"File creation detected: {data}")

    result = await runner.run_goal(
        workspace_path=workspace,
        goal=goal,
        progress_cb=progress_callback
    )

    # Check for duplicate file creation
    code_dir = workspace / "code"
    if code_dir.exists():
        collect_files = list(code_dir.glob("collect*.py"))
        logger.info(f"Collect files found: {[f.name for f in collect_files]}")

        if len(collect_files) == 1:
            logger.info("✅ SUCCESS: No duplicate collect_data files created")
            return True
        elif len(collect_files) > 1:
            logger.warning(f"❌ ISSUE: Multiple collect files created: {[f.name for f in collect_files]}")
            return False

    return result.success


async def test_with_python_api():
    """Test using the Python API directly instead of CLI subprocess."""

    workspace = Path("/tmp/test_openhands_api")
    workspace.mkdir(parents=True, exist_ok=True)

    goal = "Create a simple hello world Python script and run it."

    runner = OpenHandsGoalRunner()

    logger.info("Testing with Python API approach...")

    try:
        result = await runner.run_with_openhands_cli_api(
            workspace_path=workspace,
            goal=goal,
            progress_cb=None
        )

        logger.info(f"API Test completed: Success={result.success}")
        logger.info(f"Message: {result.message}")
        return result.success
    except ImportError as e:
        logger.warning(f"OpenHands CLI API not available: {e}")
        logger.info("This is expected if OpenHands is not installed locally")
        return None


async def main():
    """Run all tests."""

    logger.info("="*60)
    logger.info("Testing OpenHands Goal Delegation Approach")
    logger.info("="*60)

    results = {}

    # Test 1: Simple goal
    logger.info("\n--- Test 1: Simple Goal Delegation ---")
    try:
        results["simple_goal"] = await test_simple_goal_delegation()
    except Exception as e:
        logger.error(f"Simple goal test failed: {e}")
        results["simple_goal"] = False

    # Test 2: Research goal (checking for repetition)
    logger.info("\n--- Test 2: Research Goal Without Repetition ---")
    try:
        results["research_goal"] = await test_research_goal_without_repetition()
    except Exception as e:
        logger.error(f"Research goal test failed: {e}")
        results["research_goal"] = False

    # Test 3: Python API (optional)
    logger.info("\n--- Test 3: Python API Approach ---")
    try:
        results["python_api"] = await test_with_python_api()
    except Exception as e:
        logger.error(f"Python API test failed: {e}")
        results["python_api"] = False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED" if result is False else "⚠️ SKIPPED"
        logger.info(f"{test_name}: {status}")

    # Overall result
    all_passed = all(r for r in results.values() if r is not None)
    logger.info("\n" + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)