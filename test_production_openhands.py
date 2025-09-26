#!/usr/bin/env python3
"""Test the production OpenHands complete goal delegation implementation."""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_complete_delegation():
    """Test the complete goal delegation to OpenHands."""

    from backend.app.services.openhands_complete_runner import OpenHandsCompleteRunner

    workspace = Path("/tmp/test_complete_delegation")
    workspace.mkdir(parents=True, exist_ok=True)

    # Goal that previously caused multiple file creations
    goal = """
    Create a Python data collection script:
    1. Generate 100 sample TPC-H queries with randomized parameters
    2. Save the queries to a parquet file named 'queries.parquet'

    Requirements:
    - Create exactly ONE file named 'collect_data.py' in the code/ directory
    - Do not create multiple versions or variations
    - The script should be executable and tested
    """

    logger.info("Testing complete goal delegation to OpenHands...")

    runner = OpenHandsCompleteRunner(
        model="gpt-4o-mini",
        max_iterations=30,
    )

    # Track progress
    progress_events = []

    async def progress_callback(event_type: str, data: dict):
        progress_events.append((event_type, data))
        logger.info(f"Progress: {event_type}")

    try:
        result = await runner.run_complete_goal(
            workspace_path=workspace,
            goal=goal,
            progress_cb=progress_callback,
        )

        logger.info(f"Success: {result.success}")
        logger.info(f"Message: {result.message}")
        logger.info(f"Generated files: {result.generated_files}")
        logger.info(f"Total events: {result.events_count}")

        # Check for duplicate files
        collect_files = [f for f in result.generated_files if "collect" in f.lower()]
        if len(collect_files) == 1:
            logger.info("✅ SUCCESS: Only one collect file created")
            return True
        elif len(collect_files) > 1:
            logger.warning(f"❌ ISSUE: Multiple collect files: {collect_files}")
            return False
        else:
            logger.warning("⚠️ No collect files created")
            return False

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_router_integration():
    """Test the smart router with complete delegation."""

    from backend.app.routers.smart_router_complete import smart_router_complete

    session_id = "test_router_123"
    workspace_base = Path("/tmp/test_router")

    query = """
    Create a script to analyze database query performance.
    The script should generate sample queries and measure their execution time.
    """

    logger.info("Testing smart router with complete delegation...")

    try:
        result = await smart_router_complete.route(
            query=query,
            session_id=session_id,
            workspace_base=workspace_base,
            engine_override="scientific",
        )

        logger.info(f"Engine: {result['engine']}")
        logger.info(f"Success: {result['success']}")
        logger.info(f"Files: {result.get('generated_files', [])}")

        # Check for no duplicate files
        files = result.get('generated_files', [])
        file_names = [Path(f).name for f in files]
        unique_names = set(file_names)

        if len(file_names) == len(unique_names):
            logger.info("✅ SUCCESS: No duplicate file names")
            return True
        else:
            duplicates = [name for name in file_names if file_names.count(name) > 1]
            logger.warning(f"❌ ISSUE: Duplicate files found: {duplicates}")
            return False

    except Exception as e:
        logger.error(f"Router test failed: {e}", exc_info=True)
        return False


async def main():
    """Run all production tests."""

    logger.info("="*60)
    logger.info("Testing Production OpenHands Complete Goal Delegation")
    logger.info("="*60)

    # Set up API keys
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")

    results = {}

    # Test 1: Complete delegation
    logger.info("\n--- Test 1: Complete Goal Delegation ---")
    try:
        results["complete_delegation"] = await test_complete_delegation()
    except Exception as e:
        logger.error(f"Complete delegation test error: {e}")
        results["complete_delegation"] = False

    # Test 2: Router integration
    logger.info("\n--- Test 2: Smart Router Integration ---")
    try:
        results["router_integration"] = await test_router_integration()
    except Exception as e:
        logger.error(f"Router integration test error: {e}")
        results["router_integration"] = False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PRODUCTION TEST RESULTS")
    logger.info("="*60)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())

    logger.info("\n" + "="*60)
    logger.info("SOLUTION SUMMARY")
    logger.info("="*60)
    logger.info("The complete goal delegation approach solves the repetitive")
    logger.info("file creation issue by:")
    logger.info("1. Delegating the entire goal to OpenHands' agent controller")
    logger.info("2. Letting OpenHands handle planning internally")
    logger.info("3. Avoiding step-by-step external control that causes repetition")
    logger.info("="*60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)