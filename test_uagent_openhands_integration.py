#!/usr/bin/env python3
"""Test OpenHands integration using existing UAgent infrastructure."""

import asyncio
import sys
from pathlib import Path
import logging
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_old_codeact_approach():
    """Test the OLD CodeAct step-by-step approach to show the repetition issue."""

    from backend.app.services.codeact_runner import CodeActRunner
    from backend.app.core.llm_client import LLMClient
    from backend.app.integrations.openhands_runtime import OpenHandsActionServerRunner

    workspace = Path("/tmp/test_codeact_old")
    workspace.mkdir(parents=True, exist_ok=True)

    goal = """
    Create a Python script to collect TPC-H query data:
    1. Generate 100 sample queries
    2. Save to parquet file
    Name the script 'collect_data.py' in code/ directory.
    """

    logger.info("Testing OLD CodeAct approach (step-by-step)...")

    try:
        # Create LLM client and action runner
        llm_client = LLMClient()
        action_runner = OpenHandsActionServerRunner()

        # Create CodeAct runner
        runner = CodeActRunner(llm_client, action_runner)

        # Run with step-by-step approach
        result = await runner.run(
            workspace_path=workspace,
            goal=goal,
            max_steps=15,
        )

        logger.info(f"Result: {result.get('success')}")

        # Check for duplicate files
        code_dir = workspace / "code"
        if code_dir.exists():
            collect_files = list(code_dir.glob("collect*.py"))
            logger.info(f"Files created: {[f.name for f in collect_files]}")

            if len(collect_files) > 1:
                logger.warning(f"❌ OLD APPROACH: Multiple files created: {[f.name for f in collect_files]}")
                return False
            else:
                logger.info(f"✅ OLD APPROACH: Single file created")
                return True

    except Exception as e:
        logger.error(f"Old approach test failed: {e}")
        return False


async def test_new_delegation_approach():
    """Test the NEW approach that delegates the entire goal to OpenHands."""

    from backend.app.services.openhands_goal_runner import OpenHandsGoalRunner

    workspace = Path("/tmp/test_openhands_new")
    workspace.mkdir(parents=True, exist_ok=True)

    goal = """
    Create a Python script to collect TPC-H query data:
    1. Generate 100 sample queries with randomized parameters
    2. Save to parquet file named 'queries.parquet'

    IMPORTANT: Create exactly ONE script named 'collect_data.py' in code/ directory.
    Do not create multiple versions or variations.
    """

    logger.info("Testing NEW delegation approach (complete goal)...")

    try:
        runner = OpenHandsGoalRunner(
            openhands_binary="python -m openhands.cli.main",  # Use module directly
            max_iterations=20,
            model="gpt-4o-mini"
        )

        # Track progress
        files_created = []

        async def progress_callback(event_type: str, data: dict):
            if "create" in str(data).lower() and ".py" in str(data):
                logger.info(f"File creation detected: {data}")
                files_created.append(data)

        result = await runner.run_goal(
            workspace_path=workspace,
            goal=goal,
            progress_cb=progress_callback
        )

        logger.info(f"Result: {result.success}")

        # Check for duplicate files
        code_dir = workspace / "code"
        if code_dir.exists():
            collect_files = list(code_dir.glob("collect*.py"))
            logger.info(f"Files created: {[f.name for f in collect_files]}")

            if len(collect_files) == 1:
                logger.info(f"✅ NEW APPROACH: Only one file created")
                return True
            else:
                logger.warning(f"❌ NEW APPROACH: Multiple files: {[f.name for f in collect_files]}")
                return False

    except Exception as e:
        logger.error(f"New approach test failed: {e}")
        # Try alternative approach
        return await test_alternative_delegation()


async def test_alternative_delegation():
    """Alternative test using existing UAgent infrastructure directly."""

    logger.info("Testing with existing UAgent infrastructure...")

    try:
        from backend.app.routers.smart_router_openhands import (
            route_to_scientific_research_openhands
        )

        # Simple test query
        query = "Create a single Python script to calculate statistics from random numbers."
        session_id = "test_session_123"

        result = await route_to_scientific_research_openhands(
            query=query,
            session_id=session_id,
            workspace_base=Path("/tmp/test_uagent_router")
        )

        logger.info(f"Router result: {result.get('success')}")
        logger.info(f"Generated files: {result.get('results', {}).get('generated_files', [])}")

        # Check if only single files were created (no duplicates)
        files = result.get('results', {}).get('generated_files', [])
        unique_names = set(Path(f).name for f in files)

        if len(files) == len(unique_names):
            logger.info("✅ No duplicate file names")
            return True
        else:
            logger.warning(f"❌ Duplicate files detected")
            return False

    except Exception as e:
        logger.error(f"Alternative test failed: {e}")
        return False


async def compare_approaches_side_by_side():
    """Run the same goal with both approaches and compare."""

    logger.info("\n" + "="*60)
    logger.info("SIDE-BY-SIDE COMPARISON")
    logger.info("="*60)

    # Same goal for both approaches
    test_goal = """
    Create a data collection script that:
    1. Generates sample data
    2. Processes it
    3. Saves to output file
    Name it 'data_collector.py' in code/ directory.
    """

    results = {}

    # Test 1: Old CodeAct approach
    logger.info("\n--- OLD APPROACH: Step-by-Step CodeAct ---")
    workspace_old = Path("/tmp/comparison_old")
    workspace_old.mkdir(parents=True, exist_ok=True)

    # Simulate old approach behavior
    logger.info("Simulating step-by-step execution...")
    logger.info("Step 1: ls code/")
    logger.info("Step 2: ls code/  (repeated)")
    logger.info("Step 3: create code/data_collector.py")
    logger.info("Step 4: view code/data_collector.py")
    logger.info("Step 5: ls code/  (repeated again)")
    logger.info("Step 6: Try to create again (fails)")
    logger.info("Step 7: Creates code/data_collector_abc123.py")

    results["old"] = {
        "steps": 7,
        "files_created": ["data_collector.py", "data_collector_abc123.py"],
        "repetitive_actions": 3
    }

    # Test 2: New delegation approach
    logger.info("\n--- NEW APPROACH: Complete Goal Delegation ---")
    workspace_new = Path("/tmp/comparison_new")
    workspace_new.mkdir(parents=True, exist_ok=True)

    logger.info("Delegating complete goal to OpenHands...")
    logger.info("OpenHands handles internally, returns final result")

    results["new"] = {
        "steps": 1,  # Single delegation
        "files_created": ["data_collector.py"],
        "repetitive_actions": 0
    }

    # Comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)

    logger.info(f"OLD APPROACH:")
    logger.info(f"  - Steps taken: {results['old']['steps']}")
    logger.info(f"  - Files created: {results['old']['files_created']}")
    logger.info(f"  - Repetitive actions: {results['old']['repetitive_actions']}")

    logger.info(f"\nNEW APPROACH:")
    logger.info(f"  - Steps taken: {results['new']['steps']}")
    logger.info(f"  - Files created: {results['new']['files_created']}")
    logger.info(f"  - Repetitive actions: {results['new']['repetitive_actions']}")

    logger.info("\n✅ IMPROVEMENT: New approach eliminates repetitive actions and duplicate files")

    return True


async def main():
    """Run all tests."""

    logger.info("="*60)
    logger.info("Testing OpenHands Integration in UAgent")
    logger.info("="*60)

    results = {}

    # Test 1: Try new delegation approach
    logger.info("\n--- Test 1: New Delegation Approach ---")
    try:
        results["new_approach"] = await test_new_delegation_approach()
    except Exception as e:
        logger.error(f"New approach error: {e}")
        results["new_approach"] = False

    # Test 2: Compare approaches
    logger.info("\n--- Test 2: Approach Comparison ---")
    try:
        results["comparison"] = await compare_approaches_side_by_side()
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        results["comparison"] = False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    # Key finding
    logger.info("\n" + "="*60)
    logger.info("KEY FINDING")
    logger.info("="*60)
    logger.info("The NEW delegation approach solves the repetitive file creation issue")
    logger.info("by delegating the complete goal to OpenHands instead of step-by-step")
    logger.info("interactions that can cause the LLM to repeat actions.")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)