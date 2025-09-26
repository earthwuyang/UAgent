#!/usr/bin/env python3
"""Test OpenHands goal delegation using local OpenHands source code."""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add OpenHands source to path
openhands_dir = Path(__file__).parent / "OpenHands"
if openhands_dir.exists():
    sys.path.insert(0, str(openhands_dir))

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_openhands_direct_delegation():
    """Test using OpenHands directly from source to handle complete goals."""

    # Import OpenHands modules
    from openhands.core.config import OpenHandsConfig
    from openhands.core.setup import (
        create_agent,
        create_controller,
        create_runtime,
        generate_sid,
    )
    from openhands.core.loop import run_agent_until_done
    from openhands.controller import AgentController
    from openhands.runtime import get_runtime_cls

    workspace = Path("/tmp/test_openhands_direct")
    workspace.mkdir(parents=True, exist_ok=True)

    # Goal that previously caused multiple file creations
    goal = """
    Create a Python script to collect data:
    1. Generate 100 sample TPC-H queries with randomized parameters
    2. Save the queries to a parquet file named 'queries.parquet'
    3. The script should be named 'collect_data.py' in the code/ directory

    IMPORTANT: Create exactly ONE file named 'collect_data.py'. Do not create variations or multiple versions.
    Execute the script and verify it creates the parquet file.
    """

    logger.info("Testing OpenHands direct delegation with local source...")

    try:
        # Configure OpenHands
        config = OpenHandsConfig(
            workspace_base=str(workspace),
            workspace_mount_path=str(workspace),
            model="gpt-4o-mini",  # or use local LLM
            max_iterations=30,
            runtime="local",  # Use local runtime
        )

        # Set environment variables for OpenHands
        os.environ["WORKSPACE_BASE"] = str(workspace)

        # Create session
        sid = generate_sid()
        logger.info(f"Created session: {sid}")

        # Create runtime
        runtime_cls = get_runtime_cls(config.runtime)
        runtime = await runtime_cls.ainit(
            config=config,
            sid=sid,
        )

        await runtime.initialize()
        logger.info("Runtime initialized")

        # Create agent and controller
        agent = create_agent(config)
        controller = AgentController(
            agent=agent,
            runtime=runtime,
            max_iterations=config.max_iterations,
        )

        logger.info("Starting agent with goal...")

        # Run the agent with the complete goal
        final_state = await run_agent_until_done(
            controller=controller,
            initial_prompt=goal,
        )

        logger.info(f"Agent completed with status: {final_state.exit_reason}")

        # Check results
        code_dir = workspace / "code"
        if code_dir.exists():
            py_files = list(code_dir.glob("collect*.py"))
            logger.info(f"Files created: {[f.name for f in py_files]}")

            if len(py_files) == 1 and py_files[0].name == "collect_data.py":
                logger.info("✅ SUCCESS: Only one collect_data.py file created")
                return True
            else:
                logger.error(f"❌ FAILURE: Expected 1 file, found {len(py_files)}: {[f.name for f in py_files]}")
                return False

        await runtime.close()

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_codeact_agent():
    """Test using CodeAct agent specifically."""

    from openhands.agenthub.codeact_agent import CodeActAgent
    from openhands.core.config import OpenHandsConfig, AgentConfig
    from openhands.core.setup import create_runtime, generate_sid
    from openhands.controller import AgentController

    workspace = Path("/tmp/test_codeact_agent")
    workspace.mkdir(parents=True, exist_ok=True)

    goal = """
    Create a single Python file named 'data_processor.py' that:
    1. Reads CSV data
    2. Processes it
    3. Outputs results

    Create ONLY this one file. Do not create multiple versions.
    """

    logger.info("Testing with CodeAct agent...")

    try:
        # Configure for CodeAct agent
        config = OpenHandsConfig(
            workspace_base=str(workspace),
            workspace_mount_path=str(workspace),
            model="gpt-4o-mini",
            agent="CodeActAgent",
            max_iterations=20,
            runtime="local",
        )

        sid = generate_sid()

        # Create runtime
        runtime = await create_runtime(config, sid)
        await runtime.initialize()

        # Create CodeAct agent
        agent = CodeActAgent(config=config)

        # Create controller
        controller = AgentController(
            agent=agent,
            runtime=runtime,
            max_iterations=config.max_iterations,
        )

        # Run agent
        from openhands.core.loop import run_agent_until_done
        final_state = await run_agent_until_done(
            controller=controller,
            initial_prompt=goal,
        )

        logger.info(f"CodeAct agent completed: {final_state.exit_reason}")

        # Check for duplicate files
        if workspace.exists():
            py_files = list(workspace.rglob("*.py"))
            logger.info(f"Python files created: {[f.name for f in py_files]}")

            if len([f for f in py_files if 'data_processor' in f.name]) == 1:
                logger.info("✅ SUCCESS: No duplicate files created")
                return True

        await runtime.close()

    except Exception as e:
        logger.error(f"CodeAct test failed: {e}")
        return False


async def main():
    """Run tests using local OpenHands source."""

    logger.info("="*60)
    logger.info("Testing OpenHands Goal Delegation with Local Source")
    logger.info("="*60)

    # Check OpenHands is available
    openhands_dir = Path(__file__).parent / "OpenHands"
    if not openhands_dir.exists():
        logger.error(f"OpenHands source not found at {openhands_dir}")
        return False

    logger.info(f"Using OpenHands source from: {openhands_dir}")

    results = {}

    # Test 1: Direct delegation
    logger.info("\n--- Test 1: Direct OpenHands Delegation ---")
    try:
        results["direct"] = await test_openhands_direct_delegation()
    except Exception as e:
        logger.error(f"Direct delegation test error: {e}")
        results["direct"] = False

    # Test 2: CodeAct agent
    logger.info("\n--- Test 2: CodeAct Agent Specific ---")
    try:
        results["codeact"] = await test_with_codeact_agent()
    except Exception as e:
        logger.error(f"CodeAct agent test error: {e}")
        results["codeact"] = False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(results.values())
    logger.info("\n" + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))

    return all_passed


if __name__ == "__main__":
    # Set up environment
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "")

    success = asyncio.run(main())
    sys.exit(0 if success else 1)