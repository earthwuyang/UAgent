#!/usr/bin/env python3
"""
UAgent Research Extension - Installation Verification Script

Run this script to verify that the extension is properly installed and configured.
"""

import asyncio
import sys
from pathlib import Path


def print_header(text):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


async def verify_imports():
    """Verify extension can be imported"""
    print_header("Step 1: Verifying Imports")

    try:
        # Add extension to path
        ext_path = Path(__file__).parent.parent
        sys.path.insert(0, str(ext_path))

        # Test imports (skip engines/agents that require OpenHands runtime)
        from uagent_research.models import Experiment, ResearchSession, Idea, Hypothesis
        print_success("Models imported successfully")

        from uagent_research.models.base import init_database, close_database, get_session
        print_success("Database infrastructure imported successfully")

        # Skip engines and agents - they require OpenHands runtime
        print_info("Skipping engines/agents (require OpenHands runtime)")

        from uagent_research.api import router, ws_router, ws_manager
        print_success("API routes imported successfully")

        print_info(f"HTTP Router prefix: {router.prefix}")
        print_info(f"HTTP Routes: {len(router.routes)}")
        print_info(f"WebSocket Router prefix: {ws_router.prefix}")
        print_info(f"WebSocket Routes: {len(ws_router.routes)}")

        return True

    except ImportError as e:
        print_error(f"Import failed: {e}")
        return False


async def verify_database():
    """Verify database can be initialized"""
    print_header("Step 2: Verifying Database")

    try:
        from uagent_research.models.base import init_database, close_database

        # Initialize in-memory database
        await init_database("sqlite+aiosqlite:///:memory:")
        print_success("Database initialized successfully")

        # Close
        await close_database()
        print_success("Database connections closed successfully")

        return True

    except Exception as e:
        print_error(f"Database verification failed: {e}")
        return False


async def verify_models():
    """Verify models can be created"""
    print_header("Step 3: Verifying Models")

    try:
        from uagent_research.models.base import init_database, close_database, get_session
        from uagent_research.models import (
            Experiment,
            ExperimentType,
            ExperimentStatus,
            ResearchSession,
            SessionMode,
        )

        # Initialize database
        await init_database("sqlite+aiosqlite:///:memory:")

        # Test model creation
        async for session in get_session():
            # Create session
            research_session = ResearchSession(
                id="verify_session_1",
                user_id="verify_user",
                mode=SessionMode.RESEARCH,
                title="Verification Test",
            )
            session.add(research_session)
            await session.commit()
            print_success("ResearchSession created")

            # Create experiment
            experiment = Experiment(
                id="verify_exp_1",
                session_id="verify_session_1",
                experiment_type=ExperimentType.SCIENTIFIC,
                goal="Verification test",
                status=ExperimentStatus.PENDING,
            )
            session.add(experiment)
            await session.commit()
            print_success("Experiment created")

            # Test serialization
            exp_dict = experiment.to_dict()
            assert "id" in exp_dict
            assert "status" in exp_dict
            print_success("Model serialization works")

        await close_database()
        return True

    except Exception as e:
        print_error(f"Model verification failed: {e}")
        await close_database()
        return False


async def verify_api_routes():
    """Verify API routes are configured"""
    print_header("Step 4: Verifying API Routes")

    try:
        from uagent_research.api import router

        # Check routes
        routes = [route.path for route in router.routes]

        expected = [
            "/experiments/start",
            "/experiments/{experiment_id}",
            "/experiments",
            "/health",
        ]

        for expected_path in expected:
            if any(expected_path in route for route in routes):
                print_success(f"Route found: {expected_path}")
            else:
                print_error(f"Route missing: {expected_path}")
                return False

        print_info(f"Total HTTP routes: {len(routes)}")
        return True

    except Exception as e:
        print_error(f"API route verification failed: {e}")
        return False


async def verify_websocket():
    """Verify WebSocket functionality"""
    print_header("Step 5: Verifying WebSocket")

    try:
        from uagent_research.api.websocket_routes import ConnectionManager

        # Create manager
        manager = ConnectionManager()
        print_success("ConnectionManager created")

        # Mock WebSocket
        class MockWebSocket:
            def __init__(self):
                self.messages = []

            async def accept(self):
                pass

            async def send_json(self, data):
                self.messages.append(data)

        # Test connection
        ws = MockWebSocket()
        await manager.connect(ws, experiment_id="test_exp")
        print_success("WebSocket connection established")

        # Test sending update
        await manager.send_experiment_update("test_exp", {"type": "test", "data": "hello"})
        print_success("WebSocket update sent")

        # Verify message received
        assert len(ws.messages) == 1
        assert ws.messages[0]["type"] == "test"
        print_success("WebSocket message received")

        # Test disconnect
        manager.disconnect(ws, experiment_id="test_exp")
        print_success("WebSocket disconnection handled")

        return True

    except Exception as e:
        print_error(f"WebSocket verification failed: {e}")
        return False


async def verify_openhands_integration():
    """Verify OpenHands integration"""
    print_header("Step 6: Verifying OpenHands Integration")

    try:
        # Check if extension directory exists in expected location
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "openhands" / "server" / "app.py",
            Path("/home/wuy/AI/UAgent/OpenHands/openhands/server/app.py"),
        ]

        app_path = None
        for path in possible_paths:
            if path.exists():
                app_path = path
                break

        if not app_path:
            print_info("OpenHands server app.py not found (running standalone)")
            print_info("This is OK if you're just verifying the extension")
            return True  # Don't fail if running standalone

        print_success(f"OpenHands server found at: {app_path}")

        # Check if app.py contains extension loading code
        with open(app_path, 'r') as f:
            content = f.read()

        if "uagent_research" in content:
            print_success("Extension loading code found in app.py")
        else:
            print_error("Extension loading code NOT found in app.py")
            return False

        if "RESEARCH_EXTENSION_AVAILABLE" in content:
            print_success("Extension availability check found")
        else:
            print_error("Extension availability check NOT found")
            return False

        return True

    except Exception as e:
        print_error(f"OpenHands integration verification failed: {e}")
        return False


async def main():
    """Run all verification steps"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  UAgent Research Extension - Installation Verification    ║")
    print("╚════════════════════════════════════════════════════════════╝")

    results = []

    # Run verification steps
    results.append(("Imports", await verify_imports()))
    results.append(("Database", await verify_database()))
    results.append(("Models", await verify_models()))
    results.append(("API Routes", await verify_api_routes()))
    results.append(("WebSocket", await verify_websocket()))
    results.append(("OpenHands Integration", await verify_openhands_integration()))

    # Summary
    print_header("Verification Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")

    print("\n" + "=" * 60)

    if passed == total:
        print("🎉 All verification checks passed!")
        print("\nYour UAgent Research Extension is properly installed.")
        print("\nNext steps:")
        print("  1. Set database URL (optional):")
        print("     export RESEARCH_DATABASE_URL='sqlite+aiosqlite:///./research.db'")
        print("  2. Start OpenHands server:")
        print("     cd /home/wuy/AI/UAgent/OpenHands")
        print("     python -m openhands.server.listen")
        print("  3. Check for success message:")
        print("     ✅ UAgent Research Extension loaded successfully")
        return 0
    else:
        print(f"⚠️  {total - passed} verification check(s) failed.")
        print("\nPlease review the errors above and:")
        print("  1. Ensure all dependencies are installed:")
        print("     pip install -e .")
        print("  2. Check that you're in the correct directory")
        print("  3. Verify OpenHands integration was completed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
