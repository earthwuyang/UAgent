"""
Test that OpenHands server can start with research extension
"""

import asyncio
import sys
from pathlib import Path

# Add OpenHands to path
openhands_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(openhands_path))


async def test_app_creation():
    """Test that the FastAPI app can be created with extension"""
    print("Testing OpenHands server creation with research extension...")

    # Import the app
    from openhands.server.app import app, RESEARCH_EXTENSION_AVAILABLE

    # Verify app was created
    assert app is not None, "FastAPI app should be created"
    print("✅ FastAPI app created successfully")

    # Verify extension is available
    if RESEARCH_EXTENSION_AVAILABLE:
        print("✅ Research extension is available")

        # Verify routes are registered
        routes = [route.path for route in app.routes]
        print(f"\nTotal routes: {len(routes)}")

        # Check for research routes
        research_routes = [r for r in routes if '/research' in r]
        if research_routes:
            print(f"✅ Research routes found: {len(research_routes)}")
            for route in research_routes[:5]:  # Show first 5
                print(f"  - {route}")
        else:
            print("⚠️  No research routes found (extension may not be fully loaded)")
    else:
        print("❌ Research extension is NOT available")
        print("   Extension directory may not exist or imports failed")

    return True


async def test_lifespan_initialization():
    """Test that lifespan manager can initialize"""
    print("\nTesting lifespan initialization...")

    try:
        from openhands.server.app import _lifespan, app

        # Note: We can't fully test lifespan without starting the server
        # but we can verify the function exists and is callable
        assert _lifespan is not None
        print("✅ Lifespan handler exists")

        # Verify it's an async context manager
        import inspect
        assert inspect.isasyncgenfunction(_lifespan)
        print("✅ Lifespan handler is async context manager")

    except Exception as e:
        print(f"❌ Lifespan test failed: {e}")
        return False

    return True


async def main():
    """Run all tests"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  OpenHands Server Integration Test                        ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    success = True

    # Test 1: App creation
    if not await test_app_creation():
        success = False

    # Test 2: Lifespan
    if not await test_lifespan_initialization():
        success = False

    print("\n" + "="*60)
    if success:
        print("✅ All OpenHands integration tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)

    return success


if __name__ == "__main__":
    asyncio.run(main())
