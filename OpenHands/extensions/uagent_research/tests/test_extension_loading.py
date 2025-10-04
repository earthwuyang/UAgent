"""
Test extension loading logic independently
"""

import sys
from pathlib import Path


def test_extension_discovery():
    """Test that extension can be discovered"""
    print("Testing extension discovery...")

    # Simulate what OpenHands server does
    extensions_path = Path(__file__).parent.parent.parent.parent / 'extensions' / 'uagent_research'

    print(f"Extension path: {extensions_path}")
    print(f"Extension exists: {extensions_path.exists()}")

    if extensions_path.exists():
        print("✅ Extension directory found")

        # Check key files
        api_file = extensions_path / 'api' / '__init__.py'
        models_file = extensions_path / 'models' / 'base.py'

        print(f"  API file exists: {api_file.exists()}")
        print(f"  Models file exists: {models_file.exists()}")

        if api_file.exists() and models_file.exists():
            print("✅ Extension structure is valid")
        else:
            print("❌ Extension structure incomplete")
            return False
    else:
        print("❌ Extension directory not found")
        return False

    return True


def test_extension_imports():
    """Test that extension can be imported"""
    print("\nTesting extension imports...")

    try:
        # Add to path like OpenHands does
        extensions_path = Path(__file__).parent.parent.parent.parent / 'extensions' / 'uagent_research'
        sys.path.insert(0, str(extensions_path.parent))

        # Try importing
        from uagent_research.api import router as research_router
        from uagent_research.models.base import init_database, close_database

        print("✅ Extension imports successful")
        print(f"  Router: {research_router}")
        print(f"  Router prefix: {research_router.prefix}")
        print(f"  Number of routes: {len(research_router.routes)}")

        # List routes
        print("\n  Available routes:")
        for route in research_router.routes[:8]:  # Show first 8
            print(f"    - {route.path} [{', '.join(route.methods)}]")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_simulated_app_registration():
    """Simulate how OpenHands app.py registers the extension"""
    print("\nSimulating OpenHands app registration...")

    # Simulate the logic from app.py
    RESEARCH_EXTENSION_AVAILABLE = False
    research_router = None

    try:
        extensions_path = Path(__file__).parent.parent.parent.parent / 'extensions' / 'uagent_research'
        if extensions_path.exists():
            sys.path.insert(0, str(extensions_path.parent))
            from uagent_research.api import router as research_router
            from uagent_research.models.base import init_database, close_database
            RESEARCH_EXTENSION_AVAILABLE = True
            print("✅ Extension loaded successfully")
        else:
            print("❌ Extension path doesn't exist")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        RESEARCH_EXTENSION_AVAILABLE = False
        research_router = None

    # Simulate registration
    if RESEARCH_EXTENSION_AVAILABLE and research_router is not None:
        print("✅ Extension would be registered with OpenHands")
        print(f"   Router prefix: {research_router.prefix}")
        return True
    else:
        print("❌ Extension would NOT be registered")
        return False


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Extension Loading Test                                   ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    all_passed = True

    # Test 1: Discovery
    if not test_extension_discovery():
        all_passed = False

    # Test 2: Imports
    if not test_extension_imports():
        all_passed = False

    # Test 3: Registration
    if not test_simulated_app_registration():
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ All extension loading tests passed!")
        print("\nThe extension is ready to be loaded by OpenHands server.")
        print("When OpenHands starts, you should see:")
        print("  '✅ UAgent Research Extension loaded successfully'")
    else:
        print("❌ Some extension loading tests failed")
    print("="*60)


if __name__ == "__main__":
    main()
