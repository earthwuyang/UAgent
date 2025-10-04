"""
Test OpenHands server integration with research extension
"""

import asyncio
import httpx
import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_extension_can_be_imported():
    """Test that the extension can be imported from OpenHands"""
    import sys
    extensions_path = Path(__file__).parent.parent.parent.parent / 'extensions'
    sys.path.insert(0, str(extensions_path))

    from uagent_research.api import router
    from uagent_research.models.base import init_database, close_database

    assert router is not None
    assert init_database is not None
    assert close_database is not None


@pytest.mark.asyncio
async def test_database_initialization():
    """Test that database can be initialized"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'extensions'))

    from uagent_research.models.base import init_database, close_database

    # Initialize with in-memory database
    await init_database("sqlite+aiosqlite:///:memory:")

    # Close connections
    await close_database()


@pytest.mark.asyncio
async def test_api_routes_exist():
    """Test that API routes are properly configured"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'extensions'))

    from uagent_research.api import router

    # Check that router has routes
    assert len(router.routes) > 0

    # Verify expected endpoints
    route_paths = [route.path for route in router.routes]

    expected_paths = [
        "/api/research/experiments/start",
        "/api/research/experiments/{experiment_id}",
        "/api/research/experiments",
        "/api/research/health",
    ]

    for path in expected_paths:
        # Remove prefix for comparison
        path_without_prefix = path.replace("/api/research", "")
        assert any(path_without_prefix in route_path for route_path in route_paths), \
            f"Expected route {path} not found"


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_extension_can_be_imported())
    print("✅ Extension import test passed")

    asyncio.run(test_database_initialization())
    print("✅ Database initialization test passed")

    asyncio.run(test_api_routes_exist())
    print("✅ API routes test passed")

    print("\n✅ All integration tests passed!")
