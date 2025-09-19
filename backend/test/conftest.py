"""Test configuration and fixtures"""

import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator

from app.core.llm_client import create_llm_client, LLMClient
from app.core.cache import create_cache, Cache
from app.core.smart_router import SmartRouter


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def llm_client() -> LLMClient:
    """Create LLM client for testing - ALWAYS use real DashScope"""
    # Check for DashScope API key
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")

    if dashscope_key:
        return create_llm_client("dashscope", api_key=dashscope_key)
    else:
        # Skip tests if no API key available
        pytest.skip("DASHSCOPE_API_KEY not available for real LLM testing")


@pytest.fixture
def cache() -> Cache:
    """Create cache instance for testing"""
    return create_cache("memory")


@pytest.fixture
def smart_router(llm_client: LLMClient, cache: Cache) -> SmartRouter:
    """Create smart router with LLM client and cache"""
    config = {
        "max_retries": 3,
        "cache_ttl": 3600
    }
    return SmartRouter(llm_client=llm_client, cache=cache, config=config)


@pytest.fixture
def router_config() -> dict:
    """Configuration for router testing"""
    return {
        "max_retries": 2,
        "cache_ttl": 1800
    }


# Test markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "real_llm: tests that require real LLM API calls"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle real LLM tests"""
    if config.getoption("--no-real-llm", default=False):
        skip_real_llm = pytest.mark.skip(reason="--no-real-llm option given")
        for item in items:
            if "real_llm" in item.keywords:
                item.add_marker(skip_real_llm)


def pytest_addoption(parser):
    """Add command line options"""
    parser.addoption(
        "--no-real-llm",
        action="store_true",
        default=False,
        help="skip tests that require real LLM API calls"
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="run performance tests"
    )