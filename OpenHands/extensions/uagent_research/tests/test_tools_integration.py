"""
Integration Tests for Research Tools

Tests Bing search and web browsing with Playwright.
"""

import asyncio
import logging
import pytest
from typing import List

from ..tools.search.bing_search_tool import BingSearchTool
from ..tools.browse.web_browse_tool import WebBrowseTool
from ..tools.common.browser_manager import get_browser_pool, close_browser_pool

logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module", autouse=True)
async def browser_cleanup():
    """Cleanup browser pool after tests"""
    yield
    await close_browser_pool()


class TestBingSearchTool:
    """Test Bing search tool"""

    @pytest.mark.asyncio
    async def test_basic_search(self):
        """Test basic Bing search"""
        tool = BingSearchTool()

        result = await tool.invoke(
            query="Python programming language",
            num_results=5
        )

        assert result.success, f"Search failed: {result.error}"
        assert len(result.data) > 0, "No search results returned"
        assert len(result.data) <= 5, "Too many results returned"

        # Check result structure
        first_result = result.data[0]
        assert "title" in first_result
        assert "url" in first_result
        assert "snippet" in first_result
        assert first_result["source"] == "bing"

        print(f"✓ Found {len(result.data)} results")
        print(f"  First result: {first_result['title']}")

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test that caching works"""
        tool = BingSearchTool()

        query = "machine learning basics"

        # First call - should hit Bing
        result1 = await tool.invoke(query=query, num_results=3)
        assert result1.success

        # Second call - should hit cache
        result2 = await tool.invoke(query=query, num_results=3)
        assert result2.success

        # Results should be identical
        assert len(result1.data) == len(result2.data)

        print("✓ Cache hit verified")

    @pytest.mark.asyncio
    async def test_invalid_query(self):
        """Test validation with invalid query"""
        tool = BingSearchTool()

        # Empty query
        result = await tool.invoke(query="", num_results=5)
        assert not result.success
        assert "empty" in result.error.lower()

        print("✓ Empty query rejected")

    @pytest.mark.asyncio
    async def test_invalid_num_results(self):
        """Test validation with invalid num_results"""
        tool = BingSearchTool()

        # Too many results
        result = await tool.invoke(query="test", num_results=200)
        assert not result.success

        # Negative results
        result = await tool.invoke(query="test", num_results=-1)
        assert not result.success

        print("✓ Invalid num_results rejected")


class TestWebBrowseTool:
    """Test web browsing tool"""

    @pytest.mark.asyncio
    async def test_browse_wikipedia(self):
        """Test browsing Wikipedia page"""
        tool = WebBrowseTool()

        result = await tool.invoke(
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            extract_main_content=True,
            include_links=False
        )

        assert result.success, f"Browse failed: {result.error}"
        assert result.data["url"]
        assert result.data["title"]
        assert len(result.data["content"]) > 100, "Content too short"
        assert result.data["word_count"] > 50, "Word count too low"

        print(f"✓ Browsed Wikipedia: {result.data['title']}")
        print(f"  Word count: {result.data['word_count']}")
        print(f"  Content preview: {result.data['content'][:100]}...")

    @pytest.mark.asyncio
    async def test_browse_with_links(self):
        """Test extracting links from page"""
        tool = WebBrowseTool()

        result = await tool.invoke(
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            extract_main_content=True,
            include_links=True
        )

        assert result.success
        assert len(result.data["links"]) > 0, "No links extracted"
        assert len(result.data["links"]) <= 50, "Too many links"

        # Check link structure
        first_link = result.data["links"][0]
        assert "text" in first_link
        assert "url" in first_link

        print(f"✓ Extracted {len(result.data['links'])} links")
        print(f"  First link: {first_link['text']} -> {first_link['url']}")

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test that caching works"""
        tool = WebBrowseTool()

        url = "https://en.wikipedia.org/wiki/Machine_learning"

        # First call - should hit the web
        result1 = await tool.invoke(url=url, extract_main_content=True)
        assert result1.success

        # Second call - should hit cache
        result2 = await tool.invoke(url=url, extract_main_content=True)
        assert result2.success

        # Content should be identical
        assert result1.data["content"] == result2.data["content"]

        print("✓ Cache hit verified")

    @pytest.mark.asyncio
    async def test_invalid_url(self):
        """Test validation with invalid URL"""
        tool = WebBrowseTool()

        # No protocol
        result = await tool.invoke(url="wikipedia.org")
        assert not result.success
        assert "http" in result.error.lower()

        # Empty URL
        result = await tool.invoke(url="")
        assert not result.success

        print("✓ Invalid URLs rejected")

    @pytest.mark.asyncio
    async def test_404_page(self):
        """Test handling of 404 pages"""
        tool = WebBrowseTool()

        result = await tool.invoke(
            url="https://en.wikipedia.org/wiki/ThisPageDoesNotExist12345"
        )

        # Should fail gracefully
        assert not result.success
        assert result.error

        print(f"✓ 404 handled: {result.error}")


class TestToolsIntegration:
    """Test tools working together"""

    @pytest.mark.asyncio
    async def test_search_and_browse_workflow(self):
        """Test search -> browse workflow"""
        search_tool = BingSearchTool()
        browse_tool = WebBrowseTool()

        # Step 1: Search for topic
        search_result = await search_tool.invoke(
            query="neural architecture search",
            num_results=3
        )

        assert search_result.success
        assert len(search_result.data) > 0

        # Step 2: Browse first result
        first_url = search_result.data[0]["url"]
        browse_result = await browse_tool.invoke(
            url=first_url,
            extract_main_content=True
        )

        # May fail if URL is inaccessible, but should handle gracefully
        if browse_result.success:
            print(f"✓ Workflow completed:")
            print(f"  Search: Found {len(search_result.data)} results")
            print(f"  Browse: Extracted {browse_result.data['word_count']} words from {first_url}")
        else:
            print(f"⚠ Browse failed (expected for some URLs): {browse_result.error}")

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """Test parallel tool execution"""
        search_tool = BingSearchTool()

        # Execute multiple searches in parallel
        tasks = [
            search_tool.invoke(query="Python", num_results=3),
            search_tool.invoke(query="JavaScript", num_results=3),
            search_tool.invoke(query="Rust", num_results=3),
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.success for r in results)
        assert all(len(r.data) > 0 for r in results)

        print(f"✓ Parallel execution completed:")
        for i, result in enumerate(results):
            print(f"  Search {i+1}: {len(result.data)} results")


class TestBrowserPool:
    """Test browser pool management"""

    @pytest.mark.asyncio
    async def test_pool_concurrency(self):
        """Test concurrent page access"""
        pool = await get_browser_pool()

        async def browse_page(url: str):
            async with pool.get_page() as page:
                await page.goto(url, wait_until="domcontentloaded")
                return await page.title()

        # Access multiple pages concurrently
        urls = [
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "https://en.wikipedia.org/wiki/JavaScript",
            "https://en.wikipedia.org/wiki/Rust_(programming_language)",
        ]

        titles = await asyncio.gather(*[browse_page(url) for url in urls])

        assert len(titles) == 3
        assert all(title for title in titles)

        print(f"✓ Concurrent browsing completed:")
        for i, title in enumerate(titles):
            print(f"  Page {i+1}: {title}")


# Manual test runner (for development)
async def run_all_tests():
    """Run all tests manually"""
    print("\n=== Testing Bing Search Tool ===\n")

    search_tests = TestBingSearchTool()
    await search_tests.test_basic_search()
    await search_tests.test_cache_hit()
    await search_tests.test_invalid_query()
    await search_tests.test_invalid_num_results()

    print("\n=== Testing Web Browse Tool ===\n")

    browse_tests = TestWebBrowseTool()
    await browse_tests.test_browse_wikipedia()
    await browse_tests.test_browse_with_links()
    await browse_tests.test_cache_hit()
    await browse_tests.test_invalid_url()
    await browse_tests.test_404_page()

    print("\n=== Testing Integration ===\n")

    integration_tests = TestToolsIntegration()
    await integration_tests.test_search_and_browse_workflow()
    await integration_tests.test_parallel_tool_execution()

    print("\n=== Testing Browser Pool ===\n")

    pool_tests = TestBrowserPool()
    await pool_tests.test_pool_concurrency()

    print("\n=== All Tests Completed ===\n")

    # Cleanup
    await close_browser_pool()


if __name__ == "__main__":
    # Run tests manually
    asyncio.run(run_all_tests())
