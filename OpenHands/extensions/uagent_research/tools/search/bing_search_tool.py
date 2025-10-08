"""
Bing Search Tool using Playwright

Performs web searches using Bing with browser automation to mimic human behavior.
No API key required.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache

from ..common.base_tool import Tool, ToolResult
from ..common.browser_manager import get_browser_pool

logger = logging.getLogger(__name__)


class BingSearchTool(Tool):
    """
    Web search using Bing via Playwright browser automation.

    Features:
    - No API key required
    - Human-like behavior (scrolling, delays)
    - Automatic result parsing
    - Caching with TTL
    - Rate limiting
    """

    name = "bing_search"
    description = "Search the web using Bing"
    cost_per_call = 0.0  # Free (no API costs)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Cache results for 1 hour
        self.cache = TTLCache(maxsize=1000, ttl=3600)

        # Rate limiting: max 10 searches per minute
        self.rate_limit_delay = 6.0  # seconds between requests
        self.last_request_time = None

    async def invoke(self, query: str, num_results: int = 10) -> ToolResult:
        """
        Search Bing for the given query.

        Args:
            query: Search query
            num_results: Number of results to return (default: 10)

        Returns:
            ToolResult with search results

        Example:
            result = await tool.invoke(query="neural networks", num_results=5)
            for item in result.data:
                print(item['title'], item['url'])
        """
        try:
            # Validate arguments
            is_valid, error = await self.validate_args(query=query, num_results=num_results)
            if not is_valid:
                return ToolResult(success=False, data=[], error=error)

            # Check cache
            cache_key = self.cache_key(query=query, num_results=num_results)
            if cache_key in self.cache:
                logger.info(f"Cache hit for query: {query}")
                return ToolResult(success=True, data=self.cache[cache_key], cost=0.0)

            # Rate limiting
            await self._rate_limit()

            # Perform search
            results = await self._search_bing(query, num_results)

            # Cache results
            self.cache[cache_key] = results

            # Record call
            self.record_call(cost=0.0)

            return ToolResult(
                success=True,
                data=results,
                cost=0.0,
                metadata={
                    "query": query,
                    "num_results": len(results),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Bing search failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=[],
                error=str(e)
            )

    async def _rate_limit(self):
        """Apply rate limiting"""
        if self.last_request_time:
            elapsed = (datetime.utcnow() - self.last_request_time).total_seconds()
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        self.last_request_time = datetime.utcnow()

    async def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Perform actual Bing search using Playwright.

        Args:
            query: Search query
            num_results: Number of results to extract

        Returns:
            List of search results with title, url, snippet
        """
        
        logger.info(f"[BING_TOOL] _search_bing() called for query: {query}")
        pool = await get_browser_pool()

        async with pool.get_page() as page:
            try:
                # Navigate to Bing
                logger.info(f"Searching Bing for: {query}")
                await page.goto("https://www.bing.com", wait_until="domcontentloaded")

                # Wait for search box
                logger.info(f"[BING_TOOL] Waiting for search box")
                await page.wait_for_selector('input[name="q"]', state="visible")

                # Type query with human-like delays
                search_box = await page.query_selector('input[name="q"]')
                await search_box.type(query, delay=100)  # 100ms between keystrokes

                # Submit search
                await search_box.press("Enter")
                        # Wait for results
            logger.info(f"[BING_TOOL] Waiting for search results")
            await page.wait_for_selector('li.b_algo', state="visible", timeout=10000)

                # Small delay to mimic human reading
                await asyncio.sleep(1)
                        # Extract results
            logger.info(f"[BING_TOOL] Extracting search results")
            results = await page.evaluate(f"""
                    () => {{
                        const items = Array.from(document.querySelectorAll('li.b_algo'));
                        const maxResults = {num_results};

                        return items.slice(0, maxResults).map(item => {{
                            const titleEl = item.querySelector('h2 a');
                            const snippetEl = item.querySelector('.b_caption p, .b_caption');

                            return {{
                                title: titleEl ? titleEl.innerText : '',
                                url: titleEl ? titleEl.href : '',
                                snippet: snippetEl ? snippetEl.innerText : '',
                                source: 'bing'
                            }};
                        }}).filter(item => item.title && item.url);
                    }}
                """)

                logger.info(f"Extracted {len(results)} results from Bing")
                return results

            except Exception as e:
                logger.error(f"Error during Bing search: {e}")
                raise

    async def validate_args(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate search arguments"""
        query = kwargs.get("query")
        num_results = kwargs.get("num_results", 10)

        if not query or not isinstance(query, str):
            return False, "Query must be a non-empty string"

        if query.strip() == "":
            return False, "Query cannot be empty"

        if not isinstance(num_results, int) or num_results < 1 or num_results > 100:
            return False, "num_results must be an integer between 1 and 100"

        return True, None


# Example usage
async def test_bing_search():
    """Test Bing search tool"""
    tool = BingSearchTool()

    # Search for neural networks
    result = await tool.invoke(query="neural architecture search", num_results=5)

    if result.success:
        print(f"Found {len(result.data)} results:")
        for i, item in enumerate(result.data, 1):
            print(f"\n{i}. {item['title']}")
            print(f"   URL: {item['url']}")
            print(f"   Snippet: {item['snippet'][:100]}...")
    else:
        print(f"Search failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_bing_search())
