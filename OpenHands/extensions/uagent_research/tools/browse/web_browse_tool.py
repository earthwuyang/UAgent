"""
Web Browse Tool using Playwright

Fetches and extracts content from web pages with human-like behavior.
No API key required.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from cachetools import TTLCache
from bs4 import BeautifulSoup
import re

from ..common.base_tool import Tool, ToolResult
from ..common.browser_manager import get_browser_pool

logger = logging.getLogger(__name__)


class WebBrowseTool(Tool):
    """
    Web page browsing and content extraction using Playwright.

    Features:
    - JavaScript rendering
    - Human-like behavior (scrolling, waiting)
    - Automatic content extraction
    - HTML cleaning and markdown conversion
    - Caching with TTL
    """

    name = "web_browse"
    description = "Browse web pages and extract content"
    cost_per_call = 0.0  # Free (no API costs)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Cache results for 1 hour
        self.cache = TTLCache(maxsize=500, ttl=3600)

        # Rate limiting
        self.rate_limit_delay = 2.0  # seconds between requests
        self.last_request_time = None

    async def invoke(
        self,
        url: str,
        extract_main_content: bool = True,
        include_links: bool = False,
        screenshot: bool = False
    ) -> ToolResult:
        """
        Browse a web page and extract content.

        Args:
            url: URL to browse
            extract_main_content: Extract only main content (remove nav, footer, etc.)
            include_links: Include links in the extracted content
            screenshot: Take a screenshot (optional)

        Returns:
            ToolResult with extracted content

        Example:
            result = await tool.invoke(url="https://arxiv.org/abs/2301.12345")
            print(result.data['content'])
            print(result.data['title'])
        """
        try:
            # Validate arguments
            is_valid, error = await self.validate_args(url=url)
            if not is_valid:
                return ToolResult(success=False, data={}, error=error)

            # Check cache
            cache_key = self.cache_key(
                url=url,
                extract_main_content=extract_main_content,
                include_links=include_links
            )
            if cache_key in self.cache:
                logger.info(f"Cache hit for URL: {url}")
                return ToolResult(success=True, data=self.cache[cache_key], cost=0.0)

            # Rate limiting
            await self._rate_limit()

            # Browse page
            content_data = await self._browse_page(
                url,
                extract_main_content=extract_main_content,
                include_links=include_links,
                screenshot=screenshot
            )

            # Cache results
            self.cache[cache_key] = content_data

            # Record call
            self.record_call(cost=0.0)

            return ToolResult(
                success=True,
                data=content_data,
                cost=0.0,
                metadata={
                    "url": url,
                    "word_count": len(content_data.get("content", "").split()),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Web browse failed for {url}: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data={},
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

    async def _browse_page(
        self,
        url: str,
        extract_main_content: bool,
        include_links: bool,
        screenshot: bool
    ) -> Dict[str, Any]:
        """
        Browse page and extract content using Playwright.

        Args:
            url: URL to browse
            extract_main_content: Extract main content only
            include_links: Include links
            screenshot: Take screenshot

        Returns:
            Dictionary with content, title, links, etc.
        """
        pool = await get_browser_pool()

        async with pool.get_page() as page:
            try:
                # Navigate to URL
                logger.info(f"Browsing: {url}")
                response = await page.goto(url, wait_until="domcontentloaded")

                # Check response status
                if response and response.status >= 400:
                    raise Exception(f"HTTP {response.status}: {response.status_text}")

                # Wait for page to stabilize
                await asyncio.sleep(2)

                # Scroll to load lazy content
                await page.evaluate("""
                    async () => {
                        await new Promise(resolve => {
                            let totalHeight = 0;
                            const distance = 100;
                            const timer = setInterval(() => {
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                if(totalHeight >= document.body.scrollHeight){
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 100);
                        });
                    }
                """)

                # Get page title
                title = await page.title()

                # Get HTML content
                html_content = await page.content()

                # Extract main content
                if extract_main_content:
                    content = await self._extract_main_content(html_content)
                else:
                    content = await self._extract_all_text(html_content)

                # Extract links if requested
                links = []
                if include_links:
                    links = await page.evaluate("""
                        () => {
                            return Array.from(document.querySelectorAll('a[href]'))
                                .map(a => ({
                                    text: a.innerText.trim(),
                                    url: a.href
                                }))
                                .filter(link => link.text && link.url);
                        }
                    """)

                # Take screenshot if requested
                screenshot_data = None
                if screenshot:
                    screenshot_data = await page.screenshot(type="png", full_page=False)

                result = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "word_count": len(content.split()),
                    "links": links[:50] if include_links else [],  # Limit to 50 links
                    "screenshot": screenshot_data
                }

                logger.info(f"Extracted {result['word_count']} words from {url}")
                return result

            except Exception as e:
                logger.error(f"Error browsing {url}: {e}")
                raise

    async def _extract_main_content(self, html: str) -> str:
        """
        Extract main content from HTML, removing navigation, footer, ads, etc.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()

        # Try to find main content area
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find(class_=re.compile(r'(content|article|main|post)', re.I)) or
            soup.find('body')
        )

        if not main_content:
            main_content = soup

        # Extract text
        text = main_content.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        text = '\n'.join(lines)

        return text

    async def _extract_all_text(self, html: str) -> str:
        """
        Extract all text from HTML.

        Args:
            html: Raw HTML content

        Returns:
            All text content
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove scripts and styles
        for element in soup.find_all(['script', 'style', 'iframe']):
            element.decompose()

        # Get all text
        text = soup.get_text(separator='\n', strip=True)

        # Clean up
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        text = '\n'.join(lines)

        return text

    async def validate_args(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate browse arguments"""
        url = kwargs.get("url")

        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"

        if not url.startswith(('http://', 'https://')):
            return False, "URL must start with http:// or https://"

        return True, None


# Example usage
async def test_web_browse():
    """Test web browse tool"""
    tool = WebBrowseTool()

    # Browse arXiv paper
    result = await tool.invoke(
        url="https://en.wikipedia.org/wiki/Neural_architecture_search",
        extract_main_content=True,
        include_links=True
    )

    if result.success:
        print(f"Title: {result.data['title']}")
        print(f"Word count: {result.data['word_count']}")
        print(f"\nContent preview:\n{result.data['content'][:500]}...")
        print(f"\nFound {len(result.data['links'])} links")
    else:
        print(f"Browse failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_web_browse())
