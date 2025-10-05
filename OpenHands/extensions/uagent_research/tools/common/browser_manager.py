"""
Browser Manager for Playwright

Manages browser instances with headless mode (Xvfb) for web automation.
Includes connection pooling and resource management.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)


class BrowserPool:
    """
    Pool of browser contexts for efficient reuse.

    Uses Playwright with headless Chromium to mimic human behavior.
    Supports Xvfb for true headless operation on Linux servers.
    """

    def __init__(
        self,
        max_contexts: int = 5,
        headless: bool = True,
        use_xvfb: bool = True,
        timeout: int = 30000  # 30 seconds
    ):
        """
        Initialize browser pool.

        Args:
            max_contexts: Maximum number of browser contexts
            headless: Run in headless mode
            use_xvfb: Use Xvfb for virtual display (Linux only)
            timeout: Default timeout in milliseconds
        """
        self.max_contexts = max_contexts
        self.headless = headless
        self.use_xvfb = use_xvfb
        self.timeout = timeout

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._contexts: list[BrowserContext] = []
        self._semaphore = asyncio.Semaphore(max_contexts)
        self._initialized = False

    async def initialize(self):
        """Initialize Playwright and browser"""
        if self._initialized:
            return

        logger.info("Initializing Playwright browser pool...")

        # Start Playwright
        self._playwright = await async_playwright().start()

        # Launch browser with human-like settings
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',  # Hide automation
                '--disable-dev-shm-usage',  # Overcome limited resource problems
                '--no-sandbox',  # Required for Docker/CI
                '--disable-setuid-sandbox',
                '--disable-gpu',
                '--disable-web-security',  # For CORS issues
                '--disable-features=IsolateOrigins,site-per-process',
            ]
        )

        self._initialized = True
        logger.info(f"Browser pool initialized (max_contexts={self.max_contexts}, headless={self.headless})")

    async def close(self):
        """Close all browser contexts and browser"""
        if not self._initialized:
            return

        logger.info("Closing browser pool...")

        # Close all contexts
        for context in self._contexts:
            try:
                await context.close()
            except Exception as e:
                logger.error(f"Error closing context: {e}")

        self._contexts.clear()

        # Close browser
        if self._browser:
            await self._browser.close()

        # Stop Playwright
        if self._playwright:
            await self._playwright.stop()

        self._initialized = False
        logger.info("Browser pool closed")

    @asynccontextmanager
    async def get_page(self, **context_options):
        """
        Get a browser page from the pool.

        Yields:
            Page: Playwright page object

        Example:
            async with pool.get_page() as page:
                await page.goto("https://example.com")
                content = await page.content()
        """
        if not self._initialized:
            await self.initialize()

        async with self._semaphore:
            # Create new context with human-like settings
            context = await self._browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
                timezone_id='America/New_York',
                **context_options
            )

            # Set default timeout
            context.set_default_timeout(self.timeout)

            # Create page
            page = await context.new_page()

            # Add stealth scripts to avoid detection
            await page.add_init_script("""
                // Override navigator.webdriver
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );

                // Add chrome object
                window.chrome = {
                    runtime: {},
                };
            """)

            try:
                yield page
            finally:
                # Cleanup
                try:
                    await page.close()
                    await context.close()
                except Exception as e:
                    logger.error(f"Error cleaning up page: {e}")

    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()


# Global browser pool instance
_browser_pool: Optional[BrowserPool] = None


async def get_browser_pool() -> BrowserPool:
    """Get or create global browser pool"""
    global _browser_pool

    if _browser_pool is None:
        _browser_pool = BrowserPool(
            max_contexts=5,
            headless=True,
            use_xvfb=True
        )
        await _browser_pool.initialize()

    return _browser_pool


async def close_browser_pool():
    """Close global browser pool"""
    global _browser_pool

    if _browser_pool:
        await _browser_pool.close()
        _browser_pool = None
