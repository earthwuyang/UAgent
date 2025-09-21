"""Playwright-based web capture utilities for surfacing PDF content."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    async_playwright = None  # type: ignore
    PlaywrightTimeoutError = Exception  # type: ignore

LOGGER = logging.getLogger(__name__)


class PlaywrightCaptureService:
    """Capture page screenshots using Playwright (Chromium)."""

    def __init__(
        self,
        headless: bool = True,
        viewport: Optional[dict] = None,
        launch_args: Optional[List[str]] = None,
        navigation_timeout_ms: int = 45_000,
        scroll_pause: float = 1.2,
    ) -> None:
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 900}
        self.launch_args = launch_args or [
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--no-sandbox",
        ]
        self.navigation_timeout_ms = navigation_timeout_ms
        self.scroll_pause = scroll_pause

    @property
    def available(self) -> bool:
        return async_playwright is not None

    async def capture_pdf(self, url: str, max_pages: int = 3) -> List[bytes]:
        """Navigate to a PDF URL and capture scrolling screenshots."""

        if async_playwright is None:
            raise RuntimeError("playwright is not installed; install playwright to enable web capture")

        screenshots: List[bytes] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless, args=self.launch_args)
            try:
                context = await browser.new_context(viewport=self.viewport)
                page = await context.new_page()
                try:
                    await page.goto(url, wait_until="networkidle", timeout=self.navigation_timeout_ms)
                except PlaywrightTimeoutError:
                    LOGGER.warning("Timeout navigating to %s", url)
                await asyncio.sleep(self.scroll_pause)

                for page_index in range(max_pages):
                    screenshot = await page.screenshot(full_page=True)
                    screenshots.append(screenshot)
                    await page.evaluate("window.scrollBy(0, window.innerHeight);")
                    await asyncio.sleep(self.scroll_pause)

            finally:
                await browser.close()

        return screenshots


__all__ = ["PlaywrightCaptureService"]
