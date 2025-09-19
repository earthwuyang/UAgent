from __future__ import annotations

import asyncio
import os
import subprocess
from typing import List, Dict, Any

from playwright.async_api import async_playwright, Page
try:
    # Optional stealth to reduce bot detection
    from playwright_stealth import stealth_async
except Exception:  # pragma: no cover
    stealth_async = None
import random


class PlaywrightSearchEngine:
    """
    Human-like web search using Playwright with an optional Xvfb virtual display.
    Strategy: try Bing first (less strict), then DuckDuckGo, then Google.

    Note: Requires browsers installed via `python -m playwright install chromium`.
    """

    def __init__(self, timeout_ms: int = 30000) -> None:
        self.timeout = timeout_ms
        self.user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.proxy = None
        proxy = os.environ.get("UAGENT_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        if proxy:
            self.proxy = {"server": proxy}

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        # Prefer DDG Lite (static HTML, fewer challenges) → DDG → Bing → Google
        for fn in (self._duckduckgo_lite, self._duckduckgo, self._bing, self._google):
            results = await self._with_xvfb(fn, query, max_results)
            if results:
                return results[:max_results]
        return []

    async def _with_xvfb(self, fn, *args, **kwargs):
        # If DISPLAY exists, just run
        if os.environ.get("DISPLAY"):
            return await fn(*args, **kwargs)
        # Try to start Xvfb :99 and set DISPLAY
        try:
            xvfb_env = os.environ.copy()
            xvfb_env["DISPLAY"] = ":99"
            # is Xvfb already up?
            try:
                subprocess.run(["pgrep", "-f", "Xvfb :99"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                await asyncio.sleep(1.5)

            original_display = os.environ.get("DISPLAY")
            os.environ["DISPLAY"] = ":99"
            try:
                return await fn(*args, **kwargs)
            finally:
                if original_display is not None:
                    os.environ["DISPLAY"] = original_display
                else:
                    os.environ.pop("DISPLAY", None)
        except Exception:
            # Fallback: run without Xvfb
            return await fn(*args, **kwargs)

    async def _bing(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"], proxy=self.proxy)  # noqa: E501
            context = await browser.new_context(
                user_agent=self.user_agent,
                locale="en-US",
                timezone_id="UTC",
                viewport={"width": 1366, "height": 768},
            )
            page = await context.new_page()
            if stealth_async:
                await stealth_async(page)
            results: List[Dict[str, Any]] = []
            try:
                url = f"https://www.bing.com/search?q={query}"
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                await self._humanize(page)
                # Classic Bing selectors
                items = await page.query_selector_all("li.b_algo")
                for it in items:
                    a = await it.query_selector("h2 a")
                    if not a:
                        continue
                    title = (await a.inner_text()) or ""
                    link = await a.get_attribute("href")
                    if not link:
                        continue
                    snippet_el = await it.query_selector(".b_caption p, .b_snippet, p")
                    snippet = (await snippet_el.inner_text()) if snippet_el else ""
                    results.append({"title": title.strip(), "snippet": snippet.strip(), "link": link, "engine": "playwright_bing"})
                    if len(results) >= max_results:
                        break
            finally:
                await browser.close()
            return results

    async def _duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"], proxy=self.proxy)  # noqa: E501
            context = await browser.new_context(
                user_agent=self.user_agent,
                locale="en-US",
                timezone_id="UTC",
                viewport={"width": 1366, "height": 768},
            )
            page = await context.new_page()
            if stealth_async:
                await stealth_async(page)
            results: List[Dict[str, Any]] = []
            try:
                import urllib.parse as up
                url = f"https://duckduckgo.com/?q={up.quote(query)}"
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                await self._humanize(page)
                # DDG plays with DOM; try multiple selectors
                selectors = [
                    "a[data-testid='result-title-a']",
                    "article div a",
                    "#links .result__title a",
                ]
                elements = []
                for sel in selectors:
                    try:
                        elements = await page.query_selector_all(sel)
                        if elements:
                            break
                    except Exception:
                        continue
                for a in elements:
                    title = (await a.inner_text()) or ""
                    link = await a.get_attribute("href")
                    if not link or link.startswith("/"):
                        continue
                    # Try to get a nearby snippet
                    parent = await a.evaluate_handle("(el) => el.closest('article') || el.parentElement")
                    snippet = ""
                    try:
                        snippet_el = await parent.as_element().query_selector("p, .result__snippet, .result-snippet")
                        if snippet_el:
                            snippet = (await snippet_el.inner_text()) or ""
                    except Exception:
                        pass
                    results.append({"title": title.strip(), "snippet": snippet.strip(), "link": link, "engine": "playwright_duckduckgo"})
                    if len(results) >= max_results:
                        break
            finally:
                await browser.close()
            return results

    async def _google(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"], proxy=self.proxy)  # noqa: E501
            context = await browser.new_context(
                user_agent=self.user_agent,
                locale="en-US",
                timezone_id="UTC",
                viewport={"width": 1366, "height": 768},
            )
            page = await context.new_page()
            if stealth_async:
                await stealth_async(page)
            results: List[Dict[str, Any]] = []
            try:
                import urllib.parse as up
                url = f"https://www.google.com/search?q={up.quote(query)}&num={max_results}"
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                await self._humanize(page)
                # Try common Google result containers
                selectors = [
                    "div.g",
                    "#search .g",
                    "[data-ved] .g",
                ]
                elements = []
                for sel in selectors:
                    try:
                        elements = await page.query_selector_all(sel)
                        if elements:
                            break
                    except Exception:
                        continue
                for el in elements:
                    a = await el.query_selector("a[href][jsname] , a[href] h3 >> xpath=ancestor::a[1]")
                    if not a:
                        a = await el.query_selector("a[href]")
                    if not a:
                        continue
                    href = await a.get_attribute("href")
                    if not href or href.startswith("/" ):
                        continue
                    title_el = await el.query_selector("h3")
                    title = (await title_el.inner_text()) if title_el else ""
                    snippet_el = await el.query_selector(".VwiC3b, .IsZvec, .Uroaid, span")
                    snippet = (await snippet_el.inner_text()) if snippet_el else ""
                    results.append({"title": title.strip(), "snippet": snippet.strip(), "link": href, "engine": "playwright_google"})
                    if len(results) >= max_results:
                        break
            finally:
                await browser.close()
            return results

    async def _humanize(self, page: Page):
        # Random small delay
        await page.wait_for_timeout(300 + int(random.random() * 700))
        # Random scrolls
        try:
            height = await page.evaluate("() => document.body.scrollHeight")
            steps = random.randint(2, 4)
            for i in range(steps):
                y = int((i + 1) * (height / (steps + 1)))
                await page.mouse.wheel(0, y)
                await page.wait_for_timeout(200 + int(random.random() * 400))
        except Exception:
            pass

    async def _duckduckgo_lite(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Use DuckDuckGo lite (static HTML) which is bot-friendlier and stable to parse."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"], proxy=self.proxy)  # noqa: E501
            context = await browser.new_context(
                user_agent=self.user_agent,
                locale="en-US",
                timezone_id="UTC",
                viewport={"width": 1200, "height": 800},
            )
            page = await context.new_page()
            if stealth_async:
                await stealth_async(page)
            results: List[Dict[str, Any]] = []
            try:
                import urllib.parse as up
                url = f"https://lite.duckduckgo.com/lite/?q={up.quote(query)}"
                await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
                await self._humanize(page)
                # Links are td.result-link > a
                anchors = await page.query_selector_all("td.result-link a")
                for a in anchors:
                    title = (await a.inner_text()) or ""
                    link = await a.get_attribute("href")
                    if not link or link.startswith("/"):
                        continue
                    # Snippet is nearby in sibling td (result-snippet)
                    parent = await a.evaluate_handle("a => a.closest('tr')")
                    snippet = ""
                    try:
                        sib = await parent.as_element().query_selector("td.result-snippet")
                        if sib:
                            snippet = (await sib.inner_text()) or ""
                    except Exception:
                        pass
                    results.append({
                        "title": title.strip(),
                        "snippet": snippet.strip(),
                        "link": link,
                        "engine": "playwright_duckduckgo_lite",
                    })
                    if len(results) >= max_results:
                        break
            finally:
                await browser.close()
            return results
