import asyncio
import os
from pathlib import Path

from app.utils.playwright_search import PlaywrightSearchEngine
from playwright.async_api import async_playwright


async def main():
    out = Path(__file__).resolve().parent / 'playwright_debug.png'
    eng = PlaywrightSearchEngine()
    # Try a single engine for reproducibility
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])  # noqa
        context = await browser.new_context(user_agent=eng.user_agent)
        page = await context.new_page()
        await page.goto('https://www.bing.com/search?q=site:github.com%20agent%20framework', wait_until='domcontentloaded', timeout=eng.timeout)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(out), full_page=True)
        print(f"Saved screenshot to {out}")
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())

