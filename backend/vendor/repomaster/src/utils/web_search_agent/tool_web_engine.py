import os
import json
import asyncio
import logging
import subprocess
import time
import aiohttp
import requests
from bs4 import BeautifulSoup
import re
from contextlib import contextmanager
from urllib.parse import quote_plus

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from src.utils.tool_retriever_embed import WebRetriever, EmbeddingMatcher
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional, Annotated
import tiktoken

logger = logging.getLogger(__name__)


@contextmanager
def ensure_virtual_display():
    """Start a lightweight virtual display when no DISPLAY is available."""

    original_display = os.environ.get("DISPLAY")
    if original_display:
        yield
        return

    process = None
    display_name = os.environ.get("UAGENT_XVFB_DISPLAY", ":99")
    try:
        process = subprocess.Popen(
            [
                "Xvfb",
                display_name,
                "-screen",
                "0",
                "1280x720x24",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        os.environ["DISPLAY"] = display_name
    except FileNotFoundError:
        logger.warning("Xvfb not found; continuing without a virtual display")
    except Exception as exc:  # pragma: no cover - defensive safeguard
        logger.warning("Failed to start Xvfb (%s); continuing without a virtual display", exc)

    try:
        yield
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=2)
        if original_display is not None:
            os.environ["DISPLAY"] = original_display
        else:
            os.environ.pop("DISPLAY", None)

class WebBrowser:
    def __init__(self, max_browser_length=20000):
        self.search_engine = SerperSearchEngine()
        self.max_browser_length = max_browser_length

    async def searching(self, query: Annotated[str, "Query content to search for"]) -> str:
        """
        Use search engine to query information and return results
        """
        try:
            return await self.search_engine.engine_search(query, engine='bing', search_num=10, web_parse=False)
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return f"Error searching: {str(e)}"

    async def browsing(self, query: Annotated[str, "Query string for content filtering"], url: Annotated[str, "URL of the webpage to browse"]) -> str:
        """
        Browse specific URL's detailed content and extract relevant information
        """
        try:
            content = await self.browsing_url(url)
            output_content = []
            if len(content)>self.max_browser_length:
                return json.dumps({'Input Query': query, 'Search URL': url, 'Search Result': content[:self.max_browser_length]}, ensure_ascii=False)
            else:
                return json.dumps({'Input Query': query, 'Search URL': url, 'Search Result': content}, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error browsing URL {url}: {str(e)}")
            return json.dumps({'Input Query': query, 'Search URL': url, 'Search Result': 'Error browsing URL'}, ensure_ascii=False)
    
    async def parallel_browsing(
        self, 
        query: Annotated[str, "Query string for content filtering"],
        urls: Annotated[List[str], "List of webpage URLs to browse in parallel"],
        max_parallel: Annotated[int, "Maximum number of parallel processing"] = 3
    ) -> str:
        """
        Browse multiple URLs' detailed content in parallel and extract relevant information
        return: Dictionary list containing content of each URL
        """
        results = []
        
        try:
            # Split URL list into multiple batches, each batch contains at most max_parallel URLs
            for i in range(0, len(urls), max_parallel):
                batch = urls[i:i + max_parallel]
                tasks = [self.browsing(query, url) for url in batch]
                
                # Execute all tasks in current batch in parallel
                batch_results = await asyncio.gather(*tasks)
                
                # Parse JSON strings to dictionary objects
                parsed_results = [json.loads(result) for result in batch_results]
                results.extend(parsed_results)
            
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            print(f"Error parallel browsing: {str(e)}")
            return f"Error parallel browsing: {str(e)}"

    async def test_browsing(self, query: Annotated[str, "Query string for content filtering"], url: Annotated[str, "Webpage URL to browse"]):
        """
        Browse specific URL's detailed content and extract relevant information
        """
        content = await self.browsing_url(url)
        
        content_clean = await self.search_engine._clean_content(content)
        
        return content, content_clean


    async def browsing_url(self, url):
        if "r.jina.ai" not in url:
            url = "https://r.jina.ai/"+url

        if os.getenv("JINA_API_KEY"):
            headers = {
                'Authorization': "Bearer "+os.getenv("JINA_API_KEY",''),
                'X-Engine': 'direct',
                'X-Return-Format': 'markdown',
                "X-Timeout": "10"       
            }
        else:
            headers = None
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                content = await response.read()

        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        
        content = await self.search_engine._clean_content(content)

        return content

class SerperSearchEngine:
    """Search implementation backed by Playwright-powered Bing results."""

    def __init__(self, chunk_size=4000, chunk_overlap=400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # self.retriever = WebRetriever(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def _bing_playwright_search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Use Playwright to fetch Bing search results without relying on external APIs."""

        results: List[Dict[str, str]] = []
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}&setlang=en"

        with ensure_virtual_display():
            try:
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch(
                        headless=True,
                        args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
                    )
                    context = await browser.new_context()
                    page = await context.new_page()
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)

                    try:
                        await page.wait_for_selector("li.b_algo", timeout=15000)
                    except PlaywrightTimeoutError:
                        logger.warning("Bing search timed out for query '%s'", query)

                    items = await page.query_selector_all("li.b_algo")
                    for item in items:
                        title_el = await item.query_selector("h2 a")
                        link = await title_el.get_attribute("href") if title_el else None
                        title_text = await title_el.inner_text() if title_el else None

                        snippet_el = await item.query_selector(".b_caption p")
                        if snippet_el is None:
                            snippet_el = await item.query_selector(".b_paractl")
                        snippet_text = await snippet_el.inner_text() if snippet_el else ""

                        if title_text and link:
                            results.append({
                                "title": title_text.strip(),
                                "snippet": snippet_text.strip(),
                                "link": link,
                            })
                        if len(results) >= max_results:
                            break

                    await context.close()
                    await browser.close()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Playwright search failed for query '%s': %s", query, exc)

        if not results:
            logger.warning("No search results returned for query '%s'", query)

        return results

    def _scrape_search_results(self, url: Annotated[str, "The search URL"], engine: Annotated[str, "The search engine ('bing' or 'yahoo')"]) -> List[Dict[str, str]]:
        """
        Scrape search results from Bing or Yahoo.

        Args:
            url: The search URL.
            engine: The search engine ('bing' or 'yahoo').

        Returns:
            A list of search results, each containing title, snippet, and link.
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        if engine == 'bing':
            for item in soup.select('#b_results .b_algo'):
                title = item.select_one('h2 a')
                snippet = item.select_one('.b_caption p')
                if title and snippet:
                    results.append({
                        'title': title.text,
                        'snippet': snippet.text,
                        'link': title['href']
                    })
        elif engine == 'yahoo':
            for item in soup.select('div.algo'):
                title = item.select_one('h3 a')
                snippet = item.select_one('.compText')
                if title and snippet:
                    results.append({
                        'title': title.text,
                        'snippet': snippet.text,
                        'link': title['href']
                    })

        return results  # Limit to first 5 results

    def bing_search(self, query: Annotated[str, "The search query"]) -> List[Dict[str, str]]:
        """
        Perform a Bing search by scraping the results.

        Args:
            query: The search query.

        Returns:
            A list of search results, each containing title, snippet, and link.
        """
        url = f"https://www.bing.com/search?q={query}"
        return self._scrape_search_results(url, 'bing')

    def yahoo_search(self, query: Annotated[str, "The search query"]) -> List[Dict[str, str]]:
        """
        Perform a Yahoo search by scraping the results.

        Args:
            query: The search query.

        Returns:
            A list of search results, each containing title, snippet, and link.
        """
        url = f"https://search.yahoo.com/search?p={query}"
        return self._scrape_search_results(url, 'yahoo')
    
    async def _clean_content(self, content: str) -> str:
        # Remove URLs
        content = re.sub(r'http[s]?://\S+', '', content)
        
        # Remove Markdown links
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)

        # Remove image markers (keep alt text)
        content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', content)

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)        
        
        # Remove navigation lists
        content = re.sub(r'^\s*[-*]\s+(Home|About|Contact|Menu|Search|Privacy Policy|Terms of Service)\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove common footer information
        content = re.sub(r'Copyright © \d{4}.*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'All rights reserved\.?', '', content, flags=re.IGNORECASE)
        
        # Remove social media related text
        content = re.sub(r'(Follow|Like|Share|Subscribe).*(Facebook|Twitter|Instagram|LinkedIn|YouTube).*', '', content, flags=re.IGNORECASE)
        
        # Remove empty lines and extra whitespace
        content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove very short lines (likely navigation items)
        content = '\n'.join(line for line in content.split('\n') if len(line.split()) > 2)
        
        return content.strip()

    async def _parse_content_async(self, res):
        try:
            content = await WebBrowser().browsing(query='', url=res['link'])
            # Convert bytes to string if content is in bytes format
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            # print(f"content: {content}", flush=True)
            res['content'] = await self._clean_content(content)
        except Exception as e:
            print(f"Error parsing content for {res['link']}: {str(e)}")
            res['content'] = ""
        return res

    async def _enrich_results_async(self, results):
        tasks = [self._parse_content_async(res) for res in results]
        return await asyncio.gather(*tasks)
    
    async def engine_search(self, query, engine='bing', search_num=10, web_parse=True, url_filter=None):
        engine = (engine or 'bing').lower()

        try:
            results = await self._bing_playwright_search(query, max_results=search_num)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Primary Bing search failed for query '%s': %s", query, exc)
            results = []

        if url_filter:
            results = [res for res in results if res.get('link') not in url_filter]

        results = results[:min(search_num, len(results))]

        if web_parse and results:
            results = await self._enrich_results_async(results)

        return json.dumps(results, ensure_ascii=False)

    async def search(self, query: Annotated[str, "The search query"], engine: Annotated[str, "The search engine to use"] = 'google') -> List[Dict[str, str]]:
        """
        Perform a search using the specified engine and enrich results with web content.

        Args:
            query: The search query.
            engine: The search engine to use ('google', 'bing', or 'yahoo').

        Returns:
            A list of search results, each containing title, snippet, link, and content.
        """
        return await self.engine_search(query, engine)
