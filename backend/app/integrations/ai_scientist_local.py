"""
AI Scientist Local Integration - Direct access to local AI Scientist functions with Playwright search
"""

import os
import sys
import asyncio
import subprocess
import urllib.parse
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright

# Add local AI Scientist to path
AI_SCIENTIST_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ai_scientist_local"))
if AI_SCIENTIST_LOCAL_PATH not in sys.path:
    sys.path.insert(0, AI_SCIENTIST_LOCAL_PATH)

# Import AI Scientist functions
from ai_scientist.llm import get_response_from_llm, create_client
from ai_scientist.perform_ideation_temp_free import generate_temp_free_idea


class LocalAIScientist:
    """Direct interface to local AI Scientist functionality with Playwright search"""

    def __init__(self):
        self.timeout = 30000  # 30 seconds
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async def generate_idea(self, research_goal: str, context: str = "") -> Dict[str, Any]:
        """Generate research idea using AI Scientist ideation"""
        try:
            # Create temporary workshop description from research goal and context
            workshop_description = f"""
Research Goal: {research_goal}

Context: {context}

This is a research workshop focused on generating novel ideas related to the above goal and context.
"""

            # Create client and temporary file
            client = create_client("gpt-4o-mini")
            temp_filename = "/tmp/temp_idea.json"

            ideas = generate_temp_free_idea(
                idea_fname=temp_filename,
                client=client,
                model="gpt-4o-mini",
                workshop_description=workshop_description,
                max_num_generations=3,  # Reduced for faster response
                num_reflections=2,      # Reduced for faster response
                reload_ideas=False      # Don't reload existing ideas
            )
            return {
                "success": True,
                "idea": ideas[0] if ideas else {"title": research_goal, "abstract": "Generated idea placeholder"},
                "source": "ai_scientist_local"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": "ai_scientist_local"
            }

    async def search_papers(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search papers using Playwright Bing search with xvfb"""
        try:
            # Add academic terms to improve paper search results
            academic_query = f"{query} research paper academic study"
            results = await self._run_playwright_with_xvfb(self._bing_search_core, academic_query, limit)

            # Filter results to prioritize academic sources
            academic_results = []
            for result in results:
                link = result.get('link', '') or ''
                title = result.get('title', '') or ''
                snippet = result.get('snippet', '') or ''

                link_lower = link.lower()
                title_lower = title.lower()

                # Prioritize academic sources
                if any(domain in link_lower for domain in ['arxiv.org', 'scholar.google', 'ieee.org', 'acm.org', 'springer.com', 'nature.com', 'science.org', 'pubmed.ncbi', 'researchgate.net']):
                    academic_results.append({
                        'title': title,
                        'abstract': snippet,
                        'link': link,
                        'source': 'playwright_bing_academic'
                    })
                elif any(keyword in title_lower for keyword in ['paper', 'study', 'research', 'analysis', 'survey', 'review']):
                    academic_results.append({
                        'title': title,
                        'abstract': snippet,
                        'link': link,
                        'source': 'playwright_bing_academic'
                    })

            return {
                "success": True,
                "papers": academic_results[:limit],
                "source": "playwright_bing_academic"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source": "playwright_bing_academic"
            }

    async def get_llm_response(self, prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """Get LLM response using AI Scientist LLM interface"""
        try:
            client = create_client(model)
            response = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=model
            )
            return {
                "success": True,
                "response": response,
                "model": model,
                "source": "ai_scientist_llm"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "source": "ai_scientist_llm"
            }


    async def _run_playwright_with_xvfb(self, playwright_func, *args, **kwargs):
        """
        Run Playwright function using xvfb-run for virtual display
        """
        try:
            # Check if DISPLAY is already set (running in GUI environment)
            if os.environ.get('DISPLAY'):
                print("🖥️  Using existing DISPLAY")
                return await playwright_func(*args, **kwargs)
            else:
                print("🖼️  Using xvfb-run for virtual display")
                # Set up virtual display environment
                xvfb_env = os.environ.copy()
                xvfb_env['DISPLAY'] = ':99'

                # Start xvfb in background if not already running
                try:
                    subprocess.run(['pgrep', '-f', 'Xvfb :99'],
                                 check=True, capture_output=True)
                    print("✅ Xvfb already running on :99")
                except subprocess.CalledProcessError:
                    print("🚀 Starting Xvfb on :99")
                    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    await asyncio.sleep(2)  # Wait for Xvfb to start

                # Set DISPLAY for this process
                original_display = os.environ.get('DISPLAY')
                os.environ['DISPLAY'] = ':99'

                try:
                    result = await playwright_func(*args, **kwargs)
                    return result
                finally:
                    # Restore original DISPLAY
                    if original_display:
                        os.environ['DISPLAY'] = original_display
                    else:
                        os.environ.pop('DISPLAY', None)

        except Exception as e:
            print(f"❌ Error with xvfb-run: {str(e)}")
            # Fallback to regular playwright
            print("🔄 Falling back to regular Playwright")
            return await playwright_func(*args, **kwargs)

    async def _bing_search_core(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Core Bing search implementation with advanced anti-detection (adapted from RepoMaster)
        """
        results = []

        async with async_playwright() as p:
            # Advanced browser setup with maximum stealth
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-gpu',
                    '--no-first-run',
                    '--no-default-browser-check',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-background-networking',
                    '--no-zygote',
                    '--disable-ipc-flooding-protection'
                ]
            )

            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1'
                }
            )

            page = await context.new_page()

            # Add comprehensive stealth measures
            await page.add_init_script("""
                // Remove all automation indicators
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                delete navigator.webdriver;

                // Mock chrome object with realistic data
                window.chrome = {
                    runtime: {},
                    loadTimes: function() {
                        return {
                            requestTime: Date.now() / 1000 - Math.random(),
                            startLoadTime: Date.now() / 1000 - Math.random(),
                            commitLoadTime: Date.now() / 1000 - Math.random(),
                            finishDocumentLoadTime: Date.now() / 1000 - Math.random(),
                            finishLoadTime: Date.now() / 1000 - Math.random(),
                            firstPaintTime: Date.now() / 1000 - Math.random(),
                            navigationType: 'navigate'
                        };
                    },
                    csi: function() { return {}; },
                    app: { isInstalled: false }
                };

                // Mock realistic plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
                        {name: 'Chromium PDF Plugin', filename: 'chromium-pdf-viewer'},
                        {name: 'Microsoft Edge PDF Plugin', filename: 'edge-pdf-viewer'},
                        {name: 'WebKit built-in PDF', filename: 'webkit-pdf-viewer'},
                        {name: 'Native Client', filename: 'internal-nacl-plugin'}
                    ],
                });

                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
            """)

            try:
                # Human-like navigation behavior
                print(f"🔍 Navigating to Bing with human-like behavior...")

                # First visit Bing homepage to establish session
                await page.goto("https://www.bing.com", wait_until='domcontentloaded', timeout=self.timeout)

                # Random delay to simulate human behavior
                import random
                delay = random.uniform(1.0, 3.0)
                await page.wait_for_timeout(int(delay * 1000))

                # Simulate some mouse movement
                await page.mouse.move(random.randint(100, 800), random.randint(100, 600))
                await page.wait_for_timeout(random.randint(200, 500))

                # Now navigate to search results
                bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
                print(f"🔍 Searching: {bing_url}")

                await page.goto(bing_url, wait_until='domcontentloaded', timeout=self.timeout)

                # Wait for page to fully load with human-like delay
                await page.wait_for_timeout(random.randint(2000, 4000))

                # Wait for results with flexible selectors
                result_selectors = [
                    '.b_algo',
                    '[data-bm]',
                    '.b_results .b_algo',
                    'li.b_algo'
                ]

                result_elements = []
                for selector in result_selectors:
                    try:
                        await page.wait_for_selector(selector, timeout=8000)
                        elements = await page.locator(selector).all()
                        if elements:
                            result_elements = elements
                            print(f"✅ Found Bing elements using selector: {selector}")
                            break
                    except:
                        print(f"⚠️  Bing selector '{selector}' failed, trying next...")
                        continue

                print(f"📊 Found {len(result_elements)} Bing result elements")

                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        # Extract title - Bing specific selectors
                        title = ""
                        title_selectors = ['h2 a', 'h2', '.b_title a', '.b_title']
                        for sel in title_selectors:
                            try:
                                title_elem = element.locator(sel).first
                                if await title_elem.count() > 0:
                                    title = await title_elem.text_content() or ""
                                    title = title.strip()
                                    # Clean up title if it has URL appended
                                    if "http" in title:
                                        title = title.split("http")[0].strip()
                                    if title and len(title) > 3:
                                        break
                            except:
                                continue

                        # Extract link
                        link = ""
                        link_selectors = ['h2 a', '.b_title a', 'a[href]']
                        for sel in link_selectors:
                            try:
                                link_elem = element.locator(sel).first
                                if await link_elem.count() > 0:
                                    link = await link_elem.get_attribute("href") or ""
                                    if link and link.startswith(("http://", "https://")):
                                        break
                            except:
                                continue

                        # Extract snippet
                        snippet = ""
                        snippet_selectors = ['.b_caption p', '.b_caption', 'p', '.b_snippet']
                        for sel in snippet_selectors:
                            try:
                                snippet_elem = element.locator(sel).first
                                if await snippet_elem.count() > 0:
                                    snippet = await snippet_elem.text_content() or ""
                                    snippet = snippet.strip()
                                    if len(snippet) > 15:
                                        break
                            except:
                                continue

                        # Add result if we have meaningful data
                        if title and link and link.startswith(("http://", "https://")):
                            results.append({
                                "title": title,
                                "snippet": snippet or "No snippet available",
                                "link": link,
                                "engine": "playwright_bing"
                            })
                            print(f"✅ Extracted Bing result {i+1}: {title[:50]}...")

                    except Exception as e:
                        print(f"❌ Error extracting Bing result {i}: {str(e)}")
                        continue

            except Exception as e:
                print(f"❌ Error during Bing search: {str(e)}")

            finally:
                await browser.close()

        print(f"📊 Successfully extracted {len(results)} Bing search results")
        return results


# Global instance
local_ai_scientist = LocalAIScientist()