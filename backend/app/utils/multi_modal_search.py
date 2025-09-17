"""
Enhanced Multi-Modal Search Engine
Combines web search, academic papers, code repositories, and documentation
Provides unified search interface with intelligent result ranking
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, quote_plus
import json
import re
from datetime import datetime
import xml.etree.ElementTree as ET

from .searxng_engine import SearxNGSearchEngine
from .playwright_search import PlaywrightSearchEngine


class MultiModalSearchEngine:
    """
    Enhanced multi-modal search engine combining multiple sources
    Integrates web search, academic papers, code repositories, and documentation
    """

    def __init__(self):
        self.searx_engine = SearxNGSearchEngine()
        self.search_engines = {
            'web': WebSearchProvider(),
            'academic': AcademicSearchProvider(),
            'code': CodeSearchProvider(),
            'documentation': DocumentationSearchProvider(),
            'news': NewsSearchProvider()
        }

    async def unified_search(
        self,
        query: str,
        search_types: List[str] = None,
        filters: Dict[str, Any] = None,
        limit: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Perform unified search across multiple sources
        """
        # Validate query parameter
        if not isinstance(query, str):
            print(f"🚨 ERROR: unified_search received non-string query: {type(query)} = {query}")
            if hasattr(query, 'title'):
                print(f"🚨 Object has title attribute: {query.title}")
            if hasattr(query, 'goal'):
                print(f"🚨 Object has goal attribute: {query.goal}")
            raise TypeError(f"Query must be a string, got {type(query)}")

        if search_types is None:
            search_types = ['web', 'code', 'academic']

        filters = filters or {}
        results = {}

        # Execute searches in parallel
        search_tasks = []
        for search_type in search_types:
            if search_type in self.search_engines:
                provider = self.search_engines[search_type]
                task = provider.search(query, filters.get(search_type, {}), limit // len(search_types))
                search_tasks.append((search_type, task))

        # Gather results
        for search_type, task in search_tasks:
            try:
                search_results = await task
                results[search_type] = search_results
            except Exception as e:
                results[search_type] = [{"error": f"Search failed: {str(e)}"}]

        # Post-process and rank results
        results = await self._post_process_results(results, query)

        return results

    async def intelligent_search(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Intelligent search that automatically determines best search types and sources
        """
        context = context or {}

        # Analyze query to determine intent
        search_intent = await self._analyze_search_intent(query, context)

        # Select appropriate search types based on intent
        search_types = await self._select_search_types(search_intent)

        # Create optimized filters
        filters = await self._create_smart_filters(query, search_intent)

        # Execute search
        results = await self.unified_search(query, search_types, filters)

        # Add metadata about search strategy
        results['metadata'] = {
            'intent': search_intent,
            'search_types_used': search_types,
            'filters_applied': filters,
            'timestamp': datetime.now().isoformat()
        }

        return results

    async def _analyze_search_intent(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search query to determine user intent"""
        # Validate query parameter
        if query is None:
            print(f"🚨 ERROR: _analyze_search_intent received None query")
            raise TypeError(f"Query cannot be None")
        if not isinstance(query, str):
            print(f"🚨 ERROR: _analyze_search_intent received non-string query: {type(query)} = {query}")
            raise TypeError(f"Query must be a string, got {type(query)}")

        query_lower = query.lower()
        intent = {
            'type': 'general',
            'domain': 'unknown',
            'specificity': 'medium',
            'temporal': 'any',
            'technical_level': 'medium'
        }

        # Detect search type
        if any(term in query_lower for term in ['paper', 'research', 'study', 'arxiv', 'doi']):
            intent['type'] = 'academic'
        elif any(term in query_lower for term in ['code', 'github', 'function', 'class', 'api']):
            intent['type'] = 'code'
        elif any(term in query_lower for term in ['tutorial', 'documentation', 'how to', 'guide']):
            intent['type'] = 'documentation'
        elif any(term in query_lower for term in ['news', 'latest', 'recent', 'today']):
            intent['type'] = 'news'

        # Detect domain
        programming_terms = ['python', 'javascript', 'java', 'cpp', 'rust', 'go', 'programming']
        if any(term in query_lower for term in programming_terms):
            intent['domain'] = 'programming'
        elif any(term in query_lower for term in ['ai', 'ml', 'machine learning', 'neural', 'deep learning']):
            intent['domain'] = 'ai_ml'
        elif any(term in query_lower for term in ['data', 'database', 'sql', 'analytics']):
            intent['domain'] = 'data'

        # Detect specificity
        if len(query.split()) > 5 or any(char in query for char in ['"', '(', ')']):
            intent['specificity'] = 'high'
        elif len(query.split()) < 3:
            intent['specificity'] = 'low'

        # Detect temporal requirements
        if any(term in query_lower for term in ['2024', '2025', 'recent', 'latest', 'new']):
            intent['temporal'] = 'recent'

        return intent

    async def _select_search_types(self, intent: Dict[str, Any]) -> List[str]:
        """Select appropriate search types based on intent"""
        search_types = []

        if intent['type'] == 'academic':
            search_types = ['academic', 'web']
        elif intent['type'] == 'code':
            search_types = ['code', 'documentation', 'web']
        elif intent['type'] == 'documentation':
            search_types = ['documentation', 'web', 'code']
        elif intent['type'] == 'news':
            search_types = ['news', 'web']
        else:
            # General search
            if intent['domain'] == 'programming':
                search_types = ['web', 'code', 'documentation']
            elif intent['domain'] == 'ai_ml':
                search_types = ['academic', 'web', 'code']
            else:
                search_types = ['web', 'academic']

        return search_types

    async def _create_smart_filters(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligent filters based on query and intent"""
        filters = {}

        # Academic filters
        if 'academic' in await self._select_search_types(intent):
            filters['academic'] = {}
            if intent['temporal'] == 'recent':
                filters['academic']['year_min'] = 2020

        # Code filters
        if 'code' in await self._select_search_types(intent):
            filters['code'] = {}
            # Detect programming language
            lang_keywords = {
                'python': ['python', 'py', 'pip', 'django', 'flask'],
                'javascript': ['javascript', 'js', 'node', 'react', 'vue'],
                'java': ['java', 'spring', 'maven'],
                'cpp': ['cpp', 'c++', 'cmake'],
                'rust': ['rust', 'cargo'],
                'go': ['golang', 'go ']
            }

            query_lower = query.lower() if query else ''
            for lang, keywords in lang_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    filters['code']['language'] = lang
                    break

        # Web filters
        filters['web'] = {}
        if intent['temporal'] == 'recent':
            filters['web']['time_range'] = 'year'

        return filters

    async def _post_process_results(self, results: Dict[str, List[Dict]], query: str) -> Dict[str, List[Dict]]:
        """Post-process and rank search results"""
        processed = {}

        for search_type, result_list in results.items():
            # Filter and rank results
            ranked_results = []
            for result in result_list:
                if 'error' not in result:
                    relevance_score = await self._calculate_relevance(result, query)
                    result['relevance_score'] = relevance_score
                    ranked_results.append(result)

            # Sort by relevance
            ranked_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            processed[search_type] = ranked_results

        return processed

    async def _calculate_relevance(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for search result"""
        if not query:
            return 0.0
        score = 0.0
        query_terms = query.lower().split()

        # Check title relevance
        title = result.get('title', '').lower()
        title_matches = sum(1 for term in query_terms if term in title)
        score += (title_matches / len(query_terms)) * 0.4

        # Check snippet/description relevance
        snippet_text = result.get('snippet', result.get('description', ''))
        snippet = snippet_text.lower() if snippet_text else ''
        snippet_matches = sum(1 for term in query_terms if term in snippet)
        score += (snippet_matches / len(query_terms)) * 0.3

        # Check URL relevance
        url = result.get('url', '').lower()
        url_matches = sum(1 for term in query_terms if term in url)
        score += (url_matches / len(query_terms)) * 0.2

        # Boost for exact phrase matches
        query_phrase = query.lower() if query else ''
        if query_phrase in title:
            score += 0.3
        elif query_phrase in snippet:
            score += 0.2

        # Source-specific boosting
        source = result.get('source', '')
        if source == 'github_repos' and query and any(term in query.lower() for term in ['code', 'library', 'framework']):
            score += 0.1
        elif source == 'arxiv' and query and any(term in query.lower() for term in ['research', 'paper', 'study']):
            score += 0.1

        return min(score, 1.0)


class WebSearchProvider:
    """Enhanced web search provider using Playwright and SearxNG fallback"""

    def __init__(self):
        self.playwright_engine = PlaywrightSearchEngine()
        self.searx = SearxNGSearchEngine()

    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """Search web using async Playwright (primary) with SearxNG fallback"""
        filters = filters or {}

        try:
            # Primary: Use async Playwright search engine
            results = await self.playwright_engine.search(query, max_results=limit)

            # If Playwright succeeds, convert format and return
            if results:
                unified_results = []
                for result in results:
                    unified_results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'url': result.get('link', ''),
                        'engine': result.get('engine', 'playwright'),
                        'score': 1.0,  # Default score
                        'category': 'general'
                    })
                return unified_results

        except Exception as playwright_error:
            print(f"🔍 Playwright search failed: {playwright_error}")

        try:
            # Fallback: Use SearxNG via run_in_executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.searx.search(
                    query=query,
                    max_results=limit,
                    categories=filters.get('category', 'general'),
                    language=filters.get('language', 'en-US')
                )
            )

            # Convert to unified format
            unified_results = []
            for result in results:
                unified_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'snippet': result.get('snippet', ''),
                    'engine': result.get('engine', 'searx'),
                    'source': 'web'
                })

            return unified_results

        except Exception as e:
            # Fallback search
            return [{
                'title': f"Search result for: {query}",
                'url': f"https://example.com/search?q={quote_plus(query)}",
                'snippet': f"Fallback search result for query: {query}",
                'engine': 'fallback',
                'source': 'web'
            }]


class AcademicSearchProvider:
    """Academic paper search provider"""

    def __init__(self):
        self.apis = {
            'arxiv': 'http://export.arxiv.org/api/query',
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1/paper/search'
        }

    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """Search academic papers across multiple sources"""
        filters = filters or {}
        all_results = []

        # Search ArXiv
        arxiv_results = await self._search_arxiv(query, limit // 2)
        all_results.extend(arxiv_results)

        # Search Semantic Scholar
        semantic_results = await self._search_semantic_scholar(query, limit // 2)
        all_results.extend(semantic_results)

        return all_results[:limit]

    async def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search ArXiv papers"""
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': limit,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.apis['arxiv'],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_arxiv_xml(content)
        except Exception as e:
            print(f"ArXiv search error: {e}")

        return []

    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict]:
        """Parse ArXiv XML response"""
        results = []
        try:
            root = ET.fromstring(xml_content)

            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
                published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')

                authors = []
                for author in entry.findall('.//{http://www.w3.org/2005/Atom}author'):
                    name_elem = author.find('.//{http://www.w3.org/2005/Atom}name')
                    if name_elem is not None:
                        authors.append(name_elem.text)

                if title_elem is not None and id_elem is not None:
                    results.append({
                        'title': title_elem.text.strip() if title_elem.text else '',
                        'url': id_elem.text.strip() if id_elem.text else '',
                        'snippet': summary_elem.text.strip() if summary_elem is not None and summary_elem.text else '',
                        'authors': authors,
                        'published': published_elem.text.strip() if published_elem is not None and published_elem.text else '',
                        'source': 'arxiv'
                    })
        except ET.ParseError as e:
            print(f"Error parsing ArXiv XML: {e}")

        return results

    async def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar papers"""
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,abstract,authors,year,url,venue,citationCount'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.apis['semantic_scholar'],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []

                        for paper in data.get('data', []):
                            authors = [author.get('name', '') for author in paper.get('authors', [])]

                            results.append({
                                'title': paper.get('title', ''),
                                'url': paper.get('url', ''),
                                'snippet': paper.get('abstract', ''),
                                'authors': authors,
                                'published': str(paper.get('year', '')),
                                'venue': paper.get('venue', ''),
                                'citations': paper.get('citationCount', 0),
                                'source': 'semantic_scholar'
                            })

                        return results
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")

        return []


class CodeSearchProvider:
    """Code search provider for GitHub and other repositories"""

    def __init__(self):
        self.github_api = "https://api.github.com/search"

    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """Search code repositories"""
        filters = filters or {}
        results = []

        # Search GitHub repositories
        repo_results = await self._search_github_repos(query, filters, limit // 2)
        results.extend(repo_results)

        # Search GitHub code
        code_results = await self._search_github_code(query, filters, limit // 2)
        results.extend(code_results)

        return results[:limit]

    async def _search_github_repos(self, query: str, filters: Dict[str, Any], limit: int) -> List[Dict]:
        """Search GitHub repositories"""
        search_query = query

        # Add language filter if specified
        if 'language' in filters:
            search_query += f" language:{filters['language']}"

        # Add additional filters
        if 'stars' in filters:
            search_query += f" stars:>{filters['stars']}"

        params = {
            'q': search_query,
            'sort': filters.get('sort', 'stars'),
            'order': 'desc',
            'per_page': limit
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.github_api}/repositories",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []

                        for repo in data.get('items', []):
                            results.append({
                                'title': repo.get('full_name', ''),
                                'url': repo.get('html_url', ''),
                                'snippet': repo.get('description', ''),
                                'language': repo.get('language', ''),
                                'stars': repo.get('stargazers_count', 0),
                                'forks': repo.get('forks_count', 0),
                                'updated': repo.get('updated_at', ''),
                                'source': 'github_repos'
                            })

                        return results
        except Exception as e:
            print(f"GitHub repo search error: {e}")

        return []

    async def _search_github_code(self, query: str, filters: Dict[str, Any], limit: int) -> List[Dict]:
        """Search GitHub code files"""
        search_query = query

        # Add file extension filter if specified
        if 'extension' in filters:
            search_query += f" extension:{filters['extension']}"

        # Add language filter
        if 'language' in filters:
            search_query += f" language:{filters['language']}"

        params = {
            'q': search_query,
            'sort': 'indexed',
            'order': 'desc',
            'per_page': limit
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.github_api}/code",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []

                        for item in data.get('items', []):
                            repo = item.get('repository', {})
                            results.append({
                                'title': item.get('name', ''),
                                'url': item.get('html_url', ''),
                                'snippet': item.get('text_matches', [{}])[0].get('fragment', '') if item.get('text_matches') else '',
                                'path': item.get('path', ''),
                                'repository': repo.get('full_name', ''),
                                'language': repo.get('language', ''),
                                'source': 'github_code'
                            })

                        return results
        except Exception as e:
            print(f"GitHub code search error: {e}")

        return []


class DocumentationSearchProvider:
    """Documentation search provider"""

    def __init__(self):
        self.doc_sources = {
            'stackoverflow': 'https://api.stackexchange.com/2.3/search'
        }

    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """Search documentation and Q&A sites"""
        filters = filters or {}
        results = []

        # Search Stack Overflow
        so_results = await self._search_stackoverflow(query, limit)
        results.extend(so_results)

        return results[:limit]

    async def _search_stackoverflow(self, query: str, limit: int) -> List[Dict]:
        """Search Stack Overflow questions"""
        params = {
            'intitle': query,
            'site': 'stackoverflow',
            'sort': 'relevance',
            'pagesize': limit
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.doc_sources['stackoverflow'],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []

                        for item in data.get('items', []):
                            results.append({
                                'title': item.get('title', ''),
                                'url': item.get('link', ''),
                                'snippet': f"Score: {item.get('score', 0)}, Views: {item.get('view_count', 0)}",
                                'tags': item.get('tags', []),
                                'score': item.get('score', 0),
                                'views': item.get('view_count', 0),
                                'answers': item.get('answer_count', 0),
                                'is_answered': item.get('is_answered', False),
                                'source': 'stackoverflow'
                            })

                        return results
        except Exception as e:
            print(f"Stack Overflow search error: {e}")

        return []


class NewsSearchProvider:
    """News and current events search provider"""

    async def search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """Search for news and current events"""
        # For now, return placeholder results
        # In production, would integrate with news APIs
        return [{
            'title': f"News result for: {query}",
            'url': f"https://news.example.com/search?q={quote_plus(query)}",
            'snippet': f"Recent news about {query}",
            'published': datetime.now().isoformat(),
            'source': 'news_placeholder'
        }]