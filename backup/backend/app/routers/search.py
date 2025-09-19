from fastapi import APIRouter, Query
from typing import List, Dict, Any
import asyncio

from ..utils.playwright_search import PlaywrightSearchEngine


router = APIRouter(tags=["search"])
engine = PlaywrightSearchEngine()


@router.get("/search")
async def search(q: str = Query("", description="Query string. Use 'site:github.com' to constrain to GitHub."), max_results: int = 10) -> List[Dict[str, Any]]:
    query = q.strip()
    if not query:
        return []
    return await engine.search(query=query, max_results=max_results)
