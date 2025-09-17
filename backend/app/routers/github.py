from fastapi import APIRouter, Query
from typing import List, Dict, Any

from ..utils.playwright_search import PlaywrightSearchEngine


router = APIRouter(prefix="/github", tags=["github"])
engine = PlaywrightSearchEngine()


def _ensure_site_filter(q: str) -> str:
    q = q.strip()
    if "site:github.com" not in q:
        q = f"site:github.com {q}"
    return q


@router.get("/search")
async def github_search(q: str = Query("", description="Free-text GitHub query; site filter auto-applied"), max_results: int = 10) -> List[Dict[str, Any]]:
    if not q.strip():
        return []
    return await engine.search(_ensure_site_filter(q), max_results=max_results)

