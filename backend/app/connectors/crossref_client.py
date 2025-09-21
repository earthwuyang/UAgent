"""Crossref REST client."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from ..models import Author, Paper


LOGGER = logging.getLogger(__name__)


class CrossrefClient:
    """Query Crossref for DOI metadata."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 10) -> List[Paper]:
        params: Dict[str, object] = {
            "query": query,
            "rows": max(1, min(max_results, 50)),
        }

        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": "uagent-science/0.1"}) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

        data = response.json()
        items = data.get("message", {}).get("items", [])
        return [self._to_paper(item) for item in items]

    def _to_paper(self, item: Dict[str, object]) -> Paper:
        title_list = item.get("title") or []
        title = title_list[0] if title_list else "Untitled"

        authors: List[Author] = []
        for author in item.get("author", []) or []:
            given = author.get("given")
            family = author.get("family")
            display_name = " ".join(part for part in [given, family] if part)
            if not display_name:
                continue
            affiliations = [aff.get("name") for aff in author.get("affiliation", []) if aff.get("name")]
            authors.append(Author(name=display_name, orcid=author.get("ORCID"), affiliations=affiliations))

        year = None
        for field in ("published-print", "published-online", "issued"):
            date_field = item.get(field)
            if date_field and date_field.get("date-parts"):
                first_part = date_field["date-parts"][0]
                if first_part:
                    year_candidate = first_part[0]
                    if isinstance(year_candidate, int):
                        year = year_candidate
                        break

        doi = item.get("DOI")
        url = item.get("URL")
        abstract = item.get("abstract")
        if isinstance(abstract, str):
            abstract = self._strip_tags(abstract)

        open_access = bool(item.get("license"))

        return Paper(
            id=f"doi:{doi}" if doi else str(item.get("URL")),
            title=title,
            abstract=abstract,
            authors=authors,
            venue=self._extract_venue(item),
            year=year,
            url=url,
            pdf_url=self._best_pdf_url(item),
            source="crossref",
            doi=doi,
            open_access=open_access,
        )

    @staticmethod
    def _extract_venue(item: Dict[str, object]) -> Optional[str]:
        container_titles = item.get("container-title") or []
        if container_titles:
            return container_titles[0]
        publisher = item.get("publisher")
        return publisher or None

    @staticmethod
    def _best_pdf_url(item: Dict[str, object]) -> Optional[str]:
        links = item.get("link") or []
        for link in links:
            if link.get("content-type") == "application/pdf":
                return link.get("URL")
        return None

    @staticmethod
    def _strip_tags(text: str) -> str:
        import re

        return re.sub(r"<[^>]+>", "", text)


__all__ = ["CrossrefClient"]
