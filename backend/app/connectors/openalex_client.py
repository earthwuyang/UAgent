"""OpenAlex API client for scholarly metadata."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from ..models import Author, Paper


LOGGER = logging.getLogger(__name__)


class OpenAlexClient:
    """Query OpenAlex for scholarly works."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, timeout: float = 12.0, mailto: Optional[str] = None):
        self.timeout = timeout
        self.mailto = mailto

    async def search(self, query: str, max_results: int = 10) -> List[Paper]:
        params: Dict[str, object] = {
            "search": query,
            "per-page": max(1, min(max_results, 50)),
        }
        if self.mailto:
            params["mailto"] = self.mailto

        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": "uagent-science/0.1"}) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

        payload = response.json()
        results = payload.get("results", [])
        return [self._to_paper(item) for item in results]

    def _to_paper(self, item: Dict[str, object]) -> Paper:
        identifier = str(item.get("id", "")).split("/")[-1]
        abstract = self._abstract_from_inverted_index(item.get("abstract_inverted_index"))

        authors: List[Author] = []
        for authorship in item.get("authorships", []) or []:
            author_obj = authorship.get("author") or {}
            name = author_obj.get("display_name") or author_obj.get("display_name_alternatives", [""])[0]
            if name:
                orcid = author_obj.get("orcid")
                affiliations = [aff.get("display_name") for aff in authorship.get("institutions", []) if aff.get("display_name")]
                authors.append(Author(name=name, orcid=orcid, affiliations=affiliations))

        primary_location = item.get("primary_location") or {}
        host_venue = item.get("host_venue") or {}
        title = item.get("display_name") or item.get("title") or "Untitled"
        url = primary_location.get("landing_page_url") or item.get("id")
        pdf_url = primary_location.get("pdf_url")
        venue = host_venue.get("display_name") or primary_location.get("source")
        year = item.get("publication_year")
        if isinstance(year, str):
            try:
                year = int(year)
            except ValueError:
                year = None

        open_access = False
        oa = item.get("open_access") or {}
        if isinstance(oa, dict):
            open_access = bool(oa.get("is_oa"))

        doi = item.get("doi")
        if isinstance(doi, str) and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        concepts = item.get("concepts") or []
        tags = [concept.get("display_name") for concept in concepts if concept.get("display_name")]

        return Paper(
            id=f"openalex:{identifier}" if identifier else str(item.get("id")),
            title=str(title),
            abstract=abstract,
            authors=authors,
            venue=venue,
            year=year,
            url=url,
            pdf_url=pdf_url,
            source="openalex",
            doi=doi,
            open_access=open_access,
            tags=tags[:5],
        )

    @staticmethod
    def _abstract_from_inverted_index(index: Optional[Dict[str, List[int]]]) -> Optional[str]:
        if not index:
            return None

        positions: Dict[int, str] = {}
        for word, occurrences in index.items():
            for pos in occurrences:
                positions[int(pos)] = word
        ordered = [positions[pos] for pos in sorted(positions.keys())]
        return " ".join(ordered) if ordered else None


__all__ = ["OpenAlexClient"]
