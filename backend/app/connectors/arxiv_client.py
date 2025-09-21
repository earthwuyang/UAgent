"""Lightweight client for the arXiv API."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import List

import httpx

from ..models import Author, Paper


LOGGER = logging.getLogger(__name__)


class ArxivClient:
    """Query arXiv for pre-print metadata."""

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout

    async def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search arXiv and return normalized paper metadata."""

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max(1, min(max_results, 50)),
        }

        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": "uagent-science/0.1"}) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

        return self._parse_feed(response.text)

    def _parse_feed(self, feed_xml: str) -> List[Paper]:
        """Parse Atom feed returned by arXiv."""

        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(feed_xml)
        papers: List[Paper] = []

        for entry in root.findall("a:entry", ns):
            raw_id = entry.findtext("a:id", default="", namespaces=ns)
            identifier = raw_id.split("/abs/")[-1] if raw_id else ""

            title = self._clean_text(entry.findtext("a:title", default="", namespaces=ns))
            abstract = self._clean_text(entry.findtext("a:summary", default="", namespaces=ns))
            published = entry.findtext("a:published", default="", namespaces=ns)
            year = None
            if published:
                try:
                    year = int(published[:4])
                except ValueError:
                    year = None

            authors: List[Author] = []
            for author in entry.findall("a:author", ns):
                name = self._clean_text(author.findtext("a:name", default="", namespaces=ns))
                if name:
                    authors.append(Author(name=name))

            url = raw_id or None
            pdf_url = None
            for link in entry.findall("a:link", ns):
                if link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href")
                if link.attrib.get("rel") == "alternate":
                    url = link.attrib.get("href", url)

            papers.append(
                Paper(
                    id=f"arxiv:{identifier}" if identifier else raw_id,
                    title=title,
                    abstract=abstract or None,
                    authors=authors,
                    venue="arXiv",
                    year=year,
                    url=url,
                    pdf_url=pdf_url,
                    source="arxiv",
                    open_access=True,
                )
            )

        return papers

    @staticmethod
    def _clean_text(value: str) -> str:
        return " ".join(value.split()) if value else ""


__all__ = ["ArxivClient"]
