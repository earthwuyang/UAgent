"""PubMed client using NCBI E-utilities."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import httpx

from ..models import Author, Paper


LOGGER = logging.getLogger(__name__)


class PubMedClient:
    """Query PubMed for biomedical literature."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, timeout: float = 12.0, api_key: Optional[str] = None, email: Optional[str] = None):
        self.timeout = timeout
        self.api_key = api_key
        self.email = email

    async def search(self, query: str, max_results: int = 10) -> List[Paper]:
        ids = await self._esearch(query, max_results)
        if not ids:
            return []

        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": "uagent-science/0.1"}) as client:
            summaries = await self._esummary(client, ids)
            abstracts = await self._efetch_abstracts(client, ids)

        results: List[Paper] = []
        for pmid in ids:
            summary = summaries.get(pmid, {})
            if not summary:
                continue

            title = summary.get("title") or summary.get("sorttitle") or "Untitled"
            journal = summary.get("fulljournalname") or summary.get("source")
            pubdate = summary.get("pubdate", "")
            year = self._parse_year(pubdate)

            authors: List[Author] = []
            for author in summary.get("authors", []) or []:
                display_name = author.get("name")
                if not display_name:
                    last_name = author.get("lastname")
                    initials = author.get("initials")
                    display_name = " ".join(part for part in [last_name, initials] if part)
                if not display_name:
                    continue

                aff_value = author.get("affiliation")
                affiliations = [aff_value] if isinstance(aff_value, str) and aff_value else []
                authors.append(Author(name=display_name, affiliations=affiliations))

            doi = None
            for article_id in summary.get("articleids", []) or []:
                if article_id.get("idtype") == "doi":
                    doi = article_id.get("value")

            results.append(
                Paper(
                    id=f"pubmed:{pmid}",
                    title=title,
                    abstract=abstracts.get(pmid),
                    authors=authors,
                    venue=journal,
                    year=year,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    pdf_url=None,
                    source="pubmed",
                    doi=doi,
                    open_access=False,
                )
            )

        return results

    async def _esearch(self, query: str, max_results: int) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max(1, min(max_results, 50)),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": "uagent-science/0.1"}) as client:
            response = await client.get(self.ESEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

        try:
            return data["esearchresult"]["idlist"]
        except KeyError:
            LOGGER.warning("Unexpected PubMed esearch response structure: %s", data)
            return []

    async def _esummary(self, client: httpx.AsyncClient, ids: List[str]) -> Dict[str, Dict[str, object]]:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        response = await client.get(self.ESUMMARY_URL, params=params)
        response.raise_for_status()

        data = response.json()
        return {pmid: data.get("result", {}).get(pmid, {}) for pmid in ids}

    async def _efetch_abstracts(self, client: httpx.AsyncClient, ids: List[str]) -> Dict[str, Optional[str]]:
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        response = await client.get(self.EFETCH_URL, params=params)
        response.raise_for_status()

        abstracts: Dict[str, Optional[str]] = {pmid: None for pmid in ids}
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            LOGGER.exception("Failed to parse PubMed efetch response")
            return abstracts

        for article in root.findall("PubmedArticle"):
            pmid_node = article.find("MedlineCitation/PMID")
            if pmid_node is None:
                continue
            pmid = pmid_node.text
            abstract_texts: List[str] = []
            for abstract_node in article.findall("MedlineCitation/Article/Abstract/AbstractText"):
                label = abstract_node.attrib.get("Label")
                text_value = (abstract_node.text or "").strip()
                if label:
                    abstract_texts.append(f"{label}: {text_value}")
                else:
                    abstract_texts.append(text_value)
            if pmid and abstract_texts:
                abstracts[pmid] = "\n".join(abstract_texts)

        return abstracts

    @staticmethod
    def _parse_year(pubdate: str) -> Optional[int]:
        if not pubdate:
            return None
        try:
            return int(pubdate[:4])
        except ValueError:
            return None


__all__ = ["PubMedClient"]
