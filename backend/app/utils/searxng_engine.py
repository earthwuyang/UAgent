"""
Minimal SearxNG search client adapted for uagent.
Supports public or self-hosted instances and works well for GitHub queries:
  q = "site:github.com topic language:python"
"""
from __future__ import annotations

import os
import json
import requests
from typing import List, Dict, Any


class SearxNGSearchEngine:
    def __init__(self) -> None:
        self.instances = [
            os.getenv("SEARXNG_INSTANCE_URL") or "http://localhost:8080",
            "https://searx.be",
            "https://searx.tiekoetter.com",
            "https://search.sapti.me",
            "https://searx.prvcy.eu",
            "https://search.disroot.org",
        ]
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }

    def _first_working(self) -> str:
        for base in self.instances:
            try:
                r = requests.get(f"{base}/search", params={"q": "test", "format": "json"}, headers=self.headers, timeout=6)
                if r.ok and "results" in r.json():
                    return base
            except Exception:
                continue
        # last resort: first
        return self.instances[0]

    def search(self, query: str, max_results: int = 10, categories: str = "general", language: str = "en-US") -> List[Dict[str, Any]]:
        base = self._first_working()
        try:
            r = requests.get(
                f"{base}/search",
                params={"q": query, "format": "json", "categories": categories, "language": language},
                headers=self.headers,
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []
        results = []
        for item in (data.get("results") or [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "snippet": item.get("content"),
                    "link": item.get("url"),
                    "engine": item.get("engine"),
                    "category": item.get("category", categories),
                }
            )
        return results

