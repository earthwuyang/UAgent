"""Lightweight adapter around RepoMaster functionality.

This provides a thin facade that the orchestrator and routers can call without
requiring the full RepoMaster project to be vendored. If RepoMaster is not
available, we degrade gracefully by returning heuristic summaries based on
Playwright search results and local repository inspection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

from ..utils.playwright_search import PlaywrightSearchEngine


class AnalysisDepth(str, Enum):
    shallow = "shallow"
    semantic = "semantic"
    deep = "deep"


@dataclass
class RepoSummary:
    repo_id: str
    source: str
    path: str
    files_analyzed: int
    languages: Dict[str, int]
    top_files: List[str]
    metadata: Dict[str, Any]


class RepoMaster:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.search_engine = PlaywrightSearchEngine()
        self.cache_dir = cache_dir or Path(__file__).resolve().parents[2] / "data" / "repomaster_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def deep_search(self, task_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return candidate GitHub repositories for a natural language task."""
        query = task_query.strip()
        if not query:
            return []
        # ensure GitHub focus
        if "site:github.com" not in query:
            query = f"site:github.com {query}"
        results = await self.search_engine.search(query, max_results=top_k)
        ranked = []
        for item in results:
            ranked.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "score": item.get("score", 0.0),
            })
        return ranked

    async def analyze_repository(self, repo_path_or_url: str, depth: AnalysisDepth = AnalysisDepth.semantic) -> str:
        """Clone (if needed) and analyze a repository.

        Returns a stable repo_id that can be used to fetch summaries later.
        """
        repo_id = self._generate_repo_id(repo_path_or_url)
        target_dir = self.cache_dir / repo_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if repo_path_or_url.startswith("http"):
            await self._git_clone(repo_path_or_url, target_dir)
            source = repo_path_or_url
        else:
            src = Path(repo_path_or_url)
            if not src.exists():
                raise FileNotFoundError(f"Repository path {repo_path_or_url} not found")
            # copy working tree snapshot
            shutil.copytree(src, target_dir, dirs_exist_ok=True)
            source = str(src)

        summary = self._summarize_repository(repo_id, target_dir, source, depth)
        self._write_summary(summary)
        return repo_id

    async def get_repository_summary(self, repo_id: str) -> Dict[str, Any]:
        summary_path = self.cache_dir / repo_id / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary for {repo_id} not found")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Helpers

    def _generate_repo_id(self, source: str) -> str:
        digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:12]
        return f"repo_{digest}"

    async def _git_clone(self, repo_url: str, target_dir: Path) -> None:
        cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"git clone failed: {stderr.decode().strip()}")

    def _summarize_repository(self, repo_id: str, repo_dir: Path, source: str, depth: AnalysisDepth) -> RepoSummary:
        file_stats: Dict[str, int] = {}
        top_files: List[str] = []
        total_files = 0

        for root, _, files in os.walk(repo_dir):
            for name in files:
                total_files += 1
                ext = Path(name).suffix.lower() or "<no-ext>"
                file_stats[ext] = file_stats.get(ext, 0) + 1
                if len(top_files) < 20:
                    rel = str(Path(root).joinpath(name).relative_to(repo_dir))
                    top_files.append(rel)

        summary = RepoSummary(
            repo_id=repo_id,
            source=source,
            path=str(repo_dir),
            files_analyzed=total_files,
            languages=file_stats,
            top_files=top_files,
            metadata={
                "analysis_depth": depth,
                "notes": "Heuristic summary; integrate full RepoMaster pipeline for richer data.",
            },
        )
        return summary

    def _write_summary(self, summary: RepoSummary) -> None:
        summary_path = Path(summary.path) / "summary.json"
        summary_path.write_text(json.dumps({
            "repo_id": summary.repo_id,
            "source": summary.source,
            "files_analyzed": summary.files_analyzed,
            "languages": summary.languages,
            "top_files": summary.top_files,
            "metadata": summary.metadata,
        }, indent=2), encoding="utf-8")
