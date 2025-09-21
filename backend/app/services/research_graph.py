"""Lightweight research knowledge graph backed by SQLite."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from ..models import Author, Claim, Paper, VerificationResult

LOGGER = logging.getLogger(__name__)


class ResearchGraphService:
    """Persist literature metadata, claims, and supporting evidence."""

    def __init__(self, db_path: str | Path = "research_graph.db"):
        self.db_path = Path(db_path)
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    abstract TEXT,
                    venue TEXT,
                    year INTEGER,
                    url TEXT,
                    source TEXT,
                    doi TEXT,
                    authors_json TEXT,
                    open_access INTEGER
                );

                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    normalized_question TEXT
                );

                CREATE TABLE IF NOT EXISTS claim_verifications (
                    claim_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL,
                    reasoning TEXT,
                    citations_json TEXT,
                    rationales_json TEXT,
                    PRIMARY KEY (claim_id, paper_id)
                );

                CREATE TABLE IF NOT EXISTS citations (
                    from_paper TEXT NOT NULL,
                    to_paper TEXT NOT NULL,
                    PRIMARY KEY (from_paper, to_paper)
                );
                """
            )

    def upsert_paper(self, paper: Paper) -> None:
        payload = paper.model_dump()
        authors_json = json.dumps([author.model_dump() for author in paper.authors])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (id, title, abstract, venue, year, url, source, doi, authors_json, open_access)
                VALUES (:id, :title, :abstract, :venue, :year, :url, :source, :doi, :authors_json, :open_access)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    abstract=excluded.abstract,
                    venue=excluded.venue,
                    year=excluded.year,
                    url=excluded.url,
                    source=excluded.source,
                    doi=excluded.doi,
                    authors_json=excluded.authors_json,
                    open_access=excluded.open_access;
                """,
                {
                    "id": payload["id"],
                    "title": payload["title"],
                    "abstract": payload.get("abstract"),
                    "venue": payload.get("venue"),
                    "year": payload.get("year"),
                    "url": payload.get("url"),
                    "source": payload.get("source"),
                    "doi": payload.get("doi"),
                    "authors_json": authors_json,
                    "open_access": int(bool(payload.get("open_access"))),
                },
            )

    def record_claim(self, claim: Claim, claim_id: Optional[str] = None) -> str:
        claim_identifier = claim_id or claim.text[:128]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claims (id, text, normalized_question)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET text=excluded.text, normalized_question=excluded.normalized_question;
                """,
                (claim_identifier, claim.text, claim.normalized_question),
            )
        return claim_identifier

    def record_verification(self, verification: VerificationResult, claim_id: Optional[str] = None) -> None:
        claim_identifier = claim_id or verification.claim.text[:128]
        citations_json = json.dumps(verification.citations)
        rationales_json = json.dumps([rationale.model_dump() for rationale in verification.rationales])
        if verification.rationales:
            primary_paper = verification.rationales[0].paper_id
        elif verification.citations:
            primary_paper = verification.citations[0]
        else:
            primary_paper = "unknown"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claim_verifications (claim_id, paper_id, label, confidence, reasoning, citations_json, rationales_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(claim_id, paper_id) DO UPDATE SET
                    label=excluded.label,
                    confidence=excluded.confidence,
                    reasoning=excluded.reasoning,
                    citations_json=excluded.citations_json,
                    rationales_json=excluded.rationales_json;
                """,
                (
                    claim_identifier,
                    primary_paper,
                    verification.label,
                    verification.confidence,
                    verification.reasoning,
                    citations_json,
                    rationales_json,
                ),
            )

    def add_citation(self, from_paper: str, to_paper: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO citations (from_paper, to_paper) VALUES (?, ?);
                """,
                (from_paper, to_paper),
            )

    def recent_papers(self, limit: int = 20) -> List[Paper]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, abstract, venue, year, url, source, doi, authors_json, open_access
                FROM papers
                ORDER BY rowid DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()

        papers: List[Paper] = []
        for row in rows:
            authors_payload = json.loads(row["authors_json"]) if row["authors_json"] else []
            papers.append(
                Paper(
                    id=row["id"],
                    title=row["title"],
                    abstract=row["abstract"],
                    venue=row["venue"],
                    year=row["year"],
                    url=row["url"],
                    source=row["source"],
                    doi=row["doi"],
                    open_access=bool(row["open_access"]),
                    authors=[Author(**author) for author in authors_payload],
                )
            )
        return papers


__all__ = ["ResearchGraphService"]
