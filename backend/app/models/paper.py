"""Structured metadata models for scholarly papers."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class Author(BaseModel):
    """Author metadata"""

    name: str
    orcid: Optional[str] = None
    affiliations: List[str] = Field(default_factory=list)


class Paper(BaseModel):
    """Normalized paper metadata across literature providers"""

    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[Author] = Field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[int] = None
    url: Optional[HttpUrl] = None
    pdf_url: Optional[HttpUrl] = None
    source: str
    doi: Optional[str] = None
    open_access: bool = False
    tags: List[str] = Field(default_factory=list)

    def display_title(self) -> str:
        """Return a concise display title"""

        venue_part = f" ({self.venue})" if self.venue else ""
        year_part = f" {self.year}" if self.year else ""
        return f"{self.title}{venue_part}{year_part}"
