"""Naïve lexical retrieval over parsed evidence spans."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List

from ..models import EvidenceSpan

_TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


class EvidenceRetriever:
    """Simple BM25-like retriever for in-memory evidence spans."""

    def __init__(self):
        self._spans: List[EvidenceSpan] = []
        self._term_freqs: List[Counter[str]] = []
        self._token_counts: List[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._total_docs = 0

    def add_span(self, span: EvidenceSpan) -> None:
        tokens = self._tokenize(span.text)
        if not tokens:
            return

        tf = Counter(tokens)
        self._spans.append(span)
        self._term_freqs.append(tf)
        self._token_counts.append(len(tokens))
        for term in tf:
            self._doc_freq[term] += 1
        self._total_docs += 1

    def add_spans(self, spans: Iterable[EvidenceSpan]) -> None:
        for span in spans:
            self.add_span(span)

    def search(self, query: str, limit: int = 5) -> List[EvidenceSpan]:
        if self._total_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx, span in enumerate(self._spans):
            score = 0.0
            token_count = self._token_counts[idx]
            tf = self._term_freqs[idx]
            for term in query_tokens:
                term_tf = tf.get(term, 0)
                if term_tf == 0:
                    continue
                idf = math.log((self._total_docs + 1) / (self._doc_freq[term] + 0.5)) + 1
                score += (term_tf / token_count) * idf
            if score > 0:
                scores.append((score, idx))

        ranked = sorted(scores, key=lambda item: item[0], reverse=True)[:limit]
        return [self._spans[idx].model_copy(update={"score": score}) for score, idx in ranked]

    @staticmethod
    def build_spans(paper_id: str, blocks: Iterable[dict]) -> List[EvidenceSpan]:
        spans: List[EvidenceSpan] = []
        for block in blocks:
            text = block.get("text", "")
            if not text:
                continue
            spans.append(
                EvidenceSpan(
                    paper_id=paper_id,
                    text=text,
                    section=block.get("section"),
                    page=block.get("page"),
                )
            )
        return spans

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


__all__ = ["EvidenceRetriever"]
