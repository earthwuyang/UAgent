"""PubMedQA multiple-choice benchmark."""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable

from ..core.llm_client import LLMClient
from .base import BenchmarkResult
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class PubMedQAExample(BaseModel):
    """Single PubMedQA example with three-way classification."""

    id: str
    question: str
    context: str
    choices: Dict[str, str] = Field(default_factory=dict)
    answer: str


class PubMedQABenchmark:
    """Evaluate an LLM client on PubMedQA-style questions."""

    def __init__(self, llm_client: LLMClient, temperature: float = 0.0):
        self.llm_client = llm_client
        self.temperature = temperature

    async def evaluate(self, examples: Iterable[PubMedQAExample]) -> BenchmarkResult:
        total = 0
        correct = 0
        details = []

        for example in examples:
            total += 1
            prompt = self._build_prompt(example)
            try:
                response = await self.llm_client.generate(
                    prompt,
                    max_tokens=8,
                    temperature=self.temperature,
                )
            except Exception as exc:  # pragma: no cover - evaluation best effort
                LOGGER.error("PubMedQA generation failed for %s: %s", example.id, exc)
                details.append({"id": example.id, "error": str(exc)})
                continue

            predicted = self._parse_choice(response)
            is_correct = predicted.lower() == example.answer.lower()
            if is_correct:
                correct += 1

            details.append({
                "id": example.id,
                "predicted": predicted,
                "expected": example.answer,
                "raw_response": response.strip(),
            })

        return BenchmarkResult.from_counts("PubMedQA", total, correct, details)

    @staticmethod
    def _build_prompt(example: PubMedQAExample) -> str:
        options = "\n".join(
            f"{key.upper()}) {value}" for key, value in example.choices.items()
        )
        return (
            "You are evaluating biomedical research questions. "
            "Choose the single best answer (A, B, or C) and reply with just the letter.\n"
            f"Question: {example.question}\n"
            f"Context: {example.context}\n"
            f"Choices:\n{options}\n"
            "Answer:"
        )

    @staticmethod
    def _parse_choice(response: str) -> str:
        match = re.search(r"([ABC])", response.upper())
        if match:
            return match.group(1)
        return response.strip()[:1].upper()
