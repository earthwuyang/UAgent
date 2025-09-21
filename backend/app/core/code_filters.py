from __future__ import annotations

from .assertions import NO_MOCK_TOKENS


def reject_if_mock(code: str) -> None:
    if not code:
        return
    lower = code.lower()
    for tok in NO_MOCK_TOKENS:
        if tok in lower:
            raise RuntimeError(f"MOCK_CODE_DETECTED: contains '{tok}'")

