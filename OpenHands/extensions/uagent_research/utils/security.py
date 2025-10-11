"""Security helpers for the UAgent research extension."""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from threading import RLock
from typing import Dict, Optional


_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def mask_secret(secret: Optional[str], visible: int = 4) -> str:
    """Mask a secret value, revealing only the last `visible` characters."""
    if not secret:
        return "<empty>"
    if len(secret) <= visible:
        return "*" * len(secret)
    return "*" * (len(secret) - visible) + secret[-visible:]


def sanitize_identifier(name: str, value: str, pattern: Optional[re.Pattern[str]] = None) -> str:
    """Validate identifier strings to prevent injection attacks."""
    if not value:
        raise ValueError(f"{name} must not be empty")

    checker = pattern or _ID_PATTERN
    if not checker.fullmatch(value):
        raise ValueError(
            f"Invalid {name}. Allowed characters: letters, digits, '.', '_', '-', ':'; max length 128."
        )
    return value


class SlidingWindowRateLimiter:
    """Simple in-memory rate limiter using a sliding window."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Rate limiter requires positive limit and window")
        self._limit = limit
        self._window = window_seconds
        self._events: Dict[str, deque[float]] = defaultdict(deque)
        self._lock = RLock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self._events[key]
            while bucket and now - bucket[0] > self._window:
                bucket.popleft()
            if len(bucket) >= self._limit:
                return False
            bucket.append(now)
            return True

    def quota(self, key: str) -> int:
        """Return remaining quota for a key within the current window."""
        now = time.time()
        with self._lock:
            bucket = self._events[key]
            while bucket and now - bucket[0] > self._window:
                bucket.popleft()
            return max(self._limit - len(bucket), 0)
