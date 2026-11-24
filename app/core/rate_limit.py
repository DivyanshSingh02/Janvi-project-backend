"""Per-process sliding-window rate limiter.

This is intentionally in-memory for a single-worker deployment and deterministic tests.
When running multiple Uvicorn/Gunicorn workers, replace it with a shared Redis limiter.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil
from threading import Lock
from time import monotonic
from typing import Callable


@dataclass(frozen=True)
class RateLimit:
    requests: int
    window_seconds: int


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = max(1, int(retry_after))
        super().__init__("Rate limit exceeded")


class SlidingWindowRateLimiter:
    def __init__(self, clock: Callable[[], float] = monotonic):
        self.clock = clock
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Inject a deterministic clock in tests; no sleeping is required."""
        self.clock = clock

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def check(self, scope: str, key: str, limit: RateLimit) -> None:
        now = self.clock()
        bucket_key = (scope, key)
        with self._lock:
            events = self._events[bucket_key]
            cutoff = now - limit.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit.requests:
                raise RateLimitExceeded(ceil(events[0] + limit.window_seconds - now))
            events.append(now)
