"""
Lightweight in-memory rate limiter used by API endpoints.

This is process-local and works well for single-instance deployments.
For multi-instance scaling, replace with Redis or another shared store.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
import time


@dataclass(frozen=True)
class RateLimitPolicy:
    limit: int
    window_seconds: int

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and self.window_seconds > 0


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def hit(self, bucket_key: str, policy: RateLimitPolicy) -> tuple[bool, int]:
        """
        Register one event and report if the request should be allowed.

        Returns:
            (allowed, retry_after_seconds)
        """
        if not policy.enabled:
            return True, 0

        now = time.monotonic()
        cutoff = now - policy.window_seconds

        with self._lock:
            events = self._events[bucket_key]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= policy.limit:
                retry_after = int(policy.window_seconds - (now - events[0])) + 1
                return False, max(1, retry_after)

            events.append(now)
            return True, 0
