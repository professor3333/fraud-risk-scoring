"""Bounded, process-local sliding windows for the single-worker public demo."""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from ipaddress import ip_address
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request


class RateLimitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    enabled: bool = True
    window_seconds: int = Field(default=60, gt=0)
    predict: int = Field(default=60, gt=0)
    bulk: int = Field(default=5, gt=0)
    explain: int = Field(default=10, gt=0)
    max_buckets: int = Field(default=10_000, gt=0)


def client_key(request: Request, *, render: bool) -> str:
    """Trust Render's edge-overwritten header only on that public hosting target.

    Elsewhere use the ASGI client address (Uvicorn controls proxy trust).
    Missing/invalid Render headers share a bucket, never a caller-chosen fallback.
    """
    if render:
        value = request.headers.get("cf-connecting-ip", "")
        try:
            return str(ip_address(value))
        except ValueError:
            return "render-unknown"
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Charge before dispatch; rejected calls neither extend nor reset a window.

    Entries are ordered by last admission, so expired buckets can be removed
    cheaply. At capacity refuse new buckets until space expires instead of
    evicting an active client's quota. A lock makes check-and-charge atomic.
    """

    def __init__(
        self, config: RateLimitConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.config = config
        self.clock = clock
        self._buckets: OrderedDict[tuple[str, str], deque[float]] = OrderedDict()
        self._lock = Lock()

    def retry_after(self, client: str, path: str, method: str) -> int:
        """Return zero to admit, otherwise seconds until a retry can be admitted."""
        groups = {
            "/predict": ("predict", self.config.predict),
            "/predict/batch": ("bulk", self.config.bulk),
            "/predict/csv": ("bulk", self.config.bulk),
            "/explain": ("explain", self.config.explain),
        }
        route = groups.get(path.rstrip("/"))
        if not self.config.enabled or method != "POST" or route is None:
            return 0
        group, limit = route
        key = (client, group)
        with self._lock:
            now = self.clock()
            cutoff = now - self.config.window_seconds
            while self._buckets:
                oldest = next(iter(self._buckets.values()))
                if oldest[-1] > cutoff:
                    break
                self._buckets.popitem(last=False)
            calls = self._buckets.get(key)
            if calls is None:
                if len(self._buckets) >= self.config.max_buckets:
                    oldest = next(iter(self._buckets.values()))
                    return max(1, math.ceil(oldest[-1] - cutoff))
                calls = deque()
                self._buckets[key] = calls
            while calls and calls[0] <= cutoff:
                calls.popleft()
            if len(calls) >= limit:
                return max(1, math.ceil(calls[0] - cutoff))
            calls.append(now)
            self._buckets.move_to_end(key)
            return 0
