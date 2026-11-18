"""Optional client-side throttle for model calls.

One procurement request makes ~28 model calls, so a free-tier key (15 requests/minute per model)
hits 429 RESOURCE_EXHAUSTED mid-run. Set PROCURE_MAX_RPM (e.g. 12) and every model call waits for a
slot in a sliding 60 s window, per model name. Off (0) by default. List this plugin *before*
TelemetryPlugin so time spent waiting is not counted as model latency.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict, deque
from typing import Any, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin


class SlidingWindowLimiter:
    def __init__(self, max_per_minute: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = asyncio.sleep):
        self.max = max_per_minute
        self._clock, self._sleep = clock, sleep
        self._stamps: dict[str, deque[float]] = defaultdict(deque)
        self._lock: asyncio.Lock | None = None
        self.waited_s = 0.0

    async def acquire(self, key: str) -> None:
        if self.max <= 0:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:                      # serialise so concurrent callers cannot all grab one slot
            q = self._stamps[key]
            while True:
                now = self._clock()
                while q and now - q[0] >= 60:
                    q.popleft()
                if len(q) < self.max:
                    q.append(now)
                    return
                delay = 60 - (now - q[0]) + 0.05
                self.waited_s += delay
                await self._sleep(delay)


class RateLimitPlugin(BasePlugin):
    def __init__(self, max_per_minute: int | None = None, name: str = "procure_ratelimit"):
        super().__init__(name=name)
        self.limiter = SlidingWindowLimiter(
            int(os.getenv("PROCURE_MAX_RPM", "0")) if max_per_minute is None else max_per_minute)

    async def before_model_callback(self, *, callback_context, llm_request) -> Optional[Any]:
        await self.limiter.acquire(llm_request.model or "default")
        return None
