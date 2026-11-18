import asyncio

from agents.ratelimit import RateLimitPlugin, SlidingWindowLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    async def sleep(self, d):
        self.t += d


async def test_allows_up_to_limit_then_waits_for_the_window():
    c = Clock()
    lim = SlidingWindowLimiter(3, clock=c, sleep=c.sleep)
    for _ in range(3):
        await lim.acquire("m")
    assert c.t == 1000.0 and lim.waited_s == 0          # first 3 are immediate
    await lim.acquire("m")                               # 4th must wait ~60 s for the first slot to expire
    assert 60 <= c.t - 1000.0 < 61


async def test_limit_is_per_model():
    c = Clock()
    lim = SlidingWindowLimiter(1, clock=c, sleep=c.sleep)
    await lim.acquire("a")
    await lim.acquire("b")
    assert c.t == 1000.0


async def test_disabled_when_zero():
    c = Clock()
    lim = SlidingWindowLimiter(0, clock=c, sleep=c.sleep)
    for _ in range(100):
        await lim.acquire("m")
    assert c.t == 1000.0


async def test_concurrent_callers_never_exceed_limit_in_a_window():
    c = Clock()
    lim = SlidingWindowLimiter(5, clock=c, sleep=c.sleep)
    stamps = []

    async def one():
        await lim.acquire("m")
        stamps.append(c.t)

    await asyncio.gather(*[one() for _ in range(20)])
    for s in stamps:
        assert sum(1 for x in stamps if s <= x < s + 60) <= 5


def test_plugin_reads_env(monkeypatch):
    monkeypatch.setenv("PROCURE_MAX_RPM", "12")
    assert RateLimitPlugin().limiter.max == 12
    assert RateLimitPlugin(max_per_minute=0).limiter.max == 0
