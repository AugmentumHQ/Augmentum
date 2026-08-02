"""KeyedRateLimiter — per-IP throttle for public Connect endpoints (Phase 3e)."""

from __future__ import annotations

from augmentum.connect.rate_limit import KeyedRateLimiter


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def test_allows_up_to_limit_then_blocks():
    clock = _Clock()
    rl = KeyedRateLimiter(limit=3, window_s=60.0, clock=clock)
    assert rl.check("1.2.3.4") == (True, 0)
    assert rl.check("1.2.3.4") == (True, 0)
    assert rl.check("1.2.3.4") == (True, 0)
    allowed, retry = rl.check("1.2.3.4")
    assert allowed is False and retry >= 1


def test_window_slides_open_again():
    clock = _Clock()
    rl = KeyedRateLimiter(limit=2, window_s=60.0, clock=clock)
    rl.check("ip")
    rl.check("ip")
    assert rl.check("ip")[0] is False
    clock.advance(61)  # the window has passed
    assert rl.check("ip")[0] is True


def test_keys_are_isolated():
    rl = KeyedRateLimiter(limit=1, window_s=60.0, clock=_Clock())
    assert rl.check("a")[0] is True
    assert rl.check("a")[0] is False
    assert rl.check("b")[0] is True   # a different IP has its own bucket


def test_empty_key_is_allowed():
    rl = KeyedRateLimiter(limit=1, window_s=60.0, clock=_Clock())
    # No resolvable IP → don't punish (would block legit traffic behind a proxy).
    assert rl.check("")[0] is True
    assert rl.check("")[0] is True


def test_zero_limit_disables():
    rl = KeyedRateLimiter(limit=0, window_s=60.0, clock=_Clock())
    for _ in range(100):
        assert rl.check("ip")[0] is True
