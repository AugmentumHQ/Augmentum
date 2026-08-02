"""Detector resilience — circuit breaker + bounded retry.

Pins the behavior that converts the 06-14 failure mode (396/399 detectors
errored over 16 min / 811K tokens, no retry, no early bail) into a fast,
honest degraded run: queued detectors short-circuit once the backend
proves down, and isolated transient errors retry with backoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from augmentum.bug_finder.detector_resilience import (
    DetectorCircuitBreaker,
    run_with_retry,
)

# ----------------------------------------------------------------------
# Circuit breaker
# ----------------------------------------------------------------------

def test_breaker_stays_closed_below_min_samples():
    b = DetectorCircuitBreaker(min_samples=8, error_rate_threshold=0.6)
    for _ in range(7):
        b.record("error")
    assert b.is_open is False  # 100% error but only 7 < 8 samples


def test_breaker_opens_at_threshold_past_min_samples():
    b = DetectorCircuitBreaker(min_samples=8, error_rate_threshold=0.6)
    for _ in range(8):
        b.record("error")
    assert b.is_open is True
    assert b.error_rate == 1.0


def test_breaker_does_not_open_below_threshold():
    b = DetectorCircuitBreaker(min_samples=8, error_rate_threshold=0.6)
    for _ in range(5):
        b.record("complete")
    for _ in range(5):
        b.record("error")  # 5/10 = 50% < 60%
    assert b.is_open is False
    assert b.error_rate == 0.5


def test_breaker_latches_open():
    """Once open it stays open even if later detectors succeed —
    re-closing mid-run would just re-hammer the same down backend."""
    b = DetectorCircuitBreaker(min_samples=4, error_rate_threshold=0.6)
    for _ in range(4):
        b.record("error")
    assert b.is_open is True
    for _ in range(20):
        b.record("complete")
    assert b.is_open is True


def test_breaker_budget_and_stuck_are_not_errors():
    b = DetectorCircuitBreaker(min_samples=4, error_rate_threshold=0.6)
    for _ in range(10):
        b.record("budget")
    assert b.is_open is False
    assert b.error_rate == 0.0


def test_breaker_snapshot_shape():
    b = DetectorCircuitBreaker(min_samples=8, error_rate_threshold=0.6)
    b.record("error")
    snap = b.snapshot()
    assert set(snap) == {
        "total", "errored", "error_rate", "open", "min_samples", "threshold",
    }
    assert snap["total"] == 1 and snap["errored"] == 1


# ----------------------------------------------------------------------
# Bounded retry
# ----------------------------------------------------------------------

@dataclass
class _R:
    stop_reason: str


class _Runner:
    """Async factory returning a scripted sequence of results."""

    def __init__(self, sequence: list[str]) -> None:
        self._seq = list(sequence)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        # Repeat the last value once the script is exhausted.
        reason = self._seq[min(self.calls - 1, len(self._seq) - 1)]
        return _R(reason)


@pytest.fixture
def fake_sleep():
    delays: list[float] = []

    async def _sleep(d: float) -> None:
        delays.append(d)

    _sleep.delays = delays  # type: ignore[attr-defined]
    return _sleep


@pytest.mark.asyncio
async def test_retry_returns_immediately_on_success(fake_sleep):
    runner = _Runner(["complete"])
    result = await run_with_retry(
        runner, max_retries=2, base_delay_s=2.0, sleep=fake_sleep,
    )
    assert result.stop_reason == "complete"
    assert runner.calls == 1
    assert fake_sleep.delays == []  # no retry, no sleep


@pytest.mark.asyncio
async def test_retry_recovers_on_second_attempt(fake_sleep):
    runner = _Runner(["error", "complete"])
    result = await run_with_retry(
        runner, max_retries=2, base_delay_s=2.0, sleep=fake_sleep,
    )
    assert result.stop_reason == "complete"
    assert runner.calls == 2
    assert fake_sleep.delays == [2.0]  # one backoff of base * 2**0


@pytest.mark.asyncio
async def test_retry_exhausts_and_returns_last_error(fake_sleep):
    runner = _Runner(["error"])  # always errors
    result = await run_with_retry(
        runner, max_retries=2, base_delay_s=2.0, sleep=fake_sleep,
    )
    assert result.stop_reason == "error"
    assert runner.calls == 3  # initial + 2 retries
    assert fake_sleep.delays == [2.0, 4.0]  # exponential backoff


@pytest.mark.asyncio
async def test_retry_disabled_with_zero_max(fake_sleep):
    runner = _Runner(["error"])
    result = await run_with_retry(
        runner, max_retries=0, base_delay_s=2.0, sleep=fake_sleep,
    )
    assert runner.calls == 1
    assert result.stop_reason == "error"
