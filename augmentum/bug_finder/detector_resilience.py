"""Detector-stage resilience: bounded retry + circuit breaker.

Field data (06-14 run bfr_f80deb251ce5): 396/399 detector subagents
stopped with ``stop_reason="error"`` at iteration 0 — the high-concurrency
fan-out collapsed under provider rate-limits — yet the stage kept
scheduling all ~399 detectors, burning 811K tokens over 16 minutes before
reporting "no findings". Two gaps caused it:

* **No retry** — a transient backend error (429 / 5xx) permanently lost
  that chunk's detector run.
* **No circuit breaker** — once the backend went down, every remaining
  detector still fired and failed, hammering a dead provider and burning
  the whole budget on errors.

This module adds both, minimally and additively (no orchestrator
restructure):

* ``run_with_retry`` — bounded retry-with-backoff around one subagent
  run, so an isolated transient error doesn't kill a chunk.
* ``DetectorCircuitBreaker`` — a shared counter the detect stage feeds
  every result into. Once enough detectors have run AND the error rate
  crosses the threshold, the breaker OPENS; queued detectors check
  ``is_open`` and short-circuit instead of pounding a down backend. A
  16-minute / 811K-token futile grind becomes a fast, honest bail, and
  the detector-health gate (``evaluate_detector_health``) then reports
  the run degraded rather than "clean".

Async-safety: the breaker uses plain counters with no awaits between
read and mutate. Detectors run concurrently but on ONE event loop, so
``record``/``is_open`` are never truly preempted mid-operation — no lock
needed (same assumption the rest of the orchestrator's shared ledger
makes).
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class _HasStopReason(Protocol):
    stop_reason: str


@dataclass
class DetectorCircuitBreaker:
    """Trips when the detector error rate proves the backend is down.

    Conservative by construction: it needs ``min_samples`` completed
    detectors before it can open, so a couple of early flukes can't kill
    a healthy run. Once open it stays open for the rest of the stage —
    re-closing mid-run would just re-hammer the same down backend.
    """

    min_samples: int = 8
    error_rate_threshold: float = 0.6
    _total: int = 0
    _errored: int = 0
    _open: bool = False

    def record(self, stop_reason: str) -> None:
        """Feed one detector outcome. Opens the breaker if the live error
        rate has crossed the threshold past the minimum sample size."""
        self._total += 1
        if stop_reason == "error":
            self._errored += 1
        if (
            not self._open
            and self._total >= self.min_samples
            and (self._errored / self._total) >= self.error_rate_threshold
        ):
            self._open = True
            log.warning(
                "bug_finder_detector_circuit_open",
                total=self._total,
                errored=self._errored,
                error_rate=round(self.error_rate, 3),
                threshold=self.error_rate_threshold,
            )

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def error_rate(self) -> float:
        return (self._errored / self._total) if self._total else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Flat dict for SSE events + run notes."""
        return {
            "total": self._total,
            "errored": self._errored,
            "error_rate": round(self.error_rate, 3),
            "open": self._open,
            "min_samples": self.min_samples,
            "threshold": self.error_rate_threshold,
        }


def _default_retryable(result: _HasStopReason) -> bool:
    return getattr(result, "stop_reason", "") == "error"


async def run_with_retry(
    make_run: Callable[[], Awaitable[Any]],
    *,
    max_retries: int = 2,
    base_delay_s: float = 2.0,
    is_retryable: Callable[[Any], bool] = _default_retryable,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Run ``make_run()``; retry while the result is retryable.

    ``make_run`` is an async factory returning a result with a
    ``stop_reason`` (a ``SubagentResult``). On a retryable result it waits
    ``base_delay_s * 2**attempt`` (exponential backoff) and re-runs, up to
    ``max_retries`` extra attempts. Returns the last result either way —
    never raises on a retryable outcome, so the caller's ledger/parse path
    is unchanged. A permanent error (e.g. model-unavailable) simply
    exhausts retries and returns; the circuit breaker handles the
    systemic case.
    """
    result = await make_run()
    attempt = 0
    while attempt < max_retries and is_retryable(result):
        delay = base_delay_s * (2 ** attempt)
        log.info(
            "bug_finder_detector_retry",
            attempt=attempt + 1,
            max_retries=max_retries,
            delay_s=delay,
        )
        await sleep(delay)
        attempt += 1
        result = await make_run()
    return result


__all__ = ["DetectorCircuitBreaker", "run_with_retry"]
