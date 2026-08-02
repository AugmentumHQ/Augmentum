"""Time-tick publisher for the management-verb dispatcher.

Publishes ``time.tick(<interval>)`` events at 5s / 60s / 5min / 1hr /
daily boundaries on PresenceBus. The dispatcher's registered verbs
subscribe via fnmatch globs like ``time.tick(60s)`` or
``time.tick(*)``.

Distinct from the existing ``TickLoop._tick`` driver (behavior/tick.py):

- ``TickLoop`` drives a single rate based on the companion state-axis
  budget (5s present / 30s dormant / 60s asleep) and fans into seven
  hardcoded subsystems. Phase 3a/3b absorbs those subsystems into
  declared verbs subscribed to this ladder.
- This ladder publishes typed multi-rate ticks at fixed wall-clock
  cadences regardless of state. Verbs that DO want state-budgeted
  cadence can subscribe to ``behavior.activity_chosen`` instead.

Sleep granularity is 5s (the shortest interval). All other intervals
are integer multiples checked on each cycle so a single tick task
covers the whole ladder.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from augmentum.companion_runtime.bus import PROP_FULL
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Interval definitions. (label, seconds). Order matters for the loop —
# shortest first so it drives sleep granularity.
INTERVALS: tuple[tuple[str, int], ...] = (
    ("5s",    5),
    ("60s",   60),
    ("5min",  300),
    ("1hr",   3600),
    ("daily", 86400),
)

# Topic format per spec: time.tick(<label>). fnmatch handles parens.
def topic_for(label: str) -> str:
    return f"time.tick({label})"


class TickLadder:
    """Multi-rate time-tick publisher. One asyncio task per runtime.

    Lifecycle:
      ``ladder = TickLadder(runtime); await ladder.start()`` opens.
      ``await ladder.stop()`` cancels the task and waits.
    """

    def __init__(self, runtime: CompanionRuntime) -> None:
        self._runtime = runtime
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        # Track the last published epoch second per label so we don't
        # double-fire after sleep skew.
        #
        # DELIBERATELY in-memory / NOT persisted (audit 2026-06-17): the
        # ladder is a best-effort CADENCE beacon, not a durable scheduler.
        # A daily boundary that passes while the process is down is
        # silently skipped on the next start (re-seeded to the current
        # boundary below) — and that's correct for cadence/decay verbs
        # ("sample now", meaningless to back-fill). Anything that MUST
        # survive a missed window belongs in companion_standing_tasks,
        # whose next_run_at is DB-persisted and restart-safe; a verb that
        # genuinely needs miss-detection should keep its own persisted
        # watermark rather than make the whole ladder durable.
        self._last_fired_at: dict[str, int] = {}

    async def start(self) -> None:
        if self._task is not None:
            return
        # Seed each interval's last-fired boundary to the most recent
        # past boundary at startup. Without this, every interval (5s up
        # through daily) would fire on the first publish cycle because
        # last_fired_at defaulted to 0 — producing a wasteful burst plus
        # an immediate cooldown_skipped on the next true wall-clock
        # boundary (the 30s-after-startup case noted in Phase 3a notes).
        # With this seed, fires happen only when the wall-clock boundary
        # actually moves forward.
        now = int(time.time())
        for label, interval_s in INTERVALS:
            self._last_fired_at[label] = now - (now % interval_s)
        self._stopped.clear()
        self._task = asyncio.create_task(self._run(), name="tick_ladder")
        log.info("tick_ladder_started", intervals=[lbl for lbl, _ in INTERVALS])

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        log.info("tick_ladder_stopped")

    async def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                # Sleep until the next 5s wall-clock boundary so that
                # ticks land on natural seconds (xx:00, xx:05, xx:10...)
                # rather than drifting from process-start time.
                now = time.time()
                next_boundary = (int(now / INTERVALS[0][1]) + 1) * INTERVALS[0][1]
                sleep_for = max(0.001, next_boundary - now)
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=sleep_for)
                except TimeoutError:
                    pass
                if self._stopped.is_set():
                    break
                await self._maybe_publish()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("tick_ladder_run_failed")

    async def _maybe_publish(self) -> None:
        """Publish every interval that just crossed a boundary."""
        now = int(time.time())
        for label, interval_s in INTERVALS:
            # Has the boundary for this interval just been crossed since
            # the last publish? Use modular arithmetic on the absolute
            # epoch so all instances share alignment.
            boundary = now - (now % interval_s)
            last = self._last_fired_at.get(label, 0)
            if boundary > last:
                self._last_fired_at[label] = boundary
                await self._publish(label, interval_s)

    async def _publish(self, label: str, interval_s: int) -> None:
        bus = getattr(self._runtime, "bus", None)
        if bus is None:
            return
        try:
            await bus.publish_topic(
                topic_for(label),
                {"interval_s": interval_s, "label": label},
                source_companion_id=self._runtime.companion_id,
                propagation=PROP_FULL,
            )
        except Exception:
            log.warning("tick_ladder_publish_failed", label=label, exc_info=True)


__all__ = ["INTERVALS", "topic_for", "TickLadder"]
