"""Autonomous tick loop.

Event-driven. The loop wakes on a bus event *or* a state-budget
timeout, whichever comes first, then runs a single ``_tick()``. Each
tick:

1. Reads current state/role/focus.
2. Asks :mod:`role_channel` whether to act and as what role.
3. Asks :mod:`activity_selector` for a candidate activity.
4. If candidate utility ≥ threshold, performs it.
5. Updates initiative queue and recency.

Hard rate-limit per-state via ``companion_tick_state_budget_ms`` so a
chatty bus can't thrash the loop. The 250ms coalescing window batches
events received during a tick into the next tick rather than firing
multiple times back-to-back.

Registration: the runtime registers ``register_tick()`` against the
existing ``augmentum/jobs/runner.py::JobRunner`` so we share the
process-wide scheduler. No parallel scheduler.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)

# State-budget defaults (ms). Read from settings.companion_tick_state_budget_ms
# but with a safe in-code default so the loop always has a budget.
_DEFAULT_BUDGETS_MS: dict[str, int] = {
    "asleep": 60_000,      # never really ticking; safety floor
    "dormant": 30_000,     # she's around but not engaged
    "present": 5_000,      # she's here with us
}

# Coalesce burst events into one tick within this window.
_COALESCE_WINDOW_S: float = 0.25


class TickLoop:
    """Per-runtime tick loop. Owned by the runtime; not a singleton."""

    def __init__(self, runtime: CompanionRuntime) -> None:
        self.runtime = runtime
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._tick_count = 0
        self._last_tick_at = 0.0
        self._subscription = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        # Subscribe to a coarse glob — every state/role/focus change
        # plus user intents poke the loop. Filtering happens inside
        # ``_tick()`` so we don't miss correlated changes.
        self._subscription = await self.runtime.bus.subscribe(
            "**",
            slice_key="tick_loop",
        )
        self._task = asyncio.create_task(self._run(), name="companion_tick_loop")
        log.info("companion_tick_started", companion_id=self.runtime.companion_id)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._wake_event.set()
        if self._subscription is not None:
            await self.runtime.bus.unsubscribe(self._subscription)
            self._subscription = None
        try:
            await asyncio.wait_for(self._task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        log.info("companion_tick_stopped", companion_id=self.runtime.companion_id)

    def poke(self) -> None:
        """Wake the loop early. Called by external code that knows
        something interesting just happened (e.g., a journal write)."""
        self._wake_event.set()

    async def _run(self) -> None:
        """Main loop — wait for a wake signal or state-budget timeout,
        then tick once. The subscription drains itself in the background
        via ``_drain_subscription``; we only care about the wake side
        effect, not the events themselves."""
        drain_task = asyncio.create_task(
            self._drain_subscription(), name="companion_tick_drain",
        )
        try:
            while not self._stop_event.is_set():
                budget_s = self._current_budget_s()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=budget_s)
                except asyncio.TimeoutError:
                    pass
                self._wake_event.clear()
                # Coalesce a burst of events into one tick.
                await asyncio.sleep(_COALESCE_WINDOW_S)
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                # Min-interval floor: a storm of wake events right after a
                # long tick (an LLM journal/creation call) used to re-tick
                # immediately, since coalescing only de-dupes a 0.25s burst,
                # not the whole tick duration (audit 2026-06-17). Sleep off
                # any remaining interval rather than skip, so the
                # introspective tick still happens — just not faster than
                # the floor.
                min_interval = self._min_tick_interval_s()
                elapsed = time.time() - self._last_tick_at
                if min_interval > 0 and 0 <= elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
                    if self._stop_event.is_set():
                        break
                try:
                    await self._tick()
                except Exception:
                    log.exception("companion_tick_failed")
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _drain_subscription(self) -> None:
        """Continuously drain the subscription queue. Each event sets
        ``_wake_event`` so the main loop runs another tick.

        Filter contract: an event triggers a re-wake ONLY when it
        originated outside this runtime. The tick itself emits
        ``behavior.activity_chosen``, ``affect.changed``, ``affect.pad``,
        ``state.transition``, etc. — looping on those would spin the
        tick at coalesce-window cadence (0.25s) and fire the LLM
        dozens of times per minute. We bypass our own emissions by
        comparing ``source_companion_id``; external wakeups (voice,
        device, user interaction) carry a different / empty source
        and still propagate normally.
        """
        sub = self._subscription
        if sub is None:
            return
        own_id = self.runtime.companion_id
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
            # Defense in depth: legacy topic filter for tick.* in case
            # a downstream emitter forgets to set source_companion_id.
            if event.topic.startswith(("tick.", "behavior.tick")):
                continue
            # Primary filter: our own emissions never wake us.
            if getattr(event, "source_companion_id", "") == own_id:
                continue
            self._wake_event.set()

    def _current_budget_s(self) -> float:
        """State budget (seconds). Caps the time between ticks even
        without any bus traffic — ensures Becca takes at least one
        introspective look per budget window."""
        try:
            from augmentum.config import settings
            cfg = getattr(settings, "companion_tick_state_budget_ms", None)
            budgets = _DEFAULT_BUDGETS_MS if not isinstance(cfg, dict) else {
                **_DEFAULT_BUDGETS_MS, **cfg,
            }
        except Exception:
            budgets = _DEFAULT_BUDGETS_MS
        state_snap = self.runtime.state.snapshot()
        state_axis = state_snap.get("state", "dormant")
        return float(budgets.get(state_axis, budgets["dormant"])) / 1000.0

    def _min_tick_interval_s(self) -> float:
        """Lower bound on seconds between ticks — collapses post-long-tick
        wake storms. 0 disables the floor."""
        try:
            from augmentum.config import settings
            return float(getattr(settings, "companion_min_tick_interval_s", 2.0))
        except Exception:
            return 2.0

    async def _tick(self) -> None:
        """One iteration. Lazy-imports to avoid circular deps."""
        from augmentum.config import settings
        if not getattr(settings, "companion_tick_enabled", False):
            return
        self._tick_count += 1
        self._last_tick_at = time.time()
        await self.runtime.bus.publish_topic(
            "tick.fired",
            {"count": self._tick_count},
            source_companion_id=self.runtime.companion_id,
        )

        # Drive the primary state axis from observed activity + quiet
        # hours. Without this the state machine sits at ``dormant``
        # forever and every downstream rule (tick budget, activity
        # selector temperature, dream gating) reads a stale axis.
        from augmentum.companion_runtime.behavior import state_driver
        try:
            await state_driver.drive_once(self.runtime)
        except Exception:
            log.exception("state_driver_failed")

        # PAD continuous-affect bridge has moved to the
        # ``emit_pad_if_delta`` management verb (Phase 3a). It now
        # samples on ``time.tick(60s)`` rather than per-tick, since
        # PAD reflects a 60-minute window.

        # Periodic drift-audit rehearsal (Sprint 4a). Self-gates on
        # interval + flag + doc existence; cheap when not due.
        from augmentum.companion_runtime.behavior import drift_audit
        try:
            await drift_audit.run_if_due(self.runtime)
        except Exception:
            log.exception("drift_audit_failed")

        # Initiative scoring (Piece 7'). Self-gates on:
        #   - companion_initiative_enabled (kill switch)
        #   - companion_initiative_min_interval_s (default 60s) so the
        #     5-30s tick cadence doesn't fan 4 SELECTs per tick.
        # When scored, may write to companion_initiative_queue and
        # bus-emit `initiative.surfaced`. activity_selector below can
        # then boost matching candidates (Piece 9').
        from augmentum.companion_runtime.behavior import initiative
        try:
            await initiative.step(self.runtime)
        except Exception:
            log.exception("initiative_step_failed")

        # Wondering writer. Self-gates on companion_topical_aggregator_enabled,
        # presence_mode, hush, user-active, and daily cap (default 3/day).
        # A successful write fires today.maybe_regenerate as a side effect —
        # this is the path that populates the Today reflection surface.
        owner_uid = getattr(self.runtime, "owner_user_id", "") or ""
        if owner_uid:
            from augmentum.companion_runtime import wondering
            try:
                await wondering.maybe_write_wondering(
                    self.runtime, user_id=owner_uid,
                )
            except Exception:
                log.exception("wondering_step_failed")

        # Perception pass (Sovereign Perception Pipeline). The ambient
        # (in_conversation=False) run: fuse the signals we already observe
        # (browse/media/presence — zero new permissions) into insights, run
        # them through the regret-gated judgment gate, and deliver via the
        # initiative queue + bus. Self-gates on companion_perception_enabled
        # (default OFF during rollout). Ships with no fusers yet, so until real
        # fusers land it's a cheap no-op even when enabled.
        if owner_uid:
            from augmentum.companion_runtime.perception import live as _perception
            try:
                await _perception.evaluate_user(
                    self.runtime, user_id=owner_uid, in_conversation=False,
                )
            except Exception:
                log.exception("perception_pass_failed")

        # Curator. The primary autonomous writer for the notes drawer —
        # polls discovery feeds + RSS subscriptions, filters by tracked
        # topics, writes ONE structured note per debounce-window. Self-
        # gates on companion_curator_enabled + per-runtime interval +
        # presence_mode + dedup against last week of journal URLs.
        if owner_uid:
            from augmentum.companion_runtime import curator
            try:
                await curator.step(self.runtime)
            except Exception:
                log.exception("curator_step_failed")

        # Alert watch — NWS severe weather + USGS quakes near the saved
        # home location (the one weather.today learns). Self-gates on
        # companion_alert_watch_enabled + its own 10-min interval +
        # home presence; zero LLM cost.
        if owner_uid:
            from augmentum.companion_runtime import alert_watch
            try:
                await alert_watch.step(self.runtime)
            except Exception:
                log.exception("alert_watch_step_failed")

        # Standing tasks scheduling has moved to the ``tick_scheduler``
        # management verb (Phase 3a). Drains one due row per 60s tick.

        # Decide whether to act this tick. role_channel may say "stay
        # observer", in which case we return without selecting.
        from augmentum.companion_runtime.behavior import role_channel
        verdict = await role_channel.advise(self.runtime)
        if not verdict.should_act:
            return

        from augmentum.companion_runtime.behavior import activity_selector
        choice = await activity_selector.choose(self.runtime, role=verdict.role)
        if choice is None or choice.utility < choice.threshold:
            return

        await self.runtime.bus.publish_topic(
            "behavior.activity_chosen",
            {
                "kind": choice.kind,
                "utility": round(choice.utility, 3),
                "role": verdict.role,
                # Phase 3a — drive name lives in the event so the
                # apply_signal verb doesn't need to re-import the
                # candidate→drive map from activity_selector.
                "drive": choice.drive,
            },
            source_companion_id=self.runtime.companion_id,
        )
        try:
            await choice.perform(self.runtime)
        except Exception as exc:
            log.exception("activity_perform_failed", kind=choice.kind, error=str(exc))

    def snapshot(self) -> dict:
        return {
            "running": self._task is not None and not self._task.done(),
            "tick_count": self._tick_count,
            "last_tick_at": self._last_tick_at,
        }


# PAD emit moved to ``verbs/emit_pad_if_delta.py`` — Phase 3a.
