"""tick_affect_baseline — first Phase 3b management verb.

Subscribes to ``time.tick(1hr)`` and rebuilds every user's affect
baselines for all three windows (7d / 30d / 90d) via
:func:`affect.rebuild_all_baselines`. The baselines are read at
``affect.perceive`` time and at PAD projection — keeping them fresh
is what makes "felt drift" possible (the texture detector compares
last-60-min facet density against the rolling baselines).

Why this is a "first-time call" rather than a rebadge: the function
existed (perception/affect.py:248) and was docstring'd as a nightly
job, but nothing actually invoked it. This verb is its first
consumer. Until this lands, baselines are written once at the first
``build_baseline`` call (whenever some path triggers it) and then
drift further out of date with every passing day.

Cadence is hourly rather than nightly because:
  - The function is idempotent (UPSERT per window).
  - Per-user cost is three SELECTs + three UPSERTs against
    personality_facet_activations + companion_affect_baselines, both
    well-indexed.
  - Drift between baseline and now is the substrate that PAD's
    "noticeable shift" logic consumes — slower refresh = staler floor
    = jumpier emit_pad_if_delta events.

Self-gates inside the function: it iterates user_ids that already
have facet_activations rows; users with no activity yet are skipped.
"""

from __future__ import annotations

from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.perception import affect
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Cooldown sits just under the 1hr tick interval so a normal cycle
# always fires, but a duplicate publish within the same minute
# is silently coalesced via verb_log.
_BASELINE_COOLDOWN_MS = 3_590_000  # 59 min 50 s


@verb(
    "time.tick(1hr)",
    name="tick_affect_baseline",
    reads=("personality_facet_activations",),
    writes=("companion_affect_baselines",),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_BASELINE_COOLDOWN_MS,
)
async def tick_affect_baseline(event, ctx) -> None:
    """Refresh every user's affect baselines across all windows.

    Logs the per-user / per-window count of refreshed rows for
    observability; cites the baseline table once for the dispatch.
    """
    runtime = ctx.runtime
    counts = await affect.rebuild_all_baselines(runtime)
    refreshed = sum(counts.values()) if counts else 0
    ctx.cite("companion_affect_baselines", row_id=runtime.companion_id)
    # 6 ops per user (3 SELECT + 3 UPSERT); counts has one entry per user.
    ctx.db_ops += 6 * len(counts) if counts else 0
    log.debug(
        "tick_affect_baseline_refreshed",
        users=len(counts),
        windows_total=refreshed,
    )


VerbRegistry.register(tick_affect_baseline)
