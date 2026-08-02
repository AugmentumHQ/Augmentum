"""tick_scheduler — fourth Phase 3a management verb.

Subscribes to ``time.tick(60s)`` and runs one due standing task per
fire, via :func:`standing_tasks.step`. Phase 3a is a "pure rename" —
the step body (read one due row from ``companion_standing_tasks``,
dispatch its registered runner, persist ``next_run_at`` and result
summary) is unchanged; the verb just rebadges the call from inside
the legacy tick loop.

Cadence note: the previous call site at ``behavior/tick.py`` fired
every base tick (5–30s depending on the activity state axis). The
verb runs every 60s. Standing tasks are inherently minute-scale work
(RSS digests, github release polls, recurring searches) and each row
carries its own ``next_run_at`` schedule, so the cadence change just
makes the scan more honest about the work shape it dispatches.
A burst of many simultaneously-due rows now drains at ~1/min rather
than ~1/tick; raise the tick label if that turns out to be felt.

Self-gating still lives inside ``standing_tasks.step`` — it reads
``companion_standing_tasks_enabled`` and returns early when the
kill-switch is off, so the verb doesn't need to know.
"""

from __future__ import annotations

from augmentum.companion_runtime import standing_tasks
from augmentum.companion_runtime.event_bus import (
    CostEnvelope,
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Cooldown sits just under the 60s tick interval so a normal ladder
# cycle always fires, but a duplicate publish within the same second
# is silently coalesced via verb_log.
_SCHEDULER_COOLDOWN_MS = 55_000

# Wallclock budget for ONE standing-task run. The default verb envelope
# is 5s (dispatch_policy.DEFAULT_MAX_WALLCLOCK_MS) — fine for a presence
# tick, but standing tasks dispatch real work: a briefing runs several
# SearXNG gathers + a ~60s LLM synthesis, and prompt_fire has its OWN
# 120s internal cap (companion_prompt_fire_max_seconds). Under the 5s
# default every heavy kind was cancelled mid-run ("cancelled (wallclock
# exceeded)") and never delivered — briefings/scheduled requests simply
# never fired on schedule (run-now worked because it bypasses the verb).
# This backstop sits ABOVE the kinds' internal caps so a well-behaved
# kind finishes (or self-limits) before the hard kill. The dispatcher
# consumes time.tick events serially, so a long run delays — but never
# double-fires — the next tick.
_SCHEDULER_WALLCLOCK_MS = 180_000  # 3 min ceiling per run


@verb(
    "time.tick(60s)",
    name="tick_scheduler",
    reads=("companion_standing_tasks",),
    writes=("companion_standing_tasks", "companion_journal"),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_SCHEDULER_COOLDOWN_MS,
    cost_envelope=CostEnvelope(max_wallclock_ms=_SCHEDULER_WALLCLOCK_MS),
)
async def tick_scheduler(event, ctx) -> None:
    """Run one due standing task, if any.

    No-ops cleanly when there's no owner bound or no task is due;
    ``step`` itself returns silently in those cases.
    """
    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    await standing_tasks.step(runtime)
    # Cite the table even when no row was due — the SELECT happened.
    # Per-row citations would require ``step`` to thread back which
    # task it ran, which isn't worth the plumbing for Phase 3a.
    ctx.cite("companion_standing_tasks", row_id=owner)
    ctx.db_ops += 1  # Minimum 1 SELECT per call; UPDATE only when a task is due.
    log.debug("tick_scheduler_stepped", user_id=owner)


VerbRegistry.register(tick_scheduler)
