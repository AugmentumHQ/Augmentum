"""settle_today_reflection — third Phase 3b management verb.

Subscribes to ``time.tick(daily)`` and runs the nightly heal pass:
finalize yesterday's ``companion_today_reflections`` row per user
(immutable after settle, except for quarantine flips), soft-delete
journal entries past their archive expiry, and apply the forgetting
curve to old un-cited memories.

Why all three on one verb: they share the same nightly cadence,
are sequential within :func:`healing.daily_heal`, and the settle
step is the headline user-facing action — the others are bundled
hygiene riding the same wake-up.

Like ``tick_affect_baseline`` and ``tick_journal_compactor``, this
is a "first-time call" — :func:`healing.daily_heal` existed (the
``healing.py`` module Phase 1 catalog flagged as an orphan) but
was only invoked from tests. The verb is its first production
consumer.

Cadence: UTC-daily. ``daily_heal`` computes "yesterday's local
date" via ``time.localtime()``, which in the container resolves to
UTC; per-user timezone honoring is a deeper feature beyond Phase
3b scope. Self-gates on ``companion_healing_enabled`` inside the
function.
"""

from __future__ import annotations

from augmentum.companion_runtime import healing
from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Cooldown sits just under a day so a duplicate publish within the
# same minute is silently coalesced via verb_log.
_HEAL_COOLDOWN_MS = 86_390_000  # 23 h 59 m 50 s


@verb(
    "time.tick(daily)",
    name="settle_today_reflection",
    reads=("companion_journal", "companion_today_reflections"),
    writes=("companion_today_reflections", "companion_journal", "memories"),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_HEAL_COOLDOWN_MS,
)
async def settle_today_reflection(event, ctx) -> None:
    """Run the nightly daily_heal pass for this companion.

    Logs settled / soft-deleted / forgotten counts for observability;
    cites today_reflections as the headline written substrate.
    """
    runtime = ctx.runtime
    result = await healing.daily_heal(runtime)
    if result.get("skipped"):
        return
    ctx.cite("companion_today_reflections", row_id=runtime.companion_id)
    # Rough: 1 user fanout + ~3 ops per result domain. Approximate.
    ctx.db_ops += 1 + 3 * max(1, int(result.get("today_settled", 0)))
    log.debug(
        "settle_today_reflection_ran",
        today_settled=result.get("today_settled", 0),
        soft_deleted=result.get("soft_deleted", 0),
        forgetting_applied=result.get("forgetting_applied", 0),
    )


VerbRegistry.register(settle_today_reflection)
