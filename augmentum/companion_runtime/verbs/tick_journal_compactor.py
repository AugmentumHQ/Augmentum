"""tick_journal_compactor — second Phase 3b management verb.

Subscribes to ``time.tick(daily)`` and consolidates each user's
old journal entries via :func:`healing.weekly_consolidate`. Entries
older than 30 days get grouped into 7-day windows and summarized
into ``companion_journal_archive``; the source rows are stamped
``archived_at`` so the journal table stays bounded.

Like ``tick_affect_baseline``, this is a "first-time call" — the
function existed (healing.py:175, the file Phase 1 catalog flagged
as the ``healing.py`` orphan) but was only invoked from tests. The
verb is its first production consumer.

Cadence is daily because:
  - Source entries cluster into 7-day windows, so anything more
    frequent re-walks the same partitioned dataset for no benefit.
  - The function self-gates on ``companion_healing_enabled`` so it
    will no-op cleanly while the kill-switch is off.

Per-user iteration mirrors ``rebuild_all_baselines`` — distinct
user_ids that have any journal rows for this companion get
processed in turn, with per-user failures swallowed so one bad row
doesn't pause the whole fan-out.
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
_COMPACTOR_COOLDOWN_MS = 86_390_000  # 23 h 59 m 50 s


async def _journal_user_ids(runtime) -> list[str]:
    """Distinct user_ids that have any journal rows for this companion."""
    backend = runtime.backend
    cur = await backend.conn.execute(
        "SELECT DISTINCT user_id FROM companion_journal "
        "WHERE companion_id = ? AND user_id != ''",
        (runtime.companion_id,),
    )
    rows = await cur.fetchall()
    return [r[0] for r in rows]


@verb(
    "time.tick(daily)",
    name="tick_journal_compactor",
    reads=("companion_journal",),
    writes=("companion_journal", "companion_journal_archive"),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_COMPACTOR_COOLDOWN_MS,
)
async def tick_journal_compactor(event, ctx) -> None:
    """Run weekly journal consolidate for every user with journal entries."""
    runtime = ctx.runtime
    user_ids = await _journal_user_ids(runtime)
    total_windows = 0
    total_archived = 0
    for user_id in user_ids:
        try:
            result = await healing.weekly_consolidate(runtime, user_id=user_id)
        except Exception:
            log.exception("journal_compact_failed", user_id=user_id)
            continue
        if not result.get("skipped"):
            total_windows += int(result.get("windows_consolidated", 0))
            total_archived += int(result.get("entries_archived", 0))
            ctx.cite("companion_journal_archive", row_id=user_id)
    # 1 SELECT for user fanout + ~3 ops per user (SELECT old rows + INSERT
    # archive + UPDATE archived_at). Approximate is fine for telemetry.
    ctx.db_ops += 1 + 3 * len(user_ids)
    log.debug(
        "tick_journal_compactor_ran",
        users=len(user_ids),
        windows_consolidated=total_windows,
        entries_archived=total_archived,
    )


VerbRegistry.register(tick_journal_compactor)
