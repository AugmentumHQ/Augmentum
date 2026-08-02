"""tick_drive — first Phase 3a management verb.

Subscribes to ``time.tick(60s)`` and decays the four motivational
drives (curiosity / competence / connection / rest) toward their
baselines via the existing :func:`drives.decay` body. Phase 3a is a
"pure rename" — the logic body is unchanged; the verb just rebadges
it as a declared management verb that runs through the dispatcher.

Why a tick rather than an on-demand call: prior to this verb,
``drives.decay`` was only invoked lazily right before the
activity_selector scored candidates (see ``behavior/activity_selector
.py:1245``). If no activity ran for hours, drive state didn't decay —
the database row's ``last_decay_at`` lagged real time. The verb makes
the maintenance pass continuous regardless of who consumes the state.

The lazy call in activity_selector is preserved for now: ``decay()``
is idempotent on short elapsed intervals (it early-returns when
``elapsed_hours < 0.001``), so both paths converge. Phase 3a cleanup
will delete the lazy call once parity is verified.
"""

from __future__ import annotations

from augmentum.companion_runtime import drives
from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Cooldown sits just under the 60s tick interval so a normal ladder
# cycle always fires, but a duplicate publish within the same second
# (e.g. after a brief stop/start) is silently coalesced via verb_log.
_TICK_DRIVE_COOLDOWN_MS = 55_000


@verb(
    "time.tick(60s)",
    name="tick_drive",
    reads=("companion_drive_state",),
    writes=("companion_drive_state",),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_TICK_DRIVE_COOLDOWN_MS,
)
async def tick_drive(event, ctx) -> None:
    """Apply exponential decay to all four drive levels.

    No-ops cleanly when the runtime has no resolved owner yet (fresh
    install with no users) — the verb records ``ok`` with an empty
    citation rather than erroring.
    """
    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        # No bound user — nothing per-user to decay. Cite nothing; the
        # outcome is still ``ok`` so observability shows the verb is
        # running, just inert.
        return

    new_state = await drives.decay(runtime, user_id=owner)
    ctx.cite("companion_drive_state", row_id=owner)
    ctx.db_ops += 2  # 1 SELECT in load() + 1 UPDATE in decay()
    log.debug(
        "tick_drive_decayed",
        user_id=owner,
        levels={k: round(v, 3) for k, v in new_state.levels.items()},
    )


VerbRegistry.register(tick_drive)
