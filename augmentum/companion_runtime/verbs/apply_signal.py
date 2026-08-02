"""apply_signal — second Phase 3a management verb.

Subscribes to ``behavior.activity_chosen`` and reduces the drive that
the chosen activity satiates (curiosity / competence / connection /
rest), via :func:`drives.satiate`. Phase 3a is a "pure rename" — the
satiation amount and per-drive recency dampening are unchanged; the
verb just rebadges what used to be the ``_wrapped`` performer's
``finally`` block in ``activity_selector.choose``.

Why event-driven rather than tick-aligned: satiation should fire
exactly when an activity is chosen, not on a clock. The bus event is
already published at ``behavior/tick.py`` immediately before
``choice.perform``; the new event payload carries ``drive`` so the
verb doesn't need to re-import the candidate→drive map from
``activity_selector``.

Gating mirrors the prior wrapper:
  - ``companion_drives_enabled`` — kill switch, default False.
  - ``runtime.owner_user_id`` resolved — no per-user row otherwise.
  - ``drive`` field present in event payload — older publishers (none
    in production at this point) without the field are no-oped.
"""

from __future__ import annotations

from augmentum.companion_runtime import drives
from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@verb(
    "behavior.activity_chosen",
    name="apply_signal",
    reads=("companion_drive_state",),
    writes=("companion_drive_state",),
    dispatch_class=DispatchClass.EVENT_DRIVEN,
    safety_class=SafetyClass.WRITE_SELF,
    # No cooldown — every chosen activity should satiate. Activity
    # cadence is already gated by the tick loop + role_channel + the
    # utility threshold in activity_selector, so the dispatcher can
    # trust this event to be intentional.
    cooldown_ms=0,
)
async def apply_signal(event, ctx) -> None:
    """Satiate the drive tagged on the chosen activity.

    Records ``ok`` with a citation on success; silently no-ops (also
    ``ok``, no citation) when drives are gated off or the event lacks
    a drive field.
    """
    if not getattr(settings, "companion_drives_enabled", False):
        return

    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    payload = getattr(event, "payload", None) or {}
    drive = str(payload.get("drive") or "").strip()
    if not drive:
        # Older publishers that don't include the drive field — Phase 3a
        # transition fallback. Default to rest to mirror the old wrapper.
        drive = "rest"

    if drive not in drives.DRIVE_NAMES:
        log.debug("apply_signal_unknown_drive", drive=drive)
        return

    await drives.satiate(runtime, user_id=owner, drive=drive)
    ctx.cite("companion_drive_state", row_id=owner)
    ctx.db_ops += 2  # 1 SELECT in load() + 1 UPDATE in satiate()
    log.debug(
        "apply_signal_satiated",
        user_id=owner,
        drive=drive,
        kind=payload.get("kind"),
    )


VerbRegistry.register(apply_signal)
