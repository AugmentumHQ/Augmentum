"""spend_energy — the energy counterpart to ``apply_signal``.

Subscribes to ``behavior.activity_chosen`` and spends energy when a
NON-REST activity fires — the capacity cost of doing something outward.
Rest-tagged activities (``no_op`` / ``scene_update`` / ``dream_invocation``)
are how she recovers and cost nothing. ``tick_energy`` regenerates the level
toward baseline between spends; together they produce the
act -> deplete -> rest -> recover duty cycle.

This is the seam ``energy.spend`` was built for (see ``energy.py`` docstring
and ``tick_energy.py``). It mirrors ``apply_signal`` (the drive-satiation
subscriber) one-for-one, so energy stays a separate, separately-gated concern
from drives — just as ``tick_energy`` is separate from ``tick_drive``.

INVARIANT: energy shapes only what she INITIATES (this autonomous path),
never how she responds when addressed. The structural wall for that is
``tests/test_responsiveness_invariant.py``.

Gating mirrors ``apply_signal``:
  - ``companion_energy_enabled`` — kill switch, default False.
  - ``runtime.owner_user_id`` resolved — no per-user row otherwise.
  - non-rest ``drive`` in the event payload — rest activities never spend.
"""

from __future__ import annotations

from augmentum.companion_runtime import energy
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
    name="spend_energy",
    reads=("companion_energy_state",),
    writes=("companion_energy_state",),
    dispatch_class=DispatchClass.EVENT_DRIVEN,
    safety_class=SafetyClass.WRITE_SELF,
    # No cooldown — the spend should fire exactly when an activity is chosen,
    # mirroring apply_signal. Activity cadence is already gated upstream by the
    # tick loop + role_channel + the utility threshold.
    cooldown_ms=0,
)
async def spend_energy(event, ctx) -> None:
    """Spend energy on a non-rest activity; no-op on rest / when gated off."""
    if not getattr(settings, "companion_energy_enabled", False):
        return

    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    payload = getattr(event, "payload", None) or {}
    drive = str(payload.get("drive") or "rest").strip() or "rest"
    if drive == "rest":
        # Rest-tagged activities (no_op / scene_update / dream_invocation) are
        # recovery, not exertion — they cost nothing. Mirrors the rest-exempt
        # damping in activity_selector.choose, so the two stay symmetric.
        return

    await energy.spend(runtime, user_id=owner)
    ctx.cite("companion_energy_state", row_id=owner)
    ctx.db_ops += 2  # 1 SELECT in load() + 1 UPDATE in spend()
    log.debug(
        "spend_energy_spent",
        user_id=owner,
        drive=drive,
        kind=payload.get("kind"),
        amount=energy.SPEND_AMOUNT,
    )


VerbRegistry.register(spend_energy)
