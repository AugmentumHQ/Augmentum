"""tick_energy — fourth Phase 3b management verb.

Subscribes to ``time.tick(60s)`` and exponentially decays the
companion energy level toward its baseline via :func:`energy.decay`.
Mirrors :mod:`tick_drive` in structure — the difference is the
substrate (single ``energy_level`` scalar) and the cadence's
semantic (capacity-to-act, not drive appetite).

This is a "build new substrate" Phase 3b verb: migration 248 added
``companion_energy_state``; :mod:`energy` is the per-row read/write
module; this verb is the only writer in Phase 3b. A future
``apply_signal(energy)`` verb subscribed to ``behavior
.activity_chosen`` will spend energy on non-trivial activity (the
``energy.spend`` seam exists for that).

No-ops cleanly when no owner is bound — same shape as tick_drive.
"""

from __future__ import annotations

from augmentum.companion_runtime import energy
from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_TICK_ENERGY_COOLDOWN_MS = 55_000


@verb(
    "time.tick(60s)",
    name="tick_energy",
    reads=("companion_energy_state",),
    writes=("companion_energy_state",),
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.WRITE_SELF,
    cooldown_ms=_TICK_ENERGY_COOLDOWN_MS,
)
async def tick_energy(event, ctx) -> None:
    """Apply exponential decay to the energy level."""
    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    new_state = await energy.decay(runtime, user_id=owner)
    ctx.cite("companion_energy_state", row_id=owner)
    ctx.db_ops += 2  # 1 SELECT in load() + 1 UPDATE in decay()
    log.debug(
        "tick_energy_decayed",
        user_id=owner,
        level=round(new_state.level, 3),
        baseline=round(new_state.baseline, 3),
    )


VerbRegistry.register(tick_energy)
