"""propose_action — second Phase 3c management verb.

Subscribes to ``state.delta_threshold_crossed`` and translates a
substrate signal into a proposed core-verb invocation. The proposal
is emitted as ``companion.action_proposed`` on the bus; downstream
consumers (Phase 4 notify / recommend_now / the existing initiative
queue UI) can subscribe to act on it.

This is the second half of the translation layer. ``narrate_state
_to_user`` answers "say something about it"; ``propose_action``
answers "do something about it." Both feed off the same substrate
event so the model never has to compute "what should I be doing
right now" from raw drive state at output time — the runtime has
already proposed the candidate.

Phase 3c minimal logic:

  1. On a substrate threshold cross, load current drive urgencies.
  2. Pick the most urgent drive (if any is above the propose floor).
  3. Reverse-look up the activity kind that satiates it (mirrors
     ``activity_selector._CANDIDATE_DRIVES``, inverted).
  4. Emit ``companion.action_proposed`` with kind + drive + urgency.

No notification side-effect; pure substrate-to-substrate. Phase 4
verbs will pick this up.

Gating:
  - ``companion_propose_action_enabled`` — kill switch, default False.
  - Owner must be resolved.
  - Cooldown via verb_log keeps proposals to at most one per 10 min
    (more frequent overwhelms the consumer's UI surface).
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


_PROPOSE_COOLDOWN_MS = 10 * 60 * 1000
_PROPOSE_FLOOR_URGENCY = 0.65


# Inverse of activity_selector._CANDIDATE_DRIVES — given a drive,
# which activity kind would satiate it? When multiple kinds map to
# the same drive (connection ← journal/reach_out), pick the more
# externally-oriented one as the proposal default.
_DRIVE_TO_PROPOSED_KIND: dict[str, str] = {
    "curiosity": "revisit_thread",
    "competence": "creation",
    "connection": "reach_out",
    "rest": "no_op",
}


@verb(
    "state.delta_threshold_crossed",
    name="propose_action",
    reads=("companion_drive_state",),
    writes=(),  # No DB writes — pure substrate-to-event translation.
    dispatch_class=DispatchClass.EVENT_DRIVEN,
    safety_class=SafetyClass.READ,
    cooldown_ms=_PROPOSE_COOLDOWN_MS,
)
async def propose_action(event, ctx) -> None:
    """Pick the most urgent drive and emit a proposed-action event."""
    from augmentum.config import settings
    if not getattr(settings, "companion_propose_action_enabled", False):
        return

    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    state = await drives.load(runtime, user_id=owner)
    ctx.cite("companion_drive_state", row_id=owner)
    ctx.db_ops += 1

    # Pick the drive with the highest urgency above the floor. Tie-break
    # by name for determinism (matches DriveState.dominant()).
    best_drive = None
    best_urgency = _PROPOSE_FLOOR_URGENCY
    for name in drives.DRIVE_NAMES:
        u = state.urgency(name)
        if u > best_urgency:
            best_drive = name
            best_urgency = u

    if best_drive is None:
        log.debug("propose_action_below_floor",
                  user_id=owner, floor=_PROPOSE_FLOOR_URGENCY)
        return

    proposed_kind = _DRIVE_TO_PROPOSED_KIND.get(best_drive, "no_op")
    payload_in = getattr(event, "payload", None) or {}
    await ctx.emit(
        "companion.action_proposed",
        {
            "kind": proposed_kind,
            "drive": best_drive,
            "urgency": round(best_urgency, 3),
            "trigger_field": payload_in.get("field"),
        },
    )
    log.debug(
        "propose_action_emitted",
        user_id=owner,
        kind=proposed_kind,
        drive=best_drive,
        urgency=round(best_urgency, 3),
    )


VerbRegistry.register(propose_action)
