"""emit_pad_if_delta — third Phase 3a management verb.

Subscribes to ``time.tick(60s)`` and projects the current PAD
(pleasure / arousal / dominance) affect coordinate from recent facet
activations and the role vector, then publishes ``affect.pad`` and
``state.delta_threshold_crossed`` if the delta from the last
published value crosses the noticeable-shift threshold.

Phase 3a is a "pure rename" — this is the
``_maybe_emit_pad`` body from ``behavior/tick.py`` lifted into a
verb. Two consumer events are emitted because the substrate is
mid-migration:

* ``affect.pad`` — the legacy event existing UI/state consumers
  subscribe to. Payload unchanged.
* ``state.delta_threshold_crossed`` — the new substrate primitive
  the verb architecture standardizes. Payload includes the affect
  field name and the delta magnitude so future verbs (e.g.
  ``narrate_state_to_user``) can subscribe to one event source
  rather than three module-specific ones.

Why a 60s tick rather than the 5-30s tick loop: PAD reflects a
60-minute window of facet activations. Sampling more often than
once per minute is wasted SELECTs — the value can't materially
change between adjacent samples. The verb-log cooldown sits just
under the tick interval so a normal cycle always fires.
"""

from __future__ import annotations

from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Match the legacy thresholds in behavior/tick.py. Tunable via settings
# if these prove too aggressive or too quiet in production — both axes
# share the same threshold for now.
_PAD_VALENCE_DELTA_THRESHOLD = 0.15
_PAD_AROUSAL_DELTA_THRESHOLD = 0.15

# Cooldown sits just under the 60s tick interval so a normal ladder
# cycle always fires, but a duplicate publish within the same second
# is silently coalesced via verb_log.
_PAD_COOLDOWN_MS = 55_000


def _parse_role(snap: dict) -> tuple[float, float]:
    """Pull active/passive role weights out of the state snapshot.

    Mirrors the parser previously inlined in ``_maybe_emit_pad``.
    """
    role_str = str(snap.get("role", "") or "")
    role_active = 0.0
    role_passive = 1.0
    for part in role_str.split("|"):
        if ":" not in part:
            continue
        name, val = part.split(":", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        if name == "active":
            role_active = v
        elif name == "passive":
            role_passive = v
    return role_active, role_passive


@verb(
    "time.tick(60s)",
    name="emit_pad_if_delta",
    reads=("personality_facet_activations", "companion_affect_baselines"),
    writes=(),  # No DB writes — caches on runtime._last_pad in-process.
    dispatch_class=DispatchClass.TICK_ALIGNED,
    safety_class=SafetyClass.READ,
    cooldown_ms=_PAD_COOLDOWN_MS,
)
async def emit_pad_if_delta(event, ctx) -> None:
    """Compute PAD and emit threshold events if noticeable delta seen.

    No-ops cleanly when the kill-switch is off, no owner is bound, or
    the memory backend isn't attached yet (cold-boot window).
    """
    if not getattr(settings, "companion_pad_emit_enabled", False):
        return

    runtime = ctx.runtime
    user_id = getattr(runtime, "owner_user_id", "") or ""
    if not user_id:
        return

    backend = getattr(getattr(runtime, "memory", None), "_backend", None)
    if backend is None:
        # Memory subsystem not attached yet — cold-boot window before
        # lifespan completes. Skip silently.
        return

    try:
        snap = runtime.state.snapshot()
    except Exception:
        snap = {}
    role_active, role_passive = _parse_role(snap)

    from augmentum.companion_runtime.perception.pad import project_pad
    pad = await project_pad(
        backend,
        user_id=user_id,
        companion_id=runtime.companion_id,
        role_active=role_active,
        role_passive=role_passive,
    )

    ctx.cite("personality_facet_activations", row_id=user_id)
    ctx.db_ops += 1  # project_pad runs a single SELECT in practice.

    last = getattr(runtime, "_last_pad", None)
    dv = abs(pad.valence - last.valence) if last is not None else float("inf")
    da = abs(pad.arousal - last.arousal) if last is not None else float("inf")
    crossed = (
        dv >= _PAD_VALENCE_DELTA_THRESHOLD
        or da >= _PAD_AROUSAL_DELTA_THRESHOLD
    )
    if not crossed:
        return

    runtime._last_pad = pad

    # Legacy event — UI / state-driver consumers still listen here.
    await ctx.emit(
        "affect.pad",
        {
            "valence": round(pad.valence, 3),
            "arousal": round(pad.arousal, 3),
            "dominance": round(pad.dominance, 3),
            "sample_count": pad.sample_count,
        },
    )

    # New substrate primitive — future verbs (narrate_state_to_user,
    # propose_action) subscribe here rather than threading the
    # field-specific affect.pad / drive.changed / energy.shifted
    # events into one bespoke aggregator.
    #
    # Carry both the absolute PAD coords and the deltas. Translation
    # verbs need the absolutes to pick a register label; observers
    # care about the deltas to gauge "how big a shift."
    await ctx.emit(
        "state.delta_threshold_crossed",
        {
            "field": "affect.pad",
            "valence": round(pad.valence, 3),
            "arousal": round(pad.arousal, 3),
            "dominance": round(pad.dominance, 3),
            "valence_delta": round(dv, 3) if dv != float("inf") else None,
            "arousal_delta": round(da, 3) if da != float("inf") else None,
        },
    )

    log.debug(
        "emit_pad_if_delta_fired",
        user_id=user_id,
        valence=round(pad.valence, 3),
        arousal=round(pad.arousal, 3),
        dv=round(dv, 3) if dv != float("inf") else None,
        da=round(da, 3) if da != float("inf") else None,
    )


VerbRegistry.register(emit_pad_if_delta)
