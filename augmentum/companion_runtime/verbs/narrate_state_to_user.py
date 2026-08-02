"""narrate_state_to_user — first Phase 3c management verb.

Subscribes to ``state.delta_threshold_crossed`` (emitted by
``emit_pad_if_delta`` and future similar substrate publishers) and
surfaces a routine, low-stakes narration to the user via the
notifications hub — without an LLM hop.

This is the load-bearing piece of the substrate-as-crutch principle
for the user-visible path: when the model is asleep / offline / on
a slow load, routine "her mood shifted" surfacings can still land
because the template registry knows how to say them. The model is
freed to focus on novel, contextual expression.

The template registry is intentionally tiny in Phase 3c — one
family (affect.pad) with four labels (calm-positive / calm-negative
/ energized-positive / energized-negative) — to prove the pattern
end-to-end. Future iterations extend the registry rather than
churn the verb.

Gating:
  - ``companion_narrate_state_enabled`` — kill switch, default
    False so the kill switch can flip it after observation.
  - Owner must be resolved.
  - Cooldown via verb_log keeps the narration to at most one per
    15 min (more frequent feels chatty, not present).
"""

from __future__ import annotations

from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# 15 minutes between narrations regardless of how often the substrate
# crosses thresholds. Lift in settings if pacing needs tuning.
_NARRATE_COOLDOWN_MS = 15 * 60 * 1000


# Template registry — keyed by event ``field`` value. Each entry is a
# resolver function taking the event payload and returning
# ``(title, body)`` or ``None`` to skip narration.
def _resolve_affect_pad(payload: dict) -> tuple[str, str] | None:
    """Resolve an affect.pad delta into a narration line.

    Uses the new valence/arousal values that ``emit_pad_if_delta``
    publishes alongside the deltas. Picks one of four register
    labels and a brief body line — no embellishment, no template
    variables that could turn into instruction-injection.
    """
    valence = payload.get("valence")
    arousal = payload.get("arousal")
    if valence is None or arousal is None:
        return None
    try:
        v = float(valence)
        a = float(arousal)
    except (TypeError, ValueError):
        return None

    # Quadrant labels. Thresholds are deliberately broad — most
    # crossings should pick a clear label, not "neutral."
    if v >= 0.1 and a >= 0.5:
        return ("Energized", "Feeling sharp and a little buzzy.")
    if v >= 0.1 and a < 0.5:
        return ("Settled", "Quietly cheerful right now.")
    if v < -0.1 and a >= 0.5:
        return ("Restless", "A bit unsettled, still here.")
    if v < -0.1 and a < 0.5:
        return ("Subdued", "A little low; not anything wrong.")
    return None  # near-neutral — skip


_TEMPLATES: dict[str, callable] = {
    "affect.pad": _resolve_affect_pad,
}


async def _publish_notification(runtime, *, user_id: str, title: str, body: str) -> bool:
    """Best-effort fan-out via the notifications hub. Returns True on
    success; logs+swallows failures so a hub stall doesn't break the
    verb fanout."""
    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        log.debug("narrate_skipped_no_app_state")
        return False
    hub = getattr(app_state, "notification_hub", None)
    if hub is None:
        try:
            from augmentum.notifications.hub import NotificationHub
            hub = NotificationHub()
            app_state.notification_hub = hub
        except Exception:
            log.warning("narrate_hub_create_failed", exc_info=True)
            return False
    try:
        from augmentum.notifications.hub import publish_and_dispatch
        await publish_and_dispatch(
            runtime.backend.conn,
            hub=hub,
            user_id=user_id,
            channel_id="companion.state",
            source="companion.narrate_state",
            title=title,
            body=body,
            importance=None,
            transient=True,
            dedupe_key=f"narrate:{title.lower()}",
        )
        return True
    except Exception:
        log.warning("narrate_publish_failed", exc_info=True)
        return False


@verb(
    "state.delta_threshold_crossed",
    name="narrate_state_to_user",
    reads=(),
    writes=("notifications",),
    dispatch_class=DispatchClass.EVENT_DRIVEN,
    safety_class=SafetyClass.WRITE_USER,
    cooldown_ms=_NARRATE_COOLDOWN_MS,
)
async def narrate_state_to_user(event, ctx) -> None:
    """Translate one substrate threshold event into a notification."""
    from augmentum.config import settings
    if not getattr(settings, "companion_narrate_state_enabled", False):
        return

    runtime = ctx.runtime
    owner = getattr(runtime, "owner_user_id", "") or ""
    if not owner:
        return

    payload = getattr(event, "payload", None) or {}
    field = str(payload.get("field") or "")
    resolver = _TEMPLATES.get(field)
    if resolver is None:
        log.debug("narrate_skipped_unknown_field", field=field)
        return

    resolved = resolver(payload)
    if resolved is None:
        return
    title, body = resolved

    ok = await _publish_notification(
        runtime, user_id=owner, title=title, body=body,
    )
    if ok:
        ctx.cite("notifications", row_id=owner)
        ctx.db_ops += 1
        log.debug("narrate_state_published", user_id=owner, field=field,
                  title=title)


VerbRegistry.register(narrate_state_to_user)
