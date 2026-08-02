"""embody_event - translate runtime substrate events into avatar intents.

The frontend owns animation selection through companion-animation-router.js.
This verb gives the autonomous runtime a single backend outlet into that
contract: emit ``behavior.animation_intent`` with semantic roles and pose verbs,
never direct avatar calls.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.event_bus import (
    DispatchClass,
    SafetyClass,
    verb,
)
from augmentum.companion_runtime.verbs import VerbRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, n))


def _signed(value: Any, default: float = 0.0) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(-1.0, min(1.0, n))


def _action_proposal_intent(payload: dict[str, Any]) -> dict[str, Any] | None:
    kind = str(payload.get("kind") or "").strip().lower()
    urgency = _clamp01(payload.get("urgency"), 0.5)

    by_kind: dict[str, dict[str, Any]] = {
        "reach_out": {
            "roles": ["attention-seek", "reach-out"],
            "pose_verb": "reach_out",
            "emotion": {"warmth": 0.76, "energy": 0.45, "openness": 0.78, "focus": 0.65},
        },
        "revisit": {
            "roles": ["curiosity", "question", "mirror-curious"],
            "pose_verb": "curious",
            "emotion": {"warmth": 0.58, "energy": 0.42, "openness": 0.72, "focus": 0.82},
        },
        "revisit_thread": {
            "roles": ["curiosity", "question", "mirror-curious"],
            "pose_verb": "curious",
            "emotion": {"warmth": 0.58, "energy": 0.42, "openness": 0.72, "focus": 0.82},
        },
        "creation": {
            "roles": ["think", "ponder"],
            "pose_verb": "thinking",
            "emotion": {"warmth": 0.52, "energy": 0.36, "openness": 0.58, "focus": 0.9},
        },
        "journal": {
            "roles": ["think", "ponder"],
            "pose_verb": "reflect",
            "emotion": {"warmth": 0.56, "energy": 0.3, "openness": 0.52, "focus": 0.88},
        },
        "no_op": {
            "roles": ["soften"],
            "pose_verb": "settle",
            "emotion": {"warmth": 0.58, "energy": 0.22, "openness": 0.48, "focus": 0.45},
        },
    }
    base = by_kind.get(kind)
    if base is None:
        if not kind:
            return None
        base = {
            "roles": ["listen", "attentive", "idle-shift"],
            "pose_verb": "attentive",
            "emotion": {"warmth": 0.58, "energy": 0.36, "openness": 0.62, "focus": 0.72},
        }

    out = dict(base)
    out.update({
        "source": f"verb:embody_event:action:{kind or 'unknown'}",
        "priority": "situational",
        "pose_duration_ms": int(4500 + urgency * 3500),
        "reason": "companion.action_proposed",
        "explicit": False,
    })
    return out


def _pad_delta_intent(payload: dict[str, Any]) -> dict[str, Any] | None:
    if str(payload.get("field") or "").strip().lower() != "affect.pad":
        return None

    valence = _signed(payload.get("valence"), 0.0)
    arousal = _clamp01(payload.get("arousal"), 0.4)
    warmth = _clamp01(0.5 + (valence * 0.35), 0.5)
    energy = _clamp01(arousal, 0.4)

    if arousal >= 0.55 and valence >= 0.18:
        roles = ["react-positive", "excitement-peak"]
        pose_verb = "confident"
    elif arousal >= 0.55 and valence <= -0.18:
        roles = ["react-negative", "mirror-anger"]
        pose_verb = "boundary"
    elif valence <= -0.25:
        roles = ["soften"]
        pose_verb = "concerned"
    elif valence >= 0.22:
        roles = ["gratitude", "agree", "soften"]
        pose_verb = "settle"
    else:
        roles = ["listen", "attentive", "idle-shift"]
        pose_verb = "attentive"

    return {
        "roles": roles,
        "pose_verb": pose_verb,
        "emotion": {
            "warmth": round(warmth, 3),
            "energy": round(energy, 3),
            "openness": round(_clamp01(0.55 + max(valence, 0.0) * 0.25), 3),
            "focus": round(_clamp01(0.55 + arousal * 0.25), 3),
        },
        "source": "verb:embody_event:affect.pad",
        "priority": "situational",
        "pose_duration_ms": 5200,
        "reason": "state.delta_threshold_crossed",
        "explicit": False,
    }


def animation_intent_for_event(topic: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a frontend animation-intent payload for a runtime event."""
    clean_topic = str(topic or "").strip()
    clean_payload = payload if isinstance(payload, dict) else {}
    if clean_topic == "companion.action_proposed":
        return _action_proposal_intent(clean_payload)
    if clean_topic == "state.delta_threshold_crossed":
        return _pad_delta_intent(clean_payload)
    return None


@verb(
    "companion.action_proposed",
    "state.delta_threshold_crossed",
    name="embody_event",
    reads=(),
    writes=(),
    dispatch_class=DispatchClass.EVENT_DRIVEN,
    safety_class=SafetyClass.READ,
    cooldown_ms=0,
)
async def embody_event(event, ctx) -> None:
    """Emit one semantic animation intent for an autonomous runtime event."""
    payload = animation_intent_for_event(event.topic, event.payload or {})
    if payload is None:
        return
    await ctx.emit(
        "behavior.animation_intent",
        payload,
        propagation=event.propagation,
    )
    log.debug(
        "embody_event_emitted",
        topic=event.topic,
        source=payload.get("source"),
        pose_verb=payload.get("pose_verb"),
    )


VerbRegistry.register(embody_event)


__all__ = ["animation_intent_for_event", "embody_event"]
