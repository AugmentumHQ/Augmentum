"""media.{pause,next,previous} — universal playback control.

These primitives close the obvious gap left by ``media.resume``: a user
who can start playback by voice should also be able to pause, skip
forward, and skip back by voice. They're surface-agnostic on the
server side — each emits a unified ``media.transport`` channel; the
frontend's intent-action-router decides which of the three concurrent
playback systems (media-player audio, Grove music, ambient YouTube)
currently owns the foreground and dispatches there.

Why one channel for three actions: the underlying player APIs already
have pause/next/previous methods. The only thing differing between
the three primitives is which method to call, which is encoded in the
payload ``action`` field. Keeping them on one channel avoids three
near-identical case branches in intent-action-router.js.

All three are ``disruptive`` — they interrupt an in-progress audio
stream. Under the confidence-tier dispatch model that places them at
Tier B in Phase 2 (confirm-with-grace). Phase 1 force-promotes to
Tier A; the router can still elect to surface a confirm-prompt
response for ambiguous cases ("pause… everything?").
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _emit_transport(action: str, *, speak: str) -> ActionResult:
    """Shared emit helper — all three primitives produce the same shape."""
    return ActionResult(
        short_circuit=True,
        speak=speak,
        surface_emit={
            "channel": "media.transport",
            "payload": {"action": action},
        },
    )


# Receiver speak lines per action — headless control on a TV has no
# on-screen feedback from us, so the confirmation is verbal.
_RECEIVER_SPOKEN = {
    "pause": "Paused on {label}.",
    "next": "Skipped ahead on {label}.",
    "previous": "Went back on {label}.",
}


async def _transport_via_ladder(
    action: str,
    session: SessionContext,
    args: dict[str, Any] | None,
    *,
    fallback_speak: str = "",
) -> ActionResult:
    """Receiver ladder first, in-tab surface emit as the floor.

    Wiring program Phase 1 (2026-06-12): when the user's playback is a
    cast session, "pause it" must hit the receiver — emitting to the
    tab pauses nothing they can hear. A named device wins; an active
    session is preferred; the tab keeps its existing behavior when
    nothing is cast. A device hint with no session still routes to the
    receiver (DLNA/cast transports accept commands regardless of who
    started playback).
    """
    args = args or {}
    from augmentum.intent.media_devices import (
        resolve_playback_target,
        transport_on_target,
    )

    ladder = await resolve_playback_target(
        getattr(session, "app_state", None),
        session.user_id,
        device_hint=str(args.get("device") or ""),
    )
    if ladder.miss:
        return ActionResult(
            short_circuit=True,
            speak=f"I don't see a device called {ladder.miss[:60]}.",
        )
    if ladder.clarify:
        return ActionResult(
            short_circuit=True,
            speak=ladder.clarify,
            clarify={"missing": ["device"], "args": dict(args)},
        )
    had_hint = bool(str(args.get("device") or "").strip())
    if ladder.target is not None and (ladder.target.session_id or had_hint):
        res = await transport_on_target(
            getattr(session, "app_state", None), session.user_id,
            ladder.target, action,
        )
        label = ladder.target.device_label
        if res.ok:
            log.info(
                "media_transport_receiver",
                user_id=session.user_id, action=action,
                device_id=ladder.target.device_id,
            )
            return ActionResult(
                short_circuit=True,
                speak=_RECEIVER_SPOKEN.get(action, "").format(label=label),
                digest=f"{action} on {label}",
            )
        if res.code in ("unsupported_action", "unknown_action"):
            # No driver has a queue — honest beats silently pausing
            # the wrong (in-tab) playback instead.
            return ActionResult(
                short_circuit=True,
                speak=(
                    f"The cast on {label} doesn't have a queue I can "
                    "skip through — I can stop it or play something "
                    "else instead."
                ),
            )
        log.warning(
            "media_transport_receiver_failed",
            user_id=session.user_id, action=action,
            device_id=ladder.target.device_id, code=res.code,
        )
        return ActionResult(
            short_circuit=True,
            speak=f"{label} didn't take that — it may have disconnected.",
        )
    return await _emit_transport(action, speak=fallback_speak)


# --------------------------------------------------------------------- pause


async def _media_pause_handler(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    log.info("architect_media_pause", user_id=session.user_id)
    return await _transport_via_ladder("pause", session, args)


# Optional device name, shared by all three transport verbs — when
# the user names a TV/receiver ("pause the living room TV") the ladder
# routes the command there instead of the in-tab player.
_DEVICE_ARG = {
    "device": {
        "type": "string",
        "description": (
            "Device name if the user names one ('the living room TV'). "
            "Omit to control whatever is actively playing."
        ),
    },
}


register_action(
    id="media.pause",
    summary=(
        "Pause whatever is currently playing — audiobook, podcast, "
        "music, video, or ambient audio — on the casting device when "
        "something is playing on a TV/receiver, otherwise the in-tab "
        "player."
    ),
    arg_schema=dict(_DEVICE_ARG),
    examples=[
        "pause",
        "pause it",
        "pause that",
        "pause the audiobook",
        "pause the music",
        "stop the music",
        "hold on",
    ],
    handler=_media_pause_handler,
    delivery="artifact",
    surfaces=["becca", "chat"],
    stakes="disruptive",
    templates=[
        # Bare "pause" hits Tier 0 control verbs in some surfaces;
        # these templates target the explicit "pause X" / "pause that"
        # natural forms that wouldn't fire as a universal control.
        "[hey] [please] (pause|hold) [(the|that|it|my)] [(music|audiobook|podcast|video|book|show)]",
        # "stop the music" is conversationally distinct from the
        # universal ``stop`` (which kills TTS, not playback). Anchor
        # on the object so we don't poach the Tier 0 control.
        "(stop|kill) (the|that|my) (music|audiobook|podcast|video|book|playback)",
    ],
)


# --------------------------------------------------------------------- next


async def _media_next_handler(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    log.info("architect_media_next", user_id=session.user_id)
    return await _transport_via_ladder("next", session, args)


register_action(
    id="media.next",
    summary=(
        "Skip to the next track / chapter / episode in whatever is "
        "currently playing."
    ),
    arg_schema=dict(_DEVICE_ARG),
    examples=[
        "next track",
        "skip this",
        "next chapter",
        "next song",
        "play the next one",
        "skip forward",
    ],
    handler=_media_next_handler,
    delivery="artifact",
    surfaces=["becca", "chat"],
    stakes="disruptive",
    templates=[
        "(next|skip) [the] (track|song|chapter|episode|video|book)",
        "(skip|play) [the] next [(one|track|song|chapter|episode)]",
        "skip (this|that) [(track|song|chapter|episode)]",
        # "skip forward" — distinct from seek; treat as a track skip
        # in the absence of a duration. Future: media.skip with a
        # duration arg for seek behavior.
        "skip forward",
    ],
)


# --------------------------------------------------------------------- previous


async def _media_previous_handler(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    log.info("architect_media_previous", user_id=session.user_id)
    return await _transport_via_ladder("previous", session, args)


register_action(
    id="media.previous",
    summary=(
        "Go back to the previous track / chapter / episode in whatever "
        "is currently playing."
    ),
    arg_schema=dict(_DEVICE_ARG),
    examples=[
        "previous track",
        "go back",
        "previous chapter",
        "last song",
        "play the previous one",
        "back to the last song",
    ],
    handler=_media_previous_handler,
    delivery="artifact",
    surfaces=["becca", "chat"],
    stakes="disruptive",
    templates=[
        "(previous|last) [the] (track|song|chapter|episode|video)",
        "(go|play) [the] (previous|last) [(one|track|song|chapter|episode)]",
        "back to (the|that) (last|previous) (track|song|chapter|episode)",
        "go back [(a|one)] (track|song|chapter|episode)",
    ],
)
