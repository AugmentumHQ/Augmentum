"""Receiver resolution ladder for mid-experience media control.

The companion can start playback anywhere, but until 2026-06-12 she
could not touch it afterwards — volume, pause-on-the-TV, skip — even
though the device control plane (`/api/devices/{id}/{capability}/
{action}` → DeviceRegistry.invoke → DLNA / Chromecast / Emby remote)
was fully built. This module is the verb-side bridge: it resolves
"which playback does the user mean" and performs the control call
server-side, so the verbs stay thin.

Resolution ladder (companion wiring program Phase 1):

1. **Device hint wins.** "turn down the living room TV" — match the
   hint against device labels; an active session on that device is
   preferred, but a saved device without one still accepts volume.
2. **Active receiver session.** Exactly one media cast session →
   that's the playback they mean (the receiver beats the tab when
   both play; in-tab playback is what the user is *sitting at*, so
   if they wanted that they'd more likely use the on-screen control).
3. **Multiple sessions** → clarify with the device names; the
   dispatch layers park it so the answer fills the slot.
4. **Nothing cast** → empty result; the verb falls through to a
   surface emit and the in-tab foreground player handles it.

All calls are user-scoped through the registry; nothing here touches
the device store directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# One "volume up" step, in percent. Matches typical remote-control
# feel (10 presses bottom-to-top).
VOLUME_STEP = 10

_DEFAULT_CAPABILITY = "media.audio_play@1"


@dataclass(slots=True)
class PlaybackTarget:
    """A resolved receiver-side playback target."""

    device_id: str
    device_label: str
    capability_id: str = _DEFAULT_CAPABILITY
    session_id: str = ""
    title: str = ""  # what's playing, for speak lines


@dataclass(slots=True)
class LadderResult:
    """Outcome of resolution — exactly one field is meaningful.

    ``target`` set → control that receiver. ``clarify`` set → ask the
    question (dispatcher parks it). ``miss`` set → the named device
    doesn't exist; speak an honest miss. All empty → in-tab fallback.
    """

    target: PlaybackTarget | None = None
    clarify: str = ""
    miss: str = ""


def _registry(app_state: Any):
    return getattr(app_state, "device_registry", None) if app_state else None


def _media_capability(device: Any) -> str:
    for cap in getattr(device, "capabilities", None) or []:
        if str(cap).startswith("media."):
            return str(cap)
    return _DEFAULT_CAPABILITY


async def resolve_playback_target(
    app_state: Any, user_id: str, device_hint: str = "",
) -> LadderResult:
    """Resolve which receiver (if any) a media-control verb should hit."""
    reg = _registry(app_state)
    if reg is None or not user_id:
        return LadderResult()

    try:
        sessions = await reg.list_sessions(user_id=user_id)
    except Exception:  # noqa: BLE001
        log.warning("media_ladder_sessions_failed", exc_info=True)
        sessions = []
    media_sessions = [
        s for s in sessions
        if str(getattr(s, "capability_id", "")).startswith("media.")
    ]

    devices: list[Any] = []
    try:
        devices = await reg.list(user_id=user_id)
    except Exception:  # noqa: BLE001
        log.warning("media_ladder_devices_failed", exc_info=True)
    by_id = {d.id: d for d in devices}

    def _label(device_id: str) -> str:
        d = by_id.get(device_id)
        return (getattr(d, "label", "") or device_id) if d is not None else device_id

    hint = (device_hint or "").strip().lower()
    if hint:
        def _match(label: str) -> bool:
            low = (label or "").lower()
            return bool(low) and (hint in low or low in hint)

        for s in media_sessions:
            if _match(_label(s.device_id)):
                return LadderResult(target=PlaybackTarget(
                    device_id=s.device_id,
                    device_label=_label(s.device_id),
                    capability_id=s.capability_id,
                    session_id=s.id,
                    title=getattr(s, "title", "") or "",
                ))
        for d in devices:
            if _match(getattr(d, "label", "")):
                return LadderResult(target=PlaybackTarget(
                    device_id=d.id,
                    device_label=getattr(d, "label", "") or d.id,
                    capability_id=_media_capability(d),
                ))
        return LadderResult(miss=(device_hint or "").strip())

    if len(media_sessions) == 1:
        s = media_sessions[0]
        return LadderResult(target=PlaybackTarget(
            device_id=s.device_id,
            device_label=_label(s.device_id),
            capability_id=s.capability_id,
            session_id=s.id,
            title=getattr(s, "title", "") or "",
        ))
    if len(media_sessions) > 1:
        names = list(dict.fromkeys(_label(s.device_id) for s in media_sessions))
        return LadderResult(clarify=f"Which one — {', or '.join(names)}?")
    return LadderResult()


async def transport_on_target(
    app_state: Any, user_id: str, target: PlaybackTarget, action: str,
) -> Any:
    """Send pause/resume/stop/next/previous to a resolved receiver.

    Returns the registry's InvocationResult (``.ok`` / ``.code`` /
    ``.message``). Drivers that lack a queue return
    ``unsupported_action`` for next/previous — callers speak that
    honestly rather than silently routing to the tab.
    """
    reg = _registry(app_state)
    return await reg.invoke(
        user_id=user_id,
        device_id=target.device_id,
        capability=target.capability_id or _DEFAULT_CAPABILITY,
        action=action,
        args={},
    )


async def volume_on_target(
    app_state: Any,
    user_id: str,
    target: PlaybackTarget,
    *,
    direction: str,
    level: int | None = None,
) -> tuple[bool, str]:
    """Adjust volume/mute on a resolved receiver.

    Returns ``(ok, detail)`` — on success ``detail`` is the resulting
    level as a string ("" for mute/unmute); on failure it's an honest
    one-line reason suitable for speech.
    """
    reg = _registry(app_state)
    cap = target.capability_id or _DEFAULT_CAPABILITY

    if direction in ("mute", "unmute"):
        res = await reg.invoke(
            user_id=user_id, device_id=target.device_id, capability=cap,
            action="set_mute", args={"muted": direction == "mute"},
        )
        if not res.ok:
            log.warning(
                "media_ladder_mute_failed",
                device_id=target.device_id, code=res.code, message=res.message,
            )
            return False, f"{target.device_label} didn't take the mute command"
        return True, ""

    if direction == "set":
        new_level = max(0, min(100, int(level if level is not None else 50)))
    else:
        # up/down need the current level; the capability snapshot is the
        # only driver-agnostic way to read it (DLNA GetVolume, cast
        # status, Emby session state all normalize to volume_level).
        snap = await reg.snapshot(
            user_id=user_id, device_id=target.device_id, capability=cap,
        ) or {}
        current = snap.get("volume_level")
        if current is None:
            return False, (
                f"I couldn't read the current volume on "
                f"{target.device_label} to step it"
            )
        delta = VOLUME_STEP if direction == "up" else -VOLUME_STEP
        new_level = max(0, min(100, int(current) + delta))

    res = await reg.invoke(
        user_id=user_id, device_id=target.device_id, capability=cap,
        action="set_volume", args={"level": new_level},
    )
    if not res.ok:
        log.warning(
            "media_ladder_volume_failed",
            device_id=target.device_id, code=res.code, message=res.message,
        )
        return False, f"{target.device_label} didn't take the volume change"
    return True, str(new_level)
