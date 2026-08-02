"""WebSocket message protocol between Augmentum and TV receivers.

Two message types, one envelope each:

  Server → Receiver: ``ReceiverCmd``
      {"type": "cmd", "cmd": "<verb>", "id": "<corr>", "args": {...}}

  Receiver → Server: ``ReceiverEvent``
      {"type": "event", "event": "<kind>", "id": "<corr>", "data": {...}}

Forward compatibility: unknown ``cmd`` or ``event`` strings are returned
as their literal value rather than raising. Subscribers / handlers
ignore unknowns by checking against the constants below — same model
the fabric capability system uses.

The wire form is JSON (not msgpack / binary) because:
  - TVs / receiver apps often run JS which deserialises JSON natively
  - Messages are sparse (~hundreds of bytes), so binary efficiency
    doesn't pay back the maintenance cost
  - Easy to inspect in browser devtools when debugging receiver apps
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ── Commands (server → receiver) ──────────────────────────────────


CMD_PLAY = "play"
CMD_PAUSE = "pause"
CMD_RESUME = "resume"
CMD_SEEK = "seek"
CMD_VOLUME = "volume"
CMD_STOP = "stop"
CMD_SHOW_IMAGE = "show_image"
CMD_SHOW_HTML = "show_html"
CMD_SHOW_ARTIFACT = "show_artifact"
CMD_ACK = "ack"

# Surface verbs (Phase A, see docs/superpowers/specs/2026-05-20-cast-
# surface-protocol.md). One verb per lifecycle event; modality lives
# in the ``surface_kind`` string, not in the verb. Receivers that
# don't understand a verb yet are forward-compat (they ignore it).
CMD_SURFACE_OPEN = "surface_open"
CMD_SURFACE_CLOSE = "surface_close"
CMD_SURFACE_FOCUS = "surface_focus"
CMD_SURFACE_STATE = "surface_state"

# Device-level (not surface-scoped) commands. ``CMD_SYSTEM_VOLUME``
# adjusts the receiver's Android system music-stream volume via the
# AugmentumTV JS bridge — only effective on the bundled Android TV
# APK; browser receivers ignore it. Args:
#   {volume: 0.0..1.0}  — set absolute level
#   {delta: int}         — adjust by N hardware steps
#   {muted: bool}        — toggle mute
CMD_SYSTEM_VOLUME = "system_volume"

# Server → receiver identity handshake. Sent once, immediately after
# WebSocket accept, so the receiver page knows its own registration_id
# (which it can't derive otherwise — the WS-scoped wsp_ token is
# different from the registry's persistent id). Receivers forward this
# id into surface_init payloads so iframed surfaces like cast-home can
# call back through /api/cast/send for "I tapped this tile, cast it
# to myself" interactions. Args:
#   {registration_id: str}
CMD_IDENTITY = "identity"

# Follow-mode navigation events from the controller. Mirrors the
# phone's current browsing view onto the cast-home idle surface so a
# group can see what the phone-holder is picking without seeing a
# stationary screen. Args (any subset, ``view`` decides the layout):
#   {view: "home"}                 — back to the default rails
#   {view: "section", slug: str}   — show paginated section grid
#   {view: "series", file_id: str} — show series detail + episode list
# ``follow`` enable/disable is a separate concern handled by the
# controller — when off, the controller just stops emitting.
CMD_NAV = "nav"

# Per-receiver display preferences just changed. Sent by the server
# after a successful PUT to /api/cast/trusted-receivers/{id}/prefs so
# the receiver page can adopt the new bag without waiting for its
# next page reload. Args:
#   {prefs: {rails_visible: {...}, backdrop_cycle: bool, ...}}
# Receivers cache the bag in module state and re-push it into the
# mounted main-slot iframe (cast-home filters rails based on it).
CMD_PREFS_CHANGED = "prefs_changed"

# Server tells the receiver "you've been revoked" right before closing
# the WS with code 4003. Without this hint the receiver only sees a
# bare close and falls into a re-pair loop — every reconnect rediscovers
# the same revoked row + closes again. Args:
#   {trusted_id: str, reason: "revoked"|"unknown"}
# Receivers should display a terminal placeholder pointing the user at
# Settings → TVs → Revoked → Restore rather than auto-retrying.
CMD_REVOKED = "revoked"

# Couch co-op invite overlay. Host taps "+ Players" on a game tile and
# the server pushes this to the active receiver so a QR appears over
# the streaming game. Receiver renders the QR + slots_remaining count.
# Sent again as joiners claim, so the slot counter stays in sync. Args:
#   {token: str, join_url: str, slots_remaining: int,
#    expires_at: float, slots_total: int}
CMD_SHOW_INVITE_QR = "show_invite_qr"

# Dismiss the invite overlay. Sent when the invite is revoked, the
# token expires, or all slots fill. Args:
#   {token: str, reason: "expired"|"full"|"revoked"|"session_ended"}
CMD_HIDE_INVITE_QR = "hide_invite_qr"

# Couch co-op player roster update. Sent whenever a phone attaches or
# detaches from a session so the receiver's player-chip strip stays in
# sync. Receiver displays P1..P4 chips with names + colours + status.
# Args:
#   {session_id: str,
#    players: [{slot: 0..3, name: str, color: str, status: "active"|"empty"}, ...],
#    slots_remaining: int}
CMD_INVITE_SLOT_UPDATE = "invite_slot_update"

# Browser-cast gamepad input. Phone-side cast-control attaches to the
# input WS with ?receiver_id=<id> instead of ?session_id=<id> when the
# target is a kiosk play surface on a receiver (rather than an AGSP
# container). Each gamepad_state frame from the phone fans out as one
# of these commands; the receiver shell forwards the args via
# postMessage to the currently-mounted play iframe, where the
# universal-input-adapter loader dispatches it through the active
# adapter chain (gamepad_api by default; keyboard / touch / pointer
# adapters can be activated per-game via CastProfile).
#
# Args shape:
#   {"slot": int, "pad_index": int,
#    "buttons": [float × 17], "axes": [float × 4]}
#
# Frequency is the phone's frame rate (typically 60 Hz) — receivers
# MUST handle these with no allocation in the hot path. The receiver
# shell's postMessage forward is the only allocation per frame.
CMD_INPUT_GAMEPAD = "input_gamepad"


# ── Events (receiver → server) ────────────────────────────────────


EVENT_READY = "ready"
EVENT_PLAYBACK_STARTED = "playback_started"
EVENT_PLAYBACK_PROGRESS = "playback_progress"
EVENT_PLAYBACK_ENDED = "playback_ended"
EVENT_PLAYBACK_ERROR = "playback_error"
EVENT_ACK = "ack"

# Surface lifecycle events emitted by the TV shell + surface code.
EVENT_SURFACE_OPENED = "surface_opened"
EVENT_SURFACE_CLOSED = "surface_closed"
EVENT_SURFACE_STATE = "surface_state"

# Device-level echo. Emitted by the receiver after a system_volume
# cmd applies so the phone-side controller can mirror the actual
# (possibly snapped to integer steps) level + mute state. Payload:
#   {volume: 0.0..1.0, muted: bool, supported: bool}
EVENT_SYSTEM_VOLUME_STATE = "system_volume_state"

# Cast input adapter telemetry. Emitted (~every 5s) by the universal
# input adapter loader inside a cast surface and relayed by the TV
# shell as a generic surface_event. The server's demotion loop
# (augmentum.cast.games.telemetry) reads it to decide whether the
# active strategy's adapter chain is actually reaching the game.
# Payload mirrors the loader's frame:
#   {adapter: str, frames_received: int, dispatches: int, window_ms: int}
EVENT_INPUT_TELEMETRY = "input_telemetry"


# ── Layout slots ──────────────────────────────────────────────────


# Five named slots cover ~95% of TV compositions. Single occupant per
# slot — opening a new surface in an occupied slot replaces the old.
# Adding more is feasible but ratchets composition complexity; resist.
SLOT_MAIN = "main"
SLOT_PIP = "pip"
SLOT_OVERLAY = "overlay"
SLOT_TICKER = "ticker"
SLOT_COMPANION = "companion"

SLOTS: tuple[str, ...] = (
    SLOT_MAIN, SLOT_PIP, SLOT_OVERLAY, SLOT_TICKER, SLOT_COMPANION,
)


def is_valid_slot(slot: str) -> bool:
    """True if ``slot`` is one of the canonical slot names. Forward-
    compat: callers should soft-fail on invalid slots rather than
    raise, so a newer server sending an unknown slot doesn't crash
    an older receiver path."""
    return slot in SLOTS


# ── Surface kinds (open namespace) ────────────────────────────────


# Concrete kinds shipping with Phase A. Receivers handle unknown
# kinds by falling back to generic iframe load (html.generic
# semantics). Adding a kind is one constant + a /ui/<surface>/ route;
# no protocol change required.
SURFACE_HTML = "html.generic"            # any URL in iframe
SURFACE_IMAGE = "media.image"             # static <img>
SURFACE_VIDEO = "media.video"             # native <video>
SURFACE_AUDIO = "media.audio"             # native <audio>
SURFACE_COMIC = "comic.reader"            # comic with TV-mode auto-scroll
SURFACE_VRM = "vrm.avatar"                # 3D avatar; server-render fallback for lite-tier
SURFACE_ARTIFACT = "artifact.presentation"  # docs/slides/charts via studio
SURFACE_LIBRARY_APP = "library.app"       # workspace app fullscreen
SURFACE_GAME_STREAM = "game.stream"       # game streaming session
SURFACE_MIRROR = "mirror.live"            # WebRTC mirror from source device
# Generic server-rendered WebRTC stream. Anything heavy that needs to
# render on the server and stream to a thin client — VRM companion on
# weak TVs, comic reader, notebook, browse panel. Powered by the
# browser-stream profile: server-side Chromium kiosk + Selkies-gstreamer.
# Receivers iframe the stream_path (Selkies' built-in viewer); no
# bespoke WebRTC client on the TV side.
SURFACE_STREAM = "stream.webrtc"


# ── Message types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ReceiverCmd:
    """A command the server sends to a receiver.

    ``id`` is an optional correlation id — receivers that emit an
    ``ack`` event echo this id so the server can pair the response
    to the original send.
    """

    cmd: str
    id: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReceiverEvent:
    """An event a receiver pushes to the server.

    For ``ready``: ``data`` carries device fingerprint info — platform,
    version, screen size, supported codec hints. The server stores it
    on the receiver registry entry for diagnostics + future routing.
    """

    event: str
    id: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ── Serialisation ─────────────────────────────────────────────────


def serialise_cmd(cmd: ReceiverCmd) -> dict[str, Any]:
    return {
        "type": "cmd",
        "cmd": cmd.cmd,
        "id": cmd.id,
        "args": dict(cmd.args or {}),
    }


def serialise_event(event: ReceiverEvent) -> dict[str, Any]:
    return {
        "type": "event",
        "event": event.event,
        "id": event.id,
        "data": dict(event.data or {}),
    }


def deserialise_cmd(raw: dict[str, Any] | str | bytes) -> ReceiverCmd | None:
    """Parse a wire-form cmd. Returns None on shape errors.

    Accepts dict (already parsed) or str/bytes (raw JSON). Tolerant of
    missing fields — only ``cmd`` is required.
    """
    data = _ensure_dict(raw)
    if data is None or data.get("type") != "cmd":
        return None
    cmd = str(data.get("cmd") or "")
    if not cmd:
        return None
    return ReceiverCmd(
        cmd=cmd,
        id=str(data.get("id") or ""),
        args=dict(data.get("args") or {}),
    )


def deserialise_event(raw: dict[str, Any] | str | bytes) -> ReceiverEvent | None:
    """Parse a wire-form event. Returns None on shape errors."""
    data = _ensure_dict(raw)
    if data is None or data.get("type") != "event":
        return None
    event = str(data.get("event") or "")
    if not event:
        return None
    return ReceiverEvent(
        event=event,
        id=str(data.get("id") or ""),
        data=dict(data.get("data") or {}),
    )


def _ensure_dict(raw: Any) -> dict[str, Any] | None:
    """Coerce raw input to dict. Strings / bytes get JSON-parsed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None
