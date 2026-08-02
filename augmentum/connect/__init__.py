"""Connect mode — user-to-user voice/video calls + text threads.

This package owns the application-level pieces of Connect Phase 1.
Network-level WebRTC + TURN relay lives in
``augmentum/calling/`` (shared with game-stream).

Sub-modules:

* ``protocol``   — signaling envelope types + (de)serialisation
* ``hub``        — in-process WS connection registry + presence

The HTTP/WS surface for these primitives lives at
``augmentum/proxy/connect_routes.py``. Schema for persistent state
(contacts, sessions, threads, messages) is in migration 219.

See ``docs/superpowers/specs/2026-06-01-connect-and-os-positioning-design.md``
for the broader design.
"""

from __future__ import annotations

from .hub import ConnectHub
from .contacts import (
    THIS_INSTANCE_SENTINEL,
    ResolvedPeer,
    instance_handle,
    is_local_instance,
    local_did_for,
    resolve_peer_did,
)
from .protocol import (
    CALL_PROTOCOL_VERSION,
    DEFAULT_INVITE_LIFETIME_MS,
    EVENT_ACCEPT,
    EVENT_ANSWER,
    EVENT_CANDIDATES,
    EVENT_DECLINE,
    EVENT_ERROR,
    EVENT_HANGUP,
    EVENT_INVITE,
    EVENT_MUTE_STATE,
    EVENT_NEGOTIATE,
    EVENT_OFFER,
    EVENT_PONG,
    EVENT_PRESENCE_UPDATE,
    EVENT_SELECT_ANSWER,
    EVENT_TEXT_DELETE,
    EVENT_TEXT_DELIVERED,
    EVENT_TEXT_EDIT,
    EVENT_TEXT_REACT,
    EVENT_TEXT_READ,
    EVENT_TEXT_RECEIVED,
    EVENT_TYPING_START,
    EVENT_TYPING_STOP,
    EVENT_WELCOME,
    MAX_ENVELOPE_BYTES,
    MSG_ACCEPT,
    MSG_ANSWER,
    MSG_CANDIDATES,
    MSG_DECLINE,
    MSG_HANGUP,
    MSG_INVITE,
    MSG_MUTE_STATE,
    MSG_NEGOTIATE,
    MSG_OFFER,
    MSG_PING,
    MSG_PRESENCE_ANNOUNCE,
    MSG_SELECT_ANSWER,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_REACT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
    ConnectEnvelope,
    deserialise_envelope,
    serialise_envelope,
)

__all__ = [
    "CALL_PROTOCOL_VERSION",
    "ConnectEnvelope",
    "ConnectHub",
    "DEFAULT_INVITE_LIFETIME_MS",
    "EVENT_ACCEPT",
    "EVENT_ANSWER",
    "EVENT_CANDIDATES",
    "EVENT_DECLINE",
    "EVENT_ERROR",
    "EVENT_HANGUP",
    "EVENT_INVITE",
    "EVENT_MUTE_STATE",
    "EVENT_NEGOTIATE",
    "EVENT_OFFER",
    "EVENT_PONG",
    "EVENT_PRESENCE_UPDATE",
    "EVENT_SELECT_ANSWER",
    "EVENT_TEXT_DELETE",
    "EVENT_TEXT_DELIVERED",
    "EVENT_TEXT_EDIT",
    "EVENT_TEXT_REACT",
    "EVENT_TEXT_READ",
    "EVENT_TEXT_RECEIVED",
    "EVENT_TYPING_START",
    "EVENT_TYPING_STOP",
    "EVENT_WELCOME",
    "MAX_ENVELOPE_BYTES",
    "MSG_ACCEPT",
    "MSG_ANSWER",
    "MSG_CANDIDATES",
    "MSG_DECLINE",
    "MSG_HANGUP",
    "MSG_INVITE",
    "MSG_MUTE_STATE",
    "MSG_NEGOTIATE",
    "MSG_OFFER",
    "MSG_PING",
    "MSG_PRESENCE_ANNOUNCE",
    "MSG_SELECT_ANSWER",
    "MSG_TEXT_DELETE",
    "MSG_TEXT_DELIVERED",
    "MSG_TEXT_EDIT",
    "MSG_TEXT_REACT",
    "MSG_TEXT_READ",
    "MSG_TEXT_SEND",
    "MSG_TYPING_START",
    "MSG_TYPING_STOP",
    "ResolvedPeer",
    "THIS_INSTANCE_SENTINEL",
    "deserialise_envelope",
    "instance_handle",
    "is_local_instance",
    "local_did_for",
    "resolve_peer_did",
    "serialise_envelope",
]
