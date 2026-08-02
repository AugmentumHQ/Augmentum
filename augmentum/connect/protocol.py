"""Connect signaling envelope — wire protocol between UI and signaling hub.

Two envelope flavours, both JSON over WebSocket:

  Client → Server  (msg)
      {"type": "msg", "msg": "<verb>", "id": "<corr>", "to": "<peer_did>",
       "data": {...}}

  Server → Client  (event)
      {"type": "event", "event": "<kind>", "id": "<corr>", "from": "<peer_did>",
       "data": {...}}

The shape mirrors ``augmentum/cast/receiver_protocol.py`` so the two
WS-driven subsystems stay coherent (cmd/event for cast, msg/event for
Connect — the only difference is direction: cast commands go to TVs;
Connect msgs flow between user peers).

Forward compatibility: unknown verbs / event kinds round-trip as
their literal value rather than raising. Handlers ignore unknowns by
comparing against the constants below.

Why JSON not msgpack: signaling messages are sparse (~hundreds of
bytes), the UI deserialises JSON natively, and debugging in browser
devtools is night-and-day easier when payloads are human-readable.
The media itself (the SRTP-encrypted RTP) bypasses this protocol
entirely — only SDP/ICE metadata goes through here.

Call-state contract (inside ``data`` for any call-related verb):

    {
        "call_id":   "<opaque>",   # initiator-minted; same on both ends
        "party_id":  "<8 alnum>",  # PER-CONNECTION; lets multi-device
                                   # routing tell sibling devices apart
                                   # — see Matrix MSC2746
        "version":   1,            # protocol version of THIS message
                                   # — forward-compat insurance
        "lifetime":  60000,        # invite only: ms until auto-expire
        ...                        # verb-specific payload
    }

These fields are validated by the routing layer, not the envelope
codec — the wire shape stays tolerant so a newer client can speak
to an older server without protocol breakage on transport.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ── Constants ────────────────────────────────────────────────────


CALL_PROTOCOL_VERSION = 1
"""Wire-protocol version. Bumped on any breaking call-state shape
change so a router can route by ``data["version"]`` for back-compat."""

DEFAULT_INVITE_LIFETIME_MS = 60_000
"""How long an unanswered invite stays valid before the hub
auto-expires it. Matches Matrix MSC2746's default."""

MAX_ENVELOPE_BYTES = 64 * 1024
"""Cap on a single WS frame, mirroring our existing sendBeacon limit.
Sized to fit large SDPs (typically 5-20KB) with headroom for batched
ICE arrays. Anything beyond is almost certainly malicious or buggy
and is dropped at the receive layer."""


# ── Client → Server messages ─────────────────────────────────────


MSG_PRESENCE_ANNOUNCE = "presence_announce"
"""Client tells the hub it's online & ready to accept calls.

Sent immediately after WS accept. Without it, the hub treats the
connection as observing-only (e.g. a non-Connect-mode browser tab).
"""

MSG_INVITE = "invite"
"""Initiate a call to ``to``.

``data``: ``{"modalities": "audio" | "audio,video", "call_id": str}``.
``call_id`` is minted by the initiator and threaded through the rest
of the lifecycle so both ends can reconcile their per-user
``call_sessions`` rows.
"""

MSG_ACCEPT = "accept"
"""Receiver answers an inbound invite. Triggers SDP offer/answer."""

MSG_DECLINE = "decline"
"""Receiver rejects an inbound invite. Terminal — call_id is dead."""

MSG_OFFER = "offer"
"""WebRTC SDP offer. ``data``: ``{"sdp": str, "call_id": str}``."""

MSG_ANSWER = "answer"
"""WebRTC SDP answer. ``data``: ``{"sdp": str, "call_id": str}``."""

MSG_CANDIDATES = "candidates"
"""Trickle ICE candidates — batched, plural.

``data``: ``{"candidates": [{"candidate": str, "sdpMid": str,
"sdpMLineIndex": int}, ...], "call_id": str, "party_id": str}``.
An empty ``candidates`` array is the end-of-gathering sentinel.

Plural / batched matches Matrix (``m.call.candidates``), LiveKit's
``Trickle`` with candidate list, and Janus's ``trickle`` array.
Single-per-message would be ~3x chattier and trip standard
signaling rate limits.
"""

MSG_SELECT_ANSWER = "select_answer"
"""Caller tells everyone-but-the-chosen-callee to stop ringing.

``data``: ``{"call_id": str, "selected_party_id": str}``. Required
once a user has multiple devices that can both answer the same
invite — without it, two answers race and one PeerConnection ends
up orphaned. See Matrix MSC2746 § select_answer.
"""

MSG_NEGOTIATE = "negotiate"
"""Mid-call SDP renegotiation request.

``data``: ``{"call_id": str, "party_id": str, "description":
{"type": "offer"|"answer", "sdp": str}}``. Used for video on/off,
screen share start/stop, codec changes. Distinct from initial
offer/answer so the routing layer can apply different rules
(e.g. don't re-ring receiver).
"""

MSG_HANGUP = "hangup"
"""Either party ends the call.

``data``: ``{"call_id": str, "party_id": str, "reason": str}``.
Reason is informational; the hub doesn't enforce a vocabulary at
the wire level (``call_sessions.end_reason`` validates server-side).
"""

MSG_MUTE_STATE = "mute_state"
"""Either party reports their local mute state changed.

``data``: ``{"call_id": str, "party_id": str, "muted": bool}``.
Fire-and-forget metadata — no server-side state machine, just routes
through to the peer as ``EVENT_MUTE_STATE`` so their UI can show a
"Peer is muted" badge. Without this the remote side has no signal
distinguishing "muted" from "silent" / "lost connection".
"""

MSG_VIDEO_STATE = "video_state"
"""Either party reports their local camera was turned on or off.

``data``: ``{"call_id": str, "party_id": str, "video_enabled": bool}``.

The video-side counterpart of :data:`MSG_MUTE_STATE`, needed for the
same reason but more acutely: disabling a camera sets
``track.enabled = False``, which keeps sending BLACK FRAMES rather than
stopping the stream. The peer therefore cannot infer camera-off from
the media at all — an un-signalled camera-off is indistinguishable from
a frozen or dropped call. Routes through to the peer as
``EVENT_VIDEO_STATE`` so their UI can swap the tile for a "camera is
off" identity card.

Fire-and-forget, same as mute: no server-side state machine, no retry.
"""

MSG_PING = "ping"
"""Keepalive. Hub responds with ``EVENT_PONG``. Optional but lets
the UI detect stale WS connections without waiting for TCP timeouts.
"""


MSG_TEXT_SEND = "text_send"
"""Send a text message into a 1:1 thread with ``to``.

``data``: ``{"thread_id": str, "message_id": str, "body": str,
"format": "plain"|"markdown"|"voice_note"|"embed",
"reply_to": str, "attachment_ref": str, "sent_at": iso8601}``.

``message_id`` is client-minted (UUID-like) so the sender's
optimistic UI can place the message before the server round-trip
returns. The server uses ``INSERT OR IGNORE`` on (message_id, user_id)
so retries are idempotent.

``thread_id`` is client-minted on first message in a thread (a stable
hash of the sorted peer DID pair is one option; UUID is the other —
the server doesn't care, it just records what's supplied).
"""

MSG_TEXT_READ = "text_read"
"""Mark up to ``last_read_message_id`` (inclusive) as read in a thread.

``data``: ``{"thread_id": str, "last_read_message_id": str}``. Server
clears the recipient's unread counter and routes a read-receipt
``EVENT_TEXT_READ`` to the sender.
"""

MSG_TEXT_DELIVERED = "text_delivered"
"""Recipient confirms one or more messages have actually been received.

``data``: ``{"thread_id": str, "message_ids": [str, ...]}``. Sent by
the recipient's UI when ``EVENT_TEXT_RECEIVED`` arrives (or when a
catch-up fetch surfaces previously-missed messages). The server
stamps ``delivered_at`` on the original sender's row and routes
``EVENT_TEXT_DELIVERED`` back so the sender's UI can swap the
"sent" tick for a "delivered" tick.

Why: ``stamp_delivered`` at server-store time is a lie — it stamps
before the peer's WS has actually received anything. Moving the
stamp to recipient ACK makes the ticks honest at the cost of one
extra cheap round-trip per inbound message (batchable).
"""

MSG_TEXT_DELETE = "text_delete"
"""Soft-delete a message you sent.

``data``: ``{"thread_id": str, "message_id": str}``. The row stays
(audit trail) but body is cleared and ``deleted_at`` is stamped on
both sides.
"""

MSG_TYPING_START = "typing_start"
"""Ephemeral — sender is composing a message.

``data``: ``{"thread_id": str}``. No persistence; the server just
fans the event out to the peer so their UI can show the "Alice is
typing…" indicator. The client should debounce keystrokes (e.g.
emit one START + one STOP per ~3s window) so this doesn't spam.
"""

MSG_TYPING_STOP = "typing_stop"
"""Ephemeral — sender stopped composing. ``data``: ``{"thread_id": str}``.

Should be emitted on send, on field-blur, and on the debounced
keystroke-idle timeout.
"""

MSG_TEXT_EDIT = "text_edit"
"""Edit a message you sent.

``data``: ``{"thread_id": str, "message_id": str, "body": str}``.
Server stamps ``edited_at`` on both sides and routes an
``EVENT_TEXT_EDIT`` to the peer so their UI can re-render in place.
"""

MSG_TEXT_REACT = "text_react"
"""Attach or remove an emoji reaction on a message.

``data``: ``{"thread_id": str, "message_id": str, "emoji": str,
"action": "add" | "remove"}``.

Reactions are idempotent on (message_id, reactor_did, emoji) so
retries don't double-count. ``action="remove"`` deletes the row;
``action="add"`` (or absent) creates it. Server fans out via
``EVENT_TEXT_REACT`` to the peer's signaling WS.
"""


# ── Server → Client events ───────────────────────────────────────


EVENT_WELCOME = "welcome"
"""Hub identifies the peer & ships its TURN credentials.

``data``: ``{"user_did": str, "turn": {urls, username, credential,
expires_at}, "server_time": int}``. Sent once, immediately after WS
accept. The client uses ``turn`` to build its ``RTCIceServer`` config.
"""

EVENT_PRESENCE_UPDATE = "presence_update"
"""A contact's online/offline state changed.

``data``: ``{"peer_did": str, "status": "online" | "offline"}``. Sent
when a contact connects or disconnects on the same instance.
"""

EVENT_INVITE = "invite"
"""Routed from another peer's ``MSG_INVITE``. Same payload shape."""

EVENT_OFFER = "offer"
"""Routed SDP offer."""

EVENT_ANSWER = "answer"
"""Routed SDP answer."""

EVENT_CANDIDATES = "candidates"
"""Routed ICE candidates (batched, plural)."""

EVENT_ACCEPT = "accept"
"""Receiver agreed to take the call.

Routed from ``MSG_ACCEPT``. Tells the initiator's UI to proceed to
SDP offer/answer; the receiver's UI sets up its RTCPeerConnection to
receive the offer.
"""

EVENT_DECLINE = "decline"
"""Receiver refused the call.

Routed from ``MSG_DECLINE``. Terminal — the call_id is dead.
"""

EVENT_SELECT_ANSWER = "select_answer"
"""Routed ``MSG_SELECT_ANSWER`` — losing parties stop ringing."""

EVENT_NEGOTIATE = "negotiate"
"""Routed mid-call renegotiation request."""

EVENT_HANGUP = "hangup"
"""Routed hangup."""

EVENT_MUTE_STATE = "mute_state"
"""Routed ``MSG_MUTE_STATE`` — peer's mic mute state changed."""

EVENT_VIDEO_STATE = "video_state"
"""Routed ``MSG_VIDEO_STATE`` — peer's camera was turned on or off."""

EVENT_PONG = "pong"
"""Response to ``MSG_PING``. ``data``: ``{"server_time": int}``."""


EVENT_TEXT_RECEIVED = "text_received"
"""Routed from the peer's ``MSG_TEXT_SEND``.

``data`` mirrors ``MSG_TEXT_SEND.data`` plus ``sender_did`` for the
recipient's render path.
"""

EVENT_TEXT_READ = "text_read"
"""Read receipt routed from the peer's ``MSG_TEXT_READ``.

``data``: ``{"thread_id": str, "last_read_message_id": str,
"reader_did": str}``. Sender's UI updates the per-message read
indicator up to ``last_read_message_id`` (inclusive).
"""

EVENT_TEXT_DELIVERED = "text_delivered"
"""Delivery receipt routed from the peer's ``MSG_TEXT_DELIVERED``.

``data``: ``{"thread_id": str, "message_ids": [str, ...]}``. Sender's
UI marks each listed message as delivered. Delivered is one tier below
read: it means "their device received the bytes", not "they saw it".
"""

EVENT_TEXT_DELETE = "text_delete"
"""Routed from the peer's ``MSG_TEXT_DELETE`` — UI tombs the message."""

EVENT_TEXT_EDIT = "text_edit"
"""Routed from the peer's ``MSG_TEXT_EDIT`` — UI replaces the body."""


EVENT_TYPING_START = "typing_start"
"""Routed from the peer's ``MSG_TYPING_START`` — UI shows indicator."""

EVENT_TYPING_STOP = "typing_stop"
"""Routed from the peer's ``MSG_TYPING_STOP`` — UI hides indicator."""

EVENT_TEXT_REACT = "text_react"
"""Routed from the peer's ``MSG_TEXT_REACT`` — UI updates the
reaction pill stack on the targeted message."""

EVENT_ERROR = "error"
"""Hub couldn't fulfil a request.

``data``: ``{"code": str, "message": str, "ref_id": str}``. ``ref_id``
correlates back to the inbound message's ``id`` when present.
"""


@dataclass
class ConnectEnvelope:
    """One signaling message, either direction.

    ``kind`` is "msg" (client → server) or "event" (server → client).
    ``verb`` is the specific msg/event constant. ``corr_id`` is an
    arbitrary correlation token the sender chose; receivers echo it
    in any direct response (used for ping/pong + error replies).

    ``peer`` is the OTHER end:
      - on outbound msgs: the recipient peer_did (``to``)
      - on inbound events: the sender peer_did (``from``)
    """

    kind: str          # "msg" | "event"
    verb: str          # msg/event constant
    corr_id: str = ""  # optional correlation token
    peer: str = ""     # peer DID (other end)
    data: dict[str, Any] = field(default_factory=dict)


def serialise_envelope(env: ConnectEnvelope) -> str:
    """Encode an envelope to JSON for ``websocket.send_text``."""

    out: dict[str, Any] = {
        "type": env.kind,
        env.kind: env.verb,  # type: "msg" → key "msg"; type: "event" → key "event"
    }
    if env.corr_id:
        out["id"] = env.corr_id
    if env.peer:
        out["to" if env.kind == "msg" else "from"] = env.peer
    if env.data:
        out["data"] = env.data
    return json.dumps(out, separators=(",", ":"))


def deserialise_envelope(raw: str) -> ConnectEnvelope | None:
    """Decode a wire JSON message. Returns ``None`` on garbage.

    Tolerant by design — the WS receive loop drops bad messages
    silently rather than killing the connection (a buggy client
    shouldn't make a healthy peer drop its call).
    """

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None

    kind = parsed.get("type")
    if kind not in ("msg", "event"):
        return None

    verb = parsed.get(kind)
    if not isinstance(verb, str) or not verb:
        return None

    corr_id = parsed.get("id") or ""
    peer = parsed.get("to") if kind == "msg" else parsed.get("from")
    if peer is not None and not isinstance(peer, str):
        peer = ""

    data = parsed.get("data") or {}
    if not isinstance(data, dict):
        data = {}

    return ConnectEnvelope(
        kind=kind,
        verb=verb,
        corr_id=str(corr_id) if corr_id else "",
        peer=str(peer) if peer else "",
        data=data,
    )
