"""Negotiate routing — mid-call SDP renegotiation w/ modality persistence.

Covers the path the dialer/incoming UI walks when a connected call
escalates from audio-only to audio+video (or drops back). The wire
``MSG_NEGOTIATE`` carries a modality declaration; the routing layer
applies it to both perspectives' ``call_sessions`` rows so the call
history surface sees the right modality on a partially-video call.

The renegotiate is also expected to:

  * Route ``EVENT_NEGOTIATE`` to the peer's signaling WS.
  * Log a ``renegotiate`` row in ``call_events`` for the audit trail.
  * NOT publish a fresh notification (this is mid-call; no re-ring).
  * NOT cancel the missed-call timer (already cancelled at accept).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from augmentum.connect.call_routing import (
    CallRoutingResult,
    _normalise_modalities,
    handle_signaling_envelope,
    new_party_id,
)
from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_NEGOTIATE,
    MSG_NEGOTIATE,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text()
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text()


ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID = local_did_for(ALICE_ID)
BOB_DID = local_did_for(BOB_ID)


class FakeWS:
    """Async-shaped WS capture for connect-hub fan-out."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        await c.commit()
        yield c


async def _seed_connected_call(
    conn: Any, *,
    call_id: str = "negotiate-call-1",
    modalities: str = "audio",
) -> None:
    """Insert two call_sessions rows in ``connected`` state.

    Mirrors the post-accept state: both perspectives present, modality
    captured from the original invite, no end timestamp.
    """

    from datetime import UTC, datetime
    now = datetime.now(UTC).isoformat()
    for user_id in (ALICE_ID, BOB_ID):
        await conn.execute(
            """INSERT INTO call_sessions
                   (call_id, user_id, initiator_did, receiver_did,
                    modalities, state, initiated_at, connected_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                call_id, user_id, ALICE_DID, BOB_DID,
                modalities, "connected", now, now,
            ),
        )
    await conn.commit()


def _last_envelope(ws: FakeWS) -> dict[str, Any]:
    assert ws.sent, "expected at least one sent frame"
    return json.loads(ws.sent[-1])


# ── _normalise_modalities ───────────────────────────────────────


def test_normalise_modalities_string_input() -> None:
    assert _normalise_modalities("audio") == "audio"
    assert _normalise_modalities("video") == "video"
    assert _normalise_modalities("audio,video") == "audio,video"
    # Canonical ordering: audio always first regardless of input order.
    assert _normalise_modalities("video,audio") == "audio,video"
    # Whitespace tolerated.
    assert _normalise_modalities(" audio , video ") == "audio,video"


def test_normalise_modalities_list_input() -> None:
    assert _normalise_modalities(["audio"]) == "audio"
    assert _normalise_modalities(["video", "audio"]) == "audio,video"


def test_normalise_modalities_garbage_falls_back_to_audio() -> None:
    assert _normalise_modalities("") == "audio"
    assert _normalise_modalities(None) == "audio"
    assert _normalise_modalities("bogus") == "audio"
    assert _normalise_modalities({"not": "a string"}) == "audio"


# ── handle_signaling_envelope(MSG_NEGOTIATE) ────────────────────


@pytest.mark.asyncio
async def test_negotiate_routes_event_to_peer(conn) -> None:
    """Caller adds video; Bob's signaling WS sees EVENT_NEGOTIATE
    carrying the new modality declaration + the SDP offer the peer
    needs to renegotiate."""

    await _seed_connected_call(conn, modalities="audio")

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    alice_ws = FakeWS()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)
    alice_ws.sent.clear()
    bob_ws.sent.clear()

    alice_party = new_party_id()
    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg",
            verb=MSG_NEGOTIATE,
            peer=BOB_DID,
            data={
                "call_id": "negotiate-call-1",
                "description": {"type": "offer", "sdp": "v=0\r\nfake-sdp"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=alice_party,
    )

    assert isinstance(result, CallRoutingResult)
    assert result.error_code == "", result.error_message
    assert result.call_id == "negotiate-call-1"
    assert result.routed == 1, "Bob's WS should have received exactly one frame"

    env = _last_envelope(bob_ws)
    assert env["type"] == "event"
    assert env["event"] == EVENT_NEGOTIATE
    assert env["from"] == ALICE_DID
    assert env["data"]["call_id"] == "negotiate-call-1"
    assert env["data"]["modalities"] == "audio,video"
    assert env["data"]["party_id"] == alice_party
    assert env["data"]["description"]["type"] == "offer"
    assert "fake-sdp" in env["data"]["description"]["sdp"]
    # Alice should NOT see her own negotiate reflected.
    assert alice_ws.sent == []


@pytest.mark.asyncio
async def test_negotiate_updates_modalities_both_sides(conn) -> None:
    """The modality declaration on the negotiate is written to BOTH
    perspectives' call_sessions rows, so the calls-history surface
    sees a consistent view regardless of which user opens it."""

    await _seed_connected_call(conn, modalities="audio")

    connect_hub = ConnectHub()
    await connect_hub.attach(ws=FakeWS(), user_id=BOB_ID, user_did=BOB_DID)

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "call_id": "negotiate-call-1",
                "description": {"type": "offer", "sdp": "v=0\r\nfake"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    cur = await conn.execute(
        "SELECT user_id, modalities FROM call_sessions "
        "WHERE call_id = ? ORDER BY user_id",
        ("negotiate-call-1",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {(r[0], r[1]) for r in rows} == {
        (ALICE_ID, "audio,video"),
        (BOB_ID, "audio,video"),
    }


@pytest.mark.asyncio
async def test_negotiate_without_modalities_leaves_state_untouched(conn) -> None:
    """A negotiate driven by codec swap (no modality declaration)
    must still route the SDP through, but ``call_sessions.modalities``
    should be unchanged. Otherwise we'd silently corrupt history."""

    await _seed_connected_call(conn, modalities="audio,video")

    connect_hub = ConnectHub()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "call_id": "negotiate-call-1",
                "description": {"type": "offer", "sdp": "v=0\r\ncodec-swap"},
            },
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    cur = await conn.execute(
        "SELECT modalities FROM call_sessions WHERE call_id = ?",
        ("negotiate-call-1",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {r[0] for r in rows} == {"audio,video"}

    # Routed envelope still hit Bob.
    env = _last_envelope(bob_ws)
    assert env["event"] == EVENT_NEGOTIATE
    assert env["data"]["description"]["sdp"].endswith("codec-swap")


@pytest.mark.asyncio
async def test_negotiate_logs_call_event(conn) -> None:
    """A renegotiation produces a ``renegotiate`` row in call_events
    carrying both the new modality and the previous one — used by the
    Calls panel's timeline."""

    await _seed_connected_call(conn, modalities="audio")
    connect_hub = ConnectHub()
    await connect_hub.attach(ws=FakeWS(), user_id=BOB_ID, user_did=BOB_DID)

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "call_id": "negotiate-call-1",
                "description": {"type": "offer", "sdp": "v=0\r\nfake"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    cur = await conn.execute(
        "SELECT event_type, event_data FROM call_events "
        "WHERE call_id = ? AND user_id = ?",
        ("negotiate-call-1", ALICE_ID),
    )
    events = await cur.fetchall()
    await cur.close()
    assert any(e[0] == "renegotiate" for e in events)
    reneg = next(e for e in events if e[0] == "renegotiate")
    payload = json.loads(reneg[1])
    assert payload["modalities"] == "audio,video"
    assert payload["previous_modalities"] == "audio"
    assert payload["description_type"] == "offer"


@pytest.mark.asyncio
async def test_negotiate_does_not_change_state(conn) -> None:
    """Mid-call renegotiation must keep ``state='connected'`` — a
    bug where it accidentally reset to ``ringing`` would re-ring the
    receiver."""

    await _seed_connected_call(conn, modalities="audio")
    connect_hub = ConnectHub()
    await connect_hub.attach(ws=FakeWS(), user_id=BOB_ID, user_did=BOB_DID)

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "call_id": "negotiate-call-1",
                "description": {"type": "offer", "sdp": "v=0"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ?",
        ("negotiate-call-1",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {r[0] for r in rows} == {"connected"}


@pytest.mark.asyncio
async def test_negotiate_missing_call_id_returns_error(conn) -> None:
    """A negotiate without a call_id can't be persisted; the routing
    layer surfaces ``missing_call_id`` so the sender's UI can decide
    what to do."""

    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "description": {"type": "offer", "sdp": "v=0"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )
    assert result.error_code == "missing_call_id"


@pytest.mark.asyncio
async def test_negotiate_answer_passthrough(conn) -> None:
    """The renegotiate flow is symmetric — when Bob's UI answers
    Alice's offer with ``description.type='answer'``, it routes back
    just the same. The modality declaration mirrors what was on the
    offer, so the row stays consistent."""

    await _seed_connected_call(conn, modalities="audio,video")
    connect_hub = ConnectHub()
    alice_ws = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=ALICE_DID,
            data={
                "call_id": "negotiate-call-1",
                "description": {"type": "answer", "sdp": "v=0\r\nbob-answer"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=BOB_ID,
        sender_did=BOB_DID,
        sender_party_id=new_party_id(),
    )

    env = _last_envelope(alice_ws)
    assert env["event"] == EVENT_NEGOTIATE
    assert env["data"]["description"]["type"] == "answer"
    assert env["data"]["modalities"] == "audio,video"
