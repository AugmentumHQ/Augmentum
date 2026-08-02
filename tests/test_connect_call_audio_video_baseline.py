"""Phase 0 baseline — single-instance audio call + video escalation.

Pins the call-substrate happy path BEFORE Wedge B adds fabric routing on
top. If anything here regresses, fabric will inherit the bug across an
instance boundary and triple the diagnostic surface; keeping these
green means fabric errors are fabric-side, not call-substrate-side.

Coverage:
  * Full audio-only call lifecycle — INVITE → notification-accept →
    OFFER → ANSWER → trickle CANDIDATES (both directions) → HANGUP.
    Asserts every verb lands on the right peer's signaling WS with
    the right payload + the call_sessions row tracks the state
    transitions accurately.
  * Caller-initiated audio → audio+video escalation mid-call.
  * Callee-initiated audio → audio+video escalation mid-call.
  * Down-escalation audio+video → audio (e.g. user turns camera off).

These exercise the same code path the dialer + incoming-modal walk in
production; only the WS transport is replaced with FakeWS.

Manual two-browser smoke (not automated — runs against a live
docker stack):

  1. ``start.bat`` on a host with mic + webcam.
  2. Open desktop browser as user A. Open phone browser as user B
     (same LAN; iOS/Android Safari/Chrome both fine).
  3. Add B as a contact on A; place an audio-only call.
  4. Verify two-way audio (speak into both, hear on both ends).
  5. From A, toggle camera on. Verify B's UI surfaces incoming video
     and shows A's frames; verify A's UI shows B's frames once B's
     camera spins up.
  6. From B, toggle camera off. Verify A's UI drops the video tile
     and reports modality back to audio-only.
  7. Hang up from either side; verify both UIs return to home and the
     call appears in the Calls panel with the right modality summary.

If the automated tests below stay green but the manual smoke fails,
the regression is in the UI (dialer / incoming-modal / connect-client),
not in the routing substrate — start there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.call_lifecycle import reset_timers_for_test
from augmentum.connect.call_routing import (
    handle_call_action,
    handle_signaling_envelope,
    new_party_id,
)
from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_ACCEPT,
    EVENT_ANSWER,
    EVENT_CANDIDATES,
    EVENT_HANGUP,
    EVENT_INVITE,
    EVENT_NEGOTIATE,
    EVENT_OFFER,
    MSG_ANSWER,
    MSG_CANDIDATES,
    MSG_HANGUP,
    MSG_INVITE,
    MSG_NEGOTIATE,
    MSG_OFFER,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub
from augmentum.notifications.store import get_notification

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql",
).read_text(encoding="utf-8")
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql",
).read_text(encoding="utf-8")


ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID = local_did_for(ALICE_ID)
BOB_DID = local_did_for(BOB_ID)


class FakeWS:
    """Async-shaped WS capture — matches the FakeWS used elsewhere."""

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


@pytest.fixture(autouse=True)
def _reset_timers():
    """Missed-call timers are process-local — clean between cases so
    a stray ringing timer from one test can't fire during another."""
    reset_timers_for_test()
    yield
    reset_timers_for_test()


def _last_envelope(ws: FakeWS) -> dict[str, Any]:
    assert ws.sent, "expected at least one sent frame"
    return json.loads(ws.sent[-1])


def _envelopes(ws: FakeWS) -> list[dict[str, Any]]:
    return [json.loads(s) for s in ws.sent]


async def _accept_via_action(
    conn: Any, *, connect_hub: ConnectHub, notification_id: str,
    recipient_user_id: str,
) -> None:
    """Drive the notification-accept path that the incoming-modal +
    notification-action use. This is what flips both call_sessions
    rows to ``connected`` — MSG_ACCEPT-over-WS alone only cancels
    the missed-call timer."""

    from augmentum.state.backends.sqlite import SQLiteBackend

    notification = await get_notification(
        conn, user_id=recipient_user_id, notification_id=notification_id,
    )
    assert notification is not None, "notification was not published"

    fake_request = MagicMock()
    fake_request.app.state.connect_hub = connect_hub
    fake_backend = MagicMock(spec=SQLiteBackend)
    fake_backend.conn = conn
    fake_request.app.state.state_manager.backend = fake_backend

    payload = await handle_call_action(notification, "accept", fake_request)
    assert payload["status"] == "accepted"


async def _seed_connected_call(
    conn: Any, *, call_id: str = "baseline-call", modalities: str = "audio",
) -> None:
    """Insert two call_sessions rows in ``connected`` state — used by
    the escalation tests that start from a live audio call."""

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


# ── Full audio-call lifecycle ───────────────────────────────────────


@pytest.mark.asyncio
async def test_full_audio_call_lifecycle(conn) -> None:
    """End-to-end audio call: INVITE → accept → OFFER → ANSWER →
    trickle CANDIDATES (both directions) → HANGUP. Every verb
    persists state correctly and lands on the right peer's WS."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    alice_ws = FakeWS()
    bob_ws = FakeWS()
    bob_notify = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)
    await notification_hub.attach(ws=bob_notify, user_id=BOB_ID)
    alice_ws.sent.clear()
    bob_ws.sent.clear()

    alice_party = new_party_id()
    bob_party = new_party_id()

    # 1. Alice invites Bob (audio only).
    invite_result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID,
            data={"modalities": "audio"},
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=alice_party,
    )
    assert invite_result.error_code == ""
    call_id = invite_result.call_id
    notification_id = invite_result.notification_id

    invite_env = _last_envelope(bob_ws)
    assert invite_env["event"] == EVENT_INVITE
    assert invite_env["data"]["modalities"] == "audio"
    assert invite_env["data"]["party_id"] == alice_party

    # 2. Bob accepts via notification action — transitions both
    # call_sessions to 'connected'.
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await _accept_via_action(
        conn, connect_hub=connect_hub,
        notification_id=notification_id, recipient_user_id=BOB_ID,
    )

    cur = await conn.execute(
        "SELECT user_id, state FROM call_sessions WHERE call_id = ? "
        "ORDER BY user_id",
        (call_id,),
    )
    states = {r[0]: r[1] for r in await cur.fetchall()}
    await cur.close()
    assert states == {ALICE_ID: "connected", BOB_ID: "connected"}

    # Alice's WS saw the EVENT_ACCEPT echo.
    accept_env = _last_envelope(alice_ws)
    assert accept_env["event"] == EVENT_ACCEPT
    assert accept_env["data"]["call_id"] == call_id

    # 3. Alice sends SDP offer.
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_OFFER, peer=BOB_DID,
            data={"call_id": call_id, "sdp": "v=0\r\nalice-audio-offer"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id=alice_party,
    )
    offer_env = _last_envelope(bob_ws)
    assert offer_env["event"] == EVENT_OFFER
    assert offer_env["from"] == ALICE_DID
    assert offer_env["data"]["sdp"].endswith("alice-audio-offer")
    assert offer_env["data"]["party_id"] == alice_party
    assert alice_ws.sent == [], "alice should not see her own offer reflected"

    # 4. Bob answers.
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_ANSWER, peer=ALICE_DID,
            data={"call_id": call_id, "sdp": "v=0\r\nbob-audio-answer"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=bob_party,
    )
    answer_env = _last_envelope(alice_ws)
    assert answer_env["event"] == EVENT_ANSWER
    assert answer_env["from"] == BOB_DID
    assert answer_env["data"]["sdp"].endswith("bob-audio-answer")
    assert bob_ws.sent == [], "bob should not see his own answer reflected"

    # 5. Trickle ICE — Alice → Bob.
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_CANDIDATES, peer=BOB_DID,
            data={
                "call_id": call_id,
                "candidates": [
                    {"candidate": "a=cand-1", "sdpMid": "0", "sdpMLineIndex": 0},
                    {"candidate": "a=cand-2", "sdpMid": "0", "sdpMLineIndex": 0},
                ],
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id=alice_party,
    )
    cand_env = _last_envelope(bob_ws)
    assert cand_env["event"] == EVENT_CANDIDATES
    assert len(cand_env["data"]["candidates"]) == 2

    # 6. Trickle ICE — Bob → Alice.
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_CANDIDATES, peer=ALICE_DID,
            data={
                "call_id": call_id,
                "candidates": [
                    {"candidate": "b=cand-1", "sdpMid": "0", "sdpMLineIndex": 0},
                ],
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=bob_party,
    )
    cand_back = _last_envelope(alice_ws)
    assert cand_back["event"] == EVENT_CANDIDATES
    assert cand_back["data"]["candidates"][0]["candidate"] == "b=cand-1"

    # 7. End-of-gathering sentinel (empty list).
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_CANDIDATES, peer=BOB_DID,
            data={"call_id": call_id, "candidates": []},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id=alice_party,
    )
    end_env = _last_envelope(bob_ws)
    assert end_env["event"] == EVENT_CANDIDATES
    assert end_env["data"]["candidates"] == []

    # 8. Hangup from Alice.
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_HANGUP, peer=BOB_DID,
            data={"call_id": call_id, "reason": "user_hangup"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id=alice_party,
    )
    hangup_env = _last_envelope(bob_ws)
    assert hangup_env["event"] == EVENT_HANGUP
    assert hangup_env["data"]["reason"] == "user_hangup"

    # Both rows terminal.
    cur = await conn.execute(
        "SELECT user_id, state, end_reason FROM call_sessions "
        "WHERE call_id = ? ORDER BY user_id",
        (call_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {(r[0], r[1]) for r in rows} == {
        (ALICE_ID, "ended"), (BOB_ID, "ended"),
    }
    for _, _, end_reason in rows:
        assert end_reason == "user_hangup"


# ── Mid-call escalation tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_caller_escalates_audio_to_video(conn) -> None:
    """Alice was on an audio call with Bob; mid-call she turns on her
    camera. The negotiate carries modalities=audio,video + a fresh
    SDP offer. Bob's WS gets EVENT_NEGOTIATE; both call_sessions rows
    flip to audio,video; the call stays connected."""

    await _seed_connected_call(conn, modalities="audio")

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    alice_ws = FakeWS()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)
    alice_ws.sent.clear()
    bob_ws.sent.clear()

    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "call_id": "baseline-call",
                "description": {
                    "type": "offer", "sdp": "v=0\r\nalice-video-on",
                },
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    env = _last_envelope(bob_ws)
    assert env["event"] == EVENT_NEGOTIATE
    assert env["from"] == ALICE_DID
    assert env["data"]["modalities"] == "audio,video"
    assert env["data"]["description"]["type"] == "offer"

    # Both perspectives flipped to audio,video. Stays 'connected'.
    cur = await conn.execute(
        "SELECT user_id, modalities, state FROM call_sessions "
        "WHERE call_id = ? ORDER BY user_id",
        ("baseline-call",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {(r[0], r[1], r[2]) for r in rows} == {
        (ALICE_ID, "audio,video", "connected"),
        (BOB_ID, "audio,video", "connected"),
    }

    # Bob then sends his answer back (camera comes up on his side).
    alice_ws.sent.clear()
    bob_ws.sent.clear()
    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=ALICE_DID,
            data={
                "call_id": "baseline-call",
                "description": {
                    "type": "answer", "sdp": "v=0\r\nbob-video-answer",
                },
                "modalities": "audio,video",
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
    )
    alice_env = _last_envelope(alice_ws)
    assert alice_env["event"] == EVENT_NEGOTIATE
    assert alice_env["data"]["description"]["type"] == "answer"
    assert alice_env["data"]["modalities"] == "audio,video"


@pytest.mark.asyncio
async def test_callee_escalates_audio_to_video(conn) -> None:
    """Bob (the callee) initiates the video escalation instead. Same
    contract: alice's WS receives EVENT_NEGOTIATE; both rows flip."""

    await _seed_connected_call(conn, modalities="audio")

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    alice_ws = FakeWS()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)
    alice_ws.sent.clear()
    bob_ws.sent.clear()

    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=ALICE_DID,
            data={
                "call_id": "baseline-call",
                "description": {
                    "type": "offer", "sdp": "v=0\r\nbob-video-on",
                },
                "modalities": "audio,video",
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
    )

    env = _last_envelope(alice_ws)
    assert env["event"] == EVENT_NEGOTIATE
    assert env["from"] == BOB_DID
    assert env["data"]["modalities"] == "audio,video"

    cur = await conn.execute(
        "SELECT modalities FROM call_sessions WHERE call_id = ?",
        ("baseline-call",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {r[0] for r in rows} == {"audio,video"}


@pytest.mark.asyncio
async def test_video_to_audio_downgrade(conn) -> None:
    """User turns camera off mid-call — modality declaration drops
    'video'. Both rows reflect the new mode; peer is notified."""

    await _seed_connected_call(conn, modalities="audio,video")

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)

    await handle_signaling_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID,
            data={
                "call_id": "baseline-call",
                "description": {
                    "type": "offer", "sdp": "v=0\r\nalice-cam-off",
                },
                "modalities": "audio",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    env = _last_envelope(bob_ws)
    assert env["event"] == EVENT_NEGOTIATE
    assert env["data"]["modalities"] == "audio"

    cur = await conn.execute(
        "SELECT modalities FROM call_sessions WHERE call_id = ?",
        ("baseline-call",),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {r[0] for r in rows} == {"audio"}
