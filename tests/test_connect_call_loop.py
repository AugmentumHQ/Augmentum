"""End-to-end loop: Alice INVITE → Bob notification → Bob accept → Alice ACCEPT.

Exercises every layer of the production code path with no mocks:

* SQLite in-memory backend with migrations 219 (Connect substrate)
  and 221 (notification substrate) applied.
* Real ``ConnectHub``, ``NotificationHub``, and call-routing module.
* Real action handler registered for ``connect.call.*``.
* Two ``FakeWS`` clients standing in for Alice and Bob's signaling
  WSes, and one for Bob's notification subscription.

The "loop" verified here is the contract the UI relies on:

1. Alice sends ``MSG_INVITE`` with peer DID for Bob.
2. Bob's signaling WS receives ``EVENT_INVITE``.
3. Bob's notification WS receives a ``connect.call.incoming`` push.
4. Both users' ``call_sessions`` rows exist with the right states.
5. Bob's action handler routes ``EVENT_ACCEPT`` back to Alice.
6. Both ``call_sessions`` advance to ``connected``.
7. ``call_events`` carries an audit trail.

Each assertion catches a specific class of regression that would
silently break end-to-end UX.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.call_routing import (
    handle_call_action,
    handle_signaling_envelope,
    new_party_id,
)
from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_ACCEPT,
    EVENT_DECLINE,
    EVENT_INVITE,
    MSG_INVITE,
    ConnectEnvelope,
)
from augmentum.notifications import (
    IMPORTANCE_CRITICAL,
)
from augmentum.notifications.hub import NotificationHub
from augmentum.notifications.store import (
    get_notification,
    list_for_user,
)

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text()
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text()


# Two users plus DIDs in the Phase 1 same-instance form.
ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID = local_did_for(ALICE_ID)
BOB_DID = local_did_for(BOB_ID)


class FakeWS:
    """Captures sent payloads. Async-shaped to match WebSocket interface."""

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


def _last_envelope(ws: FakeWS) -> dict[str, Any]:
    """Decode the last frame the FakeWS received."""

    assert ws.sent, "expected at least one sent frame"
    return json.loads(ws.sent[-1])


@pytest.mark.asyncio
async def test_full_call_loop(conn) -> None:
    # ── Setup: hubs + attached WSes for both users ──────────────
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling_ws = FakeWS()
    bob_signaling_ws = FakeWS()
    bob_notification_ws = FakeWS()

    await connect_hub.attach(
        ws=alice_signaling_ws, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    # bob_signaling attach happens before bob_notification so the
    # ordering of presence broadcasts is deterministic.
    await connect_hub.attach(
        ws=bob_signaling_ws, user_id=BOB_ID, user_did=BOB_DID,
    )
    # Clear the presence broadcasts both saw during attach — those
    # are tested separately in test_connect_hub.py.
    alice_signaling_ws.sent.clear()
    bob_signaling_ws.sent.clear()

    await notification_hub.attach(ws=bob_notification_ws, user_id=BOB_ID)

    alice_party_id = new_party_id()
    bob_party_id = new_party_id()
    assert alice_party_id != bob_party_id

    # ── Step 1: Alice sends MSG_INVITE → Bob ────────────────────
    invite_env = ConnectEnvelope(
        kind="msg",
        verb=MSG_INVITE,
        corr_id="alice-req-1",
        peer=BOB_DID,
        data={"modalities": "audio,video"},
    )
    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=invite_env,
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=alice_party_id,
    )

    # Routing produced a call_id, notification_id, and reached Bob.
    assert result.error_code == "", f"unexpected error: {result.error_message}"
    assert result.call_id, "expected a minted call_id"
    assert result.notification_id, "expected a published notification_id"
    assert result.routed == 1, "expected Bob's signaling WS to receive 1 frame"

    call_id = result.call_id
    notification_id = result.notification_id

    # ── Step 2: Bob's signaling WS received EVENT_INVITE ────────
    assert len(bob_signaling_ws.sent) == 1
    invite_envelope = _last_envelope(bob_signaling_ws)
    assert invite_envelope["type"] == "event"
    assert invite_envelope["event"] == EVENT_INVITE
    assert invite_envelope["from"] == ALICE_DID
    assert invite_envelope["data"]["call_id"] == call_id
    assert invite_envelope["data"]["party_id"] == alice_party_id
    assert invite_envelope["data"]["modalities"] == "audio,video"

    # Alice should NOT see her own invite reflected back.
    assert alice_signaling_ws.sent == []

    # ── Step 3: Bob's notification WS received the push ─────────
    assert len(bob_notification_ws.sent) == 1
    push = json.loads(bob_notification_ws.sent[0])
    assert push["type"] == "notification"
    assert push["notification"]["channel_id"] == "connect.call.incoming"
    assert push["notification"]["importance"] == IMPORTANCE_CRITICAL
    assert push["notification"]["thread_id"] == call_id
    assert push["notification"]["payload"]["call_id"] == call_id
    assert push["notification"]["payload"]["initiator_user_id"] == ALICE_ID
    # Accept + decline buttons should be present.
    action_ids = {a["id"] for a in push["notification"]["actions"]}
    assert action_ids == {"accept", "decline"}

    # ── Step 4: call_sessions exist for both users ──────────────
    cur = await conn.execute(
        "SELECT user_id, state, initiator_did, receiver_did "
        "FROM call_sessions WHERE call_id = ? ORDER BY user_id",
        (call_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 2
    by_user = {r[0]: r for r in rows}
    assert by_user[ALICE_ID][1] == "ringing"
    assert by_user[BOB_ID][1] == "invited"
    for _, _, init, recv in rows:
        assert init == ALICE_DID
        assert recv == BOB_DID

    # ── Step 5: Bob's notification is persisted in his feed ─────
    bob_feed = await list_for_user(conn, user_id=BOB_ID)
    assert len(bob_feed) == 1
    assert bob_feed[0].notification_id == notification_id
    assert bob_feed[0].channel_id == "connect.call.incoming"
    # Alice should not see Bob's notification.
    alice_feed = await list_for_user(conn, user_id=ALICE_ID)
    assert alice_feed == []

    # ── Step 6: Bob clicks Accept (action handler) ──────────────
    # Build a fake Request whose app.state surfaces the hubs +
    # state_manager so the action handler can reach them.
    notification = await get_notification(
        conn, user_id=BOB_ID, notification_id=notification_id,
    )
    assert notification is not None

    from augmentum.state.backends.sqlite import SQLiteBackend

    fake_request = MagicMock()
    fake_request.app.state.connect_hub = connect_hub
    fake_backend = MagicMock(spec=SQLiteBackend)
    fake_backend.conn = conn
    fake_request.app.state.state_manager.backend = fake_backend

    # Clear pre-existing frames so we can assert on what Alice sees
    # next.
    alice_signaling_ws.sent.clear()
    bob_signaling_ws.sent.clear()

    result_payload = await handle_call_action(
        notification, "accept", fake_request,
    )

    assert result_payload["status"] == "accepted"
    assert result_payload["call_id"] == call_id
    assert result_payload["delivered_to_initiator"] == 1

    # ── Step 7: Alice's signaling WS received EVENT_ACCEPT ──────
    assert len(alice_signaling_ws.sent) == 1
    accept_env = _last_envelope(alice_signaling_ws)
    assert accept_env["type"] == "event"
    assert accept_env["event"] == EVENT_ACCEPT
    assert accept_env["from"] == BOB_DID
    assert accept_env["data"]["call_id"] == call_id

    # Bob's signaling WS receives a sibling-fanout echo of EVENT_ACCEPT
    # so any other tab he has open dismisses its ringing modal. The
    # HTTP banner-action path can't identify the originating tab the
    # way the WS path can, so the echo fans to every Bob tab.
    assert len(bob_signaling_ws.sent) == 1
    bob_echo = _last_envelope(bob_signaling_ws)
    assert bob_echo["event"] == EVENT_ACCEPT
    assert bob_echo["data"]["call_id"] == call_id
    assert bob_echo["data"].get("resolved_by") == "sibling"

    # ── Step 8: Both call_sessions advanced to 'connected' ──────
    cur = await conn.execute(
        "SELECT user_id, state, connected_at FROM call_sessions "
        "WHERE call_id = ? ORDER BY user_id",
        (call_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    assert {(r[0], r[1]) for r in rows} == {
        (ALICE_ID, "connected"),
        (BOB_ID, "connected"),
    }
    # connected_at should be stamped on both (non-empty).
    for _, _, connected_at in rows:
        assert connected_at and len(connected_at) > 0

    # ── Step 9: call_events carries an audit trail ──────────────
    cur = await conn.execute(
        "SELECT event_type, user_id FROM call_events "
        "WHERE call_id = ? ORDER BY event_id",
        (call_id,),
    )
    events = await cur.fetchall()
    await cur.close()
    event_types = [e[0] for e in events]
    # At minimum: the invite log + the accept log.
    assert "invited" in event_types
    assert "accepted" in event_types


@pytest.mark.asyncio
async def test_decline_loop(conn) -> None:
    """Same setup; Bob declines instead of accepts. Alice should see
    EVENT_DECLINE and both call_sessions land in 'declined'."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws = FakeWS()
    bob_signaling = FakeWS()
    bob_notify = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )
    alice_ws.sent.clear()
    bob_signaling.sent.clear()
    await notification_hub.attach(ws=bob_notify, user_id=BOB_ID)

    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID,
            data={"modalities": "audio"},
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )
    call_id = result.call_id
    notification = await get_notification(
        conn, user_id=BOB_ID, notification_id=result.notification_id,
    )

    # Set up fake request shape (same as the happy-path test).
    from augmentum.state.backends.sqlite import SQLiteBackend
    fake_request = MagicMock()
    fake_request.app.state.connect_hub = connect_hub
    fake_backend = MagicMock(spec=SQLiteBackend)
    fake_backend.conn = conn
    fake_request.app.state.state_manager.backend = fake_backend

    alice_ws.sent.clear()

    payload = await handle_call_action(notification, "decline", fake_request)
    assert payload["status"] == "declined"

    # Alice receives EVENT_DECLINE.
    assert len(alice_ws.sent) == 1
    decline_env = _last_envelope(alice_ws)
    assert decline_env["event"] == EVENT_DECLINE
    assert decline_env["data"]["call_id"] == call_id

    # Both call_sessions land in 'declined' with end_reason.
    cur = await conn.execute(
        "SELECT user_id, state, end_reason FROM call_sessions "
        "WHERE call_id = ? ORDER BY user_id",
        (call_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    for _, state, end_reason in rows:
        assert state == "declined"
        assert end_reason == "declined"


@pytest.mark.asyncio
async def test_invite_when_receiver_offline_still_persists_notification(
    conn,
) -> None:
    """If Bob has no signaling WS active, the invite still publishes
    a notification — he'll see it next time he opens Connect."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    alice_ws = FakeWS()
    await connect_hub.attach(
        ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    alice_ws.sent.clear()
    # NO bob attached.

    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID, data={},
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )

    # No live routing target, but the notification still landed.
    assert result.routed == 0
    assert result.notification_id, "notification should persist even if WS offline"

    feed = await list_for_user(conn, user_id=BOB_ID)
    assert len(feed) == 1
    assert feed[0].channel_id == "connect.call.incoming"


@pytest.mark.asyncio
async def test_invalid_peer_did_returns_error(conn) -> None:
    """A malformed peer DID surfaces ``peer_did_invalid`` so the
    sender's WS can render a useful error instead of a silent drop."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer="garbage", data={},
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )
    assert result.error_code == "peer_did_invalid"
    # Nothing should have been persisted.
    cur = await conn.execute("SELECT COUNT(*) FROM call_sessions")
    count = (await cur.fetchone())[0]
    await cur.close()
    assert count == 0


@pytest.mark.asyncio
async def test_fabric_peer_returns_unavailable_when_coordinator_missing(
    conn,
) -> None:
    """Fabric routing was wired in Wedge B Phase 4; when fabric is
    disabled on this instance (coordinator=None), the routing layer
    returns ``fabric_unavailable`` rather than silently dropping the
    invite. The error is the contract the dialer surfaces to the UI."""

    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE,
            peer="alice@peer.example.com",
            data={},
        ),
        sender_user_id=BOB_ID,
        sender_did=BOB_DID,
        sender_party_id=new_party_id(),
    )
    assert result.error_code == "fabric_unavailable"


@pytest.mark.asyncio
async def test_invite_idempotent_via_dedupe_key(conn) -> None:
    """If a flaky network causes the same invite to arrive twice
    with the same call_id, the notification should update in place
    (single feed entry), not produce two duplicate rings."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_signaling = FakeWS()
    bob_notify = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )
    bob_signaling.sent.clear()
    await notification_hub.attach(ws=bob_notify, user_id=BOB_ID)

    common = dict(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=new_party_id(),
    )
    a = await handle_signaling_envelope(
        **common,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID,
            data={"call_id": "stable-call-1"},
        ),
    )
    b = await handle_signaling_envelope(
        **common,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID,
            data={"call_id": "stable-call-1"},
        ),
    )

    assert a.call_id == b.call_id == "stable-call-1"
    # Same notification id both times (in-place update by dedupe_key).
    assert a.notification_id == b.notification_id
    feed = await list_for_user(conn, user_id=BOB_ID)
    assert len(feed) == 1
