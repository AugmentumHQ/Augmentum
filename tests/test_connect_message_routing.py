"""End-to-end test of the Connect text-message dispatch loop.

Mirrors test_connect_call_loop.py's shape: real ConnectHub +
NotificationHub + message routing module + sqlite migrations 219
(connect substrate) and 221 (notification substrate). Two users,
two FakeWS clients, no mocks.

What the loop verifies:

1. Alice sends MSG_TEXT_SEND → Bob.
2. Bob's signaling WS receives EVENT_TEXT_RECEIVED with the body.
3. Bob's notification WS receives a connect.message push with an
   "open_thread" + "mark_read" action.
4. Both users' connect_threads + connect_messages rows exist.
5. Bob's thread unread_count is incremented by the trigger.
6. Alice's outgoing row STAYS unstamped after send (delivered_at is
   set on the explicit MSG_TEXT_DELIVERED ack, not server-store).
7. Read receipt round-trip clears Bob's unread + routes EVENT_TEXT_READ
   to Alice.
8. Delete + edit round-trips persist + route the right events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import (
    handle_message_action,
    handle_message_envelope,
)
from augmentum.connect.message_store import (
    get_message,
    get_thread,
)
from augmentum.connect.protocol import (
    EVENT_TEXT_DELETE,
    EVENT_TEXT_EDIT,
    EVENT_TEXT_READ,
    EVENT_TEXT_RECEIVED,
    EVENT_TYPING_START,
    EVENT_TYPING_STOP,
    MSG_TEXT_DELETE,
    MSG_TEXT_EDIT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
    ConnectEnvelope,
)
from augmentum.notifications import (
    IMPORTANCE_DEFAULT,
)
from augmentum.notifications.hub import NotificationHub
from augmentum.notifications.store import get_notification

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
    """Records sent payloads with async-shaped send_text."""

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
    assert ws.sent, "expected at least one sent frame"
    return json.loads(ws.sent[-1])


def _envelopes_by_kind(ws: FakeWS, kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in ws.sent:
        parsed = json.loads(raw)
        if parsed.get("type") == kind:
            out.append(parsed)
    return out


# ── Happy path: Alice sends, Bob receives + notif + thread bumped ──


@pytest.mark.asyncio
async def test_text_send_full_loop(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling = FakeWS()
    bob_signaling = FakeWS()
    bob_notification = FakeWS()

    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )
    alice_signaling.sent.clear()
    bob_signaling.sent.clear()

    await notification_hub.attach(ws=bob_notification, user_id=BOB_ID)

    # ── Step 1: Alice sends a message
    send_env = ConnectEnvelope(
        kind="msg",
        verb=MSG_TEXT_SEND,
        corr_id="alice-msg-1",
        peer=BOB_DID,
        data={
            "thread_id": "thread-alice-bob",
            "message_id": "msg-001",
            "body": "hey, you around?",
            "format": "plain",
        },
    )
    result = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=send_env,
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
    )
    assert result.error_code == "", f"unexpected error {result.error_message}"
    assert result.routed == 1, "expected 1 frame to Bob"
    assert result.message_id == "msg-001"
    assert result.thread_id == "thread-alice-bob"

    # ── Step 2: Bob's signaling WS got EVENT_TEXT_RECEIVED
    bob_msg_frames = _envelopes_by_kind(bob_signaling, "event")
    assert len(bob_msg_frames) == 1
    frame = bob_msg_frames[0]
    assert frame["event"] == EVENT_TEXT_RECEIVED
    assert frame["from"] == ALICE_DID
    assert frame["data"]["body"] == "hey, you around?"
    assert frame["data"]["sender_did"] == ALICE_DID
    assert frame["data"]["thread_id"] == "thread-alice-bob"
    assert frame["data"]["message_id"] == "msg-001"

    # Alice should not see her own message reflected back.
    assert _envelopes_by_kind(alice_signaling, "event") == []

    # ── Step 3: Bob got a notification push
    assert len(bob_notification.sent) == 1
    push = json.loads(bob_notification.sent[0])
    assert push["type"] == "notification"
    n = push["notification"]
    assert n["channel_id"] == "connect.message"
    assert n["importance"] == IMPORTANCE_DEFAULT
    assert n["thread_id"] == "thread-alice-bob"
    assert n["payload"]["thread_id"] == "thread-alice-bob"
    assert n["payload"]["sender_user_id"] == ALICE_ID
    action_ids = {a["id"] for a in n["actions"]}
    assert action_ids == {"open_thread", "mark_read"}

    # ── Step 4: Both users have rows
    alice_msg = await get_message(conn, message_id="msg-001", user_id=ALICE_ID)
    bob_msg = await get_message(conn, message_id="msg-001", user_id=BOB_ID)
    assert alice_msg is not None
    assert bob_msg is not None
    assert alice_msg.body == bob_msg.body == "hey, you around?"
    assert alice_msg.sender_did == bob_msg.sender_did == ALICE_DID
    # delivered_at on sender stays None until Bob ACKs via
    # MSG_TEXT_DELIVERED — server-store time is not delivery time.
    assert alice_msg.delivered_at is None

    # ── Step 5: Bob's thread row was created + unread bumped
    bob_thread = await get_thread(
        conn, thread_id="thread-alice-bob", user_id=BOB_ID,
    )
    assert bob_thread is not None
    assert bob_thread.peer_did == ALICE_DID
    assert bob_thread.unread_count == 1
    assert "hey" in bob_thread.last_message_preview

    # Alice's own thread row stays at 0 unread.
    alice_thread = await get_thread(
        conn, thread_id="thread-alice-bob", user_id=ALICE_ID,
    )
    assert alice_thread.unread_count == 0


@pytest.mark.asyncio
async def test_text_send_when_recipient_offline_still_persists(conn) -> None:
    """No live WS for Bob — message still lands in DB and the
    notification publishes for next-load surfacing."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )

    result = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, corr_id="x",
            peer=BOB_DID, data={"body": "offline ping"},
        ),
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
    )
    assert result.routed == 0
    assert result.notification_id  # still published

    # Bob's row exists.
    bob_msg = await get_message(
        conn, message_id=result.message_id, user_id=BOB_ID,
    )
    assert bob_msg is not None
    assert bob_msg.body == "offline ping"


# ── Read receipts ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_read_routes_receipt_and_clears_unread(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling = FakeWS()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    # 1) Alice sends a message
    send_res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID,
            data={"body": "hi"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert send_res.error_code == ""
    alice_signaling.sent.clear()
    bob_signaling.sent.clear()

    # 2) Bob marks the thread read
    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_READ,
            peer=ALICE_DID,
            data={
                "thread_id": send_res.thread_id,
                "last_read_message_id": send_res.message_id,
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.error_code == ""
    assert res.routed == 1

    # 3) Alice sees the receipt
    alice_frame = _last_envelope(alice_signaling)
    assert alice_frame["event"] == EVENT_TEXT_READ
    assert alice_frame["data"]["thread_id"] == send_res.thread_id
    assert alice_frame["data"]["last_read_message_id"] == send_res.message_id
    assert alice_frame["data"]["marked"] == 1

    # 4) Bob's unread cleared
    bob_thread = await get_thread(
        conn, thread_id=send_res.thread_id, user_id=BOB_ID,
    )
    assert bob_thread.unread_count == 0


@pytest.mark.asyncio
async def test_text_read_missing_thread_id_errors(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_READ,
            peer=ALICE_DID, data={},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.error_code == "missing_thread_id"


# ── Delete ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_delete_propagates_to_both_sides(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling = FakeWS()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    send_res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "delete me"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    alice_signaling.sent.clear()
    bob_signaling.sent.clear()

    del_res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE,
            peer=BOB_DID,
            data={
                "thread_id": send_res.thread_id,
                "message_id": send_res.message_id,
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert del_res.error_code == ""
    assert del_res.routed == 1

    bob_frame = _last_envelope(bob_signaling)
    assert bob_frame["event"] == EVENT_TEXT_DELETE
    assert bob_frame["data"]["message_id"] == send_res.message_id

    bob_msg = await get_message(
        conn, message_id=send_res.message_id, user_id=BOB_ID,
    )
    assert bob_msg.body == ""
    assert bob_msg.deleted_at is not None


@pytest.mark.asyncio
async def test_text_delete_refuses_peer_messages(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    await connect_hub.attach(
        ws=FakeWS(), user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=FakeWS(), user_id=BOB_ID, user_did=BOB_DID,
    )

    send_res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "Alice's message"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )

    # Bob tries to delete Alice's message — must refuse.
    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE,
            peer=ALICE_DID,
            data={
                "thread_id": send_res.thread_id,
                "message_id": send_res.message_id,
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.error_code == "message_not_owned"


# ── Edit ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_edit_updates_both_sides(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    await connect_hub.attach(
        ws=FakeWS(), user_id=ALICE_ID, user_did=ALICE_DID,
    )
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    send_res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "typoo"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    bob_signaling.sent.clear()

    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_EDIT,
            peer=BOB_DID,
            data={
                "thread_id": send_res.thread_id,
                "message_id": send_res.message_id,
                "body": "typo",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == ""
    assert res.routed == 1

    bob_frame = _last_envelope(bob_signaling)
    assert bob_frame["event"] == EVENT_TEXT_EDIT
    assert bob_frame["data"]["body"] == "typo"

    bob_msg = await get_message(
        conn, message_id=send_res.message_id, user_id=BOB_ID,
    )
    assert bob_msg.body == "typo"
    assert bob_msg.edited_at is not None


# ── Action handler ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_action_handler_open_thread_marks_read(conn) -> None:
    """The 'open_thread' notification action clears Bob's unread."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    await connect_hub.attach(
        ws=FakeWS(), user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=FakeWS(), user_id=BOB_ID, user_did=BOB_DID,
    )

    send_res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "hi"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    notif = await get_notification(
        conn, user_id=BOB_ID, notification_id=send_res.notification_id,
    )
    assert notif is not None

    # Build a request with app.state shape that handle_message_action expects.
    request = MagicMock()
    backend = MagicMock()
    backend.conn = conn

    from augmentum.state.backends.sqlite import SQLiteBackend
    backend.__class__ = SQLiteBackend
    sm = MagicMock()
    sm.backend = backend
    request.app.state.state_manager = sm

    res = await handle_message_action(notif, "open_thread", request)
    assert res["status"] == "open_thread"
    assert res["thread_id"] == send_res.thread_id
    # Unread cleared after open_thread.
    bob_thread = await get_thread(
        conn, thread_id=send_res.thread_id, user_id=BOB_ID,
    )
    assert bob_thread.unread_count == 0


@pytest.mark.asyncio
async def test_action_handler_unknown_action_errors(conn) -> None:
    notif = MagicMock()
    notif.payload = {"thread_id": "t1"}
    request = MagicMock()
    res = await handle_message_action(notif, "made_up", request)
    assert res["status"] == "error"


# ── Empty messages ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_body_without_attachment_errors(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": ""},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == "message_empty"


@pytest.mark.asyncio
async def test_typing_start_routes_to_peer(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling = FakeWS()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )
    bob_signaling.sent.clear()

    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_START,
            peer=BOB_DID, data={"thread_id": "t1"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == ""
    assert res.routed == 1

    frame = _last_envelope(bob_signaling)
    assert frame["event"] == EVENT_TYPING_START
    assert frame["data"]["thread_id"] == "t1"
    assert frame["from"] == ALICE_DID


@pytest.mark.asyncio
async def test_typing_stop_routes_to_peer(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_STOP,
            peer=BOB_DID, data={"thread_id": "t1"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == ""
    frame = _last_envelope(bob_signaling)
    assert frame["event"] == EVENT_TYPING_STOP


@pytest.mark.asyncio
async def test_typing_missing_thread_id_errors(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_START,
            peer=BOB_DID, data={},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == "missing_thread_id"


@pytest.mark.asyncio
async def test_typing_does_not_publish_notification(conn) -> None:
    """Typing is ephemeral — no notification row should land in Bob's feed."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_notif = FakeWS()
    await notification_hub.attach(ws=bob_notif, user_id=BOB_ID)

    res = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_START,
            peer=BOB_DID, data={"thread_id": "t1"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == ""
    assert bob_notif.sent == []


@pytest.mark.asyncio
async def test_invalid_peer_did_errors(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer="not-a-did", data={"body": "hi"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == "peer_did_invalid"
