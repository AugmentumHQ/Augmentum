"""Connect block enforcement — recipient-side silent block semantics.

When user B has blocked user A (``connect_contacts.blocked = 1`` on
B's row keyed by A's DID), inbound traffic from A is silently
dropped at the routing layer:

* SEND   — sender's row persists, recipient's mirror skipped, no WS
  event, no notification. Sender's UI shows "sent" but the message
  never advances to "delivered" or "read".
* TYPING — WS event dropped (no ephemeral leak).
* READ   — receipt to the blocked sender dropped.
* DELIVERED — ack to the blocked sender dropped (and no stamp).
* REACT — recipient mirror row + WS dropped (sender's own row still
  writes so their UI is internally consistent).
* EDIT / DELETE — recipient mirror skipped; WS dropped.
* CALL INVITE — recipient row skipped, no WS, no notification;
  caller's row is transitioned straight to ``missed`` so the ringing
  UI clears quickly without revealing the block.

Mirrors test_connect_message_routing.py's substrate (FakeWS + real
ConnectHub + NotificationHub + migrations 219/221) rather than mocking
the store, so the assertions are end-to-end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from augmentum.connect.call_routing import handle_signaling_envelope
from augmentum.connect.contact_store import add_contact, set_blocked
from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.message_store import get_message
from augmentum.connect.protocol import (
    MSG_INVITE,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_REACT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub
from augmentum.notifications.store import list_for_user

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text(encoding="utf-8")
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text(encoding="utf-8")
REACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS connect_message_reactions (
    message_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    reactor_did  TEXT NOT NULL,
    emoji        TEXT NOT NULL,
    reacted_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (message_id, user_id, reactor_did, emoji)
);
"""


ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID = local_did_for(ALICE_ID)
BOB_DID = local_did_for(BOB_ID)


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        await c.executescript(REACTIONS_DDL)
        await c.commit()
        yield c


async def _bob_blocks_alice(conn: Any) -> None:
    row = await add_contact(
        conn, user_id=BOB_ID, peer_did=ALICE_DID,
        discovery_source="handle_added",
    )
    ok = await set_blocked(
        conn, user_id=BOB_ID, contact_id=row.contact_id, blocked=True,
    )
    assert ok, "set_blocked should have updated the row"


# ── SEND ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_from_blocked_sender_silently_drops(conn) -> None:
    """Alice sends to Bob, but Bob has blocked her. Alice's row
    persists (so her UI sees "sent"), but Bob's mirror does not
    exist, no WS event fires, and no notification is published."""

    await _bob_blocks_alice(conn)

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
    await notification_hub.attach(ws=bob_notification, user_id=BOB_ID)
    alice_signaling.sent.clear()
    bob_signaling.sent.clear()
    bob_notification.sent.clear()

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            corr_id="x", peer=BOB_DID,
            data={
                "thread_id": "t-1",
                "message_id": "m-1",
                "body": "you there?",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )

    # Sender's perspective: success, but routed=0.
    assert res.error_code == "", f"unexpected error '{res.error_message}'"
    assert res.routed == 0
    assert res.notification_id == ""
    assert res.message_id == "m-1"

    # Alice's row persists.
    alice_msg = await get_message(conn, message_id="m-1", user_id=ALICE_ID)
    assert alice_msg is not None
    assert alice_msg.body == "you there?"

    # Bob's mirror does NOT exist.
    bob_msg = await get_message(conn, message_id="m-1", user_id=BOB_ID)
    assert bob_msg is None

    # No WS event delivered to Bob.
    assert bob_signaling.sent == []

    # No notification published to Bob.
    assert bob_notification.sent == []
    notifs = await list_for_user(conn, user_id=BOB_ID)
    assert notifs == []


@pytest.mark.asyncio
async def test_send_when_no_contact_row_is_not_blocked(conn) -> None:
    """Block requires an explicit contact row. A never-contacted DID
    is implicitly unblocked — this is the happy-path regression that
    proves the block check doesn't accidentally drop all traffic."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            corr_id="y", peer=BOB_DID,
            data={"body": "first contact"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.routed == 1, "first-contact send should still route"

    bob_msg = await get_message(
        conn, message_id=res.message_id, user_id=BOB_ID,
    )
    assert bob_msg is not None


@pytest.mark.asyncio
async def test_unblock_restores_traffic(conn) -> None:
    """Block → unblock → send routes again. Confirms the check reads
    the live flag rather than caching."""

    row = await add_contact(
        conn, user_id=BOB_ID, peer_did=ALICE_DID,
        discovery_source="handle_added",
    )
    await set_blocked(
        conn, user_id=BOB_ID, contact_id=row.contact_id, blocked=True,
    )

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    # Blocked send drops.
    res_blocked = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID,
            data={"body": "before"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res_blocked.routed == 0

    # Unblock.
    await set_blocked(
        conn, user_id=BOB_ID, contact_id=row.contact_id, blocked=False,
    )

    # Next send routes normally.
    res_open = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID,
            data={"body": "after"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res_open.routed == 1


# ── TYPING / READ / DELIVERED ─────────────────────────────────────


@pytest.mark.asyncio
async def test_typing_from_blocked_sender_drops(conn) -> None:
    await _bob_blocks_alice(conn)

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_START,
            peer=BOB_DID,
            data={"thread_id": "t-1"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.routed == 0
    assert bob_signaling.sent == []


@pytest.mark.asyncio
async def test_read_receipt_to_blocker_drops(conn) -> None:
    """If Alice has blocked Bob, Bob's read receipts on Alice's old
    messages don't reach Alice. The block is the user expressing
    "no signal from this peer" — receipts count as signal."""

    row = await add_contact(
        conn, user_id=ALICE_ID, peer_did=BOB_DID,
        discovery_source="handle_added",
    )
    await set_blocked(
        conn, user_id=ALICE_ID, contact_id=row.contact_id, blocked=True,
    )

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_READ,
            peer=ALICE_DID,
            data={
                "thread_id": "t-1",
                "last_read_message_id": "m-1",
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.routed == 0
    assert alice_signaling.sent == []


@pytest.mark.asyncio
async def test_delivered_ack_to_blocker_drops(conn) -> None:
    """If Alice has blocked Bob, Bob's delivery acks on Alice's old
    messages don't reach Alice (no stamp + no WS event)."""

    # Set up a real message Alice sent to Bob before any block.
    row = await add_contact(
        conn, user_id=ALICE_ID, peer_did=BOB_DID,
        discovery_source="handle_added",
    )

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
            peer=BOB_DID, data={"body": "first"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert send_res.routed == 1

    # Now Alice blocks Bob.
    await set_blocked(
        conn, user_id=ALICE_ID, contact_id=row.contact_id, blocked=True,
    )
    alice_signaling.sent.clear()

    # Bob acks the old message. Routing target is Alice (the blocker).
    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED,
            peer=ALICE_DID,
            data={
                "thread_id": send_res.thread_id,
                "message_ids": [send_res.message_id],
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.routed == 0
    assert alice_signaling.sent == []

    # Alice's row's delivered_at stays None (no stamp).
    alice_msg = await get_message(
        conn, message_id=send_res.message_id, user_id=ALICE_ID,
    )
    assert alice_msg is not None
    assert alice_msg.delivered_at is None


# ── REACT / EDIT / DELETE ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_react_from_blocked_sender_skips_mirror(conn) -> None:
    """Alice reacts on a message in a thread where Bob has blocked
    her. Her own reaction row writes (so her UI is consistent), but
    Bob's mirror doesn't and the WS event is dropped."""

    # Seed a pre-block thread so a message_id exists for the react.
    seed_res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "pre-block"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    msg_id = seed_res.message_id

    await _bob_blocks_alice(conn)

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_REACT,
            peer=BOB_DID,
            data={
                "thread_id": seed_res.thread_id,
                "message_id": msg_id,
                "emoji": "👍",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.routed == 0
    assert bob_signaling.sent == []

    # Alice's reaction exists.
    cur = await conn.execute(
        "SELECT COUNT(*) FROM connect_message_reactions "
        "WHERE message_id = ? AND user_id = ?",
        (msg_id, ALICE_ID),
    )
    (count_alice,) = await cur.fetchone()
    assert count_alice == 1

    # Bob's mirror doesn't.
    cur = await conn.execute(
        "SELECT COUNT(*) FROM connect_message_reactions "
        "WHERE message_id = ? AND user_id = ?",
        (msg_id, BOB_ID),
    )
    (count_bob,) = await cur.fetchone()
    assert count_bob == 0


@pytest.mark.asyncio
async def test_edit_from_blocked_sender_freezes_mirror(conn) -> None:
    """Alice edits her own message after Bob blocked her. Alice's
    row updates; Bob's mirror stays at the original body."""

    seed_res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "original"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    msg_id = seed_res.message_id

    await _bob_blocks_alice(conn)

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_EDIT,
            peer=BOB_DID,
            data={
                "thread_id": seed_res.thread_id,
                "message_id": msg_id,
                "body": "edited",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.routed == 0
    assert bob_signaling.sent == []

    alice_msg = await get_message(conn, message_id=msg_id, user_id=ALICE_ID)
    bob_msg = await get_message(conn, message_id=msg_id, user_id=BOB_ID)
    assert alice_msg.body == "edited"
    assert bob_msg.body == "original"


@pytest.mark.asyncio
async def test_delete_from_blocked_sender_freezes_mirror(conn) -> None:
    seed_res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": "to-delete"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    msg_id = seed_res.message_id

    await _bob_blocks_alice(conn)

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE,
            peer=BOB_DID,
            data={"thread_id": seed_res.thread_id, "message_id": msg_id},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.routed == 0
    assert bob_signaling.sent == []

    alice_msg = await get_message(conn, message_id=msg_id, user_id=ALICE_ID)
    bob_msg = await get_message(conn, message_id=msg_id, user_id=BOB_ID)
    assert alice_msg.deleted_at is not None
    assert bob_msg.deleted_at is None


# ── CALL INVITE ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_invite_from_blocked_caller_marks_missed(conn) -> None:
    """Alice calls Bob. Bob has blocked her. Bob never sees an
    incoming-call event or notification. Alice's call_sessions row
    transitions ringing → missed immediately so her dialer clears."""

    await _bob_blocks_alice(conn)

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
    await notification_hub.attach(ws=bob_notification, user_id=BOB_ID)
    alice_signaling.sent.clear()
    bob_signaling.sent.clear()
    bob_notification.sent.clear()

    res = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE,
            corr_id="call-1", peer=BOB_DID,
            data={"call_id": "c-1", "modalities": "audio,video"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id="alice-party-1",
    )

    assert res.error_code == "", f"unexpected error '{res.error_message}'"
    assert res.routed == 0
    assert res.notification_id == ""

    # Bob's perspective: no WS event, no notification.
    assert bob_signaling.sent == []
    assert bob_notification.sent == []
    notifs = await list_for_user(conn, user_id=BOB_ID)
    assert notifs == []

    # Bob has no call_sessions row — block left no trace.
    cur = await conn.execute(
        "SELECT COUNT(*) FROM call_sessions WHERE call_id = ? AND user_id = ?",
        ("c-1", BOB_ID),
    )
    (count_bob,) = await cur.fetchone()
    assert count_bob == 0

    # Alice's row exists in state="missed".
    cur = await conn.execute(
        "SELECT state, end_reason FROM call_sessions "
        "WHERE call_id = ? AND user_id = ?",
        ("c-1", ALICE_ID),
    )
    row = await cur.fetchone()
    assert row is not None, "alice's call_sessions row should exist"
    state, end_reason = row
    assert state == "missed"
    assert end_reason == "no_answer"
