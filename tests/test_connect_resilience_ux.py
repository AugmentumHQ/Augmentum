"""Connect resilience + UX edge-case tests.

Failure modes a user can realistically hit that would degrade smoothness
or expose inconsistent state. Covers gaps surfaced in the
2026-06-04 coverage audit:

* Delivered-ack batch shape: empty / non-list / 200+ ids (server caps)
* Edit-after-delete / double-delete idempotency (no redundant WS event)
* Hangup races: caller hangs up before callee accepts, simultaneous
  bidirectional hangup. Both sides converge to a consistent terminal
  state.
* DID parsing edge cases: empty, missing instance, missing user, spaces.
* Catch-up cursor for never-opened threads (no crash, empty list).

Same end-to-end harness shape as test_connect_message_routing.py
(real ConnectHub + NotificationHub + migrations 219/221), so failures
implicate the substrate, not a mock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from augmentum.connect.call_routing import handle_signaling_envelope
from augmentum.connect.contacts import local_did_for, resolve_peer_did
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.message_store import (
    get_message,
    list_messages_for_thread,
)
from augmentum.connect.protocol import (
    MSG_DECLINE,
    MSG_HANGUP,
    MSG_INVITE,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_SEND,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text(encoding="utf-8")
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text(encoding="utf-8")


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
        await c.commit()
        yield c


async def _seed_message(conn: Any, body: str = "seed") -> dict:
    """Helper: send a message from Alice to Bob and return the result."""

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer=BOB_DID, data={"body": body},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    return {
        "thread_id": res.thread_id,
        "message_id": res.message_id,
    }


# ── DELIVERED ack batch shape ─────────────────────────────────────


@pytest.mark.asyncio
async def test_delivered_ack_with_empty_message_ids_errors(conn) -> None:
    """A delivered ack envelope with an empty list is malformed —
    the receiver client shouldn't be sending it. The router must
    reject rather than silently no-op (which would mask a buggy
    client that THINKS it acked something)."""

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED,
            peer=ALICE_DID,
            data={"thread_id": "t-1", "message_ids": []},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.error_code == "missing_message_ids"


@pytest.mark.asyncio
async def test_delivered_ack_with_non_list_message_ids_errors(conn) -> None:
    res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED,
            peer=ALICE_DID,
            data={"thread_id": "t-1", "message_ids": "not-a-list"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert res.error_code == "invalid_message_ids"


@pytest.mark.asyncio
async def test_delivered_ack_caps_at_200_ids(conn) -> None:
    """A buggy or hostile client could send a giant batch. The
    routing layer caps at 200 so a single envelope can't trigger
    tens of thousands of row updates."""

    seed = await _seed_message(conn)

    connect_hub = ConnectHub()
    alice_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )

    # Craft a batch of 500 ids — one real, 499 garbage. The router
    # truncates to the first 200 before stamping; we don't care which
    # 200, we care that the stamp + route only operated on a bounded
    # subset and didn't error.
    big_batch = [seed["message_id"]] + [f"junk-{i}" for i in range(499)]

    res = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED,
            peer=ALICE_DID,
            data={
                "thread_id": seed["thread_id"],
                "message_ids": big_batch,
            },
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )

    assert res.error_code == ""
    assert res.routed == 1
    # The WS frame should carry the truncated list, not the full 500.
    import json
    frame = json.loads(alice_signaling.sent[0])
    routed_ids = frame["data"]["message_ids"]
    assert len(routed_ids) <= 200, f"expected cap at 200, got {len(routed_ids)}"
    # The real id has been included (we put it at index 0).
    assert seed["message_id"] in routed_ids


# ── Edit / Delete race conditions ─────────────────────────────────


@pytest.mark.asyncio
async def test_edit_after_delete_is_refused(conn) -> None:
    """Once a message is soft-deleted, attempting to edit it must be
    refused at the routing layer — otherwise we fire EVENT_TEXT_EDIT
    with a new body while the stored row is a tombstone, leaving the
    recipient's UI showing the edit until reload (live/stored skew)."""

    seed = await _seed_message(conn, body="original")

    # Delete.
    del_res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE,
            peer=BOB_DID,
            data={
                "thread_id": seed["thread_id"],
                "message_id": seed["message_id"],
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert del_res.error_code == ""

    # Now attempt an edit — should be refused.
    edit_res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_EDIT,
            peer=BOB_DID,
            data={
                "thread_id": seed["thread_id"],
                "message_id": seed["message_id"],
                "body": "edited after deletion",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert edit_res.error_code == "message_already_deleted"

    # Stored row stays as a tombstone (body cleared, deleted_at set).
    msg = await get_message(
        conn, message_id=seed["message_id"], user_id=ALICE_ID,
    )
    assert msg is not None
    assert msg.body == ""
    assert msg.deleted_at is not None


@pytest.mark.asyncio
async def test_double_delete_is_idempotent_no_redundant_event(conn) -> None:
    """A user double-tapping the delete confirm shouldn't fire two
    EVENT_TEXT_DELETE envelopes; the second is a no-op (routed=0)."""

    seed = await _seed_message(conn)

    connect_hub = ConnectHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    # First delete fires normally.
    res1 = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE,
            peer=BOB_DID,
            data={
                "thread_id": seed["thread_id"],
                "message_id": seed["message_id"],
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res1.error_code == ""
    assert res1.routed == 1

    # Second delete is a clean no-op — no WS frame, no error.
    bob_signaling.sent.clear()
    res2 = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE,
            peer=BOB_DID,
            data={
                "thread_id": seed["thread_id"],
                "message_id": seed["message_id"],
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res2.error_code == ""
    assert res2.routed == 0
    assert bob_signaling.sent == []


# ── Call hangup races ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_caller_hangup_before_callee_accepts(conn) -> None:
    """Alice rings Bob, then immediately hangs up before he sees the
    invite. Both sides' call_sessions rows should transition to
    state=ended with end_reason set — no row is left dangling at
    ringing/invited."""

    connect_hub = ConnectHub()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    # Alice invites.
    invite_res = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE,
            peer=BOB_DID,
            data={"call_id": "c-1", "modalities": "audio"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id="a-party",
    )
    assert invite_res.error_code == ""

    # Alice immediately hangs up.
    hangup_res = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_HANGUP,
            peer=BOB_DID,
            data={"call_id": "c-1", "reason": "cancelled"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id="a-party",
    )
    assert hangup_res.error_code == ""

    # Both sides ended with the cancel reason.
    cur = await conn.execute(
        "SELECT user_id, state, end_reason FROM call_sessions WHERE call_id = ? ORDER BY user_id",
        ("c-1",),
    )
    rows = await cur.fetchall()
    assert len(rows) == 2
    for user_id, state, end_reason in rows:
        assert state == "ended", f"{user_id} stuck at {state}"
        assert end_reason == "cancelled"


@pytest.mark.asyncio
async def test_simultaneous_bidirectional_hangup_converges(conn) -> None:
    """Both sides hit the hangup button at the same time. Last write
    wins on end_reason; both rows still resolve to state=ended."""

    connect_hub = ConnectHub()
    alice_signaling = FakeWS()
    bob_signaling = FakeWS()
    await connect_hub.attach(
        ws=alice_signaling, user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=bob_signaling, user_id=BOB_ID, user_did=BOB_DID,
    )

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE,
            peer=BOB_DID, data={"call_id": "c-2"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id="a-party",
    )

    # Both hangup envelopes arrive at the dispatcher. Order doesn't
    # matter for the final state.
    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_HANGUP,
            peer=BOB_DID, data={"call_id": "c-2", "reason": "alice_end"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id="a-party",
    )
    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_HANGUP,
            peer=ALICE_DID, data={"call_id": "c-2", "reason": "bob_end"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id="b-party",
    )

    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ?",
        ("c-2",),
    )
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert all(s == "ended" for (s,) in rows)


@pytest.mark.asyncio
async def test_decline_before_offer_marks_call_declined(conn) -> None:
    """Bob declines the invite before any SDP work happens. Both
    sides' rows must reach state=declined."""

    connect_hub = ConnectHub()
    await connect_hub.attach(
        ws=FakeWS(), user_id=ALICE_ID, user_did=ALICE_DID,
    )
    await connect_hub.attach(
        ws=FakeWS(), user_id=BOB_ID, user_did=BOB_DID,
    )

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE,
            peer=BOB_DID, data={"call_id": "c-3"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        sender_party_id="a-party",
    )
    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_DECLINE,
            peer=ALICE_DID, data={"call_id": "c-3"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id="b-party",
    )

    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ? ORDER BY user_id",
        ("c-3",),
    )
    rows = await cur.fetchall()
    assert len(rows) == 2
    assert all(s == "declined" for (s,) in rows)


# ── DID parsing edge cases ────────────────────────────────────────


@pytest.mark.parametrize("bad_did", [
    "",                # empty
    "alice",           # no @
    "@instance.dev",   # no user
    "alice@",          # no instance
    "   ",             # whitespace
])
def test_resolve_peer_did_rejects_malformed(bad_did) -> None:
    resolved = resolve_peer_did(bad_did)
    assert resolved is None, f"{bad_did!r} should not resolve"


def test_resolve_peer_did_uses_rightmost_at(bad_did="alice@inst@nce") -> None:
    """Multiple ``@`` chars resolve by the rightmost one (matches email
    addressing semantics). user_part may still contain ``@`` but the
    auth layer's user_ids never do, so this is a forward-compat
    quirk rather than a hot footgun. Pinned here so any future tightening
    of the parser surfaces as a deliberate change."""

    resolved = resolve_peer_did(bad_did)
    assert resolved is not None
    assert resolved.kind == "fabric"
    assert resolved.address == "nce"


@pytest.mark.asyncio
async def test_send_with_malformed_peer_did_errors(conn) -> None:
    res = await handle_message_envelope(
        conn=conn,
        connect_hub=ConnectHub(),
        notification_hub=NotificationHub(),
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND,
            peer="alice@",  # no instance
            data={"body": "test"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    assert res.error_code == "peer_did_invalid"
    assert "alice@" in res.error_message


# ── Catch-up cursor edge cases ────────────────────────────────────


@pytest.mark.asyncio
async def test_catchup_for_nonexistent_thread_returns_empty(conn) -> None:
    """Listing messages on a thread that doesn't exist must return
    an empty list cleanly. Used by the UI when a sibling-tab broadcast
    arrives for a thread that this tab never opened — we mustn't 500."""

    rows = await list_messages_for_thread(
        conn,
        thread_id="never-existed",
        user_id=ALICE_ID,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_catchup_cursor_at_newest_returns_empty(conn) -> None:
    """Cursor exactly matching the newest message returns empty —
    strict ``>`` not ``>=``. This is the steady-state for an open
    thread: every reconnect after the last message is a no-op."""

    seed = await _seed_message(conn, body="last")
    newest = await get_message(
        conn, message_id=seed["message_id"], user_id=ALICE_ID,
    )
    assert newest is not None

    rows = await list_messages_for_thread(
        conn,
        thread_id=seed["thread_id"],
        user_id=ALICE_ID,
        after_sent_at=newest.sent_at,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_catchup_cursor_future_timestamp_returns_empty(conn) -> None:
    """Clock-skew defense: if the cursor is ahead of all server-stored
    timestamps (e.g. client clock drifted forward), don't crash — just
    return empty. The UI's responsibility to fix the cursor on next
    inbound message; the store layer stays graceful."""

    await _seed_message(conn, body="real-message")

    rows = await list_messages_for_thread(
        conn,
        thread_id="t-real",
        user_id=ALICE_ID,
        # Year 2099 — beyond any real timestamp the test would produce.
        after_sent_at="2099-01-01T00:00:00Z",
    )
    assert rows == []
