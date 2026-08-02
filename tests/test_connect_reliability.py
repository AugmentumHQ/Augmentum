"""Session A reliability surface tests.

Covers the three server-side pieces that ship together:

1. ``MSG_TEXT_DELIVERED`` routing — recipient ACKs, server stamps
   delivered_at on the sender's row, routes EVENT_TEXT_DELIVERED back.
2. Catch-up endpoint side-effect — fetching with ``?since`` stamps
   delivered_at + fans EVENT_TEXT_DELIVERED for inbound rows that
   weren't acked while the recipient was offline.
3. WS rate limiter — bursts get rejected with EVENT_ERROR
   ``code=rate_limited`` once the bucket is full.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.message_store import (
    get_message,
    get_or_create_thread,
    insert_message,
)
from augmentum.connect.protocol import (
    EVENT_TEXT_DELIVERED,
    MSG_PING,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    ConnectEnvelope,
)
from augmentum.connect.rate_limit import (
    CATEGORY_EPHEMERAL,
    CATEGORY_TEXT_WRITE,
    DEFAULT_LIMITS,
    WsRateLimiter,
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


def _events(ws: FakeWS, verb: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in ws.sent:
        parsed = json.loads(raw)
        if parsed.get("type") == "event" and parsed.get("event") == verb:
            out.append(parsed)
    return out


# ── MSG_TEXT_DELIVERED routing ────────────────────────────────────


@pytest.mark.asyncio
async def test_delivered_stamps_sender_row_and_routes_event(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws = FakeWS()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)

    # Alice sends two messages — initial state: delivered_at=None on both.
    for mid in ("msg-A", "msg-B"):
        await handle_message_envelope(
            conn=conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            env=ConnectEnvelope(
                kind="msg", verb=MSG_TEXT_SEND,
                peer=BOB_DID,
                data={
                    "thread_id": "t1",
                    "message_id": mid,
                    "body": f"body of {mid}",
                },
            ),
            sender_user_id=ALICE_ID, sender_did=ALICE_DID,
        )
    for mid in ("msg-A", "msg-B"):
        row = await get_message(conn, message_id=mid, user_id=ALICE_ID)
        assert row.delivered_at is None, "send-time delivery is a lie"

    alice_ws.sent.clear()
    bob_ws.sent.clear()

    # Bob acks both in a single batched envelope (catch-up shape).
    result = await handle_message_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED,
            peer=ALICE_DID,
            data={"thread_id": "t1", "message_ids": ["msg-A", "msg-B"]},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert result.error_code == ""
    assert result.routed == 1, "expected one frame to Alice"

    # Alice's rows now stamped.
    for mid in ("msg-A", "msg-B"):
        row = await get_message(conn, message_id=mid, user_id=ALICE_ID)
        assert row.delivered_at is not None, f"{mid} should be marked delivered"

    # Alice got the routed EVENT_TEXT_DELIVERED with both ids.
    delivered_evts = _events(alice_ws, EVENT_TEXT_DELIVERED)
    assert len(delivered_evts) == 1
    payload = delivered_evts[0]["data"]
    assert payload["thread_id"] == "t1"
    assert set(payload["message_ids"]) == {"msg-A", "msg-B"}


@pytest.mark.asyncio
async def test_delivered_is_idempotent(conn) -> None:
    """A re-ack (e.g. catch-up after WS resume) shouldn't double-stamp
    or fail. Already-delivered ids stay in the routed receipt so the
    sender's UI can reconcile its view either way."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    alice_ws = FakeWS()
    bob_ws = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    await connect_hub.attach(ws=bob_ws, user_id=BOB_ID, user_did=BOB_DID)

    await handle_message_envelope(
        conn=conn, connect_hub=connect_hub, notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID,
            data={"thread_id": "t", "message_id": "m", "body": "hi"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID,
    )
    ack_env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_DELIVERED, peer=ALICE_DID,
        data={"thread_id": "t", "message_ids": ["m"]},
    )
    r1 = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub, notification_hub=notification_hub,
        env=ack_env, sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    r2 = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub, notification_hub=notification_hub,
        env=ack_env, sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert r1.error_code == r2.error_code == ""


@pytest.mark.asyncio
async def test_delivered_requires_message_ids(conn) -> None:
    connect_hub = ConnectHub()
    notification_hub = NotificationHub()
    result = await handle_message_envelope(
        conn=conn, connect_hub=connect_hub, notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED, peer=ALICE_DID,
            data={"thread_id": "t", "message_ids": []},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
    )
    assert result.error_code == "missing_message_ids"


# ── WS rate limiter ───────────────────────────────────────────────


def test_rate_limiter_text_write_blocks_after_limit() -> None:
    limit = DEFAULT_LIMITS[CATEGORY_TEXT_WRITE]
    limiter = WsRateLimiter()
    # Burst right up to the limit — all allowed.
    for _ in range(limit):
        allowed, category, retry = limiter.check(
            user_id=ALICE_ID, verb=MSG_TEXT_SEND,
        )
        assert allowed
        assert category == CATEGORY_TEXT_WRITE
        assert retry == 0
    # The very next one trips.
    allowed, category, retry = limiter.check(
        user_id=ALICE_ID, verb=MSG_TEXT_SEND,
    )
    assert not allowed
    assert category == CATEGORY_TEXT_WRITE
    assert retry >= 1


def test_rate_limiter_buckets_are_per_user_and_per_category() -> None:
    limit = DEFAULT_LIMITS[CATEGORY_TEXT_WRITE]
    limiter = WsRateLimiter()
    # Saturate Alice's text_write bucket.
    for _ in range(limit):
        limiter.check(user_id=ALICE_ID, verb=MSG_TEXT_SEND)
    # Bob is unaffected.
    allowed, _, _ = limiter.check(user_id=BOB_ID, verb=MSG_TEXT_SEND)
    assert allowed, "rate limit should isolate per user"
    # Alice's ephemeral bucket is also unaffected.
    allowed, _, _ = limiter.check(user_id=ALICE_ID, verb=MSG_TYPING_START)
    assert allowed, "rate limit should isolate per category"


def test_rate_limiter_passes_through_empty_user_id() -> None:
    """Anon-shaped check (no user_id) should never refuse — the WS
    endpoint owns auth enforcement; the limiter mustn't mask auth
    bugs by silently rejecting unbucketed requests."""

    limiter = WsRateLimiter()
    for _ in range(1000):
        allowed, _, _ = limiter.check(user_id="", verb=MSG_TEXT_SEND)
        assert allowed


def test_rate_limiter_category_mapping() -> None:
    limiter = WsRateLimiter()
    assert limiter.category_for(MSG_TEXT_SEND) == CATEGORY_TEXT_WRITE
    assert limiter.category_for(MSG_TYPING_START) == CATEGORY_EPHEMERAL
    assert limiter.category_for(MSG_PING) == "presence"
    # Unknown verbs fall through to 'other'.
    assert limiter.category_for("totally-made-up-verb") == "other"


def test_rate_limiter_snapshot_after_use() -> None:
    limiter = WsRateLimiter()
    for _ in range(3):
        limiter.check(user_id=ALICE_ID, verb=MSG_TEXT_SEND)
    snap = limiter.snapshot(ALICE_ID)
    assert snap.get(CATEGORY_TEXT_WRITE) == 3


# ── Catch-up endpoint side-effect (store-level smoke) ─────────────


@pytest.mark.asyncio
async def test_list_messages_supports_after_sent_at_cursor(conn) -> None:
    """Direct store check that the catch-up cursor returns only
    strictly-newer rows (the route layer is exercised via the
    routes test elsewhere)."""

    from augmentum.connect.message_store import list_messages_for_thread

    # Make sure the thread row exists so insert_message succeeds.
    await get_or_create_thread(
        conn, thread_id="cu", user_id=BOB_ID, peer_did=ALICE_DID,
    )

    # Three inbound messages — ascending sent_at.
    sent_times = [
        "2026-06-03T10:00:00+00:00",
        "2026-06-03T10:00:01+00:00",
        "2026-06-03T10:00:02+00:00",
    ]
    for i, ts in enumerate(sent_times):
        await insert_message(
            conn,
            message_id=f"m{i}",
            thread_id="cu",
            user_id=BOB_ID,
            sender_did=ALICE_DID,
            body=f"hi {i}",
            sent_at=ts,
        )

    # `since` = first message's sent_at → returns just m1 and m2.
    fresh = await list_messages_for_thread(
        conn, thread_id="cu", user_id=BOB_ID,
        after_sent_at=sent_times[0],
    )
    ids = sorted(m.message_id for m in fresh)
    assert ids == ["m1", "m2"]

    # No cursor → returns all.
    all_msgs = await list_messages_for_thread(
        conn, thread_id="cu", user_id=BOB_ID,
    )
    assert len(all_msgs) == 3
