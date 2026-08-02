"""Connect message store unit tests.

Exercises the DAO directly with an in-memory SQLite using migration 219
verbatim (so trigger behavior, composite PKs, and unique constraints
all match production).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from augmentum.connect.message_store import (
    edit_message,
    get_message,
    get_or_create_thread,
    get_thread,
    insert_message,
    list_messages_for_thread,
    list_threads_for_user,
    mark_thread_read,
    new_message_id,
    new_thread_id,
    set_thread_flag,
    soft_delete_message,
    stamp_delivered,
)


CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text()


ALICE = "alice"
BOB = "bob"
ALICE_DID = "alice@this-instance"
BOB_DID = "bob@this-instance"


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(CONNECT_MIGRATION)
        await c.commit()
        yield c


# ── Thread primitives ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_or_create_thread_is_idempotent(conn) -> None:
    """Repeated creates with the same (user_id, peer_did) collapse."""

    tid1 = new_thread_id()
    tid2 = new_thread_id()
    t1 = await get_or_create_thread(
        conn, thread_id=tid1, user_id=ALICE, peer_did=BOB_DID,
        peer_display_name="Bob",
    )
    t2 = await get_or_create_thread(
        conn, thread_id=tid2, user_id=ALICE, peer_did=BOB_DID,
    )
    # The unique-pair index demotes the second create — both calls
    # return the row created by the first.
    assert t1.thread_id == t2.thread_id == tid1
    assert t1.peer_display_name == "Bob"
    # Second call doesn't overwrite the display name (INSERT OR IGNORE).
    assert t2.peer_display_name == "Bob"


@pytest.mark.asyncio
async def test_thread_isolation_per_user(conn) -> None:
    """Alice and Bob each get their own thread row for the same pair."""

    tid = new_thread_id()
    a = await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    b = await get_or_create_thread(
        conn, thread_id=tid, user_id=BOB, peer_did=ALICE_DID,
    )
    assert a.user_id == ALICE
    assert b.user_id == BOB
    assert a.thread_id == b.thread_id == tid
    # Lookups by (thread_id, user_id) scope correctly.
    assert (await get_thread(conn, thread_id=tid, user_id=ALICE)).user_id == ALICE
    assert (await get_thread(conn, thread_id=tid, user_id=BOB)).user_id == BOB


# ── Message primitives ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_message_is_idempotent(conn) -> None:
    """Re-inserting the same (message_id, user_id) is a no-op."""

    tid = new_thread_id()
    mid = new_message_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    first = await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID, body="hello",
    )
    second = await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID, body="hello again",
    )
    assert first is True
    assert second is False
    got = await get_message(conn, message_id=mid, user_id=ALICE)
    assert got is not None
    assert got.body == "hello"  # original body preserved


@pytest.mark.asyncio
async def test_insert_message_bumps_unread_and_tail(conn) -> None:
    """Trigger keeps last_message_at + unread_count fresh on Bob's side
    when Alice's message lands in Bob's thread row."""

    tid = new_thread_id()
    mid = new_message_id()
    # Bob's thread row — peer_did is Alice from Bob's perspective.
    await get_or_create_thread(
        conn, thread_id=tid, user_id=BOB, peer_did=ALICE_DID,
    )
    await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=BOB,
        sender_did=ALICE_DID,  # Alice sent, Bob received
        body="ping",
    )
    bob = await get_thread(conn, thread_id=tid, user_id=BOB)
    assert bob is not None
    assert bob.unread_count == 1
    assert bob.last_message_preview == "ping"
    assert bob.last_message_at  # stamped


@pytest.mark.asyncio
async def test_insert_outgoing_does_not_bump_unread(conn) -> None:
    """Alice's own outgoing message must not touch her unread counter."""

    tid = new_thread_id()
    mid = new_message_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID,  # Alice's own outgoing
        body="hi bob",
    )
    alice = await get_thread(conn, thread_id=tid, user_id=ALICE)
    assert alice is not None
    assert alice.unread_count == 0
    assert alice.last_message_preview == "hi bob"


@pytest.mark.asyncio
async def test_list_messages_newest_first_with_pagination(conn) -> None:
    tid = new_thread_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    ids = []
    for i in range(5):
        mid = new_message_id()
        ids.append(mid)
        await insert_message(
            conn, message_id=mid, thread_id=tid, user_id=ALICE,
            sender_did=ALICE_DID, body=f"msg {i}",
        )
        # Sleep so sent_at values are deterministically distinct on
        # Windows hosts (15.6ms clock resolution).
        await asyncio.sleep(0.02)

    first_page = await list_messages_for_thread(
        conn, thread_id=tid, user_id=ALICE, limit=3,
    )
    assert [m.body for m in first_page] == ["msg 4", "msg 3", "msg 2"]

    cursor = first_page[-1].sent_at
    next_page = await list_messages_for_thread(
        conn, thread_id=tid, user_id=ALICE, limit=3,
        before_sent_at=cursor,
    )
    assert [m.body for m in next_page] == ["msg 1", "msg 0"]


@pytest.mark.asyncio
async def test_mark_thread_read_clears_unread(conn) -> None:
    tid = new_thread_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=BOB, peer_did=ALICE_DID,
    )
    for i in range(3):
        await insert_message(
            conn, message_id=new_message_id(), thread_id=tid,
            user_id=BOB, sender_did=ALICE_DID, body=f"m{i}",
        )
        await asyncio.sleep(0.01)

    pre = await get_thread(conn, thread_id=tid, user_id=BOB)
    assert pre.unread_count == 3

    marked = await mark_thread_read(
        conn, thread_id=tid, user_id=BOB,
    )
    assert marked == 3
    post = await get_thread(conn, thread_id=tid, user_id=BOB)
    assert post.unread_count == 0

    # All rows now have read_at stamped.
    msgs = await list_messages_for_thread(
        conn, thread_id=tid, user_id=BOB,
    )
    for m in msgs:
        assert m.read_at is not None


@pytest.mark.asyncio
async def test_mark_thread_read_bounded_by_last_message_id(conn) -> None:
    """Receipt up to a specific message leaves later messages unread."""

    tid = new_thread_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=BOB, peer_did=ALICE_DID,
    )
    mids = []
    for i in range(4):
        mid = new_message_id()
        mids.append(mid)
        await insert_message(
            conn, message_id=mid, thread_id=tid, user_id=BOB,
            sender_did=ALICE_DID, body=f"m{i}",
        )
        await asyncio.sleep(0.01)

    # Mark only up to the second message read.
    marked = await mark_thread_read(
        conn, thread_id=tid, user_id=BOB,
        last_read_message_id=mids[1],
    )
    assert marked == 2

    msgs = await list_messages_for_thread(
        conn, thread_id=tid, user_id=BOB,
    )
    by_id = {m.message_id: m for m in msgs}
    assert by_id[mids[0]].read_at is not None
    assert by_id[mids[1]].read_at is not None
    assert by_id[mids[2]].read_at is None
    assert by_id[mids[3]].read_at is None


@pytest.mark.asyncio
async def test_soft_delete_clears_body_but_keeps_row(conn) -> None:
    tid = new_thread_id()
    mid = new_message_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID, body="oops",
    )
    deleted = await soft_delete_message(
        conn, message_id=mid, user_id=ALICE,
    )
    assert deleted is True
    msg = await get_message(conn, message_id=mid, user_id=ALICE)
    assert msg is not None  # row still there
    assert msg.body == ""
    assert msg.deleted_at is not None
    # Second delete is idempotent (no-op).
    second = await soft_delete_message(
        conn, message_id=mid, user_id=ALICE,
    )
    assert second is False


@pytest.mark.asyncio
async def test_edit_stamps_edited_at(conn) -> None:
    tid = new_thread_id()
    mid = new_message_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID, body="initial",
    )
    ok = await edit_message(
        conn, message_id=mid, user_id=ALICE, body="updated",
    )
    assert ok is True
    msg = await get_message(conn, message_id=mid, user_id=ALICE)
    assert msg.body == "updated"
    assert msg.edited_at is not None


@pytest.mark.asyncio
async def test_edit_skips_deleted_messages(conn) -> None:
    tid = new_thread_id()
    mid = new_message_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID, body="oops",
    )
    await soft_delete_message(conn, message_id=mid, user_id=ALICE)
    ok = await edit_message(
        conn, message_id=mid, user_id=ALICE, body="too late",
    )
    assert ok is False
    msg = await get_message(conn, message_id=mid, user_id=ALICE)
    assert msg.body == ""


@pytest.mark.asyncio
async def test_stamp_delivered_idempotent(conn) -> None:
    tid = new_thread_id()
    mid = new_message_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    await insert_message(
        conn, message_id=mid, thread_id=tid, user_id=ALICE,
        sender_did=ALICE_DID, body="hi",
    )
    first = await stamp_delivered(conn, message_id=mid, user_id=ALICE)
    second = await stamp_delivered(conn, message_id=mid, user_id=ALICE)
    assert first is True
    assert second is False


# ── Thread flags ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_thread_flag_toggles(conn) -> None:
    tid = new_thread_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    assert await set_thread_flag(
        conn, thread_id=tid, user_id=ALICE, flag="muted", value=True,
    )
    t = await get_thread(conn, thread_id=tid, user_id=ALICE)
    assert t.muted is True
    assert t.pinned is False
    assert await set_thread_flag(
        conn, thread_id=tid, user_id=ALICE, flag="pinned", value=True,
    )
    t = await get_thread(conn, thread_id=tid, user_id=ALICE)
    assert t.pinned is True


@pytest.mark.asyncio
async def test_set_unknown_flag_raises(conn) -> None:
    tid = new_thread_id()
    await get_or_create_thread(
        conn, thread_id=tid, user_id=ALICE, peer_did=BOB_DID,
    )
    with pytest.raises(ValueError):
        await set_thread_flag(
            conn, thread_id=tid, user_id=ALICE,
            flag="not_a_thing", value=True,
        )


@pytest.mark.asyncio
async def test_list_threads_returns_pinned_first(conn) -> None:
    pinned = new_thread_id()
    plain = new_thread_id()
    await get_or_create_thread(
        conn, thread_id=plain, user_id=ALICE, peer_did="bob@this-instance",
    )
    await asyncio.sleep(0.02)
    # Insert a message into the plain thread so it has a more recent
    # last_message_at than the pinned one. Pinned should still come first.
    await insert_message(
        conn, message_id=new_message_id(), thread_id=plain,
        user_id=ALICE, sender_did=ALICE_DID, body="recent",
    )
    await get_or_create_thread(
        conn, thread_id=pinned, user_id=ALICE,
        peer_did="charlie@this-instance",
    )
    await set_thread_flag(
        conn, thread_id=pinned, user_id=ALICE,
        flag="pinned", value=True,
    )
    threads = await list_threads_for_user(conn, user_id=ALICE)
    assert threads[0].thread_id == pinned
