"""Connect call store — read-side DAO tests.

Exercises list_calls_for_user / get_call / list_events_for_call /
set_quality_rating against an in-memory SQLite with migration 219
applied.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from augmentum.connect.call_routing import (
    _insert_call_session,
    _log_call_event,
    _update_call_session_state,
)
from augmentum.connect.call_store import (
    get_call,
    list_calls_for_user,
    list_events_for_call,
    set_quality_rating,
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


async def _seed_call(
    conn, *, call_id, user_id, initiator_did, receiver_did,
    state="ringing", modalities="audio",
) -> None:
    await _insert_call_session(
        conn, call_id=call_id, user_id=user_id,
        initiator_did=initiator_did, receiver_did=receiver_did,
        modalities=modalities, state=state,
    )


# ── list_calls_for_user ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_empty(conn) -> None:
    rows = await list_calls_for_user(conn, user_id=ALICE)
    assert rows == []


@pytest.mark.asyncio
async def test_list_returns_user_scoped_rows_only(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    await _seed_call(
        conn, call_id="c2", user_id=BOB,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )

    alice_rows = await list_calls_for_user(conn, user_id=ALICE)
    bob_rows = await list_calls_for_user(conn, user_id=BOB)
    assert [r.call_id for r in alice_rows] == ["c1"]
    assert [r.call_id for r in bob_rows] == ["c2"]


@pytest.mark.asyncio
async def test_list_newest_first_with_state_filter(conn) -> None:
    for i in range(3):
        cid = f"c{i}"
        await _seed_call(
            conn, call_id=cid, user_id=ALICE,
            initiator_did=ALICE_DID, receiver_did=BOB_DID,
            state="ended" if i == 1 else "missed",
        )
        # 20ms keeps each initiated_at distinct on Windows hosts.
        await asyncio.sleep(0.02)

    all_rows = await list_calls_for_user(conn, user_id=ALICE)
    # Newest first
    assert [r.call_id for r in all_rows] == ["c2", "c1", "c0"]

    missed = await list_calls_for_user(
        conn, user_id=ALICE, state_filter="missed",
    )
    assert [r.call_id for r in missed] == ["c2", "c0"]


@pytest.mark.asyncio
async def test_list_pagination_before_cursor(conn) -> None:
    for i in range(4):
        await _seed_call(
            conn, call_id=f"c{i}", user_id=ALICE,
            initiator_did=ALICE_DID, receiver_did=BOB_DID,
        )
        await asyncio.sleep(0.02)

    page1 = await list_calls_for_user(conn, user_id=ALICE, limit=2)
    assert [r.call_id for r in page1] == ["c3", "c2"]

    cursor = page1[-1].initiated_at
    page2 = await list_calls_for_user(
        conn, user_id=ALICE, limit=2, before=cursor,
    )
    assert [r.call_id for r in page2] == ["c1", "c0"]


# ── direction / peer_did / duration ──────────────────────────────


@pytest.mark.asyncio
async def test_outgoing_direction_and_peer(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    row = (await list_calls_for_user(conn, user_id=ALICE))[0]
    d = row.to_dict()
    assert d["direction"] == "outgoing"
    assert d["peer_did"] == BOB_DID


@pytest.mark.asyncio
async def test_incoming_direction_and_peer(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=BOB,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    row = (await list_calls_for_user(conn, user_id=BOB))[0]
    d = row.to_dict()
    assert d["direction"] == "incoming"
    assert d["peer_did"] == ALICE_DID


@pytest.mark.asyncio
async def test_duration_seconds_computed(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    await _update_call_session_state(
        conn, call_id="c1", user_id=ALICE, state="connected",
    )
    # Force-stamp ended_at by transitioning to ended.
    await _update_call_session_state(
        conn, call_id="c1", user_id=ALICE,
        state="ended", end_reason="hangup",
    )
    row = await get_call(conn, call_id="c1", user_id=ALICE)
    assert row is not None
    duration = row.to_dict()["duration_seconds"]
    # Very fast in tests but always >= 0.
    assert duration is not None
    assert duration >= 0


@pytest.mark.asyncio
async def test_duration_none_when_not_connected(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
        state="missed",
    )
    await _update_call_session_state(
        conn, call_id="c1", user_id=ALICE,
        state="missed", end_reason="timeout",
    )
    row = await get_call(conn, call_id="c1", user_id=ALICE)
    assert row.to_dict()["duration_seconds"] is None


# ── set_quality_rating ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_quality_rating(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    ok = await set_quality_rating(
        conn, call_id="c1", user_id=ALICE,
        rating=1, notes="crystal clear",
    )
    assert ok is True
    row = await get_call(conn, call_id="c1", user_id=ALICE)
    assert row.quality_rating == 1
    assert row.quality_notes == "crystal clear"


@pytest.mark.asyncio
async def test_set_quality_rating_validates_value(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    with pytest.raises(ValueError):
        await set_quality_rating(
            conn, call_id="c1", user_id=ALICE, rating=5,
        )


@pytest.mark.asyncio
async def test_set_quality_rating_404_returns_false(conn) -> None:
    ok = await set_quality_rating(
        conn, call_id="no-such-id", user_id=ALICE, rating=1,
    )
    assert ok is False


# ── list_events_for_call ────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events_chronological(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    for ev in ("invited", "accepted", "connected", "ended"):
        await _log_call_event(
            conn, call_id="c1", user_id=ALICE,
            event_type=ev, event_data={"step": ev},
        )
        await asyncio.sleep(0.01)
    events = await list_events_for_call(
        conn, call_id="c1", user_id=ALICE,
    )
    assert [e.event_type for e in events] == [
        "invited", "accepted", "connected", "ended",
    ]
    # event_data round-trips as JSON
    assert events[0].event_data == {"step": "invited"}


@pytest.mark.asyncio
async def test_list_events_user_scoped(conn) -> None:
    await _seed_call(
        conn, call_id="c1", user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    await _seed_call(
        conn, call_id="c1", user_id=BOB,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
    )
    await _log_call_event(
        conn, call_id="c1", user_id=ALICE,
        event_type="alice_event", event_data={},
    )
    await _log_call_event(
        conn, call_id="c1", user_id=BOB,
        event_type="bob_event", event_data={},
    )
    alice_events = await list_events_for_call(
        conn, call_id="c1", user_id=ALICE,
    )
    bob_events = await list_events_for_call(
        conn, call_id="c1", user_id=BOB,
    )
    assert [e.event_type for e in alice_events] == ["alice_event"]
    assert [e.event_type for e in bob_events] == ["bob_event"]
