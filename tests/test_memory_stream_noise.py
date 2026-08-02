# tests/test_memory_stream_noise.py
"""Memory timeline noise control.

Background (2026-06-11): the Living Stream UI was dominated by lifecycle
telemetry — every auto-promotion wrote TWO events (a rich "promotion" plus
a bare "tier_change" from update_tier), dream cycles with 0 reflections
got cards, and promotion cards never said which memory moved. These tests
pin the fix:

  - write side: _maybe_promote logs exactly one event; the compactor and
    notification-approve path log none; manual tier changes carry
    detail.source == "manual"
  - read side: _prepare_stream_events drops empty dream cycles, drops
    non-manual tier_change rows (legacy duplicates), drops events whose
    memory was deleted, and enriches survivors with memory_content
"""
from __future__ import annotations

import pytest

from augmentum.memory.models import MemoryTier, MemoryType
from augmentum.memory.store import MemoryStore
from augmentum.proxy.memory_routes import _prepare_stream_events
from augmentum.state.backends.sqlite import SQLiteBackend

_UID = "usr_stream_test"


@pytest.fixture
async def backend():
    b = SQLiteBackend(":memory:")
    await b.connect()
    await b.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (_UID, "stream_tester", "x"),
    )
    await b.conn.commit()
    yield b
    await b.close()


@pytest.fixture
async def store(backend):
    return MemoryStore(backend)


async def _events(conn, user_id=_UID):
    cur = await conn.execute(
        "SELECT event_type, memory_id, detail FROM memory_events WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    )
    import json
    return [
        {"event_type": r[0], "memory_id": r[1], "detail": json.loads(r[2] or "{}")}
        for r in await cur.fetchall()
    ]


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_tier_logs_source(store, backend):
    mid = await store.store("Name is Matt", MemoryType.FACT, user_id=_UID)
    await backend.conn.execute("DELETE FROM memory_events WHERE user_id = ?", (_UID,))
    await backend.conn.commit()

    await store.update_tier(mid, MemoryTier.CORE, user_id=_UID, source="manual")

    evts = await _events(backend.conn)
    assert len(evts) == 1
    assert evts[0]["event_type"] == "tier_change"
    assert evts[0]["detail"]["source"] == "manual"
    assert evts[0]["detail"]["to_tier"] == "core"


@pytest.mark.asyncio
async def test_update_tier_log_change_false_writes_no_event(store, backend):
    mid = await store.store("Name is Matt", MemoryType.FACT, user_id=_UID)
    await backend.conn.execute("DELETE FROM memory_events WHERE user_id = ?", (_UID,))
    await backend.conn.commit()

    await store.update_tier(mid, MemoryTier.ARCHIVE, user_id=_UID, log_change=False)

    assert await _events(backend.conn) == []


@pytest.mark.asyncio
async def test_auto_promotion_logs_exactly_one_event(store, backend):
    """The double-card bug: promotion used to write promotion + tier_change."""
    mid = await store.store("Name is Matt", MemoryType.FACT, user_id=_UID, importance=0.9)
    await backend.conn.execute(
        "UPDATE memories SET tier = 'active', access_count = ? WHERE id = ?",
        (store._PROMOTE_TO_CORE_ACCESS, mid),
    )
    await backend.conn.execute("DELETE FROM memory_events WHERE user_id = ?", (_UID,))
    await backend.conn.commit()

    await store._maybe_promote(mid, user_id=_UID)

    mem = await store.get(mid, user_id=_UID)
    tier = mem.tier if isinstance(mem.tier, str) else mem.tier.value
    assert tier == "core"

    evts = await _events(backend.conn)
    assert [e["event_type"] for e in evts] == ["promotion"]
    assert evts[0]["detail"]["to_tier"] == "core"


@pytest.mark.asyncio
async def test_provisional_promotion_logs_exactly_one_event(store, backend):
    mid = await store.store("Listens to audiobooks", MemoryType.PREFERENCE, user_id=_UID)
    await backend.conn.execute(
        "UPDATE memories SET tier = 'provisional', access_count = 3 WHERE id = ?",
        (mid,),
    )
    await backend.conn.execute("DELETE FROM memory_events WHERE user_id = ?", (_UID,))
    await backend.conn.commit()

    await store._maybe_promote(mid, user_id=_UID)

    evts = await _events(backend.conn)
    assert [e["event_type"] for e in evts] == ["promotion"]


# ---------------------------------------------------------------------------
# Read side: _prepare_stream_events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_drops_empty_dream_cycles(backend):
    events = [
        {"event_type": "dream_cycle", "memory_id": None, "detail": {"entries_count": 0}},
        {"event_type": "dream_cycle", "memory_id": None, "detail": {"entries_count": 12}},
        {"event_type": "dream_cycle", "memory_id": None, "detail": {"entries_count": 0, "portrait_updated": True}},
    ]
    out = await _prepare_stream_events(backend.conn, events, user_id=_UID)
    assert len(out) == 2
    assert out[0]["detail"]["entries_count"] == 12
    assert out[1]["detail"]["portrait_updated"] is True


@pytest.mark.asyncio
async def test_stream_drops_non_manual_tier_changes(backend):
    """Legacy auto-promotions left a bare tier_change next to every
    promotion event — those historical rows must not render."""
    events = [
        {"event_type": "tier_change", "memory_id": None, "detail": {"to_tier": "core"}},
        {"event_type": "tier_change", "memory_id": None, "detail": {"to_tier": "core", "source": "system"}},
        {"event_type": "tier_change", "memory_id": None, "detail": {"to_tier": "core", "source": "manual"}},
    ]
    out = await _prepare_stream_events(backend.conn, events, user_id=_UID)
    assert len(out) == 1
    assert out[0]["detail"]["source"] == "manual"


@pytest.mark.asyncio
async def test_stream_enriches_with_memory_content(store, backend):
    mid = await store.store("Name is Matt", MemoryType.FACT, user_id=_UID)
    events = [
        {"event_type": "promotion", "memory_id": mid, "detail": {"to_tier": "core"}},
    ]
    out = await _prepare_stream_events(backend.conn, events, user_id=_UID)
    assert len(out) == 1
    assert out[0]["memory_content"] == "Name is Matt"


@pytest.mark.asyncio
async def test_stream_drops_events_for_deleted_memories(backend):
    events = [
        {"event_type": "promotion", "memory_id": "mem_gone", "detail": {"to_tier": "core"}},
    ]
    out = await _prepare_stream_events(backend.conn, events, user_id=_UID)
    assert out == []


@pytest.mark.asyncio
async def test_stream_does_not_leak_other_users_content(store, backend):
    """Enrichment query is user-scoped: an event referencing another
    user's memory id must not pull that content."""
    other = "usr_other_stream"
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (other, "other_stream", "x"),
    )
    await backend.conn.commit()
    other_mid = await store.store("secret fact", MemoryType.FACT, user_id=other)

    events = [
        {"event_type": "promotion", "memory_id": other_mid, "detail": {"to_tier": "core"}},
    ]
    out = await _prepare_stream_events(backend.conn, events, user_id=_UID)
    assert out == []
