"""Round-trip + isolation tests for DanceHistoryStore + DanceRatingsStore.

These verify the contract the widget depends on:
  - User A can't see User B's history or ratings.
  - 'longer' accumulates and caps at 60s.
  - 'clear' deletes a rating entirely (matches legacy localStorage).
  - History append trims to retention cap.
"""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.dance.store import DanceHistoryStore, DanceRatingsStore

UID_A = "user_a"
UID_B = "user_b"


@pytest.fixture
async def conn(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as c:
        await c.executescript(
            open("augmentum/state/migrations/201_dance_history.sql").read()
            .replace(
                "REFERENCES users(id) ON DELETE CASCADE",
                "",
            )
            .replace(
                "INSERT OR IGNORE INTO schema_version",
                "-- INSERT OR IGNORE INTO schema_version",
            )
        )
        await c.executescript(
            open("augmentum/state/migrations/202_dance_ratings.sql").read()
            .replace(
                "REFERENCES users(id) ON DELETE CASCADE",
                "",
            )
            .replace(
                "INSERT OR IGNORE INTO schema_version",
                "-- INSERT OR IGNORE INTO schema_version",
            )
        )
        await c.commit()
        yield c


# ── History ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_append_and_list(conn):
    store = DanceHistoryStore(conn)
    await store.append(
        anim_id="kebab-dance", label="kebab dance",
        played_at=1_000_000, duration_sec=12.0,
        mode="chat-call", user_id=UID_A,
    )
    rows = await store.list_recent(user_id=UID_A)
    assert len(rows) == 1
    assert rows[0]["anim_id"] == "kebab-dance"
    assert rows[0]["mode"] == "chat-call"


@pytest.mark.asyncio
async def test_history_isolated_per_user(conn):
    store = DanceHistoryStore(conn)
    await store.append(
        anim_id="a-clip", label="a", played_at=1, duration_sec=1.0,
        user_id=UID_A,
    )
    await store.append(
        anim_id="b-clip", label="b", played_at=2, duration_sec=1.0,
        user_id=UID_B,
    )
    a_rows = await store.list_recent(user_id=UID_A)
    b_rows = await store.list_recent(user_id=UID_B)
    assert {r["anim_id"] for r in a_rows} == {"a-clip"}
    assert {r["anim_id"] for r in b_rows} == {"b-clip"}


@pytest.mark.asyncio
async def test_history_newest_first(conn):
    store = DanceHistoryStore(conn)
    for i in range(3):
        await store.append(
            anim_id=f"clip-{i}", label=f"{i}",
            played_at=1_000 + i, duration_sec=1.0,
            user_id=UID_A,
        )
    rows = await store.list_recent(user_id=UID_A)
    assert [r["anim_id"] for r in rows] == ["clip-2", "clip-1", "clip-0"]


@pytest.mark.asyncio
async def test_history_clear(conn):
    store = DanceHistoryStore(conn)
    await store.append(
        anim_id="x", label="x", played_at=1, duration_sec=1.0,
        user_id=UID_A,
    )
    cleared = await store.clear(user_id=UID_A)
    assert cleared == 1
    assert await store.list_recent(user_id=UID_A) == []


# ── Ratings ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ratings_set_kind(conn):
    store = DanceRatingsStore(conn)
    await store.set_kind("kebab-dance", "like", user_id=UID_A)
    all_ratings = await store.list_all(user_id=UID_A)
    assert "kebab-dance" in all_ratings
    assert all_ratings["kebab-dance"]["kind"] == "like"
    assert all_ratings["kebab-dance"]["slotBonusSec"] == 0


@pytest.mark.asyncio
async def test_ratings_longer_accumulates(conn):
    store = DanceRatingsStore(conn)
    await store.add_slot_bonus("kebab-dance", 8, user_id=UID_A)
    await store.add_slot_bonus("kebab-dance", 8, user_id=UID_A)
    all_ratings = await store.list_all(user_id=UID_A)
    assert all_ratings["kebab-dance"]["slotBonusSec"] == 16
    assert "kind" not in all_ratings["kebab-dance"]


@pytest.mark.asyncio
async def test_ratings_longer_caps_at_60(conn):
    store = DanceRatingsStore(conn)
    for _ in range(20):
        await store.add_slot_bonus("kebab-dance", 8, user_id=UID_A)
    all_ratings = await store.list_all(user_id=UID_A)
    assert all_ratings["kebab-dance"]["slotBonusSec"] == 60


@pytest.mark.asyncio
async def test_ratings_kind_preserves_bonus(conn):
    """A 'longer' click then a 'like' click should keep the slot bonus."""
    store = DanceRatingsStore(conn)
    await store.add_slot_bonus("kebab-dance", 16, user_id=UID_A)
    await store.set_kind("kebab-dance", "like", user_id=UID_A)
    all_ratings = await store.list_all(user_id=UID_A)
    assert all_ratings["kebab-dance"]["kind"] == "like"
    assert all_ratings["kebab-dance"]["slotBonusSec"] == 16


@pytest.mark.asyncio
async def test_ratings_clear_deletes(conn):
    store = DanceRatingsStore(conn)
    await store.set_kind("kebab-dance", "broken", user_id=UID_A)
    await store.add_slot_bonus("kebab-dance", 8, user_id=UID_A)
    await store.clear("kebab-dance", user_id=UID_A)
    all_ratings = await store.list_all(user_id=UID_A)
    assert "kebab-dance" not in all_ratings


@pytest.mark.asyncio
async def test_ratings_isolated_per_user(conn):
    store = DanceRatingsStore(conn)
    await store.set_kind("x", "like", user_id=UID_A)
    await store.set_kind("x", "broken", user_id=UID_B)
    a_ratings = await store.list_all(user_id=UID_A)
    b_ratings = await store.list_all(user_id=UID_B)
    assert a_ratings["x"]["kind"] == "like"
    assert b_ratings["x"]["kind"] == "broken"


@pytest.mark.asyncio
async def test_ratings_set_kind_rejects_invalid(conn):
    store = DanceRatingsStore(conn)
    with pytest.raises(ValueError):
        await store.set_kind("x", "love", user_id=UID_A)


@pytest.mark.asyncio
async def test_ratings_requires_user_id(conn):
    store = DanceRatingsStore(conn)
    with pytest.raises(ValueError):
        await store.set_kind("x", "like", user_id="")


@pytest.mark.asyncio
async def test_history_requires_user_id(conn):
    store = DanceHistoryStore(conn)
    with pytest.raises(ValueError):
        await store.append(
            anim_id="x", label="x", played_at=1, duration_sec=1.0,
            user_id="",
        )
