"""Round-trip + isolation + active-uniqueness tests for DanceLoopsStore."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.dance.loops_store import DanceLoopsStore

UID_A = "user_a"
UID_B = "user_b"


@pytest.fixture
async def conn(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as c:
        await c.executescript(
            open("augmentum/state/migrations/204_dance_loops.sql").read()
            .replace("REFERENCES users(id) ON DELETE CASCADE", "")
            .replace(
                "INSERT OR IGNORE INTO schema_version",
                "-- INSERT OR IGNORE INTO schema_version",
            )
        )
        await c.commit()
        yield c


@pytest.fixture
async def store(conn):
    return DanceLoopsStore(conn)


@pytest.mark.asyncio
async def test_create_minimal(store):
    row = await store.create(
        name="chill vibes",
        animation_ids=["kebab-dance", "user:abc_123"],
        user_id=UID_A,
    )
    assert row["id"].startswith("loop_")
    assert row["name"] == "chill vibes"
    assert row["animation_ids"] == ["kebab-dance", "user:abc_123"]
    assert row["is_active"] is False


@pytest.mark.asyncio
async def test_rejects_empty_name(store):
    with pytest.raises(ValueError):
        await store.create(name="", user_id=UID_A)


@pytest.mark.asyncio
async def test_isolated_per_user(store):
    await store.create(name="a-loop", user_id=UID_A)
    await store.create(name="b-loop", user_id=UID_B)
    a = await store.list_for_user(user_id=UID_A)
    b = await store.list_for_user(user_id=UID_B)
    assert [r["name"] for r in a] == ["a-loop"]
    assert [r["name"] for r in b] == ["b-loop"]


@pytest.mark.asyncio
async def test_set_active_marks_one(store):
    row = await store.create(name="x", user_id=UID_A)
    activated = await store.set_active(row["id"], user_id=UID_A)
    assert activated is not None
    assert activated["is_active"] is True
    active = await store.get_active(user_id=UID_A)
    assert active["id"] == row["id"]


@pytest.mark.asyncio
async def test_set_active_deactivates_previous(store):
    a = await store.create(name="a", user_id=UID_A)
    b = await store.create(name="b", user_id=UID_A)
    await store.set_active(a["id"], user_id=UID_A)
    await store.set_active(b["id"], user_id=UID_A)
    all_loops = await store.list_for_user(user_id=UID_A)
    active = [l for l in all_loops if l["is_active"]]
    assert len(active) == 1
    assert active[0]["id"] == b["id"]


@pytest.mark.asyncio
async def test_set_active_none_clears(store):
    row = await store.create(name="x", user_id=UID_A)
    await store.set_active(row["id"], user_id=UID_A)
    assert await store.set_active(None, user_id=UID_A) is None
    assert await store.get_active(user_id=UID_A) is None


@pytest.mark.asyncio
async def test_two_users_can_each_have_active(store):
    """Partial unique index is per-user — both users having one active
    loop is fine, but neither can have two."""
    a = await store.create(name="a", user_id=UID_A)
    b = await store.create(name="b", user_id=UID_B)
    await store.set_active(a["id"], user_id=UID_A)
    await store.set_active(b["id"], user_id=UID_B)
    assert (await store.get_active(user_id=UID_A))["id"] == a["id"]
    assert (await store.get_active(user_id=UID_B))["id"] == b["id"]


@pytest.mark.asyncio
async def test_update_name_and_animation_ids(store):
    row = await store.create(name="old", user_id=UID_A)
    updated = await store.update(
        row["id"], {"name": "new", "animation_ids": ["x", "y"]},
        user_id=UID_A,
    )
    assert updated["name"] == "new"
    assert updated["animation_ids"] == ["x", "y"]


@pytest.mark.asyncio
async def test_update_does_not_clear_active(store):
    """Updating the membership of the active loop must NOT inadvertently
    deactivate it — set_active is the sole path for is_active changes."""
    row = await store.create(name="x", user_id=UID_A)
    await store.set_active(row["id"], user_id=UID_A)
    updated = await store.update(
        row["id"], {"animation_ids": ["a", "b"]}, user_id=UID_A,
    )
    assert updated["is_active"] is True


@pytest.mark.asyncio
async def test_update_rejects_empty_name(store):
    row = await store.create(name="x", user_id=UID_A)
    with pytest.raises(ValueError):
        await store.update(row["id"], {"name": "  "}, user_id=UID_A)


@pytest.mark.asyncio
async def test_update_rejects_non_list_ids(store):
    row = await store.create(name="x", user_id=UID_A)
    with pytest.raises(ValueError):
        await store.update(
            row["id"], {"animation_ids": "not-a-list"}, user_id=UID_A,
        )


@pytest.mark.asyncio
async def test_delete_active_clears_active(store):
    row = await store.create(name="x", user_id=UID_A)
    await store.set_active(row["id"], user_id=UID_A)
    assert await store.delete(row["id"], user_id=UID_A) is True
    assert await store.get_active(user_id=UID_A) is None


@pytest.mark.asyncio
async def test_other_user_cannot_delete(store):
    row = await store.create(name="x", user_id=UID_A)
    assert await store.delete(row["id"], user_id=UID_B) is False
    assert await store.get(row["id"], user_id=UID_A) is not None


@pytest.mark.asyncio
async def test_get_rejects_empty_user_id(store):
    """Empty user_id must raise, not silently return another user's loop.

    Regression guard for the dropped-WHERE cross-tenant read.
    """
    row = await store.create(name="x", user_id=UID_A)
    with pytest.raises(ValueError):
        await store.get(row["id"], user_id="")
