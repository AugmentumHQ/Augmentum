"""Round-trip + isolation tests for UserAnimationStore."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.animations.store import UserAnimationStore

UID_A = "user_a"
UID_B = "user_b"


@pytest.fixture
async def conn(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as c:
        await c.executescript(
            open("augmentum/state/migrations/203_user_animations.sql").read()
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
    return UserAnimationStore(conn)


@pytest.mark.asyncio
async def test_create_minimal(store):
    row = await store.create(
        animation_type="vrma",
        source_path="/data/user_animations/user_a/foo.vrma",
        label="my dance",
        user_id=UID_A,
    )
    assert row["id"].startswith("user:")
    assert row["type"] == "vrma"
    assert row["label"] == "my dance"
    assert row["roles"] == ["dance"]
    assert row["modes"] == ["chat-call", "narrative"]
    assert row["cost"] == 0.5
    assert row["loop_flag"] == 0


@pytest.mark.asyncio
async def test_create_with_full_tags(store):
    row = await store.create(
        animation_type="bvh",
        source_path="/x/y.bvh",
        label="big spin",
        roles=["celebrate", "show-off"],
        emotion={"warmth": 0.9, "energy": 1.0, "openness": 0.8, "focus": 0.5},
        modes=["chat-call"],
        cost=0.8,
        duration_sec=5.5,
        cooldown_sec=600,
        framing="fullBody",
        speed=1.1,
        loop_flag=True,
        explicit_only=False,
        notes="user-recorded mocap",
        user_id=UID_A,
    )
    assert row["roles"] == ["celebrate", "show-off"]
    assert row["emotion"]["energy"] == 1.0
    assert row["loop_flag"] == 1
    assert row["framing"] == "fullBody"


@pytest.mark.asyncio
async def test_rejects_unknown_type(store):
    with pytest.raises(ValueError):
        await store.create(
            animation_type="fbx",
            source_path="/x.fbx", label="x", user_id=UID_A,
        )


@pytest.mark.asyncio
async def test_rejects_empty_label(store):
    with pytest.raises(ValueError):
        await store.create(
            animation_type="vrma",
            source_path="/x.vrma", label="", user_id=UID_A,
        )


@pytest.mark.asyncio
async def test_isolated_per_user(store):
    await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    await store.create(
        animation_type="vrma", source_path="/b.vrma", label="b",
        user_id=UID_B,
    )
    a_rows = await store.list_for_user(user_id=UID_A)
    b_rows = await store.list_for_user(user_id=UID_B)
    assert len(a_rows) == 1
    assert len(b_rows) == 1
    assert a_rows[0]["label"] == "a"
    assert b_rows[0]["label"] == "b"


@pytest.mark.asyncio
async def test_get_scoped_by_user(store):
    a_row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    # User B can't see User A's animation.
    assert await store.get(a_row["id"], user_id=UID_B) is None
    assert await store.get(a_row["id"], user_id=UID_A) is not None


@pytest.mark.asyncio
async def test_get_rejects_empty_user_id(store):
    """Empty user_id must raise, not silently return another user's row.

    Regression guard: get() historically dropped the WHERE clause when
    user_id was falsy, so an empty scope reaching this read could return
    any row by id (cross-tenant leak).
    """
    a_row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    with pytest.raises(ValueError):
        await store.get(a_row["id"], user_id="")


@pytest.mark.asyncio
async def test_update_partial(store):
    row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    updated = await store.update(
        row["id"], {"label": "renamed", "cost": 0.7}, user_id=UID_A,
    )
    assert updated["label"] == "renamed"
    assert updated["cost"] == 0.7
    # Untouched fields preserved.
    assert updated["type"] == "vrma"
    assert updated["roles"] == ["dance"]


@pytest.mark.asyncio
async def test_update_ignores_unknown_columns(store):
    row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    # 'source_path' / 'id' / 'user_id' must NOT be editable via update.
    updated = await store.update(
        row["id"],
        {"label": "ok", "source_path": "/evil", "id": "user:hax"},
        user_id=UID_A,
    )
    assert updated["label"] == "ok"
    assert updated["source_path"] == "/a.vrma"
    assert updated["id"] == row["id"]


@pytest.mark.asyncio
async def test_update_other_users_animation_noop(store):
    row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    # User B trying to update User A's row should not mutate it.
    result = await store.update(
        row["id"], {"label": "hacked"}, user_id=UID_B,
    )
    # update() returns the row as seen by the caller — None because
    # they can't see it.
    assert result is None
    # And the row is unchanged when User A reads it back.
    unchanged = await store.get(row["id"], user_id=UID_A)
    assert unchanged["label"] == "a"


@pytest.mark.asyncio
async def test_delete_returns_row(store):
    row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    deleted = await store.delete(row["id"], user_id=UID_A)
    assert deleted["source_path"] == "/a.vrma"
    assert await store.get(row["id"], user_id=UID_A) is None


@pytest.mark.asyncio
async def test_delete_other_users_animation_returns_none(store):
    row = await store.create(
        animation_type="vrma", source_path="/a.vrma", label="a",
        user_id=UID_A,
    )
    # User B can't delete User A's animation.
    assert await store.delete(row["id"], user_id=UID_B) is None
    # Row still exists.
    assert await store.get(row["id"], user_id=UID_A) is not None
