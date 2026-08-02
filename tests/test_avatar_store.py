"""Tests for avatar CRUD store."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.avatar.store import AvatarStore


@pytest.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as conn:
        # Create table
        await conn.executescript(open("augmentum/state/migrations/060_avatars.sql").read())
        # Apply additional migrations
        for mig in ("061_avatar_type.sql",):
            try:
                await conn.executescript(open(f"augmentum/state/migrations/{mig}").read())
            except Exception:
                pass  # Column may already exist
        # user_id column landed in migration 093 — add it directly so we
        # don't drag in the users table dependency.
        try:
            await conn.execute("ALTER TABLE avatars ADD COLUMN user_id TEXT")
        except Exception:
            pass
        await conn.commit()
        yield AvatarStore(conn)

UID = "user_test"

@pytest.mark.asyncio
async def test_create_avatar(store):
    avatar = await store.create(
        character_id="ch_test",
        vrm_path="/data/avatars/test.vrm",
        source_image_id="img_123",
        user_id=UID,
    )
    assert avatar["id"].startswith("avt_")
    assert avatar["character_id"] == "ch_test"
    assert avatar["vrm_path"] == "/data/avatars/test.vrm"

@pytest.mark.asyncio
async def test_get_avatar(store):
    created = await store.create(character_id="ch_test", vrm_path="/test.vrm", user_id=UID)
    fetched = await store.get(created["id"], user_id=UID)
    assert fetched is not None
    assert fetched["id"] == created["id"]

@pytest.mark.asyncio
async def test_get_by_character(store):
    await store.create(character_id="ch_abc", vrm_path="/a.vrm", user_id=UID)
    result = await store.get_by_character("ch_abc", user_id=UID)
    assert result is not None
    assert result["character_id"] == "ch_abc"

@pytest.mark.asyncio
async def test_get_by_character_returns_none(store):
    result = await store.get_by_character("ch_nonexistent", user_id=UID)
    assert result is None


@pytest.mark.asyncio
async def test_get_empty_user_id_returns_bundled_only(store):
    """No-scope reads see bundled (user_id IS NULL) rows only — never
    another tenant's private avatar. Regression guard for the dropped
    WHERE clause that returned any avatar by id under an empty scope.
    """
    private = await store.create(
        character_id="ch_priv", vrm_path="/priv.vrm", user_id=UID,
    )
    bundled = await store.create(vrm_path="/bundled.vrm", is_bundled=True)

    # Empty scope: private row invisible, bundled row visible.
    assert await store.get(private["id"], user_id="") is None
    got_bundled = await store.get(bundled["id"], user_id="")
    assert got_bundled is not None
    assert got_bundled["id"] == bundled["id"]

    # A different user still cannot read the private row by id...
    assert await store.get(private["id"], user_id="other_user") is None
    # ...but can see the bundled row, and the owner sees their own.
    assert await store.get(bundled["id"], user_id="other_user") is not None
    assert await store.get(private["id"], user_id=UID) is not None

@pytest.mark.asyncio
async def test_list_avatars(store):
    await store.create(character_id="ch_1", vrm_path="/1.vrm", user_id=UID)
    await store.create(character_id="ch_2", vrm_path="/2.vrm", user_id=UID)
    avatars = await store.list_all(user_id=UID)
    assert len(avatars) == 2

@pytest.mark.asyncio
async def test_delete_avatar(store):
    created = await store.create(character_id="ch_del", vrm_path="/del.vrm", user_id=UID)
    await store.delete(created["id"], user_id=UID)
    assert await store.get(created["id"], user_id=UID) is None

@pytest.mark.asyncio
async def test_update_mannerisms(store):
    created = await store.create(character_id="ch_m", vrm_path="/m.vrm", user_id=UID)
    await store.update_mannerisms(created["id"], {"gesture_frequency": 0.8}, user_id=UID)
    updated = await store.get(created["id"], user_id=UID)
    import json
    mannerisms = json.loads(updated["mannerisms"])
    assert mannerisms["gesture_frequency"] == 0.8

@pytest.mark.asyncio
async def test_assign_to_character(store):
    created = await store.create(vrm_path="/assign.vrm", user_id=UID)
    await store.assign_to_character(created["id"], "ch_target", user_id=UID)
    fetched = await store.get(created["id"], user_id=UID)
    assert fetched["character_id"] == "ch_target"
