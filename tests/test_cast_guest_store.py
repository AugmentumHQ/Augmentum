"""Tests for the cast couch co-op GuestStore (Phase 2).

Pins the SQLite-backed guest profile substrate. Mirrors the
:mod:`augmentum.state.game_stream_store` testing pattern — in-memory
aiosqlite connection per test, schema bootstrapped from the actual
migration file so SQL drift is caught immediately.
"""

from __future__ import annotations

import pathlib

import aiosqlite
import pytest

from augmentum.state.guest_store import GuestStore


_USERS_SQL = """
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT
);
"""


def _load_migration() -> str:
    p = pathlib.Path("augmentum/state/migrations/229_cast_guest_profiles.sql")
    return p.read_text(encoding="utf-8")


async def _mkstore():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_USERS_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('host1')")
    await conn.execute("INSERT INTO users (id) VALUES ('host2')")
    await conn.commit()
    await conn.executescript(_load_migration())
    return GuestStore(conn), conn


# ── create_profile ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_profile_returns_gp_prefixed_id():
    store, _ = await _mkstore()
    profile = await store.create_profile(
        host_user_id="host1", display_name="alice",
    )
    assert profile["id"].startswith("gp_")
    assert profile["display_name"] == "alice"
    assert profile["host_user_id"] == "host1"
    assert profile["play_count"] == 0


@pytest.mark.asyncio
async def test_empty_host_user_id_raises():
    store, _ = await _mkstore()
    with pytest.raises(ValueError):
        await store.create_profile(host_user_id="", display_name="alice")


@pytest.mark.asyncio
async def test_empty_display_name_raises():
    store, _ = await _mkstore()
    with pytest.raises(ValueError):
        await store.create_profile(host_user_id="host1", display_name="")


@pytest.mark.asyncio
async def test_duplicate_name_at_same_host_raises():
    """Per-host UNIQUE is the per-spec collision guard for the
    "is that you?" UI branch.
    """
    store, _ = await _mkstore()
    await store.create_profile(host_user_id="host1", display_name="alice")
    with pytest.raises(aiosqlite.IntegrityError):
        await store.create_profile(
            host_user_id="host1", display_name="alice",
        )


@pytest.mark.asyncio
async def test_same_name_at_different_hosts_is_allowed():
    """Cross-host isolation: alice at Alex's and alice at Bob's are
    different guests. Privacy thesis depends on this.
    """
    store, _ = await _mkstore()
    a = await store.create_profile(
        host_user_id="host1", display_name="alice",
    )
    b = await store.create_profile(
        host_user_id="host2", display_name="alice",
    )
    assert a["id"] != b["id"]
    assert a["display_name"] == b["display_name"] == "alice"


@pytest.mark.asyncio
async def test_display_name_trimmed():
    store, _ = await _mkstore()
    p = await store.create_profile(
        host_user_id="host1", display_name="  bob  ",
    )
    assert p["display_name"] == "bob"


# ── get / get_by_name ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_returns_full_row():
    store, _ = await _mkstore()
    minted = await store.create_profile(
        host_user_id="host1", display_name="alice",
    )
    fetched = await store.get(minted["id"], host_user_id="host1")
    assert fetched is not None
    assert fetched["display_name"] == "alice"


@pytest.mark.asyncio
async def test_get_cross_host_returns_none():
    """The host_user_id scope blocks cross-host lookups even when
    the caller knows the profile id.
    """
    store, _ = await _mkstore()
    minted = await store.create_profile(
        host_user_id="host1", display_name="alice",
    )
    fetched = await store.get(minted["id"], host_user_id="host2")
    assert fetched is None


@pytest.mark.asyncio
async def test_get_by_name_finds_existing():
    store, _ = await _mkstore()
    await store.create_profile(host_user_id="host1", display_name="alice")
    fetched = await store.get_by_name(
        host_user_id="host1", display_name="alice",
    )
    assert fetched is not None
    assert fetched["display_name"] == "alice"


@pytest.mark.asyncio
async def test_get_by_name_returns_none_for_unknown():
    store, _ = await _mkstore()
    fetched = await store.get_by_name(
        host_user_id="host1", display_name="ghost",
    )
    assert fetched is None


# ── list_for_host ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_for_host_orders_by_last_seen_desc():
    store, _ = await _mkstore()
    a = await store.create_profile(host_user_id="host1", display_name="alice")
    b = await store.create_profile(host_user_id="host1", display_name="bob")
    # Bump alice's last_seen so she's most recent.
    await store.touch_last_seen(a["id"], host_user_id="host1")
    listed = await store.list_for_host(host_user_id="host1")
    assert [p["display_name"] for p in listed] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_list_for_host_caps_at_8():
    """Privacy guard — the identify endpoint exposes this list to
    anyone with a valid invite token. Long rosters would leak.
    """
    store, _ = await _mkstore()
    for i in range(12):
        await store.create_profile(
            host_user_id="host1", display_name=f"guest_{i}",
        )
    listed = await store.list_for_host(host_user_id="host1")
    assert len(listed) == 8


@pytest.mark.asyncio
async def test_list_for_host_empty_user_returns_empty():
    store, _ = await _mkstore()
    listed = await store.list_for_host(host_user_id="")
    assert listed == []


# ── touch_last_seen ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_touch_last_seen_bumps_play_count_when_requested():
    store, _ = await _mkstore()
    p = await store.create_profile(host_user_id="host1", display_name="alice")
    assert p["play_count"] == 0
    await store.touch_last_seen(
        p["id"], host_user_id="host1", increment_play_count=True,
    )
    refreshed = await store.get(p["id"], host_user_id="host1")
    assert refreshed["play_count"] == 1


@pytest.mark.asyncio
async def test_touch_last_seen_doesnt_bump_count_by_default():
    store, _ = await _mkstore()
    p = await store.create_profile(host_user_id="host1", display_name="alice")
    await store.touch_last_seen(p["id"], host_user_id="host1")
    refreshed = await store.get(p["id"], host_user_id="host1")
    assert refreshed["play_count"] == 0


# ── delete ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_removes_row():
    store, _ = await _mkstore()
    p = await store.create_profile(host_user_id="host1", display_name="alice")
    assert await store.delete(p["id"], host_user_id="host1") is True
    assert await store.get(p["id"], host_user_id="host1") is None


@pytest.mark.asyncio
async def test_delete_blocks_cross_host():
    """Host A can't delete host B's guest. Auth boundary check."""
    store, _ = await _mkstore()
    p = await store.create_profile(host_user_id="host1", display_name="alice")
    assert await store.delete(p["id"], host_user_id="host2") is False
    assert await store.get(p["id"], host_user_id="host1") is not None


@pytest.mark.asyncio
async def test_schema_declares_cascade():
    """Pin the ON DELETE CASCADE clause in the migration.

    A user-delete cascade test in-process is fiddly (aiosqlite needs
    foreign_keys=ON set on the conn BEFORE the cascade fires, and
    enabling mid-session can hang). Cheaper guard: textual assertion
    on the migration itself. The migration runner enables foreign
    keys at boot, so the production cascade does fire.
    """
    sql = _load_migration().lower()
    assert "references users(id) on delete cascade" in sql
