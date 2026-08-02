"""Tests for ReadingPositionStore — cross-device reading-position sync."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.state.sync_store import ReadingPositionStore

MIGRATION_PATH = "augmentum/state/migrations/268_reading_positions.sql"

USER_A = "usr_a"
USER_B = "usr_b"


@pytest.fixture
async def store():
    async with aiosqlite.connect(":memory:") as conn:
        await conn.executescript(open(MIGRATION_PATH).read())
        await conn.commit()
        yield ReadingPositionStore(conn)


def _pos(key, *, kind="book", frac=0.5, detail=10, last_read=1000,
         device="dev1", title="A Book"):
    return {
        "key": key,
        "kind": kind,
        "position_fraction": frac,
        "position_detail": detail,
        "last_read_ms": last_read,
        "device_id": device,
        "title": title,
    }


class TestUpsert:
    @pytest.mark.asyncio
    async def test_insert_new_position(self, store):
        accepted, rejected, conflicts = await store.upsert_positions(
            [_pos("book:1")], user_id=USER_A,
        )
        assert accepted == 1
        assert rejected == 0
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_blank_key_is_rejected(self, store):
        accepted, rejected, conflicts = await store.upsert_positions(
            [_pos(""), _pos("   "), {"kind": "book"}], user_id=USER_A,
        )
        assert accepted == 0
        assert rejected == 3

    @pytest.mark.asyncio
    async def test_newer_write_wins(self, store):
        await store.upsert_positions([_pos("book:1", last_read=1000, detail=10)],
                                     user_id=USER_A)
        accepted, _, conflicts = await store.upsert_positions(
            [_pos("book:1", last_read=2000, detail=42)], user_id=USER_A,
        )
        assert accepted == 1
        assert conflicts == []
        rows = await store.list_since(user_id=USER_A, since_ms=0)
        assert len(rows) == 1
        assert rows[0]["position_detail"] == 42
        assert rows[0]["last_read_ms"] == 2000

    @pytest.mark.asyncio
    async def test_stale_write_is_conflict_and_preserves_newer(self, store):
        await store.upsert_positions([_pos("book:1", last_read=2000, detail=42)],
                                     user_id=USER_A)
        accepted, _, conflicts = await store.upsert_positions(
            [_pos("book:1", last_read=1000, detail=10)], user_id=USER_A,
        )
        assert accepted == 0
        assert conflicts == ["book:1"]
        rows = await store.list_since(user_id=USER_A, since_ms=0)
        assert rows[0]["position_detail"] == 42  # newer value preserved

    @pytest.mark.asyncio
    async def test_fraction_is_clamped(self, store):
        await store.upsert_positions(
            [_pos("a", frac=1.7), _pos("b", frac=-0.3)], user_id=USER_A,
        )
        rows = {r["key"]: r for r in await store.list_since(user_id=USER_A, since_ms=0)}
        assert rows["a"]["position_fraction"] == 1.0
        assert rows["b"]["position_fraction"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_title_does_not_clobber_existing(self, store):
        await store.upsert_positions([_pos("book:1", title="Real Title")],
                                     user_id=USER_A)
        await store.upsert_positions(
            [_pos("book:1", last_read=5000, title="")], user_id=USER_A,
        )
        rows = await store.list_since(user_id=USER_A, since_ms=0)
        assert rows[0]["title"] == "Real Title"

    @pytest.mark.asyncio
    async def test_malformed_entries_rejected_not_crash(self, store):
        accepted, rejected, _ = await store.upsert_positions(
            ["not a dict", 42, _pos("ok")], user_id=USER_A,
        )
        assert accepted == 1
        assert rejected == 2

    @pytest.mark.asyncio
    async def test_requires_user_id(self, store):
        with pytest.raises(ValueError):
            await store.upsert_positions([_pos("a")], user_id="")


class TestIsolation:
    @pytest.mark.asyncio
    async def test_users_are_isolated(self, store):
        await store.upsert_positions([_pos("shared:key")], user_id=USER_A)
        await store.upsert_positions([_pos("shared:key")], user_id=USER_B)
        a_rows = await store.list_since(user_id=USER_A, since_ms=0)
        b_rows = await store.list_since(user_id=USER_B, since_ms=0)
        assert len(a_rows) == 1
        assert len(b_rows) == 1

    @pytest.mark.asyncio
    async def test_list_since_empty_user(self, store):
        assert await store.list_since(user_id="", since_ms=0) == []


class TestPullCursor:
    @pytest.mark.asyncio
    async def test_since_ms_filters_older_rows(self, store):
        await store.upsert_positions([_pos("a")], user_id=USER_A)
        # A cursor at "now" (well into the future) should return nothing.
        future = 10**15
        assert await store.list_since(user_id=USER_A, since_ms=future) == []
        # A zero cursor returns everything.
        assert len(await store.list_since(user_id=USER_A, since_ms=0)) == 1

    @pytest.mark.asyncio
    async def test_exclude_device_id(self, store):
        await store.upsert_positions([_pos("a", device="phone")], user_id=USER_A)
        await store.upsert_positions([_pos("b", device="tablet")], user_id=USER_A)
        rows = await store.list_since(
            user_id=USER_A, since_ms=0, exclude_device_id="phone",
        )
        keys = {r["key"] for r in rows}
        assert keys == {"b"}
