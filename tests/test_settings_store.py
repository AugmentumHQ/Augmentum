"""Tests for augmentum/state/settings_store.py — key-value persistence."""

from __future__ import annotations

import aiosqlite
import pytest

from augmentum.state.settings_store import SettingsStore


async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    """Create an in-memory SQLite connection with app_settings + user_settings."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now'))"
        ")"
    )
    await conn.execute(
        "CREATE TABLE user_settings ("
        "  user_id TEXT NOT NULL,"
        "  key TEXT NOT NULL,"
        "  value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')),"
        "  PRIMARY KEY (user_id, key)"
        ")"
    )
    await conn.commit()
    return conn, SettingsStore(conn)


class TestSettingsStore:
    """CRUD operations on the key-value settings store."""

    async def test_set_and_get_string(self):
        conn, store = await _make_store()
        await store.set("theme", "dark")
        assert await store.get("theme") == "dark"
        await conn.close()

    async def test_get_nonexistent_returns_none(self):
        conn, store = await _make_store()
        assert await store.get("missing_key") is None
        await conn.close()

    async def test_set_overwrite(self):
        conn, store = await _make_store()
        await store.set("lang", "en")
        await store.set("lang", "fr")
        assert await store.get("lang") == "fr"
        await conn.close()

    async def test_delete_via_set_none(self):
        conn, store = await _make_store()
        await store.set("temp", "value")
        assert await store.get("temp") == "value"
        await store.set("temp", None)
        assert await store.get("temp") is None
        await conn.close()

    async def test_delete_nonexistent_key(self):
        conn, store = await _make_store()
        # Should not raise
        await store.set("ghost", None)
        assert await store.get("ghost") is None
        await conn.close()

    async def test_set_int_as_string(self):
        conn, store = await _make_store()
        await store.set("count", "42")
        val = await store.get("count")
        assert val == "42"
        assert int(val) == 42
        await conn.close()

    async def test_set_float_as_string(self):
        conn, store = await _make_store()
        await store.set("ratio", "3.14")
        val = await store.get("ratio")
        assert float(val) == pytest.approx(3.14)
        await conn.close()

    async def test_set_bool_as_string(self):
        conn, store = await _make_store()
        await store.set("enabled", "true")
        val = await store.get("enabled")
        assert val == "true"
        assert val.lower() in ("true", "1", "yes")
        await conn.close()

    async def test_get_all_empty(self):
        conn, store = await _make_store()
        assert await store.get_all() == {}
        await conn.close()

    async def test_get_all_multiple(self):
        conn, store = await _make_store()
        await store.set("a", "1")
        await store.set("b", "2")
        await store.set("c", "3")
        all_settings = await store.get_all()
        assert all_settings == {"a": "1", "b": "2", "c": "3"}
        await conn.close()

    async def test_get_all_after_delete(self):
        conn, store = await _make_store()
        await store.set("x", "10")
        await store.set("y", "20")
        await store.set("x", None)
        all_settings = await store.get_all()
        assert all_settings == {"y": "20"}
        await conn.close()

    async def test_empty_string_value(self):
        conn, store = await _make_store()
        await store.set("blank", "")
        val = await store.get("blank")
        assert val == ""
        await conn.close()

    async def test_long_value(self):
        conn, store = await _make_store()
        long_val = "x" * 10000
        await store.set("big", long_val)
        assert await store.get("big") == long_val
        await conn.close()


class TestHasAnyUserValue:
    """``has_any_user_value`` scans user_settings — used to decide
    whether a process-singleton subsystem should stay alive when any
    tenant has opted in."""

    async def test_empty_table(self):
        conn, store = await _make_store()
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is False
        await conn.close()

    async def test_single_user_match(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is True
        await conn.close()

    async def test_value_mismatch(self):
        """Stored value differs from the query value."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "false")
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is False
        await conn.close()

    async def test_key_mismatch(self):
        """Stored key differs from the query key."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.theme", "dark")
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is False
        await conn.close()

    async def test_multi_user_one_match(self):
        """One of many users has the matching value."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "false")
        await store.set_user("bob", "ui.dreamEnabled", "false")
        await store.set_user("carol", "ui.dreamEnabled", "true")
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is True
        await conn.close()

    async def test_match_disappears_on_delete(self):
        """Deleting the only matching row flips the predicate back to False."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is True
        await store.set_user("alice", "ui.dreamEnabled", None)
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is False
        await conn.close()

    async def test_global_row_ignored(self):
        """``has_any_user_value`` scans user_settings only — the global
        ``app_settings`` row is not a user."""
        conn, store = await _make_store()
        await store.set("ui.dreamEnabled", "true")
        assert await store.has_any_user_value("ui.dreamEnabled", "true") is False
        await conn.close()
