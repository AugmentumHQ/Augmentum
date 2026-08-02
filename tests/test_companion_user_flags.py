"""Multi-tenant isolation for companion feature flags.

Covers the 2026-06 fix where the companion intensity dial / menu were
written to and read from the install-wide settings store, leaking one
tenant's companion toggles into every other tenant's panel. The unit
tests here exercise the resolver spine
(``augmentum/companion_runtime/user_flags.py``) directly against a real
in-memory ``SettingsStore``; the route tests assert the endpoints use
the per-user store API.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.companion_runtime import user_flags
from augmentum.state.settings_store import SettingsStore


async def _make_store() -> SettingsStore:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT, "
        "updated_at TEXT)"
    )
    await conn.execute(
        "CREATE TABLE user_settings (user_id TEXT, key TEXT, value TEXT, "
        "updated_at TEXT, PRIMARY KEY (user_id, key))"
    )
    await conn.commit()
    return SettingsStore(conn)


class TestResolveOrder:
    @pytest.mark.asyncio
    async def test_user_override_beats_global(self):
        store = await _make_store()
        await store.set("companion_tick_enabled", "0")          # install-wide
        await store.set_user("u_a", "companion_tick_enabled", "1")  # u_a override
        assert await user_flags.resolve_bool(store, "u_a", "companion_tick_enabled") is True
        # u_b has no override → falls back to the install-wide value.
        assert await user_flags.resolve_bool(store, "u_b", "companion_tick_enabled") is False

    @pytest.mark.asyncio
    async def test_falls_back_to_settings_default(self):
        # Neither user nor global set → settings singleton default applies.
        store = await _make_store()
        got = await user_flags.resolve_bool(
            store, "u_a", "companion_journal_enabled", default=True,
        )
        assert got is True

    @pytest.mark.asyncio
    async def test_resolve_str(self):
        store = await _make_store()
        await store.set_user("u_a", "companion_presence_mode", "engaged")
        assert await user_flags.resolve_str(
            store, "u_a", "companion_presence_mode", "silent",
        ) == "engaged"
        assert await user_flags.resolve_str(
            store, "u_b", "companion_presence_mode", "silent",
        ) == "silent"


class TestWriteIsolation:
    @pytest.mark.asyncio
    async def test_non_owner_write_does_not_touch_global(self):
        """A non-owner's dial writes only their row — the install-wide value
        (which drives the single owner's background loop) is untouched."""
        store = await _make_store()
        await store.set("companion_dreams_enabled", "0")  # owner's global baseline
        await user_flags.write_user_flag(
            store, user_id="u_guest", owner_user_id="u_owner",
            key="companion_dreams_enabled", value=True,
        )
        # u_guest sees their own True; the global (owner) value stays False.
        assert await store.get_user("u_guest", "companion_dreams_enabled") == "1"
        assert await store.get("companion_dreams_enabled") == "0"

    @pytest.mark.asyncio
    async def test_owner_write_mirrors_global(self):
        """The runtime owner's dial mirrors to the install-wide store so the
        single background loop honors it — byte-identical to the old write."""
        store = await _make_store()
        await user_flags.write_user_flag(
            store, user_id="u_owner", owner_user_id="u_owner",
            key="companion_dreams_enabled", value=True,
        )
        assert await store.get_user("u_owner", "companion_dreams_enabled") == "1"
        assert await store.get("companion_dreams_enabled") == "1"

    @pytest.mark.asyncio
    async def test_anonymous_write_uses_global(self):
        store = await _make_store()
        await user_flags.write_user_flag(
            store, user_id="", owner_user_id="",
            key="companion_intensity", value="minimal",
        )
        assert await store.get("companion_intensity") == "minimal"


class TestCompanionIntensityRoute:
    def test_intensity_writes_per_user(self, app, client):
        """POST /api/companion/intensity lands on the caller's row, not the
        install-wide one, when the caller is not the runtime owner.

        `client` is authenticated as usr_test; the runtime owner is someone
        else, so the global mirror must be skipped."""
        store = MagicMock()
        store.set_user = AsyncMock()
        store.set = AsyncMock()
        app.state.settings_store = store
        app.state.companion_runtime = MagicMock(owner_user_id="usr_other")

        resp = client.post("/api/companion/intensity", json={"level": "minimal"})
        assert resp.status_code == 200
        # Every flag written for usr_test, never the install-wide row.
        assert store.set_user.await_count >= 1
        for call in store.set_user.await_args_list:
            assert call.args[0] == "usr_test"
        store.set.assert_not_called()

    def test_intensity_owner_mirrors_global(self, app, client):
        store = MagicMock()
        store.set_user = AsyncMock()
        store.set = AsyncMock()
        app.state.settings_store = store
        app.state.companion_runtime = MagicMock(owner_user_id="usr_test")

        resp = client.post("/api/companion/intensity", json={"level": "minimal"})
        assert resp.status_code == 200
        # Owner write mirrors to the install-wide store too.
        assert store.set.await_count >= 1


class TestCompanionStatusRoute:
    def test_status_reads_per_user(self, app, client):
        """GET /api/companion/status resolves the menu via the per-user
        store API so each tenant sees their own toggles."""
        store = MagicMock()
        store.get_user_or_global = AsyncMock(return_value=None)
        store.get = AsyncMock(return_value=None)
        app.state.settings_store = store

        resp = client.get("/api/companion/status")
        assert resp.status_code == 200
        assert store.get_user_or_global.await_count >= 1
        # Always scoped to the authenticated caller.
        for call in store.get_user_or_global.await_args_list:
            assert call.args[0] == "usr_test"


class TestDiscoveryFeedsRoute:
    def test_feeds_saved_per_user(self, app, client):
        store = MagicMock()
        store.set_user = AsyncMock()
        store.set = AsyncMock()
        app.state.settings_store = store

        resp = client.post("/api/discovery/feeds", json={"hn": True})
        assert resp.status_code == 200
        # Per-user write present, scoped to the caller.
        assert store.set_user.await_count >= 1
        for call in store.set_user.await_args_list:
            assert call.args[0] == "usr_test"
        # Global mirror still written (curator source) — but never instead of
        # the per-user row.
        assert store.set.await_count >= 1
