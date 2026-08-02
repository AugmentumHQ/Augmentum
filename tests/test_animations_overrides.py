"""Tests for per-user bundled-atlas overrides (migration 256).

Covers the store (upsert semantics, patch whitelist, user isolation)
and the /api/animations/overrides routes the widget's master-list
manager calls when the user disables or re-tags a bundled entry.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from augmentum.animations.store import UserAnimationStore
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager

UID_A = "user_a"
UID_B = "user_b"


@pytest.fixture
async def conn(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as c:
        await c.executescript(
            open(
                "augmentum/state/migrations/"
                "256_user_atlas_overrides_for_bundled_animation_customization.sql"
            ).read().replace("REFERENCES users(id) ON DELETE CASCADE", "")
        )
        await c.commit()
        yield c


@pytest.fixture
async def store(conn):
    return UserAnimationStore(conn)


class TestOverrideStore:
    @pytest.mark.asyncio
    async def test_set_and_list(self, store):
        await store.set_override(
            "peace-sign", disabled=True, user_id=UID_A,
        )
        rows = await store.list_overrides(user_id=UID_A)
        assert len(rows) == 1
        assert rows[0]["atlas_id"] == "peace-sign"
        assert rows[0]["disabled"] is True
        assert rows[0]["patch"] == {}

    @pytest.mark.asyncio
    async def test_disable_then_patch_keeps_disable(self, store):
        """The two halves are independent — a later patch-only write
        must not clobber an earlier disable, and vice versa."""
        await store.set_override("hello", disabled=True, user_id=UID_A)
        await store.set_override(
            "hello", patch={"roles": ["greet", "salute"]}, user_id=UID_A,
        )
        rows = await store.list_overrides(user_id=UID_A)
        assert rows[0]["disabled"] is True
        assert rows[0]["patch"]["roles"] == ["greet", "salute"]
        # And the reverse: toggling disable keeps the patch.
        await store.set_override("hello", disabled=False, user_id=UID_A)
        rows = await store.list_overrides(user_id=UID_A)
        assert rows[0]["disabled"] is False
        assert rows[0]["patch"]["roles"] == ["greet", "salute"]

    @pytest.mark.asyncio
    async def test_patch_whitelist_drops_unknown_keys(self, store):
        """source / id smuggling must be stripped — a patched entry is
        played by the conductor, so arbitrary keys are not acceptable."""
        await store.set_override(
            "hello",
            patch={
                "roles": ["greet"],
                "source": "/etc/passwd",
                "id": "evil",
                "cost": 0.2,
            },
            user_id=UID_A,
        )
        rows = await store.list_overrides(user_id=UID_A)
        assert "source" not in rows[0]["patch"]
        assert "id" not in rows[0]["patch"]
        assert rows[0]["patch"]["roles"] == ["greet"]
        assert rows[0]["patch"]["cost"] == 0.2

    @pytest.mark.asyncio
    async def test_user_isolation(self, store):
        await store.set_override("hello", disabled=True, user_id=UID_A)
        assert await store.list_overrides(user_id=UID_B) == []

    @pytest.mark.asyncio
    async def test_clear_override(self, store):
        await store.set_override("hello", disabled=True, user_id=UID_A)
        assert await store.clear_override("hello", user_id=UID_A) is True
        assert await store.list_overrides(user_id=UID_A) == []
        # Clearing a nonexistent row reports False (route 404s).
        assert await store.clear_override("hello", user_id=UID_A) is False

    @pytest.mark.asyncio
    async def test_requires_user_id(self, store):
        with pytest.raises(ValueError):
            await store.set_override("hello", disabled=True, user_id="")
        assert await store.list_overrides(user_id="") == []


@pytest.fixture
def sqlite_client(app, tmp_path, monkeypatch):
    monkeypatch.setenv("AUGMENTUM_ANIMATIONS_DIR", str(tmp_path / "anims"))
    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc
    asyncio.get_event_loop().run_until_complete(backend.close())


class TestOverrideRoutes:
    def test_overrides_start_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/animations/overrides")
        assert resp.status_code == 200
        assert resp.json() == {"overrides": []}

    def test_put_get_delete_round_trip(self, sqlite_client):
        put = sqlite_client.put(
            "/api/animations/overrides/peace-sign",
            json={"disabled": True, "patch": {"roles": ["agree"]}},
        )
        assert put.status_code == 200
        assert put.json()["override"]["disabled"] is True

        got = sqlite_client.get("/api/animations/overrides")
        rows = got.json()["overrides"]
        assert len(rows) == 1
        assert rows[0]["atlas_id"] == "peace-sign"
        assert rows[0]["patch"]["roles"] == ["agree"]

        deleted = sqlite_client.delete("/api/animations/overrides/peace-sign")
        assert deleted.status_code == 200
        assert sqlite_client.get(
            "/api/animations/overrides"
        ).json()["overrides"] == []

    def test_delete_missing_404s(self, sqlite_client):
        resp = sqlite_client.delete("/api/animations/overrides/nope")
        assert resp.status_code == 404

    def test_put_validates_types(self, sqlite_client):
        assert sqlite_client.put(
            "/api/animations/overrides/hello",
            json={"disabled": "yes"},
        ).status_code == 400
        assert sqlite_client.put(
            "/api/animations/overrides/hello",
            json={"patch": ["not", "a", "dict"]},
        ).status_code == 400

    def test_put_rejects_traversal_id(self, sqlite_client):
        # Dotted relative segment reaches the handler → 400 from the
        # id guard. (A %2F-encoded slash never matches the route at
        # all — Starlette 404s it before the handler, also safe.)
        resp = sqlite_client.put(
            "/api/animations/overrides/..evil",
            json={"disabled": True},
        )
        assert resp.status_code == 400


@pytest.fixture
def roles_client(sqlite_client):
    """sqlite_client + a settings_store (the roles snapshot persists
    per-user via user_settings, not the animations tables)."""
    from augmentum.state.settings_store import SettingsStore

    app = sqlite_client.app
    backend = app.state.state_manager.backend
    asyncio.get_event_loop().run_until_complete(
        backend.conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
    )
    asyncio.get_event_loop().run_until_complete(backend.conn.commit())
    app.state.settings_store = SettingsStore(backend.conn)
    return sqlite_client


class TestRolesSnapshot:
    def test_roles_start_empty(self, roles_client):
        resp = roles_client.get("/api/animations/roles")
        assert resp.status_code == 200
        assert resp.json() == {"roles": []}

    def test_snapshot_round_trip(self, roles_client):
        put = roles_client.put(
            "/api/animations/roles-snapshot",
            json={"roles": ["Greet", "celebrate", "greet", "  comfort  "]},
        )
        assert put.status_code == 200
        assert put.json()["count"] == 3  # lowered + deduped + stripped

        got = roles_client.get("/api/animations/roles")
        assert got.json()["roles"] == ["celebrate", "comfort", "greet"]

    def test_snapshot_replaces_wholesale(self, roles_client):
        roles_client.put(
            "/api/animations/roles-snapshot", json={"roles": ["old-role"]},
        )
        roles_client.put(
            "/api/animations/roles-snapshot", json={"roles": ["new-role"]},
        )
        assert roles_client.get(
            "/api/animations/roles"
        ).json()["roles"] == ["new-role"]

    def test_snapshot_validates_shape(self, roles_client):
        assert roles_client.put(
            "/api/animations/roles-snapshot", json={"roles": "greet"},
        ).status_code == 400
        assert roles_client.put(
            "/api/animations/roles-snapshot", json={},
        ).status_code == 400

    def test_snapshot_bounds_entries(self, roles_client):
        long_role = "x" * 200
        put = roles_client.put(
            "/api/animations/roles-snapshot", json={"roles": [long_role]},
        )
        assert put.status_code == 200
        stored = roles_client.get("/api/animations/roles").json()["roles"]
        assert len(stored[0]) == 48  # truncated to the cap
