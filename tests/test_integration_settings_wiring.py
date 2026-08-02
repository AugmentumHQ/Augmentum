"""Integration test: full 4-layer config round-trip.

Verifies settings survive: PUT -> GET -> value check, and persist across
simulated restarts (new client against the same SQLite backend).
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from augmentum.config import settings


def _setup_settings_store(app):
    """Wire a SettingsStore on the app's SQLite backend."""
    from augmentum.state.settings_store import SettingsStore

    backend = app.state.state_manager.backend
    asyncio.get_event_loop().run_until_complete(
        backend.conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
    )
    asyncio.get_event_loop().run_until_complete(backend.conn.commit())
    store = SettingsStore(backend.conn)
    app.state.settings_store = store
    return store


# ---------------------------------------------------------------------------
# Tool settings round-trip
# ---------------------------------------------------------------------------


class TestToolSettingsRoundTrip:
    def test_set_and_read_bool(self, sqlite_client):
        """PUT a bool setting, GET it back, verify value."""
        _setup_settings_store(sqlite_client.app)

        # Set
        put_resp = sqlite_client.put("/api/config/tools", json={
            "uarf_auto_search": False,
        })
        assert put_resp.status_code == 200
        assert put_resp.json()["updated"]["uarf_auto_search"] is False

        # Read back
        get_resp = sqlite_client.get("/api/config/tools")
        assert get_resp.status_code == 200
        assert get_resp.json()["uarf_auto_search"] is False

    def test_set_and_read_int(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        sqlite_client.put("/api/config/tools", json={
            "uarf_auto_search_queries": 7,
        })

        get_resp = sqlite_client.get("/api/config/tools")
        assert get_resp.json()["uarf_auto_search_queries"] == 7

    def test_set_and_read_float(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        sqlite_client.put("/api/config/tools", json={
            "search_relevance_min_score": 0.42,
        })

        get_resp = sqlite_client.get("/api/config/tools")
        assert abs(get_resp.json()["search_relevance_min_score"] - 0.42) < 0.001

    def test_set_and_read_string(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        sqlite_client.put("/api/config/tools", json={
            "timezone": "Europe/London",
        })

        get_resp = sqlite_client.get("/api/config/tools")
        assert get_resp.json()["timezone"] == "Europe/London"


# ---------------------------------------------------------------------------
# UI settings round-trip
# ---------------------------------------------------------------------------


class TestUISettingsRoundTrip:
    def test_set_and_read_ui_setting(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        sqlite_client.put("/api/config/ui", json={
            "systemPrompt": "You are a pirate.",
            "temperature": "0.8",
        })

        get_resp = sqlite_client.get("/api/config/ui")
        data = get_resp.json()
        assert data["systemPrompt"] == "You are a pirate."
        assert data["temperature"] == "0.8"

    def test_unknown_ui_key_ignored(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        resp = sqlite_client.put("/api/config/ui", json={
            "systemPrompt": "Valid",
            "totallyFakeKey": "should be dropped",
        })
        assert resp.status_code == 200
        assert "totallyFakeKey" not in resp.json()["updated"]


# ---------------------------------------------------------------------------
# Personalization round-trip
# ---------------------------------------------------------------------------


class TestPersonalizationRoundTrip:
    def test_set_and_read(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        sqlite_client.put("/api/config/personalization", json={
            "aiName": "Luna",
            "responseStyle": "warm",
        })

        get_resp = sqlite_client.get("/api/config/personalization")
        data = get_resp.json()
        assert data["aiName"] == "Luna"
        assert data["responseStyle"] == "warm"


# ---------------------------------------------------------------------------
# Simulated restart persistence
# ---------------------------------------------------------------------------


class TestSettingsPersistAcrossRestart:
    def test_tool_setting_survives_new_client(self, sqlite_client):
        """Settings persisted to SQLite survive creating a new TestClient."""
        store = _setup_settings_store(sqlite_client.app)

        # Set a value
        sqlite_client.put("/api/config/tools", json={
            "uarf_auto_search_queries": 3,
        })

        # Create a "new" client against the same app (simulates restart)
        client2 = TestClient(sqlite_client.app)
        get_resp = client2.get("/api/config/tools")
        assert get_resp.json()["uarf_auto_search_queries"] == 3

    def test_ui_setting_survives_new_client(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        sqlite_client.put("/api/config/ui", json={"aiName": "Persistent"})

        client2 = TestClient(sqlite_client.app)
        get_resp = client2.get("/api/config/ui")
        assert get_resp.json()["aiName"] == "Persistent"


# ---------------------------------------------------------------------------
# Multiple settings in one PUT
# ---------------------------------------------------------------------------


class TestBatchUpdate:
    def test_multiple_settings_in_one_put(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        resp = sqlite_client.put("/api/config/tools", json={
            "uarf_auto_search": True,
            "uarf_auto_search_queries": 4,
            "timezone": "US/Pacific",
        })
        assert resp.status_code == 200
        updated = resp.json()["updated"]
        assert updated["uarf_auto_search"] is True
        assert updated["uarf_auto_search_queries"] == 4
        assert updated["timezone"] == "US/Pacific"

    def test_mixed_valid_and_invalid(self, sqlite_client):
        _setup_settings_store(sqlite_client.app)

        resp = sqlite_client.put("/api/config/tools", json={
            "uarf_auto_search": True,
            "fake_setting": 99,
            "uarf_auto_search_queries": 999,  # out of range
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "uarf_auto_search" in data["updated"]
        assert "errors" in data
        assert len(data["errors"]) == 2  # fake_setting + out of range
