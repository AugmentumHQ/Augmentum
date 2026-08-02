"""Tests for config_routes.py — configuration management API."""

from __future__ import annotations

import asyncio

from augmentum.config import settings

# ---------------------------------------------------------------------------
# GET /api/config/
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_get_config_returns_200(self, client):
        resp = client.get("/api/config/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_get_config_contains_known_keys(self, client):
        resp = client.get("/api/config/")
        data = resp.json()
        # settings.to_safe_dict() should include at least default_backend
        assert "default_backend" in data


# ---------------------------------------------------------------------------
# GET /api/config/section/{section}
# ---------------------------------------------------------------------------


class TestGetConfigSection:
    def test_section_filters_by_prefix(self, client):
        resp = client.get("/api/config/section/uarf")
        assert resp.status_code == 200
        data = resp.json()
        # All returned keys should start with "uarf_"
        for key in data:
            assert key.startswith("uarf_"), f"Unexpected key {key} for section uarf"

    def test_section_unknown_returns_empty(self, client):
        resp = client.get("/api/config/section/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {}


# ---------------------------------------------------------------------------
# GET /api/config/tools
# ---------------------------------------------------------------------------


class TestGetToolSettings:
    def test_get_tool_settings_returns_200(self, client):
        resp = client.get("/api/config/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        # Should contain at least some known tool setting keys
        assert "uarf_auto_search" in data

    def test_sensitive_keys_redacted(self, client):
        # Set a fake API key on settings so the redaction path fires
        original = settings.anthropic_api_key
        object.__setattr__(settings, "anthropic_api_key", "sk-test-secret")
        try:
            resp = client.get("/api/config/tools")
            data = resp.json()
            assert data.get("anthropic_api_key") == "***"
        finally:
            object.__setattr__(settings, "anthropic_api_key", original)


# ---------------------------------------------------------------------------
# PUT /api/config/tools — tool (numeric/bool) settings
# ---------------------------------------------------------------------------


class TestUpdateToolSettings:
    def test_update_bool_setting(self, client):
        resp = client.put("/api/config/tools", json={"uarf_auto_search": True})
        assert resp.status_code == 200
        data = resp.json()
        assert "updated" in data
        assert data["updated"]["uarf_auto_search"] is True

    def test_update_int_setting(self, client):
        resp = client.put("/api/config/tools", json={"uarf_auto_search_queries": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"]["uarf_auto_search_queries"] == 5

    def test_out_of_range_rejected(self, client):
        resp = client.put("/api/config/tools", json={"uarf_auto_search_queries": 999})
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" in data
        assert any("out of range" in e for e in data["errors"])
        assert "uarf_auto_search_queries" not in data["updated"]

    def test_unknown_setting_rejected(self, client):
        resp = client.put("/api/config/tools", json={"totally_fake_key": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" in data
        assert any("Unknown setting" in e for e in data["errors"])

    def test_update_string_setting(self, client):
        resp = client.put("/api/config/tools", json={"timezone": "America/New_York"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"]["timezone"] == "America/New_York"

    def test_string_setting_truncated(self, client):
        # timezone has max length 64
        long_val = "x" * 200
        resp = client.put("/api/config/tools", json={"timezone": long_val})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["updated"]["timezone"]) <= 64

    def test_redacted_placeholder_ignored(self, client):
        """PUT with '***' for a sensitive key should NOT overwrite the real value."""
        original = settings.anthropic_api_key
        object.__setattr__(settings, "anthropic_api_key", "real-key")
        try:
            resp = client.put("/api/config/tools", json={"anthropic_api_key": "***"})
            assert resp.status_code == 200
            # The key should NOT appear in updated (it was skipped)
            data = resp.json()
            assert "anthropic_api_key" not in data["updated"]
            # Runtime value unchanged
            assert settings.anthropic_api_key == "real-key"
        finally:
            object.__setattr__(settings, "anthropic_api_key", original)


# ---------------------------------------------------------------------------
# GET/PUT /api/config/ui — UI settings
# ---------------------------------------------------------------------------


class TestUISettings:
    def test_get_ui_no_store_returns_empty(self, client):
        """Without a settings_store, GET /api/config/ui returns {}."""
        resp = client.get("/api/config/ui")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_put_ui_no_store_returns_503(self, client):
        resp = client.put("/api/config/ui", json={"systemPrompt": "Hello"})
        assert resp.status_code == 503

    def test_ui_settings_round_trip(self, sqlite_client):
        """Set a UI setting then read it back."""
        # We need to wire up a settings_store on the sqlite_client's app
        from augmentum.state.settings_store import SettingsStore

        app = sqlite_client.app
        backend = app.state.state_manager.backend
        # Create the app_settings table
        asyncio.get_event_loop().run_until_complete(
            backend.conn.execute(
                "CREATE TABLE IF NOT EXISTS app_settings "
                "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
            )
        )
        asyncio.get_event_loop().run_until_complete(backend.conn.commit())
        app.state.settings_store = SettingsStore(backend.conn)

        put_resp = sqlite_client.put("/api/config/ui", json={"systemPrompt": "Be helpful"})
        assert put_resp.status_code == 200
        assert put_resp.json()["updated"]["systemPrompt"] == "Be helpful"

        get_resp = sqlite_client.get("/api/config/ui")
        assert get_resp.status_code == 200
        assert get_resp.json()["systemPrompt"] == "Be helpful"

    def test_personal_state_keys_do_not_leak_global(self, sqlite_client):
        """Personal-state ui.* keys must NOT fall back to a global value.

        Regression for the coder+chat multi-window leak: a brand-new account
        with no per-user ``ui.workspace`` was inheriting the install-wide
        ``app_settings`` workspace (an owner's old coder+chat layout) via
        ``get_user_or_global`` and rebuilding both surfaces on first login.
        A genuine *default* key (aiName) still falls back; a personal-*state*
        blob (workspace) does not.
        """
        from augmentum.state.settings_store import SettingsStore

        app = sqlite_client.app
        backend = app.state.state_manager.backend
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            backend.conn.execute(
                "CREATE TABLE IF NOT EXISTS app_settings "
                "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
            )
        )
        loop.run_until_complete(backend.conn.commit())
        store = SettingsStore(backend.conn)
        app.state.settings_store = store

        # Stale install-wide rows (as left behind pre-Stage-D). The caller has
        # NO per-user override for either key.
        leaked_ws = '{"surfaces":[{"type":"coder","primary":true},{"type":"chat"}]}'
        loop.run_until_complete(store.set("ui.workspace", leaked_ws))
        # Identity keys joined the personal-state set 2026-07-18 — a new
        # account must not inherit the owner's assistant name or voice.
        loop.run_until_complete(store.set("ui.aiName", "OwnersCompanionName"))
        loop.run_until_complete(store.set("ui.companionVoice", "kokoro::af_heart"))
        loop.run_until_complete(store.set("ui.personalizeAnalytical", "true"))

        data = sqlite_client.get("/api/config/ui").json()

        # Personal STATE: no global fallback → must be absent.
        assert "workspace" not in data
        assert "aiName" not in data
        assert "companionVoice" not in data
        # Genuine DEFAULT: global fallback still applies.
        assert data.get("personalizeAnalytical") == "true"


# ---------------------------------------------------------------------------
# GET/PUT /api/config/voice-prefs/{mode}
# ---------------------------------------------------------------------------


class TestVoicePrefs:
    def test_invalid_mode_returns_400(self, client):
        resp = client.get("/api/config/voice-prefs/badmode")
        assert resp.status_code == 400

    def test_valid_mode_returns_defaults(self, client):
        """Without a store, returns 503."""
        resp = client.get("/api/config/voice-prefs/passthrough")
        # No settings_store set up on the mock client
        assert resp.status_code == 503

    def test_put_invalid_mode_returns_400(self, client):
        resp = client.put("/api/config/voice-prefs/badmode", json={"avatar_active": True})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/config/passthrough-tools
# ---------------------------------------------------------------------------


class TestPassthroughTools:
    def test_returns_tools_list(self, client):
        resp = client.get("/api/config/passthrough-tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert "defaults" in data
        assert isinstance(data["tools"], list)


# ---------------------------------------------------------------------------
# GET /api/config/personalization
# ---------------------------------------------------------------------------


class TestPersonalization:
    def test_get_personalization_no_store(self, client):
        resp = client.get("/api/config/personalization")
        assert resp.status_code == 200
        assert resp.json() == {}


# ---------------------------------------------------------------------------
# GET /api/config/kokoro-status
# ---------------------------------------------------------------------------


class TestKokoroStatus:
    def test_kokoro_status_returns_200(self, client):
        resp = client.get("/api/config/kokoro-status")
        assert resp.status_code == 200
        data = resp.json()
        # Even if KokoroTTS isn't loaded, should return a dict
        assert isinstance(data, dict)
