"""Tests for narrative_routes.py — presets, regex, groups, memory settings, lorebook."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_preset_store(monkeypatch):
    """Return a mock PromptPresetStore class and its instance."""
    mock_store = MagicMock()
    mock_preset = MagicMock(
        id="preset_1", name="Test Preset",
        system_prompt="sys", jailbreak="jb",
        post_history="ph", author_note="an",
        author_note_depth=4, is_default=False,
    )
    mock_store.return_value.list_presets = AsyncMock(return_value=[mock_preset])
    mock_store.return_value.save_preset = AsyncMock(return_value=mock_preset)
    mock_store.return_value.delete_preset = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "augmentum.modes.narrative.prompt_presets.PromptPresetStore",
        mock_store,
    )
    return mock_store


def _patch_regex_store(monkeypatch):
    mock_store = MagicMock()
    mock_script = MagicMock(
        id="rx_1", name="Test Regex",
        find_regex="foo", replace_string="bar",
        placement="output", enabled=True,
        order_num=100, character_name=None,
    )
    mock_store.return_value.list_scripts = AsyncMock(return_value=[mock_script])
    mock_store.return_value.save_script = AsyncMock(return_value=mock_script)
    mock_store.return_value.delete_script = AsyncMock(return_value=True)
    mock_store.return_value.toggle_script = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "augmentum.modes.narrative.regex_transformer.RegexScriptStore",
        mock_store,
    )
    return mock_store


def _patch_group_store(monkeypatch):
    mock_store = MagicMock()
    mock_group = MagicMock(
        id="grp_1", name="Test Group",
        description="desc", member_names=["Alice", "Bob"],
        generation_mode="round_robin",
        member_summaries={}, avatar="",
    )
    mock_store.return_value.list_groups = AsyncMock(return_value=[mock_group])
    mock_store.return_value.save_group = AsyncMock(return_value=mock_group)
    mock_store.return_value.delete_group = AsyncMock(return_value=True)
    mock_store.return_value.get_group = AsyncMock(return_value=mock_group)
    monkeypatch.setattr(
        "augmentum.modes.narrative.group_manager.GroupStore",
        mock_store,
    )
    monkeypatch.setattr(
        "augmentum.modes.narrative.group_manager.CharacterGroup",
        MagicMock,
    )
    return mock_store, mock_group


# ---------------------------------------------------------------------------
# Preset Tests
# ---------------------------------------------------------------------------


class TestPresets:
    def test_list_presets_empty_no_db(self, client):
        resp = client.get("/api/narrative/presets")
        assert resp.status_code == 200
        assert resp.json()["presets"] == []

    def test_save_preset_no_db(self, client):
        resp = client.post("/api/narrative/presets", json={"name": "New"})
        assert resp.status_code == 503

    def test_delete_preset_no_db(self, client):
        resp = client.delete("/api/narrative/presets/preset_1")
        assert resp.status_code == 503

    def test_delete_builtin_preset_locked(self, sqlite_client):
        resp = sqlite_client.delete("/api/narrative/presets/builtin_modular")
        assert resp.status_code == 400
        assert "cannot be deleted" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# Regex Tests
# ---------------------------------------------------------------------------


class TestRegex:
    def test_list_regex_no_db(self, client):
        resp = client.get("/api/narrative/regex")
        assert resp.status_code == 200
        assert resp.json()["scripts"] == []

    def test_save_regex_invalid_pattern(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/narrative/regex",
            json={"find_regex": "[invalid", "replace_string": "x"},
        )
        assert resp.status_code == 400
        assert "Invalid regex" in resp.json()["error"]

    def test_test_regex_success(self, client):
        resp = client.post(
            "/api/narrative/regex/test",
            json={
                "find_regex": r"\bfoo\b",
                "replace_string": "bar",
                "text": "foo is foo",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"] == "bar is bar"
        assert data["match_count"] == 2

    def test_test_regex_invalid_pattern(self, client):
        resp = client.post(
            "/api/narrative/regex/test",
            json={"find_regex": "[bad", "replace_string": "", "text": "test"},
        )
        assert resp.status_code == 400
        assert "Invalid regex" in resp.json()["error"]

    def test_list_regex_presets(self, client, monkeypatch):
        monkeypatch.setattr(
            "augmentum.modes.narrative.regex_presets.PRESET_PACKS",
            {
                "basic": {
                    "name": "Basic",
                    "description": "Basic pack",
                    "tier": 1,
                    "count": 3,
                    "scripts": [],
                }
            },
        )
        resp = client.get("/api/narrative/regex/presets")
        assert resp.status_code == 200
        packs = resp.json()["packs"]
        assert len(packs) == 1
        assert packs[0]["id"] == "basic"


# ---------------------------------------------------------------------------
# Group Tests
# ---------------------------------------------------------------------------


class TestGroups:
    def test_list_groups_no_db(self, client):
        resp = client.get("/api/narrative/groups")
        assert resp.status_code == 200
        assert resp.json()["groups"] == []

    def test_save_group_too_few_members(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/narrative/groups",
            json={"name": "Solo", "member_names": ["Alice"]},
        )
        assert resp.status_code == 400
        assert "at least 2" in resp.json()["error"]

    def test_delete_group_no_db(self, client):
        resp = client.delete("/api/narrative/groups/grp_1")
        assert resp.status_code == 503

    def test_deactivate_group_no_engine(self, sqlite_client):
        """Deactivation succeeds even with no active engine."""
        resp = sqlite_client.request(
            "DELETE",
            "/api/narrative/groups/deactivate",
            json={"session_id": "sess_1"},
        )
        assert resp.status_code == 200

    def test_get_turn_state_no_handler(self, client):
        resp = client.get("/api/narrative/groups/grp_1/turn-state?session_id=sess_1")
        assert resp.status_code == 200
        assert resp.json()["turn_state"] is None


# ---------------------------------------------------------------------------
# Session Memory Settings
# ---------------------------------------------------------------------------


class TestSessionMemorySettings:
    def test_get_memory_settings_no_engine(self, client):
        resp = client.get("/api/narrative/session/sess_1/memory-settings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_1"
        assert "effective" in data

    def test_update_memory_settings_bad_type(self, client):
        resp = client.put(
            "/api/narrative/session/sess_1/memory-settings",
            json={"memory_enabled": "not_a_bool"},
        )
        assert resp.status_code == 400

    def test_update_memory_settings_unknown_key(self, client):
        resp = client.put(
            "/api/narrative/session/sess_1/memory-settings",
            json={"bogus_key": True},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------


class TestBackgrounds:
    def test_get_background_empty(self, client):
        resp = client.get("/api/narrative/background/sess_1")
        assert resp.status_code == 200
        assert resp.json()["url"] is None

    def test_clear_background(self, client):
        resp = client.delete("/api/narrative/background/sess_1")
        assert resp.status_code == 200
        assert resp.json()["cleared"] is True


# ---------------------------------------------------------------------------
# Global Lorebook
# ---------------------------------------------------------------------------


class TestLorebook:
    def test_list_global_collections_no_db(self, client):
        resp = client.get("/api/narrative/lorebook/global")
        assert resp.status_code == 200
        assert resp.json()["collections"] == []

    def test_create_collection_no_name(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/narrative/lorebook/global",
            json={"name": "", "entries": []},
        )
        assert resp.status_code == 400
        assert "Name is required" in resp.json()["error"]

    def test_get_collection_not_found(self, sqlite_client):
        resp = sqlite_client.get("/api/narrative/lorebook/global/nonexistent")
        assert resp.status_code == 404

    def test_delete_collection_not_found(self, sqlite_client):
        resp = sqlite_client.delete("/api/narrative/lorebook/global/nonexistent")
        assert resp.status_code == 404
