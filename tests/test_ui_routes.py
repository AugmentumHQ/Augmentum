"""Tests for ui_routes.py — UI-specific API endpoints."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# GET /api/ui/status
# ---------------------------------------------------------------------------


class TestUIStatus:
    def test_status_returns_200(self, client):
        # The mock provider_registry.available_backends may not be set
        # Provide it as a property returning the backends keys
        registry = client.app.state.provider_registry
        registry.available_backends = list(registry.backends.keys())

        resp = client.get("/api/ui/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "active_sessions" in data
        assert "backends" in data
        assert "tools" in data
        assert "config" in data

    def test_status_config_shape(self, client):
        registry = client.app.state.provider_registry
        registry.available_backends = list(registry.backends.keys())

        resp = client.get("/api/ui/status")
        config = resp.json()["config"]
        assert "uarf_proactive_search" in config
        assert "narrative_auto_persist" in config


# ---------------------------------------------------------------------------
# GET /api/ui/settings
# ---------------------------------------------------------------------------


class TestUISettings:
    def test_settings_returns_200(self, client):
        resp = client.get("/api/ui/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "default_backend" in data

    def test_settings_has_cache_fields(self, client):
        resp = client.get("/api/ui/settings")
        data = resp.json()
        assert "prompt_cache_enabled" in data


# ---------------------------------------------------------------------------
# POST /api/ui/generate-title
# ---------------------------------------------------------------------------


class TestGenerateTitle:
    def test_empty_message_returns_default(self, client):
        resp = client.post("/api/ui/generate-title", json={"message": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "New Chat"

    def test_with_message_returns_title(self, client):
        resp = client.post("/api/ui/generate-title", json={
            "message": "What is the meaning of life?"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "title" in data
        # The mock backend returns "Hello from mock Ollama!" which gets trimmed
        assert isinstance(data["title"], str)
        assert len(data["title"]) > 0

    def test_long_message_fallback(self, client):
        """If backend fails, falls back to truncation."""
        # Make backend.chat raise
        from augmentum.models.base import ModelBackend

        original = client.app.state.provider_registry.default_backend
        broken_backend = MagicMock()
        broken_backend.chat = AsyncMock(side_effect=Exception("Broken"))
        client.app.state.provider_registry.default_backend = broken_backend

        resp = client.post("/api/ui/generate-title", json={
            "message": "A" * 200,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["title"]) <= 53  # 50 chars + "..."

        client.app.state.provider_registry.default_backend = original


# ---------------------------------------------------------------------------
# GET /api/ui/sessions
# ---------------------------------------------------------------------------


class TestUISessionsList:
    def test_sessions_returns_list(self, client):
        resp = client.get("/api/ui/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)


# ---------------------------------------------------------------------------
# Narrative inspector state
# ---------------------------------------------------------------------------


class TestUINarrativeState:
    def test_state_endpoint_hydrates_existing_empty_engine_from_db(
        self, sqlite_client, test_user,
    ):
        from augmentum.modes.narrative.engine import NarrativeEngine
        from augmentum.modes.narrative.memory import MemoryEntry, StateSnapshot

        session_id = "sess_ui_hydrate"

        persisted_engine = NarrativeEngine(session_id=session_id)
        persisted_engine._state.message_count = 12
        persisted_engine._state_snapshot = StateSnapshot(fields={
            "location": "Old harbor warehouse",
            "who_present": "Mara and June",
        })
        persisted_engine._memory_ledger = [
            MemoryEntry(
                round_num=12,
                category="discovery",
                content="Mara found the hidden dock ledger behind a loose brick.",
            ),
        ]
        persisted_engine.sync_to_state()

        asyncio.get_event_loop().run_until_complete(
            sqlite_client.app.state.state_manager.save_narrative_state(
                session_id, persisted_engine.state, user_id=test_user.id,
            )
        )

        empty_engine = NarrativeEngine(session_id=session_id)
        sqlite_client.app.state.narrative_engines[(test_user.id, session_id)] = empty_engine

        resp = sqlite_client.get(f"/api/ui/session/{session_id}/state")
        assert resp.status_code == 200

        data = resp.json()["state"]
        assert data["message_count"] == 12
        assert data["state_snapshot"]["location"] == "Old harbor warehouse"
        assert data["memory_ledger"][0]["content"] == (
            "Mara found the hidden dock ledger behind a loose brick."
        )

        hydrated_engine = sqlite_client.app.state.narrative_engines[(test_user.id, session_id)]
        assert hydrated_engine.state.message_count == 12
        assert hydrated_engine._state_snapshot is not None
        assert hydrated_engine._state_snapshot.fields["who_present"] == "Mara and June"
        assert hydrated_engine._memory_ledger[0].category == "discovery"

    def test_patch_state_updates_engine_and_persists_for_future_hydration(
        self, sqlite_client, test_user,
    ):
        from augmentum.modes.narrative.engine import NarrativeEngine

        session_id = "sess_ui_patch"
        engine = NarrativeEngine(session_id=session_id)
        sqlite_client.app.state.narrative_engines[(test_user.id, session_id)] = engine

        payload = {
            "state_snapshot": {
                "location": "Clocktower attic",
                "current_activity": "Sorting stolen maps",
            },
            "memory_ledger": [
                {
                    "round_num": 7,
                    "category": "discovery",
                    "content": "They identified the smuggler routes hidden in the map folds.",
                },
            ],
        }
        patch_resp = sqlite_client.patch(f"/api/ui/session/{session_id}/state", json=payload)
        assert patch_resp.status_code == 200
        assert patch_resp.json()["ok"] is True

        assert engine._state_snapshot is not None
        assert engine._state_snapshot.fields["location"] == "Clocktower attic"
        assert len(engine._memory_ledger) == 1
        assert engine._memory_ledger[0].content == (
            "They identified the smuggler routes hidden in the map folds."
        )

        sqlite_client.app.state.narrative_engines[(test_user.id, session_id)] = NarrativeEngine(
            session_id=session_id,
        )

        get_resp = sqlite_client.get(f"/api/ui/session/{session_id}/state")
        assert get_resp.status_code == 200

        state = get_resp.json()["state"]
        assert state["state_snapshot"]["current_activity"] == "Sorting stolen maps"
        assert state["memory_ledger"][0]["round_num"] == 7
