"""Tests for dream API routes."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.dream.models import DreamEntry, DreamEntryType, DreamPortrait
from augmentum.proxy.dream_routes import router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)

    # Mock dependencies
    app.state.dream_journal = AsyncMock()
    app.state.dream_portrait_manager = AsyncMock()
    app.state.dream_scheduler = MagicMock()

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


def test_get_journal_entries(client, app):
    mock_entry = DreamEntry(
        id="e1", persona_id="default",
        content="A reflection.", entry_type=DreamEntryType.REFLECTION,
        source_memories=[], source_sessions=[], context_window={},
        embedding=None, dream_cycle_id="c1", created_at="2026-03-25",
    )
    app.state.dream_journal.list_entries = AsyncMock(return_value=([mock_entry], 1))
    resp = client.get("/api/dream/journal")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["entries"]) == 1
    assert data["total"] == 1
    assert data["has_more"] is False
    assert data["entries"][0]["id"] == "e1"
    assert data["entries"][0]["entry_type"] == "reflection"


def test_get_journal_entries_pagination(client, app):
    """has_more is True when there are more results beyond the current page."""
    entries = [
        DreamEntry(
            id=f"e{i}", persona_id="default", content=f"Entry {i}",
            entry_type=DreamEntryType.REFLECTION, source_memories=[],
            source_sessions=[], context_window={}, embedding=None,
            dream_cycle_id="c1", created_at="2026-03-25",
        )
        for i in range(10)
    ]
    # total=20 but only 10 returned — has_more should be True
    app.state.dream_journal.list_entries = AsyncMock(return_value=(entries, 20))
    resp = client.get("/api/dream/journal?limit=10&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_more"] is True


def test_get_journal_entry_by_id(client, app):
    mock_entry = DreamEntry(
        id="e1", persona_id="default", content="Deep thought.",
        entry_type=DreamEntryType.VOICE_NOTE, source_memories=["m1"],
        source_sessions=["s1"], context_window={"key": "val"},
        embedding=None, dream_cycle_id="c1", created_at="2026-03-25",
    )
    app.state.dream_journal.get_entry = AsyncMock(return_value=mock_entry)
    resp = client.get("/api/dream/journal/e1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "e1"
    assert data["context_window"] == {"key": "val"}
    assert data["entry_type"] == "voice_note"


def test_get_journal_entry_not_found(client, app):
    app.state.dream_journal.get_entry = AsyncMock(return_value=None)
    resp = client.get("/api/dream/journal/missing")
    assert resp.status_code == 404


def test_update_journal_entry(client, app):
    mock_entry = DreamEntry(
        id="e1", persona_id="default", content="Old content.",
        entry_type=DreamEntryType.REFLECTION, source_memories=[],
        source_sessions=[], context_window={}, embedding=None,
        dream_cycle_id="c1", created_at="2026-03-25",
    )
    app.state.dream_journal.get_entry = AsyncMock(return_value=mock_entry)
    app.state.dream_journal.update_entry = AsyncMock()
    app.state.dream_portrait_manager.synthesize = AsyncMock(return_value=None)

    resp = client.put("/api/dream/journal/e1", json={"content": "New content.", "pinned": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "updated"
    app.state.dream_journal.update_entry.assert_called_once_with(
        entry_id="e1", content="New content.", weight=None, pinned=True, user_id="",
    )
    # Portrait regeneration should be triggered because content changed
    app.state.dream_portrait_manager.synthesize.assert_called_once()


def test_update_journal_entry_no_content_no_portrait_regen(client, app):
    """Updating weight/pinned without content change skips portrait regen."""
    mock_entry = DreamEntry(
        id="e1", persona_id="default", content="Content.",
        entry_type=DreamEntryType.REFLECTION, source_memories=[],
        source_sessions=[], context_window={}, embedding=None,
        dream_cycle_id="c1", created_at="2026-03-25",
    )
    app.state.dream_journal.get_entry = AsyncMock(return_value=mock_entry)
    app.state.dream_journal.update_entry = AsyncMock()
    app.state.dream_portrait_manager.synthesize = AsyncMock()

    resp = client.put("/api/dream/journal/e1", json={"weight": 1.5})
    assert resp.status_code == 200
    app.state.dream_portrait_manager.synthesize.assert_not_called()


def test_update_journal_entry_not_found(client, app):
    app.state.dream_journal.get_entry = AsyncMock(return_value=None)
    resp = client.put("/api/dream/journal/missing", json={"pinned": True})
    assert resp.status_code == 404


def test_delete_entry(client, app):
    mock_entry = DreamEntry(
        id="entry_123", persona_id="default", content="To be deleted.",
        entry_type=DreamEntryType.REFLECTION, source_memories=[],
        source_sessions=[], context_window={}, embedding=None,
        dream_cycle_id="c1", created_at="2026-03-25",
    )
    app.state.dream_journal.get_entry = AsyncMock(return_value=mock_entry)
    app.state.dream_journal.delete_entry = AsyncMock()
    app.state.dream_portrait_manager.synthesize = AsyncMock(return_value=None)

    resp = client.delete("/api/dream/journal/entry_123")
    assert resp.status_code == 200
    app.state.dream_journal.delete_entry.assert_called_once_with("entry_123", user_id="")


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def test_trigger_dream(client, app):
    app.state.dream_scheduler.trigger_manual = AsyncMock(return_value="cycle_123")
    resp = client.post("/api/dream/trigger", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["cycle_id"] == "cycle_123"
    assert data["status"] == "queued"


def test_trigger_dream_custom_persona(client, app):
    app.state.dream_scheduler.trigger_manual = AsyncMock(return_value="cycle_456")
    resp = client.post("/api/dream/trigger", json={"persona_id": "narrator"})
    assert resp.status_code == 200
    app.state.dream_scheduler.trigger_manual.assert_called_once_with(persona_id="narrator", user_id="")


# ---------------------------------------------------------------------------
# Portrait
# ---------------------------------------------------------------------------


def test_get_portrait(client, app):
    mock_portrait = DreamPortrait(
        id="p1", persona_id="default",
        voice_notes="Direct.", active_threads="Curious.", impressions="Warm.",
        source_entries=[], is_current=True, created_at="2026-03-25",
    )
    app.state.dream_portrait_manager.get_current = AsyncMock(return_value=mock_portrait)
    resp = client.get("/api/dream/portrait")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_notes"] == "Direct."
    assert data["active_threads"] == "Curious."
    assert data["id"] == "p1"


def test_get_portrait_none(client, app):
    app.state.dream_portrait_manager.get_current = AsyncMock(return_value=None)
    resp = client.get("/api/dream/portrait")
    assert resp.status_code == 200
    assert resp.json() is None


def test_regenerate_portrait(client, app):
    mock_portrait = DreamPortrait(
        id="p2", persona_id="default",
        voice_notes="Updated.", active_threads="Evolving.", impressions="Present.",
        source_entries=["e1"], is_current=True, created_at="2026-03-25",
    )
    app.state.dream_portrait_manager.synthesize = AsyncMock(return_value=mock_portrait)
    resp = client.post("/api/dream/portrait/regenerate", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_notes"] == "Updated."


def test_regenerate_portrait_no_entries(client, app):
    app.state.dream_portrait_manager.synthesize = AsyncMock(return_value=None)
    resp = client.post("/api/dream/portrait/regenerate", json={})
    assert resp.status_code == 404


def test_save_checkpoint(client, app):
    app.state.dream_portrait_manager.save_checkpoint = AsyncMock(return_value="cp_123")
    resp = client.post("/api/dream/portrait/checkpoint", json={"name": "Week 1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["checkpoint_id"] == "cp_123"
    assert data["name"] == "Week 1"


def test_save_checkpoint_no_current_portrait(client, app):
    app.state.dream_portrait_manager.save_checkpoint = AsyncMock(return_value=None)
    resp = client.post("/api/dream/portrait/checkpoint", json={"name": "Empty"})
    assert resp.status_code == 404


def test_list_checkpoints(client, app):
    cp = DreamPortrait(
        id="cp1", persona_id="default",
        voice_notes="Past.", active_threads="Settled.", impressions="Calm.",
        source_entries=[], is_current=False, checkpoint_name="Week 1",
        created_at="2026-03-20",
    )
    app.state.dream_portrait_manager.list_checkpoints = AsyncMock(return_value=[cp])
    resp = client.get("/api/dream/portrait/checkpoints")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["checkpoints"]) == 1
    assert data["checkpoints"][0]["checkpoint_name"] == "Week 1"


def test_restore_checkpoint(client, app):
    restored = DreamPortrait(
        id="cp1", persona_id="default",
        voice_notes="Restored.", active_threads=".", impressions=".",
        source_entries=[], is_current=True, checkpoint_name="Week 1",
        created_at="2026-03-20",
    )
    app.state.dream_portrait_manager.restore_checkpoint = AsyncMock(return_value=restored)
    resp = client.post("/api/dream/portrait/restore/cp1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_notes"] == "Restored."
    assert data["is_current"] is True


def test_restore_checkpoint_not_found(client, app):
    app.state.dream_portrait_manager.restore_checkpoint = AsyncMock(return_value=None)
    resp = client.post("/api/dream/portrait/restore/missing")
    assert resp.status_code == 404


def test_reset_portrait(client, app):
    app.state.dream_portrait_manager.reset_to_foundation = AsyncMock()
    resp = client.post("/api/dream/portrait/reset", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    app.state.dream_portrait_manager.reset_to_foundation.assert_called_once_with("default", user_id="")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_get_status(client, app):
    app.state.dream_scheduler.get_status = AsyncMock(return_value={
        "enabled": True, "messages_since_dream": 42,
        "approved_memories_since_dream": 3,
        "last_dream_at": None, "next_dream_eligible": False, "running": False,
    })
    resp = client.get("/api/dream/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["messages_since_dream"] == 42
    assert data["running"] is False


# ---------------------------------------------------------------------------
# System disabled (503)
# ---------------------------------------------------------------------------


def test_dream_system_disabled_journal(app):
    app.state.dream_journal = None
    client = TestClient(app)
    resp = client.get("/api/dream/journal")
    assert resp.status_code == 503
    assert "Dream system not enabled" in resp.json()["error"]


def test_dream_system_disabled_portrait(app):
    app.state.dream_portrait_manager = None
    client = TestClient(app)
    resp = client.get("/api/dream/portrait")
    assert resp.status_code == 503


def test_dream_system_disabled_scheduler(app):
    app.state.dream_scheduler = None
    client = TestClient(app)
    resp = client.get("/api/dream/status")
    assert resp.status_code == 503


def test_dream_system_disabled_trigger(app):
    app.state.dream_scheduler = None
    client = TestClient(app)
    resp = client.post("/api/dream/trigger", json={})
    assert resp.status_code == 503
