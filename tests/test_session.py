"""Tests for session management."""

from __future__ import annotations

import pytest

from augmentum.state.backends.memory import MemoryBackend
from augmentum.state.manager import StateManager


@pytest.fixture
def state_manager():
    return StateManager(MemoryBackend())


@pytest.mark.asyncio
async def test_create_session(state_manager):
    """Sessions can be created."""
    session = await state_manager.get_or_create_session("test-1")
    assert session["id"] == "test-1"
    assert session["mode"] == "passthrough"
    assert session["message_count"] == 0


@pytest.mark.asyncio
async def test_get_existing_session(state_manager):
    """Existing sessions are returned, not re-created."""
    s1 = await state_manager.get_or_create_session("test-1")
    s2 = await state_manager.get_or_create_session("test-1")
    assert s1["id"] == s2["id"]
    assert s1["created_at"] == s2["created_at"]


@pytest.mark.asyncio
async def test_update_session_mode(state_manager):
    """Session mode can be updated."""
    await state_manager.get_or_create_session("test-1")
    await state_manager.update_session("test-1", mode="narrative")
    session = await state_manager.get_or_create_session("test-1")
    assert session["mode"] == "narrative"


@pytest.mark.asyncio
async def test_increment_message_count(state_manager):
    """Message count increments correctly."""
    await state_manager.get_or_create_session("test-1")
    await state_manager.update_session("test-1", increment_messages=True)
    await state_manager.update_session("test-1", increment_messages=True)
    session = await state_manager.get_or_create_session("test-1")
    assert session["message_count"] == 2


@pytest.mark.asyncio
async def test_session_with_custom_mode(state_manager):
    """Sessions can be created with a custom mode."""
    session = await state_manager.get_or_create_session("test-1", mode="analytical")
    assert session["mode"] == "analytical"


def test_session_id_from_header(client):
    """X-Augmentum-Session header is respected."""
    resp = client.post(
        "/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
        headers={"X-Augmentum-Session": "custom-session-123"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_sqlite_backend_session():
    """SQLite backend creates and retrieves sessions."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()

    try:
        # Create session
        session = await backend.create_session("sqlite-test-1", mode="passthrough")
        assert session["id"] == "sqlite-test-1"
        assert session["mode"] == "passthrough"

        # Retrieve session
        session = await backend.get_session("sqlite-test-1")
        assert session is not None
        assert session["id"] == "sqlite-test-1"

        # Update session
        await backend.update_session(
            "sqlite-test-1", mode="analytical", increment_messages=True
        )
        session = await backend.get_session("sqlite-test-1")
        assert session["mode"] == "analytical"
        assert session["message_count"] == 1

        # Non-existent session
        assert await backend.get_session("nonexistent") is None
    finally:
        await backend.close()
