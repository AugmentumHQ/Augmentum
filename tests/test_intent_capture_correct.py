"""Tests for the intent-capture correction flywheel + companion memory floor.

Covers two Tier-1 "wire the dormant safety nets" fixes
(project_uncertainty_handling_map):
  * ``update_corrected_goal`` — the writer for the previously-dead
    ``corrected_goal`` column that feeds the export's supervised label.
  * ``CompanionMemory.recall`` now floors relevance via
    ``companion_memory_min_score`` instead of inheriting store.recall's 0.0.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.intent.capture_store import (
    VALID_GOALS,
    record_intent_capture,
    update_corrected_goal,
)


async def _migrated_conn():
    """Fresh in-memory backend with migrations applied; returns its conn."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


async def _one_capture(conn, *, user_id: str, text: str = "play some jazz") -> str:
    await record_intent_capture(
        conn,
        user_id=user_id,
        input_text=text,
        goal="drop",
        effective_goal="drop",
        confidence=0.2,
    )
    row = await (await conn.execute(
        "SELECT id FROM intent_capture WHERE user_id = ? ORDER BY captured_at DESC LIMIT 1",
        (user_id,),
    )).fetchone()
    return row[0]


@pytest.mark.asyncio
async def test_correct_sets_label_and_export_uses_it():
    backend = await _migrated_conn()
    try:
        conn = backend.conn
        cid = await _one_capture(conn, user_id="userA")

        ok = await update_corrected_goal(
            conn, user_id="userA", capture_id=cid, corrected_goal="act",
        )
        assert ok is True

        val = (await (await conn.execute(
            "SELECT corrected_goal FROM intent_capture WHERE id = ?", (cid,),
        )).fetchone())[0]
        assert val == "act"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_correct_rejects_invalid_goal():
    backend = await _migrated_conn()
    try:
        conn = backend.conn
        cid = await _one_capture(conn, user_id="userA")
        assert await update_corrected_goal(
            conn, user_id="userA", capture_id=cid, corrected_goal="banana",
        ) is False
        # unchanged
        val = (await (await conn.execute(
            "SELECT corrected_goal FROM intent_capture WHERE id = ?", (cid,),
        )).fetchone())[0]
        assert val == ""
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_correct_is_user_scoped():
    """User B cannot relabel User A's capture (cross-tenant guard)."""
    backend = await _migrated_conn()
    try:
        conn = backend.conn
        cid = await _one_capture(conn, user_id="userA")
        assert await update_corrected_goal(
            conn, user_id="userB", capture_id=cid, corrected_goal="converse",
        ) is False
        val = (await (await conn.execute(
            "SELECT corrected_goal FROM intent_capture WHERE id = ?", (cid,),
        )).fetchone())[0]
        assert val == ""
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_correct_missing_row_returns_false():
    backend = await _migrated_conn()
    try:
        assert await update_corrected_goal(
            backend.conn, user_id="userA", capture_id="nope", corrected_goal="act",
        ) is False
    finally:
        await backend.close()


def test_valid_goals_match_router_vocabulary():
    # Guards against drift from architect/voice_router.py::_GOALS.
    assert frozenset(("act", "converse", "clarify", "idle", "drop")) == VALID_GOALS


@pytest.mark.asyncio
async def test_companion_recall_applies_settings_floor(monkeypatch):
    """The wrapper floors recall with companion_memory_min_score by default."""
    from augmentum.companion_runtime.memory import CompanionMemory
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_memory_min_score", 0.42, raising=False)

    cm = CompanionMemory(backend=MagicMock(), companion_id="becca")
    fake_store = MagicMock()
    fake_store.recall = AsyncMock(return_value=[])
    cm._store = fake_store

    await cm.recall("anything", user_id="userA", k=7)

    _, kwargs = fake_store.recall.call_args
    assert kwargs["min_score"] == 0.42
    assert kwargs["limit"] == 7
    assert kwargs["user_id"] == "userA"


@pytest.mark.asyncio
async def test_companion_recall_explicit_floor_overrides_setting(monkeypatch):
    """An explicit min_score=0.0 opts a call out of the floor."""
    from augmentum.companion_runtime.memory import CompanionMemory
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_memory_min_score", 0.55, raising=False)

    cm = CompanionMemory(backend=MagicMock(), companion_id="becca")
    fake_store = MagicMock()
    fake_store.recall = AsyncMock(return_value=[])
    cm._store = fake_store

    await cm.recall("anything", user_id="userA", k=5, min_score=0.0)

    _, kwargs = fake_store.recall.call_args
    assert kwargs["min_score"] == 0.0
