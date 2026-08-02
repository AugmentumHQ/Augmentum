"""Tests for the Piece 4 journal extension (content_refs + place_ref).

Migration 176 adds two columns to companion_journal that turn the
journal into the Reference Resolver's index. The journal() write path
should accept and persist them; reads should return them. This test
hits a real :memory: SQLite backend so the migration also gets
exercised as a side effect.
"""

from __future__ import annotations

import json

import pytest


async def _boot_backend():
    """Spin up a :memory: SQLite backend with migrations applied."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


@pytest.mark.asyncio
async def test_migration_added_content_refs_and_place_ref_columns():
    """Columns must exist after migrations run."""
    backend = await _boot_backend()
    cursor = await backend.conn.execute("PRAGMA table_info(companion_journal)")
    rows = await cursor.fetchall()
    cols = {r[1] for r in rows}
    assert "content_refs" in cols
    assert "place_ref" in cols
    assert "embedding" in cols  # existing — sanity check migration 154 still in play


@pytest.mark.asyncio
async def test_journal_write_accepts_content_refs_and_place_ref():
    """The journal() helper persists the new fields."""
    from augmentum.companion_runtime.memory import CompanionMemory

    backend = await _boot_backend()
    mem = CompanionMemory(backend, companion_id="becca")

    refs = [
        {"kind": "file_index", "id": "fi_test"},
        {"kind": "chat_image", "id": "ci_test"},
    ]
    row_id = await mem.journal(
        "noticed something",
        entry_type="noticing",
        user_id="usr_test",
        content_refs=refs,
        place_ref="xrs_test_room",
        embed=False,  # skip embed for test speed
    )
    assert row_id > 0

    cursor = await backend.conn.execute(
        "SELECT content_refs, place_ref FROM companion_journal WHERE id = ?",
        (row_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    stored_refs = json.loads(row[0])
    assert stored_refs == refs
    assert row[1] == "xrs_test_room"


@pytest.mark.asyncio
async def test_journal_write_defaults_empty_refs():
    """Omitting content_refs/place_ref → empty defaults (not NULL)."""
    from augmentum.companion_runtime.memory import CompanionMemory

    backend = await _boot_backend()
    mem = CompanionMemory(backend, companion_id="becca")

    row_id = await mem.journal("plain entry", embed=False)

    cursor = await backend.conn.execute(
        "SELECT content_refs, place_ref FROM companion_journal WHERE id = ?",
        (row_id,),
    )
    row = await cursor.fetchone()
    # JSON empty array, not NULL — the resolver always iterates this.
    assert row[0] == "[]"
    assert row[1] == ""


@pytest.mark.asyncio
async def test_journal_kwarg_compatibility():
    """Old-style calls (no content_refs/place_ref) must keep working —
    Sprint 4a code still writes journal entries without these fields."""
    from augmentum.companion_runtime.memory import CompanionMemory

    backend = await _boot_backend()
    mem = CompanionMemory(backend, companion_id="becca")

    # Equivalent to what activity_selector._perform_journal does today:
    row_id = await mem.journal(
        content="legacy-shape entry",
        entry_type="noticing",
        user_id="",
        affect_tag="settled",
    )
    assert row_id > 0
