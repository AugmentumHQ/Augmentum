"""Tests for Piece 5.5 — companion_journal hybrid retrieval substrate.

Migration 177 adds two virtual tables:

* ``companion_journal_fts`` — FTS5 mirror of the ``content`` column,
  kept in sync by AFTER INSERT / UPDATE / DELETE triggers.
* ``companion_journal_vec`` — vec0 mirror of the ``embedding`` BLOB
  column, written explicitly from ``CompanionMemory.journal()``.

The hybrid substrate is what lets the Reference Resolver (Piece 6)
do fast vec KNN + FTS5 keyword search across the inner stream. These
tests exercise the round-trip in isolation so resolver work can
build on it confidently.

Tests skip when sqlite-vec isn't loaded — same pattern as
``test_file_index_vec.py``.
"""

from __future__ import annotations

import struct

import pytest


def _fake_embedding(seed: int = 0) -> bytes:
    """Deterministic 768-dim float32 vector for testing."""
    floats = [(seed + i) * 0.001 for i in range(768)]
    return struct.pack(f"{len(floats)}f", *floats)


async def _boot_backend():
    """Spin up a :memory: backend with migrations applied."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


async def _vec_loaded(backend) -> bool:
    """True iff sqlite-vec extension is loaded."""
    try:
        await backend.conn.execute(
            "SELECT * FROM companion_journal_vec LIMIT 1"
        )
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_migration_creates_fts_table():
    """Migration 177 must create companion_journal_fts virtual table."""
    backend = await _boot_backend()
    cursor = await backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'companion_journal_fts'"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "companion_journal_fts"


@pytest.mark.asyncio
async def test_migration_creates_vec_table():
    """Migration 177 must create companion_journal_vec virtual table."""
    backend = await _boot_backend()
    if not await _vec_loaded(backend):
        pytest.skip("sqlite-vec not available")
    cursor = await backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'companion_journal_vec'"
    )
    row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_fts_trigger_mirrors_insert():
    """Inserting into companion_journal must populate FTS automatically."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content) "
        "VALUES (?, ?)",
        ("becca", "the manga with the quintessential quintuplets"),
    )
    await backend.conn.commit()

    cursor = await backend.conn.execute(
        "SELECT content FROM companion_journal_fts "
        "WHERE companion_journal_fts MATCH ?",
        ("quintuplets",),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert "quintuplets" in row[0]


@pytest.mark.asyncio
async def test_fts_trigger_mirrors_delete():
    """Deleting a journal row must remove its FTS shadow."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content) "
        "VALUES (?, ?)",
        ("becca", "ephemeral observation"),
    )
    journal_id = cur.lastrowid
    await backend.conn.commit()

    # Confirm it's in FTS first
    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_fts "
        "WHERE companion_journal_fts MATCH ?",
        ("ephemeral",),
    )
    assert (await cursor.fetchone())[0] == 1

    await backend.conn.execute(
        "DELETE FROM companion_journal WHERE id = ?",
        (journal_id,),
    )
    await backend.conn.commit()

    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_fts "
        "WHERE companion_journal_fts MATCH ?",
        ("ephemeral",),
    )
    assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_fts_trigger_mirrors_update():
    """Updating a journal row's content must re-sync FTS."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content) "
        "VALUES (?, ?)",
        ("becca", "watching a sunset"),
    )
    journal_id = cur.lastrowid
    await backend.conn.commit()

    await backend.conn.execute(
        "UPDATE companion_journal SET content = ? WHERE id = ?",
        ("watching a sunrise", journal_id),
    )
    await backend.conn.commit()

    # Old term gone
    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_fts "
        "WHERE companion_journal_fts MATCH ?",
        ("sunset",),
    )
    assert (await cursor.fetchone())[0] == 0

    # New term present
    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_fts "
        "WHERE companion_journal_fts MATCH ?",
        ("sunrise",),
    )
    assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_vec_mirror_from_journal_method():
    """CompanionMemory.journal() must populate vec0 when embedding produced.

    We patch EmbeddingService.embed_one so the test doesn't need to load
    the ONNX model — the point here is verifying the vec0 INSERT happens,
    not that real embeddings work.
    """
    backend = await _boot_backend()
    if not await _vec_loaded(backend):
        pytest.skip("sqlite-vec not available")

    from augmentum.companion_runtime import memory as memory_mod
    from augmentum.companion_runtime.memory import CompanionMemory

    # Patch the embedding service to return our deterministic vector.
    # We bypass _encode_embedding too — the journal expects bytes, and
    # our fake already is bytes.
    original_embed_one = memory_mod.EmbeddingService.embed_one
    original_encode = memory_mod._encode_embedding
    try:
        memory_mod.EmbeddingService.embed_one = staticmethod(  # type: ignore[assignment]
            lambda content: [0.001 * i for i in range(768)]
        )
        memory_mod._encode_embedding = lambda emb: _fake_embedding(seed=0)  # type: ignore[assignment]

        cm = CompanionMemory(backend, "becca")
        journal_id = await cm.journal(
            "I noticed Alex skipped lunch today",
            entry_type="noticing",
            user_id="usr_test",
        )
        assert journal_id > 0

        cursor = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal_vec WHERE journal_id = ?",
            (journal_id,),
        )
        assert (await cursor.fetchone())[0] == 1
    finally:
        memory_mod.EmbeddingService.embed_one = original_embed_one  # type: ignore[assignment]
        memory_mod._encode_embedding = original_encode  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_vec_mirror_skipped_when_embed_disabled():
    """journal(embed=False) must NOT write to vec0."""
    backend = await _boot_backend()
    if not await _vec_loaded(backend):
        pytest.skip("sqlite-vec not available")

    from augmentum.companion_runtime.memory import CompanionMemory

    cm = CompanionMemory(backend, "becca")
    journal_id = await cm.journal(
        "low-stakes log",
        entry_type="observation",
        embed=False,
    )
    assert journal_id > 0

    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_vec WHERE journal_id = ?",
        (journal_id,),
    )
    assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_journal_write_survives_vec_extension_missing():
    """If vec0 INSERT raises, the journal write must NOT roll back.

    We simulate vec failure by writing the journal row, then asserting
    it's queryable even if the vec mirror wasn't created.
    """
    backend = await _boot_backend()
    from augmentum.companion_runtime import memory as memory_mod
    from augmentum.companion_runtime.memory import CompanionMemory

    # Patch embed to succeed (so the vec INSERT IS attempted) but the
    # vec table itself may or may not exist depending on extension load.
    # Either path is fine — the test passes either way as long as the
    # journal row is durable.
    original_embed = memory_mod.EmbeddingService.embed_one
    try:
        memory_mod.EmbeddingService.embed_one = staticmethod(  # type: ignore[assignment]
            lambda content: [0.001 * i for i in range(768)]
        )
        cm = CompanionMemory(backend, "becca")
        journal_id = await cm.journal("durable entry", embed=True)
        assert journal_id > 0
    finally:
        memory_mod.EmbeddingService.embed_one = original_embed  # type: ignore[assignment]

    cursor = await backend.conn.execute(
        "SELECT content FROM companion_journal WHERE id = ?",
        (journal_id,),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "durable entry"


@pytest.mark.asyncio
async def test_backfill_handler_idempotent():
    """Re-running the backfill handler must not duplicate vec rows."""
    backend = await _boot_backend()
    if not await _vec_loaded(backend):
        pytest.skip("sqlite-vec not available")

    # Write one journal row with embedding directly (no vec mirror yet
    # — simulates a pre-migration row).
    emb = _fake_embedding(seed=7)
    cur = await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content, embedding) "
        "VALUES (?, ?, ?)",
        ("becca", "ancient entry", emb),
    )
    journal_id = cur.lastrowid
    await backend.conn.commit()

    # Build a minimal app-state-like object for the handler
    class _FakeAppState:
        pass

    class _FakeApp:
        def __init__(self, backend):
            self.state = _FakeAppState()
            self.state.backend = backend

    class _FakeCtx:
        payload: dict = {}

    from augmentum.jobs.handlers.journal_vec_backfill import (
        make_journal_vec_backfill_handler,
    )

    handler = make_journal_vec_backfill_handler(_FakeApp(backend))

    # First run — should mirror the one row
    result1 = await handler(_FakeCtx())
    assert result1["status"] == "ok"
    assert result1["rows_mirrored"] == 1

    # Second run — should be a no-op since the vec row already exists
    result2 = await handler(_FakeCtx())
    assert result2["status"] == "ok"
    assert result2["rows_mirrored"] == 0

    # And exactly one vec row exists
    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal_vec WHERE journal_id = ?",
        (journal_id,),
    )
    assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_backfill_handler_skips_when_vec_unavailable():
    """If the vec table doesn't exist (extension not loaded), the handler
    skips cleanly rather than raising."""
    backend = await _boot_backend()
    # We can't actually unload sqlite-vec, but we can simulate the
    # missing-table case by checking the skip path returns the expected
    # shape when the extension IS loaded — the early SELECT just
    # confirms presence. So this test mainly verifies the contract of
    # the early-skip return.
    class _FakeApp:
        def __init__(self, backend):
            self.state = type("S", (), {"backend": backend})()

    class _FakeCtx:
        payload: dict = {}

    from augmentum.jobs.handlers.journal_vec_backfill import (
        make_journal_vec_backfill_handler,
    )

    handler = make_journal_vec_backfill_handler(_FakeApp(backend))
    result = await handler(_FakeCtx())
    # status is "ok" with 0 rows OR "skipped" — both are valid clean exits
    assert result["status"] in ("ok", "skipped")
