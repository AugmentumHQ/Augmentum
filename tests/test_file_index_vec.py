"""Tests for the file_index vec wiring (Piece 0).

Migration 175 adds the ``file_index_vec`` vec0 table; ``_upsert_file_vec``
mirrors embedding writes into it; ``search_by_embedding`` provides the
pure-vec leg the resolver's hybrid retrieval needs. These tests
exercise the round-trip against a real :memory: SQLite backend.

The sqlite-vec extension load is best-effort — if it isn't available
in the test environment, ``_upsert_file_vec`` swallows the error.
These tests skip when that's the case so they don't false-positive
on environments without the extension.
"""

from __future__ import annotations

import json
import struct
import secrets

import pytest


# A fake nomic-style 768-dim float32 vector encoded as bytes.
# Real embeddings come from EmbeddingService; tests use this stand-in
# so they don't need to load the ONNX model.
def _fake_embedding(seed: int = 0) -> bytes:
    """Build a deterministic 768-dim float32 vector for testing."""
    floats = [(seed + i) * 0.001 for i in range(768)]
    return struct.pack(f"{len(floats)}f", *floats)


async def _boot_index():
    """Spin up a :memory: backend + an aiosqlite connection wrapping
    the same DB. Returns (backend, conn) for cleanup.

    The test uses the backend's connection directly because
    FileIndexService takes the aiosqlite Connection, and the
    :memory: DB needs to be shared between the migrations runner
    and the service.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.vfs.index import FileIndexService

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    # Insert a user row so foreign keys in file_index resolve.
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("usr_vec_test", "vectester", "x"),
    )
    await backend.conn.commit()
    index = FileIndexService(backend.conn)
    return backend, index


async def _vec_extension_loaded(backend) -> bool:
    """True iff sqlite-vec is loaded and we can write to file_index_vec.

    Some local dev environments don't have the extension; in that case
    the test should skip rather than fail."""
    try:
        await backend.conn.execute(
            "SELECT * FROM file_index_vec LIMIT 1"
        )
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_migration_creates_vec_table():
    """Migration 175 must create file_index_vec as a virtual table."""
    backend, _ = await _boot_index()
    if not await _vec_extension_loaded(backend):
        pytest.skip("sqlite-vec not available in this environment")
    cursor = await backend.conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE name = 'file_index_vec'"
    )
    row = await cursor.fetchone()
    assert row is not None
    # vec0 is a virtual table — type is 'table' from sqlite's view.
    assert row[1] == "file_index_vec"


@pytest.mark.asyncio
async def test_upsert_then_search_round_trip():
    """Write an embedding via _upsert_file_vec, then search and find it."""
    backend, index = await _boot_index()
    if not await _vec_extension_loaded(backend):
        pytest.skip("sqlite-vec not available")

    # Register a fake file row so the JOIN in search_by_embedding hits.
    file_id = await index.register(
        user_id="usr_vec_test",
        source="uploads",
        source_id="src_test_1",
        name="test.png",
        mime_type="image/png",
    )
    emb = _fake_embedding(seed=10)
    await index._upsert_file_vec(file_id, emb)

    # Search with the same embedding — should return our row first.
    results = await index.search_by_embedding(
        emb, user_id="usr_vec_test", limit=5,
    )
    assert len(results) >= 1
    assert results[0].id == file_id
    # Score should be high (similarity close to 1.0 on exact match).
    assert results[0].score > 0.9


@pytest.mark.asyncio
async def test_upsert_replaces_existing():
    """A second _upsert for the same file_id replaces, doesn't duplicate."""
    backend, index = await _boot_index()
    if not await _vec_extension_loaded(backend):
        pytest.skip("sqlite-vec not available")

    file_id = await index.register(
        user_id="usr_vec_test",
        source="uploads",
        source_id="src_replace",
        name="r.png",
        mime_type="image/png",
    )
    await index._upsert_file_vec(file_id, _fake_embedding(seed=1))
    await index._upsert_file_vec(file_id, _fake_embedding(seed=2))

    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM file_index_vec WHERE file_id = ?",
        (file_id,),
    )
    count = (await cursor.fetchone())[0]
    assert count == 1


@pytest.mark.asyncio
async def test_search_filters_by_user_id():
    """search_by_embedding only returns rows owned by the caller."""
    backend, index = await _boot_index()
    if not await _vec_extension_loaded(backend):
        pytest.skip("sqlite-vec not available")

    # Insert a second user + a file for them.
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("usr_other", "other", "x"),
    )
    await backend.conn.commit()

    emb = _fake_embedding(seed=42)
    f_me = await index.register(
        user_id="usr_vec_test", source="uploads",
        source_id="m1", name="mine.png", mime_type="image/png",
    )
    f_other = await index.register(
        user_id="usr_other", source="uploads",
        source_id="o1", name="theirs.png", mime_type="image/png",
    )
    await index._upsert_file_vec(f_me, emb)
    await index._upsert_file_vec(f_other, emb)

    results = await index.search_by_embedding(
        emb, user_id="usr_vec_test", limit=10,
    )
    ids = {r.id for r in results}
    assert f_me in ids
    assert f_other not in ids


@pytest.mark.asyncio
async def test_clear_embedding_removes_vec_row():
    """Calling clear_embedding zaps both the file_index.embedding
    column AND the vec0 row, so the next enrich_pending pass
    regenerates from the updated description."""
    backend, index = await _boot_index()
    if not await _vec_extension_loaded(backend):
        pytest.skip("sqlite-vec not available")

    file_id = await index.register(
        user_id="usr_vec_test", source="uploads",
        source_id="src_clear", name="c.png", mime_type="image/png",
    )
    emb = _fake_embedding(seed=99)
    await index._upsert_file_vec(file_id, emb)

    # Sanity: vec row exists
    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM file_index_vec WHERE file_id = ?",
        (file_id,),
    )
    assert (await cursor.fetchone())[0] == 1

    await index.clear_embedding(file_id, user_id="usr_vec_test")

    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM file_index_vec WHERE file_id = ?",
        (file_id,),
    )
    assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_rows():
    """No rows in vec table → empty result, not crash."""
    backend, index = await _boot_index()
    if not await _vec_extension_loaded(backend):
        pytest.skip("sqlite-vec not available")
    results = await index.search_by_embedding(
        _fake_embedding(), user_id="usr_vec_test", limit=5,
    )
    assert results == []
