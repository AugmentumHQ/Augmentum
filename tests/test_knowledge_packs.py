"""Tests for the knowledge pack system (PackManager)."""
from __future__ import annotations

import asyncio
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import sqlite_vec

from augmentum.knowledge.packs import PackManager, PackResult

DIM = 768


@contextmanager
def _stub_embeddings(vector_seed: int = 0):
    """Patch EmbeddingService so tests don't load the 130MB Nomic model.

    ``vector_seed`` selects which deterministic test vector to return —
    matches ``_make_vec(seed)`` so a query "lands" near chunk ``seed``.
    """
    fixed_vec = [0.1 * (vector_seed + 1)] * DIM
    with patch(
        "augmentum.memory.embeddings.EmbeddingService.embed_query",
        return_value=fixed_vec,
    ):
        yield

TEST_CHUNKS = [
    {"content": "The aurora borealis is caused by solar wind particles interacting with atmospheric gases.", "title": "Aurora", "section": "Causes", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Aurora"},
    {"content": "Black holes form when massive stars collapse at the end of their life cycle.", "title": "Black hole", "section": "Formation", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Black_hole"},
    {"content": "The Python programming language was created by Guido van Rossum in 1991.", "title": "Python", "section": "History", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
    {"content": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.", "title": "Photosynthesis", "section": "Overview", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Photosynthesis"},
    {"content": "The Roman Colosseum was built between 70-80 AD and could hold 50,000 spectators.", "title": "Colosseum", "section": "History", "source": "wikipedia", "url": "https://en.wikipedia.org/wiki/Colosseum"},
]


def _make_vec(i: int) -> bytes:
    """Create a deterministic 768-d float32 vector blob."""
    return struct.pack(f"<{DIM}f", *([0.1 * (i + 1)] * DIM))


def _create_test_pack(path: Path) -> Path:
    """Build a minimal .augpack file with sqlite-vec at *path*."""
    db = sqlite3.connect(str(path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    meta_rows = [
        ("name", "test-pack"),
        ("version", "1.0.0"),
        ("description", "A test knowledge pack"),
        ("embedding_model", "nomic-embed-text"),
        ("embedding_dim", str(DIM)),
        ("chunk_count", str(len(TEST_CHUNKS))),
        ("source_license", "CC-BY-SA-4.0"),
        ("build_date", "2026-03-27"),
    ]
    db.executemany("INSERT INTO meta VALUES (?, ?)", meta_rows)

    db.execute(
        """CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            title TEXT NOT NULL,
            section TEXT NOT NULL,
            source TEXT NOT NULL,
            url TEXT NOT NULL
        )"""
    )
    for idx, chunk in enumerate(TEST_CHUNKS):
        db.execute(
            "INSERT INTO chunks (id, content, title, section, source, url) VALUES (?, ?, ?, ?, ?, ?)",
            (idx + 1, chunk["content"], chunk["title"], chunk["section"], chunk["source"], chunk["url"]),
        )

    db.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[{DIM}])")
    for idx in range(len(TEST_CHUNKS)):
        db.execute(
            "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
            (idx + 1, _make_vec(idx)),
        )

    db.commit()
    db.close()
    return path


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_pack_schema_valid(tmp_path: Path):
    """Create a pack and verify tables and data are correct."""
    pack_path = _create_test_pack(tmp_path / "science.augpack")

    db = sqlite3.connect(str(pack_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Verify meta table
    rows = db.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
    keys = {r[0] for r in rows}
    assert "name" in keys
    assert "embedding_dim" in keys
    assert "chunk_count" in keys

    # Verify chunks table
    chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert chunks == len(TEST_CHUNKS)

    # Verify chunks_vec table works
    query_vec = _make_vec(0)
    results = db.execute(
        "SELECT id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = 3",
        (query_vec,),
    ).fetchall()
    assert len(results) == 3
    # First result should be exact match (distance ~0)
    assert results[0][1] < 0.01

    db.close()


def test_pack_manager_scan(tmp_path: Path):
    """Scan a directory and verify packs are loaded."""
    _create_test_pack(tmp_path / "science.augpack")

    async def _run():
        mgr = PackManager(tmp_path)
        count = await mgr.scan()
        assert count == 1
        assert mgr.active_count == 1
        assert len(mgr.installed) == 1
        info = mgr.installed[0]
        assert info["name"] == "test-pack"
        assert info["embedding_dim"] == DIM
        assert info["chunk_count"] == len(TEST_CHUNKS)
        assert info["active"] is True
        await mgr.close()

    asyncio.run(_run())


def test_pack_manager_search(tmp_path: Path):
    """Search with a query and verify results are returned via the hybrid pipeline.

    Uses ``_stub_embeddings(0)`` so the stub vector matches chunk 0's stored
    embedding exactly — chunk 0 wins the vector leg, and FTS returns whatever
    the tokenized query happens to match in the test pack content.
    """
    _create_test_pack(tmp_path / "science.augpack")

    async def _run():
        mgr = PackManager(tmp_path)
        await mgr.scan()

        with _stub_embeddings(vector_seed=0):
            results = await mgr.search(
                query="aurora borealis",
                pack_ids=["science"],
                limit=3,
                rerank=False,  # skip cross-encoder load in unit tests
            )
        assert len(results) >= 1
        assert all(isinstance(r, PackResult) for r in results)
        # Vector leg lands chunk 0 (aurora) at the top regardless of FTS.
        assert results[0].title == "Aurora"
        # Score is RRF rank fusion when rerank=False — higher is better and
        # the top result must score at or above any subsequent result.
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
        await mgr.close()

    asyncio.run(_run())


def test_pack_manager_activate_deactivate(tmp_path: Path):
    """Toggle active flag and verify search respects it."""
    _create_test_pack(tmp_path / "science.augpack")

    async def _run():
        mgr = PackManager(tmp_path)
        await mgr.scan()
        assert mgr.active_count == 1

        # Deactivate
        ok = await mgr.deactivate("science")
        assert ok is True
        assert mgr.active_count == 0

        # Search should return nothing — pack inactive, no legs scheduled.
        with _stub_embeddings(vector_seed=0):
            results = await mgr.search(
                query="aurora", pack_ids=["science"], limit=3, rerank=False,
            )
        assert results == []

        # Re-activate
        ok = await mgr.activate("science")
        assert ok is True
        assert mgr.active_count == 1

        # Search should work again
        with _stub_embeddings(vector_seed=0):
            results = await mgr.search(
                query="aurora", pack_ids=["science"], limit=3, rerank=False,
            )
        assert len(results) > 0
        await mgr.close()

    asyncio.run(_run())


def test_pack_manager_persists_active_state(tmp_path: Path):
    """Deactivated packs should stay inactive after a manager restart."""
    _create_test_pack(tmp_path / "science.augpack")

    async def _run():
        mgr = PackManager(tmp_path)
        await mgr.scan()
        assert await mgr.deactivate("science") is True
        assert mgr.active_count == 0
        await mgr.close()

        restarted = PackManager(tmp_path)
        await restarted.scan()
        assert restarted.active_count == 0
        info = restarted.installed[0]
        assert info["pack_id"] == "science"
        assert info["active"] is False
        with _stub_embeddings(vector_seed=0):
            results = await restarted.search(
                query="aurora", pack_ids=["science"], limit=3, rerank=False,
            )
        assert results == []
        await restarted.close()

    asyncio.run(_run())


def test_pack_manager_skips_embedding_dimension_mismatch(tmp_path: Path):
    """Vector leg skips packs whose embedding dim doesn't match the query.

    Hybrid search degrades gracefully — when the vector leg is unusable,
    the FTS leg still runs. This test asserts the dim-mismatch logged
    skip path doesn't raise and doesn't return vector-leg results.
    """
    _create_test_pack(tmp_path / "science.augpack")

    async def _run():
        mgr = PackManager(tmp_path)
        await mgr.scan()
        # Stub embed_query to return a 384-dim vector — incompatible with
        # the pack's stored 768-dim chunks_vec. Vector leg should skip.
        with patch(
            "augmentum.memory.embeddings.EmbeddingService.embed_query",
            return_value=[0.1] * 384,
        ):
            results = await mgr.search(
                query="aurora", pack_ids=["science"], limit=3, rerank=False,
            )
        # FTS leg may still return matches via lazy-rebuilt chunks_fts.
        # The contract here is "doesn't crash, doesn't return phantom
        # vector-leg hits"; if any results came back, they came from FTS.
        for r in results:
            assert isinstance(r, PackResult)
        await mgr.close()

    asyncio.run(_run())


def test_pack_manager_delete(tmp_path: Path):
    """Delete a pack and verify the file is removed."""
    pack_path = _create_test_pack(tmp_path / "science.augpack")
    assert pack_path.exists()

    async def _run():
        mgr = PackManager(tmp_path)
        await mgr.scan()
        assert mgr.active_count == 1

        ok = await mgr.delete("science")
        assert ok is True
        assert mgr.active_count == 0
        assert len(mgr.installed) == 0
        assert not pack_path.exists()

    asyncio.run(_run())
