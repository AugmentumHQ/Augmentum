"""Tests for PackManager's result cache and invalidation paths.

The cache is a hot path: any bug here either serves stale results to
users (cache invalidates wrong) or makes the cache look like it works
in benchmarks but actually misses constantly (key construction wrong).
Covered separately from the broader test_knowledge_packs.py because
this is exclusively about the cache contract.

Doesn't load real models — uses the same _stub_embeddings pattern from
the existing pack tests so we exercise the cache path without paying
a 130MB embedding model load per test.

Memory: building a real .augpack (even minimal) loads sqlite-vec +
opens the file, plus PackManager.scan() opens connections. Run on a
host with at least ~1GB free to the test process. CI runners are
fine; running inside the live Augmentum container alongside the
existing app may OOM (tested 2026-05-03).
"""
from __future__ import annotations

import asyncio
import sqlite3
import struct
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlite_vec

from augmentum.knowledge.packs import PackManager

DIM = 768


@contextmanager
def _stub_embeddings():
    fixed_vec = [0.1] * DIM
    with patch(
        "augmentum.memory.embeddings.EmbeddingService.embed_query",
        return_value=fixed_vec,
    ):
        yield


def _make_vec_blob(seed: float) -> bytes:
    return struct.pack(f"<{DIM}f", *([seed] * DIM))


def _create_minimal_pack(path: Path) -> None:
    """Build a tiny augpack with two chunks. Enough for a search to
    return non-empty so the cache stores it (empty results aren't
    cached — we don't want to mask "user just installed a pack")."""
    db = sqlite3.connect(str(path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    for k, v in [
        ("name", "test-pack"),
        ("embedding_dim", str(DIM)),
        ("chunk_count", "2"),
    ]:
        db.execute("INSERT INTO meta VALUES (?, ?)", (k, v))
    db.execute(
        "CREATE TABLE chunks ("
        "id INTEGER PRIMARY KEY, content TEXT, title TEXT, section TEXT, "
        "source TEXT, url TEXT)"
    )
    db.execute("INSERT INTO chunks VALUES (1, 'apple content', 'Apple', '', 'src', '')")
    db.execute("INSERT INTO chunks VALUES (2, 'banana content', 'Banana', '', 'src', '')")
    db.execute(
        f"CREATE VIRTUAL TABLE chunks_vec USING vec0(id INTEGER PRIMARY KEY, embedding float[{DIM}])"
    )
    db.execute("INSERT INTO chunks_vec VALUES (1, ?)", (_make_vec_blob(0.1),))
    db.execute("INSERT INTO chunks_vec VALUES (2, ?)", (_make_vec_blob(0.1),))
    db.commit()
    db.close()


@pytest.fixture
def pack_manager(tmp_path) -> PackManager:
    """Build a fresh PackManager backed by one minimal pack in tmp_path."""
    pack_path = tmp_path / "testpack.augpack"
    _create_minimal_pack(pack_path)
    mgr = PackManager(tmp_path)
    asyncio.run(mgr.scan())
    assert "testpack" in mgr._packs, "pack failed to load — fixture broken"
    return mgr


# ---------------------------------------------------------------------------
# Cache hits + key construction
# ---------------------------------------------------------------------------

def test_repeat_query_hits_cache(pack_manager):
    """Same query + same pack set + same caps within TTL → cache hit.
    The hits counter increments on a cache hit, misses on a miss."""
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9), (1, 0.8)],
    ):
        # First call — miss
        r1 = asyncio.run(pack_manager.search(
            query="apple",
            pack_ids=["testpack"],
            limit=2,
            rerank=False,
        ))
        miss_after_first = pack_manager._search_cache_misses
        hit_after_first = pack_manager._search_cache_hits

        # Second call, identical args — should hit
        r2 = asyncio.run(pack_manager.search(
            query="apple",
            pack_ids=["testpack"],
            limit=2,
            rerank=False,
        ))
        miss_after_second = pack_manager._search_cache_misses
        hit_after_second = pack_manager._search_cache_hits

    assert len(r1) == 2 and len(r2) == 2
    assert miss_after_second == miss_after_first, "second call should not miss"
    assert hit_after_second == hit_after_first + 1, "second call should hit"
    # Returned objects should be FRESH instances, not the cached ones —
    # caller mutations must not poison the cache.
    assert r1[0] is not r2[0], "returned PackResults must be deep copies"


def test_different_query_misses_cache(pack_manager):
    """Different query → different cache key → miss."""
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        asyncio.run(pack_manager.search(
            query="apple", pack_ids=["testpack"], limit=2, rerank=False,
        ))
        misses_before = pack_manager._search_cache_misses

        asyncio.run(pack_manager.search(
            query="banana", pack_ids=["testpack"], limit=2, rerank=False,
        ))
        misses_after = pack_manager._search_cache_misses

    assert misses_after == misses_before + 1


def test_different_limit_misses_cache(pack_manager):
    """Limit is part of the key — different limit re-runs the search.
    Without this, the cache would over-truncate small follow-up calls."""
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9), (1, 0.8)],
    ):
        asyncio.run(pack_manager.search(
            query="apple", pack_ids=["testpack"], limit=2, rerank=False,
        ))
        misses_before = pack_manager._search_cache_misses

        asyncio.run(pack_manager.search(
            query="apple", pack_ids=["testpack"], limit=5, rerank=False,
        ))
        misses_after = pack_manager._search_cache_misses

    assert misses_after == misses_before + 1


def test_pack_id_order_does_not_affect_cache_hit(pack_manager):
    """Cache key uses sorted pack_ids — calling with [B, A] hits the
    same entry as [A, B]. Without sorting, the same logical query made
    by different callers would pay the cost twice."""
    # Add a second pack so there's something to reorder.
    second_path = pack_manager._pack_dir / "second.augpack"
    _create_minimal_pack(second_path)
    asyncio.run(pack_manager.scan())

    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        asyncio.run(pack_manager.search(
            query="apple",
            pack_ids=["testpack", "second"],
            limit=2, rerank=False,
        ))
        misses_before = pack_manager._search_cache_misses

        asyncio.run(pack_manager.search(
            query="apple",
            pack_ids=["second", "testpack"],  # reversed
            limit=2, rerank=False,
        ))
        misses_after = pack_manager._search_cache_misses

    assert misses_after == misses_before, "pack ID order must not affect cache key"


# ---------------------------------------------------------------------------
# Invalidation contract
# ---------------------------------------------------------------------------

def test_activate_invalidates_cache(pack_manager):
    """Activating a pack changes the search corpus — cached results
    that didn't include the newly-active pack are now stale."""
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        asyncio.run(pack_manager.search(
            query="apple", pack_ids=["testpack"], limit=2, rerank=False,
        ))
        assert len(pack_manager._search_cache) == 1

        asyncio.run(pack_manager.activate("testpack"))
        assert len(pack_manager._search_cache) == 0, "activate must clear cache"


def test_deactivate_invalidates_cache(pack_manager):
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        asyncio.run(pack_manager.search(
            query="apple", pack_ids=["testpack"], limit=2, rerank=False,
        ))
        assert len(pack_manager._search_cache) == 1

        asyncio.run(pack_manager.deactivate("testpack"))
        assert len(pack_manager._search_cache) == 0, "deactivate must clear cache"


def test_delete_invalidates_cache(pack_manager):
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        asyncio.run(pack_manager.search(
            query="apple", pack_ids=["testpack"], limit=2, rerank=False,
        ))
        assert len(pack_manager._search_cache) == 1

        asyncio.run(pack_manager.delete("testpack"))
        assert len(pack_manager._search_cache) == 0, "delete must clear cache"


# ---------------------------------------------------------------------------
# Eviction + bounded growth
# ---------------------------------------------------------------------------

def test_cache_size_bounded(pack_manager, monkeypatch):
    """The cache must not grow unbounded — once over capacity, oldest
    LRU entries get evicted."""
    # Shrink cap so the test fires quickly.
    from augmentum.config import settings
    monkeypatch.setattr(settings, "knowledge_search_cache_size", 3)

    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        for q in ["a", "b", "c", "d", "e"]:
            asyncio.run(pack_manager.search(
                query=q, pack_ids=["testpack"], limit=2, rerank=False,
            ))

    assert len(pack_manager._search_cache) <= 3, "cache must respect size cap"


def test_empty_results_not_cached(pack_manager):
    """We don't cache empty result sets — a user installing a pack
    after a no-result search would otherwise see "no results" until
    TTL expired despite the new pack being live."""
    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[],
    ), patch.object(
        pack_manager, "_vector_leg", return_value=[],
    ):
        asyncio.run(pack_manager.search(
            query="zzznonexistentzzz",
            pack_ids=["testpack"],
            limit=2, rerank=False,
        ))

    assert len(pack_manager._search_cache) == 0, "empty results must not be cached"


def test_cache_disabled_setting(pack_manager, monkeypatch):
    """When cache is disabled via settings, no entries land in the
    cache regardless of activity. Lets users on memory-constrained
    boxes opt out cleanly."""
    from augmentum.config import settings
    monkeypatch.setattr(settings, "knowledge_search_cache_enabled", False)

    with _stub_embeddings(), patch(
        "augmentum.memory.reranker.RerankService.rerank",
        return_value=[(0, 0.9)],
    ):
        for q in ["a", "b", "c"]:
            asyncio.run(pack_manager.search(
                query=q, pack_ids=["testpack"], limit=2, rerank=False,
            ))

    assert len(pack_manager._search_cache) == 0
    # Hits should never increment when cache is off.
    assert pack_manager._search_cache_hits == 0
