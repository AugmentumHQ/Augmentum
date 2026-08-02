"""Tests for DiscoveryStore — signals, history, and content library."""
from __future__ import annotations

import asyncio
import struct
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from augmentum.state.discovery_store import DiscoveryStore

MIGRATION_PATH = "augmentum/state/migrations/067_discovery_engine.sql"


@pytest.fixture
async def store():
    async with aiosqlite.connect(":memory:") as conn:
        sql = open(MIGRATION_PATH).read()
        # Split on virtual table creation — vec0 extension won't be present in tests
        parts = sql.split("CREATE VIRTUAL TABLE")
        await conn.executescript(parts[0])
        for part in parts[1:]:
            try:
                await conn.executescript("CREATE VIRTUAL TABLE" + part)
            except Exception:
                pass  # vec0 not available in test environment
        # user_id column landed in migration 093 — add it directly so we
        # don't drag in the users table dependency.
        for table in ("browse_history", "interaction_signals", "interest_clusters"):
            try:
                await conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
            except Exception:
                pass
        await conn.commit()
        yield DiscoveryStore(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(dim: int = 768) -> bytes:
    """Return a trivially normalised float32 embedding as bytes."""
    val = 1.0 / (dim ** 0.5)
    return struct.pack(f"{dim}f", *([val] * dim))


# ===========================================================================
# TestSignals
# ===========================================================================

class TestSignals:
    @pytest.mark.asyncio
    async def test_log_signal_creates_with_correct_fields(self, store):
        sig = await store.log_signal(
            signal_type="view",
            source_url="https://example.com/article",
            source_title="Example Article",
            content_type="article",
            weight=1.0,
            metadata={"scroll": 0.8},
            user_id="user_test",
        )
        assert sig["id"]
        assert sig["signal_type"] == "view"
        assert sig["source_url"] == "https://example.com/article"
        assert sig["source_title"] == "Example Article"
        assert sig["source_domain"] == "example.com"
        assert sig["content_type"] == "article"
        assert sig["weight"] == 1.0
        assert sig["metadata"]["scroll"] == 0.8
        assert sig["deduplicated"] is False

    @pytest.mark.asyncio
    async def test_dedup_within_30_minutes_returns_same_id(self, store):
        sig1 = await store.log_signal(
            signal_type="view",
            source_url="https://example.com/page",
            source_title="Page",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        sig2 = await store.log_signal(
            signal_type="view",
            source_url="https://example.com/page",
            source_title="Page",
            content_type="article",
            weight=1.0,
            metadata={"extra": "data"},
            user_id="user_test",
        )
        assert sig2["id"] == sig1["id"]
        assert sig2["deduplicated"] is True

    @pytest.mark.asyncio
    async def test_different_signal_types_not_deduped(self, store):
        sig1 = await store.log_signal(
            signal_type="view",
            source_url="https://example.com/page",
            source_title="Page",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        sig2 = await store.log_signal(
            signal_type="click",
            source_url="https://example.com/page",
            source_title="Page",
            content_type="article",
            weight=2.0,
            metadata={},
            user_id="user_test",
        )
        assert sig1["id"] != sig2["id"]
        assert sig2["deduplicated"] is False

    @pytest.mark.asyncio
    async def test_list_signals_returns_all(self, store):
        await store.log_signal(
            signal_type="view",
            source_url="https://a.com/1",
            source_title="A1",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        await store.log_signal(
            signal_type="view",
            source_url="https://b.com/1",
            source_title="B1",
            content_type="video",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        signals = await store.list_signals(user_id="user_test")
        assert len(signals) == 2

    @pytest.mark.asyncio
    async def test_list_signals_filter_by_type(self, store):
        await store.log_signal(
            signal_type="view",
            source_url="https://a.com/1",
            source_title="A",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        await store.log_signal(
            signal_type="bookmark",
            source_url="https://b.com/1",
            source_title="B",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        views = await store.list_signals(signal_type="view", user_id="user_test")
        assert len(views) == 1
        assert views[0]["signal_type"] == "view"

    @pytest.mark.asyncio
    async def test_list_signals_filter_by_url(self, store):
        url = "https://specific.com/post"
        await store.log_signal(
            signal_type="view",
            source_url=url,
            source_title="S",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        await store.log_signal(
            signal_type="view",
            source_url="https://other.com/post",
            source_title="O",
            content_type="article",
            weight=1.0,
            metadata={},
            user_id="user_test",
        )
        results = await store.list_signals(source_url=url, user_id="user_test")
        assert len(results) == 1
        assert results[0]["source_url"] == url


# ===========================================================================
# TestHistory
# ===========================================================================

class TestHistory:
    @pytest.mark.asyncio
    async def test_upsert_creates_with_visit_count_1(self, store):
        entry = await store.upsert_history(
            url="https://example.com/article",
            title="Example Article",
            domain="example.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        assert entry["id"]
        assert entry["url"] == "https://example.com/article"
        assert entry["visit_count"] == 1

    @pytest.mark.asyncio
    async def test_upsert_increments_visit_count(self, store):
        url = "https://example.com/article"
        await store.upsert_history(
            url=url,
            title="Example Article",
            domain="example.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        entry2 = await store.upsert_history(
            url=url,
            title="Example Article Updated",
            domain="example.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        assert entry2["visit_count"] == 2

    @pytest.mark.asyncio
    async def test_list_history_returns_newest_first(self, store):
        await store.upsert_history(
            url="https://example.com/first",
            title="First",
            domain="example.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        # Small sleep so timestamps differ
        await asyncio.sleep(0.01)
        await store.upsert_history(
            url="https://example.com/second",
            title="Second",
            domain="example.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        history = await store.list_history(user_id="user_test")
        assert history[0]["url"] == "https://example.com/second"

    @pytest.mark.asyncio
    async def test_delete_history(self, store):
        entry = await store.upsert_history(
            url="https://delete.com/me",
            title="Delete Me",
            domain="delete.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        result = await store.delete_history(entry["id"], user_id="user_test")
        assert result is True
        history = await store.list_history(user_id="user_test")
        assert all(h["id"] != entry["id"] for h in history)

    @pytest.mark.asyncio
    async def test_delete_history_nonexistent_returns_false(self, store):
        result = await store.delete_history("nonexistent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_visited_batch(self, store):
        url_a = "https://visited.com/a"
        url_b = "https://visited.com/b"
        await store.upsert_history(
            url=url_a,
            title="A",
            domain="visited.com",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        results = await store.check_visited([url_a, url_b, "https://not.visited/c"], user_id="user_test")
        assert url_a in results
        assert url_b not in results

    @pytest.mark.asyncio
    async def test_check_visited_urls_returns_only_present(self, store):
        """Light variant — returns set[str] of just the URLs that are
        present. Used by the discovery recommender's dedup path where
        the full row payload (with JSON metadata) is overhead."""
        url_a = "https://visited.com/a"
        url_b = "https://visited.com/b"
        await store.upsert_history(
            url=url_a, title="A", domain="visited.com",
            content_type="article", thumbnail="", metadata={},
            user_id="user_test",
        )
        result = await store.check_visited_urls(
            [url_a, url_b, "https://not.visited/c"],
            user_id="user_test",
        )
        assert isinstance(result, set)
        assert result == {url_a}

    @pytest.mark.asyncio
    async def test_check_visited_urls_empty_input_returns_empty_set(self, store):
        result = await store.check_visited_urls([], user_id="user_test")
        assert result == set()

    @pytest.mark.asyncio
    async def test_list_history_query_filter(self, store):
        await store.upsert_history(
            url="https://python.org/docs",
            title="Python Documentation",
            domain="python.org",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        await store.upsert_history(
            url="https://rust-lang.org/docs",
            title="Rust Documentation",
            domain="rust-lang.org",
            content_type="article",
            thumbnail="",
            metadata={},
            user_id="user_test",
        )
        results = await store.list_history(query="Python", user_id="user_test")
        assert len(results) == 1
        assert "python" in results[0]["url"].lower()


# ===========================================================================
# TestContentLibrary
# ===========================================================================

class TestContentLibrary:
    @pytest.mark.asyncio
    async def test_store_chunk_and_retrieve(self, store):
        chunk = await store.store_chunk(
            source_url="https://example.com/article",
            source_title="Example Article",
            source_type="article",
            content="This is the content of the article.",
            embedding=None,
            cluster_id=None,
        )
        assert chunk["chunk_id"]
        assert chunk["source_url"] == "https://example.com/article"
        assert chunk["content"] == "This is the content of the article."

        fetched = await store.get_chunk(chunk["chunk_id"])
        assert fetched is not None
        assert fetched["chunk_id"] == chunk["chunk_id"]
        assert fetched["content"] == chunk["content"]

    @pytest.mark.asyncio
    async def test_get_chunk_nonexistent(self, store):
        result = await store.get_chunk("nonexistent_chunk_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_chunks_by_source(self, store):
        url = "https://example.com/multi"
        for i in range(3):
            await store.store_chunk(
                source_url=url,
                source_title="Multi",
                source_type="article",
                content=f"Chunk {i}",
                embedding=None,
                cluster_id=None,
            )
        await store.store_chunk(
            source_url="https://other.com/page",
            source_title="Other",
            source_type="article",
            content="Different source",
            embedding=None,
            cluster_id=None,
        )
        chunks = await store.get_chunks_by_source(url)
        assert len(chunks) == 3
        assert all(c["source_url"] == url for c in chunks)

    @pytest.mark.asyncio
    async def test_increment_retrieved(self, store):
        chunk = await store.store_chunk(
            source_url="https://example.com/retrieve",
            source_title="Retrieve",
            source_type="article",
            content="Retrievable content",
            embedding=None,
            cluster_id=None,
        )
        assert chunk["retrieved_count"] == 0
        await store.increment_retrieved(chunk["chunk_id"])
        fetched = await store.get_chunk(chunk["chunk_id"])
        assert fetched["retrieved_count"] == 1

    @pytest.mark.asyncio
    async def test_has_source_true(self, store):
        url = "https://example.com/has-source"
        await store.store_chunk(
            source_url=url,
            source_title="Has Source",
            source_type="article",
            content="Content",
            embedding=None,
            cluster_id=None,
        )
        assert await store.has_source(url) is True

    @pytest.mark.asyncio
    async def test_has_source_false(self, store):
        assert await store.has_source("https://not-stored.com/page") is False

    @pytest.mark.asyncio
    async def test_store_chunk_with_embedding(self, store):
        emb = _make_embedding()
        chunk = await store.store_chunk(
            source_url="https://example.com/embedded",
            source_title="Embedded",
            source_type="article",
            content="Embedded content",
            embedding=emb,
            cluster_id=None,
        )
        assert chunk["chunk_id"]
        fetched = await store.get_chunk(chunk["chunk_id"])
        assert fetched["embedding"] == emb

    @pytest.mark.asyncio
    async def test_prune_old_content(self, store):
        chunk = await store.store_chunk(
            source_url="https://example.com/old",
            source_title="Old",
            source_type="article",
            content="Old content that should be pruned",
            embedding=None,
            cluster_id=None,
        )
        # Back-date the chunk
        old_ts = (datetime.now(UTC) - timedelta(days=400)).isoformat()
        async with store._conn.execute(
            "UPDATE content_library SET created_at = ? WHERE chunk_id = ?",
            (old_ts, chunk["chunk_id"]),
        ):
            pass
        await store._conn.commit()

        pruned = await store.prune_old_content(retention_days=365)
        assert pruned == 1
        fetched = await store.get_chunk(chunk["chunk_id"])
        assert "[pruned" in fetched["content"].lower()

    @pytest.mark.asyncio
    async def test_prune_skips_retrieved_chunks(self, store):
        chunk = await store.store_chunk(
            source_url="https://example.com/retrieved-old",
            source_title="Retrieved Old",
            source_type="article",
            content="Valuable content that was retrieved",
            embedding=None,
            cluster_id=None,
        )
        await store.increment_retrieved(chunk["chunk_id"])
        old_ts = (datetime.now(UTC) - timedelta(days=400)).isoformat()
        async with store._conn.execute(
            "UPDATE content_library SET created_at = ? WHERE chunk_id = ?",
            (old_ts, chunk["chunk_id"]),
        ):
            pass
        await store._conn.commit()

        pruned = await store.prune_old_content(retention_days=365)
        assert pruned == 0
        fetched = await store.get_chunk(chunk["chunk_id"])
        assert fetched["content"] == "Valuable content that was retrieved"
