"""Tests for token count cache."""
from __future__ import annotations

import time

import pytest
import pytest_asyncio

from augmentum.models.token_count_cache import TokenCountCache


@pytest_asyncio.fixture
async def cache(tmp_path):
    db_path = str(tmp_path / "test_tc.db")
    c = TokenCountCache(db_path)
    await c.init_db()
    return c


@pytest.mark.asyncio
async def test_store_and_get(cache):
    await cache.store_count("model-a", "hello world", 2)
    result = await cache.get_count("model-a", "hello world")
    assert result == 2


@pytest.mark.asyncio
async def test_miss_returns_none(cache):
    result = await cache.get_count("model-a", "not stored")
    assert result is None


@pytest.mark.asyncio
async def test_different_models_different_keys(cache):
    await cache.store_count("model-a", "same text", 10)
    await cache.store_count("model-b", "same text", 15)
    assert await cache.get_count("model-a", "same text") == 10
    assert await cache.get_count("model-b", "same text") == 15


@pytest.mark.asyncio
async def test_use_count_increments(cache):
    await cache.store_count("model-a", "hello", 5)
    # Initial store sets use_count=0, each get increments
    await cache.get_count("model-a", "hello")
    await cache.get_count("model-a", "hello")
    await cache.get_count("model-a", "hello")

    s = await cache.stats()
    assert s["total_entries"] == 1
    assert s["distinct_models"] == 1

    # Verify use_count is 3 via direct query
    row = await cache._execute(
        "SELECT use_count FROM token_counts WHERE model_id = ?",
        ("model-a",),
    )
    assert row[0][0] == 3


@pytest.mark.asyncio
async def test_evict_stale(cache):
    await cache.store_count("model-a", "keep me", 10)
    await cache.store_count("model-a", "evict me", 5)

    # Access "keep me" enough times to pass min_uses threshold
    for _ in range(4):
        await cache.get_count("model-a", "keep me")

    # Backdate "evict me" to 60 days ago
    old_ts = time.time() - 60 * 86400
    key = cache._key("model-a", "evict me")
    await cache._execute(
        "UPDATE token_counts SET last_used = ? WHERE id = ?",
        (old_ts, key),
    )

    evicted = await cache.evict_stale(max_age_days=30, min_uses=3)
    assert evicted == 1

    assert await cache.get_count("model-a", "evict me") is None
    assert await cache.get_count("model-a", "keep me") == 10


@pytest.mark.asyncio
async def test_purge_model(cache):
    await cache.store_count("model-a", "text1", 10)
    await cache.store_count("model-a", "text2", 20)
    await cache.store_count("model-b", "text1", 30)

    deleted = await cache.purge_model("model-a")
    assert deleted == 2

    assert await cache.get_count("model-a", "text1") is None
    assert await cache.get_count("model-b", "text1") == 30


@pytest.mark.asyncio
async def test_upsert_overwrites(cache):
    await cache.store_count("model-a", "text", 10)
    await cache.store_count("model-a", "text", 20)
    assert await cache.get_count("model-a", "text") == 20


@pytest.mark.asyncio
async def test_recovers_from_corrupt_cache_file(tmp_path):
    db_path = tmp_path / "corrupt_tc.db"
    cache = TokenCountCache(str(db_path))
    await cache.init_db()

    await cache.store_count("model-a", "before", 1)
    assert await cache.get_count("model-a", "before") == 1

    db_path.write_bytes(b"not a sqlite database")

    # The failing read should trigger a cache rebuild instead of bubbling up.
    assert await cache.get_count("model-a", "before") is None

    # After recovery, the cache should accept new writes normally.
    await cache.store_count("model-a", "after", 2)
    assert await cache.get_count("model-a", "after") == 2
