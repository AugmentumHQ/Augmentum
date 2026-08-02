"""Tests for the prompt caching, prefix dedup, request deduplication, and cache API."""

from __future__ import annotations  # noqa: I001

import asyncio
from collections.abc import AsyncIterator

import pytest

from augmentum.cache.dedup import RequestDeduplicator
from augmentum.cache.prefix_cache import PrefixCache
from augmentum.cache.prompt_cache import CacheStats, PromptCache
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    content: str = "Hello",
    model: str = "llama3.1:8b",
    temperature: float | None = None,
    system: str = "You are helpful.",
) -> InternalChatRequest:
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=content),
    ]
    return InternalChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def _make_response(content: str = "Hi there!", model: str = "llama3.1:8b") -> InternalChatResponse:
    return InternalChatResponse(
        message=Message(role="assistant", content=content),
        model=model,
        finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class CountingBackend(ModelBackend):
    """Backend that counts calls and returns canned responses."""

    def __init__(self) -> None:
        self.call_count = 0

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        self.call_count += 1
        return _make_response(f"Response #{self.call_count}", model=request.model)

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        response = await self.chat(request)
        yield InternalStreamChunk(
            content_delta=response.message.content,
            role="assistant",
            model=request.model,
            done=True,
        )

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


class FailingBackend(ModelBackend):
    """Backend that always raises an error."""

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        msg = "Backend failure"
        raise RuntimeError(msg)

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        msg = "Backend failure"
        raise RuntimeError(msg)
        yield  # pragma: no cover  # noqa: B027 — unreachable, makes it an async gen

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


# ===========================================================================
# PromptCache Tests
# ===========================================================================


class TestPromptCacheHitMiss:
    """Cache hit and miss for identical prompts."""

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self):
        """First request should miss, second identical request should hit."""
        cache = PromptCache()
        req = _make_request("What is 2+2?")
        resp = _make_response("4")

        # Miss
        result = await cache.get(req)
        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

        # Store
        await cache.put(req, resp)
        assert cache.stats.stores == 1

        # Hit
        result = await cache.get(req)
        assert result is not None
        assert result.message.content == "4"
        assert cache.stats.hits == 1

    @pytest.mark.asyncio
    async def test_different_prompts_dont_collide(self):
        """Different prompts should have different cache keys."""
        cache = PromptCache()
        req1 = _make_request("What is 2+2?")
        req2 = _make_request("What is 3+3?")
        resp1 = _make_response("4")
        resp2 = _make_response("6")

        await cache.put(req1, resp1)
        await cache.put(req2, resp2)

        result1 = await cache.get(req1)
        result2 = await cache.get(req2)

        assert result1 is not None
        assert result1.message.content == "4"
        assert result2 is not None
        assert result2.message.content == "6"

    @pytest.mark.asyncio
    async def test_different_models_different_keys(self):
        """Same prompt but different models should cache separately."""
        cache = PromptCache()
        req1 = _make_request("Hello", model="llama3.1:8b")
        req2 = _make_request("Hello", model="mistral:7b")
        resp1 = _make_response("From llama")
        resp2 = _make_response("From mistral")

        await cache.put(req1, resp1)
        await cache.put(req2, resp2)

        result1 = await cache.get(req1)
        result2 = await cache.get(req2)

        assert result1.message.content == "From llama"
        assert result2.message.content == "From mistral"


class TestNonDeterministicNotCached:
    """Non-deterministic requests (temperature > 0) should not be cached."""

    @pytest.mark.asyncio
    async def test_temperature_nonzero_skipped(self):
        """Requests with temperature > 0 should not be stored or returned."""
        cache = PromptCache()
        req = _make_request("Hello", temperature=0.7)
        resp = _make_response("Hi")

        await cache.put(req, resp)
        assert cache.size == 0

        result = await cache.get(req)
        assert result is None
        assert cache.stats.skipped_non_deterministic == 1

    @pytest.mark.asyncio
    async def test_temperature_zero_cached(self):
        """Requests with temperature=0 should be cached."""
        cache = PromptCache()
        req = _make_request("Hello", temperature=0)
        resp = _make_response("Hi")

        await cache.put(req, resp)
        assert cache.size == 1

        result = await cache.get(req)
        assert result is not None

    @pytest.mark.asyncio
    async def test_temperature_none_cached(self):
        """Requests with temperature=None should be cached."""
        cache = PromptCache()
        req = _make_request("Hello", temperature=None)
        resp = _make_response("Hi")

        await cache.put(req, resp)
        assert cache.size == 1


class TestTTLExpiration:
    """TTL-based cache expiration."""

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self):
        """Expired entries should return None and be removed."""
        cache = PromptCache(ttl_seconds=0.01)  # 10ms TTL
        req = _make_request("Hello")
        resp = _make_response("Hi")

        await cache.put(req, resp)
        assert cache.size == 1

        # Wait for expiration
        await asyncio.sleep(0.05)

        result = await cache.get(req)
        assert result is None
        assert cache.stats.expirations == 1
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_non_expired_entry_returned(self):
        """Non-expired entries should be returned normally."""
        cache = PromptCache(ttl_seconds=10.0)  # 10 seconds
        req = _make_request("Hello")
        resp = _make_response("Hi")

        await cache.put(req, resp)
        result = await cache.get(req)
        assert result is not None


class TestLRUEviction:
    """LRU eviction when cache is at capacity."""

    @pytest.mark.asyncio
    async def test_evicts_lru_at_capacity(self):
        """When at capacity, the least recently used entry should be evicted."""
        cache = PromptCache(max_size=2)

        req1 = _make_request("First")
        req2 = _make_request("Second")
        req3 = _make_request("Third")

        await cache.put(req1, _make_response("R1"))
        await cache.put(req2, _make_response("R2"))
        # Cache is now at capacity (2)

        await cache.put(req3, _make_response("R3"))
        # First entry should be evicted

        assert cache.size == 2
        assert cache.stats.evictions == 1

        result1 = await cache.get(req1)
        assert result1 is None  # evicted

        result2 = await cache.get(req2)
        assert result2 is not None

        result3 = await cache.get(req3)
        assert result3 is not None

    @pytest.mark.asyncio
    async def test_access_refreshes_lru_order(self):
        """Accessing an entry should move it to most recently used."""
        cache = PromptCache(max_size=2)

        req1 = _make_request("First")
        req2 = _make_request("Second")
        req3 = _make_request("Third")

        await cache.put(req1, _make_response("R1"))
        await cache.put(req2, _make_response("R2"))

        # Access req1 to make it most recently used
        await cache.get(req1)

        # Adding req3 should evict req2 (the LRU) instead of req1
        await cache.put(req3, _make_response("R3"))

        result1 = await cache.get(req1)
        assert result1 is not None  # was accessed, so not evicted

        result2 = await cache.get(req2)
        assert result2 is None  # evicted


class TestKeyGeneration:
    """Cache key generation determinism."""

    def test_same_request_same_key(self):
        """Identical requests should produce the same key."""
        req1 = _make_request("Hello")
        req2 = _make_request("Hello")

        key1 = PromptCache.make_key(req1)
        key2 = PromptCache.make_key(req2)

        assert key1 == key2

    def test_different_content_different_key(self):
        """Different content should produce different keys."""
        req1 = _make_request("Hello")
        req2 = _make_request("Goodbye")

        key1 = PromptCache.make_key(req1)
        key2 = PromptCache.make_key(req2)

        assert key1 != key2

    def test_different_model_different_key(self):
        """Different models should produce different keys."""
        req1 = _make_request("Hello", model="llama3.1:8b")
        req2 = _make_request("Hello", model="mistral:7b")

        key1 = PromptCache.make_key(req1)
        key2 = PromptCache.make_key(req2)

        assert key1 != key2

    def test_key_is_hex_string(self):
        """Key should be a hex SHA-256 hash."""
        req = _make_request("Test")
        key = PromptCache.make_key(req)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in key)


class TestPatternInvalidation:
    """Pattern-based cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_matching_entries(self):
        """Entries matching the pattern should be removed."""
        cache = PromptCache()

        req1 = _make_request("Hello", system="You are an assessment engine.")
        req2 = _make_request("Hello", system="You are a verification engine.")
        req3 = _make_request("Hello", system="You are a synthesis engine.")

        await cache.put(req1, _make_response("R1"))
        await cache.put(req2, _make_response("R2"))
        await cache.put(req3, _make_response("R3"))

        assert cache.size == 3

        removed = await cache.invalidate_pattern("assessment|verification")
        assert removed == 2
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_invalidate_no_match(self):
        """Non-matching pattern should remove nothing."""
        cache = PromptCache()
        req = _make_request("Hello")
        await cache.put(req, _make_response("R"))

        removed = await cache.invalidate_pattern("nonexistent_pattern_xyz")
        assert removed == 0
        assert cache.size == 1


class TestStatsTracking:
    """Cache statistics tracking."""

    def test_initial_stats(self):
        """Initial stats should be all zeros."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Hit rate should be hits / total_requests."""
        stats = CacheStats(hits=3, misses=7)
        assert stats.hit_rate == 0.3
        assert stats.total_requests == 10

    def test_stats_to_dict(self):
        """Stats should serialize to a dictionary."""
        stats = CacheStats(hits=5, misses=5, evictions=2, stores=10)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["stores"] == 10
        assert d["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_stats_updated_on_operations(self):
        """Stats should be updated after cache operations."""
        cache = PromptCache(max_size=1)
        req1 = _make_request("First")
        req2 = _make_request("Second")

        await cache.get(req1)  # miss
        await cache.put(req1, _make_response("R1"))  # store
        await cache.get(req1)  # hit
        await cache.put(req2, _make_response("R2"))  # store + eviction

        assert cache.stats.misses == 1
        assert cache.stats.hits == 1
        assert cache.stats.stores == 2
        assert cache.stats.evictions == 1

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Clear should remove all entries and return count."""
        cache = PromptCache()
        for i in range(5):
            await cache.put(_make_request(f"Q{i}"), _make_response(f"R{i}"))

        assert cache.size == 5
        count = await cache.clear()
        assert count == 5
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_entries_metadata(self):
        """Entries metadata should include key, model, and preview."""
        cache = PromptCache()
        req = _make_request("Hello", system="You are helpful.")
        await cache.put(req, _make_response("Hi"))

        entries = await cache.get_entries_metadata()
        assert len(entries) == 1
        entry = entries[0]
        assert "key" in entry
        assert "model" in entry
        assert "prompt_preview" in entry
        assert "age_seconds" in entry
        assert "ttl_remaining" in entry
        assert entry["expired"] is False


# ===========================================================================
# PrefixCache Tests
# ===========================================================================


class TestPrefixCache:
    """PrefixCache registration, lookup, and analytics."""

    def test_register_new_prefix(self):
        """Registering a new prefix should return a hash and set hit_count=1."""
        pc = PrefixCache()
        h = pc.register("You are an analytical engine.")

        assert isinstance(h, str)
        assert len(h) == 64
        assert pc.total_prefixes == 1

        entry = pc.lookup("You are an analytical engine.")
        assert entry is not None
        assert entry.hit_count == 1

    def test_register_same_prefix_increments_count(self):
        """Registering the same prefix again should increment hit_count."""
        pc = PrefixCache()
        pc.register("System prompt A")
        pc.register("System prompt A")
        pc.register("System prompt A")

        entry = pc.lookup("System prompt A")
        assert entry.hit_count == 3

    def test_lookup_nonexistent(self):
        """Looking up a non-registered prefix should return None."""
        pc = PrefixCache()
        assert pc.lookup("nonexistent") is None

    def test_lookup_by_hash(self):
        """Should be able to look up by hash."""
        pc = PrefixCache()
        h = pc.register("Test prefix")
        entry = pc.lookup_by_hash(h)
        assert entry is not None
        assert entry.content == "Test prefix"

    def test_detect_prefix_reuse(self):
        """Detect reuse should return True only after second registration."""
        pc = PrefixCache()
        pc.register("Reusable prompt")
        assert not pc.detect_prefix_reuse("Reusable prompt")  # hit_count==1

        pc.register("Reusable prompt")
        assert pc.detect_prefix_reuse("Reusable prompt")  # hit_count==2

    def test_multiple_prefixes(self):
        """Multiple different prefixes should be tracked independently."""
        pc = PrefixCache()
        pc.register("Prefix A")
        pc.register("Prefix B")
        pc.register("Prefix A")

        assert pc.total_prefixes == 2
        assert pc.lookup("Prefix A").hit_count == 2
        assert pc.lookup("Prefix B").hit_count == 1

    def test_get_stats(self):
        """Stats should reflect total prefixes and reuses."""
        pc = PrefixCache()
        pc.register("A")
        pc.register("A")
        pc.register("B")

        stats = pc.get_stats()
        assert stats["total_prefixes"] == 2
        assert stats["total_reuses"] == 3
        assert stats["most_reused"]["hit_count"] == 2

    def test_get_all_entries(self):
        """Should return metadata for all entries."""
        pc = PrefixCache()
        pc.register("X")
        pc.register("Y")

        entries = pc.get_all_entries()
        assert len(entries) == 2

    def test_clear(self):
        """Clear should remove all prefixes and return count."""
        pc = PrefixCache()
        pc.register("A")
        pc.register("B")

        count = pc.clear()
        assert count == 2
        assert pc.total_prefixes == 0


# ===========================================================================
# RequestDeduplicator Tests
# ===========================================================================


class TestRequestDeduplicator:
    """In-flight request deduplication tests."""

    @pytest.mark.asyncio
    async def test_first_request_gets_none(self):
        """First request for a key should get None (meaning: execute it)."""
        dedup = RequestDeduplicator()
        req = _make_request("Hello")
        key, future = await dedup.acquire(req)

        assert future is None
        assert dedup.in_flight_count == 1

        # Complete it
        await dedup.complete(key, _make_response("Hi"))
        assert dedup.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_second_request_gets_future(self):
        """Second identical request should get a future to await."""
        dedup = RequestDeduplicator()
        req = _make_request("Hello")

        key1, future1 = await dedup.acquire(req)
        assert future1 is None

        key2, future2 = await dedup.acquire(req)
        assert future2 is not None
        assert key1 == key2
        assert dedup.dedup_count == 1

        # Complete the request
        expected_response = _make_response("Hi")
        await dedup.complete(key1, expected_response)

        # The second caller should now get the response
        result = await future2
        assert result.message.content == "Hi"

    @pytest.mark.asyncio
    async def test_concurrent_requests_dedup(self):
        """Multiple concurrent identical requests should all get the same result."""
        dedup = RequestDeduplicator()
        backend = CountingBackend()
        req = _make_request("Hello")

        # Simulate concurrency: first caller acquires, then two more arrive
        # before the first completes
        key1, future1 = await dedup.acquire(req)
        assert future1 is None  # first caller executes

        key2, future2 = await dedup.acquire(req)
        assert future2 is not None  # second caller waits

        key3, future3 = await dedup.acquire(req)
        assert future3 is not None  # third caller waits

        # First caller executes and completes
        response = await backend.chat(req)
        await dedup.complete(key1, response)

        # Only one backend call should have been made
        assert backend.call_count == 1

        # Both waiters should get the same result
        result2 = await future2
        result3 = await future3
        assert result2.message.content == "Response #1"
        assert result3.message.content == "Response #1"
        assert dedup.dedup_count == 2

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Exceptions should propagate to all waiters."""
        dedup = RequestDeduplicator()
        req = _make_request("Hello")

        key1, _ = await dedup.acquire(req)
        _, future2 = await dedup.acquire(req)

        # Fail the request
        await dedup.fail(key1, RuntimeError("Backend error"))

        with pytest.raises(RuntimeError, match="Backend error"):
            await future2

    @pytest.mark.asyncio
    async def test_cleanup_after_complete(self):
        """Completed requests should be removed from in-flight tracking."""
        dedup = RequestDeduplicator()
        req = _make_request("Hello")

        key, _ = await dedup.acquire(req)
        assert dedup.in_flight_count == 1

        await dedup.complete(key, _make_response("Hi"))
        assert dedup.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_after_fail(self):
        """Failed requests should be removed from in-flight tracking."""
        dedup = RequestDeduplicator()
        req = _make_request("Hello")

        key, _ = await dedup.acquire(req)
        assert dedup.in_flight_count == 1

        await dedup.fail(key, RuntimeError("error"))
        assert dedup.in_flight_count == 0

    def test_get_stats(self):
        """Stats should report dedup count and in-flight count."""
        dedup = RequestDeduplicator()
        stats = dedup.get_stats()
        assert stats["dedup_count"] == 0
        assert stats["in_flight_count"] == 0


# ===========================================================================
# Cache API Endpoint Tests
# ===========================================================================


class TestCacheAPIEndpoints:
    """Tests for /api/cache/* endpoints."""

    def test_get_stats(self, client):
        """GET /api/cache/stats should return cache statistics."""
        response = client.get("/api/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "prompt_cache" in data
        assert "prefix_cache" in data
        assert "deduplicator" in data
        assert "hits" in data["prompt_cache"]
        assert "misses" in data["prompt_cache"]
        assert "hit_rate" in data["prompt_cache"]

    def test_clear_cache(self, client):
        """POST /api/cache/clear should clear cache and return counts."""
        response = client.post("/api/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert "prompt_cache_cleared" in data
        assert "prefix_cache_cleared" in data

    def test_list_entries_empty(self, client):
        """GET /api/cache/entries should return empty lists initially."""
        response = client.get("/api/cache/entries")
        assert response.status_code == 200
        data = response.json()
        assert "prompt_cache" in data
        assert "prefix_cache" in data
        assert isinstance(data["prompt_cache"], list)
        assert isinstance(data["prefix_cache"], list)


# ===========================================================================
# Integration: Cache with Analytical Engine
# ===========================================================================


class TestCacheEngineIntegration:
    """Integration tests for the cache wired into the analytical engine."""

    @pytest.mark.asyncio
    async def test_engine_uses_cache_for_repeated_phase(self):
        """Engine should use cached responses for identical phase requests."""
        from augmentum.modes.analytical.engine import AnalyticalEngine

        backend = CountingBackend()
        cache = PromptCache()

        # Create two engines sharing the same cache
        engine1 = AnalyticalEngine(backend, prompt_cache=cache)
        engine2 = AnalyticalEngine(backend, prompt_cache=cache)

        # Run a simple phase on engine1
        from augmentum.modes.analytical.state import AnalyticalPhase

        await engine1._run_phase(
            AnalyticalPhase.ASSESS,
            model="test-model",
            query="What is 2+2?",
        )

        first_call_count = backend.call_count
        assert first_call_count == 1

        # Run the same phase on engine2 with same params — should hit cache
        await engine2._run_phase(
            AnalyticalPhase.ASSESS,
            model="test-model",
            query="What is 2+2?",
        )

        # Backend should NOT have been called again
        assert backend.call_count == first_call_count
        assert cache.stats.hits == 1
