"""Tests for cache/prefix_cache.py — system prompt prefix tracking."""

from __future__ import annotations

from augmentum.cache.prefix_cache import PrefixCache, PrefixEntry


class TestPrefixEntry:
    def test_construct(self):
        entry = PrefixEntry(hash="abc123", content="test", registered_at=100.0)
        assert entry.hit_count == 0
        assert entry.last_used_at == 0.0

    def test_to_dict(self):
        entry = PrefixEntry(hash="abc123def456", content="test prompt", registered_at=100.0, hit_count=5)
        d = entry.to_dict()
        assert d["hash"] == "abc123def456"[:12]
        assert d["preview"] == "test prompt"
        assert d["hit_count"] == 5


class TestPrefixCache:
    def test_construct(self):
        cache = PrefixCache(max_entries=10)
        assert cache.total_prefixes == 0
        assert cache.total_reuses == 0

    def test_register_new_prefix(self):
        cache = PrefixCache(max_entries=10)
        h = cache.register("You are a helpful assistant.")
        assert h
        assert cache.total_prefixes == 1

    def test_register_same_prefix_increments_hit_count(self):
        cache = PrefixCache(max_entries=10)
        h1 = cache.register("system prompt")
        h2 = cache.register("system prompt")
        assert h1 == h2
        assert cache.total_prefixes == 1
        entry = cache.lookup("system prompt")
        assert entry.hit_count == 2

    def test_register_different_prefixes(self):
        cache = PrefixCache(max_entries=10)
        cache.register("prompt A")
        cache.register("prompt B")
        assert cache.total_prefixes == 2

    def test_lookup_found(self):
        cache = PrefixCache(max_entries=10)
        cache.register("test content")
        entry = cache.lookup("test content")
        assert entry is not None
        assert entry.content == "test content"

    def test_lookup_not_found(self):
        cache = PrefixCache(max_entries=10)
        assert cache.lookup("nonexistent") is None

    def test_lookup_by_hash(self):
        cache = PrefixCache(max_entries=10)
        h = cache.register("test content")
        entry = cache.lookup_by_hash(h)
        assert entry is not None
        assert entry.hash == h

    def test_detect_prefix_reuse_false_on_first(self):
        cache = PrefixCache(max_entries=10)
        cache.register("once only")
        assert not cache.detect_prefix_reuse("once only")

    def test_detect_prefix_reuse_true_on_second(self):
        cache = PrefixCache(max_entries=10)
        cache.register("repeated")
        cache.register("repeated")
        assert cache.detect_prefix_reuse("repeated")

    def test_eviction_when_full(self):
        cache = PrefixCache(max_entries=3)
        cache.register("A")
        cache.register("B")
        cache.register("C")
        assert cache.total_prefixes == 3

        cache.register("D")
        assert cache.total_prefixes == 3
        assert cache.lookup("A") is None  # evicted (oldest)
        assert cache.lookup("D") is not None

    def test_get_stats(self):
        cache = PrefixCache(max_entries=10)
        cache.register("prompt")
        cache.register("prompt")
        stats = cache.get_stats()
        assert stats["total_prefixes"] == 1
        assert stats["total_reuses"] == 2
        assert stats["most_reused"]["hit_count"] == 2

    def test_get_stats_empty(self):
        cache = PrefixCache(max_entries=10)
        stats = cache.get_stats()
        assert stats["total_prefixes"] == 0
        assert stats["most_reused"] is None

    def test_total_reuses(self):
        cache = PrefixCache(max_entries=10)
        cache.register("A")
        cache.register("A")
        cache.register("B")
        assert cache.total_reuses == 3  # A:2 + B:1
