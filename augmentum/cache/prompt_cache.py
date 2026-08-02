"""LRU response cache for deterministic LLM requests.

Caches model responses keyed by a hash of (model, messages, temperature).
Only caches requests with temperature=0 or None (deterministic).
Supports TTL expiration, LRU eviction, hit/miss statistics, and
pattern-based cache invalidation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

from augmentum.models.base import InternalChatRequest, InternalChatResponse
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Default cache settings
DEFAULT_MAX_SIZE = 256
DEFAULT_TTL_SECONDS = 3600  # 1 hour


@dataclass
class CacheEntry:
    """A single cached response with metadata."""

    key: str
    response: InternalChatResponse
    created_at: float
    ttl: float
    model: str
    prompt_preview: str  # First 80 chars of prompt for debugging

    @property
    def is_expired(self) -> bool:
        """Check if this entry has passed its TTL."""
        return (time.monotonic() - self.created_at) > self.ttl


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    stores: int = 0
    skipped_non_deterministic: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "stores": self.stores,
            "skipped_non_deterministic": self.skipped_non_deterministic,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
        }


class PromptCache:
    """LRU response cache with TTL expiration for deterministic LLM requests.

    Thread-safe via asyncio.Lock. Only caches requests where temperature
    is 0 or None (considered deterministic).
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Return current number of entries in the cache."""
        return len(self._cache)

    @staticmethod
    def _is_deterministic(request: InternalChatRequest) -> bool:
        """Check if a request is deterministic (cacheable).

        Only requests with temperature=0 or temperature=None are cached.
        """
        return request.temperature is None or request.temperature == 0

    @staticmethod
    def make_key(request: InternalChatRequest) -> str:
        """Generate a deterministic cache key from the request.

        Hashes model + messages + temperature to create a stable key.
        """
        key_parts = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in request.messages
            ],
            "temperature": request.temperature,
        }
        serialized = json.dumps(key_parts, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def get(self, request: InternalChatRequest) -> InternalChatResponse | None:
        """Look up a cached response for the given request.

        Returns None on cache miss, non-deterministic request, or expired entry.
        """
        if not self._is_deterministic(request):
            self._stats.skipped_non_deterministic += 1
            return None

        key = self.make_key(request)

        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                log.debug("cache_expired", key=key[:12])
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1
            log.debug("cache_hit", key=key[:12], model=entry.model)
            return entry.response

    async def put(
        self,
        request: InternalChatRequest,
        response: InternalChatResponse,
    ) -> None:
        """Store a response in the cache.

        Skips non-deterministic requests. Evicts the least recently used
        entry if the cache is at capacity.
        """
        if not self._is_deterministic(request):
            return

        key = self.make_key(request)
        prompt_preview = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt_preview = msg.content[:80]
                break

        async with self._lock:
            # If key already exists, remove it first (will be re-added at end)
            if key in self._cache:
                del self._cache[key]

            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats.evictions += 1
                log.debug("cache_evict", key=evicted_key[:12])

            self._cache[key] = CacheEntry(
                key=key,
                response=response,
                created_at=time.monotonic(),
                ttl=self._ttl,
                model=request.model,
                prompt_preview=prompt_preview,
            )
            self._stats.stores += 1

    async def invalidate_pattern(self, pattern: str) -> int:
        """Remove all cache entries whose prompt_preview matches the pattern.

        Returns the number of entries removed.
        """
        regex = re.compile(pattern, re.IGNORECASE)
        removed = 0

        async with self._lock:
            keys_to_remove = [
                key
                for key, entry in self._cache.items()
                if regex.search(entry.prompt_preview)
            ]
            for key in keys_to_remove:
                del self._cache[key]
                removed += 1

        if removed:
            log.info("cache_invalidate_pattern", pattern=pattern, removed=removed)
        return removed

    async def clear(self) -> int:
        """Remove all entries from the cache. Returns the count removed."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            log.info("cache_cleared", entries=count)
            return count

    async def get_entries_metadata(self) -> list[dict]:
        """Return metadata for all cache entries (no response bodies)."""
        async with self._lock:
            entries = []
            now = time.monotonic()
            for key, entry in self._cache.items():
                entries.append({
                    "key": key[:12],
                    "model": entry.model,
                    "prompt_preview": entry.prompt_preview,
                    "age_seconds": round(now - entry.created_at, 1),
                    "ttl_remaining": round(
                        max(0, entry.ttl - (now - entry.created_at)), 1
                    ),
                    "expired": entry.is_expired,
                })
            return entries
