"""System prompt prefix deduplication and analytics.

Tracks common system prompts by hash to detect prefix reuse across
requests. Useful for analytics and potential future prompt caching
optimizations at the model provider level.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PrefixEntry:
    """A registered system prompt prefix."""

    hash: str
    content: str
    registered_at: float
    hit_count: int = 0
    last_used_at: float = 0.0

    def to_dict(self) -> dict:
        now = time.monotonic()
        return {
            "hash": self.hash[:12],
            "preview": self.content[:80],
            "hit_count": self.hit_count,
            "age_seconds": round(now - self.registered_at, 1),
            "last_used_seconds_ago": (
                round(now - self.last_used_at, 1) if self.last_used_at else None
            ),
        }


class PrefixCache:
    """System prompt prefix deduplication tracker.

    Registers common system prompts and tracks how often they are reused.
    This is purely analytical -- it does not cache responses, but provides
    data on prefix reuse patterns.

    Uses an OrderedDict for LRU eviction when the cache exceeds
    ``max_entries`` (default from ``settings.prefix_cache_max_entries``).
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._prefixes: OrderedDict[str, PrefixEntry] = OrderedDict()
        self._max_entries = max_entries

    @property
    def max_entries(self) -> int:
        if self._max_entries is not None:
            return self._max_entries
        return settings.prefix_cache_max_entries

    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate a hash for the given content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def register(self, content: str) -> str:
        """Register a system prompt prefix and return its hash.

        If the prefix is already registered, increments its hit count.
        """
        h = self._hash_content(content)
        now = time.monotonic()

        if h in self._prefixes:
            entry = self._prefixes[h]
            entry.hit_count += 1
            entry.last_used_at = now
            self._prefixes.move_to_end(h)
            log.debug("prefix_reuse", hash=h[:12], hits=entry.hit_count)
        else:
            # Evict oldest entries if at capacity
            limit = self.max_entries
            while len(self._prefixes) >= limit:
                evicted_hash, _ = self._prefixes.popitem(last=False)
                log.debug("prefix_cache_evicted", hash=evicted_hash[:12])

            self._prefixes[h] = PrefixEntry(
                hash=h,
                content=content,
                registered_at=now,
                hit_count=1,
                last_used_at=now,
            )
            log.debug("prefix_register", hash=h[:12])

        return h

    def lookup(self, content: str) -> PrefixEntry | None:
        """Look up a prefix by content. Returns the entry or None."""
        h = self._hash_content(content)
        return self._prefixes.get(h)

    def lookup_by_hash(self, prefix_hash: str) -> PrefixEntry | None:
        """Look up a prefix by its hash. Returns the entry or None."""
        return self._prefixes.get(prefix_hash)

    def detect_prefix_reuse(self, content: str) -> bool:
        """Check if a prefix has been seen before (hit_count > 1)."""
        entry = self.lookup(content)
        return entry is not None and entry.hit_count > 1

    @property
    def total_prefixes(self) -> int:
        """Return the number of unique prefixes registered."""
        return len(self._prefixes)

    @property
    def total_reuses(self) -> int:
        """Return total reuse count across all prefixes."""
        return sum(e.hit_count for e in self._prefixes.values())

    def get_stats(self) -> dict:
        """Return prefix cache statistics."""
        entries = list(self._prefixes.values())
        return {
            "total_prefixes": len(entries),
            "total_reuses": sum(e.hit_count for e in entries),
            "most_reused": max(
                (e.to_dict() for e in entries),
                key=lambda d: d["hit_count"],
                default=None,
            ),
        }

    def get_all_entries(self) -> list[dict]:
        """Return metadata for all registered prefixes."""
        return [entry.to_dict() for entry in self._prefixes.values()]

    def clear(self) -> int:
        """Clear all registered prefixes. Returns count removed."""
        count = len(self._prefixes)
        self._prefixes.clear()
        return count
