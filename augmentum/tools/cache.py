"""Tool result caching — skip re-execution for identical calls."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.base import ToolResult

log = get_logger(__name__)


class ToolResultCache:
    """In-memory cache keyed by (tool_name, frozen_params).

    Each entry has a TTL.  A TTL of ``0`` means the entry never expires
    (useful for deterministic tools like ``calculator``).

    Uses an OrderedDict for LRU eviction when the cache exceeds
    ``max_entries`` (default from ``settings.tool_cache_max_entries``).
    """

    def __init__(self, max_entries: int | None = None) -> None:
        self._store: OrderedDict[str, tuple[ToolResult, float]] = OrderedDict()
        self._max_entries = max_entries

    @property
    def max_entries(self) -> int:
        if self._max_entries is not None:
            return self._max_entries
        return settings.tool_cache_max_entries

    @staticmethod
    def _key(tool_name: str, params: dict) -> str:
        """Build a hashable cache key."""
        try:
            frozen = json.dumps(params, sort_keys=True, default=str)
        except (TypeError, ValueError):
            frozen = str(sorted(params.items()))
        return f"{tool_name}::{frozen}"

    def get(self, tool_name: str, params: dict, ttl: float) -> ToolResult | None:
        """Return cached result if still valid, else None."""
        key = self._key(tool_name, params)
        entry = self._store.get(key)
        if entry is None:
            return None

        result, stored_at = entry
        if ttl > 0 and (time.monotonic() - stored_at) > ttl:
            del self._store[key]
            return None

        # Move to end (most recently used)
        self._store.move_to_end(key)
        log.debug("tool_cache_hit", tool=tool_name)
        return result

    def put(self, tool_name: str, params: dict, result: ToolResult) -> None:
        """Store a result in the cache."""
        key = self._key(tool_name, params)
        self._store[key] = (result, time.monotonic())
        self._store.move_to_end(key)

        # Evict oldest entries if over capacity
        limit = self.max_entries
        while len(self._store) > limit:
            evicted_key, _ = self._store.popitem(last=False)
            log.debug("tool_cache_evicted", key=evicted_key[:60])
