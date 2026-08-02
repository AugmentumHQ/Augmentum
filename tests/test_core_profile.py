"""Tests for core memory profile manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.memory.core_profile import CoreProfileManager
from augmentum.memory.models import Memory, MemoryType


def _make_memory(
    content: str,
    importance: float = 0.5,
    access_count: int = 0,
) -> Memory:
    return Memory(
        id=f"mem_{hash(content) % 10000}",
        user_id="default",
        content=content,
        memory_type=MemoryType.FACT,
        importance=importance,
        access_count=access_count,
    )


class TestCoreProfileManager:
    @pytest.mark.asyncio
    async def test_empty_store_returns_empty(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[])
        mgr = CoreProfileManager(store, max_tokens=500)

        profile = await mgr.get_profile("default")
        assert profile == ""

    @pytest.mark.asyncio
    async def test_builds_profile_with_header(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("User is a Python developer", importance=0.9),
        ])
        mgr = CoreProfileManager(store, max_tokens=500)

        profile = await mgr.get_profile("default")
        assert "[user_context]" in profile
        assert "Python developer" in profile

    @pytest.mark.asyncio
    async def test_ranks_by_importance_and_access(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("Low importance", importance=0.1, access_count=0),
            _make_memory("High importance", importance=0.9, access_count=5),
            _make_memory("Medium with access", importance=0.5, access_count=10),
        ])
        mgr = CoreProfileManager(store, max_tokens=500)

        profile = await mgr.get_profile("default")
        content_lines = [l for l in profile.strip().split("\n") if l.startswith("- ")]
        # 3 memory entries
        assert len(content_lines) == 3
        # High importance * (1 + 5*0.1) = 0.9 * 1.5 = 1.35 — should be first
        assert "High importance" in content_lines[0]

    @pytest.mark.asyncio
    async def test_respects_token_budget(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("A" * 100, importance=0.9),
            _make_memory("B" * 100, importance=0.8),
            _make_memory("C" * 100, importance=0.7),
        ])
        # Small budget — should fit fewer than all 3 entries
        mgr = CoreProfileManager(store, max_tokens=80)

        profile = await mgr.get_profile("default")
        # With 320 char budget (80 tok × 4), header reserve 80, each fact ~104 chars
        # Should fit at most 2 of 3 facts
        content_lines = [l for l in profile.strip().split("\n") if l.startswith("- ")]
        assert len(content_lines) < 3

    @pytest.mark.asyncio
    async def test_caching(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("Cached fact", importance=0.9),
        ])
        mgr = CoreProfileManager(store, max_tokens=500)

        profile1 = await mgr.get_profile("default")
        profile2 = await mgr.get_profile("default")
        assert profile1 == profile2
        # list_all should only be called once (cached)
        assert store.list_all.call_count == 1

    @pytest.mark.asyncio
    async def test_mark_stale_triggers_rebuild(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("Original fact", importance=0.9),
        ])
        mgr = CoreProfileManager(store, max_tokens=500)

        await mgr.get_profile("default")
        assert store.list_all.call_count == 1

        mgr.mark_stale("default")
        await mgr.get_profile("default")
        assert store.list_all.call_count == 2

    @pytest.mark.asyncio
    async def test_notify_extraction_triggers_rebuild(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[])
        mgr = CoreProfileManager(store, max_tokens=500, rebuild_interval=3)

        await mgr.get_profile("default")
        assert store.list_all.call_count == 1

        # Notify below threshold — no rebuild
        mgr.notify_extraction("default")
        mgr.notify_extraction("default")
        await mgr.get_profile("default")
        assert store.list_all.call_count == 1

        # Notify at threshold — triggers rebuild
        mgr.notify_extraction("default")
        await mgr.get_profile("default")
        assert store.list_all.call_count == 2

    @pytest.mark.asyncio
    async def test_invalidate(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[])
        mgr = CoreProfileManager(store, max_tokens=500)

        await mgr.get_profile("default")
        mgr.invalidate("default")
        await mgr.get_profile("default")
        assert store.list_all.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_user_isolation(self):
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("Alice's fact", importance=0.9),
        ])
        mgr = CoreProfileManager(store, max_tokens=500)

        await mgr.get_profile("alice")
        await mgr.get_profile("bob")
        assert store.list_all.call_count == 2

    @pytest.mark.asyncio
    async def test_store_error_handled_gracefully(self):
        store = MagicMock()
        store.list_all = AsyncMock(side_effect=RuntimeError("db error"))
        mgr = CoreProfileManager(store, max_tokens=500)

        profile = await mgr.get_profile("default")
        # Should not raise, return empty
        assert profile == ""

    def test_config_defaults(self):
        from augmentum.config import Settings

        s = Settings()
        assert s.memory_core_profile_enabled is True
        assert s.memory_core_profile_max_tokens == 500
        assert s.memory_core_profile_rebuild_interval == 5

    # ── Cache-only read + per-user rebuild lock ───────────────────
    #
    # ROOT CAUSE:
    #
    # Post-restart the in-memory cache is empty. Two concurrent
    # /v1/memory/context-preview hits both ran get_profile, both
    # missed the cache, both called _rebuild end-to-end, and the
    # LLM-backed _synthesize_profile burned ~130 s of engine time
    # twice. Worse: the same engine was the slow-path target for
    # the game-agent, so its first inference queued behind both
    # rebuilds and the agent appeared frozen.
    #
    # We fixed this by (a) adding a per-user rebuild lock that
    # short-circuits a second caller after the first completes,
    # and (b) introducing get_profile_cached_only which never
    # awaits the LLM — the UI memory indicator now returns
    # cached/persisted/empty immediately and schedules a rebuild
    # in the background.

    @pytest.mark.asyncio
    async def test_get_profile_cached_only_returns_cache_without_list_all(self):
        """@example: post-warmup, the lightweight endpoint must NOT re-list memories."""
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[
            _make_memory("Cached fact", importance=0.9),
        ])
        mgr = CoreProfileManager(store, max_tokens=500)

        # Warm the cache via the normal path; this calls list_all in
        # both _has_memories and _rebuild, hence call_count > 0.
        await mgr.get_profile("default")
        baseline = store.list_all.call_count

        # Cache-only read must NOT touch the store again.
        profile = await mgr.get_profile_cached_only("default")
        assert "Cached fact" in profile
        assert store.list_all.call_count == baseline

    @pytest.mark.asyncio
    async def test_get_profile_cached_only_empty_schedules_rebuild_returns_empty(self):
        """@example: cold cache + no persisted profile -> empty string + background rebuild scheduled."""
        store = MagicMock()
        store.list_all = AsyncMock(return_value=[])
        mgr = CoreProfileManager(store, max_tokens=500)

        profile = await mgr.get_profile_cached_only("default")
        # No data available yet — returns empty without blocking.
        assert profile == ""
        # Background rebuild was scheduled (user marked stale so
        # the per-user lock + _rebuild task will eventually catch up).
        assert "default" in mgr._stale

    @pytest.mark.asyncio
    async def test_rebuild_lock_collapses_concurrent_callers(self):
        """@example: two parallel stale-path get_profile callers -> only one rebuild body runs.

        Forces a real suspension inside the rebuild body via a gated
        list_all so the second caller actually reaches the lock while
        the first holds it. Without the lock, both callers would run
        the LLM-backed _synthesize_profile end-to-end (the production
        symptom: two simultaneous /v1/memory/context-preview hits
        burning ~130 s each post-restart).
        """
        import asyncio as _asyncio

        list_all_started = _asyncio.Event()
        release_list_all = _asyncio.Event()
        call_count = 0

        async def _gated_list_all(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            list_all_started.set()
            await release_list_all.wait()
            return [_make_memory("Concurrent fact", importance=0.9)]

        store = MagicMock()
        store.list_all = _gated_list_all
        mgr = CoreProfileManager(store, max_tokens=500)
        # Stale path skips _has_memories so list_all only fires from
        # inside _rebuild_locked — the exact section the lock guards.
        mgr.mark_stale("default")

        # Caller A enters the rebuild and suspends inside list_all,
        # holding the per-user lock.
        task_a = _asyncio.create_task(mgr.get_profile("default"))
        await list_all_started.wait()

        # Caller B starts; it should queue on the lock rather than
        # entering the rebuild body itself. We mark stale again
        # because Caller A hasn't cleared it yet (still mid-rebuild),
        # but it's already set anyway. Tick the loop so B reaches
        # the lock-await before we release A.
        task_b = _asyncio.create_task(mgr.get_profile("default"))
        for _ in range(3):
            await _asyncio.sleep(0)

        # Release Caller A — it now finishes the rebuild, populates
        # the cache, and clears the stale flag.
        release_list_all.set()
        results = await _asyncio.gather(task_a, task_b)

        assert results[0] == results[1]
        assert "Concurrent fact" in results[0]
        # Caller B acquired the lock AFTER Caller A finished, saw
        # the freshly populated cache, and short-circuited without
        # re-running list_all or _synthesize_profile.
        assert call_count == 1
