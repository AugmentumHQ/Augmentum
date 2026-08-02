"""Tests for background memory compaction."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.memory.compactor import MemoryCompactor, _cosine_similarity, _parse_summary_response
from augmentum.memory.models import Memory, MemoryTier, MemoryType


def _make_memory(
    content: str,
    importance: float = 0.5,
    access_count: int = 0,
    tier: str = MemoryTier.ACTIVE,
    updated_at: str | None = None,
) -> Memory:
    if updated_at is None:
        updated_at = datetime.now(UTC).isoformat()
    return Memory(
        id=f"mem_{hash(content) % 10000}",
        user_id="default",
        content=content,
        memory_type=MemoryType.FACT,
        importance=importance,
        access_count=access_count,
        tier=tier,
        updated_at=updated_at,
    )


# ===========================================================================
# _cosine_similarity
# ===========================================================================


class TestCosineSimilarity:
    def test_identical(self):
        a = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


# ===========================================================================
# _parse_summary_response
# ===========================================================================


class TestParseSummaryResponse:
    def test_valid(self):
        raw = json.dumps({"summary": "User is a developer who uses Python and Flask", "importance": 0.8})
        result = _parse_summary_response(raw)
        assert result is not None
        assert "developer" in result[0]
        assert result[1] == 0.8

    def test_invalid_json(self):
        assert _parse_summary_response("not json") is None

    def test_short_summary(self):
        assert _parse_summary_response(json.dumps({"summary": "hi"})) is None

    def test_markdown_wrapped(self):
        raw = '```json\n{"summary": "Valid summary text", "importance": 0.7}\n```'
        result = _parse_summary_response(raw)
        assert result is not None


# ===========================================================================
# MemoryCompactor
# ===========================================================================


class TestMemoryCompactor:
    @pytest.mark.asyncio
    async def test_compact_deletes_old_low_value(self):
        old_date = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        old_mem = _make_memory(
            "old low value", importance=0.1, access_count=0, updated_at=old_date,
        )

        store = MagicMock()
        store.get_compaction_candidates = AsyncMock(
            side_effect=[
                [old_mem],   # Phase 1: deletion candidates
                [],          # Phase 2: demotion candidates
            ],
        )
        store.forget = AsyncMock(return_value=True)

        compactor = MemoryCompactor(store)
        stats = await compactor.compact("default")
        assert stats["deleted"] == 1
        store.forget.assert_called_once_with(old_mem.id)

    @pytest.mark.asyncio
    async def test_compact_archives_medium_value(self):
        old_date = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        mem = _make_memory(
            "medium value", importance=0.4, access_count=3,
            tier=MemoryTier.ACTIVE, updated_at=old_date,
        )

        store = MagicMock()
        store.get_compaction_candidates = AsyncMock(
            side_effect=[
                [],     # Phase 1: no deletion candidates
                [mem],  # Phase 2: demotion candidates
            ],
        )
        store.update_tier = AsyncMock(return_value=True)

        compactor = MemoryCompactor(store)
        stats = await compactor.compact("default")
        assert stats["archived"] == 1
        store.update_tier.assert_called_once_with(mem.id, MemoryTier.ARCHIVE)

    @pytest.mark.asyncio
    async def test_compact_no_summarization_without_backend(self):
        store = MagicMock()
        store.get_compaction_candidates = AsyncMock(return_value=[])

        compactor = MemoryCompactor(store, backend=None)
        stats = await compactor.compact("default")
        assert stats["summarized"] == 0

    @pytest.mark.asyncio
    async def test_compact_empty_store(self):
        store = MagicMock()
        store.get_compaction_candidates = AsyncMock(return_value=[])

        compactor = MemoryCompactor(store)
        stats = await compactor.compact("default")
        assert stats == {"deleted": 0, "archived": 0, "summarized": 0}

    @pytest.mark.asyncio
    async def test_already_archived_not_double_demoted(self):
        mem = _make_memory(
            "already archived", importance=0.4, tier=MemoryTier.ARCHIVE,
        )

        store = MagicMock()
        store.get_compaction_candidates = AsyncMock(
            side_effect=[[], [mem]],
        )
        store.update_tier = AsyncMock(return_value=True)

        compactor = MemoryCompactor(store)
        stats = await compactor.compact("default")
        # Already archived — should not be demoted again
        assert stats["archived"] == 0
        store.update_tier.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        store = MagicMock()
        compactor = MemoryCompactor(store, interval_hours=24.0)
        compactor.start()
        assert compactor._task is not None
        await compactor.stop()
        assert compactor._task is None

    def test_config_defaults(self):
        from augmentum.config import Settings

        s = Settings()
        assert s.memory_compaction_enabled is True
        assert s.memory_compaction_interval_hours == 24.0
        assert s.memory_compaction_max_age_days == 30.0


# ===========================================================================
# Data model
# ===========================================================================


class TestMemoryTier:
    def test_tier_values(self):
        assert MemoryTier.CORE == "core"
        assert MemoryTier.ACTIVE == "active"
        assert MemoryTier.ARCHIVE == "archive"

    def test_memory_has_tier_field(self):
        mem = Memory(
            id="m1", user_id="default", content="test",
            memory_type=MemoryType.FACT, tier=MemoryTier.ARCHIVE,
        )
        assert mem.tier == MemoryTier.ARCHIVE

    def test_memory_tier_default(self):
        mem = Memory(id="m1", user_id="default", content="test", memory_type=MemoryType.FACT)
        assert mem.tier == MemoryTier.ACTIVE

    def test_memory_has_last_compacted_at(self):
        mem = Memory(
            id="m1", user_id="default", content="test",
            memory_type=MemoryType.FACT, last_compacted_at="2026-01-01T00:00:00",
        )
        assert mem.last_compacted_at == "2026-01-01T00:00:00"
