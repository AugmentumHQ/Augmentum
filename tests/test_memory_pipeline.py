"""Tests for memory pipeline — consolidator, compactor, core profile."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.memory.consolidator import (
    CONSOLIDATION_HIGH,
    CONSOLIDATION_LOW,
    _parse_merge_response,
    try_consolidate,
)
from augmentum.memory.compactor import MemoryCompactor, _cosine_similarity, _parse_summary_response
from augmentum.memory.core_profile import CoreProfileManager, _recency_weight
from augmentum.memory.models import Memory, MemoryTier, MemoryType


class TestConsolidator:
    """Consolidation-on-write — LLM merge of related memories."""

    def test_parse_merge_response_valid(self):
        raw = '{"merged": "User is a Python developer who likes AI", "importance": 0.85}'
        result = _parse_merge_response(raw)
        assert result is not None
        text, importance = result
        assert "Python" in text
        assert importance == pytest.approx(0.85)

    def test_parse_merge_response_code_fence(self):
        raw = '```json\n{"merged": "Combined fact", "importance": 0.9}\n```'
        result = _parse_merge_response(raw)
        assert result is not None
        assert result[0] == "Combined fact"

    def test_parse_merge_response_invalid_json(self):
        assert _parse_merge_response("not json at all") is None

    def test_parse_merge_response_too_short(self):
        raw = '{"merged": "Hi", "importance": 0.5}'
        assert _parse_merge_response(raw) is None

    def test_parse_merge_response_clamps_importance(self):
        raw = '{"merged": "A valid merged memory statement", "importance": 1.5}'
        result = _parse_merge_response(raw)
        assert result is not None
        assert result[1] == 1.0

    async def test_try_consolidate_no_candidates(self):
        result = await try_consolidate("new fact", [], None, None)
        assert result is None

    async def test_try_consolidate_no_backend(self):
        mem = Memory(id="m1", user_id="u", content="old fact", memory_type=MemoryType.FACT)
        result = await try_consolidate("new fact", [(mem, 0.7)], None, None)
        assert result is None

    async def test_try_consolidate_out_of_range(self):
        mem = Memory(id="m1", user_id="u", content="old fact", memory_type=MemoryType.FACT)
        backend = MagicMock()
        # Similarity too high (above CONSOLIDATION_HIGH)
        result = await try_consolidate("new fact", [(mem, 0.9)], backend, "model")
        assert result is None

    def test_consolidation_range_constants(self):
        assert CONSOLIDATION_LOW < CONSOLIDATION_HIGH
        assert CONSOLIDATION_LOW >= 0.0
        assert CONSOLIDATION_HIGH <= 1.0


class TestCompactor:
    """Background memory compaction."""

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 0.5]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_parse_summary_response_valid(self):
        raw = '{"summary": "User works in AI and likes Python", "importance": 0.8}'
        result = _parse_summary_response(raw)
        assert result is not None
        assert "AI" in result[0]

    def test_parse_summary_response_accepts_merged_key(self):
        raw = '{"merged": "A good merged fact statement", "importance": 0.7}'
        result = _parse_summary_response(raw)
        assert result is not None

    def test_parse_summary_response_invalid(self):
        assert _parse_summary_response("garbage") is None

    async def test_compactor_compact_empty(self):
        mock_store = MagicMock()
        mock_store.get_compaction_candidates = AsyncMock(return_value=[])
        mock_store.list_all = AsyncMock(return_value=[])

        compactor = MemoryCompactor(store=mock_store)
        stats = await compactor.compact(user_id="default")
        assert stats["deleted"] == 0
        assert stats["archived"] == 0


class TestCoreProfile:
    """Core profile manager — always-in-context user summary."""

    def test_recency_weight_recent(self):
        """Recent memories should have weight close to 1.0."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        weight = _recency_weight(now)
        assert weight > 0.9

    def test_recency_weight_none(self):
        """None updated_at should return default 0.5."""
        assert _recency_weight(None) == 0.5

    def test_recency_weight_has_floor(self):
        """Very old memories should still have weight >= 0.3."""
        weight = _recency_weight("2020-01-01T00:00:00+00:00")
        assert weight >= 0.3

    def test_mark_stale(self):
        mock_store = MagicMock()
        mgr = CoreProfileManager(store=mock_store)
        mgr.mark_stale("default")
        assert "default" in mgr._stale

    def test_notify_extraction_triggers_rebuild(self):
        mock_store = MagicMock()
        mgr = CoreProfileManager(store=mock_store, rebuild_interval=3)
        for _ in range(3):
            mgr.notify_extraction("default")
        assert "default" in mgr._stale

    def test_invalidate_clears_cache(self):
        mock_store = MagicMock()
        mgr = CoreProfileManager(store=mock_store)
        mgr._cache["default"] = "old profile"
        mgr.invalidate("default")
        assert "default" not in mgr._cache
