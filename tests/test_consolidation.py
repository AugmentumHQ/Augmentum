"""Tests for memory consolidation-on-write."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.memory.consolidator import (
    CONSOLIDATION_HIGH,
    CONSOLIDATION_LOW,
    _parse_merge_response,
    try_consolidate,
)
from augmentum.memory.models import Memory, MemoryType


def _make_memory(content: str, importance: float = 0.5) -> Memory:
    return Memory(
        id=f"mem_{hash(content) % 10000}",
        user_id="default",
        content=content,
        memory_type=MemoryType.FACT,
        importance=importance,
    )


# ===========================================================================
# _parse_merge_response
# ===========================================================================


class TestParseMergeResponse:
    def test_valid_json(self):
        raw = json.dumps({"merged": "User is a Python developer who uses Flask", "importance": 0.85})
        result = _parse_merge_response(raw)
        assert result is not None
        merged, importance = result
        assert merged == "User is a Python developer who uses Flask"
        assert importance == 0.85

    def test_markdown_wrapped(self):
        raw = '```json\n{"merged": "Combined fact", "importance": 0.7}\n```'
        result = _parse_merge_response(raw)
        assert result is not None
        assert result[0] == "Combined fact"

    def test_invalid_json(self):
        assert _parse_merge_response("not json") is None

    def test_short_merged(self):
        raw = json.dumps({"merged": "hi"})
        assert _parse_merge_response(raw) is None

    def test_missing_importance_defaults(self):
        raw = json.dumps({"merged": "Valid merged content"})
        result = _parse_merge_response(raw)
        assert result is not None
        assert result[1] == 0.7

    def test_importance_clamped(self):
        raw = json.dumps({"merged": "Clamped importance", "importance": 5.0})
        result = _parse_merge_response(raw)
        assert result[1] == 1.0


# ===========================================================================
# try_consolidate
# ===========================================================================


class TestTryConsolidate:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_none(self):
        result = await try_consolidate("new fact", [], MagicMock(), "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_backend_returns_none(self):
        candidates = [(_make_memory("old fact"), 0.65)]
        result = await try_consolidate("new fact", candidates, None, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_similarity_below_range(self):
        candidates = [(_make_memory("old fact"), 0.50)]  # Below CONSOLIDATION_LOW
        result = await try_consolidate("new fact", candidates, MagicMock(), "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_similarity_above_range(self):
        candidates = [(_make_memory("old fact"), 0.80)]  # Above CONSOLIDATION_HIGH
        result = await try_consolidate("new fact", candidates, MagicMock(), "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_consolidation(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = json.dumps({
            "merged": "User is a Python developer who works with Flask",
            "importance": 0.85,
        })
        backend.chat = AsyncMock(return_value=response)

        candidates = [(_make_memory("User likes Python"), 0.70)]
        result = await try_consolidate(
            "User works with Flask", candidates, backend, "model",
        )
        assert result is not None
        merged, importance = result
        assert "Python" in merged
        assert "Flask" in merged
        assert importance == 0.85

    @pytest.mark.asyncio
    async def test_backend_error_returns_none(self):
        backend = MagicMock()
        backend.chat = AsyncMock(side_effect=RuntimeError("error"))

        candidates = [(_make_memory("old fact"), 0.70)]
        result = await try_consolidate("new fact", candidates, backend, "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_picks_most_similar_candidate(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = json.dumps({
            "merged": "Best match merged", "importance": 0.8,
        })
        backend.chat = AsyncMock(return_value=response)

        candidates = [
            (_make_memory("less similar"), 0.62),
            (_make_memory("most similar"), 0.75),
            (_make_memory("also similar"), 0.68),
        ]
        result = await try_consolidate("new fact", candidates, backend, "model")
        assert result is not None

    def test_threshold_constants(self):
        assert CONSOLIDATION_LOW == 0.60
        assert CONSOLIDATION_HIGH == 0.78
        assert CONSOLIDATION_LOW < CONSOLIDATION_HIGH

    def test_config_defaults(self):
        from augmentum.config import Settings

        s = Settings()
        assert s.memory_consolidation_enabled is True

    @pytest.mark.asyncio
    async def test_brace_escaping(self):
        """Content with braces should not cause .format() errors."""
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"merged": "Merged result", "importance": 0.7}'
        backend.chat = AsyncMock(return_value=response)

        candidates = [(_make_memory("function() { return {}; }"), 0.70)]
        result = await try_consolidate(
            "const x = {key: value};", candidates, backend, "model",
        )
        # Should not raise
        assert result is not None
