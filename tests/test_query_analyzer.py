"""Tests for query analyzer — parsing, validation, fallback, caching."""

from __future__ import annotations

import json
import time

import pytest


class TestParseAnalysisResponse:
    """Test LLM response parsing resilience."""

    def test_valid_json(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = '{"strategy": "rewrite", "queries": ["financial costs"], "reason": "vague", "confidence": 0.9}'
        result = _parse_analysis_response(raw, "original query")
        assert result.strategy == "rewrite"
        assert result.queries == ["financial costs"]
        assert result.confidence == 0.9

    def test_json_in_code_fence(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = '```json\n{"strategy": "direct", "queries": ["test"], "reason": "ok", "confidence": 0.8}\n```'
        result = _parse_analysis_response(raw, "test")
        assert result.strategy == "direct"
        assert result.queries == ["test"]

    def test_malformed_json_falls_back(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = "This is not JSON at all"
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "direct"
        assert result.queries == ["original"]

    def test_invalid_strategy_falls_back(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = '{"strategy": "banana", "queries": ["test"], "reason": "ok"}'
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "direct"

    def test_empty_queries_for_non_skip(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = '{"strategy": "direct", "queries": [], "reason": "ok"}'
        result = _parse_analysis_response(raw, "original")
        assert result.queries == ["original"]

    def test_skip_allows_empty_queries(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = '{"strategy": "skip", "queries": [], "reason": "greeting"}'
        result = _parse_analysis_response(raw, "thanks")
        assert result.strategy == "skip"
        assert result.queries == []

    def test_queries_capped_at_3(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        raw = json.dumps({"strategy": "decompose", "queries": ["a", "b", "c", "d", "e"], "reason": "many"})
        result = _parse_analysis_response(raw, "original")
        assert len(result.queries) == 3

    def test_query_string_capped_at_200(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        long_query = "x" * 500
        raw = json.dumps({"strategy": "direct", "queries": [long_query], "reason": "long"})
        result = _parse_analysis_response(raw, "original")
        assert len(result.queries[0]) == 200

    def test_none_input(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        result = _parse_analysis_response(None, "original")
        assert result.strategy == "direct"
        assert result.queries == ["original"]

    def test_empty_string(self):
        from augmentum.documents.query_analyzer import _parse_analysis_response

        result = _parse_analysis_response("", "original")
        assert result.strategy == "direct"


class TestFullModeShortCircuit:
    """Test inject mode awareness."""

    @pytest.mark.asyncio
    async def test_all_full_mode_skips(self):
        from augmentum.documents.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer(backend=None)
        result = await analyzer.analyze("test query", doc_names=[], has_full_docs=True)
        assert result.strategy == "skip"
        assert result.reason == "all_docs_full_mode"

    @pytest.mark.asyncio
    async def test_no_backend_falls_back_to_direct(self):
        from augmentum.documents.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer(backend=None)
        result = await analyzer.analyze("test query", doc_names=["file.pdf"])
        assert result.strategy == "direct"
        assert result.queries == ["test query"]


class TestAnalysisCache:
    """Test cache hit/miss behavior."""

    def test_cache_key_deterministic(self):
        from augmentum.documents.query_analyzer import _cache_key

        k1 = _cache_key("test", ["a.pdf"])
        k2 = _cache_key("test", ["a.pdf"])
        assert k1 == k2

    def test_cache_key_includes_docs(self):
        from augmentum.documents.query_analyzer import _cache_key

        k1 = _cache_key("test", ["a.pdf"])
        k2 = _cache_key("test", ["b.pdf"])
        assert k1 != k2

    def test_cache_key_sorted_docs(self):
        from augmentum.documents.query_analyzer import _cache_key

        k1 = _cache_key("test", ["a.pdf", "b.pdf"])
        k2 = _cache_key("test", ["b.pdf", "a.pdf"])
        assert k1 == k2  # sorted, so order doesn't matter

    def test_cache_stores_and_retrieves(self):
        from augmentum.documents.query_analyzer import (
            QueryAnalysis,
            _analysis_cache,
            _cache_key,
        )

        _analysis_cache.clear()
        key = _cache_key("test", ["a.pdf"])
        cached = QueryAnalysis(strategy="rewrite", queries=["rewritten"], reason="cached", confidence=0.9)
        _analysis_cache[key] = (cached, time.time())

        assert key in _analysis_cache
        stored, _ = _analysis_cache[key]
        assert stored.strategy == "rewrite"
