"""Tests for document query analysis, expansion, and span filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

from augmentum.documents.query_analyzer import (
    QueryAnalysis,
    QueryAnalyzer,
    _parse_analysis_response,
)
from augmentum.documents.query_expansion import (
    expand_query_terms,
)
from augmentum.documents.query_expansion import (
    reset as reset_expansion,
)
from augmentum.documents.span_filter import _cosine_similarity, _split_sentences


class TestQueryAnalyzer:
    """Query classification and rewriting."""

    def test_parse_response_direct(self):
        raw = '{"strategy": "direct", "queries": ["test query"], "reason": "specific terms", "confidence": 0.9}'
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "direct"
        assert result.queries == ["test query"]
        assert result.confidence == 0.9

    def test_parse_response_decompose(self):
        raw = '{"strategy": "decompose", "queries": ["q1", "q2"], "reason": "multi-topic"}'
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "decompose"
        assert len(result.queries) == 2

    def test_parse_response_skip(self):
        raw = '{"strategy": "skip", "queries": [], "reason": "greeting"}'
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "skip"
        assert result.queries == []

    def test_parse_response_invalid_json(self):
        result = _parse_analysis_response("not json", "fallback query")
        assert result.strategy == "direct"
        assert result.queries == ["fallback query"]

    def test_parse_response_code_fence(self):
        raw = '```json\n{"strategy": "rewrite", "queries": ["rewritten query"]}\n```'
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "rewrite"

    def test_parse_response_invalid_strategy(self):
        raw = '{"strategy": "invalid_strategy", "queries": ["q"]}'
        result = _parse_analysis_response(raw, "original")
        assert result.strategy == "direct"

    def test_parse_response_caps_queries_at_max(self):
        raw = '{"strategy": "decompose", "queries": ["a", "b", "c", "d", "e"]}'
        result = _parse_analysis_response(raw, "original")
        assert len(result.queries) <= 3

    def test_parse_response_empty_queries_non_skip(self):
        raw = '{"strategy": "direct", "queries": []}'
        result = _parse_analysis_response(raw, "my original")
        assert result.queries == ["my original"]

    async def test_analyzer_no_backend(self):
        analyzer = QueryAnalyzer(backend=None)
        result = await analyzer.analyze("test", doc_names=["doc.pdf"])
        assert result.strategy == "direct"
        assert result.queries == ["test"]

    async def test_analyzer_all_docs_full_mode(self):
        analyzer = QueryAnalyzer(backend=MagicMock())
        result = await analyzer.analyze("test", doc_names=[], has_full_docs=True)
        assert result.strategy == "skip"

    def test_query_analysis_defaults(self):
        qa = QueryAnalysis()
        assert qa.strategy == "direct"
        assert qa.queries == []
        assert qa.confidence == 1.0


class TestQueryExpansion:
    """Embedding-based query expansion."""

    def test_expand_not_ready(self):
        reset_expansion()
        result = expand_query_terms(["python", "code"])
        assert result == []

    def test_reset_clears_state(self):
        reset_expansion()
        from augmentum.documents.query_expansion import _vocab_ready
        assert _vocab_ready is False


class TestSpanFilter:
    """Sentence-level span filtering."""

    def test_split_sentences_basic(self):
        text = "First sentence is here. Second sentence follows. Third one too."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_short_filtered(self):
        text = "Hi. This is a longer sentence that passes the length filter easily."
        sentences = _split_sentences(text)
        # "Hi." is too short (< 20 chars), should be filtered
        for s in sentences:
            assert len(s) > 20

    def test_split_sentences_empty(self):
        assert _split_sentences("") == []

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.5]
        assert _cosine_similarity(v, v) > 0.99

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 0.01

    def test_cosine_similarity_zero(self):
        assert _cosine_similarity([0.0], [1.0]) == 0.0
