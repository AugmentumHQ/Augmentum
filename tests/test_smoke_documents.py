"""Smoke tests — verify every module under augmentum/documents/ can be imported."""

from __future__ import annotations


class TestDocumentImports:
    """Import every module in the documents package."""

    def test_import_chunker(self):
        from augmentum.documents import chunker  # noqa: F401

    def test_import_contextual(self):
        from augmentum.documents import contextual  # noqa: F401

    def test_import_store(self):
        from augmentum.documents import store  # noqa: F401

    def test_import_query_analyzer(self):
        from augmentum.documents import query_analyzer  # noqa: F401

    def test_import_query_expansion(self):
        from augmentum.documents import query_expansion  # noqa: F401

    def test_import_dedup(self):
        from augmentum.documents import dedup  # noqa: F401

    def test_import_scoring(self):
        from augmentum.documents import scoring  # noqa: F401

    def test_import_answer_density(self):
        from augmentum.documents import answer_density  # noqa: F401

    def test_import_span_filter(self):
        from augmentum.documents import span_filter  # noqa: F401

    def test_import_topic_coverage(self):
        from augmentum.documents import topic_coverage  # noqa: F401

    def test_chunk_dataclass(self):
        from augmentum.documents.chunker import Chunk

        c = Chunk(index=0, text="hello world")
        assert c.index == 0
        assert c.text == "hello world"
        assert c.parent_index is None

    def test_scored_chunk_dataclass(self):
        from augmentum.documents.scoring import ScoredChunk

        sc = ScoredChunk(chunk={"content": "test"}, tier="high", score=0.9)
        assert sc.tier == "high"
        assert sc.score == 0.9

    def test_query_analysis_dataclass(self):
        from augmentum.documents.query_analyzer import QueryAnalysis

        qa = QueryAnalysis(strategy="direct", queries=["test"])
        assert qa.strategy == "direct"
        assert qa.confidence == 1.0
