"""Tests for document RAG scoring pipeline."""

from __future__ import annotations

import pytest


class TestScoreGate:
    """Test three-tier confidence classification."""

    def test_high_confidence_reranker(self):
        from augmentum.documents.scoring import score_gate

        # Sigmoid-normalized: 0.8 = strongly relevant (raw ~1.4)
        results = [{"content": "test", "score": 0.8, "filename": "a.md"}]
        scored = score_gate(results, reranker_enabled=True)
        assert len(scored) == 1
        assert scored[0].tier == "high"

    def test_uncertain_reranker(self):
        from augmentum.documents.scoring import score_gate

        # Sigmoid-normalized: 0.5 = neutral (raw ~0.0)
        results = [{"content": "test", "score": 0.5, "filename": "a.md"}]
        scored = score_gate(results, reranker_enabled=True)
        assert scored[0].tier == "uncertain"

    def test_irrelevant_reranker(self):
        from augmentum.documents.scoring import score_gate

        # Sigmoid-normalized: 0.2 = irrelevant (raw ~-1.4)
        results = [{"content": "test", "score": 0.2, "filename": "a.md"}]
        scored = score_gate(results, reranker_enabled=True)
        assert scored[0].tier == "irrelevant"

    def test_dual_source_rrf_high(self):
        from augmentum.documents.scoring import score_gate

        results = [{"content": "test", "score": 0.033, "filename": "a.md"}]
        scored = score_gate(results, reranker_enabled=False, dual_source=True)
        assert scored[0].tier == "high"

    def test_single_source_rrf_high(self):
        from augmentum.documents.scoring import score_gate

        results = [{"content": "test", "score": 0.016, "filename": "a.md"}]
        scored = score_gate(results, reranker_enabled=False, dual_source=False)
        assert scored[0].tier == "high"

    def test_single_source_rrf_uncertain(self):
        from augmentum.documents.scoring import score_gate

        results = [{"content": "test", "score": 0.01, "filename": "a.md"}]
        scored = score_gate(results, reranker_enabled=False, dual_source=False)
        assert scored[0].tier == "uncertain"

    def test_empty_results(self):
        from augmentum.documents.scoring import score_gate

        assert score_gate([], reranker_enabled=True) == []


class TestCliffDetect:
    """Test adaptive top-K with score drop-off."""

    def test_drops_below_cliff(self):
        from augmentum.documents.scoring import cliff_detect, score_gate

        results = [
            {"content": "a", "score": 0.9, "filename": "a.md"},
            {"content": "b", "score": 0.5, "filename": "a.md"},
            {"content": "c", "score": 0.1, "filename": "a.md"},
        ]
        scored = score_gate(results, reranker_enabled=True)
        clipped = cliff_detect(scored, cliff_ratio=0.3, max_results=10)
        assert len(clipped) == 2

    def test_respects_max_results(self):
        from augmentum.documents.scoring import cliff_detect, score_gate

        results = [{"content": f"c{i}", "score": 0.9 - i * 0.05, "filename": "a.md"} for i in range(10)]
        scored = score_gate(results, reranker_enabled=True)
        clipped = cliff_detect(scored, cliff_ratio=0.1, max_results=3)
        assert len(clipped) == 3

    def test_unsorted_input_gets_sorted(self):
        from augmentum.documents.scoring import cliff_detect, score_gate

        results = [
            {"content": "low", "score": 0.1, "filename": "a.md"},
            {"content": "high", "score": 0.9, "filename": "a.md"},
        ]
        scored = score_gate(results, reranker_enabled=True)
        clipped = cliff_detect(scored, cliff_ratio=0.3, max_results=10)
        assert clipped[0].chunk["content"] == "high"

    def test_empty(self):
        from augmentum.documents.scoring import cliff_detect

        assert cliff_detect([], cliff_ratio=0.3, max_results=10) == []


class TestApplyBudget:
    """Test token-aware context budget."""

    def test_fits_within_budget(self):
        from augmentum.documents.scoring import ScoredChunk, apply_budget

        chunks = [ScoredChunk(chunk={"content": "x" * 400}, tier="high", score=0.9)]
        result = apply_budget(chunks, max_tokens=200)
        assert len(result) == 1

    def test_exceeds_budget_truncates(self):
        from augmentum.documents.scoring import ScoredChunk, apply_budget

        chunks = [
            ScoredChunk(chunk={"content": "First sentence. Second sentence. Third sentence."}, tier="high", score=0.9),
        ]
        result = apply_budget(chunks, max_tokens=5)
        assert len(result) == 0

    def test_budget_packs_multiple(self):
        from augmentum.documents.scoring import ScoredChunk, apply_budget

        # 3 chunks of 200 chars each = 600 total. Budget = 150 tokens = 600 chars (exact fit).
        chunks = [
            ScoredChunk(chunk={"content": "a" * 200}, tier="high", score=0.9),
            ScoredChunk(chunk={"content": "b" * 200}, tier="high", score=0.8),
            ScoredChunk(chunk={"content": "c" * 200}, tier="high", score=0.7),
        ]
        result = apply_budget(chunks, max_tokens=150)
        assert len(result) == 3


class TestDetermineSufficiency:
    """Test sufficiency determination."""

    def test_sufficient_with_high(self):
        from augmentum.documents.scoring import ScoredChunk, determine_sufficiency

        chunks = [ScoredChunk(chunk={}, tier="high", score=0.9)]
        assert determine_sufficiency(chunks) == "sufficient"

    def test_partial_with_uncertain_only(self):
        from augmentum.documents.scoring import ScoredChunk, determine_sufficiency

        chunks = [ScoredChunk(chunk={}, tier="uncertain", score=0.3)]
        assert determine_sufficiency(chunks) == "partial"

    def test_none_with_empty(self):
        from augmentum.documents.scoring import determine_sufficiency

        assert determine_sufficiency([]) == "none"


class TestDeduplicate:
    """Test 4-gram character overlap deduplication."""

    def test_no_duplicates_passes_all(self):
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import ScoredChunk

        chunks = [
            ScoredChunk(chunk={"content": "The quick brown fox"}, tier="high", score=0.9),
            ScoredChunk(chunk={"content": "Completely different text"}, tier="high", score=0.8),
        ]
        result = deduplicate(chunks)
        assert len(result) == 2

    def test_near_duplicate_removed(self):
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import ScoredChunk

        text = "A" * 1500
        chunks = [
            ScoredChunk(chunk={"content": text}, tier="high", score=0.9),
            ScoredChunk(chunk={"content": text}, tier="high", score=0.8),
        ]
        result = deduplicate(chunks)
        assert len(result) == 1

    def test_partial_overlap_kept(self):
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import ScoredChunk

        shared = "S" * 200
        chunks = [
            ScoredChunk(chunk={"content": "A" * 1300 + shared}, tier="high", score=0.9),
            ScoredChunk(chunk={"content": shared + "B" * 1300}, tier="high", score=0.8),
        ]
        result = deduplicate(chunks)
        assert len(result) == 2

    def test_single_chunk_passes(self):
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import ScoredChunk

        chunks = [ScoredChunk(chunk={"content": "only one"}, tier="high", score=0.9)]
        result = deduplicate(chunks)
        assert len(result) == 1

    def test_empty_passes(self):
        from augmentum.documents.dedup import deduplicate

        assert deduplicate([]) == []


class TestWordNgrams:
    """Test word n-gram generation."""

    def test_basic_word_ngrams(self):
        from augmentum.documents.dedup import _word_ngrams

        result = _word_ngrams("the quick brown fox jumps", n=3)
        assert ("the", "quick", "brown") in result
        assert ("quick", "brown", "fox") in result
        assert ("brown", "fox", "jumps") in result
        assert len(result) == 3

    def test_short_text(self):
        from augmentum.documents.dedup import _word_ngrams

        result = _word_ngrams("hello world", n=3)
        assert result == {("hello", "world")}

    def test_numbers_preserved(self):
        from augmentum.documents.dedup import _word_ngrams

        result = _word_ngrams("Module-025 costs $625,000 annually", n=3)
        assert ("module-025", "costs", "$625,000") in result


class TestTokenizeFtsQuery:
    """Test AND-first/OR-fallback FTS5 tokenization."""

    def test_strips_stop_words(self):
        from augmentum.documents.store import _tokenize_fts_query

        result = _tokenize_fts_query("what are the financial deadlines")
        # "what", "are", "the" stripped -> 2 content words -> OR
        assert isinstance(result, str)
        assert '"financial"' in result
        assert '"deadlines"' in result
        assert '"what"' not in result

    def test_and_first_with_3_plus_words(self):
        from augmentum.documents.store import _tokenize_fts_query

        result = _tokenize_fts_query("financial payment deadline terms")
        assert isinstance(result, tuple)
        and_expr, or_expr = result
        assert "AND" in and_expr
        assert "OR" in or_expr

    def test_or_only_with_few_words(self):
        from augmentum.documents.store import _tokenize_fts_query

        result = _tokenize_fts_query("the financial deadlines")
        assert isinstance(result, str)
        assert "OR" in result

    def test_all_stop_words_fallback(self):
        from augmentum.documents.store import _tokenize_fts_query

        result = _tokenize_fts_query("what is the")
        assert isinstance(result, str)
        assert '"' in result

    def test_single_word(self):
        from augmentum.documents.store import _tokenize_fts_query

        result = _tokenize_fts_query("pagination")
        assert isinstance(result, str)
        assert '"pagination"' in result


class TestBuildDocumentContext:
    """Test structured injection format."""

    def test_sufficient_with_high_chunks(self):
        from augmentum.memory.integration import _build_document_context
        from augmentum.documents.scoring import ScoredChunk

        chunks = [ScoredChunk(
            chunk={"content": "Test content", "filename": "a.pdf", "page_num": 3},
            tier="high", score=0.9,
        )]
        result = _build_document_context(chunks, "sufficient")
        assert "<reference_material>" in result
        assert "(relevance: high)" in result
        assert "a.pdf p.3" in result
        assert "Ground your response" in result

    def test_partial_sufficiency_message(self):
        from augmentum.memory.integration import _build_document_context
        from augmentum.documents.scoring import ScoredChunk

        chunks = [ScoredChunk(
            chunk={"content": "Test", "filename": "b.md"},
            tier="uncertain", score=0.3,
        )]
        result = _build_document_context(chunks, "partial")
        assert "may not fully address" in result
        assert "(relevance: moderate)" in result

    def test_empty_chunks_none_returns_empty(self):
        from augmentum.memory.integration import _build_document_context

        assert _build_document_context([], "none") == ""

    def test_no_page_num(self):
        from augmentum.memory.integration import _build_document_context
        from augmentum.documents.scoring import ScoredChunk

        chunks = [ScoredChunk(
            chunk={"content": "Test", "filename": "c.md"},
            tier="high", score=0.9,
        )]
        result = _build_document_context(chunks, "sufficient")
        assert "c.md]" in result
        assert "p." not in result
