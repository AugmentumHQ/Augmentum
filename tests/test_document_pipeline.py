"""Tests for document pipeline — chunker, dedup, scoring, topic coverage."""

from __future__ import annotations

import pytest

from augmentum.documents.chunker import (
    _detect_sections,
    _split_section,
    chunk_text,
    chunk_with_parents,
    extract_text,
)
from augmentum.documents.dedup import _word_ngrams, deduplicate
from augmentum.documents.scoring import (
    ScoredChunk,
    apply_budget,
    cliff_detect,
    determine_sufficiency,
    query_quality,
    score_gate,
)
from augmentum.documents.topic_coverage import (
    _extract_topic_terms,
    build_topic_map,
    check_topic_coverage,
)
from augmentum.documents.topic_coverage import (
    reset as reset_topic_coverage,
)


class TestChunker:
    """Text chunking with section awareness."""

    def test_chunk_text_basic(self):
        pages = [("Hello world. This is a test document.", None)]
        chunks = chunk_text(pages, chunk_size=500)
        assert len(chunks) >= 1
        assert chunks[0].text.strip() != ""

    def test_chunk_text_empty(self):
        assert chunk_text([]) == []

    def test_chunk_text_splits_large_text(self):
        big_text = "Sentence number one. " * 200
        pages = [(big_text, None)]
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1

    def test_chunk_text_header_enrichment(self):
        pages = [("# Introduction\nSome content here about the topic.", None)]
        chunks = chunk_text(pages, filename="doc.pdf")
        assert len(chunks) >= 1
        # Enriched text should include filename
        assert any("doc.pdf" in c.enriched_text for c in chunks)

    def test_chunk_with_parents(self):
        text = "First paragraph.\n\n" * 20
        pages = [(text, None)]
        children, parents = chunk_with_parents(
            pages, child_size=100, parent_size=500,
        )
        assert len(parents) >= 1
        assert len(children) >= len(parents)

    def test_detect_sections_markdown(self):
        text = "# Header One\nContent under header one.\n# Header Two\nContent under header two."
        sections = _detect_sections(text)
        assert len(sections) >= 2

    def test_detect_sections_no_headings(self):
        text = "Just a plain block of text without any headings."
        sections = _detect_sections(text)
        assert len(sections) == 1

    def test_split_section_fits_in_one(self):
        result = _split_section("short text", 1000, 100)
        assert len(result) == 1

    def test_extract_text_plain(self):
        pages = extract_text(b"Hello world", "text/plain", "test.txt")
        assert len(pages) == 1
        assert "Hello" in pages[0][0]

    @staticmethod
    def _make_pptx() -> bytes:
        import io as _io

        pptx = pytest.importorskip("pptx")
        prs = pptx.Presentation()
        for title, body in (("Slide One", "First body"), ("Slide Two", "Second body")):
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = title
            slide.placeholders[1].text = body
        buf = _io.BytesIO()
        prs.save(buf)
        return buf.getvalue()

    def test_extract_text_pptx_by_filename(self):
        # The studio "Ask AI about this file" handoff uploads the artifact as
        # application/octet-stream — extraction must still route by filename and
        # NOT leak the zip's member-filename table into the model's context.
        data = self._make_pptx()
        pages = extract_text(data, "application/octet-stream", "deck.pptx")
        assert len(pages) == 2
        assert pages[0][1] == 1 and pages[1][1] == 2  # per-slide page numbers
        joined = "\n".join(t for t, _ in pages)
        assert "Slide One" in joined and "Second body" in joined
        assert "slide1.xml" not in joined  # regression: no zip plumbing leak

    def test_extract_text_pptx_coerced_text_plain(self):
        # document_routes may coerce an unknown .pptx mime to text/plain; the
        # filename-first dispatch must win over the text/* branch.
        data = self._make_pptx()
        pages = extract_text(data, "text/plain", "deck.pptx")
        assert len(pages) == 2

    def test_extract_text_binary_blob_not_leaked(self):
        # A zip/binary with no recognizable extension must yield NO text rather
        # than a garbage decode of the raw bytes.
        blob = b"PK\x03\x04" + b"ppt/slides/slide1.xml" + bytes(64)
        assert extract_text(blob, "application/octet-stream", "mystery") == []


class TestDedup:
    """Chunk deduplication via word n-gram overlap."""

    def test_word_ngrams_basic(self):
        ngrams = _word_ngrams("the quick brown fox jumps", n=3)
        assert len(ngrams) == 3  # 5 words, 3-grams = 3

    def test_word_ngrams_short_text(self):
        ngrams = _word_ngrams("hello world", n=3)
        # Less than n words -> single tuple
        assert len(ngrams) == 1

    def test_word_ngrams_empty(self):
        assert _word_ngrams("", n=3) == set()

    def test_deduplicate_no_duplicates(self):
        chunks = [
            ScoredChunk(chunk={"content": "Alpha beta gamma delta"}, tier="high", score=0.9),
            ScoredChunk(chunk={"content": "Completely different text here"}, tier="high", score=0.8),
        ]
        result = deduplicate(chunks)
        assert len(result) == 2

    def test_deduplicate_removes_duplicate(self):
        text = "The quick brown fox jumps over the lazy dog near the river"
        chunks = [
            ScoredChunk(chunk={"content": text}, tier="high", score=0.9),
            ScoredChunk(chunk={"content": text}, tier="high", score=0.8),
        ]
        result = deduplicate(chunks)
        assert len(result) == 1

    def test_deduplicate_single_chunk(self):
        chunks = [ScoredChunk(chunk={"content": "solo"}, tier="high", score=0.9)]
        assert deduplicate(chunks) == chunks

    def test_deduplicate_empty(self):
        assert deduplicate([]) == []


class TestScoring:
    """Score gate, cliff detection, budget management."""

    def test_score_gate_reranker_high(self):
        results = [{"score": 0.8}, {"score": 0.3}]
        scored = score_gate(results, reranker_enabled=True)
        assert scored[0].tier == "high"
        assert scored[1].tier == "irrelevant"

    def test_score_gate_rrf_dual(self):
        results = [{"score": 0.03}, {"score": 0.005}]
        scored = score_gate(results, reranker_enabled=False, dual_source=True)
        assert scored[0].tier == "high"

    def test_cliff_detect_drops_low_scores(self):
        scored = [
            ScoredChunk(chunk={}, tier="high", score=1.0),
            ScoredChunk(chunk={}, tier="uncertain", score=0.5),
            ScoredChunk(chunk={}, tier="uncertain", score=0.1),
        ]
        result = cliff_detect(scored, cliff_ratio=0.3)
        # 0.1 < 1.0 * 0.3 = 0.3, so should be dropped
        assert len(result) == 2

    def test_cliff_detect_empty(self):
        assert cliff_detect([]) == []

    def test_apply_budget_fits(self):
        chunks = [
            ScoredChunk(chunk={"content": "Short"}, tier="high", score=0.9),
        ]
        result = apply_budget(chunks, max_tokens=100)
        assert len(result) == 1

    def test_determine_sufficiency_high(self):
        chunks = [ScoredChunk(chunk={}, tier="high", score=0.9)]
        assert determine_sufficiency(chunks) == "sufficient"

    def test_determine_sufficiency_partial(self):
        chunks = [ScoredChunk(chunk={}, tier="uncertain", score=0.5)]
        assert determine_sufficiency(chunks) == "partial"

    def test_determine_sufficiency_none(self):
        assert determine_sufficiency([]) == "none"

    def test_query_quality_specific(self):
        score = query_quality("termination notice period employment contract")
        assert score > 0.3

    def test_query_quality_vague(self):
        score = query_quality("tell me more about that")
        assert score < 0.6


class TestTopicCoverage:
    """Topic coverage mapping for active negative detection."""

    def test_build_topic_map(self):
        reset_topic_coverage()
        docs = [
            {"id": "d1", "filename": "python.md", "content": "Python programming language variables functions classes"},
        ]
        count = build_topic_map(docs)
        assert count == 1

    def test_check_coverage_covered(self):
        reset_topic_coverage()
        docs = [
            {"id": "d1", "filename": "python.md", "content": "Python programming language variables functions classes modules packages"},
        ]
        build_topic_map(docs)
        result = check_topic_coverage("python programming functions")
        assert result["covered"] is True

    def test_check_coverage_not_covered(self):
        reset_topic_coverage()
        docs = [
            {"id": "d1", "filename": "python.md", "content": "Python programming language variables functions"},
        ]
        build_topic_map(docs)
        result = check_topic_coverage("quantum mechanics entanglement superposition")
        # Should not be covered — no overlap
        assert result["covered"] is False

    def test_extract_topic_terms(self):
        text = "# Machine Learning\nNeural networks and deep learning algorithms"
        terms = _extract_topic_terms(text)
        assert len(terms) > 0
        assert "learning" in terms or "neural" in terms or "machine" in terms

    def test_check_coverage_not_ready(self):
        reset_topic_coverage()
        result = check_topic_coverage("anything")
        assert result["covered"] is True  # Default when not ready
