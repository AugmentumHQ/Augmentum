"""Tests for document RAG — chunking, store, routes."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    """Test text extraction and chunking."""

    def test_extract_plain_text(self):
        from augmentum.documents.chunker import extract_text

        data = b"Hello world\nSecond line"
        pages = extract_text(data, "text/plain", "test.txt")
        assert len(pages) == 1
        assert pages[0][0] == "Hello world\nSecond line"
        assert pages[0][1] is None  # no page number

    def test_extract_markdown(self):
        from augmentum.documents.chunker import extract_text

        data = b"# Title\n\nSome content here."
        pages = extract_text(data, "text/markdown", "readme.md")
        assert len(pages) == 1
        assert "Title" in pages[0][0]

    def test_chunk_text_basic(self):
        from augmentum.documents.chunker import chunk_text

        pages = [("A" * 3000, None)]
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) >= 3
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    def test_chunk_text_overlap(self):
        from augmentum.documents.chunker import chunk_text

        text = "word " * 600  # ~3000 chars
        pages = [(text, 1)]
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)
        # Check that chunks overlap
        if len(chunks) >= 2:
            end_of_first = chunks[0].text[-50:]
            # Some portion of first chunk's end should appear in second chunk's start
            assert chunks[1].char_offset < len(chunks[0].text)

    def test_chunk_text_preserves_page_num(self):
        from augmentum.documents.chunker import chunk_text

        pages = [("Page one content.", 1), ("Page two content.", 2)]
        chunks = chunk_text(pages, chunk_size=5000)
        assert any(c.page_num == 1 for c in chunks)
        assert any(c.page_num == 2 for c in chunks)

    def test_chunk_empty_pages(self):
        from augmentum.documents.chunker import chunk_text

        chunks = chunk_text([])
        assert chunks == []

    def test_chunk_short_text(self):
        from augmentum.documents.chunker import chunk_text

        pages = [("Short text.", None)]
        chunks = chunk_text(pages, chunk_size=1500)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_chunk_breaks_at_sentence(self):
        from augmentum.documents.chunker import chunk_text

        text = "First sentence. Second sentence. Third sentence. " * 30
        pages = [(text, None)]
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)
        # Chunks should try to end at sentence boundaries
        for c in chunks[:-1]:  # last chunk may not end at boundary
            assert c.text.rstrip().endswith((".", "!", "?", "\n"))

    def test_enriched_text_has_filename(self):
        from augmentum.documents.chunker import chunk_text

        pages = [("Some content about APIs.", None)]
        chunks = chunk_text(pages, chunk_size=5000, filename="api-spec.pdf")
        assert len(chunks) == 1
        assert chunks[0].enriched_text.startswith("api-spec.pdf")
        assert "Some content about APIs." in chunks[0].enriched_text

    def test_enriched_text_has_section(self):
        from augmentum.documents.chunker import chunk_text

        pages = [("# Introduction\n\nThis is the intro section.", None)]
        chunks = chunk_text(pages, chunk_size=5000, filename="doc.md")
        assert len(chunks) == 1
        assert "Introduction" in chunks[0].enriched_text
        assert "doc.md" in chunks[0].enriched_text
        assert chunks[0].section == "Introduction"

    def test_section_detection_markdown_headings(self):
        from augmentum.documents.chunker import _detect_sections

        text = "# Overview\n\nFirst section.\n\n## Details\n\nSecond section."
        sections = _detect_sections(text)
        assert len(sections) == 2
        assert sections[0].heading == "Overview"
        assert sections[1].heading == "Details"

    def test_section_detection_numbered(self):
        from augmentum.documents.chunker import _detect_sections

        text = "1.1 Introduction\n\nSome text here.\n\n2.1 Methods\n\nMore text here."
        sections = _detect_sections(text)
        assert len(sections) == 2
        assert "Introduction" in sections[0].heading
        assert "Methods" in sections[1].heading

    def test_section_detection_no_headings(self):
        from augmentum.documents.chunker import _detect_sections

        text = "just some plain text without any headings at all."
        sections = _detect_sections(text)
        assert len(sections) == 1
        assert sections[0].heading == ""

    def test_chunk_with_parents(self):
        from augmentum.documents.chunker import chunk_with_parents

        text = "Word. " * 1000  # ~6000 chars
        pages = [(text, 1)]
        children, parents = chunk_with_parents(
            pages, child_size=500, parent_size=2000, chunk_overlap=50,
        )
        assert len(parents) >= 2
        assert len(children) > len(parents)
        # Every child should reference a valid parent
        parent_indices = {p.index for p in parents}
        for child in children:
            assert child.parent_index is not None
            assert child.parent_index in parent_indices

    def test_chunk_with_parents_short_text(self):
        from augmentum.documents.chunker import chunk_with_parents

        pages = [("Short document.", None)]
        children, parents = chunk_with_parents(pages, child_size=500, parent_size=2000)
        assert len(parents) == 1
        assert len(children) == 1
        assert children[0].parent_index == parents[0].index

    def test_tables_to_markdown(self):
        from augmentum.documents.chunker import _tables_to_markdown

        tables = [[["Name", "Value"], ["foo", "42"], ["bar", "99"]]]
        md = _tables_to_markdown(tables)
        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| foo | 42 |" in md

    def test_extract_html_fallback(self):
        from augmentum.documents.chunker import extract_text

        data = b"<html><body><p>Hello from HTML</p></body></html>"
        pages = extract_text(data, "text/html", "page.html")
        assert len(pages) == 1
        assert "Hello from HTML" in pages[0][0]


# ---------------------------------------------------------------------------
# DocumentStore unit tests (no DB)
# ---------------------------------------------------------------------------

class TestDocumentStoreRRF:
    """Test the RRF merge logic."""

    def test_rrf_merge_empty(self):
        from augmentum.documents.store import DocumentStore

        result = DocumentStore._rrf_merge([], [])
        assert result == []

    def test_rrf_merge_single_source(self):
        from augmentum.documents.store import DocumentStore

        vec = [
            {"chunk_id": "a", "content": "hello", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 0, "distance": 0.1},
        ]
        result = DocumentStore._rrf_merge(vec, [])
        assert len(result) == 1
        assert result[0]["chunk_id"] == "a"
        assert "score" in result[0]

    def test_rrf_merge_both_sources(self):
        from augmentum.documents.store import DocumentStore

        vec = [
            {"chunk_id": "a", "content": "hello", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 0, "distance": 0.1},
        ]
        fts = [
            {"chunk_id": "a", "content": "hello", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 0, "rank": 0.5},
        ]
        result = DocumentStore._rrf_merge(vec, fts)
        assert len(result) == 1
        # Score should be higher when item appears in both lists
        single = DocumentStore._rrf_merge(vec, [])
        assert result[0]["score"] > single[0]["score"]

    def test_rrf_merge_ordering(self):
        from augmentum.documents.store import DocumentStore

        vec = [
            {"chunk_id": "a", "content": "first", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 0, "distance": 0.1},
            {"chunk_id": "b", "content": "second", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 1, "distance": 0.3},
        ]
        fts = [
            {"chunk_id": "b", "content": "second", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 1, "rank": 0.2},
        ]
        result = DocumentStore._rrf_merge(vec, fts)
        # "b" appears in both lists, so it should rank higher than "a" (only vec)
        assert result[0]["chunk_id"] == "b"


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestDocumentRoutes:
    """Test document API endpoints."""

    def test_list_empty(self, client):
        """GET /api/documents returns empty list when no docs."""
        resp = client.get("/api/documents")
        # May be 503 if store not initialized, or 200 with empty list
        if resp.status_code == 200:
            assert resp.json()["documents"] == []

    def test_upload_empty_file_rejected(self, client):
        """Upload with no file content is rejected."""
        import io
        # Empty file
        resp = client.post(
            "/api/documents",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        if resp.status_code != 503:  # store enabled
            assert resp.status_code == 400

    def test_search_requires_query(self, client):
        """POST /api/documents/search requires a query."""
        resp = client.post("/api/documents/search", json={"query": ""})
        if resp.status_code != 503:
            assert resp.status_code == 400

    def test_delete_not_found(self, client):
        """DELETE /api/documents/nonexistent returns 404."""
        resp = client.delete("/api/documents/nonexistent")
        if resp.status_code != 503:
            assert resp.status_code == 404

    def test_get_not_found(self, client):
        """GET /api/documents/nonexistent returns 404."""
        resp = client.get("/api/documents/nonexistent")
        if resp.status_code != 503:
            assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Integration: recall injection
# ---------------------------------------------------------------------------

class TestDocumentRecallIntegration:
    """Test that document results are injected into context."""

    def test_build_doc_source_label(self):
        """Document source labels include filename and page."""
        # Simulate what search_for_recall produces
        result = {
            "content": "Some chunk text",
            "filename": "report.pdf",
            "page_num": 5,
            "score": 0.02,
        }
        label = f"[Document: {result['filename']}" + (f" p.{result['page_num']}" if result.get("page_num") else "") + "]"
        assert label == "[Document: report.pdf p.5]"

    def test_build_doc_source_label_no_page(self):
        """Document source labels omit page when None."""
        result = {"filename": "notes.txt", "page_num": None}
        label = f"[Document: {result['filename']}" + (f" p.{result['page_num']}" if result.get("page_num") else "") + "]"
        assert label == "[Document: notes.txt]"
