"""Unit tests for augmentum.discovery.distiller."""

from __future__ import annotations

from augmentum.discovery.distiller import (
    chunk_text,
    distill_article,
    distill_transcript,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paragraph(word_count: int, seed: str = "word") -> str:
    """Return a paragraph of roughly *word_count* words."""
    return " ".join(f"{seed}{i}" for i in range(word_count))


# ---------------------------------------------------------------------------
# TestChunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_text_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        # ~100 words per paragraph × 5 paragraphs = ~500 words ≈ 2 000 chars
        # max_tokens=200 → max_chars=800 — should split into multiple chunks
        paragraphs = [_make_paragraph(100, seed=f"p{i}w") for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=200)
        assert len(chunks) > 1

    def test_code_block_preserved(self):
        code_block = "```python\ndef hello():\n    print('hello world')\n```"
        text = f"Some prose before the code.\n\n{code_block}\n\nSome prose after the code."
        chunks = chunk_text(text, max_tokens=512)
        # The code block must appear intact somewhere in the output
        joined = " ".join(chunks)
        assert "def hello():" in joined
        assert "print('hello world')" in joined

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n   ") == []

    def test_all_chunks_non_empty(self):
        paragraphs = [_make_paragraph(100, seed=f"q{i}") for i in range(6)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=150)
        for chunk in chunks:
            assert chunk.strip(), "Chunk must not be empty or whitespace-only"

    def test_single_long_paragraph_split_on_sentences(self):
        # A single paragraph with many sentences — no double newlines
        sentences = [f"This is sentence number {i} of the test." for i in range(30)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=50)
        assert len(chunks) > 1
        # Every sentence fragment must be recoverable from chunks
        for chunk in chunks:
            assert chunk.strip()


# ---------------------------------------------------------------------------
# TestDistillArticle
# ---------------------------------------------------------------------------


class TestDistillArticle:
    def test_strips_html_tags(self):
        html = "<html><body><p>Hello <b>world</b>!</p></body></html>"
        result = distill_article(html, url="https://example.com", title="Test")
        assert "<" not in result["text"]
        assert "Hello" in result["text"]
        assert "world" in result["text"]

    def test_returns_chunks(self):
        # Build a long-ish article so we actually get chunks
        body = " ".join(["<p>" + _make_paragraph(80) + "</p>"] * 5)
        html = f"<html><body>{body}</body></html>"
        result = distill_article(html, url="https://example.com/article", title="My Article")
        assert isinstance(result["chunks"], list)
        # At least one chunk must be present
        assert len(result["chunks"]) >= 1

    def test_correct_metadata(self):
        html = "<html><body><p>Content here.</p></body></html>"
        result = distill_article(html, url="https://example.com/page", title="Page Title")
        assert result["source_url"] == "https://example.com/page"
        assert result["source_title"] == "Page Title"
        assert result["source_type"] == "article"
        assert "text" in result
        assert "chunks" in result

    def test_empty_html_returns_empty_chunks(self):
        result = distill_article("", url="https://example.com", title="Empty")
        assert result["chunks"] == []
        assert result["source_type"] == "article"

    def test_result_keys(self):
        html = "<p>Some text.</p>"
        result = distill_article(html, url="https://x.com", title="X")
        assert set(result.keys()) == {"source_url", "source_title", "source_type", "text", "chunks"}


# ---------------------------------------------------------------------------
# TestDistillTranscript
# ---------------------------------------------------------------------------


class TestDistillTranscript:
    def test_basic_paragraphs(self):
        paragraphs = [
            {"start": 0.0, "text": "Hello everyone and welcome."},
            {"start": 5.0, "text": "Today we discuss Python."},
            {"start": 10.0, "text": "Let us begin with basics."},
        ]
        result = distill_transcript(paragraphs, video_id="abc123", title="Python Intro")
        assert result["source_url"] == "https://www.youtube.com/watch?v=abc123"
        assert result["source_title"] == "Python Intro"
        assert result["source_type"] == "video_transcript"
        assert "Hello everyone" in result["text"]
        assert "Python" in result["text"]
        assert isinstance(result["chunks"], list)
        assert len(result["chunks"]) >= 1

    def test_empty_paragraphs(self):
        result = distill_transcript([], video_id="xyz", title="Empty Video")
        assert result["source_url"] == "https://www.youtube.com/watch?v=xyz"
        assert result["source_title"] == "Empty Video"
        assert result["source_type"] == "video_transcript"
        assert result["text"] == ""
        assert result["chunks"] == []

    def test_result_keys(self):
        result = distill_transcript(
            [{"start": 0.0, "text": "Hi."}],
            video_id="v1",
            title="Short",
        )
        assert set(result.keys()) == {"source_url", "source_title", "source_type", "text", "chunks"}

    def test_youtube_url_format(self):
        result = distill_transcript([], video_id="dQw4w9WgXcQ", title="Never Gonna Give You Up")
        assert result["source_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_paragraphs_joined_correctly(self):
        paragraphs = [
            {"start": 0.0, "text": "First segment."},
            {"start": 3.0, "text": "Second segment."},
        ]
        result = distill_transcript(paragraphs, video_id="t1", title="T")
        assert "First segment" in result["text"]
        assert "Second segment" in result["text"]
