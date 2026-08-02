"""Content distillation pipeline — extract, clean, and chunk text for embedding."""

from __future__ import annotations

import re

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Rough token estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4

# Sentence boundary: end of sentence followed by whitespace (or end of string)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Code fence marker
_CODE_FENCE = "```"


def chunk_text(text: str, *, max_tokens: int = 512) -> list[str]:
    """Split *text* into chunks of roughly *max_tokens* each.

    Splitting strategy (in priority order):
    1. Split on paragraph boundaries (double newline).
    2. If a paragraph still exceeds *max_tokens*, split on sentence boundaries.
    3. Code blocks (```…```) are kept intact where possible.

    Returns an empty list for empty / whitespace-only input.
    All returned chunks are non-empty (stripped).
    """
    if not text or not text.strip():
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN

    # ------------------------------------------------------------------ #
    # Step 1: Split into paragraphs, preserving code blocks as atomic units
    # ------------------------------------------------------------------ #
    paragraphs = _split_paragraphs_preserve_code(text)

    # ------------------------------------------------------------------ #
    # Step 2: For oversized paragraphs, fall back to sentence splitting
    # ------------------------------------------------------------------ #
    segments: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars or para.startswith(_CODE_FENCE):
            # Keep code blocks and short paragraphs as-is
            segments.append(para)
        else:
            sentences = _SENTENCE_RE.split(para)
            segments.extend(sentences)

    # ------------------------------------------------------------------ #
    # Step 3: Greedily pack segments into chunks
    # ------------------------------------------------------------------ #
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_len = len(seg)

        if current_parts and current_len + seg_len + 1 > max_chars:
            # Flush current chunk
            chunks.append(" ".join(current_parts))
            current_parts = [seg]
            current_len = seg_len
        else:
            current_parts.append(seg)
            current_len += seg_len + (1 if current_parts else 0)

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [c for c in chunks if c.strip()]


def distill_article(html: str, *, url: str, title: str) -> dict:
    """Extract clean text from *html* and chunk it for embedding.

    Uses trafilatura for extraction; falls back to a regex tag-stripper if
    trafilatura is unavailable or returns nothing.

    Returns::

        {
            "source_url":   str,
            "source_title": str,
            "source_type":  "article",
            "text":         str,
            "chunks":       list[str],
        }
    """
    text = _extract_text(html)
    chunks = chunk_text(text) if text else []
    return {
        "source_url": url,
        "source_title": title,
        "source_type": "article",
        "text": text,
        "chunks": chunks,
    }


def distill_transcript(
    paragraphs: list[dict],
    *,
    video_id: str,
    title: str,
) -> dict:
    """Convert transcript *paragraphs* to clean text chunks.

    Each paragraph is a dict with at least ``{"start": float, "text": str}``.

    Returns::

        {
            "source_url":   "https://www.youtube.com/watch?v={video_id}",
            "source_title": str,
            "source_type":  "video_transcript",
            "text":         str,
            "chunks":       list[str],
        }
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    if not paragraphs:
        return {
            "source_url": url,
            "source_title": title,
            "source_type": "video_transcript",
            "text": "",
            "chunks": [],
        }

    text = " ".join(p.get("text", "").strip() for p in paragraphs if p.get("text", "").strip())
    chunks = chunk_text(text) if text else []

    return {
        "source_url": url,
        "source_title": title,
        "source_type": "video_transcript",
        "text": text,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_text(html: str) -> str:
    """Try trafilatura first, fall back to regex tag-stripping."""
    text = _try_trafilatura(html)
    if text:
        return text
    return _strip_tags(html)


def _try_trafilatura(html: str) -> str:
    """Run trafilatura extraction and return cleaned text (or empty string)."""
    try:
        import trafilatura  # type: ignore[import-untyped]

        result = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
            deduplicate=True,
        )
        return result or ""
    except Exception:
        log.warning("trafilatura extraction failed; using fallback")
        return ""


def _strip_tags(html: str) -> str:
    """Minimal HTML tag stripper used as a fallback."""
    # Remove script/style blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_paragraphs_preserve_code(text: str) -> list[str]:
    """Split on double-newline boundaries, keeping code fences intact."""
    # Tokenise into raw segments separated by blank lines
    raw_segments = re.split(r"\n{2,}", text)

    paragraphs: list[str] = []
    in_code_block = False
    code_accumulator: list[str] = []

    for seg in raw_segments:
        seg_stripped = seg.strip()
        if not seg_stripped:
            continue

        fence_count = seg_stripped.count(_CODE_FENCE)

        if in_code_block:
            code_accumulator.append(seg)
            # An odd number of fences in this segment closes the block
            if fence_count % 2 == 1:
                in_code_block = False
                paragraphs.append("\n\n".join(code_accumulator))
                code_accumulator = []
        else:
            if fence_count % 2 == 1:
                # Fence opened but not closed — accumulate
                in_code_block = True
                code_accumulator = [seg]
            else:
                # Normal paragraph (may contain a self-contained ``` block)
                paragraphs.append(seg_stripped)

    # Flush any unclosed code block
    if code_accumulator:
        paragraphs.append("\n\n".join(code_accumulator))

    return paragraphs
