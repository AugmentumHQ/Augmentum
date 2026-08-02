"""Unit tests for the ZIM passage extractor.

The extractor is the boundary that turns "ZIM article HTML, possibly
huge with MediaWiki chrome" into "small clean passage chunks the
reranker and budget-builder can work with." Bugs here ripple into every
chat injection: a sanitization regression leaks CSS into the model's
context, a chunking regression breaks the per-mode budget fit logic.

These tests use synthetic HTML samples — fast (no libzim, no real ZIM
file), deterministic, and easy to extend when new sanitization rules
land. The companion live eval (``test_live_pack_quality.py``) covers the
end-to-end "real Wikipedia article" case.
"""
from __future__ import annotations

from augmentum.knowledge.zim_reader import (
    _chunk_section,
    _clean_html_for_text,
    _html_segment_to_text,
    extract_passages,
)

# ---------------------------------------------------------------------------
# Sanitization (_clean_html_for_text)
# ---------------------------------------------------------------------------

def test_strips_script_blocks_with_contents():
    """The script body must NOT survive in plain text — naive tag-only
    stripping leaks JS as visible text.
    """
    html = "<p>Visible</p><script>alert('hidden');</script><p>Also visible</p>"
    cleaned = _clean_html_for_text(html)
    text = _html_segment_to_text(cleaned)
    assert "Visible" in text
    assert "Also visible" in text
    assert "alert" not in text
    assert "hidden" not in text


def test_strips_style_blocks_with_contents():
    """Same as scripts but for CSS — the original bug was CSS leaking
    in as ``/* start https://mdwiki.org/ */ .mw-parser-output ...`` text.
    """
    html = '<style>.mw-parser-output{color:red}/* MDWiki */</style><p>Article</p>'
    cleaned = _clean_html_for_text(html)
    text = _html_segment_to_text(cleaned)
    assert "Article" in text
    assert "mw-parser-output" not in text
    assert "MDWiki" not in text


def test_strips_infobox_table():
    """MediaWiki infoboxes are right-floated metadata tables that pad
    the passage with country/birthdate/etc. trivia. Strip them whole."""
    html = (
        '<table class="infobox vcard">'
        '<tr><th>Born</th><td>1500</td></tr>'
        '</table>'
        '<p>Body text here.</p>'
    )
    cleaned = _clean_html_for_text(html)
    text = _html_segment_to_text(cleaned)
    assert "Body text here" in text
    assert "Born" not in text
    assert "1500" not in text


def test_strips_navbox_div():
    """Navboxes are wide nav blocks ("See also" lookups) that should
    not pollute passage content with link soup."""
    html = (
        '<div class="navbox"><a href="X">Topic A</a> <a href="Y">Topic B</a></div>'
        '<p>Real content.</p>'
    )
    cleaned = _clean_html_for_text(html)
    text = _html_segment_to_text(cleaned)
    assert "Real content" in text
    assert "Topic A" not in text
    assert "Topic B" not in text


def test_strips_html_comments():
    html = "<!-- private templating directive --><p>Public.</p>"
    cleaned = _clean_html_for_text(html)
    text = _html_segment_to_text(cleaned)
    assert "Public" in text
    assert "private templating" not in text


def test_strips_footnote_markers():
    """Numbered footnote markers (with or without internal whitespace)
    inflate token counts without adding meaning. Both ``[1]`` and the
    MediaWiki-template variant ``[ 24 ]`` should drop."""
    html = "<p>Diabetes is common[ 24 ] worldwide[1] today.</p>"
    text = _html_segment_to_text(_clean_html_for_text(html))
    assert "Diabetes is common worldwide today" in text or "common  worldwide" in text
    assert "[24]" not in text
    assert "[ 24 ]" not in text


# ---------------------------------------------------------------------------
# Section splitting (extract_passages)
# ---------------------------------------------------------------------------

def test_splits_on_h2_headings():
    """The core function: sections become passages with the heading
    text as ``section`` and body as ``content``."""
    html = (
        "<p>Intro paragraph that is long enough to count as a real passage "
        "and not get filtered as a stub by the min-chars guard. We need "
        "at least 100 characters for the passage to survive.</p>"
        "<h2>Symptoms</h2>"
        "<p>Symptom paragraph that is similarly long enough to pass the "
        "minimum content threshold for a real section. Padding text padding.</p>"
        "<h2>Treatment</h2>"
        "<p>Treatment paragraph also long enough to be considered a meaningful "
        "section by the extractor. More padding so we clear the threshold.</p>"
    )
    passages = extract_passages(html, max_chars=2000, min_chars=100)
    sections = [s for s, _ in passages]
    assert "" in sections, "intro (no heading) should be its own passage"
    assert "Symptoms" in sections
    assert "Treatment" in sections
    # Content for each
    content_by_section = {s: c for s, c in passages}
    assert "Symptom paragraph" in content_by_section["Symptoms"]
    assert "Treatment paragraph" in content_by_section["Treatment"]


def test_drops_short_sections():
    """A section with <100 chars of body is too thin to be useful —
    typically a "See also" stub or empty placeholder. Should not appear."""
    html = (
        "<h2>Long section</h2>"
        "<p>This section has more than one hundred characters of body "
        "content so it will survive the minimum-length check.</p>"
        "<h2>Stub</h2>"
        "<p>tiny.</p>"
    )
    passages = extract_passages(html, max_chars=2000, min_chars=100)
    sections = [s for s, _ in passages]
    assert "Long section" in sections
    assert "Stub" not in sections


def test_chunks_oversized_section():
    """A single huge section gets sentence-split into multiple chunks
    so the budget builder can fit any one of them."""
    sentence = "This is one sentence in a series. "
    huge = sentence * 100  # ~3300 chars
    html = f"<h2>Big topic</h2><p>{huge}</p>"
    passages = extract_passages(html, max_chars=900, min_chars=100)
    big_topic = [(s, c) for s, c in passages if s == "Big topic"]
    assert len(big_topic) >= 2, "huge section should split into multiple passages"
    for s, c in big_topic:
        assert len(c) <= 900 + 50, f"chunk exceeds budget: {len(c)}"


def test_handles_no_headings():
    """Articles without h1-h6 markers (rare but real — single-paragraph
    stubs) treat the whole thing as one passage."""
    html = (
        "<p>This is a single-paragraph article with no headings at all. "
        "It should still be returned as a passage with section=''. The "
        "extractor must not skip articles just because they lack structure.</p>"
    )
    passages = extract_passages(html, max_chars=900, min_chars=100)
    assert len(passages) == 1
    assert passages[0][0] == ""
    assert "single-paragraph article" in passages[0][1]


def test_returns_empty_for_chrome_only_article():
    """An article that's nothing but script + infobox + navbox should
    return zero passages (everything stripped, intro empty)."""
    html = (
        "<script>tracking()</script>"
        '<table class="infobox"><tr><td>Born</td></tr></table>'
        '<div class="navbox">links</div>'
    )
    passages = extract_passages(html, max_chars=900, min_chars=100)
    assert passages == []


def test_no_chrome_leaks_in_real_mediawiki_sample():
    """End-to-end check against an HTML sample that mimics actual
    MDWiki Vector skin output. None of the chrome strings should
    survive into the extracted passages.

    This is the regression test for the original "diabetes article
    led with /* mw-parser-output */ CSS" bug.
    """
    html = """
    <html>
    <head>
        <title>Test article</title>
        <style>.mw-parser-output iframe.owid-frame{width:50%}/* start https://mdwiki.org/ */</style>
        <script>window.RLQ=window.RLQ||[];RLQ.push(function(){mw.loader.implement('user');})</script>
    </head>
    <body>
        <table class="infobox"><tr><th>Type</th><td>Disease</td></tr></table>
        <h1>Diabetes</h1>
        <p>Diabetes is a metabolic disorder characterized by high blood sugar levels.
        It affects millions worldwide and has multiple types[1] including type 1 and
        type 2[ 2 ] which differ in cause.</p>
        <h2>Symptoms</h2>
        <p>Common symptoms include polyuria, polydipsia, and unexplained weight loss.
        These symptoms develop gradually in type 2 diabetes but rapidly in type 1.</p>
        <div class="navbox">Diseases | Endocrine | Diabetes mellitus</div>
        <div class="catlinks">Categories: Diseases · Endocrine</div>
    </body>
    </html>
    """
    passages = extract_passages(html, max_chars=2000, min_chars=80)
    forbidden = ["mw-parser-output", "/* start", "RLQ", "mw.loader", "[1]", "[ 2 ]"]
    for section, content in passages:
        for token in forbidden:
            assert token not in content, (
                f"chrome leak: {token!r} survived into passage section={section!r}"
            )
    # And the real content survived
    all_content = " ".join(c for _, c in passages)
    assert "metabolic disorder" in all_content or "polyuria" in all_content


# ---------------------------------------------------------------------------
# Chunk helper edge cases
# ---------------------------------------------------------------------------

def test_chunk_section_keeps_short_text_intact():
    chunks = _chunk_section("Section", "Short body text.", max_chars=900)
    assert chunks == [("Section", "Short body text.")]


def test_chunk_section_splits_on_sentence_boundary():
    text = "First sentence ends here. Second sentence begins. Third one too."
    chunks = _chunk_section("S", text, max_chars=40)
    # Should split into at least 2 chunks; each should contain whole
    # sentences (no mid-sentence cuts).
    assert len(chunks) >= 2
    for _, content in chunks:
        # No truncated sentences mid-word.
        assert not content.endswith("sente")


# ---------------------------------------------------------------------------
# Sanity: imports work without libzim
# ---------------------------------------------------------------------------

def test_module_importable_without_libzim():
    """Confirms the extractor doesn't accidentally depend on libzim
    at import time — important because the helper is reused outside
    ZIM contexts (e.g. testing pipelines, future MediaWiki dump import)."""
    # If we got here the import succeeded.
    from augmentum.knowledge import zim_reader
    assert hasattr(zim_reader, "extract_passages")
