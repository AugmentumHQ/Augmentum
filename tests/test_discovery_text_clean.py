"""Tests for augmentum.discovery.text_clean and its callers.

Pins the sanitizer behavior against the 24h log failure modes:
  - HTML entities (``&#x27;``, ``&amp;``)
  - Raw HTML tags (``<span lang="en" dir="ltr">``)
  - Trailing decorators (``| Channel``, `` - Source``)
  - Pure-junk inputs that should clean to empty (and trigger empty-query skip)
"""
from __future__ import annotations

from augmentum.discovery.clustering import extract_signal_text
from augmentum.discovery.recommender import build_search_query
from augmentum.discovery.text_clean import clean_text_for_query


class TestCleanTextForQuery:
    def test_html_entities_decoded(self):
        assert clean_text_for_query("Pope says &#x27;tyrants&#x27;") == "Pope says 'tyrants'"
        assert clean_text_for_query("AT&amp;T outage") == "AT&T outage"

    def test_html_tags_stripped(self):
        out = clean_text_for_query('<span lang="en" dir="ltr">title</span>')
        assert out == "title"

    def test_whitespace_collapsed(self):
        assert clean_text_for_query("  a   b\n\tc  ") == "a b c"

    def test_trailing_pipe_stripped(self):
        # Real failure mode: "LIVE: Putin's Speech |" — pipe stranded by truncation.
        assert clean_text_for_query("Putin Speech |") == "Putin Speech"
        assert clean_text_for_query("Title - ") == "Title"

    def test_leading_junk_stripped(self):
        assert clean_text_for_query("| residue Title") == "residue Title"

    def test_pure_junk_returns_empty(self):
        # Caller relies on empty to skip the query entirely.
        assert clean_text_for_query("<span>") == ""
        assert clean_text_for_query("") == ""
        assert clean_text_for_query("   ") == ""
        assert clean_text_for_query("|||") == ""

    def test_length_cap(self):
        out = clean_text_for_query("word " * 100, max_chars=50)
        assert len(out) <= 50

    def test_passthrough_clean_input(self):
        assert clean_text_for_query("machine learning") == "machine learning"


class TestBuildSearchQueryEmptySkip:
    """Polluted cluster names must produce empty queries, not 'introduction to '."""

    def test_html_only_name_returns_empty(self):
        assert build_search_query("<span lang=") == ""

    def test_pipe_only_name_returns_empty(self):
        assert build_search_query(" | ") == ""

    def test_real_dirty_name_keeps_meaningful_text(self):
        # YouTube title that landed in cluster.name after truncation.
        q = build_search_query("Putin&#x27;s Emergency Speech Shocks the World |")
        assert q != ""
        assert "Putin's Emergency Speech Shocks the World" in q
        assert not q.endswith("|")

    def test_clean_name_still_works(self):
        q = build_search_query("rust async runtime")
        assert "rust async runtime" in q


class TestExtractSignalTextSanitizes:
    """Prevention layer: signals should land clean before clustering."""

    def test_strips_tags_from_title(self):
        sig = {"signal_type": "video_open", "source_title": "<b>Cool Video</b>", "metadata": {}}
        assert extract_signal_text(sig) == "Cool Video"

    def test_decodes_entities_in_search_query(self):
        sig = {
            "signal_type": "search_query",
            "source_title": "",
            "metadata": {"query": "AT&amp;T outage"},
        }
        assert extract_signal_text(sig) == "AT&T outage"

    def test_video_with_channel_concatenation_then_clean(self):
        sig = {
            "signal_type": "video_watch",
            "source_title": "Funny Cats &amp; Dogs",
            "metadata": {"channel": "Pets <em>HD</em>"},
        }
        out = extract_signal_text(sig)
        assert "Funny Cats & Dogs" in out
        assert "Pets HD" in out
        assert "<" not in out
