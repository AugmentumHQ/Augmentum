"""URL parsing + boilerplate stripping for the Gutenberg fetcher.

No network: every test operates on canned strings. The live HTTP path
(``fetch_plaintext``) is covered under the handler test with a mocked
httpx client.
"""

from __future__ import annotations

import pytest

from augmentum.media.gutenberg import (
    GutenbergError,
    resolve_ebook_id,
    strip_boilerplate,
)


class TestResolveEbookId:
    def test_ebooks_url(self):
        assert resolve_ebook_id("https://www.gutenberg.org/ebooks/2701") == "2701"

    def test_etext_url(self):
        assert resolve_ebook_id("http://www.gutenberg.org/etext/123") == "123"

    def test_files_html_url(self):
        url = "https://www.gutenberg.org/files/2701/2701-h/2701-h.htm"
        assert resolve_ebook_id(url) == "2701"

    def test_files_txt_url(self):
        url = "https://www.gutenberg.org/files/12345/12345-0.txt"
        assert resolve_ebook_id(url) == "12345"

    def test_cache_epub_url(self):
        url = "https://www.gutenberg.org/cache/epub/98/pg98.txt"
        assert resolve_ebook_id(url) == "98"

    def test_case_insensitive(self):
        assert resolve_ebook_id("HTTPS://WWW.GUTENBERG.ORG/EBOOKS/42") == "42"

    def test_empty_url(self):
        with pytest.raises(GutenbergError, match="no url_text_source"):
            resolve_ebook_id("")

    def test_non_gutenberg_url(self):
        with pytest.raises(GutenbergError, match="not a Gutenberg URL"):
            resolve_ebook_id("https://librivox.org/foo")

    def test_gutenberg_landing_no_id(self):
        # The root catalog URL has no ebook id — should not resolve.
        with pytest.raises(GutenbergError):
            resolve_ebook_id("https://www.gutenberg.org/")


_SAMPLE_BOOK = """The Project Gutenberg eBook of Moby Dick; or, The Whale

Some front matter we don't want.

*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK; OR, THE WHALE ***

Call me Ishmael. Some years ago—never mind how long precisely—having
little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery
part of the world.

*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK; OR, THE WHALE ***

License footer blah blah blah.
"""


class TestStripBoilerplate:
    def test_strips_header_and_footer(self):
        body = strip_boilerplate(_SAMPLE_BOOK)
        assert "Call me Ishmael" in body
        assert "Project Gutenberg eBook of" not in body
        assert "License footer" not in body
        assert "*** START OF" not in body
        assert "*** END OF" not in body

    def test_normalises_crlf(self):
        raw = _SAMPLE_BOOK.replace("\n", "\r\n")
        body = strip_boilerplate(raw)
        assert "\r" not in body
        assert "Call me Ishmael" in body

    def test_this_variant(self):
        # Older Gutenberg files say "THIS PROJECT GUTENBERG" not "THE".
        raw = (
            "preamble\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK FOO ***\n"
            "body body body\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK FOO ***\n"
            "footer\n"
        )
        body = strip_boilerplate(raw)
        assert body.strip() == "body body body"

    def test_missing_markers_returns_input(self):
        # Old pre-marker files don't carry the *** wrappers at all.
        # Better to return the whole body than strip it to empty string.
        raw = "just some body text with no markers anywhere"
        body = strip_boilerplate(raw)
        assert "just some body text" in body

    def test_only_start_marker(self):
        # Truncated or corrupted file with a start but no end — trust
        # the start marker and take everything after.
        raw = (
            "header\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
            "actual body\n"
        )
        body = strip_boilerplate(raw)
        assert body.strip() == "actual body"
        assert "header" not in body

    def test_only_end_marker(self):
        raw = (
            "front matter with no start marker\n"
            "body body body\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
            "license\n"
        )
        body = strip_boilerplate(raw)
        assert "license" not in body
        assert "body body body" in body

    def test_empty_input(self):
        assert strip_boilerplate("") == ""

    def test_trailing_newline_guaranteed(self):
        body = strip_boilerplate(_SAMPLE_BOOK)
        assert body.endswith("\n")
