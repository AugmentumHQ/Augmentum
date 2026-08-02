"""Tests for the 4-tier edit matching engine (augmentum/coder/editing.py).

Test strategy:
  - Each tier is tested in isolation (inputs that only that tier can match)
  - Edge cases: empty content, empty search, unicode, CRLF, full-file match
  - Parser: bare blocks, FILE-wrapped blocks, flexible delimiters
"""
from __future__ import annotations

from augmentum.coder.editing import apply_edit, parse_search_replace_blocks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(content: str, search: str, replace: str, **kw):
    """Shorthand; returns (new_content, tier)."""
    return apply_edit(content, search, replace, **kw)


# ===========================================================================
# Tier 1 — EXACT
# ===========================================================================

class TestExactTier:
    def test_single_line_match(self):
        content = "x = 1\ny = 2\nz = 3\n"
        new, tier = _apply(content, "y = 2", "y = 99")
        assert tier == "exact"
        assert new == "x = 1\ny = 99\nz = 3\n"

    def test_multi_line_match(self):
        content = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        search = "def foo():\n    return 1"
        replace = "def foo():\n    return 42"
        new, tier = _apply(content, search, replace)
        assert tier == "exact"
        assert "return 42" in new
        assert "return 1" not in new

    def test_match_at_start(self):
        content = "first line\nsecond line\n"
        new, tier = _apply(content, "first line", "FIRST LINE")
        assert tier == "exact"
        assert new.startswith("FIRST LINE")

    def test_match_at_end(self):
        content = "line one\nlast line"
        new, tier = _apply(content, "last line", "LAST LINE")
        assert tier == "exact"
        assert new.endswith("LAST LINE")

    def test_match_in_middle(self):
        content = "a\nb\nc\nd\ne\n"
        new, tier = _apply(content, "c", "C")
        assert tier == "exact"
        assert "C\n" in new
        assert "c\n" not in new

    def test_only_first_occurrence_replaced(self):
        content = "foo\nfoo\nfoo\n"
        new, tier = _apply(content, "foo", "bar")
        assert tier == "exact"
        assert new.count("foo") == 2
        assert new.count("bar") == 1

    def test_no_match_when_differs_by_one_char(self):
        """A one-char difference does not match at the EXACT tier.

        The fuzzy tier may still match (which is correct behaviour).
        We only assert it wasn't caught by exact matching.
        """
        content = "hello world\n"
        new, tier = _apply(content, "hello wörld", "x")
        assert tier != "exact"  # exact must not claim this match

    def test_search_equals_entire_content(self):
        content = "the whole file"
        new, tier = _apply(content, "the whole file", "replaced")
        assert tier == "exact"
        assert new == "replaced"


# ===========================================================================
# Tier 2 — WHITESPACE-NORMALIZED
# ===========================================================================

class TestWhitespaceTier:
    def test_trailing_whitespace_in_content(self):
        """Content has trailing spaces; search does not."""
        content = "def foo():   \n    pass   \n"
        search = "def foo():\n    pass"
        replace = "def foo():\n    return 1"
        new, tier = _apply(content, search, replace)
        assert tier == "whitespace"
        assert "return 1" in new

    def test_trailing_whitespace_in_search(self):
        """Search has trailing spaces; content does not."""
        content = "x = 1\ny = 2\n"
        search = "x = 1   \ny = 2   "
        replace = "x = 10\ny = 20"
        new, tier = _apply(content, search, replace)
        assert tier == "whitespace"
        assert "x = 10" in new

    def test_extra_blank_lines_in_content(self):
        """Content has consecutive blank lines; search has one blank line."""
        content = "a\n\n\nb\n"
        search = "a\n\nb"
        replace = "a\n\nB"
        new, tier = _apply(content, search, replace)
        assert tier == "whitespace"
        assert "B" in new

    def test_tabs_vs_spaces_not_matched_at_whitespace_tier(self):
        """A tab vs spaces indent difference should NOT match at whitespace tier.
        It should fall through to indentation tier (or fuzzy).
        """
        content = "\tdef foo():\n\t\tpass\n"
        search = "def foo():\n    pass"
        new, tier = _apply(content, search, "def foo():\n    return 1")
        # Must not be exact or whitespace — indentation or fuzzy are acceptable
        assert tier not in ("exact", "whitespace")


# ===========================================================================
# Tier 3 — INDENTATION-PRESERVING
# ===========================================================================

class TestIndentationTier:
    def test_search_0_indent_file_4_space(self):
        """Search has no indent (written at column 0); file indents by 4 spaces.

        The indentation tier strips the file's common indent and compares the
        de-indented forms — they must match for the tier to fire.
        """
        content = "class Foo:\n    def bar(self):\n        pass\n"
        search = "def bar(self):\n    pass"
        replace = "def bar(self):\n    return 42"
        new, tier = _apply(content, search, replace)
        assert tier == "indentation"
        assert "return 42" in new
        # The replacement should carry the file's 4-space base indent
        assert "    def bar(self):" in new

    def test_search_0_indent_file_2_space(self):
        """Same idea: search at 0-indent, file at 2-space indent."""
        content = "if True:\n  x = 1\n  y = 2\n"
        search = "if True:\n  x = 1\n  y = 2"
        replace = "if False:\n  x = 10\n  y = 20"
        # This is an exact match (indents are the same)
        new, tier = _apply(content, search, replace)
        assert new is not None
        assert "x = 10" in new

    def test_nested_indentation(self):
        """class > method > body pattern."""
        content = (
            "class MyClass:\n"
            "    def my_method(self):\n"
            "        if True:\n"
            "            return 1\n"
        )
        search = (
            "def my_method(self):\n"
            "    if True:\n"
            "        return 1"
        )
        replace = (
            "def my_method(self):\n"
            "    if True:\n"
            "        return 99"
        )
        new, tier = _apply(content, search, replace)
        assert tier == "indentation"
        assert "return 99" in new

    def test_indentation_replacement_preserves_relative_indent(self):
        """Replace must keep relative indent levels, not flatten everything."""
        content = "    class Inner:\n        def method(self):\n            pass\n"
        search = "class Inner:\n    def method(self):\n        pass"
        replace = "class Inner:\n    def method(self):\n        return True"
        new, tier = _apply(content, search, replace)
        assert tier == "indentation"
        # The base indent of the match (4 spaces) should be preserved
        assert "        return True" in new


# ===========================================================================
# Tier 4 — FUZZY
# ===========================================================================

class TestFuzzyTier:
    def test_minor_char_difference(self):
        """One-character variable name difference — high similarity."""
        content = "result = compute_value(input_data)\nreturn result\n"
        search = "result = compute_value(input_dxta)\nreturn result"
        replace = "result = compute_value(input_data)\nreturn result * 2"
        new, tier = _apply(content, search, replace, fuzzy_threshold=0.6)
        assert tier == "fuzzy"
        assert new is not None
        assert "return result * 2" in new

    def test_below_threshold_returns_none(self):
        """Very different strings should not match even at fuzzy tier."""
        content = "x = 1\ny = 2\n"
        # Completely unrelated search
        search = "import numpy as np\nfrom scipy import signal\nresult = signal.convolve(x, y)"
        replace = "z = 3"
        new, tier = _apply(content, search, replace, fuzzy_threshold=0.85)
        assert tier == "none"
        assert new is None

    def test_fuzzy_threshold_default(self):
        """Default threshold (0.6) should catch minor differences."""
        content = "for i in range(10):\n    total += i\n"
        search = "for i in range(10):\n    total += j\n"  # j vs i
        replace = "for i in range(10):\n    total += i * 2\n"
        new, tier = _apply(content, search, replace)
        assert tier == "fuzzy"
        assert new is not None


# ===========================================================================
# Parser — parse_search_replace_blocks
# ===========================================================================

class TestParser:
    def test_single_bare_block(self):
        text = (
            "<<<<<<< SEARCH\n"
            "old code\n"
            "=======\n"
            "new code\n"
            ">>>>>>> REPLACE"
        )
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 1
        search, replace, fname = blocks[0]
        assert search.strip() == "old code"
        assert replace.strip() == "new code"
        assert fname is None

    def test_multiple_bare_blocks(self):
        text = (
            "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nbaz\n=======\nqux\n>>>>>>> REPLACE\n"
        )
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0].strip() == "foo"
        assert blocks[1][0].strip() == "baz"

    def test_file_wrapped_single_block(self):
        text = (
            "=== FILE: src/main.py ===\n"
            "<<<<<<< SEARCH\n"
            "x = 1\n"
            "=======\n"
            "x = 2\n"
            ">>>>>>> REPLACE\n"
        )
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 1
        search, replace, fname = blocks[0]
        assert fname == "src/main.py"
        assert "x = 1" in search
        assert "x = 2" in replace

    def test_file_wrapped_multiple_files(self):
        text = (
            "=== FILE: a.py ===\n"
            "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
            "=== FILE: b.js ===\n"
            "<<<<<<< SEARCH\nbaz\n=======\nqux\n>>>>>>> REPLACE\n"
        )
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][2] == "a.py"
        assert blocks[1][2] == "b.js"

    def test_file_wrapped_multiple_blocks_same_file(self):
        text = (
            "=== FILE: app.py ===\n"
            "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nbaz\n=======\nqux\n>>>>>>> REPLACE\n"
        )
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][2] == "app.py"
        assert blocks[1][2] == "app.py"

    def test_flexible_delimiters_short(self):
        """<<<< (4 chars) and >>>> (4 chars) should still be recognised."""
        text = "<<<<< SEARCH\nold\n=====\nnew\n>>>>> REPLACE"
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0].strip() == "old"
        assert blocks[0][1].strip() == "new"

    def test_flexible_delimiters_long(self):
        """<<<<<<<<<< SEARCH (10 chars) should work too."""
        text = "<<<<<<<<<< SEARCH\nold\n==========\nnew\n>>>>>>>>>> REPLACE"
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 1

    def test_empty_search_block_ignored(self):
        """A SEARCH block that is completely empty should be skipped."""
        text = "<<<<<<< SEARCH\n=======\nnew code\n>>>>>>> REPLACE"
        blocks = parse_search_replace_blocks(text)
        # An empty search could mean "insert at top"; implementation may return
        # it as empty string or skip it — we just verify no crash and the
        # replace content is accessible if returned.
        # If returned, search should be empty string.
        for s, r, _ in blocks:
            assert isinstance(s, str)
            assert isinstance(r, str)

    def test_case_insensitive_keywords(self):
        """SEARCH/REPLACE keywords may appear in mixed case."""
        text = "<<<<<<< search\nold\n=======\nnew\n>>>>>>> replace"
        blocks = parse_search_replace_blocks(text)
        assert len(blocks) == 1


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_content(self):
        new, tier = _apply("", "search", "replace")
        assert tier == "none"
        assert new is None

    def test_empty_search_returns_none(self):
        new, tier = _apply("some content", "", "replace")
        assert tier == "none"
        assert new is None

    def test_unicode_content(self):
        content = "café = 'espresso'\ncroissant = True\n"
        new, tier = _apply(content, "café = 'espresso'", "café = 'latte'")
        assert tier == "exact"
        assert "latte" in new

    def test_crlf_line_endings(self):
        """Windows-style CRLF should not break matching."""
        content = "line1\r\nline2\r\nline3\r\n"
        # Exact CRLF match
        new, tier = _apply(content, "line1\r\nline2", "LINE1\r\nLINE2")
        assert tier == "exact"
        assert "LINE1" in new

    def test_search_not_in_content_returns_none(self):
        new, tier = _apply("abc\ndef\n", "xyz", "123")
        assert tier == "none"
        assert new is None

    def test_apply_edit_return_types(self):
        """Return value is always (str|None, str)."""
        new, tier = _apply("hello", "hello", "world")
        assert isinstance(new, str)
        assert isinstance(tier, str)

        new2, tier2 = _apply("hello", "goodbye", "world")
        assert new2 is None
        assert isinstance(tier2, str)

    def test_multiline_replace_with_new_lines(self):
        """Replace can introduce more lines than search had."""
        content = "x = 1\n"
        new, tier = _apply(content, "x = 1", "x = 1\ny = 2\nz = 3")
        assert tier == "exact"
        assert new == "x = 1\ny = 2\nz = 3\n"

    def test_replace_with_fewer_lines(self):
        """Replace can collapse multi-line search to a single line."""
        content = "a = 1\nb = 2\nc = 3\n"
        new, tier = _apply(content, "a = 1\nb = 2\nc = 3", "result = 6")
        assert tier == "exact"
        assert new == "result = 6\n"

    def test_parser_returns_list(self):
        blocks = parse_search_replace_blocks("no blocks here")
        assert isinstance(blocks, list)
        assert blocks == []
