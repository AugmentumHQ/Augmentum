"""Tests for the ZIM reader wrapper and ZIM pack integration."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from augmentum.knowledge.zim_reader import (
    ZimReader,
    ZimSuggestion,
)

# ------------------------------------------------------------------
# ZimReader tests (mocked libzim)
# ------------------------------------------------------------------


def _make_mock_entry(title: str, url: str, content: str, mimetype: str = "text/html"):
    """Build a mock libzim entry/item chain."""
    item = MagicMock()
    item.content = MagicMock()
    item.content.tobytes.return_value = content.encode("utf-8")
    item.mimetype = mimetype
    item.title = title
    item.path = url

    entry = MagicMock()
    entry.get_item.return_value = item
    entry.title = title
    entry.path = url
    return entry


def _wire_search(archive, searcher, entries):
    """Set up the libzim 3.x flow: result set yields paths, archive
    resolves each path to its entry via get_entry_by_path."""
    paths = [e.path for e in entries]
    path_to_entry = {e.path: e for e in entries}
    archive.get_entry_by_path.side_effect = lambda p: path_to_entry[p]

    result_set = MagicMock()
    result_set.__iter__ = MagicMock(return_value=iter(paths))
    search_obj = MagicMock()
    search_obj.getResults.return_value = result_set
    searcher.search.return_value = search_obj


class TestZimReader:
    @patch("augmentum.knowledge.zim_reader.libzim")
    def test_article_count(self, mock_libzim):
        archive = MagicMock()
        archive.entry_count = 42
        mock_libzim.Archive.return_value = archive
        mock_libzim.Searcher.return_value = MagicMock()

        reader = ZimReader("/tmp/test.zim")
        assert reader.article_count == 42
        reader.close()

    @patch("augmentum.knowledge.zim_reader.libzim")
    def test_close_clears_refs(self, mock_libzim):
        archive = MagicMock()
        archive.entry_count = 1
        mock_libzim.Archive.return_value = archive
        mock_libzim.Searcher.return_value = MagicMock()

        reader = ZimReader("/tmp/test.zim")
        reader.close()
        assert reader._archive is None  # noqa: SLF001
        assert reader._searcher is None  # noqa: SLF001


# ------------------------------------------------------------------
# Suggest (typeahead) tests
# ------------------------------------------------------------------
#
# ``ZimReader.suggest()`` is async — it holds an ``asyncio.Lock``
# around the libzim call (which runs on a worker thread). These
# tests cover the four important branches: empty/no-archive bailouts,
# the native SuggestionSearcher path, and the full-text Searcher
# fallback when SuggestionSearcher is unavailable.


def _wire_suggester(archive, suggester, entries):
    """Mirror ``_wire_search`` but for ``SuggestionSearcher.suggest()``.
    Result iterator yields entry paths; archive resolves them back to
    Entry mocks so the reader can pull title."""
    paths = [e.path for e in entries]
    path_to_entry = {e.path: e for e in entries}
    archive.get_entry_by_path.side_effect = lambda p: path_to_entry[p]

    result_set = MagicMock()
    result_set.__iter__ = MagicMock(return_value=iter(paths))
    sugg_obj = MagicMock()
    sugg_obj.getResults.return_value = result_set
    suggester.suggest.return_value = sugg_obj


class TestZimReaderSuggest:
    @patch("augmentum.knowledge.zim_reader.libzim")
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, mock_libzim):
        archive = MagicMock()
        mock_libzim.Archive.return_value = archive
        mock_libzim.Searcher.return_value = MagicMock()
        mock_libzim.SuggestionSearcher.return_value = MagicMock()

        reader = ZimReader("/tmp/test.zim")
        assert await reader.suggest("") == []
        assert await reader.suggest("   ") == []
        reader.close()

    @patch("augmentum.knowledge.zim_reader.libzim", None)
    @pytest.mark.asyncio
    async def test_no_libzim_returns_empty(self):
        """When libzim isn't installed the constructor short-circuits;
        suggest() must not blow up — return empty cleanly."""
        reader = ZimReader("/tmp/test.zim")
        assert await reader.suggest("python") == []

    @patch("augmentum.knowledge.zim_reader.libzim")
    @pytest.mark.asyncio
    async def test_native_suggester_returns_results(self, mock_libzim):
        archive = MagicMock()
        archive.entry_count = 100
        mock_libzim.Archive.return_value = archive
        mock_libzim.Searcher.return_value = MagicMock()

        suggester = MagicMock()
        mock_libzim.SuggestionSearcher.return_value = suggester
        entry_a = _make_mock_entry("Python", "A/Python", "")
        entry_b = _make_mock_entry("Python (programming)", "A/Python_(programming)", "")
        _wire_suggester(archive, suggester, [entry_a, entry_b])

        reader = ZimReader("/tmp/test.zim")
        out = await reader.suggest("pyt", limit=5)

        assert len(out) == 2
        assert all(isinstance(s, ZimSuggestion) for s in out)
        assert out[0].title == "Python"
        assert out[0].path == "A/Python"
        assert out[1].title == "Python (programming)"
        reader.close()

    @patch("augmentum.knowledge.zim_reader.libzim")
    @pytest.mark.asyncio
    async def test_falls_back_to_searcher_when_no_suggester(self, mock_libzim):
        """Older libzim builds (or ZIMs without a title index) lack
        SuggestionSearcher. The fallback path must kick in so typeahead
        still functions, just less crisp."""
        archive = MagicMock()
        mock_libzim.Archive.return_value = archive

        # Make hasattr(libzim, "SuggestionSearcher") return False.
        del mock_libzim.SuggestionSearcher

        searcher = MagicMock()
        mock_libzim.Searcher.return_value = searcher
        entry = _make_mock_entry("Python", "A/Python", "")
        _wire_search(archive, searcher, [entry])

        reader = ZimReader("/tmp/test.zim")
        # Suggester should have stayed None; suggest() flows through
        # the fallback Searcher branch.
        assert reader._suggester is None  # noqa: SLF001
        out = await reader.suggest("pyt", limit=5)
        assert len(out) == 1
        assert out[0].title == "Python"
        reader.close()

    @patch("augmentum.knowledge.zim_reader.libzim")
    @pytest.mark.asyncio
    async def test_suggester_exception_falls_through(self, mock_libzim):
        """If the native suggester throws (corrupt index, etc.), fall
        back to the Searcher rather than 500 to the route."""
        archive = MagicMock()
        mock_libzim.Archive.return_value = archive

        suggester = MagicMock()
        suggester.suggest.side_effect = RuntimeError("index corrupt")
        mock_libzim.SuggestionSearcher.return_value = suggester

        searcher = MagicMock()
        mock_libzim.Searcher.return_value = searcher
        entry = _make_mock_entry("Python", "A/Python", "")
        _wire_search(archive, searcher, [entry])

        reader = ZimReader("/tmp/test.zim")
        out = await reader.suggest("pyt", limit=5)
        # Native failed → Searcher fallback hit → results come through.
        assert len(out) == 1
        assert out[0].title == "Python"
        reader.close()

    @patch("augmentum.knowledge.zim_reader.libzim")
    @pytest.mark.asyncio
    async def test_skips_unresolvable_paths(self, mock_libzim):
        """Suggester occasionally returns paths that disappear before
        we can look them up (rare, but possible during pack reload).
        Skip silently rather than crash the whole batch."""
        archive = MagicMock()
        mock_libzim.Archive.return_value = archive
        mock_libzim.Searcher.return_value = MagicMock()

        suggester = MagicMock()
        mock_libzim.SuggestionSearcher.return_value = suggester

        # Two paths returned; the second raises on lookup.
        valid = _make_mock_entry("Python", "A/Python", "")

        def _resolve(p):
            if p == "A/Python":
                return valid
            raise KeyError(p)

        archive.get_entry_by_path.side_effect = _resolve
        result_set = MagicMock()
        result_set.__iter__ = MagicMock(return_value=iter(["A/Python", "A/Vanished"]))
        sugg_obj = MagicMock()
        sugg_obj.getResults.return_value = result_set
        suggester.suggest.return_value = sugg_obj

        reader = ZimReader("/tmp/test.zim")
        out = await reader.suggest("py", limit=5)
        # Only the resolvable entry survives.
        assert len(out) == 1
        assert out[0].path == "A/Python"
        reader.close()
