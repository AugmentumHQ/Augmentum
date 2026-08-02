"""Tests for per-turn search-result dedup + productivity-guard accounting."""

from __future__ import annotations

import asyncio

import pytest

from augmentum.tools import turn_search_dedup as tsd
from augmentum.tools.turn_search_dedup import (
    TurnSearchDedup,
    get_turn_dedup,
    set_turn_dedup,
)


class TestMarkAndSeen:
    def test_mark_returns_true_once_then_false(self):
        d = TurnSearchDedup()
        assert d.mark("web", "https://a.com") is True
        assert d.mark("web", "https://a.com") is False  # duplicate
        assert d.total_new == 1
        assert d.total_dup == 1

    def test_trailing_slash_and_whitespace_normalized(self):
        d = TurnSearchDedup()
        assert d.mark("web", "https://a.com/page") is True
        assert d.mark("web", " https://a.com/page/ ") is False  # same after norm

    def test_buckets_are_independent(self):
        d = TurnSearchDedup()
        assert d.mark("web", "x") is True
        assert d.mark("image", "x") is True   # different bucket, still new
        assert d.mark("video", "x") is True

    def test_video_keyed_exactly_not_url_normalized(self):
        d = TurnSearchDedup()
        assert d.mark("video", "dQw4w9WgXcQ") is True
        assert d.seen("video", "dQw4w9WgXcQ") is True

    def test_seen_is_check_only(self):
        d = TurnSearchDedup()
        assert d.seen("image", "https://i.com/1.jpg") is False
        d.mark("image", "https://i.com/1.jpg")
        assert d.seen("image", "https://i.com/1.jpg") is True

    def test_empty_key_is_never_new(self):
        d = TurnSearchDedup()
        assert d.mark("web", "") is False
        assert d.seen("web", "") is False


class TestProductivityAccounting:
    def test_round_new_count_tracks_since_begin(self):
        d = TurnSearchDedup()
        d.mark("web", "a")
        d.begin_round()
        assert d.round_new_count() == 0
        d.mark("web", "b")
        d.mark("web", "c")
        assert d.round_new_count() == 2
        # A duplicate in the round doesn't count as new progress.
        d.mark("web", "b")
        assert d.round_new_count() == 2

    def test_round_of_all_duplicates_is_zero_progress(self):
        d = TurnSearchDedup()
        d.mark("web", "a")
        d.mark("web", "b")
        d.begin_round()
        d.mark("web", "a")  # already seen
        d.mark("web", "b")  # already seen
        assert d.round_new_count() == 0  # → productivity guard trips


class TestContextVar:
    def test_unset_is_none(self):
        async def _go():
            return get_turn_dedup()
        # Fresh context (separate task) — default is None.
        assert asyncio.run(_go()) is None

    def test_set_visible_within_same_async_context(self):
        async def _go():
            set_turn_dedup(TurnSearchDedup())
            # A coroutine awaited here shares the context → sees the dedup.
            async def _tool():
                d = get_turn_dedup()
                return d is not None and d.mark("web", "u")
            return await _tool()
        assert asyncio.run(_go()) is True

    def test_gathered_tasks_share_one_dedup_object(self):
        # Mirrors UARF: queries run via asyncio.gather; each task copies the
        # context but the dedup OBJECT is shared, so marks are consistent.
        async def _go():
            set_turn_dedup(TurnSearchDedup())

            async def _q(url):
                return get_turn_dedup().mark("web", url)

            # Two tasks mark the SAME url; exactly one should win "new".
            results = await asyncio.gather(_q("dup"), _q("dup"))
            return results, get_turn_dedup().total_new

        results, total_new = asyncio.run(_go())
        assert sorted(results) == [False, True]
        assert total_new == 1

    def test_reset_restores_prior(self):
        async def _go():
            token = set_turn_dedup(TurnSearchDedup())
            tsd.reset_turn_dedup(token)
            return get_turn_dedup()
        assert asyncio.run(_go()) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
