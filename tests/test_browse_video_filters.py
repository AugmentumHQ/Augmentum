"""Tests for the video-surface helpers in browse_routes + reputation.

Pure-function tests — duration parsing, bucket filtering, post-rerank
sort, and the diversity-cap bypass that lets YouTube dominate video
search. No FastAPI / DB setup required.
"""

from __future__ import annotations

from augmentum.proxy.browse_routes import (
    _apply_sort,
    _filter_by_duration,
    _parse_video_duration,
)
from augmentum.proxy.reputation import _apply_domain_diversity_cap

# ── _parse_video_duration ────────────────────────────────────────────────

def test_parse_duration_mmss():
    assert _parse_video_duration("3:45") == 225
    assert _parse_video_duration("12:00") == 720
    assert _parse_video_duration("0:30") == 30


def test_parse_duration_hhmmss():
    assert _parse_video_duration("1:00:00") == 3600
    assert _parse_video_duration("2:15:30") == 8130
    assert _parse_video_duration("10:00:00") == 36000


def test_parse_duration_int_seconds():
    """Vimeo and a few other engines return raw int seconds."""
    assert _parse_video_duration(225) == 225
    assert _parse_video_duration("225") == 225
    assert _parse_video_duration(3600.0) == 3600


def test_parse_duration_garbage_returns_none():
    assert _parse_video_duration("") is None
    assert _parse_video_duration(None) is None
    assert _parse_video_duration("LIVE") is None
    assert _parse_video_duration("just now") is None
    assert _parse_video_duration({}) is None


# ── _filter_by_duration ───────────────────────────────────────────────────

def test_duration_filter_short_keeps_only_under_4min():
    results = [
        {"url": "https://a/1", "duration": "1:30"},   # 90s short
        {"url": "https://a/2", "duration": "5:00"},   # 300s medium
        {"url": "https://a/3", "duration": "30:00"},  # 1800s long
    ]
    kept = _filter_by_duration(results, "short")
    assert [r["url"] for r in kept] == ["https://a/1"]


def test_duration_filter_medium_band():
    results = [
        {"url": "https://a/1", "duration": "3:00"},
        {"url": "https://a/2", "duration": "10:00"},
        {"url": "https://a/3", "duration": "19:59"},
        {"url": "https://a/4", "duration": "20:00"},
    ]
    kept = _filter_by_duration(results, "medium")
    # 240 <= s < 1200 — 19:59 (1199s) in, 20:00 (1200s) out, 3:00 out.
    assert [r["url"] for r in kept] == ["https://a/2", "https://a/3"]


def test_duration_filter_long_includes_anything_over_20min():
    results = [
        {"url": "https://a/1", "duration": "19:59"},
        {"url": "https://a/2", "duration": "20:00"},
        {"url": "https://a/3", "duration": "1:30:00"},
    ]
    kept = _filter_by_duration(results, "long")
    assert [r["url"] for r in kept] == ["https://a/2", "https://a/3"]


def test_duration_filter_keeps_unknown_duration():
    """Graceful: results we can't parse don't get silently dropped."""
    results = [
        {"url": "https://a/1", "duration": "3:00"},   # short
        {"url": "https://a/2", "duration": ""},       # unknown
        {"url": "https://a/3", "duration": "LIVE"},   # unknown
    ]
    kept = _filter_by_duration(results, "long")
    # Short gets dropped, unknown two are kept.
    assert [r["url"] for r in kept] == ["https://a/2", "https://a/3"]


def test_duration_filter_unknown_bucket_returns_input():
    results = [{"url": "https://a/1", "duration": "5:00"}]
    assert _filter_by_duration(results, "") == results
    assert _filter_by_duration(results, "bogus") == results


# ── _apply_sort ──────────────────────────────────────────────────────────

def test_sort_by_date_newest_first():
    results = [
        {"url": "a", "published_date": "2026-01-01"},
        {"url": "b", "published_date": "2026-05-01"},
        {"url": "c", "published_date": "2026-03-15"},
    ]
    sorted_ = _apply_sort(results, "date")
    assert [r["url"] for r in sorted_] == ["b", "c", "a"]


def test_sort_by_date_missing_dates_sink():
    results = [
        {"url": "a", "published_date": ""},
        {"url": "b", "published_date": "2026-05-01"},
        {"url": "c"},
    ]
    sorted_ = _apply_sort(results, "date")
    # 'b' wins; the two undated entries land below.
    assert sorted_[0]["url"] == "b"


def test_sort_by_duration_desc_longest_first():
    results = [
        {"url": "a", "duration": "3:00"},
        {"url": "b", "duration": "1:30:00"},
        {"url": "c", "duration": "10:00"},
    ]
    sorted_ = _apply_sort(results, "duration_desc")
    assert [r["url"] for r in sorted_] == ["b", "c", "a"]


def test_sort_by_duration_asc_shortest_first():
    results = [
        {"url": "a", "duration": "3:00"},
        {"url": "b", "duration": "1:30:00"},
        {"url": "c", "duration": "10:00"},
    ]
    sorted_ = _apply_sort(results, "duration_asc")
    assert [r["url"] for r in sorted_] == ["a", "c", "b"]


def test_sort_by_duration_unknown_sinks_to_bottom_either_direction():
    results = [
        {"url": "a", "duration": "10:00"},
        {"url": "b", "duration": ""},
        {"url": "c", "duration": "3:00"},
    ]
    asc = _apply_sort(results, "duration_asc")
    desc = _apply_sort(results, "duration_desc")
    # 'b' is unknown — always last regardless of direction.
    assert asc[-1]["url"] == "b"
    assert desc[-1]["url"] == "b"


def test_sort_unknown_value_returns_input():
    results = [{"url": "a", "duration": "10:00"}]
    assert _apply_sort(results, "") == results
    assert _apply_sort(results, "bogus") == results


# ── Diversity cap bypass ─────────────────────────────────────────────────

def test_diversity_cap_bypass_lets_youtube_dominate():
    """Video search: users want YouTube monopoly, not enforced diversity."""
    results = [
        {"url": "https://www.youtube.com/watch?v=1"},
        {"url": "https://www.youtube.com/watch?v=2"},
        {"url": "https://www.youtube.com/watch?v=3"},
        {"url": "https://www.youtube.com/watch?v=4"},
        {"url": "https://vimeo.com/x"},
    ]
    capped = _apply_domain_diversity_cap(
        results,
        max_per_domain=2,
        top_n=10,
        bypass_domains=frozenset({"youtube.com", "vimeo.com"}),
    )
    # All five preserved in original order — no overflow shuffle.
    assert [r["url"] for r in capped] == [r["url"] for r in results]


def test_diversity_cap_still_caps_non_bypass_domains():
    """Bypass set is targeted: it doesn't disable the cap for other hosts."""
    results = [
        {"url": "https://blog.example/1"},
        {"url": "https://blog.example/2"},
        {"url": "https://blog.example/3"},  # would normally overflow
        {"url": "https://other.example/x"},
    ]
    capped = _apply_domain_diversity_cap(
        results,
        max_per_domain=2,
        top_n=10,
        bypass_domains=frozenset({"youtube.com"}),  # irrelevant for these URLs
    )
    # blog.example still capped at 2; third one falls to the end.
    assert capped[-1]["url"] == "https://blog.example/3"


def test_diversity_cap_no_bypass_default_unchanged():
    """Calling without bypass_domains must behave exactly as before."""
    results = [
        {"url": "https://arxiv.org/1"},
        {"url": "https://arxiv.org/2"},
        {"url": "https://arxiv.org/3"},
        {"url": "https://wikipedia.org/x"},
    ]
    capped = _apply_domain_diversity_cap(results, max_per_domain=2, top_n=10)
    assert capped[0]["url"] == "https://arxiv.org/1"
    assert capped[1]["url"] == "https://arxiv.org/2"
    assert capped[2]["url"] == "https://wikipedia.org/x"
    assert capped[3]["url"] == "https://arxiv.org/3"
