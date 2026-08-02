"""Sprint 2 tests — topical aggregator (Piece 6).

The aggregator is a pure function over the observer's recent deque.
These tests cover:

* Below threshold (< min_events) → no threads
* Three events on same domain → one thread
* Events outside window → not in thread
* Events from other users → excluded
* Domain extraction strips www./m. prefixes
* Pareidolia: 2 events same domain + 1 elsewhere → no thread
* Keyword extraction prefers high-frequency non-stopwords
* Threads ranked by recency (last_seen DESC)
"""

from __future__ import annotations

import time

import pytest


def _evt(topic: str, payload: dict, t: float | None = None) -> dict:
    """Helper — build a deque-shaped event entry."""
    return {
        "topic": topic,
        "payload": payload,
        "t": t if t is not None else time.time(),
    }


# ── Threshold + grouping ─────────────────────────────────────────────


def test_below_min_events_returns_empty():
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt("surface.browse.opened",
             {"url": "https://example.com/a", "user_id": "u1"}, t=now),
        _evt("surface.browse.opened",
             {"url": "https://example.com/b", "user_id": "u1"}, t=now - 60),
    ]
    threads = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    assert threads == []


def test_three_events_same_domain_forms_thread():
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt("surface.browse.opened",
             {"url": f"https://example.com/article{i}",
              "user_id": "u1"}, t=now - i * 60)
        for i in range(3)
    ]
    threads = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    assert len(threads) == 1
    t = threads[0]
    assert t.topic == "example.com"
    assert t.domains == ("example.com",)
    assert t.event_count == 3


def test_events_outside_window_excluded():
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    in_window = [
        _evt("surface.browse.opened",
             {"url": "https://example.com/a", "user_id": "u1"}, t=now - 60),
        _evt("surface.browse.opened",
             {"url": "https://example.com/b", "user_id": "u1"}, t=now - 120),
    ]
    # An old event well outside the 4h window
    out_of_window = _evt(
        "surface.browse.opened",
        {"url": "https://example.com/c", "user_id": "u1"},
        t=now - 10 * 3600,
    )
    threads = aggregate_threads(
        in_window + [out_of_window], user_id="u1", min_events=3,
        window_seconds=4 * 3600, now=now,
    )
    # Only 2 in-window → below min_events
    assert threads == []


def test_other_users_events_excluded():
    """Critical isolation invariant — User A's events aren't in User B's threads."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt("surface.browse.opened",
             {"url": "https://example.com/a", "user_id": "u1"}, t=now - 60),
        _evt("surface.browse.opened",
             {"url": "https://example.com/b", "user_id": "u2"}, t=now - 60),
        _evt("surface.browse.opened",
             {"url": "https://example.com/c", "user_id": "u2"}, t=now - 120),
        _evt("surface.browse.opened",
             {"url": "https://example.com/d", "user_id": "u2"}, t=now - 180),
    ]
    threads_u1 = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    threads_u2 = aggregate_threads(events, user_id="u2", min_events=3, now=now)
    assert threads_u1 == []  # u1 only has 1 event
    assert len(threads_u2) == 1  # u2 has 3


def test_non_surface_topics_excluded():
    """Only surface.* events count — chat.turn_started shouldn't."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt("chat.turn_started",
             {"url": "https://example.com/a", "user_id": "u1"}, t=now - 60),
        _evt("chat.turn_started",
             {"url": "https://example.com/b", "user_id": "u1"}, t=now - 120),
        _evt("chat.turn_started",
             {"url": "https://example.com/c", "user_id": "u1"}, t=now - 180),
    ]
    threads = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    assert threads == []


def test_pareidolia_threshold_respected():
    """Two events on a domain + one elsewhere → no thread for either."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt("surface.browse.opened",
             {"url": "https://example.com/a", "user_id": "u1"}, t=now - 30),
        _evt("surface.browse.opened",
             {"url": "https://example.com/b", "user_id": "u1"}, t=now - 60),
        _evt("surface.browse.opened",
             {"url": "https://other.com/c", "user_id": "u1"}, t=now - 90),
    ]
    threads = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    assert threads == []  # no domain reaches 3 events


# ── Domain extraction ───────────────────────────────────────────────


def test_strips_www_prefix():
    from augmentum.companion_runtime.perception.topical import _extract_domain
    assert _extract_domain("https://www.example.com/a") == "example.com"
    assert _extract_domain("https://m.example.com/a") == "example.com"
    assert _extract_domain("https://example.com/a") == "example.com"


def test_handles_malformed_url():
    from augmentum.companion_runtime.perception.topical import _extract_domain
    assert _extract_domain("") == ""
    assert _extract_domain("not a url") == ""


# ── Keyword extraction ──────────────────────────────────────────────


def test_keyword_extraction_filters_stopwords():
    from augmentum.companion_runtime.perception.topical import _extract_keywords
    keywords = _extract_keywords(
        "The prefix caching strategy in transformers and the modern KV cache implementation"
    )
    # 'the', 'and', 'in' should be filtered. Long meaningful words kept.
    assert "the" not in keywords
    assert "and" not in keywords
    assert "prefix" in keywords or "caching" in keywords


def test_keyword_extraction_returns_top_n():
    from augmentum.companion_runtime.perception.topical import _extract_keywords
    keywords = _extract_keywords(
        "alpha beta gamma alpha beta delta alpha epsilon alpha",
        max_n=3,
    )
    assert len(keywords) <= 3
    # alpha appears 4 times — should be first
    assert keywords[0] == "alpha"


# ── Ranking + ordering ──────────────────────────────────────────────


def test_threads_ranked_by_recency():
    """Most-recent-last-seen first."""
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    # Two distinct domains, each with 3 events. example2 is more recent.
    events = [
        _evt("surface.browse.opened",
             {"url": "https://example1.com/a", "user_id": "u1"},
             t=now - 1000 - i * 60)
        for i in range(3)
    ] + [
        _evt("surface.browse.opened",
             {"url": "https://example2.com/a", "user_id": "u1"},
             t=now - i * 60)
        for i in range(3)
    ]
    threads = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    assert len(threads) == 2
    assert threads[0].topic == "example2.com"  # more recent first
    assert threads[1].topic == "example1.com"


def test_event_count_matches_event_ids_length():
    from augmentum.companion_runtime.perception.topical import aggregate_threads
    now = time.time()
    events = [
        _evt("surface.browse.opened",
             {"url": f"https://x.com/{i}", "user_id": "u1"},
             t=now - i * 60)
        for i in range(5)
    ]
    threads = aggregate_threads(events, user_id="u1", min_events=3, now=now)
    assert len(threads) == 1
    assert threads[0].event_count == 5
    assert len(threads[0].event_ids) == 5
