"""Tests for the web-search reranker in augmentum.proxy.reputation.

These are pure-function tests — no FastAPI / DB setup needed because
_rank_results takes already-collected scores and doesn't touch app state.
Coverage focuses on the query-awareness + diversity-cap behaviour that
fixed the "arxiv for polar bears" case.
"""

from __future__ import annotations

from augmentum.proxy.reputation import (
    _apply_domain_diversity_cap,
    _domain_matches_query,
    _rank_results,
    _tokenize_query,
)


def test_tokenize_drops_stopwords_and_short_tokens():
    assert _tokenize_query("the polar bears") == {"polar", "bears"}
    assert _tokenize_query("What is a transformer?") == {"transformer"}
    # Empty / whitespace-only queries produce no signal
    assert _tokenize_query("") == set()
    assert _tokenize_query("   ") == set()


def test_arxiv_off_topic_for_biology_query():
    """Regression: asking about polar bears shouldn't push arxiv to the top."""
    tokens = {"polar", "bears"}
    assert _domain_matches_query("arxiv.org", tokens) is False


def test_generalist_allowlist_matches_any_query():
    """Wikipedia + Britannica should fit factual queries without needing
    a category match in the curated source list."""
    tokens = {"polar", "bears"}
    assert _domain_matches_query("en.wikipedia.org", tokens) is True
    assert _domain_matches_query("britannica.com", tokens) is True


def test_arxiv_on_topic_for_research_query():
    tokens = {"transformer", "research", "paper"}
    assert _domain_matches_query("arxiv.org", tokens) is True


def test_rank_demotes_off_topic_excellent_sources():
    """Polar-bears scenario: excellent but off-topic arxiv should fall
    below on-topic generalists (wikipedia, natgeo, livescience)."""
    results = [
        {"url": "https://arxiv.org/1"},
        {"url": "https://arxiv.org/2"},
        {"url": "https://en.wikipedia.org/wiki/Polar_bear"},
        {"url": "https://www.nationalgeographic.com/animals/polar-bear"},
        {"url": "https://www.livescience.com/polar.html"},
    ]
    scores = {
        "arxiv.org": 5,
        "wikipedia.org": 5,
        "nationalgeographic.com": 5,
        "livescience.com": 5,
    }
    ranked = _rank_results(results, scores, query="polar bears")
    top_three_hosts = [r["url"].split("/")[2].lower().removeprefix("www.") for r in ranked[:3]]
    # Top three should NOT be arxiv — the on-topic sources must lead.
    assert not any(h.endswith("arxiv.org") for h in top_three_hosts)


def test_rank_keeps_on_topic_excellent_on_top():
    results = [
        {"url": "https://random.blog/t"},
        {"url": "https://arxiv.org/1"},
    ]
    scores = {"arxiv.org": 5}
    ranked = _rank_results(results, scores, query="transformer research paper")
    assert ranked[0]["url"] == "https://arxiv.org/1"


def test_diversity_cap_limits_same_domain():
    """Even when multiple results from one domain are excellent and
    topical, at most `max_per_domain` of them may occupy the first
    `top_n` slots — the rest get pushed further down."""
    results = [
        {"url": "https://arxiv.org/1"},
        {"url": "https://arxiv.org/2"},
        {"url": "https://arxiv.org/3"},
        {"url": "https://arxiv.org/4"},
        {"url": "https://wikipedia.org/x"},
        {"url": "https://nature.com/a"},
    ]
    capped = _apply_domain_diversity_cap(results, max_per_domain=2, top_n=10)
    # First two arxiv kept, next two pushed to end; wiki + nature move up.
    assert capped[0]["url"] == "https://arxiv.org/1"
    assert capped[1]["url"] == "https://arxiv.org/2"
    assert capped[2]["url"] == "https://wikipedia.org/x"
    assert capped[3]["url"] == "https://nature.com/a"
    # Overflow arxiv results come last, in original order.
    assert capped[4]["url"] == "https://arxiv.org/3"
    assert capped[5]["url"] == "https://arxiv.org/4"


def test_subdomain_inherits_parent_reputation():
    """en.wikipedia.org should score the same as wikipedia.org — otherwise
    half the Wikipedia corpus falls to middle tier."""
    results = [
        {"url": "https://en.wikipedia.org/wiki/Polar_bear"},
        {"url": "https://arxiv.org/1"},
    ]
    scores = {"wikipedia.org": 5, "arxiv.org": 5}
    ranked = _rank_results(results, scores, query="polar bears")
    # Wikipedia (generalist, on-topic) should be tier 0; arxiv (off-topic)
    # tier 1. So wiki leads.
    assert ranked[0]["url"] == "https://en.wikipedia.org/wiki/Polar_bear"


def test_bad_reputation_sinks_to_bottom():
    results = [
        {"url": "https://badsite.com/a"},
        {"url": "https://en.wikipedia.org/wiki/Polar_bear"},
    ]
    scores = {"badsite.com": -3, "wikipedia.org": 5}
    ranked = _rank_results(results, scores, query="polar bears")
    assert ranked[-1]["url"] == "https://badsite.com/a"


def test_no_query_preserves_original_behaviour():
    """When no query is passed, ranker should behave as it did before
    (excellent at top, good/unknown middle, bad bottom) with no
    topic-awareness filtering."""
    results = [
        {"url": "https://arxiv.org/1"},
        {"url": "https://random.blog/x"},
    ]
    scores = {"arxiv.org": 5}
    ranked = _rank_results(results, scores, query=None)
    assert ranked[0]["url"] == "https://arxiv.org/1"
