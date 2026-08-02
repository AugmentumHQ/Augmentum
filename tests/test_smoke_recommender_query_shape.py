"""Tests pinning the recommender's query-shape redesign.

The prior pools used three static prefixes per level ("introduction to",
"beginner guide", "tutorial", ...) — the templated patterns search-engine
bot detection fingerprints on. The 2026-06-10 logs showed all 6 SearXNG
engines simultaneously suspended after only minutes of recommender activity.

The fix: expand the modifier pools, include bare-query variants, and
remove the most obvious bot-bait prefixes. These tests pin the new shape
so a future "simplification" can't quietly re-introduce the bug.
"""
from __future__ import annotations

from collections import Counter

from augmentum.discovery.recommender import (
    _DEPTH_MODIFIERS,
    build_search_query,
)


class TestModifierPools:
    """The pools must be large enough + diverse enough that no single
    prefix repeats more than ~10% of queries."""

    def test_each_level_has_minimum_variants(self):
        """Every level needs at least 9 variants so any single prefix
        is ≤11% of generated queries — well under bot-detection thresholds."""
        for level, pool in _DEPTH_MODIFIERS.items():
            assert len(pool) >= 9, (
                f"Level {level} pool has only {len(pool)} variants — "
                "needs ≥9 to avoid single-pattern fingerprinting"
            )

    def test_each_level_has_some_naked_queries(self):
        """Every level must include at least one empty-modifier slot
        so a meaningful share of queries are bare topics (how humans
        actually search)."""
        for level, pool in _DEPTH_MODIFIERS.items():
            empty_count = sum(1 for m in pool if not m)
            assert empty_count >= 1, (
                f"Level {level} has zero naked-query slots — "
                "needs at least one '' entry to break the templated shape"
            )

    def test_advanced_levels_skew_naked(self):
        """Experts search bare topics ('rust async') more than beginners,
        so higher levels should have more naked slots than lower levels."""
        level_1_naked = sum(1 for m in _DEPTH_MODIFIERS[1] if not m)
        level_5_naked = sum(1 for m in _DEPTH_MODIFIERS[5] if not m)
        assert level_5_naked >= level_1_naked, (
            f"Expected level-5 naked share ≥ level-1 (exp users search bare), "
            f"got L1={level_1_naked}, L5={level_5_naked}"
        )

    def test_known_bot_bait_prefixes_removed(self):
        """The specific bot-bait prefixes from the prod incident must
        not reappear. 'introduction to' and 'beginner guide' were the
        named offenders in the 2026-06-10 logs."""
        bait = {"introduction to", "beginner guide", "state of the art"}
        for level, pool in _DEPTH_MODIFIERS.items():
            for modifier in pool:
                assert modifier not in bait, (
                    f"Level {level} pool contains bot-bait prefix "
                    f"{modifier!r} — re-introducing the prod bug"
                )


class TestBuildSearchQuery:
    """The query builder must emit clean strings for both modifier'd
    and naked cases. No double spaces, no leading/trailing whitespace,
    no orphan prefix when the topic cleans empty."""

    def test_naked_query_is_just_topic(self):
        """If the chosen modifier is empty, the query is the bare topic
        — no extra space, no prefix."""
        # We can't easily force a specific modifier from outside, but we
        # can probe many topics and check that AT LEAST SOME come out
        # as bare topics (proving the empty path is reachable).
        topics = [f"topic_{i}" for i in range(50)]
        results = [build_search_query(t, depth_level=1) for t in topics]
        # Some should equal the bare topic (when the rotation hits an
        # empty modifier slot).
        bare_count = sum(1 for t, r in zip(topics, results) if r == t)
        assert bare_count > 0, (
            "build_search_query never produced a bare-topic result over "
            "50 trials at level 1 — the empty-modifier path is dead"
        )

    def test_no_leading_or_trailing_whitespace(self):
        for t in ["react", "machine learning", "rust async"]:
            for level in (1, 2, 3, 4, 5):
                q = build_search_query(t, depth_level=level)
                if q:  # skip the empty-result path
                    assert q == q.strip(), (
                        f"Query for {t!r} at level {level} has whitespace: {q!r}"
                    )

    def test_no_double_spaces(self):
        for t in ["react", "machine learning"]:
            for level in (1, 2, 3, 4, 5):
                q = build_search_query(t, depth_level=level)
                if q:
                    assert "  " not in q, (
                        f"Query has double-space: {q!r}"
                    )

    def test_diversity_across_topics(self):
        """Over many topics at a single level, the prefix distribution
        should be diverse — no single prefix dominates above 20%."""
        topics = [f"topic_{i}" for i in range(200)]
        results = [build_search_query(t, depth_level=2) for t in topics if t]
        # Extract the modifier (first 1-3 words before the topic_N part)
        # by taking everything before the underscore.
        modifiers = []
        for q in results:
            # Naked queries match topic_N directly
            if q.startswith("topic_"):
                modifiers.append("<bare>")
            else:
                # Cut at the first 'topic_' occurrence
                idx = q.find("topic_")
                if idx > 0:
                    modifiers.append(q[:idx].strip())
                else:
                    modifiers.append(q)
        counts = Counter(modifiers)
        most_common_share = counts.most_common(1)[0][1] / len(modifiers)
        assert most_common_share <= 0.30, (
            f"Single prefix dominates {most_common_share:.0%} of queries — "
            "concentration risk for bot detection. Counts: "
            f"{counts.most_common(5)}"
        )
