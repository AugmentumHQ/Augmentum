"""Tests for zero-cost relevance filtering of search results."""

from __future__ import annotations

import pytest

from augmentum.search.relevance import (
    filter_results,
    score_relevance,
    _extract_keywords,
    _parse_result_block,
)


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_removes_stop_words(self):
        kw = _extract_keywords("what is the best way to learn Python")
        assert "what" not in kw
        assert "the" not in kw
        assert "python" in kw
        assert "learn" in kw
        assert "best" in kw

    def test_case_insensitive(self):
        kw = _extract_keywords("Python JavaScript Rust")
        assert "python" in kw
        assert "javascript" in kw
        assert "rust" in kw

    def test_short_words_filtered(self):
        kw = _extract_keywords("go is a good language")
        assert "go" not in kw  # len <= 2
        assert "good" in kw
        assert "language" in kw

    def test_empty_text(self):
        kw = _extract_keywords("")
        assert len(kw) == 0

    def test_frequency_counting(self):
        kw = _extract_keywords("war war peace war")
        assert kw["war"] == 3
        assert kw["peace"] == 1


# ---------------------------------------------------------------------------
# _parse_result_block
# ---------------------------------------------------------------------------


class TestParseResultBlock:
    def test_basic_block(self):
        block = """Just War Theory - Wikipedia
    URL: https://en.wikipedia.org/wiki/Just_war_theory
    [credibility: 0.75 — established]
    Just war theory deals with justification of warfare."""
        parsed = _parse_result_block(block)
        assert "Wikipedia" in parsed["title"]
        assert parsed["url"] == "https://en.wikipedia.org/wiki/Just_war_theory"
        assert "0.75" in parsed["credibility"]
        assert "justification" in parsed["snippet"]

    def test_no_credibility(self):
        block = """Weather Forecast 14605
    URL: https://weather.com/14605
    Partly cloudy, 48F."""
        parsed = _parse_result_block(block)
        assert parsed["credibility"] == ""
        assert parsed["url"] == "https://weather.com/14605"

    def test_empty_block(self):
        parsed = _parse_result_block("")
        assert parsed["title"] == ""
        assert parsed["url"] == ""


# ---------------------------------------------------------------------------
# score_relevance
# ---------------------------------------------------------------------------


class TestScoreRelevance:
    def test_highly_relevant(self):
        query = "is war ever okay ethics morality"
        block = """Just War Theory - Ethics of Warfare
    URL: https://example.com/war-ethics
    [credibility: 0.85 — institutional]
    Explores whether war can ever be morally justified through ethical frameworks."""
        score = score_relevance(query, block)
        assert score > 0.4

    def test_completely_irrelevant(self):
        query = "is war ever okay ethics morality"
        block = """How to Download Free eBooks Legally
    URL: https://forums.macrumors.com/threads/ebooks.12345
    [credibility: 0.40 — user-generated]
    Discussion about the legality of downloading ebooks if you own the print copy."""
        score = score_relevance(query, block)
        assert score < 0.15

    def test_partially_relevant(self):
        query = "weather forecast zip code 14605 today"
        block = """Weather Today in Seattle WA
    URL: https://weather.com/seattle
    Current conditions for Seattle, New York. Partly cloudy, 48F."""
        score = score_relevance(query, block)
        assert score > 0.2

    def test_irrelevant_to_weather(self):
        query = "weather forecast zip code 14605 today"
        block = """Zeobit Lawsuit Discussion
    URL: https://forums.macrumors.com/threads/zeobit.99999
    Director involved in Zeobit business ethics debate."""
        score = score_relevance(query, block)
        assert score < 0.15

    def test_empty_block_scores_zero(self):
        assert score_relevance("test query", "") == 0.0

    def test_no_query_keywords(self):
        # All stop words
        score = score_relevance("the is a", "Some result content here")
        assert score == 0.5  # Can't score → neutral

    def test_stem_matching(self):
        query = "warfare ethics"
        block = """War and Ethics
    URL: https://example.com/war
    The ethical implications of war throughout history."""
        score = score_relevance(query, block)
        # "war" should partially match "warfare"
        assert score > 0.1

    def test_title_bonus(self):
        query = "just war theory"
        block_title_match = """Just War Theory Overview
    URL: https://example.com/jwt
    An overview of the doctrine."""
        block_no_title = """Academic Resources
    URL: https://example.com/res
    Discussion of just war theory in context."""
        score_with_title = score_relevance(query, block_title_match)
        score_no_title = score_relevance(query, block_no_title)
        assert score_with_title > score_no_title

    def test_credibility_boost(self):
        query = "war ethics"
        high_cred = """War Ethics - Stanford
    URL: https://stanford.edu/war
    [credibility: 0.90 — institutional]
    War ethics examined."""
        low_cred = """War Ethics - Blog
    URL: https://random.xyz/war
    [credibility: 0.20 — low]
    War ethics examined."""
        score_high = score_relevance(query, high_cred)
        score_low = score_relevance(query, low_cred)
        assert score_high > score_low

    def test_precomputed_keywords(self):
        from collections import Counter
        query = "python programming"
        kw = Counter({"python": 1, "programming": 1})
        block = """Python Tutorial
    URL: https://example.com/py
    Learn Python programming basics."""
        score = score_relevance(query, block, query_keywords=kw)
        assert score > 0.3


# ---------------------------------------------------------------------------
# filter_results
# ---------------------------------------------------------------------------


class TestFilterResults:
    def test_filters_irrelevant(self):
        query = "is war ever justified"
        blocks = [
            """Just War Theory
    URL: https://example.com/war
    When is war morally justified?""",
            """Free eBook Downloads
    URL: https://forums.example.com/ebooks
    How to download ebooks legally for free.""",
            """Ethics of Armed Conflict
    URL: https://example.com/conflict
    The ethical framework for armed conflict and war.""",
        ]
        results = filter_results(query, blocks, min_score=0.15)
        # Should keep war/ethics results, drop ebook result
        texts = [block for block, _score in results]
        assert any("War Theory" in t for t in texts)
        assert any("Armed Conflict" in t for t in texts)
        assert not any("eBook" in t for t in texts)

    def test_empty_blocks(self):
        assert filter_results("test", []) == []

    def test_all_relevant_kept(self):
        query = "python tutorial"
        blocks = [
            """Python Basics
    URL: https://example.com/py1
    Learn Python fundamentals.""",
            """Advanced Python
    URL: https://example.com/py2
    Advanced Python programming techniques.""",
        ]
        results = filter_results(query, blocks, min_score=0.15)
        assert len(results) == 2

    def test_sorted_by_score(self):
        query = "war ethics"
        blocks = [
            """Cooking Recipes
    URL: https://example.com/cook
    Best chocolate cake recipe.""",
            """Just War Theory
    URL: https://example.com/war
    The ethics of war examined in depth.""",
        ]
        results = filter_results(query, blocks, min_score=0.0)
        # War should score higher than cooking
        assert "War" in results[0][0]

    def test_boost_top_n(self):
        query = "quantum physics"
        blocks = [
            f"""Result {i}
    URL: https://example.com/{i}
    Completely unrelated content about gardening."""
            for i in range(5)
        ]
        # With boost_top_n=3, should keep at least 3 regardless of score
        results = filter_results(query, blocks, min_score=0.5, boost_top_n=3)
        assert len(results) >= 3

    def test_min_score_zero_keeps_all(self):
        query = "test"
        blocks = [
            """Result A
    URL: https://a.com
    Content A.""",
            """Result B
    URL: https://b.com
    Content B.""",
        ]
        results = filter_results(query, blocks, min_score=0.0)
        assert len(results) == 2

    def test_strict_threshold(self):
        query = "war ethics morality"
        blocks = [
            """War Ethics Deep Dive
    URL: https://example.com/war-ethics
    Comprehensive analysis of war ethics and morality.""",
            """Random Forum Post
    URL: https://example.com/random
    Discussion about completely unrelated topics.""",
        ]
        results = filter_results(query, blocks, min_score=0.3)
        # High threshold should keep only clearly relevant
        assert len(results) <= 2
        if results:
            assert "War Ethics" in results[0][0]

    def test_real_world_war_query(self):
        """Test the exact scenario from the bug report: war ethics query
        with irrelevant MacRumors ebook/Zeobit results."""
        query = "is war ever okay ethics morality just war theory"
        blocks = [
            """Just War Theory - Wikipedia
    URL: https://en.wikipedia.org/wiki/Just_war_theory
    [credibility: 0.75 — established]
    Just war theory deals with the justification of how and why wars are fought.""",
            """Ethics of Warfare - University of Birmingham
    URL: https://www.birmingham.ac.uk/research/perspective/ethics-of-warfare
    [credibility: 0.85 — institutional]
    What could be more intuitive than the belief that it is morally wrong to kill.""",
            """Legality of Downloading eBooks - MacRumors Forums
    URL: https://forums.macrumors.com/threads/legality-of-downloading-ebooks.12345
    [credibility: 0.40 — user-generated]
    Is it legal to download free ebooks if you already own the physical copy?""",
            """Zeobit MacKeeper Lawsuit - MacRumors Forums
    URL: https://forums.macrumors.com/threads/zeobit-mackeeper.67890
    [credibility: 0.40 — user-generated]
    Director involvement in Zeobit business ethics and MacKeeper lawsuits.""",
            """Arguments For and Against War - ThoughtCo
    URL: https://www.thoughtco.com/arguments-for-and-against-war
    [credibility: 0.60 — mixed]
    Valid arguments exist both for and against the justification of war.""",
        ]
        results = filter_results(query, blocks, min_score=0.15)
        kept_texts = [block for block, _score in results]

        # Should keep war/ethics results
        assert any("Just War Theory" in t for t in kept_texts)
        assert any("Ethics of Warfare" in t for t in kept_texts)
        assert any("Arguments For and Against" in t for t in kept_texts)

        # Should drop ebook and Zeobit results
        assert not any("eBooks" in t for t in kept_texts)
        assert not any("Zeobit" in t for t in kept_texts)
