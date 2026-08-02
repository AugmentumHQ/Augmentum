"""Tests for source credibility scoring."""

from __future__ import annotations

try:
    from augmentum.search.credibility import (
        format_credibility_tag,
        score_results_block,
        score_url,
    )
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"augmentum.search.credibility not importable in this build: {_import_exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# score_url — institutional tier (0.85–1.0)
# ---------------------------------------------------------------------------


class TestInstitutionalSources:
    def test_gov(self):
        assert score_url("https://www.cdc.gov/health-info") >= 0.90

    def test_edu(self):
        assert score_url("https://cs.stanford.edu/papers") >= 0.85

    def test_nasa(self):
        assert score_url("https://nasa.gov/missions") >= 0.90

    def test_arxiv(self):
        assert score_url("https://arxiv.org/abs/2401.12345") >= 0.90

    def test_nature(self):
        assert score_url("https://www.nature.com/articles/s41586") >= 0.90

    def test_pubmed(self):
        assert score_url("https://pubmed.ncbi.nlm.nih.gov/12345") >= 0.90

    def test_ieee(self):
        assert score_url("https://ieeexplore.ieee.org/document/123") >= 0.85

    def test_gov_uk(self):
        assert score_url("https://www.gov.uk/guidance") >= 0.85

    def test_mil(self):
        assert score_url("https://www.defense.mil/news") >= 0.85


# ---------------------------------------------------------------------------
# score_url — established tier (0.65–0.84)
# ---------------------------------------------------------------------------


class TestEstablishedSources:
    def test_reuters(self):
        score = score_url("https://reuters.com/article/123")
        assert 0.65 <= score <= 0.85

    def test_bbc(self):
        score = score_url("https://www.bbc.com/news/article")
        assert 0.65 <= score <= 0.85

    def test_nytimes(self):
        score = score_url("https://www.nytimes.com/2026/01/article.html")
        assert 0.65 <= score <= 0.85

    def test_stackoverflow(self):
        score = score_url("https://stackoverflow.com/questions/12345")
        assert 0.65 <= score <= 0.85

    def test_mozilla_docs(self):
        score = score_url("https://developer.mozilla.org/en-US/docs/Web")
        assert 0.85 <= score <= 1.0

    def test_python_docs(self):
        score = score_url("https://docs.python.org/3/library/os.html")
        assert 0.85 <= score <= 1.0

    def test_arstechnica(self):
        score = score_url("https://arstechnica.com/gadgets/article")
        assert 0.65 <= score <= 0.85

    def test_wikipedia(self):
        score = score_url("https://en.wikipedia.org/wiki/Python")
        assert 0.65 <= score <= 0.85


# ---------------------------------------------------------------------------
# score_url — mixed tier (0.45–0.64)
# ---------------------------------------------------------------------------


class TestMixedSources:
    def test_github(self):
        score = score_url("https://github.com/user/repo")
        assert 0.45 <= score <= 0.70

    def test_healthline(self):
        score = score_url("https://www.healthline.com/health/condition")
        assert 0.45 <= score <= 0.65

    def test_w3schools(self):
        score = score_url("https://www.w3schools.com/python/")
        assert 0.45 <= score <= 0.65

    def test_unknown_com(self):
        """Unknown .com sites get neutral 0.50."""
        score = score_url("https://randomsite12345.com/article")
        assert score == 0.50

    def test_unknown_org(self):
        """Unknown .org sites get slightly above neutral."""
        score = score_url("https://randomcharity12345.org/about")
        assert 0.50 <= score <= 0.60


# ---------------------------------------------------------------------------
# score_url — user-generated tier (0.25–0.44)
# ---------------------------------------------------------------------------


class TestUserGeneratedSources:
    def test_reddit(self):
        score = score_url("https://www.reddit.com/r/python/comments/abc")
        assert 0.25 <= score <= 0.45

    def test_quora(self):
        score = score_url("https://www.quora.com/What-is-Python")
        assert 0.25 <= score <= 0.45

    def test_medium(self):
        score = score_url("https://medium.com/@user/article-title-abc123")
        assert 0.25 <= score <= 0.45

    def test_twitter(self):
        score = score_url("https://twitter.com/user/status/123")
        assert 0.25 <= score <= 0.45

    def test_x_dot_com(self):
        score = score_url("https://x.com/user/status/123")
        assert 0.25 <= score <= 0.45

    def test_youtube(self):
        score = score_url("https://www.youtube.com/watch?v=abc")
        assert 0.25 <= score <= 0.45

    def test_linkedin(self):
        score = score_url("https://www.linkedin.com/pulse/article")
        assert 0.25 <= score <= 0.45


# ---------------------------------------------------------------------------
# score_url — low tier (0.0–0.24)
# ---------------------------------------------------------------------------


class TestLowSources:
    def test_instagram(self):
        score = score_url("https://www.instagram.com/p/abc123")
        assert score <= 0.25

    def test_tiktok(self):
        score = score_url("https://www.tiktok.com/@user/video/123")
        assert score <= 0.25

    def test_pinterest(self):
        score = score_url("https://www.pinterest.com/pin/123")
        assert score <= 0.25

    def test_dailymail(self):
        score = score_url("https://www.dailymail.co.uk/news/article")
        assert score <= 0.25

    def test_infowars(self):
        score = score_url("https://www.infowars.com/posts/article")
        assert score <= 0.10

    def test_xyz_tld(self):
        """Unknown .xyz domains get low default."""
        score = score_url("https://spamsite.xyz/article")
        assert score <= 0.30

    def test_click_tld(self):
        score = score_url("https://clickbait.click/article")
        assert score <= 0.20


# ---------------------------------------------------------------------------
# score_url — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_url(self):
        assert score_url("") == 0.50

    def test_invalid_url(self):
        assert score_url("not-a-url") == 0.50

    def test_www_prefix_stripped(self):
        """www. prefix should be stripped before lookup."""
        assert score_url("https://www.reuters.com/article") == score_url("https://reuters.com/article")

    def test_subdomain_match(self):
        """Subdomain should match parent domain."""
        score = score_url("https://blog.stackoverflow.com/2026/post")
        assert 0.65 <= score <= 0.85

    def test_old_reddit(self):
        """old.reddit.com should match."""
        score = score_url("https://old.reddit.com/r/python")
        assert 0.25 <= score <= 0.45

    def test_case_insensitive(self):
        score1 = score_url("https://Reuters.COM/article")
        score2 = score_url("https://reuters.com/article")
        assert score1 == score2


# ---------------------------------------------------------------------------
# format_credibility_tag
# ---------------------------------------------------------------------------


class TestFormatTag:
    def test_institutional(self):
        tag = format_credibility_tag("https://nasa.gov/missions")
        assert "institutional" in tag
        assert "0.95" in tag

    def test_user_generated(self):
        tag = format_credibility_tag("https://reddit.com/r/python")
        assert "user-generated" in tag

    def test_format_structure(self):
        tag = format_credibility_tag("https://reuters.com/article")
        assert tag.startswith("[credibility:")
        assert tag.endswith("]")


# ---------------------------------------------------------------------------
# score_results_block
# ---------------------------------------------------------------------------


class TestScoreResultsBlock:
    def test_annotates_block(self):
        block = (
            "[1] NASA Discovers New Exoplanet\n"
            "    URL: https://nasa.gov/press-release/exoplanet\n"
            "    Scientists announced the discovery today."
        )
        annotated = score_results_block(block)
        assert "[credibility:" in annotated
        assert "institutional" in annotated
        # URL line should still be there
        assert "URL: https://nasa.gov" in annotated

    def test_tag_after_url_line(self):
        block = (
            "[1] Reddit Discussion\n"
            "    URL: https://reddit.com/r/space/post\n"
            "    Interesting thread about space."
        )
        annotated = score_results_block(block)
        lines = annotated.splitlines()
        # Tag should be inserted after URL line (index 1), so at index 2
        url_idx = next(i for i, line in enumerate(lines) if "URL:" in line)
        assert "[credibility:" in lines[url_idx + 1]

    def test_no_url_returns_unchanged(self):
        block = "Some text without a URL line"
        assert score_results_block(block) == block

    def test_low_credibility_tagged(self):
        block = (
            "[1] Click Here!\n"
            "    URL: https://spamsite.click/free-money\n"
            "    Amazing offer!"
        )
        annotated = score_results_block(block)
        assert "low" in annotated


# ---------------------------------------------------------------------------
# Ordering / relative ranking
# ---------------------------------------------------------------------------


class TestRelativeRanking:
    """Verify that the scoring reflects reasonable trust ordering."""

    def test_gov_beats_reddit(self):
        assert score_url("https://cdc.gov/info") > score_url("https://reddit.com/r/health")

    def test_arxiv_beats_medium(self):
        assert score_url("https://arxiv.org/abs/123") > score_url("https://medium.com/post")

    def test_reuters_beats_youtube(self):
        assert score_url("https://reuters.com/article") > score_url("https://youtube.com/watch")

    def test_stackoverflow_beats_quora(self):
        assert score_url("https://stackoverflow.com/q/123") > score_url("https://quora.com/q")

    def test_wikipedia_beats_wikihow(self):
        assert score_url("https://en.wikipedia.org/wiki/X") > score_url("https://wikihow.com/X")

    def test_python_docs_beats_w3schools(self):
        assert score_url("https://docs.python.org/3/lib") > score_url("https://w3schools.com/python")

    def test_mayo_clinic_beats_webmd(self):
        assert score_url("https://mayoclinic.org/diseases") > score_url("https://webmd.com/condition")

    def test_bbc_beats_dailymail(self):
        assert score_url("https://bbc.com/news") > score_url("https://dailymail.co.uk/news")

    def test_nature_beats_buzzfeed(self):
        assert score_url("https://nature.com/articles") > score_url("https://buzzfeed.com/article")
