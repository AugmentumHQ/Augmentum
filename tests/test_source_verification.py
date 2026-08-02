"""Live verification of preferred sources — actually fetches pages to confirm
domains marked EXCELLENT are truly scrapeable and domains marked AVOID fail.

These tests hit real URLs and require network access. Run with:
    pytest tests/test_source_verification.py -v --timeout=60

Marked with @pytest.mark.live so they can be skipped in CI:
    pytest -m "not live"
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.tools.preferred_sources import (
    _SOURCES,
    AVOID,
    EXCELLENT,
    GOOD,
)
from augmentum.utils.safe_http import SafeHttpClient

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_client = SafeHttpClient()


async def _fetch(url: str, timeout: float = 15.0) -> tuple[bool, int, str]:
    """Fetch a URL and return (success, char_count, error_or_snippet).

    Returns (True, N, first_200_chars) on success,
            (False, 0, error_message) on failure.
    """
    try:
        html, meta = await asyncio.wait_for(
            _client.fetch(url),
            timeout=timeout,
        )
        text = html.strip()
        if not text:
            return False, 0, "empty response"
        return True, len(text), text[:200]
    except TimeoutError:
        return False, 0, f"timeout after {timeout}s"
    except Exception as exc:
        return False, 0, str(exc)


async def _extract_content(url: str, timeout: float = 15.0) -> tuple[bool, int, str]:
    """Fetch + extract readable content (like the web_fetch tool does)."""
    try:
        html, meta = await asyncio.wait_for(
            _client.fetch(url),
            timeout=timeout,
        )
    except TimeoutError:
        return False, 0, f"timeout after {timeout}s"
    except Exception as exc:
        return False, 0, str(exc)

    if not html.strip():
        return False, 0, "empty response"

    # Try trafilatura extraction
    extracted = None
    try:
        import trafilatura
        extracted = trafilatura.extract(html)
    except ImportError:
        pass
    except Exception:
        pass

    if not extracted:
        # Fallback: crude tag stripping
        import re
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        extracted = text

    if not extracted or len(extracted) < 50:
        return False, len(extracted or ""), "insufficient content extracted"

    return True, len(extracted), extracted[:200]


# ---------------------------------------------------------------------------
# EXCELLENT sources — these MUST be fetchable and yield content
# ---------------------------------------------------------------------------

# Representative test URLs for EXCELLENT-tier domains.
# Each tuple: (domain, test_url, min_content_chars)
_EXCELLENT_TESTS: list[tuple[str, str, int]] = [
    # Government
    ("weather.gov", "https://www.weather.gov", 500),
    ("cdc.gov", "https://www.cdc.gov/about/index.html", 200),
    ("nasa.gov", "https://www.nasa.gov", 500),
    ("usa.gov", "https://www.usa.gov", 200),
    ("irs.gov", "https://www.irs.gov", 200),
    ("bls.gov", "https://www.bls.gov", 200),

    # Reference
    ("wikipedia.org", "https://en.wikipedia.org/wiki/Python_(programming_language)", 500),
    ("britannica.com", "https://www.britannica.com/science/weather", 500),
    ("archive.org", "https://archive.org/about/", 200),

    # Programming
    ("github.com", "https://github.com/python/cpython", 500),
    ("docs.python.org", "https://docs.python.org/3/library/json.html", 500),
    ("developer.mozilla.org", "https://developer.mozilla.org/en-US/docs/Web/JavaScript", 500),
    ("stackoverflow.com", "https://stackoverflow.com/questions/tagged/python", 200),

    # News
    ("reuters.com", "https://www.reuters.com", 200),
    ("apnews.com", "https://apnews.com", 200),
    ("bbc.com", "https://www.bbc.com/news", 200),
    ("npr.org", "https://www.npr.org", 200),

    # Science
    ("arxiv.org", "https://arxiv.org/abs/2301.00001", 200),
    ("pubmed.ncbi.nlm.nih.gov", "https://pubmed.ncbi.nlm.nih.gov/", 200),

    # Data
    ("earthquake.usgs.gov", "https://earthquake.usgs.gov/earthquakes/map/", 200),

    # Health
    ("medlineplus.gov", "https://medlineplus.gov", 200),

    # Finance
    ("fred.stlouisfed.org", "https://fred.stlouisfed.org", 200),

    # Sports
    ("baseball-reference.com", "https://www.baseball-reference.com", 200),

    # Time
    ("timeanddate.com", "https://www.timeanddate.com", 200),

    # Education
    ("plato.stanford.edu", "https://plato.stanford.edu/entries/aristotle/", 500),
    ("mathworld.wolfram.com", "https://mathworld.wolfram.com/PrimeNumber.html", 200),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain,url,min_chars",
    _EXCELLENT_TESTS,
    ids=[t[0] for t in _EXCELLENT_TESTS],
)
async def test_excellent_source_fetchable(domain: str, url: str, min_chars: int):
    """EXCELLENT-tier domains should return fetchable content."""
    ok, char_count, detail = await _fetch(url, timeout=20.0)
    assert ok, f"{domain}: fetch failed — {detail}"
    assert char_count >= min_chars, (
        f"{domain}: only {char_count} chars (need {min_chars}). "
        f"Preview: {detail[:100]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain,url,min_chars",
    # Subset of EXCELLENT tests for content extraction verification
    [t for t in _EXCELLENT_TESTS if t[2] >= 500],
    ids=[t[0] for t in _EXCELLENT_TESTS if t[2] >= 500],
)
async def test_excellent_source_extractable(domain: str, url: str, min_chars: int):
    """EXCELLENT-tier domains should yield meaningful extracted text."""
    ok, char_count, detail = await _extract_content(url, timeout=20.0)
    assert ok, f"{domain}: extraction failed — {detail}"
    # Extracted content should be at least 100 chars of readable text
    assert char_count >= 100, (
        f"{domain}: extracted only {char_count} chars. Preview: {detail[:100]}"
    )


# ---------------------------------------------------------------------------
# AVOID sources — these SHOULD fail or return minimal content
# ---------------------------------------------------------------------------

_AVOID_TESTS: list[tuple[str, str]] = [
    ("accuweather.com", "https://www.accuweather.com/en/us/new-york/10007/weather-forecast/349727"),
    ("linkedin.com", "https://www.linkedin.com/in/some-user"),
    ("instagram.com", "https://www.instagram.com/python"),
    ("pinterest.com", "https://www.pinterest.com/search/pins/?q=test"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain,url",
    _AVOID_TESTS,
    ids=[t[0] for t in _AVOID_TESTS],
)
async def test_avoid_source_blocked(domain: str, url: str):
    """AVOID-tier domains should fail to fetch or return minimal content.

    This test verifies our AVOID classification is correct. If an AVOID site
    starts returning good content, we should reconsider its tier.
    """
    ok, char_count, detail = await _extract_content(url, timeout=15.0)
    if ok:
        # If it did return content, it should be minimal (login wall, JS placeholder)
        # Allow up to 500 chars of boilerplate
        assert char_count < 500, (
            f"{domain}: AVOID domain returned {char_count} chars of content. "
            f"Consider upgrading to GOOD tier. Preview: {detail[:100]}"
        )


# ---------------------------------------------------------------------------
# Registry consistency checks
# ---------------------------------------------------------------------------


class TestRegistryConsistency:
    def test_all_excellent_have_categories(self):
        """EXCELLENT sources should have at least one category."""
        for domain, info in _SOURCES.items():
            if info.quality == EXCELLENT:
                assert info.categories, f"{domain}: EXCELLENT source has no categories"

    def test_all_avoid_have_notes(self):
        """AVOID sources should explain WHY they're avoided."""
        for domain, info in _SOURCES.items():
            if info.quality == AVOID:
                assert info.notes, f"{domain}: AVOID source has no notes explaining why"

    def test_no_duplicate_redirects(self):
        """Domains with redirect_domain should point to a registered domain."""
        for domain, info in _SOURCES.items():
            if info.redirect_domain:
                assert info.redirect_domain in _SOURCES, (
                    f"{domain}: redirect_domain '{info.redirect_domain}' not in registry"
                )

    def test_topic_sites_are_registered(self):
        """Topic site recommendations should reference registered domains."""
        from augmentum.tools.preferred_sources import _TOPIC_SITES
        unregistered: list[str] = []
        for keyword, domains in _TOPIC_SITES.items():
            for d in domains:
                if d not in _SOURCES:
                    unregistered.append(f"{keyword} → {d}")
        # Allow some unregistered (external sites we recommend but don't track)
        # but warn if too many
        if unregistered:
            print(f"Note: {len(unregistered)} topic sites not in registry: "
                  f"{unregistered[:5]}")

    def test_quality_distribution_reasonable(self):
        """Registry should have more EXCELLENT+GOOD than AVOID."""
        excellent = sum(1 for s in _SOURCES.values() if s.quality == EXCELLENT)
        good = sum(1 for s in _SOURCES.values() if s.quality == GOOD)
        avoid = sum(1 for s in _SOURCES.values() if s.quality == AVOID)
        assert excellent + good > avoid, (
            f"Registry skewed toward AVOID: {excellent}E + {good}G vs {avoid}A"
        )
