"""Recommendation & search quality pipelines — tailored per consumer.

Every SearXNG consumer has different needs. A music search in the Grove
cares about livestream detection and playability. An AI web_search tool
cares about fetchability and content density. An image search cares about
resolution and format. One pipeline doesn't fit all.

Exported pipelines:
  - filter_and_rank()       — Discovery For You recommendations
  - filter_for_llm()        — AI web_search tool (text results for LLM context)
  - filter_for_video_ui()   — YouTube panel, Grove ambient, related videos
  - filter_for_images()     — AI image_search tool
  - filter_for_docs()       — Coder doc_search tool
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

_VIDEO_DOMAINS: set[str] = {
    "youtube.com", "youtu.be", "m.youtube.com",
    "vimeo.com", "player.vimeo.com",
    "dailymotion.com", "dai.ly",
    "tiktok.com",
    "twitch.tv", "clips.twitch.tv",
    "rumble.com",
}

_DOWNLOAD_EXTENSIONS: set[str] = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".msi", ".dmg", ".deb", ".rpm",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".flv",
    ".iso", ".img", ".bin",
}

# Non-English script blocks (3+ consecutive = likely not English)
_NON_LATIN_RE = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F"
    r"\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]{3,}",
)

_JUNK_TITLE_RE = re.compile(
    r"^("
    r"page not found|404|403|401|500|502|503|"
    r"access denied|just a moment|attention required|"
    r"checking your browser|please enable|enable javascript|"
    r"security check|captcha|blocked|unauthorized|"
    r"sign.?in|log.?in|cookie|subscribe to|are you a robot"
    r")",
    re.IGNORECASE,
)

_NON_CONTENT_RE = re.compile(
    r"(terms (of|and) (service|use)|privacy policy|cookie policy|"
    r"404 error|page cannot be|maintenance mode|under construction|coming soon)",
    re.IGNORECASE,
)


# Generic landing pages that the LLM-facing search filter and the curator
# both want to reject: dictionary defs for shared tokens, login walls,
# "Welcome to X" homepages, Reddit 403 placeholders, "What is X" SEO bait.
# When SearXNG returns "TOP Definition & Meaning - Merriam-Webster" for a
# query about cryptocurrencies, the LLM wastes a tool call on a useless
# definition page. Same pattern across consumers — share the rejection.
_LOW_VALUE_LANDING_DOMAINS: frozenset[str] = frozenset({
    # Dictionaries / thesauri
    "merriam-webster.com", "dictionary.com", "vocabulary.com",
    "thesaurus.com", "wordreference.com", "macmillandictionary.com",
    "collinsdictionary.com", "oxfordlearnersdictionaries.com",
    "ldoceonline.com", "freedictionary.com", "yourdictionary.com",
    # Generic reference landing pages
    "britannica.com",
    # Image-only platforms — the LLM gets a title but no fetchable text
    "pinterest.com", "pinterest.ca", "pinterest.co.uk",
    "pinterest.fr", "pinterest.de", "pinterest.com.au",
    # Defunct service whose archived pages still show in SearXNG
    "answers.yahoo.com",
    # Embedded PDFs / decks — title-only, the LLM can't read the slide content
    "slideshare.net",
})

_LOW_VALUE_LANDING_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "CHECK Definition & Meaning - Merriam-Webster"
    re.compile(r"^\s*\w[\w'-]*\s+definition\s*(&|and)\s*meaning\b", re.IGNORECASE),
    # "Define check - dictionary.com"
    re.compile(r"^\s*define\s+\w", re.IGNORECASE),
    # "Transcript of 'Google Drive introduction' video"
    re.compile(r"^\s*transcript\s+of\b", re.IGNORECASE),
    # "Sign-in to X", "Sign In to X", "Log in to X", "Login to X"
    re.compile(r"^\s*(sign[\s-]?in|sign[\s-]?up|log[\s-]?in|login)\s+(to|\w)", re.IGNORECASE),
    # "Welcome to <product homepage>" — homepage landing, not content
    re.compile(r"^\s*welcome\s+to\b", re.IGNORECASE),
    # Reddit's "we'd like to show you a description here" 403 placeholder
    re.compile(r"^\s*reddit\s*[—–-]\s*we\s+would\s+like\s+to\s+show", re.IGNORECASE),
    # "What is X?" / "What are X?" — definitional Q-style landing pages
    re.compile(r"^\s*what\s+(is|are)\b", re.IGNORECASE),
    # Wikipedia disambiguation / stub pages — title-only signal
    re.compile(r"\(disambiguation\)\s*[-—–]?\s*wikipedia", re.IGNORECASE),
    # "Topic - Quora" / "What is X? - Quora" — Quora SEO landings
    re.compile(r"^\s*\w[\w'\- ]{0,40}\s+[-—–]\s*quora\s*$", re.IGNORECASE),
    # AI-content-farm clickbait — "The Ultimate Guide to X", "Ultimate Guide:"
    re.compile(r"^\s*(the\s+)?ultimate\s+guide\s+(to|:)\b", re.IGNORECASE),
    # "10 Best X You Need to Know" / "7 Best Y You Must Try" listicle farms
    re.compile(r"^\s*\d+\s+(best|top)\s+.+\s+you\s+(need|must|should)\s+(know|try|see|use)", re.IGNORECASE),
    # "[YYYY] Best X" / "Best X in [YYYY]" — year-stuffed SEO listicles
    re.compile(r"^\s*(best|top)\s+\d+\s+\w+", re.IGNORECASE),
)


def _is_low_value_landing(domain: str, title: str) -> bool:
    """True when the result is a generic landing/definition page.

    Caller passes the already-normalized domain (lowercased, www. stripped)
    and the raw title. Used by both ``filter_for_llm`` (web_search tool)
    and the curator's note-pick filter so the same SearXNG firehose result
    is rejected on both surfaces with the same logic.
    """
    if domain in _LOW_VALUE_LANDING_DOMAINS:
        return True
    if not title:
        return False
    return any(pat.search(title) for pat in _LOW_VALUE_LANDING_TITLE_PATTERNS)


def _extract_domain(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_non_english(text: str) -> bool:
    """True if text is predominantly non-Latin script.

    Misnamed for historical reasons — this is a Latin-script-dominance
    test, not an actual language detector. A page in French is "English"
    by this measure; a page in Mandarin is not. Discovery's For-You
    pipeline uses it as a coarse "skip content I probably can't read"
    filter, on the assumption that scoring + domain reputation were
    tuned against an English corpus.

    The filter is bypassable via the ``discovery_allow_non_latin``
    setting — see ``_reject_rec`` for the gate. The other consumers of
    this function (web search, browse, video UI) intentionally keep the
    bias for now because their downstream paths (LLM context, embedded
    images) have their own per-locale concerns.
    """
    if not _NON_LATIN_RE.search(text):
        return False
    latin = len(re.findall(r"[a-zA-Z]", text))
    total = len(re.findall(r"\w", text))
    return total > 0 and latin / total < 0.5


def _is_junk_title(title: str) -> bool:
    if not title or len(title.strip()) < 5:
        return True
    return bool(_JUNK_TITLE_RE.search(title.strip()))


def _url_has_download_ext(url: str) -> bool:
    path = urlparse(url).path.lower()
    tail = path.split("/")[-1]
    if "." in tail:
        ext = "." + tail.rsplit(".", 1)[-1]
        return ext in _DOWNLOAD_EXTENSIONS
    return False


def _detect_content_type(url: str, domain: str) -> str:
    base = domain
    parts = base.split(".")
    parent = ".".join(parts[-2:]) if len(parts) > 2 else base
    if base in _VIDEO_DOMAINS or parent in _VIDEO_DOMAINS:
        return "video"
    path = urlparse(url).path.lower()
    if any(seg in path for seg in ("/video/", "/watch", "/embed/", "/clip/")):
        return "video"
    return "article"


def _extract_youtube_thumbnail(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if "youtube.com" in host:
            from urllib.parse import parse_qs
            vid = parse_qs(parsed.query).get("v", [None])[0]
            if vid:
                return f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
        elif host == "youtu.be":
            vid = parsed.path.lstrip("/").split("/")[0]
            if vid:
                return f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
    except (ValueError, AttributeError, KeyError) as exc:
        # Malformed URL or unexpected parse_qs shape — return empty so
        # the card just lacks a thumbnail.
        log.debug("youtube_thumbnail_extract_failed", url=url, error=str(exc))
    return ""


def _clean_title(title: str, domain: str) -> str:
    """Strip redundant site name suffixes and wrapping quotes."""
    title = title.strip()
    if " - " in title and len(title) > 60:
        parts = title.rsplit(" - ", 1)
        if len(parts) == 2 and domain:
            suffix = parts[1].lower().replace(" ", "")
            if suffix in domain.replace(".", "") or domain.replace(".", "") in suffix:
                title = parts[0].strip()
    if " | " in title and len(title) > 60:
        parts = title.rsplit(" | ", 1)
        if len(parts) == 2 and domain:
            suffix = parts[1].lower().replace(" ", "")
            if suffix in domain.replace(".", "") or domain.replace(".", "") in suffix:
                title = parts[0].strip()
    if title.startswith('"') and title.endswith('"') and title.count('"') == 2:
        title = title[1:-1]
    return title


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 1: Discovery For You — recommendations shown as cards
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_rec(result: dict) -> dict:
    url = result.get("url", "").strip()
    domain = _extract_domain(url) or result.get("domain", "")
    title = _clean_title(result.get("title") or "", domain)
    snippet = (result.get("snippet") or result.get("content") or "").strip()
    content_type = _detect_content_type(url, domain)
    thumbnail = result.get("thumbnail") or result.get("img_src") or ""
    if not thumbnail and content_type == "video":
        thumbnail = _extract_youtube_thumbnail(url)
    return {**result, "url": url, "title": title, "snippet": snippet,
            "domain": domain, "content_type": content_type, "thumbnail": thumbnail}


def _reject_rec(result: dict, *, allow_non_latin: bool = False) -> str | None:
    url = result.get("url", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    if not url or not url.startswith(("http://", "https://")):
        return "bad_url"
    if _url_has_download_ext(url):
        return "file_download"
    if _is_junk_title(title):
        return "junk_title"
    if _NON_CONTENT_RE.search(title):
        return "non_content"
    if not allow_non_latin and _is_non_english(title + " " + snippet):
        return "non_english"
    if len(url) > 500:
        return "url_too_long"
    if not result.get("domain"):
        return "no_domain"
    return None


def _score_rec(result: dict, domain_scores: dict[str, int] | None = None,
               source_fn=None) -> float:
    domain = result.get("domain", "")
    total = 0.1
    if domain_scores and domain:
        rep = domain_scores.get(domain)
        if rep is not None:
            total += 0.35 if rep >= 5 else 0.25 if rep >= 2 else 0.15 if rep >= 0 else 0.0
    if source_fn:
        try:
            q = source_fn(result.get("url", ""))
            total += 0.25 if q >= 2 else 0.15 if q >= 1 else -0.10 if q < 0 else 0.0
        except Exception:
            log.debug("quality_score_source_fn_failed", exc_info=True)
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    if 20 <= len(title) <= 100:
        total += 0.05
    if len(snippet) > 50:
        total += 0.08
    elif len(snippet) > 20:
        total += 0.04
    if result.get("thumbnail"):
        total += 0.07
    return min(total, 1.0)


async def filter_and_rank(
    results: list[dict], *, store=None,
    domain_scores: dict[str, int] | None = None,
    seen_urls: set[str] | None = None,
    allow_non_latin: bool = False,
) -> list[dict]:
    """Discovery For You — normalize, reject, score, dedup vs history, rank.

    ``allow_non_latin`` bypasses the Latin-script-dominance filter used to
    skip CJK/Arabic/Hebrew/Cyrillic content. Wired from the user setting
    ``discovery_allow_non_latin`` by recommender callers — quality.py
    stays settings-free so it remains pure-functional and testable.
    """
    if seen_urls is None:
        seen_urls = set()
    try:
        from augmentum.tools.preferred_sources import domain_quality
        source_fn = domain_quality
    except ImportError:
        source_fn = None

    normalized = [_normalize_rec(r) for r in results]
    passed = []
    for r in normalized:
        reason = _reject_rec(r, allow_non_latin=allow_non_latin)
        if reason:
            log.debug("rec_rejected", url=r.get("url", "")[:80], reason=reason)
            continue
        passed.append(r)

    visited_urls: set[str] | None = None
    if store and passed:
        try:
            # Use the URL-only variant — we only need existence for
            # dedup. The full-row variant pulls JSON metadata we
            # discard, paying a JSON parse per row.
            visited_urls = await store.check_visited_urls(
                [r["url"] for r in passed],
            )
        except Exception:
            # Visited-history is enrichment for re-rank; degrade
            # gracefully if the store call fails.
            log.debug("filter_visited_check_failed", exc_info=True)

    domain_counts: dict[str, int] = {}
    deduped = []
    for r in passed:
        url, domain = r["url"], r.get("domain", "")
        if url in seen_urls:
            continue
        if visited_urls and url in visited_urls:
            continue
        dc = domain_counts.get(domain, 0)
        if dc >= 3:
            continue
        r["_score"] = _score_rec(r, domain_scores, source_fn)
        seen_urls.add(url)
        domain_counts[domain] = dc + 1
        deduped.append(r)

    deduped.sort(key=lambda r: r.get("_score", 0), reverse=True)
    for r in deduped:
        r.pop("_score", None)
    return deduped


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 2: AI web_search tool — text results for LLM context
# ═══════════════════════════════════════════════════════════════════════════
# The LLM is making decisions based on these results. A junk URL wastes a
# tool call. A non-English result confuses the model. An unfetchable page
# means the follow-up web_fetch will fail. Prioritize fetchability.

# Domains known to block programmatic access (JS walls, CAPTCHAs)
_LLM_UNFETCHABLE_DOMAINS: set[str] = {
    "facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "tiktok.com", "pinterest.com",
    "quora.com",  # heavy JS, anti-bot
}
_LLM_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "but", "with", "from", "into", "onto", "this", "that",
    "these", "those", "what", "when", "where", "which", "while", "who", "whom",
    "how", "why", "can", "does", "did", "will", "would", "should", "could",
    "are", "was", "were", "been", "being", "have", "has", "had", "its",
    "tell", "show", "find", "get", "about", "some", "more", "most", "best",
    "good", "bad", "any", "all", "new", "old", "top", "info", "information",
    "help", "please", "thanks", "thank",
})


def _tokenize_llm_query(text: str) -> set[str]:
    if not text:
        return set()
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {token for token in tokens if token not in _LLM_QUERY_STOPWORDS}


def _source_topic_score(url: str, query_tokens: set[str], info_fn) -> int:
    if not query_tokens or not info_fn:
        return 0
    try:
        info = info_fn(url)
    except Exception:
        return 0
    if not info or not getattr(info, "categories", None):
        return 0

    score = 0
    for category in info.categories:
        cat = str(category or "").lower()
        if not cat:
            continue
        if cat in query_tokens:
            score += 3
            continue
        for token in query_tokens:
            if cat.startswith(token) or token.startswith(cat):
                score += 1
                break
    return score


def _query_relevance_score(query_tokens: set[str], *, title: str, snippet: str, url: str) -> int:
    if not query_tokens:
        return 0
    title_tokens = _tokenize_llm_query(title)
    body_tokens = _tokenize_llm_query(f"{snippet} {url}")
    title_overlap = len(query_tokens & title_tokens)
    body_overlap = len(query_tokens & body_tokens)
    return (title_overlap * 4) + (body_overlap * 2)


def filter_for_llm(
    results: list[dict],
    *,
    domain_scores: dict[str, int] | None = None,
    query: str | None = None,
) -> list[dict]:
    """Filter SearXNG results for LLM consumption (web_search tool).

    Goals: every result should be fetchable, in English, have a meaningful
    snippet for the LLM to reason about, and not waste a tool call slot.
    """
    try:
        from augmentum.tools.preferred_sources import domain_quality, get_source_info
        source_fn = domain_quality
        info_fn = get_source_info
    except ImportError:
        source_fn = None
        info_fn = None

    seen: set[str] = set()
    passed: list[dict] = []
    query_tokens = _tokenize_llm_query(query or "")

    for r in results:
        url = r.get("url", "").strip()
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        domain = _extract_domain(url)

        # Hard rejects
        if not url or not url.startswith(("http://", "https://")):
            continue
        if url in seen:
            continue
        if _url_has_download_ext(url):
            continue
        if _is_junk_title(title):
            continue
        if _is_non_english(title + " " + snippet):
            continue

        # Skip generic landing pages — dictionary defs for shared tokens
        # (the SearXNG firehose returning "TOP Definition & Meaning" for a
        # crypto query), login walls, Welcome-to homepages, Reddit 403s,
        # What-is SEO bait. Same logic the curator uses.
        if _is_low_value_landing(domain, title):
            continue

        # Skip domains the LLM can't fetch (JS walls, login required)
        if domain in _LLM_UNFETCHABLE_DOMAINS:
            continue

        # Check preferred_sources for JS requirement / paywall
        if info_fn:
            try:
                info = info_fn(url)
                if info and info.requires_js:
                    continue  # LLM fetch will get empty content
                if info and info.has_paywall:
                    continue  # LLM fetch will get paywall text
            except Exception:
                log.debug("filter_llm_info_fn_failed", url=url, exc_info=True)

        # Rank primarily on textual/query fit, with snippet richness and
        # source reputation acting as tie-breakers.
        query_score = _query_relevance_score(query_tokens, title=title, snippet=snippet, url=url)
        topic_score = _source_topic_score(url, query_tokens, info_fn)

        # Hard relevance floor — drop results that share ZERO tokens
        # with the query (after stopword strip). Without this, a query
        # like "new movies 2026 streaming" was getting back high-rep
        # news homepages (ABC News, BBC, CNN) whose titles share no
        # tokens with the query; the filter then ranked them by
        # domain reputation and surfaced 7 confidently-wrong sources.
        # Skipping the floor when query_tokens is empty (all-stopwords
        # query like "what is the best") because every result would
        # score zero by definition and dropping all of them helps no one.
        if query_tokens and query_score == 0 and topic_score == 0:
            continue

        snippet_score = 2.0 if len(snippet) > 120 else 1.0 if len(snippet) > 40 else 0.4 if len(snippet) > 10 else 0.1
        rep_score = 1.0
        if domain_scores:
            rep = domain_scores.get(domain)
            if rep is not None:
                rep_score = 1.5 if rep >= 5 else 1.2 if rep >= 2 else 1.0 if rep >= 0 else 0.5
        if source_fn:
            try:
                q = source_fn(url)
                if q >= 2:
                    rep_score *= 1.3
                elif q < 0:
                    rep_score *= 0.6
            except Exception:
                log.debug("filter_llm_source_fn_failed", url=url, exc_info=True)

        seen.add(url)
        r["_llm_score"] = (query_score * 10) + (topic_score * 5) + (snippet_score * 2) + rep_score
        r["title"] = _clean_title(title, domain)
        passed.append(r)

    # Stable sort by query fit first, then source trust and snippet richness.
    passed.sort(key=lambda r: r.get("_llm_score", 0), reverse=True)
    for r in passed:
        r.pop("_llm_score", None)
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 2b: Browse-tab search results — human-facing reading surface
# ═══════════════════════════════════════════════════════════════════════════
# The human user is looking at a search results list, deciding what to
# click. Different rules from the LLM pipeline:
#   * Keep non-English results — the user might be looking for them
#   * Keep file downloads (PDF/code/etc.) — browse_fetch renders them
#   * Keep "unfetchable" platform domains (Pinterest, LinkedIn, etc.) —
#     browse_fetch returns a clean "open in browser" card for these,
#     which is more useful than silently disappearing the result
# Still drop:
#   * Error pages, CAPTCHA walls, "checking your browser" pages
#   * Terms/privacy/maintenance/coming-soon stubs
#   * Bare URLs (no title, junk-titled results)
#   * URLs over 500 chars (tracking-laden affiliate redirects)


# SearXNG sometimes wraps snippet matches in <b>/<strong> highlight tags
# or returns lightly-escaped HTML entities. Cleanups happen here so all
# downstream consumers see clean text.
_SEARCH_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_search_snippet(snippet: str) -> str:
    """Strip HTML tags and unescape entities from a SearXNG snippet."""
    if not snippet:
        return ""
    import html as _html_module
    text = _SEARCH_HTML_TAG_RE.sub(" ", snippet)
    text = _html_module.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_for_browse(
    results: list[dict],
    *,
    query: str | None = None,
) -> list[dict]:
    """Filter SearXNG results for the human-facing browse search tab.

    Softer than filter_for_llm — keeps non-English, downloadable files,
    and platform domains that the browse pipeline can still render or at
    least show a clean "open in browser" card for. Drops only hard junk
    (error pages, CAPTCHA/JS walls, stub pages with no real content).
    """
    seen: set[str] = set()
    passed: list[dict] = []
    query_tokens = _tokenize_llm_query(query or "")

    for r in results:
        url = (r.get("url") or "").strip()
        title_raw = (r.get("title") or "").strip()
        snippet_raw = (r.get("content") or r.get("snippet") or "").strip()

        # Hard rejects — these waste a click no matter who's looking.
        if not url or not url.startswith(("http://", "https://")):
            continue
        if url in seen:
            continue
        if len(url) > 500:
            continue  # tracker-laden affiliate redirect

        # Clean before junk-detection so <b>404</b> still trips _is_junk_title.
        title = _clean_search_snippet(title_raw)
        snippet = _clean_search_snippet(snippet_raw)

        if _is_junk_title(title):
            continue
        if _NON_CONTENT_RE.search(title):
            continue
        if not title and not snippet:
            continue

        domain = _extract_domain(url)

        # Keep the cleaned text so callers don't have to re-clean.
        r["content"] = snippet
        if "snippet" in r:
            r["snippet"] = snippet
        r["title"] = _clean_title(title, domain)

        # Score for ranking — same shape as filter_for_llm but without
        # the source-fn JS/paywall penalty (the browse pipeline handles
        # those gracefully via the hostile-domain card).
        query_score = _query_relevance_score(
            query_tokens, title=r["title"], snippet=snippet, url=url
        )
        snippet_score = (
            2.0 if len(snippet) > 120
            else 1.0 if len(snippet) > 40
            else 0.4 if len(snippet) > 10
            else 0.1
        )

        seen.add(url)
        r["_browse_score"] = (query_score * 10) + (snippet_score * 2)
        passed.append(r)

    passed.sort(key=lambda r: r.get("_browse_score", 0), reverse=True)
    for r in passed:
        r.pop("_browse_score", None)
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 3: Video UI — YouTube panel, Grove ambient, related videos
# ═══════════════════════════════════════════════════════════════════════════
# Users see these as visual cards with thumbnails. A dead video, a
# non-music result in the Grove, or a title in another language breaks
# the experience. Each caller can pass context to tune the filter.

_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})")

# Words that indicate non-music/non-ambient content in Grove context
_GROVE_NOISE_RE = re.compile(
    r"\b(react|tutorial|how to|review|unboxing|prank|vlog|podcast|debate|"
    r"news|politics|interview|drama|trailer|gameplay|let.?s play|walkthrough|"
    r"mukbang|asmr eating|compilation|fails?|try not to)\b",
    re.IGNORECASE,
)

# Positive signals for ambient/music content
_GROVE_SIGNAL_RE = re.compile(
    r"\b(ambient|lofi|lo-fi|chill|relax|jazz|rain|fireplace|nature|"
    r"sleep|study|meditation|piano|guitar|acoustic|instrumental|"
    r"cafe|coffee shop|ocean|forest|thunderstorm|white noise|"
    r"soundtrack|ost|mix|playlist|hours?|stream|live)\b",
    re.IGNORECASE,
)


def filter_for_video_ui(
    results: list[dict],
    *,
    context: str = "general",
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Filter SearXNG video results for UI display.

    Args:
        context: "grove" for ambient/music (strict noise filtering),
                 "related" for related videos (moderate),
                 "general" for YouTube panel search (permissive).
        exclude_ids: Video IDs to skip (e.g., currently playing video).
    """
    if exclude_ids is None:
        exclude_ids = set()

    seen_ids: set[str] = set()
    passed: list[dict] = []

    for r in results:
        url = r.get("url", "")
        title = (r.get("title") or "").strip()

        # Extract YouTube video ID
        m = _YT_ID_RE.search(url)
        if not m:
            continue
        vid = m.group(1)

        # Dedup + exclude
        if vid in seen_ids or vid in exclude_ids:
            continue

        # Title quality
        if not title or len(title) < 3:
            continue
        if _is_junk_title(title):
            continue
        if _is_non_english(title):
            continue

        # Context-specific filtering
        score = 1.0

        if context == "grove":
            # Grove ambient: reject non-music content aggressively
            if _GROVE_NOISE_RE.search(title):
                continue
            # Boost ambient signals
            if _GROVE_SIGNAL_RE.search(title):
                score += 0.5
            # Short videos (<5min) are unlikely ambient content
            duration = r.get("length") or r.get("duration") or ""
            if duration and _parse_duration_seconds(duration) < 300:
                score -= 0.3

        elif context == "related":
            # Related videos: moderate filtering, reject obvious junk
            if _is_non_english(title):
                continue

        # General context: permissive, just basic validation

        seen_ids.add(vid)
        r["_vscore"] = score
        r["title"] = _clean_title(title, "youtube.com")
        passed.append(r)

    passed.sort(key=lambda r: r.get("_vscore", 0), reverse=True)
    for r in passed:
        r.pop("_vscore", None)
    return passed


def _parse_duration_seconds(duration: str) -> int:
    """Parse 'mm:ss' or 'hh:mm:ss' into seconds. Returns 0 on failure."""
    try:
        parts = duration.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 4: Image search — AI image_search tool
# ═══════════════════════════════════════════════════════════════════════════
# Images the LLM embeds in responses. Broken images, wrong aspect ratios,
# icons/favicons instead of content images, and watermarked stock photos
# all degrade the experience.

_IMAGE_JUNK_DOMAINS: set[str] = {
    "facebook.com", "fbcdn.net", "instagram.com", "cdninstagram.com",
    "twitter.com", "twimg.com", "x.com",
    "pinterest.com", "pinimg.com",
    "tiktok.com",
    "youtube.com", "ytimg.com",  # thumbnails, not content images
    "google.com", "gstatic.com", "googleapis.com",
    "gravatar.com", "wp.com",  # avatars, not content
    "cloudfront.net",  # too generic, often broken
}

_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}

# Patterns suggesting the image is a UI element, not content
_IMAGE_NOISE_RE = re.compile(
    r"(logo|icon|avatar|favicon|banner|button|arrow|spinner|"
    r"placeholder|thumbnail|default|blank|spacer|pixel|badge)",
    re.IGNORECASE,
)

_STOCK_DOMAINS: set[str] = {
    "shutterstock", "istockphoto", "gettyimages", "dreamstime",
    "depositphotos", "123rf", "alamy", "bigstockphoto",
    "stock.adobe", "canstockphoto",
}


def filter_for_images(
    results: list[dict],
    query: str,
    *,
    prefer_charts: bool = False,
) -> list[dict]:
    """Filter SearXNG image results for AI image_search tool.

    Goals: relevant, displayable, reasonable resolution, not UI junk.
    """
    query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
    seen: set[str] = set()
    scored: list[tuple[float, dict]] = []

    for r in results:
        img_url = r.get("img_src") or r.get("url", "")
        if not img_url or img_url in seen:
            continue

        domain = _extract_domain(img_url)
        title = (r.get("title") or "").strip()

        # Block junk domains
        if any(d in domain for d in _IMAGE_JUNK_DOMAINS):
            continue

        # Check extension (allow extensionless CDN URLs)
        path_lower = img_url.split("?")[0].lower()
        tail = path_lower.split("/")[-1]
        if "." in tail:
            ext = "." + tail.rsplit(".", 1)[-1]
            if ext not in _IMAGE_EXTENSIONS and len(ext) < 6:
                continue

        # Filter UI noise from URL path and title
        if _IMAGE_NOISE_RE.search(tail) or _IMAGE_NOISE_RE.search(title):
            continue

        # Non-English title
        if title and _is_non_english(title):
            continue

        # Score: keyword relevance
        title_words = set(re.findall(r"\b\w{3,}\b", title.lower()))
        source_words = set(re.findall(r"\b\w{3,}\b", (r.get("source") or "").lower()))
        overlap = len(query_words & (title_words | source_words))
        if overlap < 1 and len(query_words) > 1:
            continue  # no keyword match at all
        score = overlap / max(len(query_words), 1)

        # Chart preference boost
        if prefer_charts:
            chart_words = {"chart", "graph", "diagram", "figure", "plot",
                           "data", "statistics", "infographic", "visualization"}
            if (title_words | source_words) & chart_words:
                score += 0.4

        # Resolution scoring
        res = r.get("resolution", "")
        if res:
            try:
                parts = re.split(r"[x×]", res)
                if len(parts) == 2:
                    w, h = int(parts[0].strip()), int(parts[1].strip())
                    if w >= 800 and h >= 600:
                        score += 0.3
                    elif w >= 400 and h >= 300:
                        score += 0.1
                    elif w < 100 or h < 100:
                        continue  # too small to be useful
                    elif w < 200 or h < 200:
                        score -= 0.2
            except (ValueError, IndexError):
                pass

        # Stock photo penalty
        if any(d in (r.get("source") or "").lower() for d in _STOCK_DOMAINS):
            score -= 0.3

        seen.add(img_url)
        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored]


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline 5: Coder doc_search — documentation for coding agent
# ═══════════════════════════════════════════════════════════════════════════
# The coding agent needs authoritative, fetchable documentation — not SEO
# spam, not paywalled tutorials, not outdated blog posts. Prioritize
# official docs, recency, and code-rich content.

_DOC_TRUSTED: set[str] = {
    "docs.python.org", "pypi.org", "peps.python.org",
    "developer.mozilla.org", "nodejs.org", "typescriptlang.org", "npmjs.com",
    "doc.rust-lang.org", "docs.rs", "crates.io",
    "go.dev", "pkg.go.dev",
    "docs.oracle.com", "kotlinlang.org",
    "cppreference.com", "cplusplus.com",
    "learn.microsoft.com", "dotnet.microsoft.com",
    "ruby-doc.org", "rubygems.org",
    "php.net",
    "developer.apple.com",
    "devdocs.io", "stackoverflow.com", "github.com",
    "flask.palletsprojects.com", "fastapi.tiangolo.com", "djangoproject.com",
    "expressjs.com", "react.dev", "vuejs.org", "nextjs.org",
    "tailwindcss.com", "getbootstrap.com",
    "postgresql.org", "dev.mysql.com", "redis.io", "mongodb.com",
    "docs.docker.com", "kubernetes.io", "nginx.org",
}

_DOC_AVOID: set[str] = {
    "w3schools.com", "geeksforgeeks.org", "tutorialspoint.com",
    "javatpoint.com", "programiz.com",
}

# Domains that are JS-heavy and won't extract well for coder agent
_DOC_UNFETCHABLE: set[str] = {
    "medium.com",  # paywall + JS
    "dev.to",      # JS rendering needed for full content
    "hashnode.com",
    "substack.com",
}

# Max results any one domain may contribute to the front of the ranked
# doc_search output (see filter_for_docs). Keeps a single source — arxiv
# on a research query, one GitHub org, a doc farm — from drowning the rest.
_DOC_DOMAIN_CAP: int = 2


def filter_for_docs(
    results: list[dict],
    query: str,
    *,
    language: str = "",
) -> list[dict]:
    """Filter SearXNG results for coder doc_search tool.

    Goals: authoritative, fetchable, code-rich, in the right language.
    """
    seen: set[str] = set()
    scored: list[tuple[float, dict]] = []

    # Query-term relevance tokens (stopword-stripped, ≥3 chars) — shared
    # tokenizer with the haystack below so overlap is measured fairly.
    # Without a DOMINANT relevance term a trusted-domain page about the
    # WRONG topic outranks a mid-trust page about the right one — the
    # "unrelated official docs" failure mode small models kept hitting
    # (2026-07-06), and the broader "engine returned junk first" flood
    # (percona docker images for 'llama.cpp', NSFW reddit for 'MLX', an
    # ice-cream quiz, MDN WebRTC docs for a benchmark query) diagnosed
    # 2026-08-01.
    q_tokens = _tokenize_llm_query(query)

    for r in results:
        url = r.get("url", "").strip()
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        domain = _extract_domain(url)

        if not url or url in seen:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        if _url_has_download_ext(url):
            continue
        if _is_junk_title(title):
            continue
        if _is_non_english(title + " " + snippet):
            continue
        # Dictionary defs / "Welcome to X" homepages / login walls that
        # token-match a single query word (merriam-webster "media" for a
        # 'Media Molecule' query, egrammarbook "speculative meaning" for
        # 'speculative decoding') — the relevance floor CAN'T drop these
        # because the token legitimately overlaps. Same check filter_for_llm
        # already runs; propagated here 2026-08-01.
        if _is_low_value_landing(domain, title):
            continue

        # Block known-bad domains
        if domain in _DOC_AVOID or domain in _DOC_UNFETCHABLE:
            continue

        # ── Relevance-first scoring with a hard floor ──────────────────
        # Query fit is the DOMINANT term, not a multiplier on a positional
        # baseline. The old design seeded score from search position
        # (1/(n+1)) and multiplied by a soft 0.3–1.8× relevance factor, so
        # a zero-overlap page the engine returned FIRST (a percona docker
        # image for 'llama.cpp', NSFW reddit for 'MLX', an ice-cream quiz)
        # outranked an on-topic page found later — and nothing was ever
        # dropped. This mirrors the fix already in filter_for_llm.
        hay_tokens = _tokenize_llm_query(f"{title} {snippet} {url}")
        title_tokens = _tokenize_llm_query(title)
        title_overlap = len(q_tokens & title_tokens)
        body_overlap = len(q_tokens & hay_tokens)

        # Hard relevance floor — drop results that share ZERO significant
        # tokens with the query. Skipped only when the query itself has no
        # significant tokens (all-stopword ask: nothing to measure, and
        # dropping everything helps no one).
        if q_tokens and title_overlap == 0 and body_overlap == 0:
            continue

        # Query fit dominates; domain trust / richness / recency are only
        # tie-breakers and can no longer carry an off-topic page.
        score = (title_overlap * 4.0) + (body_overlap * 2.0)

        if domain in _DOC_TRUSTED:
            score += 3.0
        elif domain.endswith((".dev", ".io")) and "/docs" in url:
            score += 2.0
        elif "github.com" in domain and "/blob/" not in url and "/gist" not in url:
            score += 1.5

        # Snippet richness — code indicators are gold for a coder agent.
        if snippet:
            if any(c in snippet for c in ("```", "def ", "function ", "class ", "import ", "const ")):
                score += 1.5
            if len(snippet) > 100:
                score += 0.5

        # Language relevance — if the caller pinned a language, nudge matches.
        if language and language.lower() in (title + " " + snippet + " " + url).lower():
            score += 1.0

        # Recency signal — newer docs are more relevant (additive, small).
        pub = r.get("publishedDate", "")
        if pub:
            try:
                from datetime import datetime, timezone
                if "T" in pub or len(pub) == 10:
                    pub_date = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    if pub_date.tzinfo is None:
                        pub_date = pub_date.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - pub_date).days
                    if age_days < 90:
                        score += 1.0
                    elif age_days > 1095:
                        score -= 0.5
            except (ValueError, TypeError):
                # Human-readable dates ("3 days ago") — skip the bonus
                # rather than crash the rank.
                pass

        seen.add(url)
        r["_doc_score"] = score
        r["title"] = _clean_title(title, domain)
        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])

    # Per-domain diversity cap — no single source may take more than
    # ``_DOC_DOMAIN_CAP`` of the ranked output. Query-agnostic anti-drown:
    # it replaces the old keyword "is this a research query" gate (which
    # encoded one author's phrasing) — arxiv can return 20 token-matching
    # papers for a practical query, but only the top 2 survive, leaving
    # room for the repo/writeup the user actually wanted. Applies uniformly
    # to every domain, so a doc-farm or a single GitHub org can't flood
    # either. Overflow is appended after the capped set (not discarded) so
    # a thin pool still returns everything.
    per_domain: dict[str, int] = {}
    primary: list[dict] = []
    overflow: list[dict] = []
    for _, r in scored:
        dom = _extract_domain(r.get("url", ""))
        per_domain[dom] = per_domain.get(dom, 0) + 1
        (primary if per_domain[dom] <= _DOC_DOMAIN_CAP else overflow).append(r)
    result = primary + overflow
    for r in result:
        r.pop("_doc_score", None)
    return result
