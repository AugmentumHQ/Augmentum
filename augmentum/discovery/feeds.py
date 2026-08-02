"""External feed ingestion for Discovery — HN, Reddit, arxiv, RSS.

Pulls from high-signal non-SearXNG sources and normalizes each result into
the same dict shape SearXNG returns, so the quality pipeline can score and
rank them alongside cluster-based recommendations.

Results are cached in-memory (per-source TTL) to avoid rate limits and
make repeat /for-you calls snappy.
"""
from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ─── in-memory TTL cache ────────────────────────────────────────────────
# Stores (value, stored_at, ttl) so successes and failures can use different
# windows: a working fetch is good for 20min, but a transient outage gets a
# 60s negative-cache entry so we don't stampede an upstream on every poll
# while it's down. Without this, the curator's parallel topic poller can
# fire a dozen requests against a 301-ing endpoint in a single tick — see
# the arxiv http→https regression that prompted this design.
_cache: dict[str, tuple[list[dict], float, float]] = {}
_CACHE_TTL = 20 * 60        # 20 min — success default
_NEG_CACHE_TTL = 60         # 60s — failure cooldown
_UA = "Augmentum/1.0 (discovery feed ingest)"


def _cache_get(key: str) -> list[dict] | None:
    entry = _cache.get(key)
    if not entry:
        return None
    value, stored_at, ttl = entry
    if (monotonic() - stored_at) < ttl:
        return value
    _cache.pop(key, None)
    return None


def _cache_set(key: str, value: list[dict], ttl: float = _CACHE_TTL) -> None:
    _cache[key] = (value, monotonic(), ttl)


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _normalize(
    *, url: str, title: str, snippet: str, source_label: str,
    thumbnail: str = "",
) -> dict[str, Any]:
    """Shape items to match what _search_searxng() produces."""
    return {
        "title": (title or "").strip(),
        "url": (url or "").strip(),
        "snippet": (snippet or "").strip()[:500],
        "domain": _domain_of(url),
        "thumbnail": thumbnail,
        "content_type": "web",
        "_feed_source": source_label,
    }


# ─── Hacker News (Algolia) ──────────────────────────────────────────────

async def fetch_hn_top(http_client, *, limit: int = 15) -> list[dict[str, Any]]:
    """HN front-page by points via Algolia search API (no key required)."""
    key = f"hn:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        resp = await http_client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": "front_page", "hitsPerPage": limit},
            timeout=10.0,
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("hn_fetch_failed", error=str(exc))
        _cache_set(key, [], _NEG_CACHE_TTL)
        return []

    out: list[dict] = []
    for hit in data.get("hits", []):
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        title = hit.get("title") or hit.get("story_title") or ""
        if not title:
            continue
        points = hit.get("points") or 0
        comments = hit.get("num_comments") or 0
        snippet = f"{points} points · {comments} comments on Hacker News"
        out.append(_normalize(
            url=url, title=title, snippet=snippet, source_label="hn",
        ))
    _cache_set(key, out)
    return out


# ─── Reddit (public .json) ──────────────────────────────────────────────

async def fetch_reddit(http_client, subs: list[str], *, limit: int = 8) -> list[dict[str, Any]]:
    """Public subreddit JSON listings. No auth required for read-only."""
    subs = [s.strip().lstrip("/").removeprefix("r/") for s in subs if s.strip()]
    if not subs:
        return []
    key = f"reddit:{','.join(sorted(subs))}:{limit}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    async def _one(sub: str) -> list[dict]:
        try:
            resp = await http_client.get(
                f"https://www.reddit.com/r/{sub}/hot.json",
                params={"limit": limit, "raw_json": 1},
                timeout=10.0,
                headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        items = []
        for child in (data.get("data", {}).get("children") or []):
            d = child.get("data", {})
            # Prefer the linked article; fall back to the self-post
            url = d.get("url_overridden_by_dest") or d.get("url") or ""
            if not url or url.startswith("/r/"):
                continue
            title = d.get("title", "")
            if not title:
                continue
            score = d.get("score") or 0
            comments = d.get("num_comments") or 0
            selftext = (d.get("selftext") or "").strip()
            snippet = selftext[:300] if selftext else f"{score} upvotes · {comments} comments on r/{sub}"
            thumb = d.get("thumbnail") or ""
            if thumb in ("self", "default", "nsfw", "spoiler", ""):
                thumb = ""
            items.append(_normalize(
                url=url, title=title, snippet=snippet,
                source_label=f"reddit:{sub}", thumbnail=thumb,
            ))
        return items

    results = await asyncio.gather(*[_one(s) for s in subs], return_exceptions=False)
    merged: list[dict] = []
    for r in results:
        merged.extend(r)
    _cache_set(key, merged)
    return merged


# ─── arxiv (Atom) ───────────────────────────────────────────────────────

_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


async def fetch_arxiv(
    http_client,
    categories: list[str],
    *,
    limit: int = 50,
    terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """arxiv recent papers in the given categories.

    Optional ``terms`` add a topic filter so results match the category set
    AND at least one of the terms (full-text across all fields). This is
    the difference between "latest 50 in cs.AI" (firehose — score-filter
    locally and pray) and "latest 50 in cs.AI mentioning speculative
    decoding" (targeted — arxiv does the relevance work, local score is
    a sanity check). The targeted form has dramatically higher hit rate
    for specific topics because arxiv's full-text index is far better than
    our crude keyword-overlap tokenizer.

    Terms are sanitized (alphanumeric + space + hyphen only), lowercased,
    de-duplicated, and capped at 8 to keep the query string sane. Empty
    or sub-3-char terms are dropped so single-letter remnants from
    tokenization don't blow up the query.
    """
    cats = [c.strip() for c in categories if c.strip()]
    if not cats:
        return []

    seen_terms: set[str] = set()
    term_list: list[str] = []
    for raw in (terms or []):
        cleaned = re.sub(r"[^\w\s-]", "", str(raw)).strip().lower()
        if len(cleaned) < 3 or cleaned in seen_terms:
            continue
        seen_terms.add(cleaned)
        term_list.append(cleaned)
        if len(term_list) >= 8:
            break

    key = f"arxiv:{','.join(sorted(cats))}:{limit}:{','.join(sorted(term_list))}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    cat_clause = "+OR+".join(f"cat:{c}" for c in cats)
    if term_list:
        # Multi-word terms are split rather than phrase-quoted — arxiv's
        # full-text index returns more useful candidates when each word
        # contributes independently. Local score still filters down to
        # topic-coherent ones after.
        atomic: list[str] = []
        for t in term_list:
            atomic.extend(t.split())
        atomic = [w for w in atomic if len(w) >= 3][:12]
        terms_clause = "+OR+".join(f"all:{w}" for w in atomic)
        query = f"({cat_clause})+AND+({terms_clause})"
    else:
        query = cat_clause

    try:
        resp = await http_client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": limit,
            },
            timeout=15.0,
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
    except Exception as exc:
        log.warning("arxiv_fetch_failed", error=f"{type(exc).__name__}: {exc}".rstrip(": "))
        _cache_set(key, [], _NEG_CACHE_TTL)
        return []

    out: list[dict] = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        _cache_set(key, [], _NEG_CACHE_TTL)
        return []

    for entry in root.findall("a:entry", _ATOM_NS):
        title_el = entry.find("a:title", _ATOM_NS)
        summary_el = entry.find("a:summary", _ATOM_NS)
        link_el = entry.find("a:id", _ATOM_NS)
        if title_el is None or link_el is None:
            continue
        title = re.sub(r"\s+", " ", (title_el.text or "").strip())
        summary = re.sub(r"\s+", " ", (summary_el.text or "").strip()) if summary_el is not None else ""
        url = (link_el.text or "").strip().replace("http://", "https://")
        if not title or not url:
            continue
        out.append(_normalize(
            url=url, title=title, snippet=summary[:400],
            source_label="arxiv",
        ))

    _cache_set(key, out)
    return out


# ─── Generic RSS/Atom ───────────────────────────────────────────────────

_RSS_ITEM_TAGS = ("item", "{http://www.w3.org/2005/Atom}entry")


def _expand_rsshub(feed_urls: list[str]) -> list[str]:
    """Expand ``rsshub://route/path`` shorthands against the configured
    RSSHub base URL (compose.rsshub overlay). Lets a subscription read
    ``rsshub://github/release/owner/repo`` instead of hardcoding the
    service address. Unknown base (setting emptied) drops the entry —
    a shorthand without a resolver isn't a fetchable URL anyway.
    """
    try:
        from augmentum.config import settings
        base = (getattr(settings, "rsshub_base_url", "") or "").rstrip("/")
    except Exception:  # noqa: BLE001 — config unavailable in some test rigs
        base = ""
    out: list[str] = []
    for url in feed_urls:
        url = (url or "").strip()
        if url.lower().startswith("rsshub://"):
            if base:
                out.append(f"{base}/{url[len('rsshub://'):].lstrip('/')}")
        else:
            out.append(url)
    return out


async def fetch_rss(http_client, feed_urls: list[str], *, per_feed: int = 5) -> list[dict[str, Any]]:
    """Parse user-supplied RSS/Atom feeds using stdlib XML."""
    feed_urls = _expand_rsshub(feed_urls)
    feed_urls = [u.strip() for u in feed_urls if u.strip().startswith(("http://", "https://"))]
    if not feed_urls:
        return []
    key = f"rss:{','.join(sorted(feed_urls))}:{per_feed}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    async def _one(url: str) -> list[dict]:
        try:
            resp = await http_client.get(
                url, timeout=10.0, headers={"User-Agent": _UA},
            )
            resp.raise_for_status()
        except Exception:
            return []
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return []

        items: list[dict] = []
        # RSS 2.0: <channel><item>; Atom: <feed><entry>
        for tag in _RSS_ITEM_TAGS:
            for el in root.iter(tag):
                title = _text(el, "title") or _text(el, "{http://www.w3.org/2005/Atom}title")
                link = _text(el, "link") or _atom_link(el)
                desc = (
                    _text(el, "description")
                    or _text(el, "{http://www.w3.org/2005/Atom}summary")
                    or _text(el, "{http://www.w3.org/2005/Atom}content")
                    or ""
                )
                # Strip HTML from descriptions
                desc = re.sub(r"<[^>]+>", " ", desc)
                desc = re.sub(r"\s+", " ", desc).strip()
                if not title or not link:
                    continue
                items.append(_normalize(
                    url=link, title=title, snippet=desc[:400],
                    source_label=f"rss:{_domain_of(url)}",
                ))
                if len(items) >= per_feed:
                    break
            if items:
                break
        return items

    results = await asyncio.gather(*[_one(u) for u in feed_urls], return_exceptions=False)
    merged: list[dict] = []
    for r in results:
        merged.extend(r)
    _cache_set(key, merged)
    return merged


def _text(el, tag: str) -> str:
    child = el.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def _atom_link(el) -> str:
    """Atom <link href="..."/> form."""
    for child in el.findall("{http://www.w3.org/2005/Atom}link"):
        href = child.get("href")
        if href:
            return href
    return ""


# ─── Top-level collector ────────────────────────────────────────────────

async def gather_feeds(
    http_client,
    *,
    hn: bool = False,
    reddit_subs: list[str] | None = None,
    arxiv_cats: list[str] | None = None,
    rss_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all configured feed fetchers in parallel and merge results."""
    tasks: list = []
    if hn:
        tasks.append(fetch_hn_top(http_client))
    if reddit_subs:
        tasks.append(fetch_reddit(http_client, reddit_subs))
    if arxiv_cats:
        tasks.append(fetch_arxiv(http_client, arxiv_cats))
    if rss_urls:
        tasks.append(fetch_rss(http_client, rss_urls))

    if not tasks:
        return []

    batches = await asyncio.gather(*tasks, return_exceptions=True)
    merged: list[dict] = []
    for b in batches:
        if isinstance(b, Exception):
            continue
        merged.extend(b)
    log.info("discovery_feeds_fetched", count=len(merged))
    return merged
