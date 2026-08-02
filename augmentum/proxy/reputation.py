"""Domain reputation system — tracks quality scores for fetched domains."""

from __future__ import annotations

from datetime import UTC
from urllib.parse import urlparse

from fastapi import Request

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _get_db(request: Request):
    """Get the SQLite connection from app state."""
    store = getattr(request.app.state, "settings_store", None)
    if store:
        return store._conn
    return None


_reputation_sync_counter = 0

async def _update_reputation(
    request: Request,
    url: str,
    *,
    success: bool = False,
    junk: bool = False,
    user_action: bool = False,
    structured_data: bool = False,
) -> None:
    """Update domain reputation score after a fetch or user action."""
    db = await _get_db(request)
    if not db:
        return

    hostname = urlparse(url).hostname
    if not hostname:
        return
    # Normalize: strip www.
    domain = hostname.lower().removeprefix("www.")

    try:
        # Upsert the domain
        await db.execute(
            """INSERT INTO domain_reputation (domain, score, fetch_count, success_count,
                   fail_count, user_action_count, last_fetched, created_at)
               VALUES (?, 0, 0, 0, 0, 0, datetime('now'), datetime('now'))
               ON CONFLICT(domain) DO NOTHING""",
            (domain,),
        )

        # Calculate score delta
        delta = 0
        updates = []

        if success:
            delta += 1
            updates.append("fetch_count = fetch_count + 1")
            updates.append("success_count = success_count + 1")
            updates.append("last_fetched = datetime('now')")
        elif junk:
            delta -= 2
            updates.append("fetch_count = fetch_count + 1")
            updates.append("fail_count = fail_count + 1")
            updates.append("last_fetched = datetime('now')")
        else:
            # Thin content
            delta -= 1
            updates.append("fetch_count = fetch_count + 1")
            updates.append("fail_count = fail_count + 1")
            updates.append("last_fetched = datetime('now')")

        if structured_data:
            delta += 1  # bonus for having JSON-LD/AMP/RSS

        if user_action:
            delta += 3
            updates.append("user_action_count = user_action_count + 1")
            updates.append("last_action = datetime('now')")

        updates.append(f"score = MIN(score + {delta}, 50)")  # cap at 50

        sql = f"UPDATE domain_reputation SET {', '.join(updates)} WHERE domain = ?"
        await db.execute(sql, (domain,))
        await db.commit()
    except Exception:
        log.debug("reputation_update_failed", domain=domain, exc_info=True)
        return

    # Periodically sync learned reputation to the web tool's quality lookup
    global _reputation_sync_counter
    _reputation_sync_counter += 1
    if _reputation_sync_counter % 20 == 0:
        try:
            all_scores = await _get_domain_scores(request)
            if all_scores:
                from augmentum.tools.preferred_sources import merge_learned_reputation
                merge_learned_reputation(all_scores)
        except Exception as exc:
            log.debug("reputation_sync_merge_failed", error=str(exc))


async def _seed_preferred_sources(request: Request) -> None:
    """Seed the reputation table with curated preferred sources.

    EXCELLENT sources start at +5 (immediately in the top tier).
    GOOD sources start at +2.
    AVOID sources start at -3.
    Only seeds domains that don't already have a row (won't overwrite
    learned scores from actual usage).
    """
    db = await _get_db(request)
    if not db:
        return
    try:
        from augmentum.tools.preferred_sources import _SOURCES, AVOID, EXCELLENT, GOOD

        score_map = {EXCELLENT: 5, GOOD: 2, AVOID: -3}

        for domain, info in _SOURCES.items():
            score = score_map.get(info.quality, 0)
            await db.execute(
                """INSERT INTO domain_reputation (domain, score, created_at)
                   VALUES (?, ?, datetime('now'))
                   ON CONFLICT(domain) DO NOTHING""",
                (domain.lower(), score),
            )
        await db.commit()
        log.info("preferred_sources_seeded", count=len(_SOURCES))
    except Exception:
        log.debug("preferred_sources_seed_failed", exc_info=True)

    # Also run daily decay while we're here
    await _maybe_decay_scores(request)


async def _maybe_decay_scores(request: Request) -> None:
    """Decay reputation scores toward neutral once per day.

    High scores (>5) decrease by 1. Low scores (<-3) increase by 1.
    Scores in the -3 to 5 range are stable (the 'neutral zone').
    This prevents stale scores from dominating and lets sites
    recover from temporary issues.
    """
    db = await _get_db(request)
    if not db:
        return

    try:
        # Check when decay last ran
        cursor = await db.execute(
            "SELECT value FROM app_settings WHERE key = 'reputation_last_decay'"
        )
        row = await cursor.fetchone()

        if row:
            from datetime import datetime, timedelta
            last_decay = datetime.fromisoformat(row[0])
            now = datetime.now(UTC)
            if now - last_decay < timedelta(days=1):
                return  # too soon

        # Decay high scores
        await db.execute(
            "UPDATE domain_reputation SET score = score - 1 WHERE score > 5"
        )
        # Decay low scores
        await db.execute(
            "UPDATE domain_reputation SET score = score + 1 WHERE score < -3"
        )

        # Record decay time
        await db.execute(
            """INSERT INTO app_settings (key, value, updated_at)
               VALUES ('reputation_last_decay', datetime('now'), datetime('now'))
               ON CONFLICT(key) DO UPDATE SET value = datetime('now'), updated_at = datetime('now')"""
        )
        await db.commit()
        log.info("reputation_scores_decayed")
    except Exception:
        log.debug("reputation_decay_failed", exc_info=True)


async def _get_domain_scores(request: Request) -> dict[str, int]:
    """Get all domain scores as a dict for search result ranking."""
    db = await _get_db(request)
    if not db:
        return {}
    try:
        cursor = await db.execute("SELECT domain, score FROM domain_reputation")
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


# Platforms we can embed inline (ad-free player, transcript sync, etc.)
_EMBEDDABLE_VIDEO_DOMAINS: set[str] = {
    "youtube.com", "youtu.be", "m.youtube.com",
    "vimeo.com", "player.vimeo.com",
    "dailymotion.com", "dai.ly",
    "tiktok.com",
    "twitch.tv", "clips.twitch.tv",
}


# Tiny stopword set for query tokenisation. Kept inline rather than pulled
# from a corpus file so the reranker has no extra load-time cost; these are
# the high-frequency words most likely to cause false category matches.
_QUERY_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "but", "with", "from", "into", "onto", "this", "that",
    "these", "those", "what", "when", "where", "which", "while", "who", "whom",
    "how", "why", "can", "does", "did", "will", "would", "should", "could",
    "are", "was", "were", "been", "being", "have", "has", "had", "its",
    "tell", "show", "find", "get", "about", "some", "more", "most", "best",
    "good", "bad", "any", "all", "new", "old", "top", "info", "information",
    "help", "please", "thanks", "thank",
})

# Domains that provide truly general-purpose reference coverage across almost
# any topic. These always qualify for the tier-0 boost regardless of whether
# their SourceInfo categories overlap the query — a query like "polar bears"
# has no topic a curated category can perfectly match, but Wikipedia is still
# the right answer. Keep this list extremely small: news brands and topical
# magazines should still prove relevance through categories/query matching.
_GENERALIST_DOMAINS: frozenset[str] = frozenset({
    "wikipedia.org",
    "britannica.com",
})


def _tokenize_query(query: str) -> set[str]:
    """Extract topic-candidate tokens from a search query.

    Lowercase, strip non-word characters, drop stopwords and <3-char
    tokens. The result is compared against SourceInfo.categories to decide
    whether a domain is topically relevant. Empty set means "no signal" —
    callers fall back to the plain reputation sort.
    """
    if not query:
        return set()
    import re
    tokens = re.findall(r"[a-z]{3,}", query.lower())
    return {t for t in tokens if t not in _QUERY_STOPWORDS}


def _domain_matches_query(domain: str, query_tokens: set[str]) -> bool:
    """True if the domain's curated categories overlap with query tokens,
    or if the domain is a recognised generalist reference source.

    Deferred import of preferred_sources — reputation.py is imported at
    request time and we don't want to pull a 5 kLOC module in unless we
    actually rerank.
    """
    if not domain:
        return False
    if not query_tokens:
        # No query signal — treat every EXCELLENT domain as a potential
        # match so we don't break the pre-existing behaviour when a caller
        # opts out of query-aware ranking.
        return True
    # Generalist allowlist covers the common case (Wikipedia et al.)
    # without requiring a category match.
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in _GENERALIST_DOMAINS:
            return True
    try:
        from augmentum.tools.preferred_sources import get_source_info
    except Exception:
        return True  # module unavailable — fail open, don't over-restrict
    info = get_source_info(f"https://{domain}/")
    if not info or not info.categories:
        return False
    # Substring match either direction: "bear" matches category "biology"?
    # No — we want "biology" to match queries containing "biology" or
    # "biological". Keep the matcher exact-ish: token equals category, OR
    # one contains the other as a word-boundary prefix/suffix to catch
    # "physics"/"astrophysics" style drift.
    for category in info.categories:
        cat = category.lower()
        if cat in query_tokens:
            return True
        for token in query_tokens:
            if cat.startswith(token) or token.startswith(cat):
                return True
    return False


def _apply_domain_diversity_cap(
    results: list[dict],
    *,
    max_per_domain: int = 2,
    top_n: int = 10,
    bypass_domains: frozenset[str] | set[str] | None = None,
) -> list[dict]:
    """Prevent any single domain from monopolising the top N results.

    Once a domain has contributed `max_per_domain` results to the first
    `top_n` slots, further results from that same domain are pushed
    further down the list. Preserves relative order otherwise. Gives
    users a visible mix of sources even when one provider dominates the
    raw ranking — critical for queries where the curated preferred-source
    list boosts a single domain (e.g. arxiv across science queries).

    `bypass_domains`: roots in this set are exempt from the cap. Used by
    the video pipeline to let YouTube / Vimeo / Twitch dominate video
    search the way users expect — the diversity-cap reasoning that
    applies to text search (no arxiv-monoculture for biology queries)
    inverts here: users want the canonical video host, not "diverse"
    video sources.
    """
    if max_per_domain <= 0 or top_n <= 0:
        return results
    bypass = bypass_domains or frozenset()
    kept: list[dict] = []
    overflow: list[dict] = []
    domain_count: dict[str, int] = {}
    for item in results:
        url = item.get("url", "")
        hostname = urlparse(url).hostname or ""
        domain = hostname.lower().removeprefix("www.")
        # Collapse subdomains so api.x.com and x.com share a budget
        parts = domain.split(".")
        root = ".".join(parts[-2:]) if len(parts) > 2 else domain
        if root in bypass or domain in bypass:
            kept.append(item)
            continue
        if len(kept) < top_n and domain_count.get(root, 0) >= max_per_domain:
            overflow.append(item)
            continue
        kept.append(item)
        domain_count[root] = domain_count.get(root, 0) + 1
    return kept + overflow


def _rank_results(
    results: list[dict],
    scores: dict[str, int],
    *,
    boost_embeddable_video: bool = False,
    prefer_english: bool = False,
    query: str | None = None,
) -> list[dict]:
    """Re-rank search results by domain reputation, topic relevance, and
    cross-domain diversity.

    Tiers (stable-sorted, so SearXNG order breaks ties within a tier):
      0 — EXCELLENT reputation AND topically relevant to the query
      0 — embeddable video platform (when boost_embeddable_video)
      1 — GOOD, UNKNOWN, or EXCELLENT-but-off-topic
      2 — bad reputation (learned negative score)

    Demoting off-topic EXCELLENT sources fixes the "arxiv for polar bears"
    failure mode: arxiv is curated as EXCELLENT but its categories are
    ("science", "research", "papers", "physics", "math", "cs"). A biology
    query doesn't match any of those, so arxiv falls to tier 1 and gives
    space to Wikipedia / National Geographic / Britannica.

    After tier sort, a same-domain diversity cap (max 2 per root domain
    in the top 10) prevents a single provider from dominating visible
    results. Downstream order is preserved otherwise.
    """
    if not scores and not boost_embeddable_video and not query and not prefer_english:
        return results

    query_tokens = _tokenize_query(query or "")

    _is_non_english = None
    if prefer_english:
        # Imported lazily so the ranker stays usable when discovery.quality
        # isn't importable (tests, bare-proxy builds).
        try:
            from augmentum.discovery.quality import _is_non_english as _ine
            _is_non_english = _ine
        except Exception:
            _is_non_english = None

    def _lookup_score(domain: str) -> int | None:
        """Find the reputation score for a domain, walking up subdomains.

        `en.wikipedia.org` should inherit `wikipedia.org`'s score — the
        curated list keys by root domain but real fetches often land on
        a language or shard subdomain. Without this walk-up, half the
        Wikipedia corpus shows up as "unknown" and falls to middle tier.
        """
        if domain in scores:
            return scores[domain]
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            candidate = ".".join(parts[i:])
            if candidate in scores:
                return scores[candidate]
        return None

    def _tier(item) -> int:
        url = item.get("url", "")
        hostname = urlparse(url).hostname or ""
        domain = hostname.lower().removeprefix("www.")

        # For video searches, boost platforms we can embed inline
        if boost_embeddable_video:
            base = domain
            # Check both the domain and its parent (e.g. clips.twitch.tv → twitch.tv)
            parts = base.split(".")
            parent = ".".join(parts[-2:]) if len(parts) > 2 else base
            if base in _EMBEDDABLE_VIDEO_DOMAINS or parent in _EMBEDDABLE_VIDEO_DOMAINS:
                return 0  # top tier — embeddable

        score = _lookup_score(domain)
        if score is None:
            return 1  # unknown — middle
        if score >= 5:
            # Only boost to tier 0 when the source topically fits the
            # query; otherwise off-topic "excellent" sites (arxiv, pubmed,
            # semanticscholar for a non-research query) fall to middle
            # tier alongside other unknowns.
            if query_tokens and not _domain_matches_query(domain, query_tokens):
                return 1
            return 0
        if score >= 0:
            return 1
        return 2

    def _sort_key(item):
        tier = _tier(item)
        # English-first as a soft sub-sort so non-English content stays
        # visible (just below) rather than being dropped — unlike
        # filter_for_video_ui which hard-rejects non-Latin titles.
        if _is_non_english is not None:
            title = item.get("title", "") or ""
            return (tier, 1 if _is_non_english(title) else 0)
        return (tier,)

    ranked = sorted(results, key=_sort_key)
    # In video category, exempt embeddable video hosts from the
    # diversity cap — users expect YouTube to dominate video search,
    # the same way they expect Wikipedia to dominate factual lookups.
    bypass = _EMBEDDABLE_VIDEO_DOMAINS if boost_embeddable_video else None
    return _apply_domain_diversity_cap(ranked, bypass_domains=bypass)
