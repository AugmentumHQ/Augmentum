"""Discovery Phase 2 — recommendation generator (Core / Frontier / Adjacent).

Results pass through the quality pipeline (quality.py) before reaching the user:
normalize → reject junk → score by reputation → dedup against history → rank.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import structlog

log = structlog.get_logger(__name__)

# Depth-level query modifiers. Each level has:
#   * Many variants (no single pattern repeats more than ~10% of queries)
#   * A meaningful share of bare/naked queries (empty modifier)
#   * Phrasings that mimic how humans search, not how scrapers do
#
# Why the redesign: the prior pools were short ("introduction to X",
# "beginner guide X", "tutorial X") — exactly the templated patterns
# search-engine bot detection fingerprints on. Logs from 2026-06-10
# showed all 6 SearXNG engines simultaneously suspended for 3 min after
# only a few minutes of the recommender's curator loop, because every
# query started with one of nine words. Larger pools + bare-query
# variants spread the shape across natural-search distribution.
#
# An empty string ("") in the list means "use the bare cluster name with
# no modifier" — humans frequently search for "react hooks" not
# "introduction to react hooks". Bare frequency increases with depth
# because experts search for bare topics + look for context themselves.
_DEPTH_MODIFIERS: dict[int, list[str]] = {
    1: [
        "what is", "explained", "for beginners", "basics",
        "overview", "simple terms", "in plain english",
        "how does it work", "first steps",
        "", "",
    ],
    2: [
        "tutorial", "guide", "examples", "how to",
        "learn", "step by step", "reference", "cheat sheet",
        "getting started",
        "", "",
    ],
    3: [
        "best practices", "patterns", "comparison",
        "pros and cons", "common mistakes", "tips",
        "troubleshooting", "gotchas",
        "", "", "",
    ],
    4: [
        "deep dive", "internals", "performance",
        "architecture", "under the hood", "advanced",
        "design patterns",
        "", "", "", "",
    ],
    5: [
        "production", "at scale", "case study",
        "real world", "in practice", "lessons learned",
        "", "", "", "", "",
    ],
}


def distribute_slots(total: int = 15) -> dict[str, int]:
    """Split *total* slots across the three recommendation zones.

    Roughly 60 % core, 27 % frontier, remainder adjacent (min 1 each).
    """
    core = max(1, round(total * 0.6))
    frontier = max(1, round(total * 0.27))
    adjacent = max(1, total - core - frontier)
    return {"core": core, "frontier": frontier, "adjacent": adjacent}


def build_search_query(cluster_name: str, depth_level: int = 1) -> str:
    """Build a SearXNG query from a cluster name and depth level.

    Uses time-based rotation (4 windows/day) so results vary across visits.

    Returns empty string in two cases (caller must skip the query):

      * The cluster name cleans down to nothing (HTML-only junk from
        a legacy interaction-signal row).
      * The cluster name contains an NSFW token. The curator already
        filters unsafe RESULTS, but if we built the query anyway the
        unsafe *string* would already be transmitted to upstream
        search engines — privacy + reputation leak. Dropping at the
        build step closes that gap. See augmentum/discovery/safety.py
        for the token policy.
    """
    from augmentum.discovery.safety import is_nsfw_text
    from augmentum.discovery.text_clean import clean_text_for_query

    name = clean_text_for_query(cluster_name.rstrip("."))
    if not name:
        return ""
    if is_nsfw_text(name):
        log.info("recommender_query_dropped_unsafe", reason="nsfw_token")
        return ""
    level = min(max(depth_level, 1), 5)
    modifiers = _DEPTH_MODIFIERS[level]
    window = datetime.now(UTC).hour // 6
    idx = (window + hash(name)) % len(modifiers)
    modifier = modifiers[idx]
    # Empty modifier slots ("") mean "naked query — just the topic".
    # The pool deliberately includes these so a meaningful share of
    # queries look like how humans search ("react hooks") not how
    # scrapers do ("introduction to react hooks"). Don't emit the leading
    # space in that case.
    if not modifier:
        return name
    return f"{modifier} {name}"


# ---------------------------------------------------------------------------
# SearXNG helper
# ---------------------------------------------------------------------------

async def _search_searxng(
    query: str,
    *,
    searxng_base: str,
    http_client,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Run a single SearXNG search and return raw result dicts.

    Over-fetches by 2x so the quality pipeline has room to reject junk
    and still fill the requested slot count.
    """
    try:
        resp = await http_client.get(
            f"{searxng_base}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "general",
                "pageno": random.randint(1, 3),
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning(
            "searxng_search_failed",
            query=query,
            error=str(exc)[:200],
            error_class=type(exc).__name__,
        )
        return []

    # Return raw results — quality.py handles normalization
    results: list[dict[str, Any]] = []
    fetch_limit = limit * 2  # over-fetch to compensate for pipeline rejections
    for item in (data.get("results") or [])[:fetch_limit]:
        url = item.get("url", "")
        domain = ""
        try:
            domain = urlparse(url).hostname or ""
        except (ValueError, AttributeError):
            # Malformed URL — leave domain empty so downstream filters
            # treat it as an unknown source rather than crashing.
            pass
        results.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": item.get("content", ""),
            "domain": domain,
            "thumbnail": item.get("thumbnail") or item.get("img_src") or "",
            "content_type": item.get("engine", "web"),
        })
    return results


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

async def generate_recommendations(
    store,
    searxng_base: str = "http://searxng:8080",
    total: int = 15,
    *,
    http_client=None,
    seed: int | None = None,
    domain_scores: dict[str, int] | None = None,
    hidden_urls: set[str] | None = None,
    feed_config: dict | None = None,
    user_id: str = "",
    allow_non_latin: bool = False,
    autonomous: bool = False,
) -> list[dict[str, Any]]:
    """Generate recommendations across Core / Frontier / Adjacent / Fresh zones.

    * **Core** — top clusters by frecency, depth-aware SearXNG queries.
    * **Frontier** — knowledge-gap SearXNG queries from clusters.
    * **Adjacent** — adjacent-topic SearXNG queries from clusters.
    * **Fresh** — explicit-subscribed external feeds (HN / Reddit / arXiv / RSS).

    All results pass through the quality pipeline (normalize, reject, score,
    dedup against browse history, rank by composite quality score).

    ``autonomous=True`` marks the caller as a background path (curator tick,
    UI poll). When the ``companion_autonomous_web_search_enabled`` setting
    is False (default), autonomous callers skip Core / Frontier / Adjacent
    (the SearXNG zones) and return only Fresh items from explicitly-subscribed
    feeds. This is the policy gate that closed the 2026-06-10 cascade where
    autonomous SearXNG fan-out got us bot-suspended across 6 engines.

    Explicit user requests (voice tool calls, future scheduled-briefing
    infrastructure, manual UI search) pass ``autonomous=False`` and always
    run the full pipeline.
    """
    from augmentum.config import settings as _settings
    from augmentum.discovery.quality import filter_and_rank

    autonomous_web_allowed = bool(
        getattr(_settings, "companion_autonomous_web_search_enabled", False),
    )
    # The gate is single-purpose: skip the SearXNG fan-out on autonomous
    # paths unless the operator explicitly enabled it. User-initiated
    # requests (autonomous=False) bypass entirely.
    skip_searxng_zones = autonomous and not autonomous_web_allowed
    if skip_searxng_zones:
        log.info(
            "recommender_searxng_gated",
            user_id=user_id[:16],
            reason="autonomous_off_by_setting",
        )

    if http_client is None:
        import httpx
        # Explicit timeout + scoped close. Without these a stalled SearXNG
        # call hangs this function (and any awaiting callers) indefinitely,
        # and the client leaks across calls when ``http_client`` defaults
        # to None. ``aclose`` in the finally below fires on every exit
        # path (success, exception, cancellation).
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
        _owns_http_client = True
    else:
        _owns_http_client = False

    if seed is not None:
        random.seed(seed)

    clusters = await store.list_clusters(include_dampened=False, user_id=user_id)
    # Consumption entities (kind='entity' — audiobooks, comics, shows)
    # are not web-search topics. Their titles as SearXNG queries is the
    # keyword-guessing failure this lane replaced; they get the
    # catalog-first ladder (discovery/entity_recommender.py) instead.
    clusters = [
        c for c in clusters
        if str(c.get("kind") or "topic") == "topic"
    ]
    feed_config = feed_config or {}
    have_feeds = bool(
        feed_config.get("hn")
        or feed_config.get("reddit_subs")
        or feed_config.get("arxiv_cats")
        or feed_config.get("rss_urls"),
    )
    # No clusters AND no feeds → nothing to show.
    if not clusters and not have_feeds:
        return []

    slots = distribute_slots(total)
    # Carve out a "fresh" zone for external feed items. Steal from adjacent
    # first (smallest zone), then frontier, keeping at least 1 each.
    fresh_budget = 0
    if have_feeds:
        fresh_budget = max(2, round(total * 0.2))
        steal = fresh_budget
        take = min(steal, max(0, slots["adjacent"] - 1))
        slots["adjacent"] -= take; steal -= take
        if steal:
            take = min(steal, max(0, slots["frontier"] - 1))
            slots["frontier"] -= take; steal -= take
        if steal:
            slots["core"] = max(1, slots["core"] - steal)

    # Pre-seed dedup set with user-hidden URLs so they never reappear.
    seen_urls: set[str] = set(hidden_urls or set())
    all_recs: list[dict[str, Any]] = []

    async def _fetch_and_filter(
        query: str, zone: str, cluster_id: str, cluster_name: str, limit: int,
    ) -> list[dict]:
        """Search, run quality pipeline, tag with zone/cluster metadata."""
        raw = await _search_searxng(
            query, searxng_base=searxng_base, http_client=http_client, limit=limit,
        )
        if not raw:
            return []

        # Run quality pipeline — normalizes, rejects junk, scores, deduplicates
        cleaned = await filter_and_rank(
            raw,
            store=store,
            domain_scores=domain_scores,
            seen_urls=seen_urls,
            allow_non_latin=allow_non_latin,
        )

        # Tag with zone and cluster metadata
        tagged: list[dict] = []
        for r in cleaned:
            tagged.append({
                **r,
                "zone": zone,
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
            })
        return tagged

    # --- Core ---
    # Skipped entirely on autonomous paths when the operator hasn't opted
    # in to autonomous web search. Feeds (Fresh zone) below still run.
    core_budget = slots["core"] if clusters and not skip_searxng_zones else 0
    core_clusters = clusters[:max(1, len(clusters) // 2 + 1)] if clusters else []
    random.shuffle(core_clusters)
    per_cluster = max(1, core_budget // len(core_clusters)) if core_clusters else core_budget
    for cl in core_clusters:
        if len([r for r in all_recs if r["zone"] == "core"]) >= core_budget:
            break
        query = build_search_query(cl["name"], cl.get("depth_level", 1))
        if not query:
            continue
        tagged = await _fetch_and_filter(
            query, "core", cl["cluster_id"], cl["name"], per_cluster + 2,
        )
        all_recs.extend(tagged[:per_cluster])

    # --- Frontier (knowledge gaps) ---
    # NSFW gate applies the same outbound-query safety rule as
    # build_search_query above; gap / adjacent queries don't go through
    # that helper so the check is inline.
    from augmentum.discovery.safety import is_nsfw_text
    from augmentum.discovery.text_clean import clean_text_for_query

    # Same autonomous-gate check as Core.
    frontier_budget = slots["frontier"] if not skip_searxng_zones else 0
    for cl in clusters:
        if len([r for r in all_recs if r["zone"] == "frontier"]) >= frontier_budget:
            break
        gaps = cl.get("knowledge_gaps") or ""
        if not gaps:
            continue
        gap_query = clean_text_for_query(gaps.split(",")[0])
        if not gap_query:
            continue
        if is_nsfw_text(gap_query):
            log.info(
                "recommender_query_dropped_unsafe",
                reason="nsfw_token", zone="frontier",
            )
            continue
        tagged = await _fetch_and_filter(
            gap_query, "frontier", cl["cluster_id"], cl["name"], 3,
        )
        all_recs.extend(tagged[:2])

    # --- Adjacent ---
    # Same autonomous-gate check as Core / Frontier.
    adjacent_budget = slots["adjacent"] if not skip_searxng_zones else 0
    for cl in clusters:
        if len([r for r in all_recs if r["zone"] == "adjacent"]) >= adjacent_budget:
            break
        adj = cl.get("adjacent_topics") or ""
        if not adj:
            continue
        adj_query = clean_text_for_query(adj.split(",")[0])
        if not adj_query:
            continue
        if is_nsfw_text(adj_query):
            log.info(
                "recommender_query_dropped_unsafe",
                reason="nsfw_token", zone="adjacent",
            )
            continue
        tagged = await _fetch_and_filter(
            adj_query, "adjacent", cl["cluster_id"], cl["name"], 3,
        )
        all_recs.extend(tagged[:2])

    # --- Fresh (external feeds: HN, Reddit, arxiv, RSS) ---
    if fresh_budget > 0 and have_feeds:
        try:
            from augmentum.discovery.feeds import gather_feeds
            from augmentum.discovery.quality import filter_and_rank

            feed_raw = await gather_feeds(
                http_client,
                hn=bool(feed_config.get("hn")),
                reddit_subs=feed_config.get("reddit_subs") or [],
                arxiv_cats=feed_config.get("arxiv_cats") or [],
                rss_urls=feed_config.get("rss_urls") or [],
            )
            if feed_raw:
                random.shuffle(feed_raw)
                cleaned = await filter_and_rank(
                    feed_raw,
                    store=store,
                    domain_scores=domain_scores,
                    seen_urls=seen_urls,
                    allow_non_latin=allow_non_latin,
                )
                for r in cleaned[:fresh_budget]:
                    all_recs.append({
                        **r,
                        "zone": "fresh",
                        "cluster_id": "",
                        "cluster_name": r.get("_feed_source", "fresh"),
                    })
        except Exception as exc:
            log.warning("fresh_zone_failed", error=str(exc))

    log.info(
        "recommendations_generated",
        total=len(all_recs),
        core=len([r for r in all_recs if r["zone"] == "core"]),
        frontier=len([r for r in all_recs if r["zone"] == "frontier"]),
        adjacent=len([r for r in all_recs if r["zone"] == "adjacent"]),
        fresh=len([r for r in all_recs if r["zone"] == "fresh"]),
    )

    if _owns_http_client:
        await http_client.aclose()
    return all_recs[:total]
