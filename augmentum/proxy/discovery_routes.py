"""Discovery Engine API — signals, history, knowledge search."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.config import settings
from augmentum.discovery.clustering import (
    assign_signal_to_cluster,
    compose_narration,
    extract_signal_text,
)
from augmentum.discovery.entities import (
    ENTITY_SIGNAL_TYPES as _ENTITY_SIGNAL_TYPES,
)
from augmentum.discovery.frecency import compute_frecency_from_signals
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

_LIBRARY_SIGNAL_TYPES = {"page_visit", "video_watch", "ai_action", "note_save"}

# Signal types that should bump domain reputation as a positive user action.
# Hides are NOT included — they're negative (tracked via list_hidden_urls).
_REPUTATION_BUMP_TYPES = {"discovery_click", "discovery_favorite"}

# Signal types that feed the History view. The original triple
# (page_visit / video_open / video_watch) covered only web content;
# the four new types capture media-server consumption so audiobook,
# comic, movie, show, and podcast plays surface in History too.
# Each player POSTs one of these on first-play of a new item with
# source_url=``augm:media:<file_id>`` and metadata.cover_url for the
# thumbnail.
_HISTORY_SIGNAL_TYPES = (
    "page_visit", "video_open", "video_watch",
    "media_play", "comic_read",
)

# Synthetic-URL prefix for media-server items in browse_history. The
# real URL doesn't exist (the item is a file_id, not a web page), so
# we mint one. Matched against the start of source_url to route
# domain extraction + thumbnail logic.
_MEDIA_URL_PREFIX = "augm:media:"


def _history_domain_and_thumb(*, source_url: str, content_type: str, meta: dict) -> tuple[str, str]:
    """Return (domain, thumbnail) for a history row, branching on URL kind.

    Web URLs get hostname-derived domains and YouTube-derived thumbnails
    (existing behavior). Synthetic ``augm:media:<file_id>`` URLs get an
    empty domain and the cover_url passed in metadata — covers come
    from /api/media/cover/<file_id> via the player module that emitted
    the signal.
    """
    if source_url.startswith(_MEDIA_URL_PREFIX):
        return "", str(meta.get("cover_url") or "")
    from urllib.parse import urlparse
    hostname = urlparse(source_url).hostname or ""
    domain = hostname.lower().removeprefix("www.")
    thumb = ""
    if content_type == "video" and meta.get("video_id"):
        thumb = f"https://img.youtube.com/vi/{meta['video_id']}/hqdefault.jpg"
    return domain, thumb


def _get_store(request: Request):
    return getattr(request.app.state, "discovery_store", None)


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


# ---------------------------------------------------------------------------
# Feed config (user-configurable external sources for the "fresh" zone)
# ---------------------------------------------------------------------------

# Sensible first-run defaults — HN on, everything else opt-in.
_FEED_DEFAULTS = {
    "discovery_feeds_hn": "1",
    "discovery_feeds_reddit": "",        # comma-sep subreddits
    "discovery_feeds_arxiv": "",         # comma-sep categories (e.g. cs.AI,cs.LG)
    "discovery_feeds_rss": "",           # comma-sep RSS URLs
}


async def _load_feed_config(request: Request) -> dict:
    """Read feed configuration from settings_store, scoped to the caller.

    Multi-tenant: resolves each tenant's own feed prefs (user override →
    install-wide → default) so one user's discovery sources never show
    up in another's panel.
    """
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return {"hn": True, "reddit_subs": [], "arxiv_cats": [], "rss_urls": []}

    uid = _user_id(request)

    async def _get(key: str, default: str) -> str:
        try:
            val = (
                await store.get_user_or_global(uid, key)
                if uid else await store.get(key)
            )
            return val if val is not None else default
        except Exception:
            return default

    hn_raw = await _get("discovery_feeds_hn", _FEED_DEFAULTS["discovery_feeds_hn"])
    reddit_raw = await _get("discovery_feeds_reddit", _FEED_DEFAULTS["discovery_feeds_reddit"])
    arxiv_raw = await _get("discovery_feeds_arxiv", _FEED_DEFAULTS["discovery_feeds_arxiv"])
    rss_raw = await _get("discovery_feeds_rss", _FEED_DEFAULTS["discovery_feeds_rss"])

    return {
        "hn": str(hn_raw).strip() in ("1", "true", "True", "yes"),
        "reddit_subs": [s.strip() for s in (reddit_raw or "").split(",") if s.strip()],
        "arxiv_cats": [s.strip() for s in (arxiv_raw or "").split(",") if s.strip()],
        "rss_urls": [s.strip() for s in (rss_raw or "").split(",") if s.strip()],
    }


# ---------------------------------------------------------------------------
# Feed sources (the For-You input config — previously settings-store
# keys with NO edit path; the fetchers read them but nothing wrote
# them. RSS accepts rsshub:// shorthands when the overlay is running.)
# ---------------------------------------------------------------------------


@router.get("/feeds")
async def get_feeds(request: Request) -> JSONResponse:
    """Current feed source configuration."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    cfg = await _load_feed_config(request)
    return JSONResponse(cfg)


@router.put("/feeds")
async def put_feeds(request: Request) -> JSONResponse:
    """Update feed sources. Body: {hn: bool, reddit_subs: [..],
    arxiv_cats: [..], rss_urls: [..]}. Lists are normalized (trimmed,
    deduped, capped) before persisting to the settings store."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        return JSONResponse(
            {"error": "settings store unavailable"}, status_code=503,
        )
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    def _clean_list(key: str, *, cap: int, max_len: int) -> list[str]:
        raw = body.get(key)
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            s = str(item or "").strip()[:max_len]
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
            if len(out) >= cap:
                break
        return out

    reddit_subs = _clean_list("reddit_subs", cap=24, max_len=64)
    arxiv_cats = _clean_list("arxiv_cats", cap=24, max_len=32)
    rss_urls = [
        u for u in _clean_list("rss_urls", cap=48, max_len=512)
        if u.lower().startswith(("http://", "https://", "rsshub://"))
    ]
    hn = bool(body.get("hn"))

    try:
        # Per-user (multi-tenant): each tenant's discovery panel reads
        # their own feeds. Mirror to the install-wide keys too so the
        # companion curator's autonomous "for you" pass — which reads the
        # global keys for the runtime owner — keeps working unchanged.
        for _key, _val in (
            ("discovery_feeds_hn", "1" if hn else "0"),
            ("discovery_feeds_reddit", ",".join(reddit_subs)),
            ("discovery_feeds_arxiv", ",".join(arxiv_cats)),
            ("discovery_feeds_rss", ",".join(rss_urls)),
        ):
            await store.set_user(uid, _key, _val)
            await store.set(_key, _val)
    except Exception:
        log.warning("discovery_feeds_save_failed", exc_info=True)
        return JSONResponse({"error": "failed to save"}, status_code=500)

    log.info(
        "discovery_feeds_updated",
        user_id=uid, hn=hn, reddit=len(reddit_subs),
        arxiv=len(arxiv_cats), rss=len(rss_urls),
    )
    return JSONResponse({
        "hn": hn, "reddit_subs": reddit_subs,
        "arxiv_cats": arxiv_cats, "rss_urls": rss_urls,
    })


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


@router.post("/signal")
async def log_signal(request: Request) -> JSONResponse:
    """Log an interaction signal and optionally distill content into the library."""
    if not settings.discovery_enabled:
        return JSONResponse({"status": "disabled"})

    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    signal_type = body.get("signal_type")
    if not signal_type:
        return JSONResponse({"error": "signal_type is required"}, status_code=400)

    source_url = body.get("source_url", "")
    source_title = body.get("source_title", "")
    content_type = body.get("content_type", "")

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    result = await store.log_signal(
        signal_type=signal_type,
        source_url=source_url,
        source_title=source_title,
        content_type=content_type,
        weight=float(body.get("weight", 1.0)),
        metadata=body.get("metadata") or {},
        user_id=uid,
    )

    # Cluster assignment
    signal_id = result.get("id")
    title = body.get("source_title", "")
    metadata = body.get("metadata") or {}
    weight = float(body.get("weight", 1.0))
    if signal_id and not result.get("deduplicated"):
        # Consumption signals resolve to series-level ENTITY clusters
        # via the catalog (the typed hop — spec 2026-06-12), never to
        # topic clusters built from title keywords. Falls through to
        # topic clustering only when entity resolution fully misses.
        entity_clustered = False
        if signal_type in _ENTITY_SIGNAL_TYPES:
            from augmentum.discovery.entities import assign_entity_signal
            file_id = ""
            if isinstance(metadata, dict):
                file_id = str(metadata.get("file_id") or "")
            try:
                entity_cluster = await assign_entity_signal(
                    store, signal_id,
                    user_id=uid, file_id=file_id, fallback_title=title,
                )
            except Exception:
                log.warning("entity_signal_assign_failed", exc_info=True)
                entity_cluster = None
            if entity_cluster:
                entity_clustered = True
                log.info(
                    "signal_clustered",
                    signal_id=signal_id, cluster_id=entity_cluster,
                    kind="entity",
                )
        signal_text = None if entity_clustered else extract_signal_text({
            "source_title": title,
            "signal_type": signal_type,
            "metadata": metadata if isinstance(metadata, dict) else {},
        })
        if signal_text:
            try:
                cluster_id = await assign_signal_to_cluster(
                    store, signal_id, signal_text, signal_type, weight,
                    user_id=uid,
                )
                if cluster_id:
                    # Update frecency + narration for the cluster. Narration
                    # is the dormant `interest_clusters.narration` field —
                    # template-driven label string the UI can show as a
                    # human-readable summary instead of just the cluster
                    # name. Recomputed alongside frecency so it tracks the
                    # cluster's evolving signal mix without a separate job.
                    signals = await store.list_signals(
                        cluster_id=cluster_id, limit=100, user_id=uid,
                    )
                    if signals:
                        short, long_ = compute_frecency_from_signals(signals)
                        # Pull the current cluster name for narration. If the
                        # cluster row is missing (shouldn't happen here, but
                        # be defensive), skip narration update — frecency
                        # still updates.
                        narration: str | None = None
                        try:
                            name_row = await store._conn.execute(
                                "SELECT name, signal_count FROM interest_clusters "
                                "WHERE cluster_id = ? AND user_id = ?",
                                (cluster_id, uid),
                            )
                            name_data = await name_row.fetchone()
                            if name_data:
                                cl_name, cl_signal_count = name_data
                                narration = compose_narration(
                                    name=cl_name or "",
                                    signal_count=int(cl_signal_count or len(signals)),
                                    signals=signals,
                                )
                        except Exception:
                            log.debug("narration_compose_failed", exc_info=True)
                            narration = None

                        if narration is not None:
                            await store._conn.execute(
                                """UPDATE interest_clusters
                                   SET frecency_short = ?, frecency_long = ?,
                                       narration = ?, updated_at = datetime('now')
                                   WHERE cluster_id = ?""",
                                (short, long_, narration, cluster_id),
                            )
                        else:
                            await store._conn.execute(
                                """UPDATE interest_clusters
                                   SET frecency_short = ?, frecency_long = ?, updated_at = datetime('now')
                                   WHERE cluster_id = ?""",
                                (short, long_, cluster_id),
                            )
                        await store._conn.commit()
                    log.info("signal_clustered", signal_id=signal_id, cluster_id=cluster_id)
            except Exception as exc:
                log.warning("signal_clustering_failed", error=str(exc))

    # Upsert browse history for visit/open signals.
    #
    # The set of signal types that feed history was expanded from the
    # original web-only triple (page_visit / video_open / video_watch)
    # to include media-server consumption events (media_play / comic_read).
    # That way a unified History view shows every item the user has
    # actually consumed — articles, YouTube clips, audiobooks, comics,
    # movies, shows, and podcasts — keyed by either the original URL
    # (web content) or a synthetic ``augm:media:<file_id>`` URL (library
    # content). The synthetic-URL convention lets the existing browse_
    # history table absorb media items without a schema migration; the
    # frontend's history renderer detects the prefix and routes clicks
    # through ``discovery:open-file`` instead of ``discovery:open-url``.
    if signal_type in _HISTORY_SIGNAL_TYPES and source_url:
        try:
            meta = body.get("metadata") or {}
            domain, thumb = _history_domain_and_thumb(
                source_url=source_url,
                content_type=content_type,
                meta=meta,
            )
            await store.upsert_history(
                url=source_url,
                title=source_title,
                domain=domain,
                content_type=content_type or "article",
                thumbnail=thumb,
                metadata=meta,
                user_id=uid,
            )
        except Exception as exc:
            log.debug("discovery_history_upsert_skipped", error=str(exc))

    # Engagement → domain reputation: clicks/favorites on Discovery items teach
    # the quality pipeline which sources this user actually finds valuable.
    if signal_type in _REPUTATION_BUMP_TYPES and source_url:
        try:
            from augmentum.proxy.reputation import _update_reputation
            await _update_reputation(request, source_url, user_action=True)
        except Exception:
            log.debug("reputation_bump_failed", url=source_url[:80])

    # Optionally distill raw content into the knowledge library
    raw_content = body.get("raw_content")
    if (
        raw_content
        and settings.knowledge_library_enabled
        and signal_type in _LIBRARY_SIGNAL_TYPES
    ):
        source_url = body.get("source_url", "")
        already_stored = await store.has_source(source_url, user_id=uid)
        if not already_stored:
            # Hold a ref so the distill-and-store work isn't GC'd
            # mid-flight — silently dropped "save to library" articles
            # have no recovery path.
            from augmentum.utils.bg_tasks import track
            track(
                _distill_and_store(
                    store,
                    source_url=source_url,
                    source_title=body.get("source_title", ""),
                    source_type=body.get("source_type", "article"),
                    raw_content=raw_content,
                    is_html=bool(body.get("is_html", False)),
                    user_id=uid,
                )
            )

    return JSONResponse(result)


async def _distill_and_store(
    store,
    *,
    source_url: str,
    source_title: str,
    source_type: str,
    raw_content: str,
    is_html: bool,
    user_id: str = "",
) -> None:
    """Background task: distill content into chunks and embed into the library."""
    try:
        from augmentum.discovery.distiller import chunk_text, distill_article, distill_transcript
        from augmentum.memory.embeddings import EmbeddingService

        if source_type == "video_transcript":
            chunks = chunk_text(raw_content)
        elif is_html:
            distilled = distill_article(raw_content, url=source_url, title=source_title)
            chunks = distilled.get("chunks") or chunk_text(distilled.get("text", raw_content))
        else:
            chunks = chunk_text(raw_content)

        # Embed the whole chunk set in ONE offloaded batch — embed_one in
        # the loop ran synchronous ONNX inference on the event loop once
        # per chunk, stalling every other request while an article was
        # distilled (2026-06-13 loop-stall audit).
        vecs = (
            await asyncio.to_thread(EmbeddingService.embed, chunks)
            if chunks else []
        )
        for chunk, vec in zip(chunks, vecs, strict=False):
            blob = EmbeddingService.to_blob(vec)
            await store.store_chunk(
                source_url=source_url,
                source_title=source_title,
                source_type=source_type,
                content=chunk,
                embedding=blob,
                cluster_id=None,
                user_id=user_id,
            )

        log.info("discovery_distill_stored", source_url=source_url, chunks=len(chunks))
    except Exception:
        log.warning("discovery_distill_failed", source_url=source_url, exc_info=True)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@router.get("/history")
async def list_history(
    request: Request,
    page: int = 1,
    q: str = "",
    days: int = 0,
) -> JSONResponse:
    """Get paginated browse history for the authenticated user.

    Library media items (URLs prefixed with ``augm:media:``) are enriched
    at read-time with current playback / read progress from the
    file_index. The history row's stored metadata is whatever was
    captured at signal-emit time and goes stale as the user keeps
    listening; merging the live source_metadata fields gives the
    history list real-time "Continue from 47%" UX without round-tripping
    twice from the client.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    limit = 50
    offset = (page - 1) * limit

    # Fetch one extra to detect has_more
    rows = await store.list_history(
        limit=limit + 1, offset=offset, query=q, days=days, user_id=uid,
    )
    has_more = len(rows) > limit
    items = rows[:limit]

    # Enrich library media items with live progress from file_index.
    idx = getattr(request.app.state, "file_index", None)
    if idx is not None:
        await _enrich_history_with_progress(items, idx, uid)

    return JSONResponse({"items": items, "page": page, "has_more": has_more})


async def _enrich_history_with_progress(items: list[dict], idx, user_id: str) -> None:
    """Mutate ``items`` in place: for each library-media row, look up
    its current source_metadata in file_index and merge progress fields
    into the history metadata. Skips silently for unreadable rows so a
    deleted file entry doesn't blow up the whole list.
    """
    for item in items:
        url = str(item.get("url") or "")
        if not url.startswith(_MEDIA_URL_PREFIX):
            continue
        file_id = url[len(_MEDIA_URL_PREFIX):]
        if not file_id:
            continue
        try:
            entry = await idx.get(file_id, user_id=user_id)
        except Exception as exc:
            log.debug("discovery_idx_get_failed", file_id=file_id, error=str(exc))
            continue
        if not entry:
            continue
        meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
        # Merge live progress on top of whatever was captured at signal
        # emit. Field names mirror what the resume-listening endpoint
        # already returns so the frontend can reuse one rendering path.
        live = item.get("metadata") or {}
        if isinstance(live, str):
            try:
                import json as _json
                live = _json.loads(live)
            except Exception:
                live = {}
        progress_pct = float(meta.get("progress_pct") or 0.0)
        if progress_pct < 0:
            progress_pct = 0.0
        elif progress_pct > 1:
            progress_pct = 1.0
        live["current_progress_pct"] = progress_pct
        live["current_time_s"] = float(meta.get("current_time_s") or 0.0)
        live["duration_s"] = float(meta.get("duration_s") or live.get("duration_s") or 0.0)
        live["is_finished"] = bool(meta.get("is_finished") or False)
        # Comic-specific: page progress (current_page / page_count). The
        # file_index row stores these in source_metadata.extra for Komga
        # / Suwayomi rows; surfacing both lets the frontend pick whichever
        # framing is most readable per kind.
        extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
        if extra.get("page_count"):
            live["page_count"] = int(extra.get("page_count") or 0)
            live["current_page"] = int(extra.get("current_page") or 0)
        item["metadata"] = live


@router.delete("/history/{history_id}")
async def delete_history(request: Request, history_id: str) -> JSONResponse:
    """Delete one of the authenticated user's browse history entries."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    deleted = await store.delete_history(history_id, user_id=uid)
    if not deleted:
        return JSONResponse({"error": "History entry not found"}, status_code=404)

    return JSONResponse({"status": "deleted", "id": history_id})


# ---------------------------------------------------------------------------
# Knowledge Library
# ---------------------------------------------------------------------------


@router.get("/knowledge/search")
async def search_knowledge(request: Request, q: str = "") -> JSONResponse:
    """Search the knowledge library by semantic similarity."""
    uid = _user_id(request)
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    if not q:
        return JSONResponse({"results": []})

    from augmentum.memory.embeddings import EmbeddingService

    # Offloaded — synchronous embed on the loop blocked every knowledge
    # search request (2026-06-13 audit).
    vec = await asyncio.to_thread(EmbeddingService.embed_query, q)
    query_blob = EmbeddingService.to_blob(vec)

    results = await store.search_library(query_blob, limit=5, user_id=uid)

    # Increment retrieved_count for each returned chunk
    for chunk in results:
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            await store.increment_retrieved(chunk_id, user_id=uid)

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# Interests / For You / Dismiss
# ---------------------------------------------------------------------------


@router.get("/interests")
async def get_interests(request: Request) -> JSONResponse:
    """Return the authenticated user's interest clusters."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        clusters = await store.list_clusters(include_dampened=False, user_id=uid)
    except Exception as exc:
        log.warning("interests_list_failed", error=str(exc))
        return JSONResponse({"error": "Failed to list clusters"}, status_code=500)

    return JSONResponse({"clusters": clusters})


@router.get("/for-you")
async def get_for_you(request: Request, seed: int | None = None) -> JSONResponse:
    """Generate personalised recommendations for the authenticated user.

    Accepts repeated ``exclude=<url>`` query params so background polling can
    ask for items *other than* what's already on screen. Each URL is folded
    into the per-user hidden-URL set just for this call (not persisted), so
    the recommender treats them as already-seen and picks different ones.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        from augmentum.discovery.recommender import generate_recommendations
        from augmentum.proxy.reputation import _get_domain_scores

        http_client = getattr(request.app.state, "http_client", None)
        searxng_base = settings.searxng_base_url

        # Load learned domain reputation scores for quality scoring
        domain_scores = await _get_domain_scores(request)

        # URLs the user hid from Discovery — never reappear.
        hidden_urls = await store.list_hidden_urls(user_id=uid)

        # Caller-provided exclusions (already-on-screen URLs, for polling).
        # Cap at 200 to bound query size and cluster-overlap work.
        exclude_list = request.query_params.getlist("exclude")[:200]
        if exclude_list:
            hidden_urls = set(hidden_urls) | set(exclude_list)

        # External feed config from settings_store (HN / Reddit / arxiv / RSS).
        feed_config = await _load_feed_config(request)

        allow_non_latin = bool(getattr(settings, "discovery_allow_non_latin", False))
        recs = await generate_recommendations(
            store,
            searxng_base=searxng_base,
            total=15,
            http_client=http_client,
            seed=seed,
            domain_scores=domain_scores,
            hidden_urls=hidden_urls,
            feed_config=feed_config,
            user_id=uid,
            allow_non_latin=allow_non_latin,
            # The For-You panel polls this endpoint — autonomous from the
            # web-search-policy lens. User-initiated searches go through
            # the WebSearchTool, not the recommender, and are unaffected.
            autonomous=True,
        )

        zones: dict[str, list] = {"core": [], "frontier": [], "adjacent": [], "fresh": []}
        for r in recs:
            zone = r.get("zone", "core")
            if zone in zones:
                zones[zone].append(r)

    except Exception as exc:
        log.warning("for_you_generation_failed", error=str(exc))
        return JSONResponse(
            {"recommendations": [], "zones": {"core": [], "frontier": [], "adjacent": [], "fresh": []}},
        )

    return JSONResponse({"recommendations": recs, "zones": zones})


# ---------------------------------------------------------------------------
# Library zones — surfaces the user's own media (comics / audiobooks /
# movies / shows) on the For You page alongside the web feed zones.
#
# Why a single endpoint, not per-kind: one HTTP round-trip beats four,
# and the response is already small (≤40 items total). Each kind's
# query hits the file_index in <50ms on a warm SQLite, so wall-clock is
# dominated by the fan-out, not the per-query cost — gather() in
# parallel and we're done in roughly one query's time.
#
# Why "in_progress first, then recently added": the user's intent on
# "For You" is "what should I do next." In-progress items are the
# strongest answer to that question; recently-added items are a softer
# "in case you want to start something." Capping the in_progress slice
# at 6 keeps any single zone from becoming a wall of resume-cards.
# ---------------------------------------------------------------------------

# Map zone-name → (kind filter, entity_kinds filter). Any zone whose query
# returns zero rows is dropped from the response so the frontend doesn't
# render an empty strip for media types the user hasn't installed yet.
_LIBRARY_ZONES: dict[str, dict] = {
    # Comics: Komga / Suwayomi / Kavita all sync as kind="document" with
    # entity_kind in {manga, comic}. We don't try to dedupe to series
    # level here — chapter-level rows are what file_index has, and the
    # card title (e.bse e.name) is already chapter-or-series shaped.
    "comics":     {"kind": "document", "entity_kinds": ["manga", "comic"]},
    # Audiobooks share the same query as /api/media/resume-listening but
    # we extend it to include "newest" too so users with no in-progress
    # audio still see their library.
    "audiobooks": {"kind": "audio",    "entity_kinds": ["book"]},
    "movies":     {"kind": "video",    "entity_kinds": ["movie"]},
    "shows":      {"kind": "video",    "entity_kinds": ["series"]},
}

_LIBRARY_ZONE_LIMIT = 12     # max cards per zone
_LIBRARY_INPROGRESS_CAP = 6  # in-progress slice ceiling


# Maps `(kind, entity_kind)` → the Files-panel virtual chip the user
# should land on if they click "show in Files" for this card. Mirrors
# the chip vocabulary in ui/scripts/files/state.js / files_routes.py.
# Used by both the chip field below and the frontend filter dispatch.
_KIND_TO_CHIP: dict[tuple[str, str], str] = {
    ("audio",    "book"):    "audiobooks",
    ("audio",    "podcast"): "podcasts",
    ("document", "manga"):   "comics",
    ("document", "comic"):   "comics",
    ("video",    "movie"):   "movies",
    ("video",    "series"):  "shows",
}


def _library_card(entry, status: str) -> dict:
    """Shape a FileEntry into a card-friendly dict for the frontend.

    Only fields the renderer actually needs — keeps the wire payload
    small (a 4-zone × 12-item response is <8KB even with metadata).
    The shape is consistent across kinds so the frontend has a single
    card-renderer rather than four parallel ones.

    Two new fields support the "click to show in Files" navigation:
      * ``chip`` — the virtual Files chip that owns this media type.
      * ``search_hint`` — the text the Files search should be seeded
        with when the user clicks the subtitle. For audiobooks this
        is the author; for comics it's the author or series name;
        for movies/shows we leave it blank because the subtitle is
        a composite (year · director) that wouldn't search well.
    """
    meta = entry.source_metadata if isinstance(entry.source_metadata, dict) else {}
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    entity_kind_raw = str(meta.get("entity_kind") or "").lower()

    # Subtitle picks the most relevant secondary line per kind. We
    # collapse to a single string here (rather than passing structured
    # fields) because the renderer only ever needs to display it as
    # one line of muted text.
    subtitle = ""
    search_hint = ""
    if entry.kind == "audio":
        author = str(meta.get("author") or extra.get("narrator") or "")
        subtitle = author
        # Author-search is the natural drill — "more from this author"
        # is the most-asked navigation on a now-playing card. Narrator-
        # only entries (rare; LibriVox solo recordings) still resolve
        # via author search since LibriVox stores the same name in both.
        search_hint = str(meta.get("author") or "")
    elif entry.kind == "document":
        author = str(meta.get("author") or extra.get("series_name") or "")
        subtitle = author
        # Comic users tend to think series-first ("more Saga chapters")
        # rather than author-first, so prefer series_name when present.
        search_hint = str(extra.get("series_name") or meta.get("author") or "")
    elif entry.kind == "video":
        # Movies prefer year/director; series prefer status/genre tag.
        if entity_kind_raw == "movie":
            year = extra.get("year") or meta.get("year") or ""
            director = extra.get("director") or ""
            parts = [str(p) for p in (year, director) if p]
            subtitle = " · ".join(parts)
            search_hint = str(director or "")  # year-only search is noise
        else:  # series
            status_label = extra.get("status") or ""
            year = extra.get("year") or meta.get("year") or ""
            parts = [str(p) for p in (year, status_label) if p]
            subtitle = " · ".join(parts)
            # Series subtitle is "Year · Continuing" — neither half is
            # a useful search seed, so leave blank.
            search_hint = ""

    progress_pct = float(meta.get("progress_pct") or 0.0)
    if progress_pct < 0:
        progress_pct = 0.0
    elif progress_pct > 1:
        progress_pct = 1.0

    chip = _KIND_TO_CHIP.get((entry.kind or "", entity_kind_raw), "")

    return {
        "file_id":       entry.id,
        "title":         entry.name,
        "subtitle":      subtitle,
        "search_hint":   search_hint,
        "cover_url":     f"/api/media/cover/{entry.id}",
        "source":        entry.source or "",
        "kind":          entry.kind or "",
        "entity_kind":   entity_kind_raw,
        "chip":          chip,
        "progress_pct":  progress_pct,
        "status":        status,  # "in_progress" | "recent"
        "updated_at":    entry.updated_at or "",
    }


async def _build_library_zone(idx, user_id: str, *, kind: str, entity_kinds: list[str]) -> list[dict]:
    """Build one zone: in-progress slice (up to 6) + recent additions
    (filling to 12), with no overlap between slices.
    """
    in_progress = await idx.list_recent(
        user_id=user_id,
        kind=kind,
        entity_kinds=entity_kinds,
        media_status="in_progress",
        sort="progress",  # most-recently-touched first
        limit=_LIBRARY_INPROGRESS_CAP,
    )
    seen_ids = {e.id for e in in_progress}

    remaining = max(0, _LIBRARY_ZONE_LIMIT - len(in_progress))
    recent: list = []
    if remaining > 0:
        # Pull a wider slice than `remaining` so we have headroom after
        # filtering out any rows that already showed up in_progress.
        # 2× headroom is enough for the worst case (every recent row
        # also being in-progress).
        candidates = await idx.list_recent(
            user_id=user_id,
            kind=kind,
            entity_kinds=entity_kinds,
            sort="newest",
            limit=remaining * 2,
        )
        for e in candidates:
            if e.id in seen_ids:
                continue
            recent.append(e)
            if len(recent) >= remaining:
                break

    cards = [_library_card(e, status="in_progress") for e in in_progress]
    cards.extend(_library_card(e, status="recent") for e in recent)
    return cards


@router.get("/library")
async def get_library_zones(request: Request) -> JSONResponse:
    """Return the user's own media as four zones for the For You page.

    Empty zones are omitted from the response so the frontend doesn't
    render placeholder strips for media types the user hasn't installed.
    Local SQLite queries — typical wall-clock <100ms even with all
    four zones populated.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    idx = getattr(request.app.state, "file_index", None)
    if idx is None:
        return JSONResponse({"zones": {}})

    try:
        results = await asyncio.gather(
            *(
                _build_library_zone(idx, uid, **cfg)
                for cfg in _LIBRARY_ZONES.values()
            ),
            return_exceptions=True,
        )
    except Exception as exc:
        log.warning("library_zones_failed", error=str(exc))
        return JSONResponse({"zones": {}})

    zones: dict[str, list[dict]] = {}
    for (zone_name, _cfg), result in zip(_LIBRARY_ZONES.items(), results):
        if isinstance(result, Exception):
            log.warning("library_zone_failed", zone=zone_name, error=str(result))
            continue
        if result:
            zones[zone_name] = result

    return JSONResponse({"zones": zones})


@router.post("/dismiss")
async def dismiss_recommendation(request: Request) -> JSONResponse:
    """Dampen one of the caller's clusters so it appears less in recs."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    cluster_id = body.get("cluster_id")
    if not cluster_id:
        return JSONResponse({"error": "cluster_id is required"}, status_code=400)

    try:
        await store.dampen_cluster(cluster_id, user_id=uid)
    except Exception as exc:
        log.warning("dismiss_failed", cluster_id=cluster_id, error=str(exc))
        return JSONResponse({"error": "Failed to dampen cluster"}, status_code=500)

    return JSONResponse({"status": "dampened", "cluster_id": cluster_id})


# ---------------------------------------------------------------------------
# Visited URL check
# ---------------------------------------------------------------------------


@router.get("/check-visited")
async def check_visited(request: Request, urls: str = "") -> JSONResponse:
    """Check which URLs the authenticated user has visited."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Discovery store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    url_list = [u.strip() for u in urls.split(",") if u.strip()] if urls else []
    visited = await store.check_visited(url_list, user_id=uid)
    return JSONResponse({"visited": visited})
