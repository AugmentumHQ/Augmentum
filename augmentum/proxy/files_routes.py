"""File browser REST API."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import JSONResponse, Response

from augmentum.config import settings
from augmentum.utils.logging import get_logger
from augmentum.vfs.index import _ENTRY_COLUMNS
from augmentum.vfs.tags import normalize_tags, suggest_tags
from augmentum.vfs.validation import (
    is_mime_mismatch,
    sanitize_filename,
    sniff_mime,
)

log = get_logger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"])

# Sources whose rows have no bytes on disk — playback goes through
# /api/media/stream/{id} (chapter-aware, Range-forwarding, auth-handling
# proxy). Kept in sync with augmentum/vfs/index.py:_MEDIA_SOURCES and the
# frontend MEDIA_SOURCES set in ui/scripts/files/state.js.
_MEDIA_STREAMING_SOURCES = frozenset({"audiobookshelf", "librivox", "emby", "jellyfin"})

# Cloud sources — anything that lives on a remote service the user
# connected via /api/media/servers (or the built-in LibriVox integration).
# Files from these sources never have bytes on disk; they're streamed
# through the media proxy. The Files panel's Local/Cloud scope toggle
# uses this set to isolate user-authored/uploaded content from
# catalog-sourced content (the Berserk-drowns-your-recent-images case).
# Extend as Emby/Jellyfin/Plex/Kavita land; ``scope=cloud`` will
# automatically include them.
_CLOUD_SOURCES: frozenset[str] = frozenset({
    "audiobookshelf", "librivox",
    "suwayomi", "komga", "kavita",
    "jellyfin", "plex", "emby",
})

# Virtual chip slugs: the UI chip name → concrete list of row sources to
# filter by. Lets one "Audiobooks" chip cover both a user's connected ABS
# library and their pinned LibriVox catalog entries without duplicating
# chips per provider. Extend as Emby / Jellyfin / etc. come online.
_SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "audiobooks": ("audiobookshelf", "librivox"),
    "podcasts":   ("audiobookshelf",),
    "comics":     ("suwayomi", "komga", "kavita"),
}

# Some virtual source chips need an extra entity-kind fence in addition to
# their provider list. Audiobookshelf can hold both books and podcasts, so:
#   - "audiobooks" keeps ABS books + LibriVox rows, excluding podcasts
#   - "podcasts" isolates ABS podcast container rows
_SOURCE_GROUP_ENTITY_FILTERS: dict[str, dict[str, tuple[str, ...]]] = {
    "audiobooks": {
        "exclude_entity_kinds": ("podcast",),
    },
    "podcasts": {
        "entity_kinds": ("podcast",),
    },
}

_VIDEO_PROVIDER_SOURCES: tuple[str, ...] = ("emby", "jellyfin")

_VIDEO_CHIP_FILTERS: dict[str, dict[str, tuple[str, ...] | str]] = {
    "shows": {
        "sources": _VIDEO_PROVIDER_SOURCES,
        "entity_kinds": ("series",),
    },
    "movies": {
        "sources": _VIDEO_PROVIDER_SOURCES,
        "entity_kinds": ("movie",),
    },
    "music_videos": {
        "sources": _VIDEO_PROVIDER_SOURCES,
        "entity_kinds": ("music_video",),
    },
}


class _PerUserTTLCache:
    """Async-safe per-user TTL cache with stampede protection.

    Concurrent calls for the same user inside the TTL window:
      - First caller acquires the per-user lock, computes, stores.
      - Subsequent callers wait on the same lock, then read the freshly
        stored value (no duplicate compute).
    Concurrent calls for *different* users run in parallel — locks are
    per-user, not global.

    Bounded by ``max_users`` — when full, the oldest entry by store
    timestamp is evicted before insertion. Idle locks are evicted with
    their entry; locks held by an in-flight compute are left alone (the
    in-flight call repopulates the entry, so eviction is a no-op).
    """

    def __init__(self, ttl_s: float, max_users: int) -> None:
        self._ttl = ttl_s
        self._max = max_users
        self._cache: dict[str, tuple[float, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Guards structural ops on _locks so two concurrent first-time
        # callers for the same user don't create two competing locks.
        self._lock_factory_guard = asyncio.Lock()

    def _is_fresh(self, entry: tuple[float, Any]) -> bool:
        return (time.monotonic() - entry[0]) < self._ttl

    async def get_or_compute(
        self,
        user_id: str,
        compute_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        # Fast path — fresh hit, no locking.
        entry = self._cache.get(user_id)
        if entry is not None and self._is_fresh(entry):
            return entry[1]
        # Slow path — lazily create per-user lock under a guard so
        # concurrent first callers see the same lock instance.
        lock = self._locks.get(user_id)
        if lock is None:
            async with self._lock_factory_guard:
                lock = self._locks.get(user_id)
                if lock is None:
                    lock = asyncio.Lock()
                    self._locks[user_id] = lock
        async with lock:
            entry = self._cache.get(user_id)
            if entry is not None and self._is_fresh(entry):
                return entry[1]
            value = await compute_fn()
            self._store(user_id, value)
            return value

    def _store(self, user_id: str, value: Any) -> None:
        self._cache[user_id] = (time.monotonic(), value)
        if len(self._cache) > self._max:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest_key, None)
            stale_lock = self._locks.get(oldest_key)
            if stale_lock is not None and not stale_lock.locked():
                self._locks.pop(oldest_key, None)

    def invalidate(self, user_id: str) -> None:
        """Drop a user's cached value. Safe to call from write paths."""
        self._cache.pop(user_id, None)

    def clear(self) -> None:
        """Drop all cached entries. Used by tests and admin reset paths."""
        self._cache.clear()


# /api/files/stats result cache. Endpoint runs ~8 sequential GROUP BY
# scans on file_index — cumulative ~5-10s on a 64k-row library. Chip
# badges drive the call (Files-tab open + every chip click), so the
# uncached path was the dominant Files-tab latency.
#
# 30s TTL — chip-badge counts are advisory; users tolerate brief
# staleness after upload/delete in exchange for the perf win. Active
# write paths can call ``_FILE_STATS_CACHE.invalidate(user_id)`` if
# they need immediate freshness.
_FILE_STATS_CACHE = _PerUserTTLCache(ttl_s=30.0, max_users=256)

# /api/video/genres + /api/comics/genres result caches. The underlying
# SQL unnests JSON arrays per row (json_each) and groups by lowercased
# value — 500-660ms each on a 64k-row library. The Filter dropdown
# polls them on every chip click. 60s TTL: genre lists rarely change
# (new media gets new genres maybe a few times per import); brief
# staleness is invisible in the dropdown experience.
_VIDEO_GENRES_CACHE = _PerUserTTLCache(ttl_s=60.0, max_users=256)
_COMIC_GENRES_CACHE = _PerUserTTLCache(ttl_s=60.0, max_users=256)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").strip()
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    client = request.scope.get("client")
    return client[0] if client else ""


async def _audit(request: Request, action: str, *, detail: str = "") -> None:
    """Best-effort audit log write for file operations.

    Reuses the auth audit table — `action` namespacing (`file.upload` etc.)
    keeps the rows distinguishable.  Silently no-ops when the session
    manager isn't available so tests and stripped-down configs don't break.
    """
    sm = getattr(request.app.state, "session_manager", None)
    if not sm:
        return
    actor = request.scope.get("user")
    try:
        await sm.write_audit(
            actor=actor, target=None, action=action,
            detail=detail or "", ip_address=_client_ip(request),
        )
    except Exception:
        log.warning("file_audit_write_failed", action=action, exc_info=True)


def _get_index(request: Request):
    return getattr(request.app.state, "file_index", None)


def _get_vfs(request: Request):
    return getattr(request.app.state, "vfs", None)


def _paging(request: Request, default_limit: int = 60, max_limit: int = 200):
    q = request.query_params
    try:
        raw_limit = int(q.get("limit", str(default_limit)))
    except (TypeError, ValueError):
        raw_limit = default_limit
    try:
        raw_offset = int(q.get("offset", "0"))
    except (TypeError, ValueError):
        raw_offset = 0
    limit = max(1, min(raw_limit, max_limit))
    offset = max(0, raw_offset)
    sort = q.get("sort") or None
    return limit, offset, sort


@router.get("/search")
async def search_files(request: Request):
    """Search files across all sources, paginated."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"files": [], "has_more": False, "offset": 0})

    query = request.query_params.get("q", "")
    source = request.query_params.get("source")
    # Keep the raw user-intent separate from any post-expansion ``sources``.
    # ``source=comics`` means "user picked the Comics chip" → comics should
    # show. But ``scope=cloud`` later SETS ``sources`` to all cloud providers,
    # which would incorrectly signal "user pinned a source" to the comic-
    # exclusion guard. Using ``raw_source`` below keeps the two separate.
    raw_source = source
    mime = request.query_params.get("mime")
    kind = request.query_params.get("kind")
    entity_kind = request.query_params.get("entity_kind")
    # media_status: playback-state filter for audiobookshelf-like
    # sources. Value is whitelisted in the index layer; bad input is
    # silently ignored rather than erroring so typos don't 500 the UI.
    media_status = request.query_params.get("media_status")
    # Year range: bracket video rows by source_metadata.extra.year. Used
    # by the Filter dropdown's Year slider on the Shows/Movies/Music
    # Videos chips. Empty string or missing → no bound. Both bounds
    # optional and independent. Rows with year=0 (no data) only stay
    # visible when neither bound is set; otherwise they drop from the
    # result so the user isn't surprised by undated entries when they
    # explicitly asked for "1990 — 2010."
    try:
        year_from = int(request.query_params.get("year_from") or 0)
    except (TypeError, ValueError):
        year_from = 0
    try:
        year_to = int(request.query_params.get("year_to") or 0)
    except (TypeError, ValueError):
        year_to = 0
    # Genre filter for video rows (Shows/Movies/Music Videos). Reads
    # source_metadata.extra.genres[] — case-insensitive substring
    # match so "supernatural" hits both "Supernatural" and "Supernatural
    # Action" without forcing the user to know exact upstream casing.
    # Empty string skips the filter entirely.
    genre_filter = (request.query_params.get("genre") or "").strip().lower()
    # scope: 'local' (uploads, notes, artifacts — things that live here)
    # vs 'cloud' (audiobooks, comics, future video from remote servers).
    # The Files panel's Local/Cloud toggle lives here; unknown values
    # fall through to "no scope filter" so old clients don't break.
    scope = (request.query_params.get("scope") or "").strip().lower()
    limit, offset, sort = _paging(request)

    # Source groups: a chip slug like "audiobooks" stands in for a set of
    # concrete source names. Expand at the route layer so the store stays
    # unaware of UI chip vocabulary.
    sources: list[str] | None = None
    exclude_sources: list[str] | None = None
    entity_kinds: list[str] | None = None
    exclude_entity_kinds: list[str] | None = None
    if source and source in _SOURCE_GROUPS:
        sources = list(_SOURCE_GROUPS[source])
        cfg = _SOURCE_GROUP_ENTITY_FILTERS.get(source) or {}
        if cfg.get("entity_kinds"):
            entity_kinds = list(cfg.get("entity_kinds") or [])
        if cfg.get("exclude_entity_kinds"):
            exclude_entity_kinds = list(cfg.get("exclude_entity_kinds") or [])
        source = None
    elif source and source in _VIDEO_CHIP_FILTERS:
        cfg = _VIDEO_CHIP_FILTERS[source]
        sources = list(cfg.get("sources") or [])
        entity_kinds = list(cfg.get("entity_kinds") or [])
        source = None

    # Scope translation. ``cloud`` adds an include filter; ``local`` adds an
    # exclude filter for every known cloud source. If the caller also pinned
    # a source group (e.g. ``source=comics``), we let the store apply both
    # — contradictory combos naturally return empty instead of the wrong
    # rows, which is what we want semantically.
    if scope == "cloud":
        if sources is None:
            sources = list(_CLOUD_SOURCES)
    elif scope == "local":
        exclude_sources = list(_CLOUD_SOURCES)

    # Comic chapter rows are the only "content" in file_index that exists
    # at a sub-item granularity — a user's Suwayomi library is 20k+
    # individual chapter rows, one per chapter. Showing those flat in the
    # All tab buries everything else. Exclude kind='comic' when the caller
    # hasn't explicitly asked for comics via a chip OR a kind filter OR
    # a search query. Users reach comics through the Comics chip (rendered
    # as series) instead; searching by chapter name still works.
    #
    # Gate on ``raw_source`` (pre-expansion) not ``sources``: scope=cloud
    # sets ``sources = _CLOUD_SOURCES`` to implement the cloud filter,
    # which would otherwise look like "user pinned Comics" and skip the
    # exclusion. Before this check ``raw_source`` is the exact string the
    # user's chip put on the wire — 'all'/None/'comics'/etc.
    exclude_comic_kind = (
        not raw_source
        and not kind
        and not query
    )

    exclude_kinds = ["comic"] if exclude_comic_kind else None
    exclude_video_detail_entities = (
        scope == "cloud"
        and not raw_source
        and not kind
        and not query
    )
    if exclude_video_detail_entities:
        exclude_entity_kinds = list(exclude_entity_kinds or [])
        for entity in ("season", "episode"):
            if entity not in exclude_entity_kinds:
                exclude_entity_kinds.append(entity)
    if not query:
        results = await idx.list_recent(
            user_id=uid, limit=limit, offset=offset, sort=sort,
            source=source, sources=sources, exclude_sources=exclude_sources,
            kind=kind, exclude_kinds=exclude_kinds,
            entity_kind=entity_kind,
            entity_kinds=entity_kinds,
            exclude_entity_kinds=exclude_entity_kinds,
            media_status=media_status,
            year_from=year_from, year_to=year_to,
            genre=genre_filter,
        )
    else:
        # Search: intentionally DO include comic chapters — if the user
        # searches "Berserk 12" they want the chapter hit, not just the
        # series. exclude_comic_kind is False when query is non-empty.
        results = await idx.search(
            query, user_id=uid, source=source, sources=sources,
            exclude_sources=exclude_sources,
            mime_filter=mime, kind=kind,
            entity_kind=entity_kind,
            entity_kinds=entity_kinds,
            exclude_entity_kinds=exclude_entity_kinds,
            media_status=media_status,
            year_from=year_from, year_to=year_to,
            genre=genre_filter,
            limit=limit, offset=offset, sort=sort,
        )

    return JSONResponse({
        "files": [f.to_dict() for f in results],
        "has_more": len(results) == limit,
        "offset": offset,
    })


@router.get("/comics/series")
async def list_comic_series(request: Request):
    """Series-level rollups for the Comics chip.

    Aggregates ``file_index`` chapter rows (one per chapter for Suwayomi,
    one per volume for Komga) up to their parent ``comic_series`` entry
    and returns per-series summaries: chapter count, read progress
    breakdown, cover art reference, and display metadata.

    At 20k+ chapters the raw ``/list`` firehose is unusable — this
    endpoint collapses to ~hundreds of series cards instead, matching how
    a human actually browses a manga library.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"series": [], "has_more": False, "offset": 0})

    limit, offset, sort = _paging(request, default_limit=60, max_limit=500)
    search_q = (request.query_params.get("q") or "").strip().lower()
    # Series-level filters. All are optional; unknown values fall through
    # as no-op filters so stale client state can't empty the grid.
    #
    # `status` — series read-state rollup, computed from the chapters:
    #     reading    : at least one chapter in progress
    #     caught-up  : every chapter finished (series fully read)
    #     unread     : no chapters read, no chapters in progress
    #
    # `completion` — publisher status (ongoing / completed / hiatus),
    # read straight off `comic_series.status`. Normalised by trimming
    # whitespace + case so catalogs that write "Ongoing" vs "ongoing"
    # match the same filter bucket.
    #
    # `genre` — substring match against the JSON genres array. We unwrap
    # via json_each + LIKE for flexibility ("supernatural" matches both
    # "supernatural" and "supernatural action"). A real library typically
    # has a few dozen distinct genres, so full table scan at filter time
    # is trivially fast.
    status_filter = (request.query_params.get("status") or "").strip().lower()
    completion_filter = (request.query_params.get("completion") or "").strip().lower()
    genre_filter = (request.query_params.get("genre") or "").strip().lower()

    # User-scoped + no-trash + cheap per-row progress extraction via
    # json_extract. The GROUP BY is on series id; HAVING chapter_count>0
    # hides orphan series (user removed all chapters but series row
    # survived) so the grid never shows empty cards.
    #
    # Search matches series-level identity — canonical_name, author, and
    # description. Single-field match (just canonical_name) misses the
    # common "I remember it was by Kentaro Miura" and "there was a vampire"
    # cases. LIKE is fine here: the series table typically has hundreds
    # of rows, not thousands, so full-table scan is fast enough without
    # needing FTS5 infrastructure.
    search_pred = ""
    params: list[Any] = [uid]
    if search_q:
        needle = f"%{search_q}%"
        search_pred = (
            " AND ("
            "LOWER(cs.canonical_name) LIKE ? "
            "OR LOWER(COALESCE(cs.alias_names, '')) LIKE ? "
            "OR LOWER(COALESCE(cs.author, '')) LIKE ? "
            "OR LOWER(COALESCE(cs.description, '')) LIKE ?"
            ")"
        )
        params.extend([needle, needle, needle, needle])

    # Completion-status filter lives in the outer WHERE (it's a per-series
    # column, not a chapter rollup). Only three known values are accepted;
    # anything else is treated as "no filter."
    completion_pred = ""
    if completion_filter in ("ongoing", "completed", "hiatus"):
        completion_pred = " AND LOWER(TRIM(COALESCE(cs.status, ''))) = ?"
        params.append(completion_filter)

    # Genre filter — EXISTS over the JSON array so a series matches when
    # ANY of its genres contains the query substring. Paramised cleanly
    # via json_each's value column.
    genre_pred = ""
    if genre_filter:
        genre_pred = (
            " AND EXISTS ("
            "SELECT 1 FROM json_each(COALESCE(cs.genres, '[]')) g "
            "WHERE LOWER(g.value) LIKE ?"
            ")"
        )
        params.append(f"%{genre_filter}%")

    # Sort vocabulary — any unknown/empty value falls through to `name`,
    # which is the historical default. Series-specific sorts (`updated`,
    # `unread`) live here; generic file sorts (`newest`, `oldest`, `name`)
    # map to series-analogous columns so the Files panel's shared sort
    # dropdown has sensible semantics on the Comics chip.
    order_clauses = {
        "name":    "cs.sort_name ASC",
        # "newest" / "oldest": use MAX(fi.updated_at) as the series-level
        # signal — that's when a chapter was added or read-state updated.
        # Falls back to canonical_name for series with no chapters yet so
        # ordering is deterministic.
        "newest":  "MAX(fi.updated_at) DESC, cs.sort_name ASC",
        "oldest":  "MAX(fi.updated_at) ASC,  cs.sort_name ASC",
        # "updated": alias for newest — makes the sort-label vocabulary
        # match what the UI chip offers ("Recently updated").
        "updated": "MAX(fi.updated_at) DESC, cs.sort_name ASC",
        # "continue": series the user is actively reading, by recency of
        # their last reading session. Filters the MAX() to only chapters
        # with progress (current_time_s > 0 OR is_finished = 1) so a
        # series the user has never opened ranks last regardless of when
        # its chapters were synced. `last_read_at` is stamped by the
        # progress-push route; older read chapters fall back to
        # fi.updated_at so the sort is meaningful immediately rather
        # than after a re-read backfill. NULLs (untouched series) sort
        # last in DESC, then by name for tiebreak.
        "continue":
            "MAX(CASE "
            "  WHEN COALESCE(json_extract(fi.source_metadata, '$.is_finished'), 0) = 1 "
            "    OR CAST(COALESCE(json_extract(fi.source_metadata, "
            "                                  '$.current_time_s'), 0) AS REAL) > 0 "
            "  THEN COALESCE(json_extract(fi.source_metadata, '$.last_read_at'), "
            "                fi.updated_at) "
            "END) DESC, cs.sort_name ASC",
        # "unread": series with the most unread chapters first. Uses the
        # same expression as the unread_count we emit below so the order
        # matches what the user sees on the cards.
        "unread":
            "(COUNT(fi.id) - "
            "SUM(CASE WHEN json_extract(fi.source_metadata, '$.is_finished') = 1 "
            "THEN 1 ELSE 0 END) - "
            "SUM(CASE WHEN CAST(COALESCE(json_extract(fi.source_metadata, "
            "'$.current_time_s'), 0) AS REAL) > 0 "
            "AND COALESCE(json_extract(fi.source_metadata, '$.is_finished'), 0) != 1 "
            "THEN 1 ELSE 0 END)) DESC, cs.sort_name ASC",
    }
    order_by = order_clauses.get(sort or "name", order_clauses["name"])

    # Read-status filter operates on the aggregated counts so it has to
    # live in HAVING, not WHERE. Expressions reference the same column
    # aliases we emit below — SQLite allows aliases in HAVING.
    #   reading    → any chapter in-progress (regardless of finished count)
    #   caught-up  → every chapter finished, no unread, no in-progress
    #   unread     → nothing touched yet (no finished, no in-progress)
    status_having = ""
    if status_filter == "reading":
        status_having = " AND in_progress_count > 0"
    elif status_filter == "caught-up" or status_filter == "caught_up":
        status_having = " AND finished_count = chapter_count AND chapter_count > 0"
    elif status_filter == "unread":
        status_having = " AND finished_count = 0 AND in_progress_count = 0"

    sql = f"""
        SELECT
            cs.id                   AS series_id,
            cs.canonical_name,
            cs.author,
            cs.publisher,
            cs.description,
            cs.genres,
            cs.status,
            cs.year_started,
            cs.year_ended,
            cs.language_iso,
            cs.accent_color,
            COUNT(fi.id) AS chapter_count,
            SUM(CASE
                WHEN json_extract(fi.source_metadata, '$.is_finished') = 1
                THEN 1 ELSE 0 END
            ) AS finished_count,
            SUM(CASE
                WHEN CAST(COALESCE(json_extract(fi.source_metadata,
                                                '$.current_time_s'),
                                    0) AS REAL) > 0
                 AND COALESCE(json_extract(fi.source_metadata,
                                            '$.is_finished'), 0) != 1
                THEN 1 ELSE 0 END
            ) AS in_progress_count,
            -- Cover: any chapter's file_id works since they all share the
            -- series thumbnail via provider. MIN keeps it deterministic
            -- so the cover doesn't flicker between grid refreshes.
            MIN(fi.id) AS sample_chapter_file_id,
            MAX(fi.updated_at) AS last_updated
        FROM comic_series cs
        -- ``INDEXED BY idx_file_index_series`` forces the planner to walk
        -- the per-series (user_id, series_id) index instead of the wider
        -- ``idx_file_index_user_recent`` (user_id, is_trashed, created_at)
        -- the planner started preferring after that index landed for the
        -- Files-tab All/newest fix. The wrong index meant scanning all
        -- 63k user rows in memory and filtering by series_id afterwards,
        -- producing a 6.9s query against a 168-series library. With the
        -- hint, each series resolves via an O(log n) index lookup and
        -- the same query runs in ~150ms — 44x speedup, no caching, no
        -- schema change. SQLite's INDEXED BY is exactly the override
        -- mechanism for cases like this where the planner's cost
        -- estimate guesses wrong.
        LEFT JOIN file_index fi INDEXED BY idx_file_index_series
            ON fi.series_id = cs.id
           AND fi.user_id = cs.user_id
           AND fi.is_trashed = 0
        WHERE cs.user_id = ?{search_pred}{completion_pred}{genre_pred}
        GROUP BY cs.id
        HAVING chapter_count > 0{status_having}
        ORDER BY {order_by}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    cursor = await idx._db.execute(sql, params)
    rows = await cursor.fetchall()

    import json as _json  # avoid shadowing the top-level module name
    series_list = []
    for r in rows:
        total = int(r["chapter_count"] or 0)
        finished = int(r["finished_count"] or 0)
        in_progress = int(r["in_progress_count"] or 0)
        # ``genres`` is stored as a JSON array in the DB column. Decode
        # defensively so a bad write never breaks the grid.
        try:
            genres = _json.loads(r["genres"] or "[]")
            if not isinstance(genres, list):
                genres = []
        except (_json.JSONDecodeError, TypeError):
            genres = []
        series_list.append({
            "id":                r["series_id"],
            "name":              r["canonical_name"],
            "author":            r["author"] or "",
            "publisher":         r["publisher"] or "",
            "description":       r["description"] or "",
            "genres":            genres,
            "status":            r["status"] or "",
            "year_started":      r["year_started"],
            "year_ended":        r["year_ended"],
            "language_iso":      r["language_iso"] or "",
            "accent_color":      r["accent_color"] or "",
            "chapter_count":     total,
            "finished_count":    finished,
            "in_progress_count": in_progress,
            "unread_count":      max(0, total - finished - in_progress),
            # Frontend hits /api/media/cover/{file_id} with this; any
            # chapter's cover_url is the series cover for remote providers.
            "cover_file_id":     r["sample_chapter_file_id"] or "",
            "last_updated":      r["last_updated"] or "",
        })

    return JSONResponse({
        "series":   series_list,
        "has_more": len(series_list) == limit,
        "offset":   offset,
    })


@router.get("/comics/series/{series_id}")
async def get_comic_series(series_id: str, request: Request):
    """Single-series detail lookup for the in-reader nav drawer.

    The series grid endpoint above already aggregates the same shape from
    `comic_series` + `file_index`. This route is the same query with
    `cs.id = ?` and a single-row read so the reader can populate its
    series-info card lazily — used when a chapter is opened via the flat
    Files panel (no series context threaded through), so the drawer can
    show description / status / counts without bouncing through the full
    list endpoint.

    Shape matches one element of `/comics/series.series[]` exactly so the
    client doesn't need a separate decoder.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if not series_id:
        return JSONResponse({"error": "Missing series_id"}, status_code=400)

    sql = """
        SELECT
            cs.id                   AS series_id,
            cs.canonical_name,
            cs.author,
            cs.publisher,
            cs.description,
            cs.genres,
            cs.status,
            cs.year_started,
            cs.year_ended,
            cs.language_iso,
            cs.accent_color,
            COUNT(fi.id) AS chapter_count,
            SUM(CASE
                WHEN json_extract(fi.source_metadata, '$.is_finished') = 1
                THEN 1 ELSE 0 END
            ) AS finished_count,
            SUM(CASE
                WHEN CAST(COALESCE(json_extract(fi.source_metadata,
                                                '$.current_time_s'),
                                    0) AS REAL) > 0
                 AND COALESCE(json_extract(fi.source_metadata,
                                            '$.is_finished'), 0) != 1
                THEN 1 ELSE 0 END
            ) AS in_progress_count,
            MIN(fi.id) AS sample_chapter_file_id,
            MAX(fi.updated_at) AS last_updated
        FROM comic_series cs
        LEFT JOIN file_index fi
            ON fi.series_id = cs.id
           AND fi.user_id = cs.user_id
           AND fi.is_trashed = 0
        WHERE cs.user_id = ? AND cs.id = ?
        GROUP BY cs.id
        LIMIT 1
    """
    try:
        cursor = await idx._db.execute(sql, (uid, series_id))
        row = await cursor.fetchone()
    except Exception as exc:
        log.warning("comic_series_detail_query_failed",
                    series_id=series_id, error=str(exc))
        return JSONResponse({"error": "Lookup failed"}, status_code=500)
    if not row:
        return JSONResponse({"error": "Series not found"}, status_code=404)

    import json as _json
    try:
        genres = _json.loads(row["genres"] or "[]")
        if not isinstance(genres, list):
            genres = []
    except (_json.JSONDecodeError, TypeError):
        genres = []

    total = int(row["chapter_count"] or 0)
    finished = int(row["finished_count"] or 0)
    in_progress = int(row["in_progress_count"] or 0)
    return JSONResponse({
        "id":                row["series_id"],
        "name":              row["canonical_name"],
        "author":            row["author"] or "",
        "publisher":         row["publisher"] or "",
        "description":       row["description"] or "",
        "genres":            genres,
        "status":            row["status"] or "",
        "year_started":      row["year_started"],
        "year_ended":        row["year_ended"],
        "language_iso":      row["language_iso"] or "",
        "accent_color":      row["accent_color"] or "",
        "chapter_count":     total,
        "finished_count":    finished,
        "in_progress_count": in_progress,
        "unread_count":      max(0, total - finished - in_progress),
        "cover_file_id":     row["sample_chapter_file_id"] or "",
        "last_updated":      row["last_updated"] or "",
    })


@router.get("/comics/genres")
async def list_comic_genres(request: Request):
    """Distinct genres across the user's comic library, with counts.

    Powers the genre filter dropdown on the Comics chip. Each genre in
    the ``comic_series.genres`` JSON array is unnested via ``json_each``,
    normalised to lowercase for grouping, and returned with the count of
    series that carry it so the UI can rank "romance (42)" above "isekai
    (3)" without a client-side pass. Trims obvious non-informative
    strings ("", "unknown") so the dropdown isn't cluttered.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"genres": []})

    async def _compute() -> list[dict[str, Any]]:
        sql = """
            SELECT LOWER(TRIM(g.value)) AS genre_key,
                   COUNT(DISTINCT cs.id) AS series_count
            FROM comic_series cs,
                 json_each(COALESCE(cs.genres, '[]')) g
            WHERE cs.user_id = ?
              AND TRIM(g.value) != ''
              AND LOWER(TRIM(g.value)) NOT IN ('unknown', 'n/a')
            GROUP BY genre_key
            ORDER BY series_count DESC, genre_key ASC
        """
        try:
            cursor = await idx._db.execute(sql, (uid,))
            rows = await cursor.fetchall()
        except Exception as exc:
            # Missing comic_series table (fresh install without comics) or
            # json_each unavailable on older SQLite builds → return empty
            # rather than 500. The dropdown gracefully hides itself.
            log.warning("comic_genres_query_failed", error=str(exc))
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            key = r["genre_key"]
            if not key:
                continue
            # Re-case for display: "romance" → "Romance", "slice of life" →
            # "Slice of Life". Keeps the filter dropdown looking hand-curated
            # even when upstream providers write arbitrary-case strings.
            display = " ".join(word.capitalize() for word in key.split())
            out.append({
                "key":   key,
                "label": display,
                "count": int(r["series_count"] or 0),
            })
        return out

    genres = await _COMIC_GENRES_CACHE.get_or_compute(uid, _compute)
    return JSONResponse({"genres": genres})


@router.get("/video/genres")
async def list_video_genres(request: Request):
    """Distinct genres across the user's video rows (shows + movies +
    music videos), with counts.

    Mirrors the comics genre endpoint but reads from ``file_index``
    where Emby/Jellyfin sync writes ``source_metadata.extra.genres``.
    Powers the Filter dropdown's Genre section on the video chips.

    Series are counted at the series-level (entity_kind=series) where
    available so a 13-episode show counts as one Drama, not thirteen;
    movies count individually since each is its own entity. Episodes
    and seasons are excluded from the count to prevent inflation.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"genres": []})

    # Provider sources that write rich video metadata. New providers
    # added later (Plex, Trakt, etc.) get added here. The list mirrors
    # MEDIA_SOURCES for video on the frontend.
    video_sources = ("emby", "jellyfin")

    async def _compute() -> list[dict[str, Any]]:
        # Reads from the canonical top-level paths in source_metadata
        # (`genres`, `entity_kind`) — these are the explicitly mapped
        # fields in augmentum.media.sync._video_to_row. The same data
        # also exists under $.extra.genres / $.extra.entity_kind as a
        # secondary copy of item.extra; falling back via COALESCE means
        # the query keeps working even if a future sync change breaks
        # one of the two.
        sql = """
            SELECT LOWER(TRIM(g.value)) AS genre_key,
                   COUNT(DISTINCT fi.id) AS row_count
            FROM file_index fi,
                 json_each(
                     COALESCE(
                         json_extract(fi.source_metadata, '$.genres'),
                         json_extract(fi.source_metadata, '$.extra.genres'),
                         '[]'
                     )
                 ) g
            WHERE fi.user_id = ?
              AND fi.is_trashed = 0
              AND fi.source IN ({sources})
              AND TRIM(g.value) != ''
              AND LOWER(TRIM(g.value)) NOT IN ('unknown', 'n/a')
              -- Roll up to series-level for shows; keep movies as-is.
              -- Episodes + seasons would inflate counts artificially.
              AND COALESCE(
                      json_extract(fi.source_metadata, '$.entity_kind'),
                      json_extract(fi.source_metadata, '$.extra.entity_kind'),
                      ''
                  ) IN ('series', 'movie', 'music_video')
            GROUP BY genre_key
            ORDER BY row_count DESC, genre_key ASC
        """.format(sources=",".join("?" * len(video_sources)))
        params = (uid, *video_sources)
        try:
            cursor = await idx._db.execute(sql, params)
            rows = await cursor.fetchall()
        except Exception as exc:
            log.warning("video_genres_query_failed", error=str(exc))
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            key = r["genre_key"]
            if not key:
                continue
            display = " ".join(word.capitalize() for word in key.split())
            out.append({
                "key":   key,
                "label": display,
                "count": int(r["row_count"] or 0),
            })
        return out

    genres = await _VIDEO_GENRES_CACHE.get_or_compute(uid, _compute)
    return JSONResponse({"genres": genres})


@router.get("/comics/series/{series_id}/chapters")
async def list_series_chapters(series_id: str, request: Request):
    """Every chapter in one series, ordered by source-provider's ordering.

    Serves the drill-down from a series card. Returns full ``FileEntry``
    rows so the existing card renderer works without translation — a
    chapter in the drill-down looks identical to the same chapter in the
    flat list view.

    Ordering: ``source_metadata.extra.chapter_source_order`` ascending.
    That's how Suwayomi/Komga/Kavita address chapters internally, so this
    matches the in-library reading order the user sees on the source.
    """
    uid = _user_id(request)
    idx = _get_index(request)
    if not uid or not idx:
        return JSONResponse({"files": [], "has_more": False, "offset": 0})

    limit, offset, _ = _paging(request, default_limit=500, max_limit=2000)

    # sourceOrder sort: chapter 1 before chapter 2, etc. Tie-break on id
    # so the order is stable across refreshes even if multiple chapters
    # share a sourceOrder (shouldn't happen but defensive).
    #
    # MUST use the explicit ``_ENTRY_COLUMNS`` list, not ``SELECT *``:
    # ``_row_to_entry`` is built around _ENTRY_COLUMNS (no embedding BLOB,
    # shifted positions). ``SELECT *`` returns columns in physical table
    # order which puts embedding at index 11 and pushes source_metadata
    # off the position _row_to_entry reads it from — chapters come back
    # with empty source_metadata, breaking is_finished / chapter_source_
    # _order / progress display and the "mark up to here" action.
    sql = f"""
        SELECT {_ENTRY_COLUMNS} FROM file_index
        WHERE series_id = ? AND user_id = ? AND is_trashed = 0
        ORDER BY
            CAST(COALESCE(
                json_extract(source_metadata, '$.extra.chapter_source_order'),
                0
            ) AS INTEGER) ASC,
            id ASC
        LIMIT ? OFFSET ?
    """
    cursor = await idx._db.execute(sql, (series_id, uid, limit, offset))
    rows = await cursor.fetchall()
    results = [e for r in rows if (e := idx._row_to_entry(r))]

    return JSONResponse({
        "files":    [f.to_dict() for f in results],
        "has_more": len(results) == limit,
        "offset":   offset,
    })


@router.get("/favorites")
async def list_favorites(request: Request):
    """List favorited files, paginated."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"files": [], "has_more": False, "offset": 0})
    limit, offset, sort = _paging(request)
    query = request.query_params.get("q") or None
    results = await idx.list_favorites(
        user_id=uid, limit=limit, offset=offset, sort=sort, query=query,
    )
    return JSONResponse({
        "files": [f.to_dict() for f in results],
        "has_more": len(results) == limit,
        "offset": offset,
    })


@router.get("/list/{source}")
async def list_by_source(source: str, request: Request):
    """List files from a specific source."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"files": [], "has_more": False, "offset": 0})

    limit, offset, sort = _paging(request, default_limit=50)
    results = await idx.list_by_source(
        source, user_id=uid, limit=limit, offset=offset, sort=sort,
    )
    return JSONResponse({
        "files": [f.to_dict() for f in results],
        "has_more": len(results) == limit,
        "offset": offset,
    })


@router.get("/stats")
async def file_stats(request: Request):
    """Return file counts and sizes by source.

    Cached per user for ~30s (see ``_FILE_STATS_CACHE``). The endpoint
    runs ~8 sequential GROUP BY scans over file_index for chip-badge
    counts; uncached the call is the dominant Files-tab latency.
    """
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"by_source": {}, "total_count": 0, "total_size": 0})

    async def _compute_stats() -> dict[str, Any]:
        return await _build_file_stats(request, idx, uid)

    data = await _FILE_STATS_CACHE.get_or_compute(uid, _compute_stats)
    return JSONResponse(data)


async def _build_file_stats(request: Request, idx: Any, uid: str) -> dict[str, Any]:
    """Run the full chip-badge aggregation. Extracted so ``file_stats``
    can route through the per-user TTL cache without intermediate dicts
    leaking across users.
    """
    data = await idx.stats(user_id=uid)

    # Materialise synthetic group counts (e.g. "audiobooks" = ABS + LibriVox)
    # so the chip badge has a number to show without the frontend needing
    # to sum per-provider counts itself.
    by_source = data.get("by_source") or {}
    for group, members in _SOURCE_GROUPS.items():
        count = sum((by_source.get(m) or {}).get("count", 0) for m in members)
        size = sum((by_source.get(m) or {}).get("size_bytes", 0) for m in members)
        if count or size:
            by_source[group] = {"count": count, "size_bytes": size}

    async def _filtered_source_totals(
        *,
        sources: list[str],
        entity_kinds: list[str] | None = None,
        exclude_entity_kinds: list[str] | None = None,
    ) -> tuple[int, int]:
        if not sources:
            return 0, 0
        src_placeholders = ", ".join("?" for _ in sources)
        sql = (
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) "
            "FROM file_index "
            f"WHERE user_id = ? AND is_trashed = 0 AND source IN ({src_placeholders})"
        )
        params: list[Any] = [uid, *sources]
        if entity_kinds:
            placeholders = ", ".join("?" for _ in entity_kinds)
            sql += (
                " AND COALESCE(json_extract(source_metadata, '$.entity_kind'), '') "
                f"IN ({placeholders})"
            )
            params.extend(entity_kinds)
        if exclude_entity_kinds:
            placeholders = ", ".join("?" for _ in exclude_entity_kinds)
            sql += (
                " AND COALESCE(json_extract(source_metadata, '$.entity_kind'), '') "
                f"NOT IN ({placeholders})"
            )
            params.extend(exclude_entity_kinds)
        cursor = await idx._db.execute(sql, params)
        row = await cursor.fetchone()
        return (
            int(row[0] or 0) if row else 0,
            int(row[1] or 0) if row else 0,
        )

    # Groups with entity-kind filters cannot be derived by simply summing
    # provider totals because the same provider can contain multiple media
    # types (ABS books + podcasts). Recompute those group badges exactly.
    try:
        for group, cfg in _SOURCE_GROUP_ENTITY_FILTERS.items():
            group_sources = list(_SOURCE_GROUPS.get(group) or [])
            count, size = await _filtered_source_totals(
                sources=group_sources,
                entity_kinds=list(cfg.get("entity_kinds") or []),
                exclude_entity_kinds=list(cfg.get("exclude_entity_kinds") or []),
            )
            if count or size:
                by_source[group] = {"count": count, "size_bytes": size}
            else:
                by_source.pop(group, None)
    except Exception as exc:
        log.debug("files_stats_group_aggregation_failed", error=str(exc))

    # Comics show as series, not chapters, throughout the UI — ``20k+
    # chapter rows`` is just the wire shape; the user sees ``~500 series``.
    # Count the user's comic_series rows and substitute into the Comics
    # chip badge + All tab count so the numbers match what the grid renders.
    comic_series_count = 0
    try:
        cursor = await idx._db.execute(
            "SELECT COUNT(*) FROM comic_series WHERE user_id = ?", (uid,),
        )
        row = await cursor.fetchone()
        comic_series_count = int(row[0]) if row else 0
    except Exception:
        # comic_series table may be missing on fresh installs pre-migration
        # 101; count stays 0 and the Comics chip badge is empty. Harmless.
        pass

    # Kind counts split by scope — lets the Files panel's chip badges show
    # accurate numbers for "Audio", "Documents", etc. based on which
    # scope the user is currently in. Without this, the Local/Audio chip
    # showed the GLOBAL audio count, including cloud audiobooks that the
    # scope=local filter would then strip from the list query (so the
    # badge said "120" but the grid was empty).
    by_kind_by_scope: dict[str, dict[str, dict[str, int]]] = {
        "local": {}, "cloud": {},
    }
    try:
        cloud_list = list(_CLOUD_SOURCES)
        placeholders = ", ".join("?" for _ in cloud_list)
        cursor = await idx._db.execute(
            f"""SELECT
                  CASE WHEN source IN ({placeholders}) THEN 'cloud' ELSE 'local' END AS scope,
                  kind, COUNT(*), COALESCE(SUM(size_bytes), 0)
                FROM file_index
                WHERE user_id = ? AND is_trashed = 0 AND kind != ''
                GROUP BY scope, kind""",
            (*cloud_list, uid),
        )
        for scope_label, k, count, size in await cursor.fetchall():
            by_kind_by_scope.setdefault(scope_label, {})[k] = {
                "count":      int(count or 0),
                "size_bytes": int(size or 0),
            }
    except Exception:
        # Defensive — if the query fails, frontend falls back to the
        # global by_kind counts (the old, pre-scope behavior).
        pass
    data["by_kind_by_scope"] = by_kind_by_scope
    by_kind = data.get("by_kind") or {}
    chapter_count = int((by_kind.get("comic") or {}).get("count", 0))
    by_source["comics"] = {
        "count":      comic_series_count,
        "size_bytes": 0,
    }
    try:
        for slug, cfg in _VIDEO_CHIP_FILTERS.items():
            entity_list = list(cfg.get("entity_kinds") or [])
            source_list = list(cfg.get("sources") or [])
            if not entity_list or not source_list:
                continue
            src_placeholders = ", ".join("?" for _ in source_list)
            ent_placeholders = ", ".join("?" for _ in entity_list)
            cursor = await idx._db.execute(
                f"""SELECT COUNT(*), COALESCE(SUM(size_bytes), 0)
                    FROM file_index
                    WHERE user_id = ? AND is_trashed = 0
                      AND source IN ({src_placeholders})
                      AND COALESCE(json_extract(source_metadata, '$.entity_kind'), '')
                          IN ({ent_placeholders})""",
                (uid, *source_list, *entity_list),
            )
            row = await cursor.fetchone()
            count = int(row[0] or 0) if row else 0
            size = int(row[1] or 0) if row else 0
            if count or size:
                by_source[slug] = {"count": count, "size_bytes": size}
    except Exception as exc:
        log.debug("files_stats_per_source_query_failed", error=str(exc))
    data["by_source"] = by_source

    hidden_video_detail_count = 0
    try:
        cursor = await idx._db.execute(
            """SELECT COUNT(*)
               FROM file_index
               WHERE user_id = ? AND is_trashed = 0
                 AND source IN (?, ?)
                 AND COALESCE(json_extract(source_metadata, '$.entity_kind'), '')
                     IN ('season', 'episode')""",
            (uid, *_VIDEO_PROVIDER_SOURCES),
        )
        row = await cursor.fetchone()
        hidden_video_detail_count = int(row[0] or 0) if row else 0
    except Exception as exc:
        log.debug("files_stats_video_detail_count_failed", error=str(exc))

    # Scope totals — the Local/Cloud pill badges. Cloud count rolls up
    # audiobooks + series (NOT raw chapters), matching what the user
    # actually sees in the grid when they click into a scope.
    raw_cloud_count = sum(
        (by_source.get(s) or {}).get("count", 0) for s in _CLOUD_SOURCES
    )
    cloud_count = max(
        0,
        raw_cloud_count - chapter_count - hidden_video_detail_count,
    ) + comic_series_count
    cloud_size = sum(
        (by_source.get(s) or {}).get("size_bytes", 0) for s in _CLOUD_SOURCES
    )
    total_count = int(data.get("total_count") or 0)
    total_size = int(data.get("total_size") or 0)
    local_count = max(0, total_count - raw_cloud_count)
    data["by_scope"] = {
        "local": {
            "count":      local_count,
            "size_bytes": max(0, total_size - cloud_size),
        },
        "cloud": {
            "count":      cloud_count,
            "size_bytes": cloud_size,
        },
    }

    # Bolt on blob-level stats so the UI can surface dedup savings
    blob_store = getattr(request.app.state, "blob_store", None)
    if blob_store:
        try:
            data["blobs"] = await blob_store.totals()
        except Exception as exc:
            log.debug("files_blob_totals_failed", error=str(exc))
    return data


@router.post("/upload")
async def upload_files(request: Request):
    """Accept one or more user-uploaded files. Multipart form — any field
    name; every file part is stored. Dedups by SHA-256: identical bytes
    across multiple uploads share a single blob on disk.

    Hardening:
      * Per-file size cap (settings.files_upload_max_file_bytes)
      * Per-request file count cap (settings.files_upload_max_files_per_request)
      * Per-request aggregate cap (settings.files_upload_max_request_bytes)
      * Per-user storage quota (settings.files_user_storage_quota_bytes; 0 = off)
      * Filename sanitisation (path traversal, control chars, reserved names)
      * MIME sniffing (server-detected MIME stored alongside client claim)
    """
    uid = _user_id(request)
    adapter = getattr(request.app.state, "uploads_adapter", None)
    if not adapter or not uid:
        return JSONResponse({"error": "Uploads not available"}, status_code=503)

    max_file_bytes = settings.files_upload_max_file_bytes
    max_files = settings.files_upload_max_files_per_request
    max_total_bytes = settings.files_upload_max_request_bytes
    quota_bytes = settings.files_user_storage_quota_bytes

    # Pre-fetch the user's current usage once; we update locally as we go
    # so a single multi-file POST can't sneak past the quota by racing
    # itself against repeated DB reads.
    used_bytes = await adapter.storage_used(uid) if quota_bytes > 0 else 0

    form = await request.form()
    results: list[dict] = []
    errors: list[dict] = []
    total_bytes = 0
    for _key, value in form.multi_items():
        if not hasattr(value, "read") or not hasattr(value, "filename"):
            continue
        raw_name = getattr(value, "filename", None) or "upload"
        filename = sanitize_filename(raw_name)
        if len(results) + len(errors) >= max_files:
            errors.append({"filename": filename,
                           "error": f"too many files in one request (max {max_files})"})
            break
        claimed_mime = getattr(value, "content_type", None) or ""
        try:
            data = await value.read()
        except Exception as err:
            errors.append({"filename": filename, "error": f"read failed: {err}"})
            continue
        if not data:
            errors.append({"filename": filename, "error": "empty file"})
            continue
        size = len(data)
        if size > max_file_bytes:
            errors.append({"filename": filename,
                           "error": f"file exceeds per-file limit ({max_file_bytes} bytes)"})
            continue
        total_bytes += size
        if total_bytes > max_total_bytes:
            errors.append({"filename": filename,
                           "error": f"request exceeds aggregate limit ({max_total_bytes} bytes)"})
            break
        if quota_bytes > 0 and used_bytes + size > quota_bytes:
            errors.append({
                "filename": filename,
                "error": f"storage quota exceeded ({used_bytes + size} > {quota_bytes} bytes)",
            })
            break
        sniffed_mime = sniff_mime(data, fallback=claimed_mime)
        try:
            info = await adapter.save(
                data, filename,
                mime_type=claimed_mime,
                mime_sniffed=sniffed_mime,
                user_id=uid,
            )
            if is_mime_mismatch(claimed_mime, sniffed_mime):
                info["mime_mismatch"] = True
                log.warning(
                    "upload_mime_mismatch", id=info.get("id"), user_id=uid,
                    filename=filename, claimed=claimed_mime, sniffed=sniffed_mime,
                )
            results.append(info)
            used_bytes += size
        except Exception as err:
            log.warning("upload_failed", filename=filename, err=str(err))
            errors.append({"filename": filename, "error": str(err)})

    if results or errors:
        await _audit(
            request, "file.upload",
            detail=f"ok={len(results)} err={len(errors)} bytes={total_bytes}",
        )
    return JSONResponse({"uploaded": results, "errors": errors})


@router.post("/bookmarks")
async def save_bookmark(request: Request):
    """Save a video/article URL to the Files panel.

    Idempotent — re-saving the same URL updates the title/thumbnail
    rather than creating a duplicate row (source_id is a hash of the
    URL).  Bookmarks appear under the Videos kind chip alongside
    uploaded video files.

    Body: {url, title, thumbnail?, channel?, duration?, platform?, video_id?}
    """
    adapter = getattr(request.app.state, "bookmarks_adapter", None)
    uid = _user_id(request)
    if not adapter or not uid:
        return JSONResponse({"error": "Bookmarks not available"}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    url = (body.get("url") or "").strip()
    title = sanitize_filename(body.get("title") or "Untitled video")
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    if not (url.startswith("http://") or url.startswith("https://")):
        return JSONResponse({"error": "url must be http(s)"}, status_code=400)

    duration = body.get("duration")
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None

    try:
        info = await adapter.save(
            url=url, title=title, user_id=uid,
            thumbnail=str(body.get("thumbnail") or "")[:500],
            channel=str(body.get("channel") or "")[:200],
            duration=duration,
            platform=str(body.get("platform") or "")[:32],
            video_id=str(body.get("video_id") or "")[:64],
        )
    except ValueError as err:
        return JSONResponse({"error": str(err)}, status_code=400)
    except Exception as err:
        log.warning("bookmark_save_failed", url=url, err=str(err))
        return JSONResponse({"error": "save failed"}, status_code=500)

    await _audit(request, "file.bookmark", detail=f"url={url[:200]}")
    return JSONResponse({"ok": True, "bookmark": info})


@router.get("/entry/{file_id}")
async def get_file_by_id(file_id: str, request: Request):
    """Single file_index row as JSON — for deep-link navigation.

    Used by the media detail view's "Also by X" strip and the mini-
    player expand handler when the target file isn't in the currently
    loaded grid page. User-scoped: a guess of someone else's id resolves
    to 404 at ``idx.get``'s WHERE user_id = ? clause.

    Path prefix is ``/entry/`` (not a bare ``/{file_id}``) to avoid
    colliding with existing named routes like ``/browse``, ``/stats``,
    etc. — FastAPI matches by declaration order, and a catch-all here
    would swallow any new named GET added under /api/files/ later.
    """
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)
    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(entry.to_dict())


@router.get("/thumb/{file_id}")
async def get_file_thumb(file_id: str, request: Request):
    """Cached, size-capped thumbnail for a file_index row.

    Canonical entry point for every gallery/grid/tile in the app. Size
    must be one of ``150 / 300 / 800`` — anything else 400s. Returns a
    WebP stream with long-lived cache headers (new image → new file_id →
    new URL, so thumbs are safely immutable).
    """
    from fastapi.responses import FileResponse

    idx = _get_index(request)
    uid = _user_id(request)
    svc = getattr(request.app.state, "thumbnail_service", None)
    if not idx or not uid or not svc:
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        size = int(request.query_params.get("size", "300"))
    except ValueError:
        return JSONResponse({"error": "Invalid size"}, status_code=400)

    from augmentum.vfs.thumbnails import ALLOWED_SIZES
    if size not in ALLOWED_SIZES:
        return JSONResponse(
            {"error": f"size must be one of {sorted(ALLOWED_SIZES)}"},
            status_code=400,
        )

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    result = await svc.get(entry, size, user_id=uid)
    if not result:
        return JSONResponse({"error": "No thumbnail available"}, status_code=404)

    path, mime = result
    return FileResponse(
        path, media_type=mime,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/thumb/by-source/{source}/{source_id}")
async def get_thumb_by_source(source: str, source_id: str, request: Request):
    """Alternate thumbnail lookup for callers that only track (source, id).

    The image gallery works in image_ids, not file_ids, so wiring the
    gallery to the canonical ``/thumb/{file_id}`` route would require
    every history response to carry file_ids. This resolver lets callers
    keep their existing data shape and still share the cache + producer
    pipeline.
    """
    from fastapi.responses import FileResponse

    idx = _get_index(request)
    uid = _user_id(request)
    svc = getattr(request.app.state, "thumbnail_service", None)
    if not idx or not uid or not svc:
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        size = int(request.query_params.get("size", "300"))
    except ValueError:
        return JSONResponse({"error": "Invalid size"}, status_code=400)

    from augmentum.vfs.thumbnails import ALLOWED_SIZES
    if size not in ALLOWED_SIZES:
        return JSONResponse(
            {"error": f"size must be one of {sorted(ALLOWED_SIZES)}"},
            status_code=400,
        )

    entry = await idx.get_by_source(source, source_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)

    result = await svc.get(entry, size, user_id=uid)
    if not result:
        return JSONResponse({"error": "No thumbnail available"}, status_code=404)

    path, mime = result
    return FileResponse(
        path, media_type=mime,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/browse")
async def browse_vfs(request: Request):
    """Browse the virtual filesystem."""
    vfs = _get_vfs(request)
    uid = _user_id(request)
    if not vfs or not uid:
        return JSONResponse({"nodes": []})

    path = request.query_params.get("path", "/")
    nodes = await vfs.list(path, user_id=uid)
    return JSONResponse({
        "path": path,
        "nodes": [
            {"path": n.path, "name": n.name, "is_dir": n.is_dir,
             "size": n.size, "mime_type": n.mime_type, "source": n.source}
            for n in nodes
        ],
    })


async def _resolve_by_source(request: Request, entry, uid: str) -> tuple[bytes | str | None, str]:
    """Source-specific content lookup by source_id.

    Returns (payload, media_type). Payload is bytes OR a filesystem path
    (caller opens it). None on miss. Bridges are a last resort because their
    name-based lookups don't reliably find renamed/multi-extension files.
    """
    import os

    from augmentum.config import settings

    state = request.app.state
    mime = entry.mime_type or "application/octet-stream"

    if entry.source == "images":
        persistence = getattr(state, "image_persistence", None)
        if persistence:
            try:
                gen = await persistence.get_generation(entry.source_id, user_id=uid)
            except Exception:
                gen = None
            fp = (gen or {}).get("file_path") or ""
            if fp and os.path.exists(fp):
                return fp, mime or "image/png"
        # Cloud-generated images land in image_output dir by id
        output_dir = getattr(settings, "image_output_dir", "") or f"{settings.data_dir}/image_output"
        fp = os.path.join(output_dir, f"{entry.source_id}.png")
        if os.path.exists(fp):
            return fp, mime or "image/png"
        return None, mime

    if entry.source == "artifacts":
        store = getattr(state, "artifact_store", None)
        if store:
            try:
                info = await store.get(entry.source_id, user_id=uid)
            except Exception:
                info = None
            if info and info.get("path"):
                full = store.get_file_path(info["path"])
                if full and full.is_file():
                    return str(full), mime
        return None, mime

    if entry.source == "chat_images":
        # Chat image blobs live in the chat_images table; read by id directly
        backend = getattr(state, "state_manager", None)
        conn = getattr(backend, "_backend", None) if backend else None
        if not conn:
            conn = getattr(state, "sqlite_backend", None)
        db = getattr(conn, "_conn", conn)
        if db is not None:
            try:
                cur = await db.execute(
                    "SELECT data, mime_type FROM chat_images WHERE id = ? AND user_id = ?",
                    (entry.source_id, uid),
                )
                row = await cur.fetchone()
                if row and row[0]:
                    return row[0], row[1] or mime or "image/png"
            except Exception as exc:
                log.debug(
                    "files_resolve_chat_image_failed",
                    source_id=entry.source_id,
                    error=str(exc),
                )
        return None, mime

    if entry.source == "documents":
        # Documents are chunked in the DB; no reassembly path — nothing to serve
        return None, mime

    if entry.source == "voices":
        voices_dir = str(Path(settings.data_dir) / "voices")
        fp = os.path.join(voices_dir, entry.name)
        if os.path.isfile(fp):
            return fp, mime
        return None, mime

    # Registry fallthrough — any adapter registered at startup
    # (uploads, future Dropbox/S3/etc.) handles its own source slug
    # without needing a hardcoded branch here.
    from augmentum.vfs import get_adapter
    adapter = get_adapter(entry.source)
    if adapter:
        try:
            resolved = await adapter.resolve(entry.source_id, user_id=uid)
            if resolved:
                return resolved, mime
        except Exception:
            log.warning("adapter_resolve_failed", source=entry.source, exc_info=True)
    return None, mime


@router.get("/download/{file_id}")
async def download_file(file_id: str, request: Request):
    """Download a file by ID. Resolves by source_id, not filename.

    Order: fast-path real_path on disk → source-specific lookup →
    last-resort VFS bridge by name. When the first two miss but the row still
    points at something that exists, we repair real_path inline so the next
    request short-circuits.

    Disk-backed responses use Starlette's FileResponse so the body is
    streamed (no whole-file read into memory) and Range requests get a
    proper 206 Partial Content reply — important for video seeking,
    progressive PDF load, and not OOM-ing on large uploads.
    """
    import os

    from fastapi.responses import FileResponse

    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Streaming sources (Audiobookshelf, LibriVox, ...) have no bytes on
    # disk to serve — playback goes through /api/media/stream which handles
    # chapter selection, Range forwarding, and upstream auth. Redirect any
    # stray /api/files/download/ hit at that path so audio elements or
    # external links pointed at the wrong endpoint still play (chapter 0).
    if entry.source in _MEDIA_STREAMING_SOURCES:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(
            url=f"/api/media/stream/{file_id}?file=0", status_code=302,
        )

    mime = entry.mime_type or "application/octet-stream"
    headers = {"Content-Disposition": f'attachment; filename="{entry.name}"'}

    # Fast path
    if entry.real_path and os.path.exists(entry.real_path):
        return FileResponse(
            entry.real_path, media_type=mime, headers=headers, filename=entry.name,
        )

    # Source-specific resolve
    payload, resolved_mime = await _resolve_by_source(request, entry, uid)
    if payload is not None:
        media = resolved_mime or mime
        if isinstance(payload, str):
            # Filesystem path — heal real_path so future requests skip the lookup
            try:
                if payload != (entry.real_path or ""):
                    await getattr(request.app.state, "file_index", None)._db.execute(
                        "UPDATE file_index SET real_path = ?, "
                        "size_bytes = CASE WHEN size_bytes > 0 THEN size_bytes ELSE ? END, "
                        "updated_at = datetime('now') WHERE id = ?",
                        (payload, os.path.getsize(payload), entry.id),
                    )
                    await getattr(request.app.state, "file_index", None)._db.commit()
            except Exception as exc:
                # Was debug — but if the heal silently fails, every future
                # download of this file re-runs the expensive source lookup
                # forever with no operator signal. Warning at first failure
                # surfaces a stuck heal pattern early (e.g. WAL locked,
                # FK constraint, disk full).
                log.warning("files_real_path_heal_failed", entry_id=entry.id, error=str(exc))
            return FileResponse(
                payload, media_type=media, headers=headers, filename=entry.name,
            )
        # bytes — small payloads (chat_images blobs etc.); in-memory is fine here
        return Response(content=payload, media_type=media, headers=headers)

    # Last resort: VFS by name (covers voices with unusual filenames)
    vfs = _get_vfs(request)
    if vfs:
        source_prefix = {
            "artifacts": "/Artifacts", "images": "/Images",
            "chat_images": "/Chat Images",
            "voices": "/Voices", "documents": "/Documents",
        }.get(entry.source, "")
        if source_prefix:
            try:
                data = await vfs.read(f"{source_prefix}/{entry.name}", user_id=uid)
            except Exception:
                data = None
            if data:
                return Response(content=data, media_type=mime, headers=headers)

    return JSONResponse({"error": "File content not available"}, status_code=404)


@router.post("/bulk-delete")
async def bulk_delete_files(request: Request):
    """Soft-delete multiple files."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not available"}, status_code=404)

    body = await request.json()
    ids = body.get("ids", [])
    if not ids or len(ids) > 200:
        return JSONResponse({"error": "Provide 1-200 IDs"}, status_code=400)

    deleted = 0
    for fid in ids:
        if await idx.soft_delete(fid, user_id=uid):
            deleted += 1
    await _audit(request, "file.bulk_delete", detail=f"requested={len(ids)} deleted={deleted}")
    return JSONResponse({"ok": True, "deleted": deleted})


@router.post("/favorite/{file_id}")
async def toggle_favorite(file_id: str, request: Request):
    """Toggle favorite status for a file."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)
    is_fav = await idx.toggle_favorite(file_id, user_id=uid)
    return JSONResponse({"ok": True, "is_favorite": is_fav})


@router.get("/trash")
async def list_trash(request: Request):
    """List trashed files — accepts `sort` and `q` for parity with the
    rest of the file listing endpoints. Default sort (when unspecified)
    is deleted-time DESC so the "I just deleted the wrong thing" flow
    still lands the target row near the top."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"files": [], "has_more": False, "offset": 0})
    limit, offset, sort = _paging(request)
    query = request.query_params.get("q") or None
    results = await idx.list_trash(
        user_id=uid, limit=limit, offset=offset, sort=sort, query=query,
    )
    return JSONResponse({
        "files": [f.to_dict() for f in results],
        "has_more": len(results) == limit,
        "offset": offset,
    })


@router.post("/restore/{file_id}")
async def restore_file(file_id: str, request: Request):
    """Restore a file from trash."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)
    ok = await idx.restore(file_id, user_id=uid)
    if not ok:
        return JSONResponse({"error": "File not in trash"}, status_code=404)
    await _audit(request, "file.restore", detail=f"id={file_id}")
    return JSONResponse({"ok": True})


@router.post("/bulk-restore")
async def bulk_restore_files(request: Request):
    """Restore multiple files from trash in one call."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not available"}, status_code=404)

    body = await request.json()
    ids = body.get("ids", [])
    if not ids or len(ids) > 200:
        return JSONResponse({"error": "Provide 1-200 IDs"}, status_code=400)

    restored = 0
    for fid in ids:
        if await idx.restore(fid, user_id=uid):
            restored += 1
    await _audit(request, "file.bulk_restore", detail=f"requested={len(ids)} restored={restored}")
    return JSONResponse({"ok": True, "restored": restored})


@router.post("/purge-trash")
async def purge_trash(request: Request):
    """Permanently delete all trashed files.

    For rows whose source has a registered adapter (uploads, future cloud
    sources), delegate to the adapter's delete path so blob refcounts get
    released and physical files actually go away. Rows without an adapter
    (images / artifacts / etc. — those sources are deleted from their own
    native UIs) get their file_index row dropped only.
    """
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not available"}, status_code=404)

    from augmentum.vfs import get_adapter

    # Gather trashed rows first so we can dispatch per-source before the
    # index wipes them. Cheap because list_trash is paginated + indexed.
    trashed = await idx.list_trash(user_id=uid, limit=1000, offset=0)
    adapter_deleted = 0
    for entry in trashed:
        adapter = get_adapter(entry.source)
        if adapter:
            try:
                if await adapter.delete(entry.source_id, user_id=uid):
                    adapter_deleted += 1
            except Exception:
                log.warning("adapter_purge_failed", source=entry.source, exc_info=True)

    # Whatever the adapters didn't cover (native-source rows) drops via the
    # straight index purge. Adapter.delete() already removed their index
    # rows, so this only catches the non-adapter leftovers.
    deleted = await idx.purge_trash(user_id=uid)
    total = deleted + adapter_deleted
    await _audit(
        request, "file.purge_trash",
        detail=f"deleted={total} adapter_deleted={adapter_deleted}",
    )
    return JSONResponse({
        "ok": True,
        "deleted": total,
        "adapter_deleted": adapter_deleted,
    })


@router.patch("/tags/{file_id}")
async def update_tags(file_id: str, request: Request):
    """Update tags for a file. Tags are normalized (NFKC, dedup by canonical
    form) before persisting so casing/whitespace variants collapse.
    """
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)
    body = await request.json()
    raw_tags = body.get("tags", [])
    if not isinstance(raw_tags, list) or len(raw_tags) > 50:
        return JSONResponse({"error": "Tags must be a list (max 50)"}, status_code=400)
    tags = normalize_tags(raw_tags)
    ok = await idx.update_tags(file_id, tags=tags, user_id=uid)
    if not ok:
        return JSONResponse({"error": "File not found"}, status_code=404)
    return JSONResponse({"ok": True, "tags": tags})


@router.get("/tags/suggest")
async def tags_suggest(request: Request):
    """Autocomplete: return up to N tags matching the prefix, ranked by
    use frequency.  Backs the tag input UI so users gravitate to existing
    tags instead of inventing inconsistent variants.
    """
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"tags": []})
    q = request.query_params.get("q", "")
    try:
        limit = max(1, min(int(request.query_params.get("limit", "20")), 50))
    except ValueError:
        limit = 20
    # FileIndexService owns the connection — reach in rather than
    # threading another dependency through to the route. Documented
    # accessor would be cleaner; leaving that for a follow-up.
    suggestions = await suggest_tags(idx._db, user_id=uid, prefix=q, limit=limit)
    return JSONResponse({"tags": suggestions})


@router.post("/zip")
async def zip_download(request: Request):
    """Download multiple files as a zip archive."""
    import io
    import zipfile

    idx = _get_index(request)
    vfs = _get_vfs(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not available"}, status_code=404)

    body = await request.json()
    ids = body.get("ids", [])
    if not ids or len(ids) > 100:
        return JSONResponse({"error": "Provide 1-100 IDs"}, status_code=400)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fid in ids:
            entry = await idx.get(fid, user_id=uid)
            if not entry:
                continue
            data = None
            if entry.real_path:
                import os
                if os.path.exists(entry.real_path):
                    with open(entry.real_path, "rb") as f:
                        data = f.read()
            if data is None and vfs:
                source_prefix = {
                    "artifacts": "/Artifacts", "images": "/Images",
                    "chat_images": "/Chat Images",
                    "voices": "/Voices",
                }.get(entry.source, "")
                if source_prefix:
                    data = await vfs.read(f"{source_prefix}/{entry.name}", user_id=uid)
            if data:
                zf.writestr(entry.name, data)

    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="files.zip"'},
    )


@router.get("/preview/{file_id}")
async def file_preview(file_id: str, request: Request):
    """Return a small text snippet for content-peek tiles. Text/code only.

    Reads at most 4KB and returns up to 1200 chars of UTF-8 text. Returns an
    empty snippet (200) for binary or unreadable files so the UI can fall back
    to its source-themed default tile without erroring.
    """
    import os

    idx = _get_index(request)
    vfs = _get_vfs(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"snippet": "", "ext": ""}, status_code=404)
    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"snippet": "", "ext": ""}, status_code=404)

    data: bytes | None = None
    if entry.real_path:
        try:
            if os.path.exists(entry.real_path):
                with open(entry.real_path, "rb") as f:
                    data = f.read(4096)
        except OSError:
            data = None
    if data is None and vfs:
        prefix = {
            "artifacts": "/Artifacts", "images": "/Images",
            "chat_images": "/Chat Images",
            "voices": "/Voices", "documents": "/Documents",
        }.get(entry.source, "")
        if prefix:
            try:
                full = await vfs.read(f"{prefix}/{entry.name}", user_id=uid)
                if full:
                    data = full[:4096]
            except Exception:
                data = None

    ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
    if not data:
        return JSONResponse({"snippet": "", "ext": ext})

    # Reject obvious binary (null bytes in the head)
    if b"\x00" in data[:200]:
        return JSONResponse({"snippet": "", "ext": ext})

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return JSONResponse({"snippet": "", "ext": ext})

    return JSONResponse({"snippet": text[:1200], "ext": ext})


@router.get("/text/{file_id}")
async def file_text(file_id: str, request: Request):
    """Return the full extracted plain text of a file, for TTS / AI
    summarize / export. Handles UTF-8 text directly and binary
    documents (PDF / DOCX / PPTX / XLSX / EPUB) via the document
    parsers in augmentum.tools.document_parse.

    Capped at 2 MB of output to keep response bodies bounded — truly
    huge files should round-trip through /transform or /summarize
    which are designed for it. Returns { text, truncated, ext } on
    success; 404 if the file is missing, 415 if the file type isn't
    extractable, 501 if an optional decoder dep isn't installed.
    """
    import os

    idx = _get_index(request)
    vfs = _get_vfs(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "not found"}, status_code=404)
    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "not found"}, status_code=404)

    # Load the full file bytes (not capped like /preview — we need
    # the whole document). Cap on the OUTPUT text below, not input.
    data: bytes | None = None
    if entry.real_path:
        try:
            if os.path.exists(entry.real_path):
                with open(entry.real_path, "rb") as f:
                    data = f.read()
        except OSError:
            data = None
    if data is None and vfs:
        prefix = {
            "artifacts": "/Artifacts", "images": "/Images",
            "chat_images": "/Chat Images",
            "voices": "/Voices", "documents": "/Documents",
        }.get(entry.source, "")
        if prefix:
            try:
                data = await vfs.read(f"{prefix}/{entry.name}", user_id=uid)
            except Exception:
                data = None
    if data is None:
        return JSONResponse({"error": "could not load file"}, status_code=404)

    ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
    max_chars = 2_000_000  # 2 MB of text ≈ one big book

    # Plain-text family: decode directly, no parser needed.
    _TEXTUAL_EXTS = {
        "txt", "md", "markdown", "rst", "log", "csv", "tsv",
        "json", "jsonc", "yaml", "yml", "toml", "ini", "env",
        "xml", "html", "htm", "svg", "py", "pyi", "js", "mjs", "ts",
        "tsx", "jsx", "css", "scss", "less", "sh", "bash", "zsh",
        "rb", "go", "rs", "c", "h", "cpp", "hpp", "cs", "java",
        "kt", "swift", "php", "pl", "lua", "sql", "dockerfile",
        "makefile", "vue", "svelte",
    }
    if ext in _TEXTUAL_EXTS or not ext:
        # Null-byte sniff to reject true binaries that happen to lack
        # a distinguishing extension.
        if b"\x00" in data[:4096]:
            return JSONResponse(
                {"error": "binary content — not extractable as text"},
                status_code=415,
            )
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = data.decode("latin-1", errors="replace")
        truncated = len(text) > max_chars
        return JSONResponse({
            "text": text[:max_chars],
            "truncated": truncated,
            "ext": ext,
        })

    # Document family — dispatch to the shared parser registry.
    try:
        from augmentum.tools.document_parse import _PARSERS
    except ImportError:
        return JSONResponse(
            {"error": "document parsers unavailable"},
            status_code=501,
        )
    parser = _PARSERS.get(f".{ext}")
    if not parser:
        return JSONResponse(
            {"error": f"unsupported file type: {ext}"},
            status_code=415,
        )

    # Parsers take a filesystem path, so spill to a tempfile first.
    import tempfile
    suffix = f".{ext}"
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tf.write(data)
        tf.close()
        try:
            text = parser(tf.name)
        except ImportError as err:
            return JSONResponse(
                {"error": f"optional decoder missing: {err}"},
                status_code=501,
            )
        except Exception as err:
            return JSONResponse(
                {"error": f"extraction failed: {err}"},
                status_code=422,
            )
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass

    truncated = len(text) > max_chars
    return JSONResponse({
        "text": text[:max_chars],
        "truncated": truncated,
        "ext": ext,
    })


@router.get("/render/{file_id}")
async def render_file(file_id: str, request: Request):
    """Render a file as an HTML preview. Thin wrapper around the artifact
    preview helpers so files-mode can reuse the same EPUB/DOCX/PPTX/XLSX
    renderers regardless of which subsystem owns the row.
    """
    import os

    from fastapi.responses import FileResponse, HTMLResponse

    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Resolve to a filesystem path (the renderers all open by path)
    fp_str: str | None = None
    if entry.real_path and os.path.exists(entry.real_path):
        fp_str = entry.real_path
    else:
        payload, _ = await _resolve_by_source(request, entry, uid)
        if isinstance(payload, str) and os.path.exists(payload):
            fp_str = payload
        # bytes-only payloads (chat_images blobs) aren't anything we'd render —
        # browser handles those as images directly via /download.

    if not fp_str:
        # Documents are indexed as chunks — no whole-file preview available.
        # Return a styled shell instead of a 404 so the iframe doesn't show
        # the browser's default error page.
        from augmentum.proxy.artifact_routes import _preview_shell

        size_kb = f"{entry.size_bytes / 1024:.1f} KB" if entry.size_bytes else ""
        body = (
            '<div class="wrap" style="text-align:center;padding-top:48px">'
            '<div style="font-size:48px;margin-bottom:16px;opacity:0.3">\U0001F4C4</div>'
            f'<div style="margin-bottom:8px;font-size:16px;word-break:break-word">{entry.name}</div>'
            f'<div style="color:#6b6b80;margin-bottom:24px">{size_kb}</div>'
            '<div style="color:#6b6b80;margin:0 auto;line-height:1.55">'
            "Preview not available — this file was indexed as chunks, "
            "so there's no whole-file preview to render."
            '</div></div>'
        )
        from fastapi.responses import HTMLResponse
        return HTMLResponse(_preview_shell(entry.name, f"/api/files/download/{file_id}", body))

    fp = Path(fp_str)
    ext = entry.name.rsplit(".", 1)[-1].lower() if "." in entry.name else ""
    download_url = f"/api/files/download/{file_id}"
    display_name = entry.name

    from augmentum.proxy.artifact_routes import (
        _docx_to_html,
        _epub_to_html,
        _pptx_to_html,
        _preview_shell,
        _xlsx_to_html,
    )

    if ext == "pdf":
        return FileResponse(
            str(fp), media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )
    # Browser-unfriendly image formats — transcode to JPEG when we can.
    # HEIC/HEIF need pillow-heif (optional dep); TIFF works with stock Pillow.
    if ext in ("heic", "heif", "tif", "tiff"):
        rendered = _transcode_image_to_jpeg(fp, ext)
        if rendered is not None:
            return Response(content=rendered, media_type="image/jpeg",
                            headers={"Content-Disposition": "inline"})
        # No transcoder available → friendly fallback shell with download link.
        body = (
            '<div class="wrap" style="text-align:center;padding-top:48px">'
            '<div style="font-size:48px;margin-bottom:16px;opacity:0.3">\U0001F5BC</div>'
            f'<div style="margin-bottom:8px;font-size:16px;word-break:break-word">{display_name}</div>'
            f'<div style="color:#6b6b80;margin-bottom:24px">{ext.upper()} preview unavailable</div>'
            '<div style="color:#6b6b80;line-height:1.55;max-width:380px;margin:0 auto">'
            f"{ext.upper()} files require a server-side transcoder that isn't installed. "
            "Use Download to open in a native viewer."
            '</div></div>'
        )
        return HTMLResponse(_preview_shell(display_name, download_url, body))
    if ext in ("html", "htm"):
        try:
            return HTMLResponse(fp.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            return JSONResponse({"error": "Could not read file"}, status_code=500)
    if ext in ("zip", "tar", "gz", "tgz", "bz2"):
        html = _archive_to_html(fp, display_name, download_url, ext)
        if html:
            return HTMLResponse(html)
    # Document parsers (EPUB/DOCX/PPTX/XLSX) are sync, GIL-bound, and can take
    # hundreds of ms on large files. Hand off to a worker thread so the event
    # loop keeps serving the healthcheck and concurrent chat streams.
    if ext == "epub":
        html = await asyncio.to_thread(_epub_to_html, fp, display_name, download_url)
        if html:
            return HTMLResponse(html)
    if ext == "docx":
        html = await asyncio.to_thread(_docx_to_html, fp, display_name, download_url)
        if html:
            return HTMLResponse(html)
    if ext == "pptx":
        html = await asyncio.to_thread(_pptx_to_html, fp, display_name, download_url)
        if html:
            return HTMLResponse(html)
    if ext in ("xlsx", "csv"):
        html = await asyncio.to_thread(_xlsx_to_html, fp, ext, display_name, download_url)
        if html:
            return HTMLResponse(html)
    if ext in ("md", "markdown", "mdown", "mkd"):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return JSONResponse({"error": "Could not read file"}, status_code=500)
        # Defer to the same renderer artifact_document uses so formatting,
        # lists, headers, inline code all look right.
        from augmentum.tools.artifact_document import _md_to_html
        rendered = _md_to_html(text)
        body = (
            '<div class="wrap" style="font-size:14px;line-height:1.65;color:#d4d4df">'
            f'{rendered}'
            '</div>'
        )
        return HTMLResponse(_preview_shell(display_name, download_url, body))

    if ext == "json":
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return JSONResponse({"error": "Could not read file"}, status_code=500)
        # Pretty-print when possible; fall back to raw text on parse failure.
        import json as _json
        try:
            formatted = _json.dumps(_json.loads(raw), indent=2, ensure_ascii=False)
        except _json.JSONDecodeError:
            formatted = raw
        return HTMLResponse(_preview_shell(
            display_name, download_url, _highlight_text(formatted, "json"),
        ))

    if ext in ("txt", "rst", "log", "yaml", "yml", "toml", "ini", "cfg", "env"):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return JSONResponse({"error": "Could not read file"}, status_code=500)
        # Pygments handles yaml/toml/ini/rst with proper highlighting; txt/log/
        # env have no lexer and fall through the helper as plain styled blocks.
        return HTMLResponse(_preview_shell(
            display_name, download_url, _highlight_text(text, ext),
        ))

    # Source-code files — pygments by extension. Falls back to plain block
    # if a language is unknown or its lexer is missing.
    _CODE = {
        "py","js","ts","jsx","tsx","vue","svelte","rs","go","java","kt","swift",
        "cs","cpp","cc","c","h","hpp","rb","php","sh","bash","zsh","fish","sql",
        "css","scss","less","xml","html","htm",
    }
    if ext in _CODE:
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return JSONResponse({"error": "Could not read file"}, status_code=500)
        return HTMLResponse(_preview_shell(
            display_name, download_url, _highlight_text(text, ext),
        ))

    # Fallback: styled download card (uses shell .wrap for responsive padding)
    size_kb = f"{entry.size_bytes / 1024:.1f} KB"
    html = _preview_shell(
        display_name, download_url,
        '<div class="wrap" style="text-align:center;padding-top:48px">'
        '<div style="font-size:48px;margin-bottom:16px;opacity:0.3">\U0001F4C4</div>'
        f'<div style="margin-bottom:8px;font-size:16px;word-break:break-word">{display_name}</div>'
        f'<div style="color:#6b6b80;margin-bottom:24px">{ext.upper() or "FILE"} &middot; {size_kb}</div>'
        f'<a href="{download_url}" download style="color:#6c8aff;text-decoration:none;'
        'padding:10px 24px;border:1px solid #2d2d45;border-radius:8px;'
        'display:inline-block">Download</a>'
        '</div>',
    )
    return HTMLResponse(html)


@router.get("/epub-text/{file_id}")
async def epub_text_file(file_id: str, request: Request):
    """Return an indexed EPUB's spine as plain-text chapters.

    Mirrors :func:`render_file`'s path resolution so files-mode read-aloud
    works regardless of which subsystem owns the row. Fetched by the
    parent-side reader controls (ui/scripts/epub-reader-controls.js).
    """
    import os

    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)
    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "File not found"}, status_code=404)
    if not entry.name.lower().endswith(".epub"):
        return JSONResponse({"error": "Not an EPUB"}, status_code=400)

    fp_str: str | None = None
    if entry.real_path and os.path.exists(entry.real_path):
        fp_str = entry.real_path
    else:
        payload, _ = await _resolve_by_source(request, entry, uid)
        if isinstance(payload, str) and os.path.exists(payload):
            fp_str = payload
    if not fp_str:
        return JSONResponse({"error": "File not available on disk"}, status_code=404)

    from augmentum.vfs import epub_extractor

    chapters = await asyncio.to_thread(epub_extractor.chapters_text, fp_str)
    if not chapters:
        return JSONResponse({"error": "Could not extract text from this EPUB"}, status_code=422)
    return JSONResponse({"title": entry.name, "chapters": chapters})


@router.get("/narration/{file_id}")
async def get_file_narration(file_id: str, request: Request):
    """Status of this EPUB file's paired TTS narration (if any)."""
    from augmentum.proxy.narration_common import narration_status
    return JSONResponse(await narration_status(request, "file", file_id))


@router.post("/narration/{file_id}")
async def start_file_narration(file_id: str, request: Request, force: int = 0):
    """Record (synthesize) a TTS narration for this indexed EPUB.

    Idempotent unless ``force=1``. Body may carry ``{"voice": "...", "format": "mp3"|"wav"}``.
    """
    from augmentum.proxy.narration_common import narration_start

    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        raise HTTPException(404, "Not found")
    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        raise HTTPException(404, "File not found")
    if not entry.name.lower().endswith(".epub"):
        raise HTTPException(400, "Not an EPUB")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    body = body if isinstance(body, dict) else {}
    voice = str(body.get("voice") or "")
    out_fmt = str(body.get("format") or "mp3")
    return JSONResponse(await narration_start(
        request, "file", file_id, title=entry.name, voice=voice,
        output_format=out_fmt, force=bool(force),
    ))


_TTS_STUDIO_MAX_CHARS = 20000
_TTS_STUDIO_SINGLE_CALL_CHARS = 4000   # ≤ this → one TTS call (any provider)
_TTS_STUDIO_FORMATS = {"mp3", "wav", "opus", "flac"}


@router.post("/tts-studio")
async def tts_studio_synthesize(request: Request):
    """TTS recording studio — synthesize text to an audio file in Files.

    Body: ``{text, voice?, speed?, name?, format?}``. Short text goes
    through a single TTS call (works with any provider); longer text needs
    the built-in Kokoro voice (it's split into chunks and the WAV segments
    are stitched). The result is saved as an audio artifact, so it appears
    under Files → Audio.
    """
    from augmentum.proxy.audio_routes import tts_synthesize_bytes

    uid = _user_id(request)
    if not uid:
        raise HTTPException(401, "Not authenticated")
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    if backend is None:
        raise HTTPException(503, "Database not available")
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    body = body if isinstance(body, dict) else {}
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Text is required")
    if len(text) > _TTS_STUDIO_MAX_CHARS:
        raise HTTPException(413, f"Text is too long (max {_TTS_STUDIO_MAX_CHARS:,} characters)")
    voice = str(body.get("voice") or "")
    try:
        speed = float(body.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    speed = max(0.5, min(2.0, speed))
    fmt = str(body.get("format") or "mp3").lower()
    if fmt not in _TTS_STUDIO_FORMATS:
        fmt = "mp3"
    name = str(body.get("name") or "").strip()

    if len(text) <= _TTS_STUDIO_SINGLE_CALL_CHARS:
        audio, _ = await tts_synthesize_bytes(
            backend.conn, text, voice=voice, speed=speed, response_format=fmt,
        )
    else:
        # Long text: chunk + stitch — requires a built-in engine (Kokoro /
        # Pocket TTS) since they yield well-formed per-segment WAV.
        from augmentum.jobs.handlers.narration_synth import (
            _chunk_text,
            _concat_wav,
            _engine_wav_blobs,
            _resolve_synth_engine,
        )
        from augmentum.proxy.audio_routes import _BUILTIN_TTS_IDS, resolve_voice_provider
        provider, _ = await resolve_voice_provider(backend.conn, voice or "")
        engine_id = provider.get("id") if provider else ""
        if engine_id not in _BUILTIN_TTS_IDS:
            raise HTTPException(
                422,
                "Text over 4,000 characters is split and stitched — that needs a "
                "built-in voice (Kokoro or Pocket TTS). Shorten the text or switch your TTS provider.",
            )
        engine = _resolve_synth_engine(engine_id)
        fmt = "wav"
        blobs: list[bytes] = []
        for chunk in _chunk_text(text):
            blobs.extend(await _engine_wav_blobs(engine, chunk, voice))
        audio = await asyncio.to_thread(_concat_wav, blobs)

    if not audio:
        raise HTTPException(502, "TTS produced no audio")

    # Derive a friendly name/filename.
    base = name or " ".join(text.split()[:8]) or "TTS recording"
    base = re.sub(r"[^\w\- ]+", "", base).strip()[:80] or "TTS recording"
    filename = f"{base}.{fmt}"

    store = getattr(request.app.state, "artifact_store", None)
    if store is None:
        from augmentum.tools.artifact_storage import ArtifactStore
        store = ArtifactStore(backend.conn)
    saved = await store.save(
        audio, filename, fmt,
        display_name=base,
        user_id=uid,
        metadata={"tts_studio": True, "voice": voice, "chars": len(text), "speed": speed},
    )
    return JSONResponse({
        "artifact_id": saved["id"],
        "download_url": f"/api/artifacts/{saved['id']}/download",
        "name": base,
        "filename": filename,
        "format": fmt,
        "size_bytes": len(audio),
    })


@router.post("/transform/{file_id}")
async def transform_file(file_id: str, request: Request):
    """Apply a deterministic content transform to a file.

    Body:
      {"operation": "convert_image",
       "params":   {"target": "png" | "jpg" | "jpeg" | "webp"},
       "disposition": "download" | "new_file"}

    With ``download``, the response body is the transformed bytes as an
    attachment. With ``new_file``, the bytes are stored as a new file
    (subject to per-file size cap and storage quota) and the new file's
    info row is returned as JSON.

    Adding new operations is a one-line addition to ``_TRANSFORM_REGISTRY``;
    handler signature is ``(src_bytes, src_name, params) -> (out_bytes,
    out_name, out_mime)``.
    """
    import os

    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    operation = (body.get("operation") or "").strip()
    params = body.get("params") or {}
    disposition = (body.get("disposition") or "download").strip()
    if not isinstance(params, dict):
        return JSONResponse({"error": "params must be an object"}, status_code=400)
    if disposition not in ("download", "new_file"):
        return JSONResponse({"error": "disposition must be 'download' or 'new_file'"},
                            status_code=400)

    handler = _TRANSFORM_REGISTRY.get(operation)
    if not handler:
        return JSONResponse({"error": f"unknown operation: {operation}"}, status_code=400)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Resolve source bytes — disk path first, then source-specific lookup
    src_bytes: bytes | None = None
    if entry.real_path and os.path.exists(entry.real_path):
        try:
            with open(entry.real_path, "rb") as f:
                src_bytes = f.read()
        except OSError as err:
            return JSONResponse({"error": f"could not read source: {err}"}, status_code=500)
    else:
        payload, _ = await _resolve_by_source(request, entry, uid)
        if isinstance(payload, str) and os.path.exists(payload):
            try:
                with open(payload, "rb") as f:
                    src_bytes = f.read()
            except OSError as err:
                return JSONResponse({"error": f"could not read source: {err}"}, status_code=500)
        elif isinstance(payload, bytes):
            src_bytes = payload

    if src_bytes is None:
        return JSONResponse({"error": "source content not available"}, status_code=404)

    # Per-file size cap on input — same envelope as /upload
    if len(src_bytes) > settings.files_upload_max_file_bytes:
        return JSONResponse(
            {"error": f"source exceeds per-file limit "
                      f"({settings.files_upload_max_file_bytes} bytes)"},
            status_code=413,
        )

    try:
        out_bytes, out_name, out_mime = handler(src_bytes, entry.name, params)
    except ValueError as err:
        return JSONResponse({"error": str(err)}, status_code=400)
    except RuntimeError as err:
        # Reserved for "operation supported but optional dep missing" cases
        return JSONResponse({"error": str(err)}, status_code=501)
    except Exception as err:
        log.warning("file_transform_failed", file_id=file_id, operation=operation,
                    error=str(err), exc_info=True)
        return JSONResponse({"error": "transform failed"}, status_code=500)

    await _audit(
        request, "file.transform",
        detail=f"id={file_id} op={operation} disposition={disposition} "
               f"in={len(src_bytes)} out={len(out_bytes)}",
    )

    if disposition == "download":
        return Response(
            content=out_bytes, media_type=out_mime,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )

    # disposition == "new_file": persist via the same adapter /upload uses,
    # so quota accounting and dedup behave identically.
    adapter = getattr(request.app.state, "uploads_adapter", None)
    if not adapter:
        return JSONResponse({"error": "uploads not available"}, status_code=503)

    if len(out_bytes) > settings.files_upload_max_file_bytes:
        return JSONResponse(
            {"error": f"output exceeds per-file limit "
                      f"({settings.files_upload_max_file_bytes} bytes)"},
            status_code=413,
        )

    quota = settings.files_user_storage_quota_bytes
    if quota > 0:
        used = await adapter.storage_used(uid)
        if used + len(out_bytes) > quota:
            return JSONResponse(
                {"error": f"storage quota exceeded "
                          f"({used + len(out_bytes)} > {quota} bytes)"},
                status_code=507,
            )

    safe_name = sanitize_filename(out_name)
    sniffed = sniff_mime(out_bytes, fallback=out_mime)
    try:
        info = await adapter.save(
            out_bytes, safe_name,
            mime_type=out_mime, mime_sniffed=sniffed,
            user_id=uid,
        )
    except Exception as err:
        log.warning("file_transform_save_failed", file_id=file_id,
                    operation=operation, error=str(err))
        return JSONResponse({"error": f"save failed: {err}"}, status_code=500)

    return JSONResponse({"file": info, "operation": operation}, status_code=201)


@router.post("/summarize/{file_id}")
async def summarize_file(file_id: str, request: Request):
    """Generate an AI summary for a file using the LLM backend."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "File not found"}, status_code=404)

    body = await request.json()
    model = body.get("model", "")

    card = entry.to_card("card_content")

    content_snippet = ""
    vfs = _get_vfs(request)
    if entry.mime_type and entry.mime_type.startswith("text/") and vfs:
        source_prefix = {
            "artifacts": "/Artifacts", "images": "/Images",
            "chat_images": "/Chat Images",
            "voices": "/Voices",
        }.get(entry.source, "")
        if source_prefix:
            try:
                raw = await vfs.read(f"{source_prefix}/{entry.name}", user_id=uid)
                if raw:
                    content_snippet = raw[:4000].decode("utf-8", errors="replace")
            except Exception as exc:
                log.debug("files_vfs_read_for_summary_failed", entry_id=entry.id, error=str(exc))

    from augmentum.models.base import InternalChatRequest, Message
    registry = getattr(request.app.state, "provider_registry", None)
    if not registry:
        return JSONResponse({"error": "No provider registry"}, status_code=503)
    try:
        backend, model = await registry.resolve_model_for_role(
            "utility",
            override=model,
            settings=settings,
        )
    except Exception:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)
    if not backend:
        return JSONResponse({"error": "No LLM backend available"}, status_code=503)

    user_msg = f"Summarize this file in 2-3 sentences:\n\n{card}"
    if content_snippet:
        user_msg += f"\n\nFile content (first 4000 chars):\n{content_snippet}"

    chat_req = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content="You are a file analysis assistant. Provide a concise summary of the given file. Be specific about what it contains and its purpose."),
            Message(role="user", content=user_msg),
        ],
        stream=False,
        temperature=0.3,
        max_tokens=300,
    )

    try:
        resp = await backend.chat(chat_req)
        summary = resp.message.content.strip()
        await idx.update_enrichment(file_id, user_id=uid, description=summary)
        return JSONResponse({"ok": True, "summary": summary})
    except Exception as exc:
        log.warning("summarize_failed", file_id=file_id, error=str(exc))
        return JSONResponse({"error": f"Summarize failed: {exc}"}, status_code=502)


@router.patch("/{file_id}")
async def rename_file(file_id: str, request: Request):
    """Rename a file in the index."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)

    entry = await idx.get(file_id, user_id=uid)
    if not entry:
        return JSONResponse({"error": "File not found"}, status_code=404)

    body = await request.json()
    raw_name = str(body.get("name", "")).strip()
    if not raw_name:
        return JSONResponse({"error": "Name cannot be empty"}, status_code=400)
    if len(raw_name) > 255:
        return JSONResponse({"error": "Name too long"}, status_code=400)
    new_name = sanitize_filename(raw_name)

    await idx._db.execute(
        "UPDATE file_index SET name = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
        (new_name, file_id, uid),
    )
    await idx._db.commit()
    return JSONResponse({"ok": True, "name": new_name})


@router.delete("/{file_id}")
async def delete_file(file_id: str, request: Request):
    """Soft-delete a file (move to trash)."""
    idx = _get_index(request)
    uid = _user_id(request)
    if not idx or not uid:
        return JSONResponse({"error": "Not found"}, status_code=404)
    ok = await idx.soft_delete(file_id, user_id=uid)
    if not ok:
        return JSONResponse({"error": "File not found"}, status_code=404)
    await _audit(request, "file.delete", detail=f"id={file_id}")
    return JSONResponse({"ok": True})


def _transcode_image_to_jpeg(fp: Path, ext: str) -> bytes | None:
    """Transcode HEIC/HEIF/TIFF to JPEG bytes for browser display.

    Returns None when no decoder is available so the caller can fall back
    to a friendly preview shell. pillow-heif is an optional dependency
    only relevant on systems that handle iPhone photos; TIFF works with
    stock Pillow on every install.
    """
    try:
        from PIL import Image
        if ext in ("heic", "heif"):
            try:
                import pillow_heif  # type: ignore[import-untyped]
                pillow_heif.register_heif_opener()
            except ImportError:
                log.info("heic_preview_skipped", reason="pillow-heif not installed")
                return None
        with Image.open(fp) as img:
            # Drop alpha for JPEG output (paste over white).
            if img.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                rgba = img.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[-1])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            return buf.getvalue()
    except Exception:
        log.warning("image_transcode_failed", path=str(fp), ext=ext, exc_info=True)
        return None


# --- Transform registry ---------------------------------------------------
#
# Handlers map an input file's bytes + original name + caller params to
# output bytes + a suggested output filename + a MIME type. They raise
# ValueError for bad input/params (→ 400), RuntimeError when a feature is
# unavailable because an optional dep is missing (→ 501), and any other
# exception turns into a generic 500 at the route.

def _convert_image(src_bytes: bytes, src_name: str,
                   params: dict) -> tuple[bytes, str, str]:
    """Re-encode an image into one of png / jpeg / webp using Pillow.

    Preserves EXIF where the source has it and the target supports it.
    JPEG can't carry alpha, so transparent inputs get flattened onto a
    white background before encoding.
    """
    import io as _io

    from PIL import Image

    target = (params.get("target") or "").lower().strip()
    target = {"jpg": "jpeg"}.get(target, target)
    if target not in {"png", "jpeg", "webp"}:
        raise ValueError(
            f"unsupported target format: {target!r} (expected png|jpg|jpeg|webp)"
        )

    src_ext = src_name.rsplit(".", 1)[-1].lower() if "." in src_name else ""
    if src_ext in ("heic", "heif"):
        try:
            import pillow_heif  # type: ignore[import-untyped]
            pillow_heif.register_heif_opener()
        except ImportError:
            raise RuntimeError(
                "HEIC/HEIF input requires the optional pillow-heif package"
            )

    try:
        img = Image.open(_io.BytesIO(src_bytes))
        img.load()  # force decode now so errors surface here, not at save()
    except Exception as err:
        raise ValueError(f"could not decode source image: {err}")

    exif = img.info.get("exif")
    save_kwargs: dict = {}

    if target == "jpeg":
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            rgba = img.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        save_kwargs = {"quality": 90, "optimize": True}
        if exif:
            save_kwargs["exif"] = exif
        out_ext, out_mime, pil_format = "jpg", "image/jpeg", "JPEG"
    elif target == "png":
        save_kwargs = {"optimize": True}
        out_ext, out_mime, pil_format = "png", "image/png", "PNG"
    else:  # webp
        save_kwargs = {"quality": 90, "method": 6}
        if exif:
            save_kwargs["exif"] = exif
        out_ext, out_mime, pil_format = "webp", "image/webp", "WEBP"

    buf = _io.BytesIO()
    img.save(buf, format=pil_format, **save_kwargs)

    base = src_name.rsplit(".", 1)[0] if "." in src_name else src_name
    return buf.getvalue(), f"{base}.{out_ext}", out_mime


def _extract_text(src_bytes: bytes, src_name: str,
                  params: dict) -> tuple[bytes, str, str]:
    """Extract plain text from a document (PDF/DOCX/PPTX/XLSX) into a .txt.

    Reuses the parsers in ``augmentum.tools.document_parse`` so the eventual
    LLM tool surface and this files-mode action share one extraction
    implementation. Parsers take a filesystem path, so source bytes are
    spilled to a tempfile first.
    """
    import os as _os
    import tempfile as _tempfile

    from augmentum.tools.document_parse import _PARSERS

    src_ext = src_name.rsplit(".", 1)[-1].lower() if "." in src_name else ""
    parser = _PARSERS.get(f".{src_ext}")
    if not parser:
        raise ValueError(
            f"unsupported source for text extraction: {src_ext or '(no extension)'}"
        )

    suffix = f".{src_ext}" if src_ext else ""
    tf = _tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tf.write(src_bytes)
        tf.close()
        try:
            text = parser(tf.name)
        except ImportError as err:
            # Optional decoder dep missing → caller surfaces as 501
            raise RuntimeError(str(err))
        except Exception as err:
            raise ValueError(f"could not extract text: {err}")
    finally:
        try:
            _os.unlink(tf.name)
        except OSError:
            pass

    base = src_name.rsplit(".", 1)[0] if "." in src_name else src_name
    return text.encode("utf-8"), f"{base}.txt", "text/plain; charset=utf-8"


def _md_to_sections(md_text: str, fallback_title: str) -> tuple[str, list[dict]]:
    """Split a markdown doc into ``(title, [section])`` for the DOCX/PDF renderers.

    The first H1 (if any) is lifted out as the document title; everything
    after it becomes a single section body. The downstream
    ``_render_docx_body`` / ``_render_pdf`` body parser already handles
    inline H2/H3, lists, and fenced code, so we don't need to subdivide
    further at this layer.
    """
    import re

    title = fallback_title
    body_start = 0
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            title = m.group(1).strip() or fallback_title
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).lstrip("\n")
    return title, [{"heading": "", "level": 1, "body": body}]


def _convert_document(src_bytes: bytes, src_name: str,
                      params: dict) -> tuple[bytes, str, str]:
    """Convert a markdown document to DOCX or PDF via the artifact toolchain.

    Phase-1 scope: markdown → docx/pdf only. The artifact renderers also
    cover EPUB and richer themes, and reverse directions (DOCX → MD via
    document_parse, etc.) can layer on by extending the dispatch below
    or adding sibling handlers.
    """
    target = (params.get("target") or "").lower().strip()
    target = {"doc": "docx"}.get(target, target)
    if target not in {"docx", "pdf"}:
        raise ValueError(
            f"unsupported target format: {target!r} (expected docx|pdf)"
        )

    src_ext = src_name.rsplit(".", 1)[-1].lower() if "." in src_name else ""
    if src_ext not in {"md", "markdown", "mdown", "mkd"}:
        raise ValueError(
            f"unsupported source for document convert: {src_ext!r} "
            "(only markdown is supported in this slice)"
        )

    md_text = src_bytes.decode("utf-8", errors="replace")
    base = src_name.rsplit(".", 1)[0] if "." in src_name else src_name
    title, sections = _md_to_sections(md_text, fallback_title=base)

    try:
        if target == "docx":
            from augmentum.tools.artifact_document import _render_docx
            data = _render_docx(title, "", sections)
            mime = ("application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document")
            return data, f"{base}.docx", mime
        # target == "pdf"
        from augmentum.tools.artifact_document import _render_pdf
        data = _render_pdf(title, "", sections)
        return data, f"{base}.pdf", "application/pdf"
    except ImportError as err:
        # Optional dep missing (python-docx for docx, fpdf2 for pdf) → 501
        raise RuntimeError(f"renderer dependency missing: {err}")


_TRANSFORM_REGISTRY: dict = {
    "convert_image":    _convert_image,
    "extract_text":     _extract_text,
    "convert_document": _convert_document,
}


def _highlight_text(text: str, hint: str) -> str:
    """Render text as a syntax-highlighted code block via Pygments.

    ``hint`` is a filename extension without leading dot (``"py"``, ``"json"``,
    etc.). Lookup goes filename-first so common extensions resolve cleanly,
    falling back to language-name lookup. On any failure (pygments missing,
    unknown extension, oversized input) the function returns a plain
    ``<div class="highlight"><pre>...</pre></div>`` so the visual chrome
    matches the highlighted output and downstream CSS targets one selector.

    Capped at 200 KB to keep the tokenizer's O(n) cost off the request path
    for huge files; over that we serve the plain block.
    """
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    plain = f'<div class="highlight"><pre>{safe}</pre></div>'
    if not text or len(text) > 200_000:
        return plain
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
        from pygments.util import ClassNotFound
        try:
            lexer = get_lexer_for_filename(f"file.{hint}", stripall=False)
        except ClassNotFound:
            try:
                lexer = get_lexer_by_name(hint)
            except ClassNotFound:
                return plain
        return highlight(text, lexer, HtmlFormatter(cssclass="highlight"))
    except Exception:
        return plain


def _archive_to_html(file_path: Path, title: str, download_url: str, ext: str) -> str | None:
    """List an archive's contents as a directory tree preview.

    Supports zip / tar / tar.gz / tar.bz2 via stdlib. 7z is not previewable
    (no stdlib support) — caller falls back to the download card.
    Caps at 2000 entries so a massive archive doesn't spike memory.
    """
    from html import escape

    from augmentum.proxy.artifact_routes import _preview_shell

    MAX_ENTRIES = 2000
    entries: list[tuple[str, int, bool]] = []  # (path, size, is_dir)
    truncated = False

    try:
        if ext == "zip":
            import zipfile

            with zipfile.ZipFile(str(file_path)) as zf:
                for info in zf.infolist():
                    if len(entries) >= MAX_ENTRIES:
                        truncated = True
                        break
                    entries.append((
                        info.filename,
                        info.file_size,
                        info.filename.endswith("/") or info.is_dir(),
                    ))
        else:
            import tarfile

            mode_map = {"tar": "r", "gz": "r:gz", "tgz": "r:gz", "bz2": "r:bz2"}
            with tarfile.open(str(file_path), mode_map.get(ext, "r:*")) as tf:
                for member in tf:
                    if len(entries) >= MAX_ENTRIES:
                        truncated = True
                        break
                    entries.append((member.name, member.size, member.isdir()))
    except Exception as exc:
        log.warning("archive_preview_failed", path=str(file_path), error=str(exc))
        return None

    if not entries:
        return _preview_shell(title, download_url, (
            '<div class="wrap" style="text-align:center;padding-top:48px;color:#6b6b80">'
            'Archive is empty or could not be read.'
            '</div>'
        ))

    # Normalize + sort: directories first, then alphabetical. Size display
    # is best-effort — directories report 0.
    def _fmt_size(n: int) -> str:
        if n <= 0:
            return ""
        units = ("B", "KB", "MB", "GB", "TB")
        i = 0
        size = float(n)
        while size >= 1024 and i < len(units) - 1:
            size /= 1024
            i += 1
        return f"{size:.1f} {units[i]}" if i else f"{int(size)} {units[i]}"

    entries.sort(key=lambda e: (not e[2], e[0].lower()))

    total_files = sum(1 for e in entries if not e[2])
    total_size = sum(e[1] for e in entries if not e[2])

    rows = []
    for path, size, is_dir in entries:
        icon = "\U0001F4C1" if is_dir else "\U0001F4C4"  # folder vs page
        size_col = "—" if is_dir else _fmt_size(size)
        display = escape(path.rstrip("/") or path)
        row_cls = "dir" if is_dir else "file"
        rows.append(
            f'<tr class="{row_cls}"><td class="icon">{icon}</td>'
            f'<td class="path">{display}</td>'
            f'<td class="size">{size_col}</td></tr>'
        )

    summary = f'{total_files} file{"s" if total_files != 1 else ""} \u00B7 {_fmt_size(total_size)}'
    if truncated:
        summary += f' \u00B7 showing first {MAX_ENTRIES}'

    body = (
        '<style>'
        '.archive{max-width:900px;margin:0 auto}'
        '.archive-summary{color:#8a8a9c;font-size:13px;padding:18px 24px 10px}'
        '.archive table{width:100%;border-collapse:collapse;font-size:13px}'
        '.archive thead th{position:sticky;top:44px;background:#161625;color:#a1a1b5;'
        'font-weight:600;text-align:left;padding:10px 12px;font-size:11px;'
        'text-transform:uppercase;letter-spacing:0.03em;border-bottom:1px solid #2d2d45}'
        '.archive td{padding:6px 12px;border-bottom:1px solid #1c1c2e;color:#ececf1;'
        'vertical-align:middle}'
        '.archive tr:hover td{background:#1a1a2c}'
        '.archive td.icon{width:24px;opacity:0.6}'
        '.archive td.path{word-break:break-all;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px}'
        '.archive td.size{text-align:right;color:#8a8a9c;white-space:nowrap;width:90px}'
        '.archive tr.dir td.path{color:#6c8aff}'
        '@media (max-width:640px){'
        '.archive-summary{padding:12px 14px 8px;font-size:12px}'
        '.archive td,.archive thead th{padding:6px 10px;font-size:12px}'
        '.archive td.path{font-size:11.5px}'
        '}'
        '</style>'
        '<div class="archive">'
        f'<div class="archive-summary">{summary}</div>'
        '<table><thead><tr><th></th><th>Path</th><th style="text-align:right">Size</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        '</div>'
    )
    return _preview_shell(title, download_url, body)

