"""Game Portal routes — browse and pin free browser games.

Mirrors the LibriVox pattern from ``media_routes.py``: browse-live (live
fetch from external catalogs, user-agnostic cache) and pin-to-persist
(one endpoint writes into the shared ``artifacts`` table with a game
sentinel in metadata). No migrations required — games live as
``metadata.kind = "game"`` artifact rows.

Active sources: js13k. itch.io was removed — every embed got punted
to a security interstitial by upstream anti-embed detection, so users
never reached the actual game. Future sources (game jams, AGSP-streamed
games via game_stream_routes) land here.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.games.providers import js13k as js13k_provider
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient

log = get_logger(__name__)

router = APIRouter(prefix="/api/games", tags=["games"])


# Sentinel stamped on every pinned game artifact so future code can
# recognise portal-sourced entries without parsing metadata.source. Not
# used in Phase 0 browse but reserved here to keep the pattern aligned
# with ``BUILTIN_LIBRIVOX``.
BUILTIN_GAME_PORTAL = "builtin-game-portal"


# One SafeHttpClient for the module. Default 5 MB cap is plenty for the
# js13k GitHub directory listing (~200-400 KB) with headroom for any
# inlined base64 thumbnails. DNS rebinding / private-IP SSRF is
# handled inside the client.
_safe_client = SafeHttpClient()

# Separate client for detail-page fetches. js13k entries are normally
# tiny; the 3 MB cap is defensive against a malformed entry-point HTML.
_details_client = SafeHttpClient(max_response_size=3_145_728)


# --- Browse cache -------------------------------------------------------
#
# Mirrors ``media_routes.py:874-896`` exactly. Ten-minute TTL, ~200-entry
# LRU cap. The cache key intentionally excludes user_id so all users on
# a box share a single fetch against the upstream catalog; the per-user
# ``pinned`` flag gets decorated *after* lookup.

_BROWSE_CACHE_TTL_S = 600
_browse_cache: dict[tuple, tuple[float, list[dict]]] = {}


def _browse_cache_get(key: tuple) -> list[dict] | None:
    entry = _browse_cache.get(key)
    if not entry:
        return None
    expires_at, payload = entry
    if time.monotonic() > expires_at:
        _browse_cache.pop(key, None)
        return None
    return payload


def _browse_cache_set(key: tuple, payload: list[dict]) -> None:
    if len(_browse_cache) > 200:
        oldest = min(_browse_cache.items(), key=lambda kv: kv[1][0])
        _browse_cache.pop(oldest[0], None)
    _browse_cache[key] = (time.monotonic() + _BROWSE_CACHE_TTL_S, payload)


# --- Details cache ------------------------------------------------------
#
# Detail fetches are expensive (full HTML page + JSON-LD parse) and the
# underlying data barely changes once a game is shipped — ratings creep
# up over weeks, the input-method list is usually set once and forgotten,
# cover art is static. A 7-day TTL keeps the browse-strip lazy-enrichment
# flow essentially free after a user's first pass through the catalog.
# Capacity of 2000 fits ~50 pages of browse-surface enrichment without
# evicting actively-viewed entries.

_DETAILS_CACHE_TTL_S = 7 * 86_400
_DETAILS_CACHE_MAX = 2000
_details_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def _details_cache_get(key: tuple[str, str]) -> dict | None:
    entry = _details_cache.get(key)
    if not entry:
        return None
    expires_at, payload = entry
    if time.monotonic() > expires_at:
        _details_cache.pop(key, None)
        return None
    return payload


def _details_cache_set(key: tuple[str, str], payload: dict) -> None:
    if len(_details_cache) > _DETAILS_CACHE_MAX:
        oldest = min(_details_cache.items(), key=lambda kv: kv[1][0])
        _details_cache.pop(oldest[0], None)
    _details_cache[key] = (time.monotonic() + _DETAILS_CACHE_TTL_S, payload)


# --- Per-user rate limiter for detail fetches --------------------------
#
# Cheap sliding-window counter. Sized for the browse-strip enrichment
# pattern: one page (~17 cards) on entry + one page on Load More + any
# click-into-detail moves. 180/min leaves room for two full pages of
# lazy fetches plus user-driven reads without blocking real usage, while
# still stopping anyone using this endpoint as a crawl proxy against
# upstream catalogs. Server-side cache absorbs repeat views so the
# enrichment cost is paid once per game per week (7-day TTL).

_RATE_WINDOW_S = 60.0
_RATE_MAX = 180
_rate_buckets: dict[str, list[float]] = {}


def _rate_limit_ok(user_id: str) -> bool:
    now = time.monotonic()
    bucket = _rate_buckets.get(user_id, [])
    bucket = [t for t in bucket if now - t < _RATE_WINDOW_S]
    if len(bucket) >= _RATE_MAX:
        _rate_buckets[user_id] = bucket  # keep the pruned list
        return False
    bucket.append(now)
    _rate_buckets[user_id] = bucket
    # Cheap occasional GC: when the bucket map grows beyond 1k users,
    # drop anyone whose newest timestamp is older than the window.
    if len(_rate_buckets) > 1000:
        cutoff = now - _RATE_WINDOW_S
        for uid in list(_rate_buckets.keys()):
            if not _rate_buckets[uid] or max(_rate_buckets[uid]) < cutoff:
                _rate_buckets.pop(uid, None)
    return True


# --- Helpers -----------------------------------------------------------


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


# Known sources. Adding a source is: add an entry here plus branches in
# ``_fetch_source`` / ``_fetch_details_for`` and (for local-mode sources)
# a branch in the pin handler. js13k is alphabetical-only -- 'newest'
# is the canonical request; 'popular' is accepted but ignored.
_KNOWN_SOURCES: set[str] = {"js13k", "marketplace"}
_KNOWN_SORTS: set[str] = {"newest", "popular"}


async def _fetch_source(source: str, sort: str, page: int) -> list[dict]:
    """Delegate to the right provider and normalise to dicts."""
    if source == "js13k":
        hits = await js13k_provider.browse(sort, page, _safe_client)
        return [h.to_dict() for h in hits]
    # Shouldn't be reached — route layer validates source first.
    return []


async def _fetch_details_for(source: str, source_id: str) -> dict:
    """Delegate to the right provider's detail fetcher."""
    if source == "js13k":
        return await js13k_provider.fetch_details(source_id, _details_client)
    return {"ok": False, "reason": "unsupported_source"}


def _get_store(request: Request):
    store = getattr(request.app.state, "artifact_store", None)
    if not store:
        return JSONResponse({"error": "Artifact storage unavailable"}, status_code=503)
    return store


async def _find_pinned_artifact(
    store, source: str, source_id: str, user_id: str
) -> str | None:
    """Return artifact id if this (source, source_id) is already pinned for this user.

    Phase 0 uses ``json_extract`` directly against the artifacts table
    rather than extending ArtifactStore. If this grows beyond pin lookups,
    promote to a proper ``ArtifactStore.find_by_source_id`` method.
    """
    query = (
        "SELECT id FROM artifacts "
        "WHERE json_extract(metadata, '$.kind') = 'game' "
        "  AND json_extract(metadata, '$.source') = ? "
        "  AND json_extract(metadata, '$.source_id') = ?"
    )
    params: list[Any] = [source, source_id]
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    cursor = await store._db.execute(query, params)
    row = await cursor.fetchone()
    return row[0] if row else None


async def _build_pinned_map(
    store, results: list[dict], user_id: str
) -> dict[tuple[str, str], str]:
    """Return {(source, source_id): artifact_id} for all pinned items in one query.

    Rather than relying on SQLite's (col,col) IN tuple syntax (which is
    supported but noisy), we fetch every pinned game for the user and
    filter client-side. At Phase 0 pin counts this is trivial; at scale
    this becomes a bounded LIMIT query over an indexed user_id column.
    """
    if not results or not user_id:
        return {}
    wanted: set[tuple[str, str]] = {
        (r.get("source", ""), r.get("source_id", ""))
        for r in results
        if r.get("source_id")
    }
    if not wanted:
        return {}
    query = (
        "SELECT id, "
        "  json_extract(metadata, '$.source') as src, "
        "  json_extract(metadata, '$.source_id') as sid "
        "FROM artifacts "
        "WHERE json_extract(metadata, '$.kind') = 'game' "
        "  AND user_id = ?"
    )
    cursor = await store._db.execute(query, [user_id])
    out: dict[tuple[str, str], str] = {}
    for row in await cursor.fetchall():
        artifact_id, src, sid = row
        key = (src or "", sid or "")
        if key in wanted:
            out[key] = artifact_id
    return out


# --- Routes ------------------------------------------------------------


@router.get("/browse")
async def browse_games(request: Request) -> JSONResponse:
    """Live browse against a public game catalog.

    Query params:
    - ``source`` — ``js13k`` (default). Future: ``jam``.
    - ``sort`` — ``newest`` (default) or ``popular``.
    - ``page`` — 1-indexed page number, default 1.

    Returns ``{results, source, sort, page}``. Results are normalised
    ``GameBrowseResult`` dicts regardless of source. Failures produce a
    structured error body; the frontend distinguishes 403 (source
    blocked), 503 (timeout/slow), 502 (parse/upstream) for UX copy.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    qp = request.query_params
    source = (qp.get("source") or "js13k").strip().lower()
    sort = (qp.get("sort") or "newest").strip().lower()
    try:
        page = max(1, int(qp.get("page", "1")))
    except ValueError:
        return JSONResponse({"error": "Invalid pagination"}, status_code=400)

    if source not in _KNOWN_SOURCES:
        return JSONResponse(
            {"error": f"Unknown source: {source}"}, status_code=400
        )
    if sort not in _KNOWN_SORTS:
        return JSONResponse(
            {"error": f"Unknown sort: {sort}"}, status_code=400
        )

    cache_key = (source, sort, page)
    cached = _browse_cache_get(cache_key)
    if cached is not None:
        results: list[dict] = cached
    else:
        try:
            results = await _fetch_source(source, sort, page)
        except Exception as exc:
            # The provider already logs structured warnings and swallows
            # its own errors, so if we reach here it's an unexpected
            # exception in the cache layer or router glue.
            log.warning(
                "games_browse_failed",
                source=source,
                sort=sort,
                page=page,
                error=str(exc),
            )
            return JSONResponse(
                {"error": "Failed to fetch games"}, status_code=502
            )
        _browse_cache_set(cache_key, results)

    # Decorate with per-user ``pinned`` flag AFTER the cache lookup so
    # the cached payload stays user-agnostic and every user on the same
    # box benefits from one upstream fetch. Mirrors
    # ``media_routes.py:939-968``.
    store = getattr(request.app.state, "artifact_store", None)
    pinned_map: dict[tuple[str, str], str] = {}
    if store is not None:
        try:
            pinned_map = await _build_pinned_map(store, results, uid)
        except Exception as exc:
            # Decoration failure is non-fatal — the user still sees the
            # catalog, just without pinned state.
            log.warning("games_pinned_decoration_failed", error=str(exc))

    decorated: list[dict[str, Any]] = []
    for r in results:
        entry = dict(r)
        key = (r.get("source", ""), r.get("source_id", ""))
        artifact_id = pinned_map.get(key)
        entry["pinned"] = artifact_id is not None
        entry["pinned_artifact_id"] = artifact_id
        decorated.append(entry)

    return JSONResponse(
        {
            "results": decorated,
            "source": source,
            "sort": sort,
            "page": page,
            "has_more": len(results) > 0,
        }
    )


# --- Details ------------------------------------------------------------


@router.get("/details")
async def details_game(request: Request) -> JSONResponse:
    """Expanded metadata for a single game.

    Triggers a live page fetch (JSON-LD + OpenGraph parse) with a 24h
    cache in front. Rate-limited per-user to 60 fetches/minute so the
    endpoint can't be used as a crawl proxy against the upstream
    catalog.

    Query params:
    - ``source`` — ``js13k`` (only currently-supported source).
    - ``source_id`` — stable id within the source.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    qp = request.query_params
    source = (qp.get("source") or "js13k").strip().lower()
    source_id = (qp.get("source_id") or "").strip()
    if source not in _KNOWN_SOURCES:
        return JSONResponse({"error": f"Unknown source: {source}"}, status_code=400)
    if not source_id:
        return JSONResponse({"error": "source_id is required"}, status_code=400)

    cache_key = (source, source_id)
    cached = _details_cache_get(cache_key)
    if cached is not None:
        return JSONResponse(cached)

    if not _rate_limit_ok(uid):
        return JSONResponse(
            {"error": "Too many detail requests — slow down"},
            status_code=429,
        )

    payload = await _fetch_details_for(source, source_id)
    # Always cache, even on not-ok payloads — avoids hammering a broken
    # page on repeated clicks. The user can retry via a refresh in 24h.
    _details_cache_set(cache_key, payload)
    if not payload.get("ok"):
        # Partial payloads still render something useful in the UI.
        return JSONResponse(payload, status_code=200)
    return JSONResponse(payload)


# --- Pin / unpin --------------------------------------------------------


class GamePinRequest(BaseModel):
    source: str
    source_id: str
    name: str
    author: str = ""
    tagline: str = ""
    thumbnail_url: str = ""
    source_url: str = ""
    embed_url: str = ""
    play_mode: str = "embed"        # "embed" | "local"
    genre: list[str] = Field(default_factory=list)
    size_bytes: int = 0
    load_estimate_ms: int = 0
    extra: dict = Field(default_factory=dict)


@router.post("/pin")
async def pin_game(body: GamePinRequest, request: Request) -> JSONResponse:
    """Save a browse result to the user's library as a game artifact.

    Phase 0 supports ``play_mode="embed"`` — the artifact is a small JSON
    manifest bookmark; play time reads ``metadata.embed_url`` and mounts
    it in a sandboxed iframe. ``play_mode="local"`` (zip download) is
    reserved for a later phase when js13k lands.

    Idempotent: re-pinning the same ``(source, source_id)`` returns the
    existing artifact id.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)

    source = (body.source or "").strip().lower()
    if source not in _KNOWN_SOURCES:
        return JSONResponse(
            {"error": f"Unknown source: {source}"}, status_code=400
        )
    if not body.source_id:
        return JSONResponse({"error": "source_id is required"}, status_code=400)
    if body.play_mode not in {"embed", "local"}:
        return JSONResponse(
            {"error": f"Unsupported play_mode: {body.play_mode}"}, status_code=400
        )
    if body.play_mode == "local" and source != "js13k":
        # Only js13k supports local-mode in Phase 0 — other sources either
        # don't expose redistribution-friendly zips or haven't been wired.
        return JSONResponse(
            {"error": f"local-mode pin not supported for source: {source}"},
            status_code=501,
        )
    if body.play_mode == "embed" and not body.embed_url:
        return JSONResponse(
            {"error": "embed_url is required for embed-mode games"},
            status_code=400,
        )

    store = _get_store(request)

    existing = await _find_pinned_artifact(store, source, body.source_id, uid)
    if existing:
        return JSONResponse(
            {"id": existing, "already_pinned": True}, status_code=200
        )

    # Enrichment: fetch detail page server-side before persisting. Gives
    # the pinned artifact the richer metadata (real author, description,
    # genre, rating) without relying on the frontend-sent fields. Uses
    # the same 24h cache so a user who already previewed the game gets
    # an effectively free lookup. A failure here is non-fatal — we fall
    # back to whatever the browse card carried.
    details: dict = {}
    try:
        cached = _details_cache_get((source, body.source_id))
        if cached is not None:
            details = cached
        else:
            details = await _fetch_details_for(source, body.source_id)
            _details_cache_set((source, body.source_id), details)
    except Exception as exc:
        log.warning(
            "game_pin_enrichment_failed",
            source=source,
            source_id=body.source_id,
            error=str(exc),
        )
        details = {}

    # Prefer enriched fields when the fetch succeeded; otherwise keep the
    # frontend-sent values. We never blank a field the user already saw.
    enriched_author = body.author
    enriched_description = ""
    enriched_cover = body.thumbnail_url
    enriched_genre = list(body.genre)
    enriched_rating_value = 0.0
    enriched_rating_count = 0
    enriched_date_published = ""
    enriched_platforms: list[str] = []
    enriched_inputs: list[str] = []
    enriched_mobile_friendly = False
    # Default to the frontend-supplied embed_url so sources without a
    # detail-fetcher (e.g. the curated marketplace list, where every URL
    # was hand-verified iframe-friendly) still get a playable iframe at
    # play time. Detail-fetched embed_src wins when present.
    enriched_embed_src = body.embed_url or ""
    if details.get("ok"):
        enriched_author = details.get("author_name") or enriched_author
        enriched_description = details.get("description") or ""
        enriched_cover = details.get("cover_url") or enriched_cover
        if details.get("genre"):
            enriched_genre = details["genre"]
        enriched_rating_value = float(details.get("rating_value") or 0)
        enriched_rating_count = int(details.get("rating_count") or 0)
        enriched_date_published = details.get("date_published") or ""
        enriched_platforms = list(details.get("platforms") or [])
        enriched_inputs = list(details.get("inputs") or [])
        enriched_mobile_friendly = bool(details.get("mobile_friendly"))
        enriched_embed_src = str(details.get("embed_src") or body.embed_url or "")

    # For local-mode pins (js13k today) we fetch and unpack the game's
    # zip bundle now so subsequent Plays can be served from our own
    # origin without another round-trip to the upstream. Failure here is
    # FATAL — if we can't download the bundle, the pin has no playable
    # payload and should not be persisted (otherwise the user sees a
    # broken card in their library with no way to fix it short of
    # unpinning and retrying).
    bundle_source_json: str | None = None
    bundle_entry: str = ""
    bundle_single_file: bool = False
    bundle_size_bytes: int = 0
    if body.play_mode == "local":
        if source != "js13k":
            return JSONResponse(
                {"error": f"local-mode not implemented for source: {source}"},
                status_code=501,
            )
        bundle = await js13k_provider.fetch_bundle(body.source_id)
        if not bundle.get("ok"):
            log.warning(
                "game_pin_bundle_failed",
                source=source,
                source_id=body.source_id,
                reason=bundle.get("reason"),
            )
            return JSONResponse(
                {"error": f"Failed to download game bundle: {bundle.get('reason', 'unknown')}"},
                status_code=502,
            )
        # Store the unpacked files inside ``source_json`` (TEXT column).
        # Shape mirrors the app-builder artifact pattern so the frontend
        # can reuse ``assemble.js``-style rendering if we ever want to.
        bundle_source_json = json.dumps(
            {
                "kind": "game_bundle",
                "source": source,
                "source_id": body.source_id,
                "entry": bundle.get("entry") or "index.html",
                "files": bundle.get("files") or [],
            }
        )
        bundle_entry = bundle.get("entry") or "index.html"
        bundle_single_file = bool(bundle.get("single_file"))
        bundle_size_bytes = int(bundle.get("size_bytes") or 0)

    # Build a tiny manifest payload that doubles as the artifact's file.
    # Keeps ArtifactStore.save happy (it wants bytes) and gives the
    # download endpoint something meaningful to return without hitting
    # the external source again.
    manifest = {
        "kind": "game",
        "source": source,
        "source_id": body.source_id,
        "source_url": body.source_url,
        "embed_url": body.embed_url,
        "play_mode": body.play_mode,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    # Filename follows a consistent pattern so VFS/Files surfacing looks
    # clean: ``{slug}.game.json``. ``.game.json`` is recognised in the
    # frontend TYPE_MAP as type=game.
    slug = body.source_id.replace("/", "_").replace(" ", "_")[:60] or "game"
    filename = f"{slug}.game.json"

    metadata = {
        "kind": "game",
        "source": source,
        "source_id": body.source_id,
        "source_url": body.source_url,
        "embed_url": body.embed_url,
        "author": enriched_author,
        "tagline": body.tagline,
        "description": enriched_description,
        "thumbnail_url": enriched_cover,
        "play_mode": body.play_mode,
        "genre": enriched_genre,
        "rating_value": enriched_rating_value,
        "rating_count": enriched_rating_count,
        "date_published": enriched_date_published,
        "platforms": enriched_platforms,
        "inputs": enriched_inputs,
        "mobile_friendly": enriched_mobile_friendly,
        # Framable URL for the iframe. ``embed_url`` (the canonical
        # source page) often ships ``frame-ancestors`` CSP that blocks
        # third-party framing; ``embed_src`` is the resolved framable
        # variant for sources where one exists. Empty string means
        # the source requires a click-to-play handshake -- the
        # frontend falls back to opening in a new tab.
        "embed_src": enriched_embed_src,
        # Local-mode bundle markers. ``entry`` is the filename inside the
        # bundle that ``game-surface.js`` mounts as the iframe srcdoc;
        # ``single_file`` lets the UI warn about multi-file games that
        # need the blob-URL bridge (not wired in Phase 0).
        "bundle_entry": bundle_entry,
        "bundle_single_file": bundle_single_file,
        "size_bytes": bundle_size_bytes or body.size_bytes,
        "load_estimate_ms": body.load_estimate_ms,
        "enriched": bool(details.get("ok")),
        "extra": body.extra,
    }

    artifact = await store.save(
        data=manifest_bytes,
        filename=filename,
        fmt="game",
        display_name=body.name or slug,
        metadata=metadata,
        source_json=bundle_source_json,
        user_id=uid,
    )
    log.info(
        "game_pinned",
        id=artifact["id"],
        source=source,
        source_id=body.source_id,
        user_id=uid,
    )
    return JSONResponse(
        {"id": artifact["id"], "already_pinned": False}, status_code=201
    )


@router.delete("/pin/{artifact_id}")
async def unpin_game(artifact_id: str, request: Request) -> JSONResponse:
    """Remove a pinned game. Reuses ArtifactStore.delete for user scoping."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    store = _get_store(request)
    ok = await store.delete(artifact_id, user_id=uid)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Also drop any stored save state — unpinning is the user saying
    # "this is gone"; keeping save data around after would be surprising.
    try:
        ss = getattr(request.app.state, "settings_store", None)
        if ss is not None:
            await ss.set_user(uid, _save_key(artifact_id), None)
    except Exception as exc:
        log.warning("game_save_cleanup_failed", id=artifact_id, error=str(exc))
    log.info("game_unpinned", id=artifact_id, user_id=uid)
    return JSONResponse({"ok": True})


# --- Save state ---------------------------------------------------------
#
# Per-user, per-artifact save blobs. Stored in the user_settings table
# via SettingsStore.set_user so users on the same browser stay isolated
# from each other, and saves travel with the account rather than the
# browser. Only useful for ``play_mode == "local"`` games (where we
# control the iframe's HTML via srcdoc and can inject a bridge);
# embed-mode games run in their own cross-origin context and manage
# their own storage -- browser partitioning already keeps those saves
# Augmentum-scoped per-device.

# Hard size cap. Most game save files are tiny (<10 KB); this leaves
# comfortable headroom for complex RPG state without letting a runaway
# game balloon the user_settings table. The UI warns before PUT if the
# payload approaches the cap.
_SAVE_MAX_BYTES = 262_144  # 256 KB


def _save_key(artifact_id: str) -> str:
    return f"game_save:{artifact_id}"


async def _settings_store(request: Request):
    return getattr(request.app.state, "settings_store", None)


async def _assert_game_ownership(
    request: Request, artifact_id: str, uid: str
) -> tuple[bool, dict | None]:
    """Confirm (artifact exists, is a game, belongs to this user).

    Returns (ok, artifact_dict). Save routes use this before reading or
    writing a save so we never persist saves against artifacts the user
    doesn't own.
    """
    store = _get_store(request)
    art = await store.get(artifact_id, user_id=uid)
    if not art:
        return False, None
    if art.get("metadata", {}).get("kind") != "game":
        return False, None
    return True, art


@router.get("/saves/{artifact_id}")
async def get_save(artifact_id: str, request: Request) -> JSONResponse:
    """Return the persisted save blob for this artifact, if any."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    ok, _ = await _assert_game_ownership(request, artifact_id, uid)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)

    ss = await _settings_store(request)
    if ss is None:
        return JSONResponse({"data": {}, "exists": False})
    raw = await ss.get_user(uid, _save_key(artifact_id))
    if not raw:
        return JSONResponse({"data": {}, "exists": False})
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("game_save_corrupt", id=artifact_id, user_id=uid)
        return JSONResponse({"data": {}, "exists": False})
    return JSONResponse({"data": data, "exists": True})


class GameSavePut(BaseModel):
    data: dict


@router.put("/saves/{artifact_id}")
async def put_save(
    artifact_id: str, body: GameSavePut, request: Request
) -> JSONResponse:
    """Write the save blob for this artifact. Idempotent; overwrites prior."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    ok, _ = await _assert_game_ownership(request, artifact_id, uid)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)

    serialized = json.dumps(body.data, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > _SAVE_MAX_BYTES:
        return JSONResponse(
            {"error": f"Save too large (max {_SAVE_MAX_BYTES} bytes)"},
            status_code=413,
        )

    ss = await _settings_store(request)
    if ss is None:
        return JSONResponse({"error": "Storage unavailable"}, status_code=503)
    await ss.set_user(uid, _save_key(artifact_id), serialized)
    return JSONResponse({"ok": True, "bytes": len(serialized)})


@router.delete("/saves/{artifact_id}")
async def delete_save(artifact_id: str, request: Request) -> JSONResponse:
    """Clear the save blob for this artifact."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    ok, _ = await _assert_game_ownership(request, artifact_id, uid)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    ss = await _settings_store(request)
    if ss is not None:
        await ss.set_user(uid, _save_key(artifact_id), None)
    return JSONResponse({"ok": True})
