"""Grove routes — stations, favorites, and system vitals."""

from __future__ import annotations

import asyncio
import json
import re as _re

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["grove"])


def _user_id(request: Request) -> str:
    """Extract the authenticated user's id, or '' for anon/no-auth installs."""
    user = request.scope.get("user")
    return user.id if user else ""

# ---------------------------------------------------------------------------
# SomaFM curated channels
# ---------------------------------------------------------------------------

SOMAFM_CHANNELS: list[dict[str, str]] = [
    {
        "id": "groovesalad",
        "name": "Groove Salad",
        "desc": "A nicely chilled plate of ambient/downtempo beats and grooves.",
        "genre": "ambient, downtempo",
        "url": "https://ice1.somafm.com/groovesalad-128-mp3",
    },
    {
        "id": "dronezone",
        "name": "Drone Zone",
        "desc": "Served best chilled, safe with most medications. Atmospheric textures with minimal beats.",
        "genre": "ambient, drone",
        "url": "https://ice1.somafm.com/dronezone-128-mp3",
    },
    {
        "id": "deepspaceone",
        "name": "Deep Space One",
        "desc": "Deep ambient electronic, experimental and space music.",
        "genre": "ambient, space",
        "url": "https://ice1.somafm.com/deepspaceone-128-mp3",
    },
    {
        "id": "defcon",
        "name": "DEF CON Radio",
        "desc": "Music heard at the DEF CON hacker convention.",
        "genre": "electronic, hacker",
        "url": "https://ice1.somafm.com/defcon-128-mp3",
    },
    {
        "id": "spacestation",
        "name": "Space Station Soma",
        "desc": "Spaced-out ambient and mid-tempo electronica.",
        "genre": "ambient, electronic",
        "url": "https://ice1.somafm.com/spacestation-128-mp3",
    },
    {
        "id": "lush",
        "name": "Lush",
        "desc": "Sensuous and mellow vocals, mostly female, with an electronic influence.",
        "genre": "electronic, vocal",
        "url": "https://ice1.somafm.com/lush-128-mp3",
    },
    {
        "id": "seventies",
        "name": "Left Coast 70s",
        "desc": "Mellow album rock from the Seventies. Yacht rock without the irony.",
        "genre": "rock, 70s",
        "url": "https://ice1.somafm.com/seventies-128-mp3",
    },
    {
        "id": "thistle",
        "name": "ThistleRadio",
        "desc": "Exploring the pointed encyclopaedia of encyclopaedic encyclopaedism. Celtic music.",
        "genre": "celtic, folk",
        "url": "https://ice1.somafm.com/thistle-128-mp3",
    },
    {
        "id": "bootliquor",
        "name": "Boot Liquor",
        "desc": "Americana roots music for Cowhands, Cowpokes and Saddle Tramps.",
        "genre": "americana, country",
        "url": "https://ice1.somafm.com/bootliquor-128-mp3",
    },
    {
        "id": "cliqhop",
        "name": "cliqhop idm",
        "desc": "Blips, clicks and beats. Intelligent dance music.",
        "genre": "idm, electronic",
        "url": "https://ice1.somafm.com/cliqhop-128-mp3",
    },
]


# ---------------------------------------------------------------------------
# Station endpoints
# ---------------------------------------------------------------------------

@router.get("/api/grove/stations/soma")
async def soma_stations() -> JSONResponse:
    """Return curated SomaFM channel list."""
    return JSONResponse([{**ch, "source": "somafm"} for ch in SOMAFM_CHANNELS])


# ── RadioBrowser server discovery ─────────────────────────────────────
# Per API docs: DNS-lookup all.api.radio-browser.info, randomize, fallback.
_radiobrowser_servers: list[str] = []
_radiobrowser_servers_ts: float = 0
_RB_CACHE_TTL = 3600  # re-discover servers every hour
_RB_USER_AGENT = "Augmentum/1.0"

_RB_FALLBACK_SERVERS = [
    "https://de1.api.radio-browser.info",
    "https://de2.api.radio-browser.info",
    "https://nl1.api.radio-browser.info",
    "https://at1.api.radio-browser.info",
]


async def _get_radiobrowser_servers() -> list[str]:
    """Discover RadioBrowser servers via DNS, cache for 1 hour."""
    global _radiobrowser_servers, _radiobrowser_servers_ts
    import random
    import socket
    import time

    now = time.monotonic()
    if _radiobrowser_servers and (now - _radiobrowser_servers_ts) < _RB_CACHE_TTL:
        return _radiobrowser_servers

    try:
        # DNS lookup as recommended by RadioBrowser docs
        ips = socket.getaddrinfo("all.api.radio-browser.info", 443, socket.AF_INET)
        hosts = list({ip[4][0] for ip in ips})
        # Reverse DNS to get server names — skip IPs that fail reverse lookup
        # (raw IPs don't have valid TLS certs for HTTPS)
        servers = []
        for ip in hosts:
            try:
                name = socket.gethostbyaddr(ip)[0]
                servers.append(f"https://{name}")
            except socket.herror:
                pass  # skip — can't use raw IP over HTTPS
        if servers:
            random.shuffle(servers)
            _radiobrowser_servers = servers
            _radiobrowser_servers_ts = now
            return servers
    except Exception:
        log.warning("radiobrowser_dns_discovery_failed")

    # Fallback to hardcoded servers
    fallback = list(_RB_FALLBACK_SERVERS)
    random.shuffle(fallback)
    _radiobrowser_servers = fallback
    _radiobrowser_servers_ts = now
    return fallback


async def _fetch_radio_stations(q: str, tag: str, limit: int) -> list[dict]:
    """Fetch and normalize stations from RadioBrowser API with server failover."""
    params: dict[str, str | int] = {
        "limit": limit,
        "hidebroken": "true",
        "order": "clickcount",
        "reverse": "true",
    }
    if q:
        params["name"] = q
    if tag:
        params["tag"] = tag

    servers = await _get_radiobrowser_servers()
    raw = None

    for server in servers:
        try:
            async with httpx.AsyncClient(
                timeout=10,
                headers={"User-Agent": _RB_USER_AGENT},
            ) as client:
                resp = await client.get(
                    f"{server}/json/stations/search",
                    params=params,
                )
                resp.raise_for_status()
                raw = resp.json()
                break  # success — stop trying servers
        except Exception:
            log.warning("radiobrowser_server_failed", server=server, query=q)
            continue

    if raw is None:
        return []

    return [
        {
            "id": s.get("stationuuid", ""),
            "name": s.get("name", "").strip(),
            "desc": s.get("tags", ""),
            "genre": s.get("tags", ""),
            "url": s.get("url_resolved") or s.get("url", ""),
            "source": "radiobrowser",
            "favicon": s.get("favicon", ""),
            "country": s.get("countrycode", ""),
            "bitrate": s.get("bitrate", 0),
        }
        for s in raw
        if s.get("url_resolved") or s.get("url")
    ]


@router.get("/api/grove/stations/radio")
async def radio_stations(
    q: str = Query("", description="Search query"),
    tag: str = Query("", description="Tag filter"),
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    """Proxy RadioBrowser API and return normalized station objects."""
    return JSONResponse(await _fetch_radio_stations(q, tag, limit))


@router.get("/api/grove/stations/search")
async def unified_search(
    q: str = Query("", description="Search query"),
    tag: str = Query("", description="Genre/tag filter"),
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    """Unified search: filter SomaFM locally + fan out to RadioBrowser and FMA."""
    # Local SomaFM filter — match on query text AND genre tag
    q_lower = q.lower()
    tag_lower = tag.lower()
    soma_all = [{**ch, "source": "somafm"} for ch in SOMAFM_CHANNELS]

    soma_results = []
    for s in soma_all:
        # If tag filter set, station must match the genre
        if tag_lower and tag_lower not in s["genre"].lower():
            continue
        # If text query set, station must match name/desc/genre
        if q_lower and not (
            q_lower in s["name"].lower()
            or q_lower in s["desc"].lower()
            or q_lower in s["genre"].lower()
        ):
            continue
        soma_results.append(s)

    # If no filters at all, show all SomaFM
    if not q and not tag:
        soma_results = soma_all

    # Fan out to RadioBrowser in parallel with SomaFM filtering
    radio_results = await _fetch_radio_stations(q=q, tag=tag, limit=limit)

    merged = soma_results + radio_results
    return JSONResponse(merged[:limit])


# ---------------------------------------------------------------------------
# Favorites persistence
# ---------------------------------------------------------------------------

@router.get("/api/grove/favorites")
async def get_favorites(request: Request) -> JSONResponse:
    """Read soundscape favorites from settings store."""
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse([])
    # Per-user, with fallback to the legacy install-wide value so favorites
    # saved before this became tenant-scoped still surface for that user.
    raw = await store.get_user_or_global(_user_id(request), "soundscape_favorites")
    if not raw:
        return JSONResponse([])
    try:
        return JSONResponse(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return JSONResponse([])


@router.post("/api/grove/favorites")
async def save_favorites(request: Request) -> JSONResponse:
    """Save soundscape favorites to settings store."""
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse({"error": "Settings store not available"}, status_code=503)
    body = await request.json()
    uid = _user_id(request)
    if uid:
        await store.set_user(uid, "soundscape_favorites", json.dumps(body))
    else:
        await store.set("soundscape_favorites", json.dumps(body))
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# YouTube ambient search (via SearXNG)
# ---------------------------------------------------------------------------

_YT_ID_RE = _re.compile(r'(?:v=|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})')


async def _check_playable(client: httpx.AsyncClient, video_id: str) -> bool:
    """Check if a YouTube video is playable via the free oEmbed endpoint."""
    try:
        resp = await client.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            follow_redirects=True,
        )
        return resp.status_code == 200
    except Exception:
        return True  # On network error, assume playable (don't penalize)


async def _filter_playable(candidates: list[dict], limit: int) -> list[dict]:
    """Filter out unavailable YouTube videos by checking oEmbed in parallel."""
    if not candidates:
        return []
    import asyncio

    async with httpx.AsyncClient(timeout=5.0) as client:
        checks = [_check_playable(client, c["videoId"]) for c in candidates]
        playable = await asyncio.gather(*checks)

    results = [c for c, ok in zip(candidates, playable) if ok]
    return results[:limit]


# ---------------------------------------------------------------------------
# YouTube search cache (server-side TTL)
#   Shared by /youtube/search and /youtube/prewarm. One canned response
#   serves all users, which is both cheap and insulates against SearXNG or
#   YouTube rate-limit burps.
# ---------------------------------------------------------------------------

_YT_SEARCH_CACHE: dict[str, tuple[float, list[dict]]] = {}
_YT_CACHE_TTL_S = 60 * 60  # 1h — grove results don't need to be fresh
_YT_PREWARM_LOCK = asyncio.Lock()

# Default queries for prewarm — keys match frontend GENRE_CHIPS (lowercased).
# The values mirror grove.js's genreMap so cache keys line up with what
# the frontend will query when the user clicks a chip.
_YT_PREWARM_QUERIES: dict[str, str] = {
    "ambient":    "ambient music",
    "lo-fi":      "lofi hip hop music",
    "electronic": "electronic ambient music",
    "classical":  "classical music relaxing",
    "jazz":       "jazz music relaxing",
    "focus":      "focus music study",
    "nature":     "nature sounds ambiance",
    "synthwave":  "synthwave retrowave music",
}


def _norm_yt_query(q: str) -> str:
    """Normalize a query for cache-key equality: trim, lower, collapse spaces."""
    return " ".join((q or "").lower().strip().split())


async def _youtube_search_impl(
    base: str,
    q: str,
    limit: int,
) -> list[dict]:
    """Core SearXNG → YouTube search. Returns normalized video dicts."""
    params = {
        "q": q,
        "categories": "videos",
        "engines": "youtube",
        "format": "json",
        "pageno": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        # repr() preserves the type for message-less exceptions
        # (httpx ConnectTimeout(TimeoutError()) → empty str, which
        # made these logs unactionable — same pattern as
        # ollama_routes.tags_list_failed).
        log.warning("youtube_search_failed", error=repr(exc), query=q)
        return []

    raw_results = data.get("results", [])
    try:
        from augmentum.discovery.quality import filter_for_video_ui
        raw_results = filter_for_video_ui(raw_results, context="grove")
    except Exception as exc:
        log.debug("grove_youtube_filter_failed", error=str(exc))

    candidates = []
    seen_ids: set[str] = set()
    for item in raw_results:
        url = item.get("url", "")
        m = _YT_ID_RE.search(url)
        if not m:
            continue
        vid = m.group(1)
        if vid in seen_ids:
            continue
        seen_ids.add(vid)

        title = item.get("title", "")
        is_live = bool(
            "LIVE" in title.upper()
            or item.get("livestream")
            or item.get("length") in (None, "", "0:00")
        )

        candidates.append({
            "videoId": vid,
            "title": title,
            "channel": item.get("author", item.get("channel", "")),
            "duration": item.get("length", ""),
            "isLivestream": is_live,
            "thumbnail": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
        })

        if len(candidates) >= limit + 6:
            break

    return await _filter_playable(candidates, limit)


async def _youtube_search_cached(base: str, q: str, limit: int) -> list[dict]:
    """Cache-aware wrapper around _youtube_search_impl."""
    import time
    key = f"{_norm_yt_query(q)}::{limit}"
    entry = _YT_SEARCH_CACHE.get(key)
    now = time.monotonic()
    if entry and (now - entry[0]) < _YT_CACHE_TTL_S:
        return entry[1]

    results = await _youtube_search_impl(base, q, limit)
    if results:  # Only cache non-empty results — failures shouldn't stick
        _YT_SEARCH_CACHE[key] = (now, results)
        # Bound cache size
        if len(_YT_SEARCH_CACHE) > 64:
            oldest = min(_YT_SEARCH_CACHE, key=lambda k: _YT_SEARCH_CACHE[k][0])
            _YT_SEARCH_CACHE.pop(oldest, None)
    return results


@router.get("/api/grove/youtube/search")
async def youtube_search(
    request: Request,
    q: str = Query("ambient music", description="Search query"),
    limit: int = Query(12, ge=1, le=30),
) -> JSONResponse:
    """Search YouTube videos via SearXNG for ambient/music content."""
    searxng = getattr(request.app.state, "settings", None)
    base = searxng.searxng_base_url if searxng else "http://searxng:8080"

    results = await _youtube_search_cached(base, q, limit)
    if not results:
        return JSONResponse({"error": "search_unavailable", "results": []}, status_code=502)
    return JSONResponse(results)


@router.get("/api/grove/youtube/prewarm")
async def youtube_prewarm(request: Request) -> JSONResponse:
    """Return cached YouTube results for all default genres.

    The grove frontend calls this on panel-open so switching to YouTube in
    Discover feels instant. Results are cached server-side for 1h, so this
    endpoint is cheap even under heavy user load — the cost is 8 SearXNG
    queries per hour, not per user.

    Response shape: { "<genre>": [<video>, ...], ... }

    A genre key is always present in the response; the value may be an empty
    list if that genre's fetch failed. Callers should treat missing lists as
    "no cache available, fall back to on-demand search."
    """
    searxng = getattr(request.app.state, "settings", None)
    base = searxng.searxng_base_url if searxng else "http://searxng:8080"

    async with _YT_PREWARM_LOCK:
        tasks = [
            _youtube_search_cached(base, query, 12)
            for query in _YT_PREWARM_QUERIES.values()
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

    bundle: dict[str, list[dict]] = {}
    for (genre, query), result in zip(_YT_PREWARM_QUERIES.items(), all_results):
        if isinstance(result, Exception):
            log.warning("prewarm_genre_failed", genre=genre, query=query, error=str(result))
            bundle[genre] = []
        else:
            bundle[genre] = result

    return JSONResponse({
        "queries": _YT_PREWARM_QUERIES,
        "results": bundle,
    })


# ---------------------------------------------------------------------------
# Ambient video favorites
# ---------------------------------------------------------------------------


@router.get("/api/grove/ambient-favorites")
async def get_ambient_favorites(request: Request) -> JSONResponse:
    """Read ambient video favorites from settings store."""
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse([])
    raw = await store.get_user_or_global(_user_id(request), "ambient_favorites")
    if not raw:
        return JSONResponse([])
    try:
        return JSONResponse(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return JSONResponse([])


@router.post("/api/grove/ambient-favorites")
async def save_ambient_favorites(request: Request) -> JSONResponse:
    """Save ambient video favorites to settings store."""
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse({"error": "Settings store not available"}, status_code=503)
    body = await request.json()
    uid = _user_id(request)
    if uid:
        await store.set_user(uid, "ambient_favorites", json.dumps(body))
    else:
        await store.set("ambient_favorites", json.dumps(body))
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# System vitals
# ---------------------------------------------------------------------------

@router.get("/api/system/vitals")
async def system_vitals(request: Request) -> JSONResponse:
    """Aggregate resource snapshot into a compact vitals payload."""
    ledger = getattr(request.app.state, "resource_ledger", None)
    if not ledger:
        return JSONResponse({"error": "Resource ledger not available"}, status_code=503)

    snap = await ledger.collect()

    # CPU (non-blocking with interval=0)
    cpu_pct = 0
    if _psutil is not None:
        try:
            cpu_pct = round(_psutil.cpu_percent(interval=0))
        except Exception as exc:
            log.debug("grove_cpu_pct_probe_failed", error=str(exc))

    # GPU utilization percentage
    gpu_pct = 0.0
    if snap.gpu_total_mb > 0:
        gpu_pct = round(snap.gpu_used_mb / snap.gpu_total_mb * 100, 1)

    # Map model status
    def _model_status(m) -> str:
        if m.active and m.device == "cpu":
            return "resting"
        if m.active:
            return "thriving"
        return "dormant"

    models = [
        {
            "name": m.name,
            "subsystem": m.subsystem,
            "backend": m.backend,
            "status": _model_status(m),
            "vram_mb": m.vram_mb,
            "ram_mb": m.ram_mb,
            "quantization": m.quantization,
            "parameter_size": m.parameter_size,
        }
        for m in snap.models
    ]

    return JSONResponse({
        "gpu_pct": gpu_pct,
        "vram_used_mb": snap.gpu_used_mb,
        "vram_total_mb": snap.gpu_total_mb,
        "ram_used_mb": snap.ram_used_mb,
        "ram_total_mb": snap.ram_total_mb,
        "cpu_pct": cpu_pct,
        "gpu_name": snap.gpu_name,
        "models": models,
    })
