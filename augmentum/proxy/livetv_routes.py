"""Live TV API surface.

Two related surfaces:

1. **Rail browser** — GET /api/livetv/rails returns the categorized
   channel set the UI renders (Files panel → Live TV chip).
   Aggregates across every Emby/JF server the caller can see, runs
   them through the rail categorizer, ships JSON to the UI.
2. **Stream proxy** — POST /api/livetv/play mints an opaque session
   token; GET /api/livetv/stream/{token}/{...} proxies the HLS
   manifest + segments, stripping the upstream api_key so the
   browser never sees the media-server token. Tear-down via
   POST /api/livetv/stop releases the upstream tuner.

All routes are user-scoped. The play session checks user_id on
every fetch, and the rail data plane composes:

  user_media_servers (Emby + JF rows visible to caller)
    │
    └── EmbyCompatBase.fetch_live_channels()
            │
            └── livetv_rails.categorize_channels()
                    │
                    └── JSON {rails: [...]} for the UI

A per-user 60s in-memory TTL cache wraps the rails chain so the UI
can re-poll cheaply (rail re-fetch on tab focus, on play start, …)
without hammering every connected media server. ``?refresh=1``
bypasses the cache for manual reload + tests.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from augmentum.media.livetv_rails import categorize_channels
from augmentum.media.livetv_sessions import LiveTvSession, get_default_store
from augmentum.media.providers.base import CatalogItem
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.emby_compat import EmbyCompatBase
from augmentum.media.providers.jellyfin import JellyfinProvider
from augmentum.media.store import MediaServerStore
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/livetv", tags=["livetv"])

# Per-user in-memory rail cache. {user_id: (expires_at_epoch, payload)}.
# 60s window matches the channel-metadata refresh cadence on Emby/JF
# — EPG ``CurrentProgram`` flips on the hour/half-hour, so a fresher
# cache wouldn't surface anything new. Cache is process-local;
# horizontal scale would need Redis but that's not where we are.
_CACHE_TTL_S = 60.0
_rail_cache: dict[str, tuple[float, dict]] = {}


# ── Helpers ───────────────────────────────────────────────────────

def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request) -> MediaServerStore | None:
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    if isinstance(backend, SQLiteBackend):
        return MediaServerStore(backend.conn)
    return None


def _http(request: Request) -> httpx.AsyncClient | None:
    return getattr(request.app.state, "http_client", None)


def _provider_client(provider: str, http: httpx.AsyncClient) -> EmbyCompatBase | None:
    """Only Emby/JF expose Live TV today. Other providers return None
    and the caller skips them."""
    if provider == "emby":
        return EmbyProvider(http)
    if provider == "jellyfin":
        return JellyfinProvider(http)
    return None


async def _fetch_for_server(
    server,
    http: httpx.AsyncClient,
) -> list[CatalogItem]:
    """Per-server fetch with isolated error handling.

    A broken server (DNS gone, token expired, network blip) must not
    take out the rails for the user's other servers. Errors get
    logged at warning so a recurring failure shows up rather than
    silently presenting an incomplete rail set as complete.
    """
    if not server.access_token:
        return []
    client = _provider_client(server.provider, http)
    if client is None:
        return []
    try:
        items = await client.fetch_live_channels(server.base_url, server.access_token)
    except Exception as exc:
        log.warning(
            "livetv_fetch_failed",
            server_id=server.id,
            provider=server.provider,
            error=str(exc),
        )
        return []
    # Tag each channel with the originating server so the play path
    # can route back to the right server without re-querying.
    for item in items:
        if isinstance(item.extra, dict):
            item.extra["server_id"] = server.id
    return items


def _build_payload(channels: list[CatalogItem]) -> dict[str, Any]:
    rails = categorize_channels(channels)
    return {
        "rails":        [r.to_dict() for r in rails],
        "channel_count": len(channels),
        "rail_count":    len(rails),
        "generated_at":  time.time(),
    }


# ── Routes ────────────────────────────────────────────────────────

@router.get("/rails")
async def get_livetv_rails(request: Request) -> JSONResponse:
    """Return the categorized rail set for the current user.

    Aggregates Live TV channels across every Emby/Jellyfin server
    the user has visible (their own + admin-shared), categorizes via
    :func:`augmentum.media.livetv_rails.categorize_channels`, and
    caches the response for 60s per user. ``?refresh=1`` bypasses
    the cache. Returns ``{rails: [], channel_count: 0}`` when the
    user has no Emby/JF servers configured — the UI uses that to
    render the empty-state setup card rather than crashing.
    """
    user_id = _user_id(request)
    if not user_id:
        return JSONResponse({"rails": [], "channel_count": 0, "rail_count": 0})

    refresh = (request.query_params.get("refresh") or "").lower() in ("1", "true")
    now = time.time()
    if not refresh:
        cached = _rail_cache.get(user_id)
        if cached and cached[0] > now:
            payload = dict(cached[1])
            payload["cached"] = True
            return JSONResponse(payload)

    store = _get_store(request)
    http = _http(request)
    if store is None or http is None:
        return JSONResponse({"rails": [], "channel_count": 0, "rail_count": 0})

    servers = await store.list_visible(user_id=user_id)
    live_capable = [s for s in servers if s.provider in ("emby", "jellyfin")]
    if not live_capable:
        return JSONResponse({"rails": [], "channel_count": 0, "rail_count": 0})

    # Fetch in parallel — a slow server doesn't block a fast one.
    fetches = [_fetch_for_server(s, http) for s in live_capable]
    per_server = await asyncio.gather(*fetches, return_exceptions=False)

    channels: list[CatalogItem] = []
    for batch in per_server:
        channels.extend(batch)

    payload = _build_payload(channels)
    _rail_cache[user_id] = (now + _CACHE_TTL_S, payload)
    return JSONResponse({**payload, "cached": False})


def invalidate_user_cache(user_id: str) -> None:
    """Drop the cached rail payload for ``user_id``.

    Called from server-management routes when a media server is added,
    removed, or has its token rotated — so the next /rails fetch
    rebuilds against the new server set instead of waiting up to 60s.
    """
    _rail_cache.pop(user_id, None)


# Logo variants exposed by Emby/JF for live channels. Anything outside
# the allowlist falls back to ``Primary`` rather than echoing whatever
# arbitrary string a caller smuggled through — the variant interpolates
# into the upstream URL.
_LOGO_VARIANTS = {
    "primary": "Primary",
    "light":   "LogoLight",
    "dark":    "LogoDark",
}

# Fallback chain per requested variant. Emby's ImageTags often
# advertise multiple variants but only some are actually served (the
# others 404 / 400). Trying alternatives in priority order makes the
# proxy resilient to ImageTags-vs-actual-image drift without forcing
# every UI tile to re-roundtrip on its own retry.
_LOGO_FALLBACK_CHAIN = {
    "Primary":   ("Primary", "LogoLight", "LogoDark"),
    "LogoLight": ("LogoLight", "Primary", "LogoDark"),
    "LogoDark":  ("LogoDark", "Primary", "LogoLight"),
}


@router.get("/logo/{server_id}/{channel_id}")
async def get_channel_logo(
    server_id: str,
    channel_id: str,
    request: Request,
) -> Response:
    """Proxy a Live TV channel logo from the user's Emby/JF server.

    Browser-side rails need theme-aware logos but MUST NOT see the
    user's media-server access token. This endpoint mints a one-shot
    image proxy: the route holds the token, fetches the variant, and
    streams the bytes back with a long cache header (logos are
    effectively immutable per channel).

    Authorized to the calling user — the lookup is scoped through
    ``MediaServerStore.get_visible`` with the caller's ``user_id`` so
    a user can't pull logos from another user's private server.
    Shared servers are reachable through the same path (visible to
    every user by definition).
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    store = _get_store(request)
    http = _http(request)
    if store is None or http is None:
        raise HTTPException(status_code=503, detail="service unavailable")

    server = await store.get_visible(server_id, user_id=user_id)
    if server is None:
        raise HTTPException(status_code=404, detail="server not found")
    if server.provider not in ("emby", "jellyfin"):
        raise HTTPException(status_code=400, detail="provider does not support live TV")
    if not server.access_token:
        raise HTTPException(status_code=409, detail="server not connected")

    variant_q = (request.query_params.get("variant") or "primary").lower()
    upstream_variant = _LOGO_VARIANTS.get(variant_q, "Primary")

    client = _provider_client(server.provider, http)
    if client is None:
        raise HTTPException(status_code=500, detail="provider not available")

    # Cascade through fallback variants until one returns image bytes.
    # The first 200 wins; if every variant fails, surface a 404 so the
    # UI renders its initials tile rather than a broken-image icon.
    chain = _LOGO_FALLBACK_CHAIN.get(upstream_variant, (upstream_variant,))
    last_status = 502
    for variant in chain:
        url = client.build_channel_logo_url(
            server.base_url, channel_id, server.access_token,
            variant=variant, max_height=240,
        )
        try:
            upstream = await http.get(url, timeout=15.0)
        except Exception as exc:
            log.warning(
                "livetv_logo_fetch_failed",
                server_id=server_id, channel_id=channel_id,
                variant=variant, error=str(exc),
            )
            last_status = 502
            continue
        if upstream.status_code == 200:
            content_type = upstream.headers.get("content-type", "image/png")
            return Response(
                content=upstream.content,
                media_type=content_type,
                headers={
                    # Logos are effectively per-channel-immutable.
                    # A day is short enough that a re-skin lands
                    # within a day, long enough to spare every rail
                    # open from re-hitting upstream.
                    "Cache-Control": "public, max-age=86400",
                },
            )
        last_status = upstream.status_code

    raise HTTPException(status_code=404, detail="no logo variant available")


# ──────────────────────────────────────────────────────────────────
# Stream proxy — POST /play → mint session, GET /stream/{token}/…
# proxy HLS, POST /stop → release upstream tuner.
# ──────────────────────────────────────────────────────────────────

# Where to chunk segment downloads. 64 KiB matches the existing
# /api/media/stream pipeline.
_STREAM_CHUNK = 64 * 1024
# Upstream segment fetch ceiling. A live HLS segment is typically
# 4-10s of video and well under 5 MB; 60s is plenty for the slowest
# legitimate transcoder and bounds the cost of a wedged upstream.
_SEGMENT_FETCH_TIMEOUT_S = 60.0
# Manifest fetch ceiling. The variant playlist (live.m3u8) is the
# slow one — Emby's live transcoder blocks until the first segment
# is ready, which on a cold tuner with MPEG-2 source can take 20-
# 30s. 60s covers cold-start warmup without leaving the player
# wedged forever on a genuinely broken upstream.
_MANIFEST_FETCH_TIMEOUT_S = 60.0


def _strip_api_key(query: str) -> str:
    """Drop ``api_key`` from a query string; keep everything else.

    Emby's HLS m3u8 inlines the api_key as a query param on every
    variant + segment URI. The browser must NOT see those, so we
    strip on the way out and re-attach server-side on each upstream
    fetch.
    """
    if not query:
        return ""
    pairs = [
        (k, v) for k, v in parse_qsl(query, keep_blank_values=True)
        if k.lower() != "api_key"
    ]
    return urlencode(pairs)


def _rewrite_m3u8(body: str, *, session_token: str) -> str:
    """Rewrite a master / variant m3u8 so every URI inside points at
    our proxy.

    Lines starting with ``#`` are HLS directives; everything else
    that's non-empty is a URI to a sub-playlist or segment. We
    rewrite the URI's PATH component to live under ``/api/livetv/
    stream/{token}/seg/{path}`` and strip the upstream api_key
    from the query string. Absolute URIs (rare for Emby live) keep
    their full path — the seg proxy detects an absolute path via
    a leading ``/`` versus a relative one and reconstructs the
    upstream URL accordingly.
    """
    out_lines = []
    for raw in body.splitlines():
        line = raw.rstrip("\r")
        if not line or line.startswith("#"):
            out_lines.append(line)
            continue
        parsed = urlparse(line)
        # Schemeful absolute (https://…) — preserve the path AS-IS
        # but mark it so the segment proxy can route it back to the
        # right upstream. For Emby live this rarely happens, but we
        # handle defensively rather than dropping the line.
        if parsed.scheme:
            # Just pass through unchanged — better a broken thumbnail
            # than a silent wrong rewrite. Real Emby outputs are
            # relative-path, so this branch is the unhappy path.
            out_lines.append(line)
            continue
        cleaned_query = _strip_api_key(parsed.query)
        path = parsed.path.lstrip("/")
        new_uri = f"/api/livetv/stream/{session_token}/seg/{path}"
        if cleaned_query:
            new_uri = f"{new_uri}?{cleaned_query}"
        out_lines.append(new_uri)
    return "\n".join(out_lines) + "\n"


async def _resolve_play_session(
    *,
    client: EmbyCompatBase,
    server,
    channel_id: str,
) -> tuple[str, str] | None:
    """Mint an upstream PlaySessionId for ``channel_id``. Returns
    ``(play_session_id, media_source_id)`` on success, ``None`` if
    the upstream call failed or didn't return what we need.

    Emby's PlaybackInfo endpoint opens the live stream as a side
    effect (``AutoOpenLiveStream=true``) which is what we want —
    once we have a PlaySessionId, the tuner is reserved and the
    transcoder is warming up by the time the browser asks for
    master.m3u8.
    """
    raw = await client.fetch_playback_info(
        server.base_url, server.access_token, external_id=channel_id,
    )
    if not isinstance(raw, dict):
        return None
    play_session_id = str(raw.get("PlaySessionId") or "").strip()
    if not play_session_id:
        return None
    sources = raw.get("MediaSources") or []
    if not isinstance(sources, list) or not sources:
        return None
    # Prefer a source that explicitly supports transcoding (browser
    # can't play MPEG-2 / AC-3 direct). The first source IS usually
    # the right one; the check just hardens against the rare server
    # that lists a direct-only entry first.
    chosen = None
    for src in sources:
        if isinstance(src, dict) and src.get("SupportsTranscoding"):
            chosen = src
            break
    if chosen is None and isinstance(sources[0], dict):
        chosen = sources[0]
    if not isinstance(chosen, dict):
        return None
    source_id = str(chosen.get("Id") or "").strip()
    if not source_id:
        return None
    return play_session_id, source_id


@router.post("/play/{server_id}/{channel_id}")
async def start_play_session(
    server_id: str,
    channel_id: str,
    request: Request,
) -> JSONResponse:
    """Mint a play session token and return the proxied manifest URL.

    Hand the manifest_url to hls.js (or a native <video src=> on
    Safari) and the player will pull all the way down to segments
    through our proxy without ever seeing the upstream api_key.
    """
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthorized")

    store = _get_store(request)
    http = _http(request)
    if store is None or http is None:
        raise HTTPException(status_code=503, detail="service unavailable")

    server = await store.get_visible(server_id, user_id=user_id)
    if server is None:
        raise HTTPException(status_code=404, detail="server not found")
    if server.provider not in ("emby", "jellyfin"):
        raise HTTPException(status_code=400, detail="provider does not support live TV")
    if not server.access_token:
        raise HTTPException(status_code=409, detail="server not connected")

    client = _provider_client(server.provider, http)
    if client is None:
        raise HTTPException(status_code=500, detail="provider not available")

    resolved = await _resolve_play_session(
        client=client, server=server, channel_id=channel_id,
    )
    if resolved is None:
        log.warning(
            "livetv_play_resolve_failed",
            server_id=server_id, channel_id=channel_id,
        )
        raise HTTPException(status_code=502, detail="could not start playback")

    play_session_id, media_source_id = resolved
    # DeviceId is what Emby uses to attribute the tuner reservation.
    # Per-session uniqueness here means two browser tabs on the same
    # channel get two separate tuner allocations (matching Emby Web's
    # behavior) rather than fighting over one PlaySessionId.
    device_id = f"augmentum-livetv-{secrets.token_hex(8)}"

    session = get_default_store().create(
        user_id=user_id,
        server_id=server_id,
        provider=server.provider,
        base_url=server.base_url,
        access_token=server.access_token,
        channel_id=channel_id,
        play_session_id=play_session_id,
        media_source_id=media_source_id,
        title=request.query_params.get("title", "") or channel_id,
        device_id=device_id,
    )

    manifest_url = f"/api/livetv/stream/{session.token}/master.m3u8"
    return JSONResponse({
        "session_token":   session.token,
        "manifest_url":    manifest_url,
        "channel_id":      channel_id,
        "server_id":       server_id,
        "title":           session.title,
    })


def _api_prefix_for(provider: str) -> str:
    """Mirror what EmbyProvider / JellyfinProvider use for api_prefix."""
    return "/emby" if provider == "emby" else ""


def _build_upstream_url(session: LiveTvSession, path: str, query: str) -> str:
    """Reconstruct the upstream URL for a proxied HLS path.

    ``path`` is what the browser asked for under ``/seg/`` — already
    stripped of leading slash by the route's path parameter. We
    prepend the provider's API prefix + the channel's Videos path,
    then re-attach api_key from the session.
    """
    base = session.base_url.rstrip("/") + _api_prefix_for(session.provider)
    cleaned_query = _strip_api_key(query)
    # api_key always last so we can spot it in logs / debug dumps.
    suffix = (f"?{cleaned_query}&" if cleaned_query else "?") + f"api_key={session.access_token}"
    return f"{base}/Videos/{session.channel_id}/{path}{suffix}"


@router.get("/stream/{session_token}/master.m3u8")
async def stream_master_playlist(
    session_token: str,
    request: Request,
) -> Response:
    """Mint + serve the rewritten master playlist for the session.

    Fetches Emby's master.m3u8 (using ``build_live_stream_url`` so
    the codec hints / transcoder params land), rewrites every
    variant URI to point at our seg proxy, and serves it with
    no-cache. Manifests change on every transcoder restart.
    """
    user_id = _user_id(request)
    session = get_default_store().get(session_token, user_id=user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    http = _http(request)
    if http is None:
        raise HTTPException(status_code=503, detail="service unavailable")

    # Reconstruct the master.m3u8 URL via the provider helper so the
    # transcoder hints (Static=false, VideoCodec=h264, etc.) match
    # what we'd send normally — keeps Emby on the codec path that
    # actually works in browsers.
    client = _provider_client(session.provider, http)
    if client is None:
        raise HTTPException(status_code=500, detail="provider not available")
    upstream_url = client.build_live_stream_url(
        session.base_url,
        channel_external_id=session.channel_id,
        media_source_id=session.media_source_id,
        play_session_id=session.play_session_id,
        token=session.access_token,
        max_audio_channels=2,
    )
    if not upstream_url:
        raise HTTPException(status_code=500, detail="could not build manifest url")

    try:
        upstream = await http.get(
            upstream_url,
            timeout=_MANIFEST_FETCH_TIMEOUT_S,
            follow_redirects=True,
        )
    except Exception as exc:
        log.warning(
            "livetv_master_fetch_failed",
            token_prefix=session_token[:8], error=str(exc),
        )
        raise HTTPException(status_code=502, detail="upstream master fetch failed")
    if upstream.status_code != 200:
        raise HTTPException(
            status_code=upstream.status_code,
            detail=f"upstream returned {upstream.status_code}",
        )
    rewritten = _rewrite_m3u8(upstream.text, session_token=session_token)
    return Response(
        content=rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/stream/{session_token}/seg/{seg_path:path}")
async def stream_segment(
    session_token: str,
    seg_path: str,
    request: Request,
) -> Response:
    """Proxy a sub-playlist (.m3u8) or a media segment (.ts / .mp4).

    Single endpoint for both because Emby's transcoded HLS keeps
    variants AND segments under the same ``/Videos/{channel_id}/``
    prefix. The path discriminant is the extension: ``.m3u8`` → fetch
    + rewrite, anything else → stream bytes.
    """
    user_id = _user_id(request)
    session = get_default_store().get(session_token, user_id=user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    http = _http(request)
    if http is None:
        raise HTTPException(status_code=503, detail="service unavailable")

    upstream_url = _build_upstream_url(
        session, seg_path, request.url.query,
    )

    is_playlist = seg_path.lower().endswith(".m3u8")
    if is_playlist:
        try:
            upstream = await http.get(
                upstream_url,
                timeout=_MANIFEST_FETCH_TIMEOUT_S,
                follow_redirects=True,
            )
        except Exception as exc:
            # repr() preserves the type for message-less httpx
            # exceptions (ConnectTimeout, RemoteProtocolError) that
            # otherwise log as empty error= and lose diagnostic info.
            log.warning(
                "livetv_variant_fetch_failed",
                token_prefix=session_token[:8], seg_path=seg_path,
                error=repr(exc),
            )
            raise HTTPException(status_code=502, detail="upstream variant fetch failed")
        if upstream.status_code != 200:
            raise HTTPException(
                status_code=upstream.status_code,
                detail=f"upstream returned {upstream.status_code}",
            )
        rewritten = _rewrite_m3u8(upstream.text, session_token=session_token)
        return Response(
            content=rewritten,
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-store"},
        )

    # Segment — stream bytes through so a long segment can start
    # playing in the browser before the upstream finishes sending.
    # No range support: HLS segments are atomic units, the player
    # asks for whole segments.
    try:
        upstream_resp = await http.send(
            http.build_request("GET", upstream_url),
            stream=True,
            follow_redirects=True,
        )
    except Exception as exc:
        log.warning(
            "livetv_segment_fetch_failed",
            token_prefix=session_token[:8], seg_path=seg_path,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail="upstream segment fetch failed")
    if upstream_resp.status_code != 200:
        await upstream_resp.aclose()
        raise HTTPException(
            status_code=upstream_resp.status_code,
            detail=f"upstream returned {upstream_resp.status_code}",
        )

    async def _iter():
        try:
            async for chunk in upstream_resp.aiter_bytes(_STREAM_CHUNK):
                yield chunk
        finally:
            await upstream_resp.aclose()

    content_type = upstream_resp.headers.get("content-type", "video/mp2t")
    return StreamingResponse(
        _iter(),
        media_type=content_type,
        headers={
            # Each segment is unique to this transcoder session; cache
            # for the segment's typical lifetime (10s) so a quick
            # rewind hits the cache instead of re-transcoding.
            "Cache-Control": "public, max-age=10",
        },
    )


@router.post("/stop/{session_token}")
async def stop_play_session(
    session_token: str,
    request: Request,
) -> JSONResponse:
    """Tear down the upstream PlaySessionId so the tuner releases.

    Without this, Emby keeps the tuner pinned for several minutes
    after the user closes the player — and on a single-tuner setup
    that means the NEXT play attempt fails. We POST to
    ``/Sessions/Playing/Stopped`` with the session's PlaySessionId,
    then drop the token from our store.

    Idempotent: a second stop call returns ``{ok: true, already:
    true}`` rather than 404 so reload-races don't surface as errors.
    """
    user_id = _user_id(request)
    session = get_default_store().remove(session_token, user_id=user_id)
    if session is None:
        return JSONResponse({"ok": True, "already": True})

    http = _http(request)
    if http is None:
        # Session is gone from our store regardless; the upstream
        # tuner will time out on its own.
        return JSONResponse({"ok": True, "upstream_released": False})

    base = session.base_url.rstrip("/") + _api_prefix_for(session.provider)
    try:
        await http.post(
            f"{base}/Sessions/Playing/Stopped",
            params={
                "api_key":       session.access_token,
                "PlaySessionId": session.play_session_id,
                "ItemId":        session.channel_id,
                "MediaSourceId": session.media_source_id,
            },
            timeout=10.0,
        )
        released = True
    except Exception as exc:
        log.warning(
            "livetv_stop_upstream_failed",
            token_prefix=session_token[:8], error=str(exc),
        )
        released = False
    return JSONResponse({"ok": True, "upstream_released": released})
