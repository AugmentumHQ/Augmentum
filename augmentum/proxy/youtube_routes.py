"""YouTube transcript API — frontend-initiated video data fetch."""

from __future__ import annotations

import asyncio
from time import monotonic

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["youtube"])

_cache: dict[tuple[str, str], tuple[dict, float]] = {}
_CACHE_TTL = 3600.0
_CACHE_MAX = 100


def _cache_get(video_id: str, lang: str) -> dict | None:
    key = (video_id, lang)
    entry = _cache.get(key)
    if entry and (monotonic() - entry[1]) < _CACHE_TTL:
        return entry[0]
    if entry:
        del _cache[key]
    return None


def _cache_set(video_id: str, lang: str, data: dict) -> None:
    if len(_cache) >= _CACHE_MAX:
        oldest_key = min(_cache, key=lambda k: _cache[k][1])
        del _cache[oldest_key]
    _cache[(video_id, lang)] = (data, monotonic())


@router.get("/api/youtube/transcript")
async def get_transcript(request: Request, v: str = "", lang: str = "en") -> JSONResponse:
    """Fetch video metadata + transcript for the YouTube panel."""
    v = v.strip()
    if not v or len(v) != 11:
        return JSONResponse({"error": "Invalid video ID"}, status_code=400)

    cached = _cache_get(v, lang)
    if cached:
        return JSONResponse(cached)

    http_client = getattr(request.app.state, "http_client", None)
    meta = {}
    if http_client:
        try:
            url = f"https://www.youtube.com/watch?v={v}"
            resp = await http_client.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                meta = resp.json()
        except Exception:
            log.debug("youtube_oembed_failed", video_id=v)

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        raw = await asyncio.to_thread(ytt_api.fetch, v, languages=[lang, "en"])
        segments = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in raw
        ]
    except ImportError:
        return JSONResponse({"error": "Transcript library not installed"}, status_code=503)
    except Exception as exc:
        data = {
            "video_id": v,
            "title": meta.get("title", ""),
            "channel": meta.get("author_name", ""),
            "thumbnail": meta.get("thumbnail_url", f"https://img.youtube.com/vi/{v}/hqdefault.jpg"),
            "url": f"https://www.youtube.com/watch?v={v}",
            "transcript": [],
            "paragraphs": [],
            "transcript_error": str(exc),
        }
        _cache_set(v, lang, data)
        return JSONResponse(data)

    from augmentum.tools.youtube import _group_into_paragraphs
    paragraphs = _group_into_paragraphs(segments)

    data = {
        "video_id": v,
        "title": meta.get("title", ""),
        "channel": meta.get("author_name", ""),
        "thumbnail": meta.get("thumbnail_url", f"https://img.youtube.com/vi/{v}/hqdefault.jpg"),
        "url": f"https://www.youtube.com/watch?v={v}",
        "transcript": segments,
        "paragraphs": paragraphs,
    }

    _cache_set(v, lang, data)
    return JSONResponse(data)


@router.get("/api/youtube/related")
async def get_related_videos(request: Request, q: str = "", exclude: str = "") -> JSONResponse:
    """Search for related YouTube videos. Used by the panel for rabbit-hole browsing."""
    q = q.strip()
    if not q:
        return JSONResponse({"results": []})

    from augmentum.config import settings
    from augmentum.tools.youtube import _extract_video_id, _humanize_date, _humanize_views

    http_client = getattr(request.app.state, "http_client", None)
    if not http_client or not settings.searxng_base_url:
        return JSONResponse({"results": []})

    try:
        resp = await http_client.get(
            f"{settings.searxng_base_url.rstrip('/')}/search",
            params={"q": q, "format": "json", "categories": "videos"},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return JSONResponse({"results": []})

    # Quality filter: reject junk titles, non-English results
    raw_results = data.get("results", [])
    try:
        from augmentum.discovery.quality import filter_for_video_ui
        exclude_ids = {exclude} if exclude else set()
        raw_results = filter_for_video_ui(raw_results, context="related", exclude_ids=exclude_ids)
    except Exception as exc:
        log.debug("youtube_related_filter_failed", error=str(exc))

    candidates = []
    seen_ids = {exclude} if exclude else set()
    for r in raw_results:
        url = r.get("url", "")
        vid = _extract_video_id(url)
        if not vid or vid in seen_ids:
            continue
        seen_ids.add(vid)
        candidates.append({
            "video_id": vid,
            "title": r.get("title", ""),
            "channel": r.get("author", r.get("engine", "")),
            "thumbnail": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
            "duration": r.get("length", r.get("duration", "")),
            "views": _humanize_views(r.get("metadata", "")),
            "published": _humanize_date(r.get("publishedDate", "")),
        })
        if len(candidates) >= 10:
            break

    # Verify playability via oEmbed (free, no API key)
    import asyncio

    async def _check(vid: str) -> bool:
        try:
            r = await http_client.get(
                f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={vid}&format=json",
                timeout=5.0, follow_redirects=True,
            )
            return r.status_code == 200
        except Exception:
            return True  # Assume playable on network error

    checks = await asyncio.gather(*[_check(c["video_id"]) for c in candidates])
    results = [c for c, ok in zip(candidates, checks) if ok][:5]

    return JSONResponse({"results": results})
