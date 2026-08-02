"""YouTube tool — find videos, fetch transcripts, search via SearXNG."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC
from urllib.parse import parse_qs, urlparse

import httpx

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?.*v=|embed/|shorts/|live/)"
    r"|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)


def _extract_video_id(url_or_id: str) -> str | None:
    url_or_id = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = _VIDEO_ID_RE.search(url_or_id)
    if m:
        return m.group(1)
    # Fallback: parse query string (handles uppercase ?V= and unusual formats)
    parsed = urlparse(url_or_id)
    qs = parse_qs(parsed.query)
    # Check both 'v' and 'V' (some copy-pasted URLs have uppercase)
    v = qs.get("v") or qs.get("V")
    if v and len(v[0]) == 11:
        return v[0]
    return None


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _group_into_paragraphs(
    segments: list[dict], gap_threshold: float = 2.0,
) -> list[dict]:
    """Group caption segments into readable paragraphs by sentence boundaries and time gaps."""
    paragraphs: list[dict] = []
    current: dict = {"text": "", "start": 0.0, "lines": []}

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if not current["text"]:
            current["start"] = seg.get("start", 0.0)
        if current["lines"]:
            prev = current["lines"][-1]
            time_gap = seg["start"] - (prev["start"] + prev.get("duration", 0))
            ends_sentence = current["text"].rstrip().endswith((".", "!", "?", '."', '!"', '?"'))
            if time_gap > gap_threshold or (ends_sentence and time_gap > 0.5):
                paragraphs.append(current)
                current = {"text": "", "start": seg.get("start", 0.0), "lines": []}
        current["text"] += (" " if current["text"] else "") + text
        current["lines"].append(seg)

    if current["text"]:
        paragraphs.append(current)
    return paragraphs


def _humanize_views(count) -> str:
    try:
        n = int(count)
    except (TypeError, ValueError):
        return str(count) if count else ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M views"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K views"
    return f"{n} views"


def _humanize_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        now = datetime.now(UTC)
        days = (now - dt).days
        if days < 1:
            return "today"
        if days < 7:
            return f"{days}d ago"
        if days < 30:
            return f"{days // 7}w ago"
        if days < 365:
            return f"{days // 30}mo ago"
        return f"{days // 365}y ago"
    except Exception:
        return date_str[:10] if len(date_str) >= 10 else date_str


class YouTubeTool(Tool):
    """Find and watch YouTube videos with transcripts."""

    def __init__(self, http_client: httpx.AsyncClient, searxng_url: str = "") -> None:
        self._client = http_client
        self._searxng_url = searxng_url.rstrip("/")

    @property
    def name(self) -> str:
        return "youtube"

    @property
    def description(self) -> str:
        return (
            "Find and watch YouTube videos with transcripts. "
            "Provide a YouTube URL or video ID to get the transcript directly, "
            "or describe what you want to watch to discover videos."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "YouTube URL, video ID, or search terms",
                },
                "language": {
                    "type": "string",
                    "description": "Language code (default: en)",
                    "default": "en",
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 25.0

    @property
    def cacheable(self) -> bool:
        return True

    @property
    def cache_ttl(self) -> float:
        return 3600.0

    @property
    def produces(self) -> list[str]:
        return ["text", "video_metadata"]

    @property
    def requires_services(self) -> list[str]:
        return ["searxng"]

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "Transcripts are disabled": "This video has captions disabled. The video can still be watched but no transcript is available.",
            "not found": "Could not find that video. Check the URL or try different search terms.",
            "No YouTube results": "No YouTube videos found. Try broader or different terms.",
            "youtube-transcript-api": "Transcript library not installed. Answer from your knowledge.",
        }

    @property
    def model_hint(self) -> str:
        return "Use this when the user asks to find, watch, or learn from a video. Pass a YouTube URL for direct transcript, or search terms to discover videos."

    @property
    def auto_invoke_when_enabled(self) -> bool:
        # The user turning this tool on IS the video-search intent
        # signal — don't make them repeat it in the prompt.
        return True

    def health_check(self) -> bool:
        return bool(self._searxng_url)

    async def execute(self, *, query: str, language: str = "en", **kwargs) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(success=False, error="No query provided")
        video_id = _extract_video_id(query)
        if video_id:
            return await self._direct_mode(video_id, language)
        return await self._search_mode(query, language)

    async def _direct_mode(self, video_id: str, language: str) -> ToolResult:
        meta = await self._fetch_oembed(video_id)
        title = meta.get("title", f"Video {video_id}")
        channel = meta.get("author_name", "")
        thumbnail = meta.get("thumbnail_url", f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")
        transcript, error = await self._fetch_transcript(video_id, language)
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        if error:
            return ToolResult(
                success=True,
                output=f"Video: {title} by {channel}\n\nTranscript unavailable: {error}",
                metadata={
                    "youtube_mode": "direct",
                    "video_id": video_id,
                    "title": title,
                    "channel": channel,
                    "thumbnail": thumbnail,
                    "url": video_url,
                    "transcript": [],
                    "paragraphs": [],
                    "transcript_error": error,
                },
            )

        paragraphs = _group_into_paragraphs(transcript)
        full_text = " ".join(seg["text"] for seg in transcript)
        # The model reads `output` — a direct fetch exists so it can answer FROM
        # the transcript, so give it the real text, not an 800-char teaser (that
        # stub was why "grab the transcript" produced nothing usable). Only cap
        # absurdly long ones, and say so explicitly instead of a silent "...".
        _TRANSCRIPT_CAP = 20000
        if len(full_text) > _TRANSCRIPT_CAP:
            body = (
                full_text[:_TRANSCRIPT_CAP]
                + f"\n\n[Transcript truncated here — {len(full_text) - _TRANSCRIPT_CAP}"
                " more characters follow in the video; ask to continue if you need them.]"
            )
        else:
            body = full_text

        from augmentum.security.untrusted import wrap_untrusted
        return ToolResult(
            success=True,
            output=wrap_untrusted(
                "web/youtube",
                f"Video: {title} by {channel}\n\nTranscript:\n{body}",
            ),
            metadata={
                "youtube_mode": "direct",
                "video_id": video_id,
                "title": title,
                "channel": channel,
                "thumbnail": thumbnail,
                "url": video_url,
                "transcript": transcript,
                "paragraphs": paragraphs,
            },
        )

    async def _search_mode(self, query: str, language: str) -> ToolResult:
        if not self._searxng_url:
            return ToolResult(success=False, error="Video search unavailable (no search service configured)")
        try:
            resp = await self._client.get(
                f"{self._searxng_url}/search",
                params={"q": query, "format": "json", "categories": "videos"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ToolResult(success=False, error=f"Video search failed: {exc}")

        # Quality filter: reject junk titles, non-English, bad results
        raw_results = data.get("results", [])
        try:
            from augmentum.discovery.quality import filter_for_video_ui
            raw_results = filter_for_video_ui(raw_results, context="general")
        except Exception as exc:
            log.debug("youtube_tool_quality_filter_failed", error=str(exc))

        # Cross-round dedup (turn_search_dedup): skip videos already shown to the
        # user earlier this turn so repeated rounds don't return the same clip;
        # the over-fetch (up to 6) backfills with fresh ones.
        from augmentum.tools.turn_search_dedup import get_turn_dedup
        dedup = get_turn_dedup()
        skipped_dup = 0

        # Collect candidates (more than 3 so we can filter dead ones)
        candidates = []
        seen_ids: set[str] = set()
        for r in raw_results:
            url = r.get("url", "")
            vid = _extract_video_id(url)
            if not vid or vid in seen_ids:
                continue
            seen_ids.add(vid)
            if dedup is not None and dedup.seen("video", vid):
                skipped_dup += 1
                continue
            candidates.append({
                "video_id": vid,
                "title": r.get("title", ""),
                "channel": r.get("author", r.get("engine", "")),
                "thumbnail": f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
                "duration": r.get("length", r.get("duration", "")),
                "views": _humanize_views(r.get("metadata", "")),
                "published": _humanize_date(r.get("publishedDate", "")),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
            if len(candidates) >= 6:  # grab extras in case some are dead
                break

        if not candidates:
            if skipped_dup:
                return ToolResult(success=False, error=(
                    f"All matching videos were already shown to the user earlier this "
                    f"turn ({skipped_dup} skipped). Don't search for the same video "
                    f"again — answer with what's already shown."
                ))
            return ToolResult(success=False, error=f"No YouTube results found for: {query}")

        # Validate candidates in parallel — oEmbed returns 200 for live, embeddable videos
        results = await self._validate_videos(candidates)

        # Mark the surfaced videos as shown this turn.
        if dedup is not None:
            for r in results:
                dedup.mark("video", r.get("video_id", ""))

        lines = [f"Found {len(results)} videos:"]
        for i, r in enumerate(results, 1):
            dur = f" ({r['duration']})" if r["duration"] else ""
            lines.append(f"{i}. {r['title']}{dur} — {r['channel']}")
            # The URL is what the model needs to fetch a transcript next — it
            # MUST be in the model-facing output, not only in metadata (which
            # goes to the UI panel). Without it the model can't act on these
            # results and re-searches in a loop.
            lines.append(f"   {r['url']}")
        lines.append(
            "\nTo read a video's transcript, call this tool again with its URL "
            "above (you can pass several in one turn). These ARE the results — "
            "do not search again for the same thing."
        )
        if skipped_dup:
            lines.append(
                f"(Note: {skipped_dup} video(s) already shown earlier this turn were "
                f"omitted. Avoid re-searching the same topic.)"
            )

        from augmentum.security.untrusted import wrap_untrusted
        return ToolResult(
            success=True,
            output=wrap_untrusted("web/youtube", "\n".join(lines)),
            metadata={"youtube_mode": "search", "results": results},
        )

    async def _validate_videos(self, candidates: list[dict], max_results: int = 3) -> list[dict]:
        """Validate video availability via oEmbed in parallel. Returns up to max_results live videos."""
        async def _check(candidate: dict) -> dict | None:
            try:
                url = f"https://www.youtube.com/watch?v={candidate['video_id']}"
                resp = await self._client.get(
                    "https://www.youtube.com/oembed",
                    params={"url": url, "format": "json"},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    # Enrich with oEmbed metadata (more accurate title/channel)
                    oembed = resp.json()
                    if oembed.get("title"):
                        candidate["title"] = oembed["title"]
                    if oembed.get("author_name"):
                        candidate["channel"] = oembed["author_name"]
                    return candidate
            except Exception as exc:
                log.debug(
                    "youtube_tool_oembed_check_failed",
                    vid=candidate.get("id"),
                    error=str(exc),
                )
            return None

        # Check all candidates concurrently
        checks = await asyncio.gather(*[_check(c) for c in candidates], return_exceptions=True)
        results = []
        for result in checks:
            if isinstance(result, dict):
                results.append(result)
                if len(results) >= max_results:
                    break
        return results

    async def _fetch_oembed(self, video_id: str) -> dict:
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            resp = await self._client.get(
                "https://www.youtube.com/oembed",
                params={"url": url, "format": "json"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            log.debug("youtube_oembed_failed", video_id=video_id)
        return {}

    async def _fetch_transcript(self, video_id: str, language: str) -> tuple[list[dict], str]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return [], "youtube-transcript-api is not installed"
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = await asyncio.to_thread(ytt_api.fetch, video_id, languages=[language, "en"])
            segments = [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]
            return segments, ""
        except Exception as exc:
            error_msg = str(exc)
            if "TranscriptsDisabled" in error_msg or "disabled" in error_msg.lower():
                return [], "Transcripts are disabled for this video"
            if "NoTranscriptFound" in error_msg:
                return [], f"No transcript found in language '{language}'"
            return [], f"Transcript fetch failed: {error_msg}"
