"""Shared Emby/Jellyfin provider helpers."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from augmentum.media.library_classification import classify_library
from augmentum.media.providers.base import (
    CatalogItem,
    LibraryView,
    ProviderInfo,
    RemoteSession,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

_TIMEOUT_S = 10.0
_CATALOG_TIMEOUT_S = 60.0
_PAGE_SIZE = 500
_TICKS_PER_SECOND = 10_000_000
_CLIENT_NAME = "Augmentum"
_CLIENT_DEVICE = "Augmentum Server"
_CLIENT_VERSION = "1.0"
_CLIENT_DEVICE_ID = "augmentum-media"
_PRIMARY_SAMPLE_TYPES = ("Movie", "Series", "Season", "Episode", "MusicVideo", "Program")
_SUPPORTED_GROUPS = frozenset({"movies", "shows", "music_videos"})
_REMOTE_PLAY_COMMANDS = frozenset({"PlayNow", "PlayNext", "PlayLast"})
_REMOTE_PLAYSTATE_COMMANDS = frozenset({
    "Stop",
    "Pause",
    "Unpause",
    "NextTrack",
    "PreviousTrack",
    "Seek",
    "Rewind",
    "FastForward",
    "PlayPause",
    "SeekRelative",
})
_REMOTE_GENERAL_COMMANDS = frozenset({
    "VolumeUp",
    "VolumeDown",
    "SetVolume",
    "Mute",
    "Unmute",
    "ToggleMute",
    "SetAudioStreamIndex",
    "SetSubtitleStreamIndex",
})
_USER_ID_CACHE: dict[tuple[str, str, str], str] = {}


class EmbyCompatBase:
    name = ""
    api_prefix = ""
    auth_scheme = "MediaBrowser"
    item_list_path_uses_user = False
    user_views_path_uses_user = False
    user_data_path_uses_user = False

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._http = http_client

    async def ping(self, base_url: str) -> ProviderInfo | None:
        url = self._api_base(base_url)
        try:
            resp = await self._http.get(
                f"{url}/System/Info/Public",
                headers={"Accept": "application/json"},
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return None
            body = resp.json() or {}
            version = str(body.get("Version") or "")
            if not version:
                return None
            server_name = str(
                body.get("ServerName")
                or body.get("LocalAddress")
                or body.get("WanAddress")
                or self.name
            )
            return ProviderInfo(
                provider=self.name,
                base_url=base_url.rstrip("/"),
                server_name=server_name,
                version=version,
                is_initialized=True,
            )
        except Exception as exc:
            log.debug("emby_compat_ping_failed", provider=self.name, error=str(exc))
            return None

    async def login(self, base_url: str, username: str, password: str) -> str:
        resp = await self._http.post(
            f"{self._api_base(base_url)}/Users/AuthenticateByName",
            json={"Username": username, "Pw": password},
            headers=self._headers(),
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        if resp.status_code in (401, 403):
            raise ValueError("Invalid username or password")
        if resp.status_code != 200:
            raise RuntimeError(f"Login failed: HTTP {resp.status_code}")
        body = resp.json() or {}
        token = str(body.get("AccessToken") or body.get("accessToken") or "")
        if not token:
            raise RuntimeError("Login response missing access token")
        user = body.get("User") or {}
        user_id = str(user.get("Id") or user.get("id") or "").strip()
        if user_id:
            _USER_ID_CACHE[self._cache_key(base_url, token)] = user_id
        return token

    async def change_password(
        self, base_url: str, username: str,
        current_password: str, new_password: str,
    ) -> str:
        """Change the managed account's password; return a fresh token.

        Emby/Jellyfin: ``POST /Users/{userId}/Password`` with
        ``{CurrentPw, NewPw}`` and the current token. The existing token
        survives the change, but we re-login with the new password so the
        caller stores a credential that matches what's persisted.
        """
        token = await self.login(base_url, username, current_password)
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            raise RuntimeError("could not resolve user id for password change")
        resp = await self._http.post(
            f"{self._api_base(base_url)}/Users/{user_id}/Password",
            headers=self._headers(token),
            json={"CurrentPw": current_password, "NewPw": new_password},
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"change password failed: HTTP {resp.status_code}")
        return await self.login(base_url, username, new_password)

    async def verify_token(self, base_url: str, token: str) -> bool:
        try:
            resp = await self._http.get(
                f"{self._api_base(base_url)}/Sessions",
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def discover_libraries(self, base_url: str, token: str) -> list[LibraryView]:
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            return []
        url = self._views_url(base_url, user_id)
        resp = await self._http.get(
            url,
            headers=self._headers(token),
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"List libraries failed: HTTP {resp.status_code}")
        body = resp.json() or {}
        if isinstance(body, list):
            rows = body
        else:
            rows = body.get("Items") or body.get("items") or []
        libraries: list[LibraryView] = []
        for raw in rows:
            library_id = str(raw.get("Id") or raw.get("id") or "").strip()
            if not library_id:
                continue
            sample_counts = await self._sample_library_types(
                base_url, token, user_id=user_id, library_id=library_id,
            )
            view = LibraryView(
                external_id=library_id,
                name=str(raw.get("Name") or raw.get("name") or "Library").strip() or "Library",
                view_type=str(raw.get("Type") or raw.get("type") or "").strip(),
                collection_type=str(
                    raw.get("CollectionType")
                    or raw.get("CollectionType")
                    or raw.get("collectionType")
                    or ""
                ).strip(),
                sample_type_counts=sample_counts,
                extra={},
            )
            classified = classify_library(
                collection_type=view.collection_type,
                sample_type_counts=sample_counts,
                view_type=view.view_type,
            )
            view.extra.update({
                "detected_group": classified.detected_group,
                "detected_primary_entity": classified.detected_primary_entity,
                "detection_confidence": classified.detection_confidence,
            })
            libraries.append(view)
        return libraries

    async def list_live_channels(
        self, base_url: str, token: str, *,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        """Return the user's Live TV channels.

        Emby/Jellyfin expose Live TV as a first-class subsystem when the
        user has configured a tuner (HDHR, m3u, Plex DVR, etc.). Each
        channel returned is a BaseItemDto with ``ChannelType``,
        ``ChannelNumber``, ``ImageTags.Primary``, and optionally
        ``CurrentProgram`` if the server has EPG.

        Returns the raw item dicts (caller converts to ``CatalogItem``).
        Empty list on any error or when Live TV isn't enabled for the
        user — Live TV is optional infrastructure, missing it isn't a
        provider failure.
        """
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            return []
        params: dict[str, Any] = {
            "UserId": user_id,
            "EnableImages": "true",
            "EnableUserData": "true",
            "AddCurrentProgram": "true",
            # ChannelInfo carries ChannelNumber + Tags (broadcast network).
            # Overview gives us description for the channel itself.
            "Fields": "ChannelInfo,Overview,Genres",
        }
        if limit > 0:
            params["Limit"] = int(limit)
        try:
            resp = await self._http.get(
                f"{self._api_base(base_url)}/LiveTv/Channels",
                headers=self._headers(token),
                params=params,
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                log.debug(
                    "livetv_channels_non_200",
                    provider=self.name, status=resp.status_code,
                )
                return []
            body = resp.json() or {}
            if isinstance(body, dict):
                items = body.get("Items") or body.get("items") or []
            elif isinstance(body, list):
                items = body
            else:
                items = []
            return [it for it in items if isinstance(it, dict)]
        except Exception as exc:
            log.debug(
                "livetv_channels_failed", provider=self.name, error=str(exc),
            )
            return []

    async def list_live_programs(
        self, base_url: str, token: str, *,
        channel_external_ids: tuple[str, ...] = (),
        max_results: int = 200,
        hours_ahead: float = 12.0,
    ) -> list[dict[str, Any]]:
        """Return upcoming EPG entries (programmes) within a time window.

        Default window: now → +12h, capped at ``max_results``. Pass
        ``channel_external_ids`` to scope to specific channels; empty =
        all channels the user has access to.

        Uses ISO 8601 with 'Z' suffix per Emby/Jellyfin's expectation;
        their server normalises to UTC internally.

        Empty list on error or when no EPG is available — many setups
        run without EPG (no XMLTV / Schedules Direct configured) and
        that's not a failure.
        """
        import datetime as _dt
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            return []
        now = _dt.datetime.now(_dt.UTC)
        end = now + _dt.timedelta(hours=max(0.5, float(hours_ahead)))
        params: dict[str, Any] = {
            "UserId": user_id,
            "MinStartDate": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "MaxStartDate": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "Limit": int(max(1, max_results)),
            "EnableImages": "false",
            "Fields": "Overview,Genres",
        }
        if channel_external_ids:
            # Server expects comma-separated; cap to keep URL under 2K.
            params["ChannelIds"] = ",".join(channel_external_ids[:50])
        try:
            resp = await self._http.get(
                f"{self._api_base(base_url)}/LiveTv/Programs",
                headers=self._headers(token),
                params=params,
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                log.debug(
                    "livetv_programs_non_200",
                    provider=self.name, status=resp.status_code,
                )
                return []
            body = resp.json() or {}
            items = (
                body.get("Items") or body.get("items") or []
                if isinstance(body, dict) else body if isinstance(body, list) else []
            )
            return [it for it in items if isinstance(it, dict)]
        except Exception as exc:
            log.debug(
                "livetv_programs_failed", provider=self.name, error=str(exc),
            )
            return []

    async def fetch_live_channels(
        self, base_url: str, token: str,
    ) -> list[CatalogItem]:
        """Return Live TV channels as typed ``CatalogItem`` rows.

        Sister to :meth:`fetch_catalog` (VOD). Live channels stay on a
        separate code path because their identity is fundamentally
        different — no duration, no chapters, no resume position, EPG-
        driven 'now playing' instead of static metadata. Shoehorning
        them through :meth:`_item_from_row` would lose information.

        Each ``CatalogItem`` carries:

        - ``kind='live_video'``
        - ``extra.channel_number`` (e.g. '6.2', '13.1')
        - ``extra.current_program`` inline (drives the 'Live now' UX
          without needing a separate EPG query)
        - ``extra.has_logo_{primary,light,dark}`` flags — the UI picks
          the right variant for the current theme
        - ``extra.is_favorite`` + ``extra.play_count`` from
          UserData — feeds the Favorites + Recently Watched rails
        - ``extra.source_provider`` = 'emby' | 'jellyfin'

        ``stream_path`` is empty by design — live URLs need a
        ``fetch_playback_info`` call at play-time to mint a
        ``PlaySessionId``. The route layer composes
        ``fetch_playback_info`` + ``build_live_stream_url`` per play.
        """
        raw = await self.list_live_channels(base_url, token)
        items: list[CatalogItem] = []
        for row in raw:
            ext_id = str(row.get("Id") or "").strip()
            name = str(row.get("Name") or "").strip()
            if not ext_id or not name:
                continue
            cur_prog_raw = row.get("CurrentProgram") or {}
            cur_prog_raw = cur_prog_raw if isinstance(cur_prog_raw, dict) else {}
            user_data = row.get("UserData") or {}
            user_data = user_data if isinstance(user_data, dict) else {}
            image_tags = row.get("ImageTags") or {}
            image_tags = image_tags if isinstance(image_tags, dict) else {}
            current_program: dict[str, Any] | None = None
            if cur_prog_raw:
                current_program = {
                    "id":         str(cur_prog_raw.get("Id") or ""),
                    "name":       str(cur_prog_raw.get("Name") or ""),
                    "start_date": str(cur_prog_raw.get("StartDate") or ""),
                    "end_date":   str(cur_prog_raw.get("EndDate") or ""),
                    "overview":   str(cur_prog_raw.get("Overview") or ""),
                    "is_news":    bool(cur_prog_raw.get("IsNews") or False),
                    "is_sports":  bool(cur_prog_raw.get("IsSports") or False),
                    "is_kids":    bool(cur_prog_raw.get("IsKids") or False),
                    "is_movie":   bool(cur_prog_raw.get("IsMovie") or False),
                    "is_series":  bool(cur_prog_raw.get("IsSeries") or False),
                }
            extra: dict[str, Any] = {
                "channel_number":   str(row.get("Number") or row.get("ChannelNumber") or "").strip(),
                "media_type":       str(row.get("MediaType") or "Video"),
                "channel_type":     str(row.get("ChannelType") or row.get("Type") or "TvChannel"),
                "is_live":          True,
                "source_provider":  self.name,
                "is_favorite":      bool(user_data.get("IsFavorite") or False),
                "play_count":       int(user_data.get("PlayCount") or 0),
                "has_logo_primary": bool(image_tags.get("Primary")),
                "has_logo_light":   bool(image_tags.get("LogoLight")),
                "has_logo_dark":    bool(image_tags.get("LogoDark")),
                "current_program":  current_program,
            }
            items.append(CatalogItem(
                external_id=ext_id,
                name=name,
                kind="live_video",
                mime_type="application/vnd.apple.mpegurl",   # HLS — Emby transcodes
                size_bytes=0,
                duration_ms=0,
                progress_pct=0.0,
                cover_url="",       # built by route layer with auth
                author="",
                narrator="",
                stream_path="",     # resolved at play-time via PlaybackInfo
                extra=extra,
            ))
        return items

    async def fetch_catalog(self, base_url: str, token: str) -> list[CatalogItem]:
        libraries = await self.discover_libraries(base_url, token)
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            return []

        items: list[CatalogItem] = []
        for library in libraries:
            classified = classify_library(
                collection_type=library.collection_type,
                sample_type_counts=library.sample_type_counts,
                view_type=library.view_type,
            )
            if classified.detected_group not in _SUPPORTED_GROUPS:
                continue
            include_types = _include_types_for_group(classified.detected_group)
            rows = await self._fetch_library_items(
                base_url,
                token,
                user_id=user_id,
                library=library,
                include_types=include_types,
            )
            for raw in rows:
                item = self._item_from_row(
                    raw,
                    library=library,
                    detected_group=classified.detected_group,
                )
                if item is not None:
                    items.append(item)
        return items

    def build_stream_url(self, base_url: str, stream_path: str, token: str) -> str:
        path = stream_path if stream_path.startswith("/") else f"/{stream_path}"
        sep = "&" if "?" in path else "?"
        return f"{self._api_base(base_url)}{path}{sep}api_key={token}"

    def build_browser_video_stream_url(
        self,
        base_url: str,
        *,
        external_id: str,
        media_source_id: str,
        play_session_id: str,
        token: str,
        audio_stream_index: int | None = None,
        audio_codec: str = "aac",
        max_audio_channels: int | None = 2,
        start_time_ticks: int | None = None,
    ) -> str:
        if not external_id or not media_source_id or not play_session_id:
            return ""
        params: dict[str, str | int] = {
            "MediaSourceId": media_source_id,
            "PlaySessionId": play_session_id,
            # A/V sync correction params — partial-transcode (audio re-encode +
            # video stream-copy + container remux) is where lip-sync drift
            # originates. Without these hints, FFmpeg's audio path can pick
            # up timestamps that don't align with the kept-original video
            # PTS, and drift accumulates over the file.
            #
            # Static=false explicitly engages the dynamic-transcode pipeline
            # in Emby/Jellyfin, which enables their A/V sync filter that
            # otherwise stays dormant on copy paths.
            #
            # AudioSampleRate=48000 pins the output sample rate. Most TV /
            # film audio is already 48kHz (so usually a no-op), but for
            # 44.1kHz sources the resampler bypass-vs-engage decision is
            # what causes timing errors — pinning forces a single, known
            # path. Both Emby and Jellyfin ignore unknown params, so the
            # one extra hint is safe across forks/versions.
            "Static": "false",
            "AudioSampleRate": 48000,
        }
        if audio_codec:
            params["AudioCodec"] = str(audio_codec).strip().lower()
        if audio_stream_index is not None:
            params["AudioStreamIndex"] = int(audio_stream_index)
        if max_audio_channels is not None and max_audio_channels > 0:
            params["MaxAudioChannels"] = int(max_audio_channels)
        if start_time_ticks is not None and start_time_ticks > 0:
            params["StartTimeTicks"] = int(start_time_ticks)
        query = urlencode(params)
        return (
            f"{self._api_base(base_url)}/Videos/{external_id}/stream.mp4"
            f"?{query}&api_key={token}"
        )

    def build_cover_url(self, base_url: str, external_id: str, token: str) -> str:
        return f"{self._api_base(base_url)}/Items/{external_id}/Images/Primary?api_key={token}"

    def build_channel_logo_url(
        self,
        base_url: str,
        external_id: str,
        token: str,
        *,
        variant: str = "Primary",
        max_height: int = 240,
    ) -> str:
        """Build a channel-logo URL with explicit variant + size.

        Emby/Jellyfin channels carry up to three logo image variants in
        ``ImageTags``: ``Primary`` (full-color, the safe default),
        ``LogoLight`` (for dark UIs), ``LogoDark`` (for light UIs).
        Picking the right variant per UI theme is what makes the
        channel tiles look native instead of "stretched cable-box logo".

        Capped at ``max_height`` so we don't pull a 2K source asset just
        to render a 120px tile — saves bandwidth and keeps the proxy
        cache hit-rate sane.
        """
        v = variant if variant in ("Primary", "LogoLight", "LogoDark") else "Primary"
        h = max(48, min(int(max_height or 240), 1024))
        return (
            f"{self._api_base(base_url)}/Items/{external_id}/Images/{v}"
            f"?maxHeight={h}&api_key={token}"
        )

    def build_live_stream_url(
        self,
        base_url: str,
        *,
        channel_external_id: str,
        media_source_id: str,
        play_session_id: str,
        token: str,
        max_audio_channels: int = 2,
    ) -> str:
        """Build the HLS master playlist URL for a live channel.

        Emby's tuner feeds are typically MPEG-2 / AC-3 (cable / OTA
        codecs) — not browser-playable. Emby's transcoder converts to
        H.264 / AAC on the fly and serves an HLS master playlist.

        Returns the master.m3u8 URL with codec hints + auth. The
        browser hits this via hls.js (Chrome/Firefox/Edge) or
        natively (Safari). Emby starts a transcoding worker per
        unique PlaySessionId — the session is tracked server-side
        and torn down when playback stops.

        The ``Static=false`` hint is what tells Emby this is a
        transcode request rather than a remux pass-through (matches
        the rationale in :meth:`build_browser_video_stream_url`).
        """
        if not channel_external_id or not media_source_id or not play_session_id:
            return ""
        params = {
            "PlaySessionId":            play_session_id,
            "MediaSourceId":            media_source_id,
            "VideoCodec":               "h264",
            "AudioCodec":               "aac",
            "MaxAudioChannels":         int(max(1, max_audio_channels)),
            "TranscodingMaxAudioChannels": int(max(1, max_audio_channels)),
            "AudioSampleRate":          48000,
            # Static=false engages the dynamic-transcode pipeline (see
            # build_browser_video_stream_url comment for the lip-sync
            # rationale — same applies here).
            "Static":                   "false",
            # SegmentContainer=ts is the safe default — fmp4 is supported
            # on iOS 17+ + recent hls.js but TS is universally playable.
            "SegmentContainer":         "ts",
            # Most live transcoders run a single ABR rung, but if the
            # server *can* produce multiple, this hint biases toward
            # ones that fit a typical wifi link.
            "TranscodingMaxWidth":      1280,
        }
        query = urlencode(params)
        return (
            f"{self._api_base(base_url)}/Videos/{channel_external_id}/master.m3u8"
            f"?{query}&api_key={token}"
        )

    def build_subtitle_url(
        self,
        base_url: str,
        *,
        external_id: str,
        media_source_id: str,
        subtitle_stream_index: int,
        token: str,
        format: str = "vtt",
    ) -> str:
        if not external_id or not media_source_id or subtitle_stream_index < 0:
            return ""
        fmt = (format or "vtt").strip().lower() or "vtt"
        return (
            f"{self._api_base(base_url)}/Videos/{external_id}/{media_source_id}"
            f"/Subtitles/{subtitle_stream_index}/Stream.{fmt}?api_key={token}"
        )

    async def fetch_progress(self, base_url: str, token: str) -> dict[str, dict]:
        return {}

    async def fetch_item_details(
        self, base_url: str, token: str, *, external_id: str,
    ) -> dict | None:
        user_id = await self._resolve_user_id(base_url, token)
        url = self._item_url(base_url, external_id, user_id=user_id)
        try:
            resp = await self._http.get(
                url,
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return None
            raw = resp.json() or {}
            item_type = str(raw.get("Type") or "").strip()
            if item_type == "Series":
                raw["_augmentum_children"] = await self._fetch_children(
                    base_url, token, user_id=user_id, parent_id=external_id,
                    include_types=["Season"],
                    sort_by="IndexNumber",
                )
                raw["_augmentum_episodes"] = await self._fetch_children(
                    base_url, token, user_id=user_id, parent_id=external_id,
                    include_types=["Episode"],
                    sort_by="ParentIndexNumber,IndexNumber",
                    limit=2000,
                )
            elif item_type == "Season":
                raw["_augmentum_children"] = await self._fetch_children(
                    base_url, token, user_id=user_id, parent_id=external_id,
                    include_types=["Episode"],
                    sort_by="IndexNumber",
                )
            elif item_type in {"Movie", "Episode", "MusicVideo"}:
                playback = await self.fetch_playback_info(
                    base_url, token, external_id=external_id,
                )
                if playback:
                    raw["_augmentum_playback_info"] = playback
            return raw
        except Exception as exc:
            log.debug("emby_compat_details_failed", provider=self.name, error=str(exc))
            return None

    async def fetch_playback_info(
        self, base_url: str, token: str, *, external_id: str,
    ) -> dict | None:
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            return None
        try:
            resp = await self._http.get(
                f"{self._api_base(base_url)}/Items/{external_id}/PlaybackInfo",
                headers=self._headers(token),
                params={
                    "UserId": user_id,
                    "IsPlayback": "true",
                    "AutoOpenLiveStream": "true",
                    "StartTimeTicks": 0,
                },
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return None
            body = resp.json() or {}
            return body if isinstance(body, dict) else None
        except Exception as exc:
            log.debug(
                "emby_compat_playback_info_failed",
                provider=self.name,
                error=str(exc),
            )
            return None

    async def push_progress(
        self,
        base_url: str,
        token: str,
        *,
        external_id: str,
        current_time_s: float,
        duration_s: float,
        is_finished: bool = False,
    ) -> bool:
        user_id = await self._resolve_user_id(base_url, token)
        if not user_id:
            return False
        ticks = max(0, int(current_time_s * _TICKS_PER_SECOND))
        payload = {
            "PlaybackPositionTicks": ticks,
            "Played": bool(is_finished),
        }
        try:
            resp = await self._http.post(
                self._user_data_url(base_url, user_id=user_id, external_id=external_id),
                json=payload,
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            ok = 200 <= resp.status_code < 300
        except Exception:
            ok = False
        if is_finished:
            ok = await self._mark_played(base_url, token, user_id=user_id, external_id=external_id) or ok
        elif current_time_s <= 1:
            await self._mark_unplayed(base_url, token, user_id=user_id, external_id=external_id)
        return ok

    async def list_remote_sessions(
        self,
        base_url: str,
        token: str,
        *,
        media_type: str = "Video",
    ) -> list[RemoteSession]:
        user_id = await self._resolve_user_id(base_url, token)
        try:
            resp = await self._http.get(
                f"{self._api_base(base_url)}/Sessions",
                headers=self._headers(token),
                params=self._sessions_params(user_id=user_id),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return []
            rows = resp.json() or []
        except Exception as exc:
            log.debug("emby_compat_remote_sessions_failed", provider=self.name, error=str(exc))
            return []
        if not isinstance(rows, list):
            return []

        target_media = str(media_type or "").strip().lower()
        sessions: list[RemoteSession] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            session_id = str(row.get("Id") or "").strip()
            if not session_id:
                continue
            device_id = str(row.get("DeviceId") or "").strip()
            if device_id == _CLIENT_DEVICE_ID:
                continue
            supports_media = bool(row.get("SupportsMediaControl") or False)
            supports_remote = bool(
                row.get("SupportsRemoteControl")
                or row.get("SupportsMediaControl")
                or False
            )
            playable_types = _string_list(
                row.get("PlayableMediaTypes")
                or row.get("SupportedMediaTypes")
                or [],
            )
            if target_media and playable_types:
                playable_norm = {str(kind).strip().lower() for kind in playable_types if str(kind).strip()}
                if target_media not in playable_norm:
                    continue
            if not supports_media and not supports_remote:
                continue
            device_name = str(
                row.get("DeviceName")
                or row.get("DeviceFriendlyName")
                or row.get("FriendlyName")
                or ""
            ).strip()
            client_name = str(
                row.get("Client")
                or row.get("AppName")
                or ""
            ).strip()
            user_name = str(row.get("UserName") or "").strip()
            now_playing = row.get("NowPlayingItem") or {}
            play_state = row.get("PlayState") or {}
            now_playing_title = ""
            now_playing_item_id = ""
            duration_s = 0.0
            if isinstance(now_playing, dict):
                now_playing_title = str(now_playing.get("Name") or "").strip()
                now_playing_item_id = str(now_playing.get("Id") or "").strip()
                duration_s = _ticks_to_seconds(now_playing.get("RunTimeTicks"))
            supported_commands = _string_list(row.get("SupportedCommands") or [])
            current_time_s = 0.0
            is_paused = False
            is_muted = False
            can_seek = False
            volume_level = _int_or_none(row.get("VolumeLevel"))
            audio_stream_index = None
            subtitle_stream_index = None
            if isinstance(play_state, dict):
                current_time_s = _ticks_to_seconds(play_state.get("PositionTicks"))
                is_paused = bool(play_state.get("IsPaused") or False)
                is_muted = bool(play_state.get("IsMuted") or False)
                can_seek = bool(play_state.get("CanSeek") or False)
                play_state_volume = _int_or_none(play_state.get("VolumeLevel"))
                if play_state_volume is not None:
                    volume_level = play_state_volume
                audio_stream_index = _int_or_none(play_state.get("AudioStreamIndex"))
                subtitle_stream_index = _int_or_none(play_state.get("SubtitleStreamIndex"))
            name_parts = [part for part in (device_name, client_name) if part]
            session_name = " - ".join(name_parts) if name_parts else (user_name or session_id)
            sessions.append(RemoteSession(
                session_id=session_id,
                name=session_name,
                client=client_name,
                device_name=device_name,
                user_name=user_name,
                supports_media_control=supports_media,
                supports_remote_control=supports_remote,
                playable_media_types=playable_types,
                supported_commands=supported_commands,
                now_playing_title=now_playing_title,
                now_playing_item_id=now_playing_item_id,
                current_time_s=current_time_s,
                duration_s=duration_s,
                is_paused=is_paused,
                is_muted=is_muted,
                can_seek=can_seek,
                volume_level=volume_level,
                audio_stream_index=audio_stream_index,
                subtitle_stream_index=subtitle_stream_index,
                extra={
                    "device_id": device_id,
                    "icon_url": str(row.get("AppIconUrl") or "").strip(),
                    "last_activity_date": str(row.get("LastActivityDate") or "").strip(),
                    "last_playback_check_in": str(row.get("LastPlaybackCheckIn") or "").strip(),
                    "now_playing_item_type": str(now_playing.get("Type") or "").strip() if isinstance(now_playing, dict) else "",
                    "series_name": str(now_playing.get("SeriesName") or "").strip() if isinstance(now_playing, dict) else "",
                },
            ))
        sessions.sort(key=lambda session: (
            0 if session.now_playing_title else 1,
            session.name.lower(),
        ))
        return sessions

    async def remote_play(
        self,
        base_url: str,
        token: str,
        *,
        session_id: str,
        external_id: str,
        start_time_s: float = 0.0,
        play_command: str = "PlayNow",
        media_source_id: str = "",
        audio_stream_index: int | None = None,
        subtitle_stream_index: int | None = None,
    ) -> bool:
        session_id = str(session_id or "").strip()
        external_id = str(external_id or "").strip()
        play_command = str(play_command or "PlayNow").strip() or "PlayNow"
        if not session_id or not external_id or play_command not in _REMOTE_PLAY_COMMANDS:
            return False
        user_id = await self._resolve_user_id(base_url, token)
        start_ticks = max(0, int(float(start_time_s or 0.0) * _TICKS_PER_SECOND))
        payload: dict[str, Any] = {
            "ControllingUserId": user_id or None,
            "ItemIds": [external_id],
            "PlayCommand": play_command,
            "StartPositionTicks": start_ticks,
        }
        params: dict[str, Any] = {
            "ItemIds": external_id,
            "PlayCommand": play_command,
            "StartPositionTicks": start_ticks,
        }
        if user_id:
            params["ControllingUserId"] = user_id
        media_source_id = str(media_source_id or "").strip()
        if media_source_id:
            payload["MediaSourceId"] = media_source_id
            params["MediaSourceId"] = media_source_id
        if audio_stream_index is not None:
            payload["AudioStreamIndex"] = int(audio_stream_index)
            params["AudioStreamIndex"] = int(audio_stream_index)
        if subtitle_stream_index is not None:
            payload["SubtitleStreamIndex"] = int(subtitle_stream_index)
            params["SubtitleStreamIndex"] = int(subtitle_stream_index)
        try:
            resp = await self._http.post(
                f"{self._api_base(base_url)}/Sessions/{session_id}/Playing",
                headers=self._headers(token),
                params=params,
                json=payload,
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:
            log.debug("emby_compat_remote_play_failed", provider=self.name, error=str(exc))
            return False

    async def remote_command(
        self,
        base_url: str,
        token: str,
        *,
        session_id: str,
        command: str,
        seek_position_s: float | None = None,
    ) -> bool:
        session_id = str(session_id or "").strip()
        command = str(command or "").strip()
        if not session_id or command not in _REMOTE_PLAYSTATE_COMMANDS:
            return False
        user_id = await self._resolve_user_id(base_url, token)
        payload: dict[str, Any] = {
            "Command": command,
            "ControllingUserId": user_id or None,
        }
        params: dict[str, Any] = {}
        if user_id:
            params["ControllingUserId"] = user_id
        if seek_position_s is not None:
            seek_ticks = max(
                0,
                int(float(seek_position_s or 0.0) * _TICKS_PER_SECOND),
            )
            payload["SeekPositionTicks"] = seek_ticks
            params["SeekPositionTicks"] = seek_ticks
        try:
            resp = await self._http.post(
                f"{self._api_base(base_url)}/Sessions/{session_id}/Playing/{command}",
                headers=self._headers(token),
                params=params or None,
                json=payload,
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:
            log.debug("emby_compat_remote_command_failed", provider=self.name, error=str(exc))
            return False

    async def remote_general_command(
        self,
        base_url: str,
        token: str,
        *,
        session_id: str,
        command: str,
        arguments: dict[str, Any] | None = None,
    ) -> bool:
        session_id = str(session_id or "").strip()
        command = str(command or "").strip()
        if not session_id or command not in _REMOTE_GENERAL_COMMANDS:
            return False
        user_id = await self._resolve_user_id(base_url, token)
        clean_args = _string_map(arguments or {})
        payload: dict[str, Any] = {
            "ControllingUserId": user_id or None,
        }
        if self.name == "jellyfin" and clean_args:
            payload["Name"] = command
            payload["Arguments"] = clean_args
            endpoint = f"{self._api_base(base_url)}/Sessions/{session_id}/Command"
        else:
            if clean_args:
                payload["Arguments"] = clean_args
            endpoint = f"{self._api_base(base_url)}/Sessions/{session_id}/Command/{command}"
        try:
            resp = await self._http.post(
                endpoint,
                headers=self._headers(token),
                json=payload,
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            return 200 <= resp.status_code < 300
        except Exception as exc:
            log.debug("emby_compat_remote_general_command_failed", provider=self.name, error=str(exc))
            return False

    async def _fetch_library_items(
        self,
        base_url: str,
        token: str,
        *,
        user_id: str,
        library: LibraryView,
        include_types: list[str],
    ) -> list[dict]:
        out: list[dict] = []
        start_index = 0
        while True:
            resp = await self._http.get(
                self._items_url(base_url, user_id=user_id),
                headers=self._headers(token),
                params=self._items_params(
                    user_id=user_id,
                    parent_id=library.external_id,
                    start_index=start_index,
                    limit=_PAGE_SIZE,
                    include_types=include_types,
                ),
                timeout=_CATALOG_TIMEOUT_S,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"List items failed for {library.name}: HTTP {resp.status_code}"
                )
            body = resp.json() or {}
            rows = body.get("Items") or body.get("items") or []
            if not isinstance(rows, list):
                rows = []
            out.extend(rows)
            if len(rows) < _PAGE_SIZE:
                break
            start_index += len(rows)
        return out

    async def _fetch_children(
        self,
        base_url: str,
        token: str,
        *,
        user_id: str,
        parent_id: str,
        include_types: list[str],
        sort_by: str = "SortName",
        limit: int = 200,
    ) -> list[dict]:
        resp = await self._http.get(
            self._items_url(base_url, user_id=user_id),
            headers=self._headers(token),
            params=self._items_params(
                user_id=user_id,
                parent_id=parent_id,
                start_index=0,
                limit=limit,
                include_types=include_types,
                sort_by=sort_by,
            ),
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return []
        body = resp.json() or {}
        rows = body.get("Items") or body.get("items") or []
        return rows if isinstance(rows, list) else []

    async def _sample_library_types(
        self,
        base_url: str,
        token: str,
        *,
        user_id: str,
        library_id: str,
    ) -> dict[str, int]:
        resp = await self._http.get(
            self._items_url(base_url, user_id=user_id),
            headers=self._headers(token),
            params=self._items_params(
                user_id=user_id,
                parent_id=library_id,
                start_index=0,
                limit=60,
                include_types=list(_PRIMARY_SAMPLE_TYPES),
            ),
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
        if resp.status_code != 200:
            return {}
        body = resp.json() or {}
        rows = body.get("Items") or body.get("items") or []
        counter: Counter[str] = Counter()
        for raw in rows if isinstance(rows, list) else []:
            item_type = str(raw.get("Type") or raw.get("type") or "").strip()
            if item_type in _PRIMARY_SAMPLE_TYPES:
                counter[item_type] += 1
        return dict(counter)

    async def _resolve_user_id(self, base_url: str, token: str) -> str:
        cache_key = self._cache_key(base_url, token)
        cached = _USER_ID_CACHE.get(cache_key)
        if cached:
            return cached
        api_base = self._api_base(base_url)
        try:
            me_resp = await self._http.get(
                f"{api_base}/Users/Me",
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if me_resp.status_code == 200:
                body = me_resp.json() or {}
                user_id = str(body.get("Id") or body.get("id") or "").strip()
                if user_id:
                    _USER_ID_CACHE[cache_key] = user_id
                    return user_id
        except Exception as exc:
            # Falls through to /Sessions lookup; debug log so an Emby
            # API change is findable.
            log.debug("emby_user_id_users_me_failed", error=str(exc))
        try:
            sess_resp = await self._http.get(
                f"{api_base}/Sessions",
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            if sess_resp.status_code == 200:
                rows = sess_resp.json() or []
                if isinstance(rows, list):
                    for row in rows:
                        if str(row.get("DeviceId") or "") == _CLIENT_DEVICE_ID:
                            user_id = str(row.get("UserId") or "").strip()
                            if user_id:
                                _USER_ID_CACHE[cache_key] = user_id
                                return user_id
                    for row in rows:
                        user_id = str(row.get("UserId") or "").strip()
                        if user_id:
                            _USER_ID_CACHE[cache_key] = user_id
                            return user_id
        except Exception as exc:
            log.debug("emby_user_id_sessions_fallback_failed", error=str(exc))
        return ""

    def _headers(self, token: str = "") -> dict[str, str]:
        auth = (
            f'{self.auth_scheme} Client="{_CLIENT_NAME}", '
            f'Device="{_CLIENT_DEVICE}", DeviceId="{_CLIENT_DEVICE_ID}", '
            f'Version="{_CLIENT_VERSION}"'
        )
        if token:
            auth = f'{auth}, Token="{token}"'
        headers = {
            "Accept": "application/json",
            "X-Application": f"{_CLIENT_NAME}/{_CLIENT_VERSION}",
            "Authorization": auth,
            "X-Emby-Authorization": auth,
        }
        if token:
            headers["X-Emby-Token"] = token
            headers["X-MediaBrowser-Token"] = token
        return headers

    def _api_base(self, base_url: str) -> str:
        base = base_url.rstrip("/")
        if not self.api_prefix:
            return base
        lower = base.lower()
        if lower.endswith(self.api_prefix.lower()):
            return base
        return f"{base}{self.api_prefix}"

    def _cache_key(self, base_url: str, token: str) -> tuple[str, str, str]:
        return (self.name, self._api_base(base_url), token)

    def _views_url(self, base_url: str, user_id: str) -> str:
        if self.user_views_path_uses_user:
            return f"{self._api_base(base_url)}/Users/{user_id}/Views"
        return f"{self._api_base(base_url)}/UserViews"

    def _items_url(self, base_url: str, *, user_id: str) -> str:
        if self.item_list_path_uses_user:
            return f"{self._api_base(base_url)}/Users/{user_id}/Items"
        return f"{self._api_base(base_url)}/Items"

    def _item_url(self, base_url: str, external_id: str, *, user_id: str) -> str:
        if self.item_list_path_uses_user:
            return f"{self._api_base(base_url)}/Users/{user_id}/Items/{external_id}"
        return f"{self._api_base(base_url)}/Items/{external_id}"

    def _sessions_params(self, *, user_id: str) -> dict[str, Any]:
        if self.name != "jellyfin" or not user_id:
            return {}
        return {
            "ControllableByUserId": user_id,
            "ActiveWithinSeconds": 60 * 60 * 24,
        }

    def _items_params(
        self,
        *,
        user_id: str,
        parent_id: str,
        start_index: int,
        limit: int,
        include_types: list[str],
        sort_by: str = "SortName",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "ParentId": parent_id,
            "Recursive": "true",
            "IncludeItemTypes": ",".join(include_types),
            "Fields": "Overview,Genres,ParentId,Path,People,Studios,ChildCount,MediaSources,UserData,ProductionYear,ParentIndexNumber,IndexNumber,SeriesName,Tagline,Status,EndDate,OfficialRating,CommunityRating,RecursiveItemCount,PremiereDate,RunTimeTicks",
            "EnableUserData": "true",
            "EnableImages": "true",
            "StartIndex": start_index,
            "Limit": limit,
            "SortBy": sort_by,
            "SortOrder": "Ascending",
        }
        if not self.item_list_path_uses_user:
            params["UserId"] = user_id
        return params

    def _item_from_row(
        self,
        raw: dict,
        *,
        library: LibraryView,
        detected_group: str,
    ) -> CatalogItem | None:
        external_id = str(raw.get("Id") or raw.get("id") or "").strip()
        item_type = str(raw.get("Type") or raw.get("type") or "").strip()
        entity_kind = _entity_kind_for_type(item_type)
        if not external_id or entity_kind == "other":
            return None
        name = str(raw.get("Name") or raw.get("name") or "Untitled").strip() or "Untitled"
        media_sources = raw.get("MediaSources") or raw.get("mediaSources") or []
        first_media = media_sources[0] if isinstance(media_sources, list) and media_sources else {}
        container = str(
            first_media.get("Container")
            or raw.get("Container")
            or ""
        ).strip().lower()
        mime_type = _mime_for_entity(entity_kind, container)
        run_time_ticks = int(raw.get("RunTimeTicks") or 0)
        duration_ms = int(run_time_ticks / 10_000) if run_time_ticks > 0 else 0
        user_data = raw.get("UserData") or raw.get("userData") or {}
        current_ticks = int(user_data.get("PlaybackPositionTicks") or 0)
        progress_pct = 0.0
        if run_time_ticks > 0 and current_ticks > 0:
            progress_pct = min(100.0, max(0.0, current_ticks / run_time_ticks * 100.0))
        is_finished = bool(
            user_data.get("Played")
            or user_data.get("PlayedPercentage") == 100
            or progress_pct >= 99.9
        )
        stream_path = ""
        if entity_kind in {"movie", "episode", "music_video"}:
            stream_path = f"/Videos/{external_id}/stream?static=true"
        image_tags = raw.get("ImageTags") or {}
        primary_tag = image_tags.get("Primary") or raw.get("PrimaryImageTag") or ""
        has_cover = bool(primary_tag or image_tags)
        cover_url = ""
        if has_cover:
            cover_url = f"/Items/{external_id}/Images/Primary"
            if primary_tag:
                cover_url = f"{cover_url}?tag={primary_tag}"
        size_bytes = int(
            first_media.get("Size")
            or raw.get("Size")
            or 0
        )
        overview = str(raw.get("Overview") or "").strip()
        genres = raw.get("Genres") or []
        if not isinstance(genres, list):
            genres = []
        series_name = str(raw.get("SeriesName") or "").strip()
        parent_external_id = str(raw.get("ParentId") or "").strip()
        grandparent_external_id = str(raw.get("SeriesId") or raw.get("SeasonId") or "").strip()
        season_number = int(raw.get("ParentIndexNumber") or 0)
        episode_number = int(raw.get("IndexNumber") or 0)
        production_year = raw.get("ProductionYear")
        unplayed_count = int(
            user_data.get("UnplayedItemCount")
            or raw.get("RecursiveUnplayedItemCount")
            or raw.get("UnplayedItemCount")
            or 0
        )
        return CatalogItem(
            external_id=external_id,
            name=name,
            kind="video",
            mime_type=mime_type,
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            progress_pct=progress_pct,
            cover_url=cover_url,
            author="",
            narrator="",
            stream_path=stream_path,
            extra={
                "library_view_id": library.external_id,
                "library_name": library.name,
                "provider_collection_type": library.collection_type,
                "provider_item_type": item_type,
                "entity_kind": entity_kind,
                "detected_group": detected_group,
                "parent_external_id": parent_external_id,
                "grandparent_external_id": grandparent_external_id,
                "year": int(production_year or 0) if production_year else 0,
                "genres": [str(g).strip() for g in genres if str(g).strip()],
                "series_name": series_name,
                "season_number": season_number,
                "episode_number": episode_number,
                "unplayed_count": unplayed_count,
                "overview": overview,
                "has_cover": has_cover,
                "current_time_s": current_ticks / _TICKS_PER_SECOND if current_ticks > 0 else 0.0,
                "is_finished": is_finished,
                "index_without_stream": entity_kind in {"series", "season"},
            },
        )

    def _user_data_url(self, base_url: str, *, user_id: str, external_id: str) -> str:
        if self.user_data_path_uses_user:
            return f"{self._api_base(base_url)}/Users/{user_id}/Items/{external_id}/UserData"
        return f"{self._api_base(base_url)}/UserItems/{external_id}/UserData"

    async def _mark_played(
        self, base_url: str, token: str, *, user_id: str, external_id: str,
    ) -> bool:
        url = self._played_url(base_url, user_id=user_id, external_id=external_id)
        try:
            resp = await self._http.post(
                url,
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    async def _mark_unplayed(
        self, base_url: str, token: str, *, user_id: str, external_id: str,
    ) -> bool:
        url = self._played_url(base_url, user_id=user_id, external_id=external_id)
        try:
            resp = await self._http.delete(
                url,
                headers=self._headers(token),
                timeout=_TIMEOUT_S,
                follow_redirects=True,
            )
            return 200 <= resp.status_code < 300
        except Exception:
            return False

    def _played_url(self, base_url: str, *, user_id: str, external_id: str) -> str:
        if self.user_data_path_uses_user:
            return f"{self._api_base(base_url)}/Users/{user_id}/PlayedItems/{external_id}"
        return f"{self._api_base(base_url)}/UserPlayedItems/{external_id}"


def _include_types_for_group(group: str) -> list[str]:
    if group == "movies":
        return ["Movie"]
    if group == "shows":
        return ["Series", "Season", "Episode"]
    if group == "music_videos":
        return ["MusicVideo"]
    return []


def _entity_kind_for_type(item_type: str) -> str:
    return {
        "Movie": "movie",
        "Series": "series",
        "Season": "season",
        "Episode": "episode",
        "MusicVideo": "music_video",
    }.get(item_type, "other")


def _mime_for_entity(entity_kind: str, container: str) -> str:
    if entity_kind == "series":
        return "video/vnd.augmentum.series"
    if entity_kind == "season":
        return "video/vnd.augmentum.season"
    if container == "mkv":
        return "video/x-matroska"
    if container in {"mp4", "m4v"}:
        return "video/mp4"
    if container == "webm":
        return "video/webm"
    if container == "avi":
        return "video/x-msvideo"
    if container == "mov":
        return "video/quicktime"
    return "video/mp4"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _ticks_to_seconds(value: Any) -> float:
    try:
        ticks = int(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if ticks <= 0:
        return 0.0
    return ticks / _TICKS_PER_SECOND


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_map(value: dict[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, raw in (value or {}).items():
        name = str(key or "").strip()
        if not name:
            continue
        out[name] = None if raw is None else str(raw)
    return out
