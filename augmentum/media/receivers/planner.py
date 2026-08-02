"""Provider-to-receiver playback blueprint planning.

This is the seam between "where the media lives" (Emby/Jellyfin) and
"how the playback is transported" (Cast, DLNA, etc.). The planner does
not start playback itself; it only returns a receiver-friendly launch
plan that a sender/control stack can consume.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from augmentum.media.playback_selection import (
    int_or_none,
    normalise_emby_compat_playback,
)
from augmentum.media.receivers.base import ReceiverLaunchPlan
from augmentum.media.receivers.profiles import get_receiver_profile
from augmentum.media.store import MediaServer


def _append_query_params(url: str, params: dict[str, str]) -> str:
    if not params:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query),
        parts.fragment,
    ))


def _is_loopback_base_url(url: str) -> bool:
    host = str(urlsplit(url).hostname or "").strip().lower()
    return (
        not host
        or host == "localhost"
        or host == "127.0.0.1"
        or host == "::1"
        or host == "0.0.0.0"
        or host.endswith(".localhost")
    )


def _preferred_stream_choice(meta: dict, *, include_subtitle: bool = True) -> dict[str, str]:
    choice: dict[str, str] = {}
    media_source_id = str(meta.get("preferred_media_source_id") or "").strip()
    if media_source_id:
        choice["MediaSourceId"] = media_source_id
    audio_idx = int_or_none(meta.get("preferred_audio_stream_index"))
    if audio_idx is not None:
        choice["AudioStreamIndex"] = str(audio_idx)
    subtitle_idx = int_or_none(meta.get("preferred_subtitle_stream_index"))
    if include_subtitle and subtitle_idx is not None and subtitle_idx >= 0:
        choice["SubtitleStreamIndex"] = str(subtitle_idx)
    return choice


def _content_type_for_container(container: str) -> str:
    value = str(container or "").strip().lower()
    if value == "mkv":
        return "video/x-matroska"
    if value in {"mp4", "m4v"}:
        return "video/mp4"
    if value == "webm":
        return "video/webm"
    if value == "mov":
        return "video/quicktime"
    if value == "avi":
        return "video/x-msvideo"
    if value in {"ts", "mpegts"}:
        return "video/mp2t"
    return "video/mp4"


def _selected_playback_source(playback: dict | None) -> dict | None:
    if not isinstance(playback, dict):
        return None
    sources = playback.get("media_sources") or []
    if not isinstance(sources, list) or not sources:
        return None
    selected_id = str(playback.get("selected_media_source_id") or "").strip()
    for source in sources:
        if str(source.get("id") or "").strip() == selected_id:
            return source
    return sources[0] if isinstance(sources[0], dict) else None


def _selected_subtitle_track(playback: dict | None) -> dict | None:
    source = _selected_playback_source(playback)
    if not isinstance(source, dict):
        return None
    selected_idx = int_or_none(playback.get("selected_subtitle_stream_index") if isinstance(playback, dict) else None)
    for track in source.get("subtitle_tracks") or []:
        if not isinstance(track, dict):
            continue
        if int_or_none(track.get("index")) == selected_idx:
            return track
    return None


async def _ensure_playback(client, server: MediaServer, *, external_id: str, cached_meta: dict) -> dict | None:
    playback = cached_meta.get("playback")
    if isinstance(playback, dict) and isinstance(playback.get("media_sources"), list):
        return playback
    if not callable(getattr(client, "fetch_item_details", None)):
        return None
    raw = await client.fetch_item_details(
        server.base_url,
        server.access_token,
        external_id=external_id,
    )
    if not isinstance(raw, dict):
        return None
    return normalise_emby_compat_playback(raw, cached_meta=cached_meta)


async def build_receiver_launch_plan(
    *,
    server: MediaServer,
    client,
    file_id: str,
    entry_name: str,
    cached_meta: dict,
    receiver_profile_id: str,
) -> ReceiverLaunchPlan:
    profile = get_receiver_profile(receiver_profile_id)
    if profile is None:
        return ReceiverLaunchPlan(
            supported=False,
            receiver_profile=receiver_profile_id,
            receiver_kind="",
            control_plane="",
            capability_tier="",
            reason="Unknown receiver profile",
            provider=server.provider,
            server_id=server.id,
            file_id=file_id,
        )

    entity_kind = str(cached_meta.get("entity_kind") or "").strip()
    external_id = str(cached_meta.get("external_id") or "").strip()
    stream_path = str(cached_meta.get("stream_path") or "").strip()
    if server.provider not in {"emby", "jellyfin"}:
        return ReceiverLaunchPlan(
            supported=False,
            receiver_profile=profile.id,
            receiver_kind=profile.kind,
            control_plane=profile.transport,
            capability_tier=profile.capability_tier,
            reason="Receiver launch planning currently supports Emby and Jellyfin video items",
            provider=server.provider,
            server_id=server.id,
            file_id=file_id,
            external_id=external_id,
            title=entry_name,
            expected_controls=profile.expected_controls,
        )
    if entity_kind not in {"movie", "episode", "music_video"}:
        return ReceiverLaunchPlan(
            supported=False,
            receiver_profile=profile.id,
            receiver_kind=profile.kind,
            control_plane=profile.transport,
            capability_tier=profile.capability_tier,
            reason="Item is not directly playable on a receiver target",
            provider=server.provider,
            server_id=server.id,
            file_id=file_id,
            external_id=external_id,
            title=entry_name,
            expected_controls=profile.expected_controls,
        )
    if not external_id or not stream_path:
        return ReceiverLaunchPlan(
            supported=False,
            receiver_profile=profile.id,
            receiver_kind=profile.kind,
            control_plane=profile.transport,
            capability_tier=profile.capability_tier,
            reason="Media stream path is unavailable for this item",
            provider=server.provider,
            server_id=server.id,
            file_id=file_id,
            external_id=external_id,
            title=entry_name,
            expected_controls=profile.expected_controls,
        )
    if profile.requires_receiver_reachable_url and _is_loopback_base_url(server.base_url):
        return ReceiverLaunchPlan(
            supported=False,
            receiver_profile=profile.id,
            receiver_kind=profile.kind,
            control_plane=profile.transport,
            capability_tier=profile.capability_tier,
            reason="Receiver cannot reach a localhost-only media server URL",
            provider=server.provider,
            server_id=server.id,
            file_id=file_id,
            external_id=external_id,
            title=entry_name,
            expected_controls=profile.expected_controls,
        )

    playback = await _ensure_playback(
        client,
        server,
        external_id=external_id,
        cached_meta=cached_meta,
    )
    selected_source = _selected_playback_source(playback)
    selected_subtitle_track = _selected_subtitle_track(playback)
    selected_subtitle_idx = int_or_none(
        playback.get("selected_subtitle_stream_index") if isinstance(playback, dict) else cached_meta.get("preferred_subtitle_stream_index"),
    )
    selected_audio_idx = int_or_none(
        playback.get("selected_audio_stream_index") if isinstance(playback, dict) else cached_meta.get("preferred_audio_stream_index"),
    )
    selected_source_id = str(
        playback.get("selected_media_source_id") if isinstance(playback, dict) else cached_meta.get("preferred_media_source_id") or ""
    ).strip()

    use_external_subtitle = bool(
        selected_subtitle_track
        and int_or_none(selected_subtitle_track.get("index")) is not None
        and int_or_none(selected_subtitle_track.get("index")) >= 0
        and bool(selected_subtitle_track.get("is_text"))
        and selected_source_id
        and callable(getattr(client, "build_subtitle_url", None))
    )
    stream_query = _preferred_stream_choice(
        cached_meta,
        include_subtitle=not use_external_subtitle,
    )
    if selected_source_id:
        stream_query["MediaSourceId"] = selected_source_id
    if selected_audio_idx is not None:
        stream_query["AudioStreamIndex"] = str(selected_audio_idx)

    content_url = client.build_stream_url(
        server.base_url,
        stream_path,
        server.access_token,
    )
    content_url = _append_query_params(content_url, stream_query)

    poster_url = ""
    if bool(cached_meta.get("has_cover") or False):
        poster_url = client.build_cover_url(
            server.base_url,
            external_id,
            server.access_token,
        )

    subtitle_url = ""
    subtitle_type = ""
    subtitle_delivery = "server_decides"
    if use_external_subtitle:
        subtitle_url = client.build_subtitle_url(
            server.base_url,
            external_id=external_id,
            media_source_id=selected_source_id,
            subtitle_stream_index=int(selected_subtitle_idx or 0),
            token=server.access_token,
        )
        subtitle_type = "text/vtt"
        subtitle_delivery = "external_text_track"
    elif selected_subtitle_idx is not None and selected_subtitle_idx >= 0:
        subtitle_delivery = "embedded_or_burned"

    title = str(
        cached_meta.get("selected_episode_title")
        or cached_meta.get("playback_title")
        or entry_name
        or "Video"
    ).strip() or "Video"
    container = str(selected_source.get("container") or "").strip().lower() if isinstance(selected_source, dict) else ""
    video_codec = str(selected_source.get("video_codec") or "").strip().lower() if isinstance(selected_source, dict) else ""
    requires_server_transcode = False
    if profile.supported_containers and container and container not in set(profile.supported_containers):
        requires_server_transcode = True
    if profile.supported_video_codecs and video_codec and video_codec not in set(profile.supported_video_codecs):
        requires_server_transcode = True

    return ReceiverLaunchPlan(
        supported=True,
        receiver_profile=profile.id,
        receiver_kind=profile.kind,
        control_plane=profile.transport,
        capability_tier=profile.capability_tier,
        provider=server.provider,
        server_id=server.id,
        file_id=file_id,
        external_id=external_id,
        title=title,
        poster_url=poster_url,
        content_url=content_url,
        content_type=_content_type_for_container(container),
        subtitle_url=subtitle_url,
        subtitle_type=subtitle_type,
        subtitle_delivery=subtitle_delivery,
        media_source_id=selected_source_id,
        audio_stream_index=selected_audio_idx,
        subtitle_stream_index=selected_subtitle_idx,
        start_time_s=float(cached_meta.get("current_time_s") or 0.0),
        expected_controls=profile.expected_controls,
        requires_server_transcode=requires_server_transcode,
        extra={
            "receiver_label": profile.label,
            "container": container,
            "video_codec": video_codec,
            "subtitle_is_text": bool(selected_subtitle_track.get("is_text")) if isinstance(selected_subtitle_track, dict) else False,
            "selected_source_label": str(selected_source.get("label") or "").strip() if isinstance(selected_source, dict) else "",
            "selection_origin": "playback_info" if playback else "cached_metadata",
        },
    )
