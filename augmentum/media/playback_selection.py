"""Shared media-server playback selection helpers.

These helpers normalize provider playback metadata into one Augmentum-owned
shape so routes, senders, and future receiver adapters can all reason about
the same selected media source / audio / subtitle choices.
"""

from __future__ import annotations

from typing import Any


def int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def stream_label(stream: dict, *, kind: str, fallback: str) -> str:
    display = str(stream.get("DisplayTitle") or "").strip()
    if display:
        return display
    parts: list[str] = []
    language = (
        str(stream.get("DisplayLanguage") or "").strip()
        or str(stream.get("Language") or "").strip()
        or str(stream.get("Title") or "").strip()
    )
    if language:
        parts.append(language)
    title = str(stream.get("Title") or "").strip()
    if title and title.lower() not in {language.lower(), fallback.lower()}:
        parts.append(title)
    codec = str(stream.get("Codec") or "").strip().upper()
    if codec:
        parts.append(codec)
    if kind == "audio":
        channels = stream.get("Channels")
        if channels not in (None, ""):
            try:
                parts.append(f"{float(channels):g}ch")
            except (TypeError, ValueError):
                pass
    if kind == "subtitle" and bool(stream.get("IsForced")):
        parts.append("Forced")
    return " • ".join(p for p in parts if p) or fallback


def media_source_label(source: dict, index: int, audio_tracks: list[dict]) -> str:
    name = str(source.get("Name") or "").strip()
    if name:
        return name
    parts: list[str] = []
    video_codec = str(source.get("VideoCodec") or "").strip().upper()
    container = str(source.get("Container") or "").strip().upper()
    height = int_or_none(source.get("Height"))
    if height is None:
        media_streams = source.get("MediaStreams") or []
        if isinstance(media_streams, list):
            for stream in media_streams:
                if not isinstance(stream, dict):
                    continue
                if str(stream.get("Type") or "").lower() == "video":
                    height = int_or_none(stream.get("Height"))
                    break
    if height:
        parts.append(f"{height}p")
    if video_codec:
        parts.append(video_codec)
    if container:
        parts.append(container)
    langs = []
    for track in audio_tracks:
        lang = str(track.get("language") or "").strip()
        if lang and lang not in langs:
            langs.append(lang)
    if langs:
        parts.append("/".join(langs[:2]))
    return " • ".join(parts) or f"Version {index + 1}"


def choose_track_index(
    tracks: list[dict],
    *,
    preferred: int | None,
    default_idx: int | None = None,
    allow_none: bool = False,
) -> int | None:
    valid = {int(track.get("index")) for track in tracks if track.get("index") is not None}
    if preferred is not None and (preferred in valid or (allow_none and preferred == -1)):
        return preferred
    if default_idx is not None and default_idx in valid:
        return default_idx
    for track in tracks:
        if track.get("is_default") and track.get("index") is not None:
            return int(track["index"])
    if allow_none:
        return -1
    if tracks:
        first = tracks[0].get("index")
        return int(first) if first is not None else None
    return -1 if allow_none else None


def normalise_emby_compat_playback(raw: dict, *, cached_meta: dict) -> dict | None:
    playback_raw = raw.get("_augmentum_playback_info")
    if not isinstance(playback_raw, dict):
        playback_raw = {}
    media_sources_raw = playback_raw.get("MediaSources")
    if not isinstance(media_sources_raw, list) or not media_sources_raw:
        media_sources_raw = raw.get("MediaSources") or []
    if not isinstance(media_sources_raw, list) or not media_sources_raw:
        return None

    preferred_source = str(cached_meta.get("preferred_media_source_id") or "").strip()
    preferred_audio = int_or_none(cached_meta.get("preferred_audio_stream_index"))
    preferred_subtitle = int_or_none(cached_meta.get("preferred_subtitle_stream_index"))

    sources: list[dict[str, Any]] = []
    for idx, source in enumerate(media_sources_raw):
        if not isinstance(source, dict):
            continue
        source_id = str(
            source.get("Id")
            or source.get("MediaSourceId")
            or f"source_{idx}"
        ).strip() or f"source_{idx}"
        streams_raw = source.get("MediaStreams") or []
        if not isinstance(streams_raw, list):
            streams_raw = []
        audio_tracks: list[dict[str, Any]] = []
        subtitle_tracks: list[dict[str, Any]] = []
        for stream in streams_raw:
            if not isinstance(stream, dict):
                continue
            stream_type = str(stream.get("Type") or "").lower()
            stream_index = int_or_none(stream.get("Index"))
            if stream_index is None:
                continue
            if stream_type == "audio":
                display_language = str(
                    stream.get("DisplayLanguage")
                    or stream.get("Language")
                    or ""
                ).strip()
                language_code = str(stream.get("Language") or "").strip().lower()
                audio_tracks.append({
                    "index": stream_index,
                    "label": stream_label(
                        stream, kind="audio", fallback=f"Audio {len(audio_tracks) + 1}",
                    ),
                    "language": display_language,
                    "language_code": language_code,
                    "codec": str(stream.get("Codec") or "").strip().lower(),
                    "channels": int_or_none(stream.get("Channels")),
                    "title": str(stream.get("Title") or "").strip(),
                    "is_default": bool(stream.get("IsDefault") or False),
                })
            elif stream_type == "subtitle":
                display_language = str(
                    stream.get("DisplayLanguage")
                    or stream.get("Language")
                    or ""
                ).strip()
                language_code = str(stream.get("Language") or "").strip().lower()
                subtitle_tracks.append({
                    "index": stream_index,
                    "label": stream_label(
                        stream,
                        kind="subtitle",
                        fallback=f"Subtitle {len(subtitle_tracks) + 1}",
                    ),
                    "language": display_language,
                    "language_code": language_code,
                    "title": str(stream.get("Title") or "").strip(),
                    "codec": str(stream.get("Codec") or "").strip().lower(),
                    "is_default": bool(stream.get("IsDefault") or False),
                    "is_forced": bool(stream.get("IsForced") or False),
                    "is_external": bool(stream.get("IsExternal") or False),
                    "is_text": bool(stream.get("IsTextSubtitleStream") or False),
                })
        subtitle_tracks.insert(0, {
            "index": -1,
            "label": "Off",
            "language": "",
            "language_code": "",
            "title": "",
            "codec": "",
            "is_default": not any(track.get("is_default") for track in subtitle_tracks),
            "is_forced": False,
            "is_external": False,
            "is_none": True,
            "is_text": True,
        })
        default_audio_idx = int_or_none(source.get("DefaultAudioStreamIndex"))
        default_subtitle_idx = int_or_none(source.get("DefaultSubtitleStreamIndex"))
        sources.append({
            "id": source_id,
            "label": media_source_label(source, idx, audio_tracks),
            "container": str(source.get("Container") or "").strip().lower(),
            "video_codec": str(source.get("VideoCodec") or "").strip().lower(),
            "is_default": idx == 0,
            "audio_tracks": audio_tracks,
            "subtitle_tracks": subtitle_tracks,
            "default_audio_stream_index": default_audio_idx,
            "default_subtitle_stream_index": default_subtitle_idx,
        })

    if not sources:
        return None

    selected_source = preferred_source if preferred_source in {
        str(source.get("id") or "") for source in sources
    } else str(sources[0].get("id") or "")
    source_obj = next(
        (source for source in sources if str(source.get("id") or "") == selected_source),
        sources[0],
    )
    selected_audio = choose_track_index(
        source_obj.get("audio_tracks") or [],
        preferred=preferred_audio,
        default_idx=int_or_none(source_obj.get("default_audio_stream_index")),
    )
    selected_subtitle = choose_track_index(
        source_obj.get("subtitle_tracks") or [],
        preferred=preferred_subtitle,
        default_idx=int_or_none(source_obj.get("default_subtitle_stream_index")),
        allow_none=True,
    )

    for source in sources:
        is_selected_source = str(source.get("id") or "") == selected_source
        source["is_selected"] = is_selected_source
        for track in source.get("audio_tracks") or []:
            track["is_selected"] = bool(
                is_selected_source and track.get("index") == selected_audio
            )
        for track in source.get("subtitle_tracks") or []:
            track["is_selected"] = bool(
                is_selected_source and track.get("index") == selected_subtitle
            )

    return {
        "selected_media_source_id": selected_source,
        "selected_audio_stream_index": selected_audio,
        "selected_subtitle_stream_index": selected_subtitle,
        "media_sources": sources,
        "has_multiple_versions": len(sources) > 1,
    }
