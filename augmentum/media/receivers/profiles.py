"""Receiver profile catalog used by the launch-planning layer."""

from __future__ import annotations

from augmentum.media.receivers.base import ReceiverProfile


_PROFILES: dict[str, ReceiverProfile] = {
    "cast_video": ReceiverProfile(
        id="cast_video",
        kind="cast",
        label="Google Cast",
        transport="cast_sender",
        capability_tier="receiver_owned",
        expected_controls=["play", "pause", "seek", "stop", "volume", "mute"],
        supported_containers=["mp4", "m4v", "webm", "mkv", "mov"],
        supported_video_codecs=["h264", "hevc", "vp8", "vp9", "av1"],
        supported_audio_codecs=["aac", "mp3", "opus", "vorbis", "ac3", "eac3"],
        subtitle_strategy="external_text_track",
        extra={
            "notes": [
                "Receiver must fetch the media URL directly.",
                "Text subtitles should be passed as external tracks when available.",
            ],
        },
    ),
    "dlna_generic_video": ReceiverProfile(
        id="dlna_generic_video",
        kind="dlna",
        label="Generic DLNA TV",
        transport="dlna_avtransport",
        capability_tier="receiver_owned",
        expected_controls=["play", "pause", "seek", "stop", "volume", "mute"],
        supported_containers=["mp4", "m4v", "mkv", "avi", "mov", "ts"],
        supported_video_codecs=["h264", "mpeg2video", "hevc"],
        supported_audio_codecs=["aac", "mp3", "ac3", "eac3", "pcm"],
        subtitle_strategy="server_decides",
        extra={
            "notes": [
                "DLNA capability reporting is inconsistent across TVs.",
                "Transport and subtitle support must be capability-gated at runtime.",
            ],
        },
    ),
}


def list_receiver_profiles() -> list[ReceiverProfile]:
    return list(_PROFILES.values())


def get_receiver_profile(profile_id: str) -> ReceiverProfile | None:
    return _PROFILES.get(str(profile_id or "").strip())
