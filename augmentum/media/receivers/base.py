"""Receiver-side playback planning contracts.

This layer is intentionally transport-agnostic. Providers prepare a
launchable media blueprint; concrete sender/control stacks (Cast, DLNA,
AirPlay, browser remote playback) consume that blueprint later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ReceiverProfile:
    """Capabilities and operational expectations for one receiver family."""

    id: str
    kind: str
    label: str
    transport: str
    capability_tier: str
    expected_controls: list[str] = field(default_factory=list)
    supported_containers: list[str] = field(default_factory=list)
    supported_video_codecs: list[str] = field(default_factory=list)
    supported_audio_codecs: list[str] = field(default_factory=list)
    subtitle_strategy: str = "receiver_managed"
    requires_receiver_reachable_url: bool = True
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "transport": self.transport,
            "capability_tier": self.capability_tier,
            "expected_controls": list(self.expected_controls or []),
            "supported_containers": list(self.supported_containers or []),
            "supported_video_codecs": list(self.supported_video_codecs or []),
            "supported_audio_codecs": list(self.supported_audio_codecs or []),
            "subtitle_strategy": self.subtitle_strategy,
            "requires_receiver_reachable_url": bool(self.requires_receiver_reachable_url),
            "extra": self.extra,
        }


@dataclass(slots=True)
class ReceiverLaunchPlan:
    """Provider-prepared playback blueprint for a concrete receiver profile."""

    supported: bool
    receiver_profile: str
    receiver_kind: str
    control_plane: str
    capability_tier: str
    reason: str = ""
    provider: str = ""
    server_id: str = ""
    file_id: str = ""
    external_id: str = ""
    title: str = ""
    poster_url: str = ""
    content_url: str = ""
    content_type: str = ""
    subtitle_url: str = ""
    subtitle_type: str = ""
    subtitle_delivery: str = ""
    media_source_id: str = ""
    audio_stream_index: int | None = None
    subtitle_stream_index: int | None = None
    start_time_s: float = 0.0
    expected_controls: list[str] = field(default_factory=list)
    sync_back_plane: str = "provider_progress"
    requires_server_transcode: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "supported": bool(self.supported),
            "receiver_profile": self.receiver_profile,
            "receiver_kind": self.receiver_kind,
            "control_plane": self.control_plane,
            "capability_tier": self.capability_tier,
            "reason": self.reason,
            "provider": self.provider,
            "server_id": self.server_id,
            "file_id": self.file_id,
            "external_id": self.external_id,
            "title": self.title,
            "poster_url": self.poster_url,
            "content_url": self.content_url,
            "content_type": self.content_type,
            "subtitle_url": self.subtitle_url,
            "subtitle_type": self.subtitle_type,
            "subtitle_delivery": self.subtitle_delivery,
            "media_source_id": self.media_source_id,
            "audio_stream_index": self.audio_stream_index,
            "subtitle_stream_index": self.subtitle_stream_index,
            "start_time_s": float(self.start_time_s or 0.0),
            "expected_controls": list(self.expected_controls or []),
            "sync_back_plane": self.sync_back_plane,
            "requires_server_transcode": bool(self.requires_server_transcode),
            "extra": self.extra,
        }
