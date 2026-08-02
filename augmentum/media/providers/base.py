"""Provider protocol — what every media-server client must implement.

The store owns credentials and persistence; providers own the
HTTP-speaks-to-their-specific-server logic. Sync/stream/progress land in
the route layer, which composes the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Default ports the detector probes. Providers whose defaults live in
# this table get silent first-open detection; others require manual entry.
DEFAULT_PORTS: dict[str, int] = {
    "audiobookshelf": 13378,
    "emby": 8096,
    "jellyfin": 8096,
    "komga": 25600,
    "kavita": 5000,
    "suwayomi": 4567,
}


@dataclass(slots=True)
class CatalogItem:
    """One library entry surfaced to Augmentum's file index.

    ``external_id`` + ``server_id`` is the uniqueness key — re-syncing
    upserts on (source='audiobookshelf', source_id=external_id) by way of
    file_index's ON CONFLICT handling, so progress changes don't create
    duplicates.
    """

    external_id: str
    name: str
    kind: str  # 'audio' | 'video' | 'document'
    mime_type: str
    size_bytes: int = 0
    duration_ms: int = 0
    progress_pct: float = 0.0
    cover_url: str = ""
    author: str = ""
    narrator: str = ""
    stream_path: str = ""          # server-side path we proxy, no token in it
    extra: dict = field(default_factory=dict)


@dataclass(slots=True)
class ProviderInfo:
    """What a successful detection/test returns to the UI."""

    provider: str
    base_url: str
    server_name: str = ""
    version: str = ""
    is_initialized: bool = True    # ABS-specific; True for other providers


@dataclass(slots=True)
class BrowseResult:
    """Ephemeral search/browse hit from a remote catalog.

    Used by built-in free providers (LibriVox, and future Gutenberg) where
    the catalog is too large to mirror into file_index. Results are live,
    not persisted; the user promotes one into their library via /api/media/pin,
    which then materialises a CatalogItem row through the normal sync path.
    """

    external_id: str
    name: str
    author: str = ""
    narrator: str = ""
    duration_ms: int = 0
    cover_url: str = ""
    description: str = ""
    license: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "external_id": self.external_id,
            "name":        self.name,
            "author":      self.author,
            "narrator":    self.narrator,
            "duration_ms": self.duration_ms,
            "cover_url":   self.cover_url,
            "description": self.description,
            "license":     self.license,
            "extra":       self.extra,
        }


@dataclass(slots=True)
class LibraryView:
    """One provider-native library/view discovered on a remote server."""

    external_id: str
    name: str
    view_type: str = ""
    collection_type: str = ""
    sample_type_counts: dict[str, int] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "external_id": self.external_id,
            "name": self.name,
            "view_type": self.view_type,
            "collection_type": self.collection_type,
            "sample_type_counts": self.sample_type_counts,
            "extra": self.extra,
        }


@dataclass(slots=True)
class RemoteSession:
    """One provider-native playback session/client that can accept remote control."""

    session_id: str
    name: str
    client: str = ""
    device_name: str = ""
    user_name: str = ""
    supports_media_control: bool = False
    supports_remote_control: bool = False
    playable_media_types: list[str] = field(default_factory=list)
    supported_commands: list[str] = field(default_factory=list)
    now_playing_title: str = ""
    now_playing_item_id: str = ""
    current_time_s: float = 0.0
    duration_s: float = 0.0
    is_paused: bool = False
    is_muted: bool = False
    can_seek: bool = False
    volume_level: int | None = None
    audio_stream_index: int | None = None
    subtitle_stream_index: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "client": self.client,
            "device_name": self.device_name,
            "user_name": self.user_name,
            "supports_media_control": self.supports_media_control,
            "supports_remote_control": self.supports_remote_control,
            "playable_media_types": list(self.playable_media_types or []),
            "supported_commands": list(self.supported_commands or []),
            "now_playing_title": self.now_playing_title,
            "now_playing_item_id": self.now_playing_item_id,
            "current_time_s": float(self.current_time_s or 0.0),
            "duration_s": float(self.duration_s or 0.0),
            "is_paused": bool(self.is_paused),
            "is_muted": bool(self.is_muted),
            "can_seek": bool(self.can_seek),
            "volume_level": self.volume_level,
            "audio_stream_index": self.audio_stream_index,
            "subtitle_stream_index": self.subtitle_stream_index,
            "extra": self.extra,
        }


@runtime_checkable
class MediaProvider(Protocol):
    """Protocol for per-server clients.

    Implementations are stateless — the store holds credentials, the
    provider is handed what it needs per call. This keeps the provider
    easy to mock and avoids mutable state aliasing across users.
    """

    name: str  # matches ``user_media_servers.provider`` column and file_index.source

    async def ping(self, base_url: str) -> ProviderInfo | None:
        """Unauthenticated health + fingerprint check."""
        ...

    async def login(self, base_url: str, username: str, password: str) -> str:
        """Exchange user/pass for an access token. Raises on failure."""
        ...

    async def verify_token(self, base_url: str, token: str) -> bool:
        """Return True if the token still authenticates against this server."""
        ...

    async def fetch_catalog(self, base_url: str, token: str) -> list[CatalogItem]:
        """Pull library items. Deep per-chapter data lives in extra{}."""
        ...

    def build_stream_url(self, base_url: str, stream_path: str, token: str) -> str:
        """Assemble the upstream URL we GET with Range when the client plays."""
        ...

    def build_cover_url(self, base_url: str, external_id: str, token: str) -> str:
        """Assemble the upstream URL for the item's cover art."""
        ...

    async def fetch_progress(self, base_url: str, token: str) -> dict[str, dict]:
        """Return { external_id: {current_time_s, duration_s, progress, is_finished} }."""
        ...

    async def fetch_item_details(
        self,
        base_url: str,
        token: str,
        *,
        external_id: str,
        episode_id: str = "",
    ) -> dict | None:
        """Pull the rich per-item record (chapters + full arrays). None on miss."""
        ...

    async def push_progress(
        self, base_url: str, token: str, *,
        external_id: str, episode_id: str = "",
        current_time_s: float, duration_s: float,
        is_finished: bool = False,
    ) -> bool:
        """Write user position back upstream. Returns True on success."""
        ...


def provider_supports_browse(provider: object) -> bool:
    """Optional-method feature check — does this provider expose browse()?

    Kept separate from the Protocol so ABS and existing concrete providers
    don't need to define a no-op stub. The browse route uses this to gate
    live-search endpoints to providers that actually implement them.
    """
    return callable(getattr(provider, "browse", None))


def provider_supports_library_discovery(provider: object) -> bool:
    """Optional-method feature check for remote library/view discovery."""
    return callable(getattr(provider, "discover_libraries", None))


def provider_supports_remote_control(provider: object) -> bool:
    """Optional-method feature check for provider-native remote session control."""
    return callable(getattr(provider, "list_remote_sessions", None)) and callable(
        getattr(provider, "remote_play", None),
    )


def provider_supports_remote_general_control(provider: object) -> bool:
    """Optional-method feature check for provider general-command support."""
    return provider_supports_remote_control(provider) and callable(
        getattr(provider, "remote_general_command", None),
    )
