"""Shared dataclass for normalized browse results across providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GameBrowseResult:
    """A single game as returned from a provider's browse() call.

    Matches the shape of ``augmentum.media.providers.base.BrowseResult`` so
    the frontend can render one card component regardless of source.
    """

    source: str                               # "js13k" | "jam"
    source_id: str                            # stable id within source
    name: str
    author: str = ""
    tagline: str = ""                         # one-liner, from source description
    thumbnail_url: str = ""
    source_url: str = ""                      # user-facing page on the source
    embed_url: str = ""                       # iframe-friendly URL if available
    play_mode: str = "embed"                  # "embed" | "local"
    genre: list[str] = field(default_factory=list)
    size_bytes: int = 0                       # 0 if unknown
    load_estimate_ms: int = 0                 # 0 if unknown
    extra: dict = field(default_factory=dict)  # provider-specific passthrough

    def to_dict(self) -> dict:
        return asdict(self)
