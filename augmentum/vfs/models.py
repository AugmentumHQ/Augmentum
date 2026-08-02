"""Data models for the file index and VFS."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileEntry:
    """A file in the unified index."""
    id: str
    user_id: str
    source: str
    source_id: str
    name: str
    mime_type: str = ""
    size_bytes: int = 0
    real_path: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    thumbnail: str | None = None
    embedding: bytes | None = None
    is_directory: bool = False
    parent_id: str | None = None
    source_metadata: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    is_favorite: bool = False
    is_trashed: bool = False
    trashed_at: str | None = None
    kind: str = ""
    score: float = 0.0
    # ISO-8601 UTC timestamp set ONLY by the progress endpoint, never
    # by catalog sync. Powers the Continue rail's "most recently
    # played" ordering. Empty for files that have never been played
    # (those fall to the rail's secondary sort by updated_at).
    last_played_at: str = ""
    # Comic-library series linkage (migration 16x scan pipeline). The
    # comic reader keys per-series prefs AND sibling-chapter resolution
    # (prev/next, auto-advance) on this — without it every chapter opens
    # as a one-chapter "series".
    series_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "source_id": self.source_id,
            "name": self.name,
            "mime_type": self.mime_type,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "description": self.description,
            "tags": self.tags,
            "thumbnail": self.thumbnail,
            "is_directory": self.is_directory,
            "source_metadata": self.source_metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_favorite": self.is_favorite,
            "is_trashed": self.is_trashed,
            "series_id": self.series_id or "",
        }

    def to_card(self, tier: str = "card") -> str:
        size = _human_size(self.size_bytes)
        date = self.created_at[:10] if self.created_at else ""
        card = f"[File: {self.name} | {self.mime_type or 'unknown'} | {size} | {date}]"
        if self.description:
            card += f"\nDescription: {self.description[:500]}"
        if self.tags:
            card += f"\nTags: {', '.join(self.tags[:10])}"
        if self.source_metadata:
            extras = []
            if "prompt" in self.source_metadata:
                extras.append(f"Prompt: {self.source_metadata['prompt'][:200]}")
            if "model" in self.source_metadata:
                extras.append(f"Model: {self.source_metadata['model']}")
            if "chunk_count" in self.source_metadata:
                extras.append(f"Chunks: {self.source_metadata['chunk_count']}")
            if extras:
                card += "\n" + " | ".join(extras)
        return card


@dataclass
class VFSNode:
    path: str
    name: str
    is_dir: bool = False
    size: int = 0
    mime_type: str = ""
    modified_at: str = ""
    source: str = ""
    source_id: str = ""
    real_path: str | None = None
    file_entry: FileEntry | None = None


@dataclass
class SearchResult:
    source: str
    item: object
    score: float = 0.0
    card: str = ""


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f}{unit}" if unit != "B" else f"{nbytes}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"
