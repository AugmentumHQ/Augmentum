from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DreamEntryType(str, Enum):
    REFLECTION = "reflection"
    VOICE_NOTE = "voice_note"
    ACTIVE_THREAD = "active_thread"
    IMPRESSION = "impression"


@dataclass
class DreamEntry:
    id: str
    persona_id: str
    content: str
    entry_type: DreamEntryType
    source_memories: list[str]
    source_sessions: list[str]
    context_window: dict
    embedding: bytes | None
    weight: float = 1.0
    pinned: bool = False
    dream_cycle_id: str = ""
    created_at: str = ""
    expires_at: str | None = None


@dataclass
class DreamPortrait:
    id: str
    persona_id: str
    voice_notes: str
    active_threads: str
    impressions: str
    source_entries: list[str]
    is_current: bool = True
    checkpoint_name: str | None = None
    created_at: str = ""


@dataclass
class DreamCycle:
    id: str
    persona_id: str
    trigger_reason: str
    memories_count: int = 0
    entries_count: int = 0
    model_used: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0
    status: str = "pending"
    error: str | None = None
    started_at: str = ""
    completed_at: str | None = None


@dataclass
class ContextSegment:
    """A clustered group of memories with their surrounding conversation context."""

    memories: list[dict]
    messages: list[dict]
    session_id: str
    timestamp_range: tuple[str, str]
    relative_age: str
