"""In-memory state backend for testing and development."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class MemorySession:
    id: str
    mode: str = "passthrough"
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0
    metadata: str = "{}"

    def __post_init__(self) -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


class MemoryBackend:
    """In-memory state backend — no persistence across restarts."""

    def __init__(self) -> None:
        self._sessions: dict[str, MemorySession] = {}

    async def get_session(self, session_id: str) -> dict | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "id": session.id,
            "mode": session.mode,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": session.message_count,
            "metadata": session.metadata,
        }

    async def create_session(self, session_id: str, mode: str = "passthrough") -> dict:
        session = MemorySession(id=session_id, mode=mode)
        self._sessions[session_id] = session
        return await self.get_session(session_id)  # type: ignore[return-value]

    async def update_session(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        increment_messages: bool = False,
        metadata: str | None = None,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if mode is not None:
            session.mode = mode
        if increment_messages:
            session.message_count += 1
        if metadata is not None:
            session.metadata = metadata
