"""In-memory capability-session runtime.

Sessions are ephemeral — they only exist while a stateful capability is
active. Provider progress (Emby watch state, Hue bulb actual color, etc.)
is the durable source of truth; sessions are the controller's handle for
in-flight playback or interaction.

User scoping is enforced on every read.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CapabilitySession:
    """One stateful capability invocation (e.g. an active video cast)."""

    user_id: str
    device_id: str
    driver: str
    capability_id: str
    title: str = ""
    thumbnail: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"sess_{secrets.token_hex(8)}")
    created_at: float = field(default_factory=time.time)
    last_event_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def update_state(self, partial: dict[str, Any]) -> None:
        if not isinstance(partial, dict):
            return
        merged = dict(self.state or {})
        merged.update(partial)
        self.state = merged
        self.last_event_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "driver": self.driver,
            "capability_id": self.capability_id,
            "title": self.title,
            "thumbnail": self.thumbnail,
            "state": dict(self.state or {}),
            "created_at": float(self.created_at or 0.0),
            "last_event_at": float(self.last_event_at or 0.0),
            "extra": dict(self.extra or {}),
        }


class SessionRuntime:
    """Process-local capability-session store. User-scoped on every read."""

    def __init__(self) -> None:
        self._sessions: dict[str, CapabilitySession] = {}

    def list(self, *, user_id: str) -> list[CapabilitySession]:
        return [s for s in self._sessions.values() if s.user_id == user_id]

    def list_for_device(self, *, user_id: str, device_id: str) -> list[CapabilitySession]:
        return [
            s for s in self._sessions.values()
            if s.user_id == user_id and s.device_id == device_id
        ]

    def get(self, session_id: str, *, user_id: str) -> CapabilitySession | None:
        session = self._sessions.get(str(session_id or "").strip())
        if session is None:
            return None
        if session.user_id != user_id:
            return None
        return session

    def put(self, session: CapabilitySession) -> CapabilitySession:
        self._sessions[session.id] = session
        return session

    def remove(self, session_id: str, *, user_id: str) -> bool:
        session = self.get(session_id, user_id=user_id)
        if session is None:
            return False
        self._sessions.pop(session.id, None)
        return True

    def remove_all_for_device(self, *, user_id: str, device_id: str) -> int:
        ids = [
            s.id for s in self._sessions.values()
            if s.user_id == user_id and s.device_id == device_id
        ]
        for sid in ids:
            self._sessions.pop(sid, None)
        return len(ids)

    def reap_idle(self, *, idle_seconds: float) -> int:
        cutoff = time.time() - max(0.0, float(idle_seconds))
        ids = [
            s.id for s in self._sessions.values()
            if s.last_event_at < cutoff
        ]
        for sid in ids:
            self._sessions.pop(sid, None)
        return len(ids)
