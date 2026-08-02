"""In-memory receiver/transport session runtime.

This is intentionally ephemeral. Receiver discovery and remote playback state
only need to survive for the current app process; provider progress remains the
durable source of truth.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransportSession:
    """Augmentum-owned remote playback session for a transport adapter."""

    user_id: str
    transport_kind: str
    receiver_id: str
    receiver_label: str
    receiver_profile: str
    provider: str
    server_id: str
    file_id: str
    external_id: str
    title: str
    thumbnail: str = ""
    session_id: str = field(default_factory=lambda: f"ts_{secrets.token_hex(8)}")
    current_time_s: float = 0.0
    duration_s: float = 0.0
    is_paused: bool = False
    is_muted: bool = False
    can_seek: bool = False
    volume_level: int | None = None
    supported_commands: list[str] = field(default_factory=list)
    receiver_state: str = ""
    receiver: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update_from_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        snap = snapshot or {}
        self.current_time_s = max(0.0, float(snap.get("current_time_s") or 0.0))
        self.duration_s = max(0.0, float(snap.get("duration_s") or 0.0))
        self.is_paused = bool(snap.get("is_paused") or False)
        self.is_muted = bool(snap.get("is_muted") or False)
        self.can_seek = bool(snap.get("can_seek") or False)
        volume = snap.get("volume_level")
        self.volume_level = int(volume) if isinstance(volume, (int, float)) else None
        commands = snap.get("supported_commands") or []
        self.supported_commands = [
            str(command).strip() for command in commands if str(command).strip()
        ]
        self.receiver_state = str(snap.get("receiver_state") or "").strip()
        if isinstance(snap.get("extra"), dict):
            merged = dict(self.extra or {})
            merged.update(snap["extra"])
            self.extra = merged
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "transport_kind": self.transport_kind,
            "receiver_id": self.receiver_id,
            "receiver_label": self.receiver_label,
            "receiver_profile": self.receiver_profile,
            "provider": self.provider,
            "server_id": self.server_id,
            "file_id": self.file_id,
            "external_id": self.external_id,
            "title": self.title,
            "thumbnail": self.thumbnail,
            "current_time_s": float(self.current_time_s or 0.0),
            "duration_s": float(self.duration_s or 0.0),
            "is_paused": bool(self.is_paused),
            "is_muted": bool(self.is_muted),
            "can_seek": bool(self.can_seek),
            "volume_level": self.volume_level,
            "supported_commands": list(self.supported_commands or []),
            "receiver_state": self.receiver_state,
            "extra": dict(self.extra or {}),
        }


class ReceiverRuntime:
    """Process-local store for discovered receivers and active transport sessions."""

    def __init__(self) -> None:
        self._receivers: dict[str, tuple[float, Any]] = {}
        self._sessions: dict[str, TransportSession] = {}

    def remember_receivers(self, receivers: list[Any], *, ttl_s: float = 300.0) -> None:
        expires_at = time.time() + max(30.0, float(ttl_s or 300.0))
        for receiver in receivers or []:
            receiver_id = str(getattr(receiver, "receiver_id", "") or "").strip()
            if not receiver_id:
                continue
            self._receivers[receiver_id] = (expires_at, receiver)

    def get_receiver(self, receiver_id: str) -> Any | None:
        item = self._receivers.get(str(receiver_id or "").strip())
        if not item:
            return None
        expires_at, receiver = item
        if expires_at <= time.time():
            self._receivers.pop(str(receiver_id or "").strip(), None)
            return None
        return receiver

    def put_session(self, session: TransportSession) -> TransportSession:
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str, *, user_id: str = "") -> TransportSession | None:
        session = self._sessions.get(str(session_id or "").strip())
        if not session:
            return None
        if user_id and session.user_id != user_id:
            return None
        return session

    def remove_session(self, session_id: str, *, user_id: str = "") -> None:
        session = self.get_session(session_id, user_id=user_id)
        if session is not None:
            self._sessions.pop(session.session_id, None)
