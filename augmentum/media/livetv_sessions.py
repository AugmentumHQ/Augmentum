"""In-memory session store for Live TV playback proxying.

A play session = an opaque token the browser holds + the upstream
state we need to (a) reconstruct segment URLs into Emby/JF and
(b) tell the upstream "stop the tuner" when the user closes the
player.

Sessions are intentionally NOT persisted. They're created on
``POST /api/livetv/play`` and torn down on ``POST /api/livetv/stop``
(or via the idle-timeout sweep). If Augmentum restarts mid-session,
the user's existing tab will start failing segment fetches and
re-trigger a new play call — which is the right behavior, since
the upstream PlaySessionId is also dead at that point.

Token entropy: ``secrets.token_urlsafe(32)`` → 256 bits. Unguessable
within the session lifetime, scoped per-user by the route layer
checking ``user_id`` on every fetch.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Optional

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Sessions older than this with no segment activity get swept. Most
# live TV viewing fits in this window comfortably; a real session
# that's actively pulling segments bumps last_activity on each fetch
# so it never goes idle.
DEFAULT_IDLE_TTL_S = 600.0


@dataclass(slots=True)
class LiveTvSession:
    token: str
    user_id: str
    server_id: str
    provider: str           # 'emby' | 'jellyfin'
    base_url: str
    access_token: str
    channel_id: str
    play_session_id: str    # Emby's PlaySessionId — needed for /Sessions/Playing/Stopped
    media_source_id: str
    title: str
    device_id: str          # the DeviceId we sent to PlaybackInfo
    created_at: float
    last_activity: float


class LiveTvSessionStore:
    """Thread-safe-enough in-memory session store.

    asyncio-single-loop semantics mean no lock needed for the common
    create/get/remove path. The sweep runs synchronously inside the
    same loop, so no race with fetches either.
    """

    def __init__(self, *, idle_ttl_s: float = DEFAULT_IDLE_TTL_S) -> None:
        self._sessions: dict[str, LiveTvSession] = {}
        self._idle_ttl_s = idle_ttl_s

    def create(
        self,
        *,
        user_id: str,
        server_id: str,
        provider: str,
        base_url: str,
        access_token: str,
        channel_id: str,
        play_session_id: str,
        media_source_id: str,
        title: str,
        device_id: str,
    ) -> LiveTvSession:
        now = time.time()
        session = LiveTvSession(
            token=secrets.token_urlsafe(32),
            user_id=user_id,
            server_id=server_id,
            provider=provider,
            base_url=base_url,
            access_token=access_token,
            channel_id=channel_id,
            play_session_id=play_session_id,
            media_source_id=media_source_id,
            title=title,
            device_id=device_id,
            created_at=now,
            last_activity=now,
        )
        self._sessions[session.token] = session
        log.info(
            "livetv_session_created",
            token_prefix=session.token[:8],
            user_id=user_id,
            server_id=server_id,
            channel_id=channel_id,
        )
        return session

    def get(self, token: str, *, user_id: str) -> Optional[LiveTvSession]:
        """Return the session if it exists, belongs to ``user_id``, and
        isn't stale. Bumps last_activity as a side effect — a session
        that's actively being polled by the HLS player never expires.
        """
        session = self._sessions.get(token)
        if session is None:
            return None
        if session.user_id != user_id:
            # Cross-user access — treat as not-found rather than 403 so
            # we never confirm token existence to a wrong user.
            return None
        now = time.time()
        if now - session.last_activity > self._idle_ttl_s:
            self._sessions.pop(token, None)
            return None
        session.last_activity = now
        return session

    def remove(self, token: str, *, user_id: str) -> Optional[LiveTvSession]:
        session = self._sessions.get(token)
        if session is None or session.user_id != user_id:
            return None
        self._sessions.pop(token, None)
        log.info("livetv_session_removed", token_prefix=token[:8], user_id=user_id)
        return session

    def sweep_expired(self) -> int:
        """Drop sessions past the idle TTL. Returns the count removed.
        Safe to call from a periodic task; otherwise gets called
        opportunistically from ``get`` on each access."""
        cutoff = time.time() - self._idle_ttl_s
        dead = [t for t, s in self._sessions.items() if s.last_activity < cutoff]
        for t in dead:
            self._sessions.pop(t, None)
        if dead:
            log.info("livetv_sessions_swept", count=len(dead))
        return len(dead)

    def count(self) -> int:
        """Useful for ops/diagnostics. Doesn't sweep first — call
        ``sweep_expired`` if you need a live count."""
        return len(self._sessions)


# Process-singleton. Routes import this directly. Replace via
# monkeypatch in tests where state isolation matters.
_default_store = LiveTvSessionStore()


def get_default_store() -> LiveTvSessionStore:
    return _default_store
