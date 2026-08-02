"""Voice session fanout — non-invasive multi-target voice rendering.

Lets the existing voice WS pipeline emit to additional consumers (TV
receivers, future DLNA renderers, etc.) without rewriting 100+
``websocket.send_*`` call sites in ``proxy/voice_routes.py``.

How it works:

  The voice route wraps its inbound ``WebSocket`` in a
  ``VoiceFanoutSocket`` proxy at handler entry. The proxy quacks
  exactly like a Starlette WebSocket — same ``send_json``,
  ``send_text``, ``send_bytes`` signatures — so every existing
  call site in pipeline.py / voice_routes.py keeps working
  unmodified. On every send, the proxy ALSO publishes the payload
  to the central ``VoiceFanout`` keyed by ``voice_session_id``.

  Separate consumers (cast-vrm surface on the TV, etc.) call
  ``VoiceFanout.subscribe(session_id)`` to receive a per-subscriber
  async iterator of those same payloads. Each subscriber is
  independent; one consumer being slow does not block others.

Why fanout-and-mirror instead of a sink abstraction:

  The pipeline calls ``websocket.send_bytes`` at 20+ sites across
  multiple files. Refactoring all of them to a generic sink would
  ripple through many test fixtures and edge cases. The proxy
  approach is additive — we wrap the WS at one point, every emit
  goes to both the original socket AND the fanout. Zero risk to
  the existing voice loop, full reuse for new consumers.

Lifecycle:

  - VoiceFanout lives on ``app.state.voice_fanout`` (one per process).
  - Voice WS connect → register a new session_id in the fanout.
  - On each send: publish to all subscribers for that session.
  - Voice WS disconnect → ``close_session(session_id)`` drops every
    subscriber's queue so they wake up and exit cleanly.
  - Subscribers track their own lifecycle (e.g., a TV WS closes →
    its subscription is torn down automatically via the iterator's
    finally clause).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import WebSocket

log = get_logger(__name__)


# Each event the fanout broadcasts is one of these. Subscribers
# pattern-match on ``kind`` to route to the right local handler.
FANOUT_KIND_JSON = "json"      # JSON control messages (transcript, tts_start, etc.)
FANOUT_KIND_TEXT = "text"      # Raw text messages (uncommon, kept for parity)
FANOUT_KIND_BYTES = "bytes"    # Audio frames + other binary payloads


@dataclass
class FanoutEvent:
    kind: str
    payload: Any                 # dict for json, str for text, bytes for bytes
    seq: int = 0                 # Monotonic per-session sequence number
    ts_ns: int = 0               # Nanoseconds since session start (for sync diagnostics)


@dataclass
class _SessionState:
    session_id: str
    user_id: str
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    seq: int = 0
    start_ns: int = 0
    closed: bool = False


class VoiceFanout:
    """Per-session multi-subscriber broadcaster.

    Process-local, no persistence. Voice sessions are ephemeral by
    design — a process restart means every in-flight call is over
    anyway, so subscribers reconnect rather than the fanout being
    serialized. Same lifecycle model as the receiver registry.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = asyncio.Lock()

    # ── Session lifecycle (called by voice WS handler) ────────────

    async def open_session(self, session_id: str, user_id: str) -> None:
        """Register a new voice session. Subsequent publish() calls
        for ``session_id`` will reach any current subscribers."""
        import time
        async with self._lock:
            if session_id in self._sessions:
                # Re-opening is a contract violation upstream — log + ignore.
                log.warning("voice_fanout_double_open", session_id=session_id)
                return
            self._sessions[session_id] = _SessionState(
                session_id=session_id,
                user_id=user_id,
                start_ns=time.monotonic_ns(),
            )
        log.info("voice_fanout_session_opened", session_id=session_id, user_id=user_id)

    async def close_session(self, session_id: str) -> None:
        """Tear down a session — drops every subscriber queue so
        consumers wake up and exit cleanly."""
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is None:
                return
            state.closed = True
            subscribers = list(state.subscribers)
        # Signal end-of-stream to every subscriber.
        for q in subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                # Subscriber is so slow it can't even take a sentinel;
                # drop it on the floor — they'll time out on their own.
                pass
        log.info("voice_fanout_session_closed", session_id=session_id)

    # ── Publish (called by VoiceFanoutSocket on every send) ───────

    def publish_nowait(self, session_id: str, kind: str, payload: Any) -> None:
        """Non-blocking publish. Used from inside the voice pipeline
        which doesn't have time to await fanout deliveries."""
        import time
        state = self._sessions.get(session_id)
        if state is None or state.closed:
            return
        state.seq += 1
        event = FanoutEvent(
            kind=kind,
            payload=payload,
            seq=state.seq,
            ts_ns=time.monotonic_ns() - state.start_ns,
        )
        # Direct fanout. Drop-oldest on backpressure: a slow consumer
        # falling behind the live stream isn't worth blocking the
        # voice loop. They'll resync from the next frame.
        for q in state.subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # ── Subscribe (called by TV consumers, future renderers) ──────

    async def subscribe(
        self,
        session_id: str,
        *,
        user_id: str,
        maxsize: int = 256,
    ) -> AsyncIterator[FanoutEvent]:
        """Yield events for a session as they arrive. Caller cancels
        the consuming task to unsubscribe; we clean up on iterator
        teardown via the finally clause."""
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                # Session doesn't exist — bail immediately rather than
                # silently waiting. Callers should validate session
                # existence before subscribing.
                return
            if state.user_id != user_id:
                # Cross-user subscription attempt. Log + refuse silently.
                log.warning(
                    "voice_fanout_cross_user_subscribe_denied",
                    session_id=session_id, requester=user_id, owner=state.user_id,
                )
                return
            q: asyncio.Queue = asyncio.Queue(maxsize=max(1, maxsize))
            state.subscribers.append(q)

        try:
            while True:
                event = await q.get()
                if event is None:
                    # Sentinel from close_session.
                    break
                yield event
        finally:
            async with self._lock:
                fresh_state = self._sessions.get(session_id)
                if fresh_state is not None:
                    try:
                        fresh_state.subscribers.remove(q)
                    except ValueError:
                        pass

    # ── Read API (diagnostics) ────────────────────────────────────

    def session_count(self) -> int:
        return len(self._sessions)

    def subscriber_count(self, session_id: str) -> int:
        state = self._sessions.get(session_id)
        return len(state.subscribers) if state else 0


class VoiceFanoutSocket:
    """Proxy that wraps a Starlette WebSocket and mirrors every send
    to the central fanout. Quacks identically to a real WebSocket for
    the methods the voice pipeline actually uses.

    Non-send methods (``receive_bytes``, ``receive_text``, etc.) pass
    through unchanged via ``__getattr__`` so the pipeline can still
    receive client audio frames without modification.
    """

    def __init__(
        self,
        websocket: WebSocket,
        fanout: VoiceFanout,
        session_id: str,
    ) -> None:
        self._ws = websocket
        self._fanout = fanout
        self._session_id = session_id

    # ── Send overrides ────────────────────────────────────────────

    async def send_json(self, data: Any) -> None:
        await self._ws.send_json(data)
        self._fanout.publish_nowait(self._session_id, FANOUT_KIND_JSON, data)

    async def send_text(self, text: str) -> None:
        await self._ws.send_text(text)
        self._fanout.publish_nowait(self._session_id, FANOUT_KIND_TEXT, text)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)
        self._fanout.publish_nowait(self._session_id, FANOUT_KIND_BYTES, data)

    # ── Pass-through for everything else ──────────────────────────

    def __getattr__(self, name: str) -> Any:
        # Delegated only when the attribute isn't on the proxy itself.
        # __getattr__ is only called for missing attrs, so methods we've
        # defined above always win.
        return getattr(self._ws, name)


def wrap_websocket(
    websocket: WebSocket,
    fanout: VoiceFanout,
    session_id: str,
) -> VoiceFanoutSocket:
    """Convenience constructor used at the top of the voice WS
    handler. Returns a proxy that the pipeline treats identically
    to the underlying websocket."""
    return VoiceFanoutSocket(websocket, fanout, session_id)
