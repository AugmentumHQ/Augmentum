"""In-RAM registry of connected TV / receiver WebSocket sessions.

Mirrors the connectivity primitives of cast_tokens.py + the device
substrate's session runtime: a single registry holds every open
connection, user-scoped on every read.

Lifecycle: a receiver opens a WebSocket → ``attach()`` returns a
ConnectedReceiver record (with a fresh registration_id). The receiver
sends its ``ready`` event; the registry records the device fingerprint.
Server-side callers ``send()`` cmds; the receiver responds with
events. On WS close the route handler calls ``detach()`` to clean up.

No persistence — receivers re-register on reconnect with a new id.
That's deliberate: a TV that loses power should not silently come back
into a half-active state; the cast orchestration layer above should
notice and re-dispatch.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.cast.receiver_protocol import (
    ReceiverCmd,
    ReceiverEvent,
    serialise_cmd,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import WebSocket

    from augmentum.cast.cast_events import CastEventStore
    from augmentum.cast.trusted_receivers import TrustedReceiverStore

log = get_logger(__name__)


@dataclass(slots=True)
class ConnectedReceiver:
    registration_id: str
    user_id: str
    ws: WebSocket
    info: dict[str, Any] = field(default_factory=dict)
    connected_at: float = 0.0
    last_event_at: float = 0.0
    # Best-effort label populated from ready info (e.g. "Onn 4K Box");
    # falls back to registration_id when receiver didn't supply one.
    label: str = ""
    # Trusted-receiver binding — populated when the receiver's ready
    # event carried a stable device_id and the row was upserted.
    # Empty for browser-tab receivers (no device_id, ephemeral by
    # design).
    trusted_id: str = ""
    # Latest playback state per mounted surface_id, merged from
    # surface_state events emitted by the surface (native media
    # shortcuts and html.generic iframes that opt in via the
    # ``augmentum.surface_state`` postMessage). Read by controller-side
    # UIs (cast-shelf, cast-control) to render scrubbers / pause
    # buttons that match reality. Cleared when the surface closes.
    surface_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    # surface_id → stream_session_id for browser-stream renders mounted
    # on this receiver. Populated by /api/cast/render-stream/start so
    # detach() and surface_closed events can reap the orphan container
    # via the registry's stream_session_stopper callable. Without this,
    # losing a TV WS leaves the server-side Chrome+Selkies container
    # running indefinitely, burning ~4 cores on software rasterization.
    stream_sessions: dict[str, str] = field(default_factory=dict)

    def to_dict(self, *, include_user_id: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "registration_id": self.registration_id,
            "label": self.label or self.registration_id,
            "info": dict(self.info),
            "connected_at": self.connected_at,
            "last_event_at": self.last_event_at,
            "trusted_id": self.trusted_id,
        }
        if include_user_id:
            out["user_id"] = self.user_id
        return out


class ReceiverRegistry:
    """Manages connected receivers + their event subscribers.

    Thread-/task-safety: writes happen from WS handlers (one event loop
    per process); the registry uses no locks. Subscribers receive
    events via asyncio queues which serialise the cross-task fan-out.
    """

    def __init__(
        self,
        *,
        trusted_store: TrustedReceiverStore | None = None,
        event_store: CastEventStore | None = None,
        stream_session_stopper: Callable[[str, str], Awaitable[Any]] | None = None,
    ) -> None:
        self._receivers: dict[str, ConnectedReceiver] = {}
        # (user_id_filter, registration_id_filter, queue). Filters of
        # "" match anything; concrete strings match only the named entity.
        self._subscribers: list[tuple[str, str, asyncio.Queue]] = []
        self._trusted_store = trusted_store
        self._event_store = event_store
        # Callable that stops a browser-stream render session by
        # (stream_session_id, user_id). Wired in server.py from
        # GameStreamRuntime.stop_session. Optional: tests + deployments
        # without the game_stream module pass None and the registry
        # silently skips the stop step. Returns Any because we treat
        # the result as opaque — registry just needs the side-effect.
        self._stream_session_stopper = stream_session_stopper
        # Per-(user, cmd) last-broadcast clock used by broadcast_debounced.
        # In-process only; survives until restart, which is enough for the
        # cadence-shaping it does — every host process gets its own first
        # broadcast through. Keyed (user_id, cmd_name).
        self._last_broadcast_at: dict[tuple[str, str], float] = {}

    # ── Store accessors (used by routes for higher-level features) ─

    @property
    def trusted_store(self) -> TrustedReceiverStore | None:
        return self._trusted_store

    @property
    def event_store(self) -> CastEventStore | None:
        return self._event_store

    # ── Attach / detach ──────────────────────────────────────────

    def attach(self, *, ws: WebSocket, user_id: str) -> ConnectedReceiver:
        """Register a freshly-connected WebSocket. The caller has
        already accepted the WS handshake."""
        now = time.time()
        registration_id = f"rcv_{secrets.token_hex(10)}"
        receiver = ConnectedReceiver(
            registration_id=registration_id,
            user_id=user_id,
            ws=ws,
            connected_at=now,
        )
        self._receivers[registration_id] = receiver
        log.info(
            "cast_receiver_attached",
            registration_id=registration_id, user_id=user_id,
        )
        return receiver

    def detach(self, registration_id: str) -> bool:
        receiver = self._receivers.pop(registration_id, None)
        if receiver is None:
            return False
        log.info(
            "cast_receiver_detached",
            registration_id=registration_id,
            user_id=receiver.user_id,
            duration_s=round(time.time() - receiver.connected_at, 2),
        )
        # Fire-and-forget close any open cast events tied to this
        # runtime registration. Failures here must not block detach.
        if self._event_store is not None:
            try:
                asyncio.create_task(self._event_store.mark_end_by_registration(
                    user_id=receiver.user_id,
                    registration_id=registration_id,
                    reason="disconnected",
                ))
            except Exception as exc:
                log.warning(
                    "cast_event_mark_end_failed",
                    registration_id=registration_id, error=str(exc),
                )
        # Reap any browser-stream render containers mounted on this
        # receiver. Without this, losing the TV WS leaves the server-
        # side Chrome+Selkies container running indefinitely (the
        # game_stream idle reaper only catches sessions in IDLE status,
        # and render-stream sessions never transition there).
        self._reap_stream_sessions(receiver, reason="receiver_disconnected")
        return True

    # ── Render-stream linkage ─────────────────────────────────────

    def note_stream_session(
        self, registration_id: str, surface_id: str, stream_session_id: str,
    ) -> None:
        """Record that ``surface_id`` on this receiver is backed by a
        running browser-stream session ``stream_session_id``. Called by
        /api/cast/render-stream/start after the runtime hands back the
        new session id. Allows detach() + surface_closed teardown.
        """
        receiver = self._receivers.get(registration_id)
        if receiver is None or not surface_id or not stream_session_id:
            return
        receiver.stream_sessions[surface_id] = stream_session_id

    def forget_stream_session(
        self, registration_id: str, surface_id: str,
    ) -> str:
        """Drop the surface_id→stream_session_id mapping and return the
        stream_session_id (or '' when absent). Caller stops the session.
        """
        receiver = self._receivers.get(registration_id)
        if receiver is None or not surface_id:
            return ""
        return receiver.stream_sessions.pop(surface_id, "")

    def _reap_stream_sessions(
        self, receiver: ConnectedReceiver, *, reason: str,
    ) -> None:
        """Fire-and-forget stop every browser-stream session bound to
        the receiver. No-op when no stopper was wired in (tests / non-
        game-stream deployments) so the registry stays usable without
        the runtime."""
        if self._stream_session_stopper is None:
            return
        sessions = list(receiver.stream_sessions.values())
        receiver.stream_sessions.clear()
        if not sessions:
            return
        for stream_session_id in sessions:
            try:
                asyncio.create_task(self._stop_one_stream_session(
                    stream_session_id, receiver.user_id, reason,
                ))
            except Exception as exc:
                log.warning(
                    "cast_stream_session_stop_schedule_failed",
                    stream_session_id=stream_session_id,
                    user_id=receiver.user_id, error=str(exc),
                )

    async def _stop_one_stream_session(
        self, stream_session_id: str, user_id: str, reason: str,
    ) -> None:
        try:
            await self._stream_session_stopper(stream_session_id, user_id)
            log.info(
                "cast_stream_session_stopped",
                stream_session_id=stream_session_id,
                user_id=user_id, reason=reason,
            )
        except Exception as exc:
            log.warning(
                "cast_stream_session_stop_failed",
                stream_session_id=stream_session_id,
                user_id=user_id, reason=reason, error=str(exc),
            )

    # ── Read API ──────────────────────────────────────────────────

    def get(self, registration_id: str) -> ConnectedReceiver | None:
        return self._receivers.get(registration_id)

    def list_for_user(self, user_id: str) -> list[ConnectedReceiver]:
        return [r for r in self._receivers.values() if r.user_id == user_id]

    def count(self) -> int:
        return len(self._receivers)

    # ── Capability negotiation (Phase C) ──────────────────────────

    def receiver_capabilities(self, registration_id: str) -> dict[str, Any]:
        """Return the receiver's advertised surface_capabilities dict.

        Empty dict when the receiver hasn't sent ``ready`` yet or
        doesn't carry capability info. Caller compares against
        surface_kind strings to decide rendering strategy.
        """
        receiver = self._receivers.get(registration_id)
        if receiver is None:
            return {}
        caps = (receiver.info or {}).get("surface_capabilities")
        return dict(caps) if isinstance(caps, dict) else {}

    def receiver_supports(self, registration_id: str, surface_kind: str) -> bool:
        """Does this receiver advertise native support for ``surface_kind``?

        Soft signal — unknown kinds fall back to iframe rendering on
        every receiver, so ``False`` does NOT mean "can't render at
        all." It means "no fast path; consider server-side render or
        a different node before shipping a heavy URL the TV can't
        handle natively."
        """
        if not surface_kind:
            return False
        return surface_kind in self.receiver_capabilities(registration_id)

    def find_receivers_with_capability(
        self,
        user_id: str,
        surface_kind: str,
    ) -> list[ConnectedReceiver]:
        """List user-owned receivers that advertise the kind natively.

        Used by the smart dispatcher (future) to prefer native-render
        receivers when multiple are connected. Empty when nothing
        matches — caller decides whether to fall back to server-render
        or skip.
        """
        return [
            r for r in self._receivers.values()
            if r.user_id == user_id
            and surface_kind in self.receiver_capabilities(r.registration_id)
        ]

    # ── Send ──────────────────────────────────────────────────────

    async def send(self, registration_id: str, cmd: ReceiverCmd) -> bool:
        """Send a cmd to a single receiver. Returns False if the
        receiver isn't connected or the send fails (which auto-detaches
        the bad connection)."""
        receiver = self._receivers.get(registration_id)
        if receiver is None:
            return False
        return await self._send_to(receiver, cmd)

    async def broadcast(self, user_id: str, cmd: ReceiverCmd) -> int:
        """Send a cmd to every receiver owned by ``user_id``. Returns
        the count of successful sends."""
        sent = 0
        for receiver in list(self._receivers.values()):
            if receiver.user_id != user_id:
                continue
            if await self._send_to(receiver, cmd):
                sent += 1
        return sent

    async def broadcast_debounced(
        self,
        user_id: str,
        cmd: ReceiverCmd,
        *,
        min_interval_s: float,
    ) -> int:
        """``broadcast`` with a per-(user, cmd) rate gate.

        Use for cmds emitted from high-frequency hot paths where the
        receiver only needs an occasional nudge — e.g. ``library_invalidate``
        from /api/media/progress, which fires every few seconds during
        playback but only needs to reach the TV once per ~30 s to keep
        the Continue rail fresh. The first call after the gate clears
        passes through; subsequent calls inside the window are dropped.

        Returns the count of successful sends, or 0 when the call was
        skipped by the gate.
        """
        key = (user_id, cmd.cmd)
        now = time.time()
        last = self._last_broadcast_at.get(key, 0.0)
        if now - last < min_interval_s:
            return 0
        self._last_broadcast_at[key] = now
        return await self.broadcast(user_id, cmd)

    async def _send_to(self, receiver: ConnectedReceiver, cmd: ReceiverCmd) -> bool:
        try:
            await receiver.ws.send_json(serialise_cmd(cmd))
            return True
        except Exception as exc:
            log.warning(
                "cast_receiver_send_failed",
                registration_id=receiver.registration_id,
                cmd=cmd.cmd, error=str(exc)[:160],
            )
            # The WS is broken — drop the receiver so callers don't
            # keep retrying against a dead connection.
            self._receivers.pop(receiver.registration_id, None)
            return False

    # ── Events ────────────────────────────────────────────────────

    def record_event(self, registration_id: str, event: ReceiverEvent) -> None:
        """Apply an inbound event to the receiver's state + fan out
        to subscribers. Called by the WS route on every received msg."""
        receiver = self._receivers.get(registration_id)
        if receiver is None:
            return
        receiver.last_event_at = time.time()

        # Merge surface_state deltas into the per-surface bag so
        # controller UIs can read live playback state without polling
        # the receiver directly. Partial patches are MERGED (not
        # replaced) — a position_s update mid-stream shouldn't clobber
        # a previously-reported duration_s. Surface_closed events drop
        # the entry so a stale state can't leak across remounts.
        if event.event == "surface_state":
            data = event.data or {}
            surface_id = str(data.get("surface_id") or "").strip()
            state = data.get("state")
            if surface_id and isinstance(state, dict):
                bag = receiver.surface_states.get(surface_id) or {}
                bag.update(state)
                receiver.surface_states[surface_id] = bag
        elif event.event == "surface_closed":
            data = event.data or {}
            surface_id = str(data.get("surface_id") or "").strip()
            if surface_id:
                receiver.surface_states.pop(surface_id, None)
                # If this surface was a browser-stream render, stop the
                # backing container. Same reaper as detach() — receivers
                # that close a stream cleanly (vs. dropping the WS)
                # arrive here first, and we must not let the container
                # outlive the surface.
                stream_session_id = receiver.stream_sessions.pop(surface_id, "")
                if stream_session_id and self._stream_session_stopper is not None:
                    try:
                        asyncio.create_task(self._stop_one_stream_session(
                            stream_session_id, receiver.user_id, "surface_closed",
                        ))
                    except Exception as exc:
                        log.warning(
                            "cast_stream_session_stop_schedule_failed",
                            stream_session_id=stream_session_id,
                            user_id=receiver.user_id, error=str(exc),
                        )
                # Close the matching cast_events row so the controller's
                # "currently_showing" list doesn't accumulate orphaned
                # active entries — every cast (image/video/audio) records
                # a start row but only explicit /api/cast/close closed it
                # before this. The receiver-side reason maps to the
                # cast_events vocabulary; fall through to user_stop for
                # anything unrecognised so the row still ends.
                if self._event_store is not None:
                    from augmentum.cast.cast_events import (
                        END_REASON_ENDED,
                        END_REASON_REPLACED,
                        END_REASON_USER_STOP,
                    )
                    reason = str(data.get("reason") or "").strip()
                    end_reason = {
                        "ended": END_REASON_ENDED,
                        "remote_stop": END_REASON_USER_STOP,
                        "replaced": END_REASON_REPLACED,
                        "cmd": END_REASON_USER_STOP,
                    }.get(reason, END_REASON_USER_STOP)
                    try:
                        asyncio.create_task(self._event_store.mark_end_by_surface(
                            user_id=receiver.user_id,
                            surface_id=surface_id,
                            reason=end_reason,
                        ))
                    except Exception as exc:
                        log.warning(
                            "cast_event_mark_end_on_close_failed",
                            registration_id=registration_id,
                            surface_id=surface_id, error=str(exc),
                        )

        if event.event == "ready":
            # ready carries device fingerprint (platform, version, etc.)
            receiver.info = dict(event.data or {})
            receiver.label = str(
                event.data.get("label") or event.data.get("name") or "",
            )
            # Trusted-receiver binding — if the receiver supplied a
            # stable device_id, look up/create the durable row so
            # this connection is recognized across reboots. Runs as
            # a background task because we don't want a slow DB to
            # delay event fanout.
            device_id = str(event.data.get("device_id") or "").strip()
            if device_id and self._trusted_store is not None:
                try:
                    asyncio.create_task(self._bind_trusted(
                        registration_id, device_id, event.data,
                    ))
                except Exception as exc:
                    log.warning(
                        "cast_receiver_trusted_bind_schedule_failed",
                        registration_id=registration_id, error=str(exc),
                    )

        self._fanout(receiver, event)

    async def _bind_trusted(
        self,
        registration_id: str,
        device_id: str,
        ready_info: dict[str, Any],
    ) -> None:
        """Async upsert of the trusted_receivers row + binding back
        onto the runtime record. Revoked devices are kicked out by
        closing the WS — every send afterwards will fail and the WS
        route will detach normally."""
        receiver = self._receivers.get(registration_id)
        if receiver is None or self._trusted_store is None:
            return
        try:
            trusted = await self._trusted_store.upsert_on_connect(
                user_id=receiver.user_id,
                device_id=device_id,
                platform=str(ready_info.get("platform") or ""),
                info=ready_info,
                label=str(ready_info.get("label") or ""),
            )
        except Exception as exc:
            log.warning(
                "cast_receiver_trusted_upsert_failed",
                registration_id=registration_id, error=str(exc),
            )
            return
        if trusted is None:
            # upsert returned None — either the existing row is revoked
            # OR user_id/device_id were empty. The first case is the
            # common one (user explicitly revoked this TV earlier); the
            # second indicates an upstream bug. Look up the revoked row
            # to surface a useful payload (trusted_id) to the receiver
            # so it can stop the re-pair loop and point the user at the
            # restore UI.
            existing = await self._trusted_store.get_by_device(
                device_id, user_id=receiver.user_id,
            )
            revoke_reason = (
                "revoked" if (existing and existing.is_revoked)
                else "unknown"
            )
            log.warning(
                "cast_receiver_revoked_device_disconnect",
                registration_id=registration_id, device_id=device_id,
                trusted_id=(existing.id if existing else ""),
                reason=revoke_reason,
            )
            # Send a structured cmd before the close so the receiver
            # can render a terminal placeholder instead of looping. The
            # send may fail if the WS is already gone — that's fine,
            # the subsequent close() is what actually matters.
            from augmentum.cast.receiver_protocol import (
                CMD_REVOKED,
                ReceiverCmd,
            )
            try:
                await self._send_to(receiver, ReceiverCmd(
                    cmd=CMD_REVOKED,
                    args={
                        "trusted_id": existing.id if existing else "",
                        "reason": revoke_reason,
                    },
                ))
            except Exception:
                log.debug("receiver_revoke_cmd_send_failed", exc_info=True)
            try:
                await receiver.ws.close(
                    code=4003, reason="receiver revoked",
                )
            except Exception:
                log.debug("receiver_revoke_ws_close_failed", exc_info=True)
            self._receivers.pop(registration_id, None)
            return
        # Bind so subsequent send/list/cast helpers can map back.
        receiver.trusted_id = trusted.id
        # Prefer the user-chosen label over whatever the device announced.
        if trusted.label:
            receiver.label = trusted.label

    def _fanout(self, receiver: ConnectedReceiver, event: ReceiverEvent) -> None:
        for user_filter, id_filter, queue in list(self._subscribers):
            if user_filter and user_filter != receiver.user_id:
                continue
            if id_filter and id_filter != receiver.registration_id:
                continue
            try:
                queue.put_nowait((receiver.registration_id, event))
            except asyncio.QueueFull:
                # Drop the oldest to make room — better to lose an old
                # progress tick than block the receiver thread.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait((receiver.registration_id, event))
                except asyncio.QueueFull:
                    pass

    async def subscribe(
        self,
        *,
        user_id: str = "",
        registration_id: str = "",
        maxsize: int = 64,
    ) -> AsyncIterator[tuple[str, ReceiverEvent]]:
        """Yield (registration_id, event) tuples matching the filters.

        Empty filter = match-all. When both filters are given, both
        must match. The async iterator never terminates; cancel the
        consuming task to stop subscribing.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, maxsize))
        entry = (user_id, registration_id, queue)
        self._subscribers.append(entry)
        try:
            while True:
                yield await queue.get()
        finally:
            try:
                self._subscribers.remove(entry)
            except ValueError:
                pass

    # ── Shutdown ──────────────────────────────────────────────────

    async def close_all(self) -> None:
        """Close every WS gracefully — called at app shutdown so
        receivers see a clean close frame instead of a torn TCP."""
        receivers = list(self._receivers.values())
        self._receivers.clear()
        for receiver in receivers:
            try:
                await receiver.ws.close(code=1001, reason="server shutdown")
            except Exception:
                log.debug("receiver_shutdown_ws_close_failed", exc_info=True)
