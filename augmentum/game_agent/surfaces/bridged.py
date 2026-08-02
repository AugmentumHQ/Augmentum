"""WebSocket-bridged surface adapter.

A single adapter implementation that fits any surface whose wire is
"open a WebSocket, send JSON frames both ways". js13k browser shims
and the Luanti agent mod both speak this shape; the routes layer
hands the live :class:`starlette.websockets.WebSocket` to
:class:`BridgedAdapter` once the client has connected.

Wire protocol
-------------
Bridge -> adapter (one JSON object per frame)::

    {"kind": "event", "data": {...arbitrary surface vocabulary...}}
    {"kind": "event", "data": {...}, "confidence": 0.8}    # optional confidence
    {"kind": "frame", "png_b64": "<base64-encoded PNG>"}   # cached for snapshot_frame()
    {"kind": "ping"}                                       # heartbeat; ignored
    {"kind": "bye"}                                        # bridge requests session stop

Adapter -> bridge::

    {"action": "<semantic_id>", "duration_ms": <int>}

The adapter does not interpret ``data``; the surface-specific
vocabulary is declared in :class:`SurfaceCapsPayload.log_schema` and
the slow-path agent learns it from the live log.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections import deque
from collections.abc import Callable
from typing import Any, cast

import structlog
from starlette.websockets import WebSocket, WebSocketDisconnect

from augmentum.game_agent.control import ComposedProfile
from augmentum.game_agent.schema import (
    EventPayload,
    ObservationModality,
    SurfaceCapsPayload,
    SurfaceKind,
)
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn

log = structlog.get_logger(__name__)


# Sentinel pumped through emit() if the bridge announces "bye". The
# orchestrator treats it like any other event; the *route* layer
# observes this entry kind in the log and triggers Orchestrator.stop().
_BYE_EVENT_DATA = {"event": "bridge_bye"}


class BridgedAdapter:
    """Live-WebSocket surface adapter.

    Use when:
    - A client (browser shim or Luanti Lua mod) has just connected to
      the bridge endpoint and we need an adapter that owns that
      connection for the duration of the session.

    Expects:
    - ``websocket`` is already :meth:`accept`-ed by the caller.
    - ``surface_kind`` and ``log_schema`` are the strings the route
      layer derived from the URL + the original session request.
    - ``semantic_inputs`` was declared by the client during session
      creation or negotiated on the WS handshake.

    Returns:
    - From :meth:`start`, control to the orchestrator; the adapter
      spawns a read task that pushes events through ``emit`` until
      the bridge disconnects or :meth:`stop` is called.
    """

    def __init__(
        self,
        *,
        websocket: WebSocket,
        surface_kind: SurfaceKind,
        semantic_inputs: list[str],
        log_schema: str,
        observation_modalities: list[ObservationModality] | None = None,
        on_bridge_disconnect: Callable[[], None] | None = None,
        profile: ComposedProfile | None = None,
    ) -> None:
        self._ws = websocket
        self._surface_kind = surface_kind
        # When a ComposedProfile is supplied, the profile is the source
        # of truth for vocabulary + hints. The semantic_inputs argument
        # is ignored in favor of profile.semantic_inputs() so a caller
        # can't accidentally desync the agent vocabulary from the
        # profile that's actually doing the resolution.
        self._profile = profile
        if profile is not None:
            self._semantic_inputs = profile.semantic_inputs()
            self._input_hints: dict[str, str] | None = profile.hints()
        else:
            self._semantic_inputs = list(semantic_inputs)
            self._input_hints = None
        self._log_schema = log_schema
        # Default modalities cover the common case for both js13k and
        # luanti -- structured events + optional frames pushed by the
        # bridge. Adapters that want OCR or memory channels should
        # pass them in explicitly.
        self._observation_modalities: list[ObservationModality] = list(
            observation_modalities or ["log", "frame"]
        )
        self._on_bridge_disconnect = on_bridge_disconnect

        self._resolver = SemanticInputResolver()
        for semantic in self._semantic_inputs:
            self._resolver.bind(semantic, self._make_outbound_resolver(semantic))
        # Chords ride the same wire as single presses (one frame, one
        # ack) — the iframe holds every part simultaneously. Bound
        # unconditionally: the wire shape degrades gracefully on an
        # older client (unknown ``chord`` field is ignored and only the
        # primary lands), and the ack path is identical.
        self._resolver.bind_chord(self._dispatch_chord)

        self._read_task: asyncio.Task[None] | None = None
        # Frame ring buffer (Phase B: temporal chunks).
        #
        # We keep the last N decoded PNG frames so the slow path can
        # snapshot a short sequence -- the agent reasons about CHANGE
        # (text scroll, character motion, animation progress) rather
        # than a single still. Maxlen 8 gives headroom for callers
        # that want longer windows; the orchestrator picks how many
        # to actually send to the LLM per turn.
        self._frame_buffer: deque[bytes] = deque(maxlen=8)
        self._stopped = asyncio.Event()
        # Serializes outbound writes -- starlette's WebSocket is not
        # safe for concurrent send_text from multiple tasks.
        self._send_lock = asyncio.Lock()
        # ── Input delivery guarantee ──────────────────────────────────
        #
        # _pending_acks holds an asyncio.Event per in-flight input. The
        # outbound resolver mints a request_id, registers an Event,
        # sends the WS frame, then waits for the read loop to ``set()``
        # it when the matching input_ack arrives. Times out at
        # ``duration_ms + _ACK_OVERHEAD_MS`` (the iframe's hold +
        # post-release settle + WS roundtrip).
        #
        # We do NOT auto-retry on timeout: not all presses are
        # idempotent and double-firing could cause real harm
        # (move-twice when expected once). Instead the timeout is
        # surfaced as an ``agent_error`` log entry the slow path reads
        # next turn, letting the agent itself decide whether to re-emit.
        self._pending_acks: dict[str, asyncio.Event] = {}
        self._next_request_seq: int = 0
        # Captured from start() so the outbound resolver can emit
        # delivery-status events (input_dispatched / input_ack_timeout)
        # into the orchestrator's normal event pipeline. The agent
        # sees them in LIVE_LOG_TAIL on its next planning turn.
        self._emit: EmitEventFn | None = None

    # ── SurfaceAdapter Protocol ───────────────────────────────────

    @property
    def resolver(self) -> SemanticInputResolver:
        return self._resolver

    def caps(self) -> SurfaceCapsPayload:
        controller_id = (
            self._profile.controller.id if self._profile is not None else None
        )
        game_id = (
            self._profile.game.id if self._profile is not None else None
        )
        return SurfaceCapsPayload(
            semantic_inputs=self._semantic_inputs,
            log_schema=self._log_schema,
            observation_modalities=self._observation_modalities,
            input_hints=self._input_hints,
            controller_profile=controller_id,
            game_profile=game_id,
        )

    async def start(self, emit: EmitEventFn) -> None:
        self._stopped.clear()
        self._emit = emit
        self._read_task = asyncio.create_task(self._read_loop(emit), name="bridge-read")

    async def stop(self) -> None:
        self._stopped.set()
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None
        # Close the socket if still open; ignore double-close errors.
        try:
            await self._ws.close()
        except Exception as exc:
            log.debug("game_bridged_ws_close_failed", error=str(exc))

    async def snapshot_frame(self) -> bytes | None:
        """Return the most-recent frame, or None if the buffer is empty.

        Legacy single-frame API kept for callers that haven't migrated
        to :meth:`snapshot_frames`. New code should prefer the
        multi-frame variant so the agent has temporal context.
        """

        return self._frame_buffer[-1] if self._frame_buffer else None

    async def snapshot_frames(self, n: int = 3) -> list[bytes]:
        """Return up to the last ``n`` frames in oldest-first order.

        Use when:
        - The orchestrator is preparing a slow-path turn and the LLM
          accepts multi-image prompts (every modern VLM does). A short
          sequence lets the model reason about animation, motion, and
          action causality -- not just a single snapshot.

        Returns:
        - A list with length ``min(n, len(buffer))``. Empty when no
          frames have arrived yet. Frames are independent byte
          strings; the caller may reorder, drop, or re-encode them.
        """

        if n <= 0 or not self._frame_buffer:
            return []
        if n >= len(self._frame_buffer):
            return list(self._frame_buffer)
        # Take the last n (newest end of the deque) preserving order.
        return list(self._frame_buffer)[-n:]

    async def push_audio(self, *, mime: str, bytes_b64: str, utterance: str) -> None:
        """Push a companion speech frame down the WS to the browser.

        Used by the orchestrator when ``plan.say`` is non-empty and
        the voice bridge has produced audio. The wire frame shape:

            {kind: "audio", mime: "audio/mpeg", bytes_b64: "...",
             utterance: "the actual text"}

        Failures (closed socket) are swallowed and logged once — the
        session keeps running, only the voice cuts out.
        """

        if not bytes_b64:
            return
        payload = json.dumps(
            {
                "kind": "audio",
                "mime": mime,
                "bytes_b64": bytes_b64,
                "utterance": utterance,
            },
            separators=(",", ":"),
        )
        async with self._send_lock:
            try:
                await self._ws.send_text(payload)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "bridge.audio_send_failed",
                    error=str(exc),
                    utterance_len=len(utterance),
                )
                self._stopped.set()
                if self._on_bridge_disconnect is not None:
                    self._on_bridge_disconnect()

    # ── Internals ─────────────────────────────────────────────────

    async def _read_loop(self, emit: EmitEventFn) -> None:
        try:
            while not self._stopped.is_set():
                try:
                    raw = await self._ws.receive_text()
                except WebSocketDisconnect:
                    log.info("bridge.disconnected", surface=self._surface_kind)
                    if self._on_bridge_disconnect is not None:
                        self._on_bridge_disconnect()
                    return
                msg = self._safe_parse(raw)
                if msg is None:
                    continue
                await self._handle_frame(msg, emit)
        except asyncio.CancelledError:
            return

    async def _handle_frame(self, msg: dict[str, Any], emit: EmitEventFn) -> None:
        kind = msg.get("kind")
        if kind == "event":
            data = cast(dict[str, Any], msg.get("data") or {})
            # input_ack events from the parent EmulatorBridge close the
            # delivery-guarantee loop: if we have a pending Event for
            # this request_id, set it so the resolver returns. We also
            # always emit the event normally so the agent sees it in
            # LIVE_LOG_TAIL alongside its action history.
            if data.get("event") == "input_ack":
                req_id = data.get("request_id")
                if isinstance(req_id, str) and req_id:
                    pending = self._pending_acks.get(req_id)
                    if pending is not None:
                        pending.set()
            confidence = msg.get("confidence")
            await emit(
                EventPayload(
                    channel="log",
                    data=data,
                    confidence=confidence if isinstance(confidence, int | float) else None,
                )
            )
        elif kind == "frame":
            b64 = msg.get("png_b64")
            if isinstance(b64, str):
                try:
                    self._frame_buffer.append(base64.b64decode(b64, validate=True))
                except (ValueError, TypeError):
                    log.warning("bridge.bad_frame_b64")
        elif kind == "ping":
            return  # heartbeat
        elif kind == "bye":
            await emit(EventPayload(channel="log", data=_BYE_EVENT_DATA))
            self._stopped.set()
            if self._on_bridge_disconnect is not None:
                self._on_bridge_disconnect()
        else:
            log.warning("bridge.unknown_kind", kind=kind)

    def _safe_parse(self, raw: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("bridge.bad_json", preview=raw[:80])
            return None
        if not isinstance(obj, dict):
            log.warning("bridge.non_object", preview=raw[:80])
            return None
        return cast(dict[str, Any], obj)

    # Wall-clock budget between WS dispatch and the iframe's
    # AGENT_INPUT_ACK arriving back. iframe-side: hold for duration_ms
    # via rAF, release, settle 150ms, capture fingerprint, post ACK.
    # WS roundtrip + postMessage hops add ~50ms. 500ms total overhead
    # covers all of it with comfortable headroom.
    _ACK_OVERHEAD_MS = 500

    def _wire_fields(self, semantic: str) -> dict[str, Any]:
        """Resolve one semantic to its wire transport fields (may be {})."""

        if self._profile is None:
            return {}
        button = self._profile.resolve(semantic)
        if button is None:
            return {}
        return {
            "wire_kind": self._profile.controller.wire_kind,
            "wire_code": button.wire_code,
        }

    async def _dispatch_chord(self, semantics: list[str], duration_ms: int) -> None:
        """Chord entry point bound on the resolver (see bind_chord)."""

        if not semantics:
            return
        await self._dispatch_input(semantics[0], duration_ms, extras=semantics[1:])

    def _make_outbound_resolver(self, semantic: str):  # type: ignore[no-untyped-def]
        async def _resolver(duration_ms: int) -> None:
            await self._dispatch_input(semantic, duration_ms)

        return _resolver

    async def _dispatch_input(
        self, semantic: str, duration_ms: int, extras: list[str] | None = None,
    ) -> None:
        """Send one input frame (single press or chord) and await its ack.

        Mints a request_id, registers the pending event, sends, and
        waits for the iframe's AGENT_INPUT_ACK to close the loop. On
        timeout we surface an ``input_ack_timeout`` event so the slow
        path sees the gap and can adapt. We do NOT retry: not every
        press is idempotent and silently double-firing could move the
        character twice when the agent intended once. The agent is the
        right decision point for retry semantics.
        """

        self._next_request_seq += 1
        request_id = f"act-{self._next_request_seq}"
        ack_event = asyncio.Event()
        self._pending_acks[request_id] = ack_event

        # When a profile is active, resolve Layer-1 -> wire code so
        # the iframe can dispatch directly without a second lookup.
        # Both the resolved data AND the original semantic name go
        # on the wire: the iframe prefers wire_code when present;
        # the semantic name lives in the log entry for replay /
        # debugging / journal references.
        payload_dict: dict[str, Any] = {
            "action": semantic,
            "duration_ms": duration_ms,
            "request_id": request_id,
            **self._wire_fields(semantic),
        }
        if extras:
            # Chord parts: each carries its own wire resolution so the
            # iframe holds every button simultaneously with the primary.
            payload_dict["chord"] = [
                {"button": s, **self._wire_fields(s)} for s in extras
            ]
        payload = json.dumps(payload_dict, separators=(",", ":"))
        send_ok = False
        try:
            async with self._send_lock:
                try:
                    await self._ws.send_text(payload)
                    send_ok = True
                except Exception as exc:  # noqa: BLE001
                    # If the bridge has gone away, mark stopped so
                    # the orchestrator's action worker doesn't keep
                    # trying. Logged as a single warning per outage.
                    log.warning(
                        "bridge.send_failed",
                        semantic=semantic, error=str(exc),
                    )
                    self._stopped.set()
                    if self._on_bridge_disconnect is not None:
                        self._on_bridge_disconnect()
                    return
            if not send_ok:
                return
            # Wait for the matching ACK. Timeout scales with the
            # press duration so a long hold (e.g. 2000 ms duration)
            # doesn't false-positive as a lost ACK.
            timeout_s = max(
                0.6, (duration_ms + self._ACK_OVERHEAD_MS) / 1000.0,
            )
            try:
                await asyncio.wait_for(ack_event.wait(), timeout=timeout_s)
            except TimeoutError:
                log.warning(
                    "bridge.input_ack_timeout",
                    semantic=semantic,
                    duration_ms=duration_ms,
                    timeout_s=timeout_s,
                    request_id=request_id,
                )
                # Surface to the log so the slow path observes it.
                if self._emit is not None:
                    try:
                        await self._emit(
                            EventPayload(
                                channel="log",
                                data={
                                    "event": "input_ack_timeout",
                                    "semantic": semantic,
                                    "duration_ms": duration_ms,
                                    "request_id": request_id,
                                },
                            )
                        )
                    except Exception as emit_exc:
                        log.debug(
                            "game_bridged_input_ack_timeout_emit_failed",
                            error=str(emit_exc),
                        )
        finally:
            # Always reap the pending entry, ack or not, so a
            # long-running session doesn't accumulate stale events.
            self._pending_acks.pop(request_id, None)
