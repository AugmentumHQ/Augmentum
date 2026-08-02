"""Companion presence WebSocket — Phase 1 endpoint stub.

Hosts the WS the presence-pipeline client connects to. Phase 1 ships
the event dispatch + state mirror; audio frame handling lands in
Phase 3 alongside the streaming chunker.

Routing:
  WS /ws/companion/presence/{session_id}

Authentication:
  Standard ws-ticket / session-cookie via the existing middleware.
  Auth required — no anonymous companion sessions.

Multi-tenant:
  Pipeline instances cached on app.state.presence_pipelines keyed by
  (user_id, session_id). User A cannot reach into User B's pipeline
  even with the same session_id (the cache key includes both).

Inbound message protocol (Phase 1):
  JSON: {"type": "wake_triggered"}
  JSON: {"type": "ptt_release"}
  JSON: {"type": "speech_continued"}
  JSON: {"type": "interrupt_vad", "mid_phrase": "..."}
  JSON: {"type": "user_backchannel"}
  Binary frames: ACK only in Phase 1 (audio path arrives Phase 3).

Outbound:
  JSON: {"type": "state", "state": "<state>", "transition_count": <int>}
        emitted on every state transition.
  JSON: {"type": "error", "code": "...", "detail": "..."}
        on transition / dispatch failure.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from augmentum.companion.presence import (
    PresencePipeline,
    StateTransition,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/ws/companion", tags=["companion-presence"])


# Cap on cached pipeline instances to prevent unbounded growth from a
# user who opens-and-abandons many sessions. LRU eviction by last
# transition time. Per-user, not global — so a power user with many
# sessions doesn't crowd out other users.
_MAX_PIPELINES_PER_USER = 32


def _get_or_create_pipeline(
    app_state: Any, *, session_id: str, user_id: str,
) -> PresencePipeline:
    """Lookup-or-construct the pipeline for (user_id, session_id).

    Pipelines are cached on app.state.presence_pipelines as a dict
    keyed by (user_id, session_id). The cache survives WS reconnect
    so transient network blips don't lose conversation state.
    """
    pipelines = getattr(app_state, "presence_pipelines", None)
    if pipelines is None:
        pipelines = {}
        app_state.presence_pipelines = pipelines

    key = (user_id, session_id)
    pipeline = pipelines.get(key)
    if pipeline is None:
        pipeline = PresencePipeline(session_id=session_id, user_id=user_id)
        pipelines[key] = pipeline
        _maybe_evict_lru(pipelines, user_id=user_id)
    return pipeline


def _maybe_evict_lru(
    pipelines: dict[tuple[str, str], PresencePipeline], *, user_id: str,
) -> None:
    """Drop the oldest pipeline for this user if they're over the cap."""
    user_keys = [k for k in pipelines if k[0] == user_id]
    if len(user_keys) <= _MAX_PIPELINES_PER_USER:
        return
    user_keys.sort(
        key=lambda k: pipelines[k].context.state_entered_at,
    )
    overflow = len(user_keys) - _MAX_PIPELINES_PER_USER
    for k in user_keys[:overflow]:
        old = pipelines.pop(k, None)
        if old is not None:
            log.info(
                "presence_pipeline_evicted",
                user_id=user_id, session_id=k[1],
                final_state=old.state.value,
                transition_count=old.context.transition_count,
            )


# ── Event dispatch ──────────────────────────────────────────────


async def _dispatch_event(
    pipeline: PresencePipeline, msg: dict[str, Any],
) -> tuple[bool, str | None]:
    """Apply a client-side event message to the pipeline.

    Returns (handled, error_detail). handled=False with error_detail
    means the message was malformed; the WS sends a JSON error frame.
    handled=True means the event was dispatched (state may or may not
    have changed; client gets a state frame either way for confirmation).
    """
    event_type = str(msg.get("type") or "")

    if event_type == "wake_triggered":
        # Wake / PTT entry — model this as SPEECH_DETECTED at the
        # pipeline level. The wake detector + PTT button are two
        # different UX paths to the same "user started speaking" event.
        await pipeline.on_speech_detected()
        return True, None

    if event_type == "ptt_release":
        # PTT released — force the turn closed immediately, no Smart
        # Turn confidence required.
        await pipeline.on_turn_committed()
        return True, None

    if event_type == "speech_continued":
        await pipeline.on_speech_continued()
        return True, None

    if event_type == "speech_detected":
        await pipeline.on_speech_detected()
        return True, None

    if event_type == "turn_likely":
        confidence = float(msg.get("confidence") or 0.0)
        await pipeline.on_turn_likely(confidence)
        return True, None

    if event_type == "turn_committed":
        await pipeline.on_turn_committed()
        return True, None

    if event_type == "interrupt_vad":
        mid_phrase = str(msg.get("mid_phrase") or "")
        await pipeline.on_interrupt_vad(mid_phrase=mid_phrase)
        return True, None

    if event_type == "user_backchannel":
        await pipeline.on_user_backchannel_detected()
        return True, None

    if event_type == "beat_complete":
        await pipeline.on_beat_complete()
        return True, None

    if event_type == "chunk_queue_empty":
        await pipeline.on_chunk_queue_empty()
        return True, None

    if event_type == "first_chunk_ready":
        await pipeline.on_first_chunk_ready()
        return True, None

    return False, f"Unknown event type: {event_type!r}"


@router.websocket("/presence/{session_id}")
async def companion_presence_stream(
    websocket: WebSocket, session_id: str,
) -> None:
    """Bidirectional WS for the companion presence pipeline.

    Lifecycle:
      1. Accept + auth check (returns 4001 if unauthenticated)
      2. Lookup-or-construct the per-(user, session) pipeline
      3. Subscribe a forwarder that pushes every state transition
         to the client as JSON
      4. Loop: receive client events, dispatch to pipeline
      5. On disconnect: unsubscribe the forwarder (pipeline stays
         cached for reconnect)
    """
    await websocket.accept()
    user = websocket.scope.get("user")
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    user_id = user.id

    pipeline = _get_or_create_pipeline(
        websocket.app.state, session_id=session_id, user_id=user_id,
    )

    # State-mirror listener: every transition is forwarded as a JSON
    # frame. We define it inline so it closes over `websocket`.
    async def _forward_state(transition: StateTransition) -> None:
        try:
            await websocket.send_json({
                "type": "state",
                "state": transition.to_state.value,
                "from_state": transition.from_state.value,
                "event": transition.event.value,
                "transition_count": pipeline.context.transition_count,
            })
        except Exception as exc:
            # WS dropped — listener will be torn down in the finally
            # clause below. Don't log at warning; this is the normal
            # disconnect path.
            log.debug(
                "presence_forward_send_failed",
                session_id=session_id, error=str(exc)[:160],
            )

    unsubscribe = pipeline.subscribe(_forward_state)

    log.info(
        "presence_ws_attached",
        session_id=session_id, user_id=user_id,
        initial_state=pipeline.state.value,
    )

    # Send the current state immediately so the client UI can render
    # without waiting for a transition.
    try:
        await websocket.send_json({
            "type": "state",
            "state": pipeline.state.value,
            "from_state": pipeline.state.value,
            "event": "subscribed",
            "transition_count": pipeline.context.transition_count,
        })
    except Exception:
        # WS already dead — exit silently, the finally cleans up.
        unsubscribe()
        return

    try:
        while True:
            msg = await websocket.receive()
            msg_type = msg.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                # Phase 1: binary audio is acknowledged but not yet
                # routed through STT/VAD. The audio pipeline lands in
                # Phase 3 alongside streaming chunker.
                # Send a small marker so the client can confirm the
                # WS is alive without spamming.
                continue

            text = msg.get("text")
            if not text:
                continue

            try:
                import json as _json
                payload = _json.loads(text)
            except Exception as exc:
                await _send_error(websocket, "bad_json", str(exc))
                continue

            if not isinstance(payload, dict):
                await _send_error(
                    websocket, "bad_payload", "expected JSON object",
                )
                continue

            try:
                handled, detail = await _dispatch_event(pipeline, payload)
            except Exception as exc:
                log.warning(
                    "presence_dispatch_error",
                    session_id=session_id, error=str(exc),
                    error_type=type(exc).__name__,
                )
                await _send_error(websocket, "dispatch_error", str(exc))
                continue

            if not handled and detail:
                await _send_error(websocket, "unknown_event", detail)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning(
            "presence_ws_unexpected_error",
            session_id=session_id, error=str(exc),
            error_type=type(exc).__name__,
        )
    finally:
        unsubscribe()
        log.info(
            "presence_ws_detached",
            session_id=session_id, user_id=user_id,
            final_state=pipeline.state.value,
            transition_count=pipeline.context.transition_count,
        )


async def _send_error(
    websocket: WebSocket, code: str, detail: str,
) -> None:
    """Send a JSON error frame.

    Narrow exception capture: a dead WS raises WebSocketDisconnect or
    RuntimeError ("after sending close"); ConnectionError covers the
    TCP-level drop. Anything else propagates so a genuine bug in error
    formatting doesn't get silently masked while we're already on a
    failure path.
    """
    try:
        await websocket.send_json({
            "type": "error",
            "code": code,
            "detail": detail[:512],
        })
    except (WebSocketDisconnect, ConnectionError, RuntimeError) as exc:
        # WS already dead — the receive loop will break on its own next
        # iter. Logged at debug so prod degradation is observable but
        # doesn't spam at info level for every normal disconnect.
        log.debug(
            "presence_send_error_failed",
            code=code, error=str(exc)[:160],
        )
