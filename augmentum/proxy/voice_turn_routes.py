"""HTTPS companion turn — the cert-free, model-driven agency path.

The voice WebSocket (``voice_routes.py``) is the rich, always-listening
path. It needs a trusted TLS cert (WebSockets can't ride a self-signed
cert in an Android WebView), which the lock-screen / on-device-STT
surfaces don't have.

This module is the **cert-free seam**: a plain HTTPS POST that takes a
*final* transcript (already produced on-device by Moonshine STT) and runs
it through the **same companion tool loop the web app uses** — the
``becca_direct`` handler, which composes her prompt and consumes
``native_loop_events`` with the model-driven roster from
``select_companion_tools``. The MODEL decides which verbs to fire (it
sees each tier-3 primitive as a tool with name + description + arg
schema), including ``app.act`` for the dynamic in-app-button catalog.

No regex intent matching. Verb selection is the model's job — the
lesson of the regex switchboard (open-slot templates after permissive
openers) is not re-learned here. The only thing this path adds over the
WS is the transport: it collects the loop's events into ONE JSON
response so the on-device client can speak the reply (on-device TTS) and
route any surface events through ``intent-action-router.js`` — identical
behavior to the WS path, no socket required.

Response::

    {"handled": true, "reply": "...",
     "surfaces": [{"channel": "navigate.open_surface",
                   "payload": {"surface": "browse"}}, ...]}

    {"handled": false}   # companion unavailable — client sends to chat

Always 200 so the client can reliably fall through. See
``docs/superpowers/specs/2026-06-10-companion-headless-agency-design.md``.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["voice"])


class VoiceTurnRequest(BaseModel):
    transcript: str = Field(..., max_length=4000)
    session_id: str = Field("", max_length=128)
    # Client surface hint — informational; the companion loop offers the
    # same roster regardless. Kept for the opt-in capture row.
    surface: str = Field("voice", max_length=32)
    # Optional image attachments for THIS turn — inline ``data:`` URLs or
    # ``/api/chat-images/<id>`` refs. Lets the voice/lock-screen companion
    # SEE something the user shows it (and is the seam a live-camera frame
    # loop feeds). Routed through the same vision pipeline the chat path
    # uses: VL primary reads them directly; a text-only primary gets the
    # sibling-captioner's text. Capped to keep the cert-free POST small.
    images: list[str] = Field(default_factory=list, max_length=6)
    # Prior conversation turns so the cert-free turn has MEMORY — without this
    # each call is stateless and "forgets" the previous utterance. Each entry is
    # ``{"role": "user"|"assistant", "content": "..."}``; newest last. The
    # becca_direct composer renders them as its transcript window
    # (``_intent_from_request`` already derives recent_turns from request
    # messages). Capped to bound the prompt; the client trims to a window.
    history: list[dict] = Field(default_factory=list, max_length=40)
    # Optional per-turn model PIN (the phone's chosen "voice model"). When set,
    # this turn runs on that model instead of the companion's primary chat model
    # — letting a daily-driver phone pin a small/fast model for lower latency
    # WITHOUT changing the server-global chat model. Blank = companion primary.
    model: str = Field("", max_length=128)


def _resolve_user_id(request: Request) -> str:
    """user_id from the authenticated scope; '' when unauthenticated."""
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


async def _resolve_voice_backend(runtime, app_state, body: VoiceTurnRequest):
    """Resolve ``(backend, model_name)`` for the turn, honoring an optional
    per-turn model pin (``body.model``). Falls back to the companion's primary
    chat model when unset or unresolvable, so a bad/stale pin never drops a turn.
    """
    pinned = (body.model or "").strip()
    if pinned:
        registry = getattr(app_state, "provider_registry", None)
        if registry is not None:
            try:
                backend, served = await registry.resolve_backend_for_model(pinned)
                if backend is not None:
                    return backend, (served or pinned)
            except Exception as exc:  # noqa: BLE001 — fall back to primary
                log.warning("voice_turn_pinned_model_unresolved", model=pinned, error=str(exc))
    from augmentum.companion_runtime import tiers
    return await tiers.primary(runtime)


async def _prepare_turn(request: Request, body: VoiceTurnRequest):
    """Resolve the companion + backend and build the streaming handler + chat
    request — the shared setup behind both the collected (`/api/voice/turn`)
    and streaming (`/api/voice/turn/stream`) routes.

    Returns ``(handler, req)`` or ``None`` when the companion path is
    unavailable (no user / runtime off / signed-out / backend down) — callers
    surface that as ``handled: false`` so the client falls through to chat.
    """
    transcript = (body.transcript or "").strip()
    if not transcript:
        return None

    user_id = _resolve_user_id(request)
    app_state = request.app.state
    if not user_id:
        return None
    if not getattr(settings, "companion_runtime_enabled", False):
        return None

    companions = getattr(app_state, "companions", None) or {}
    companion = companions.get("becca")
    if companion is None or not getattr(companion, "started", False):
        return None
    runtime = getattr(companion, "runtime", None)
    if runtime is None:
        return None

    try:
        backend, model_name = await _resolve_voice_backend(runtime, app_state, body)
    except Exception as exc:  # noqa: BLE001 — degrade to chat fall-through
        log.warning("voice_turn_backend_unavailable", error=str(exc))
        return None
    if backend is None:
        return None

    from augmentum.models.base import (
        InternalChatRequest,
        Message,
        apply_vision_pipeline,
    )
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    images = [i for i in (body.images or []) if isinstance(i, str) and i.strip()]
    # Prior turns → leading messages so the composer renders them as history
    # (this is what gives the cert-free turn conversational memory).
    history_msgs: list[Message] = []
    for h in body.history or []:
        if not isinstance(h, dict):
            continue
        role = str(h.get("role") or "").strip()
        content = str(h.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            history_msgs.append(Message(role=role, content=content))
    req = InternalChatRequest(
        model=model_name or "",
        messages=[
            *history_msgs,
            Message(role="user", content=transcript, images=images or None),
        ],
        stream=True,
    )
    if images:
        try:
            await apply_vision_pipeline(req, app_state, backend, reason_on_vision=True)
        except Exception as exc:  # noqa: BLE001 — never fail a turn over vision
            log.warning("voice_turn_vision_pipeline_failed", error=str(exc))

    handler = BeccaDirectHandler(
        backend=backend,
        app_state=app_state,
        session_id=body.session_id,
        user_id=user_id,
        surface=(body.surface or "assist"),
    )
    return handler, req


@router.post("/api/voice/turn")
async def voice_turn(request: Request, body: VoiceTurnRequest) -> JSONResponse:
    """Run a final transcript through the companion's model-driven loop.

    Mirrors the voice pipeline's ``_consume_native_loop`` and the chat
    path's ``BeccaDirectHandler`` — resolve the companion, build a chat
    request with the transcript, stream the handler, and collect the
    reply text + any surface events the model's tool calls produced.

    Returns ``handled: false`` (200) whenever the companion path isn't
    available (runtime off, signed-out, compose bypass) so the client
    falls through to its normal chat send instead of special-casing a
    non-200.
    """
    transcript = (body.transcript or "").strip()
    if not transcript:
        return JSONResponse({"handled": False, "empty": True})

    user_id = _resolve_user_id(request)
    app_state = request.app.state

    # Companion path only — no user / runtime off → tell the client to
    # use its own chat send (which already carries conversational agency).
    if not user_id:
        return JSONResponse({"handled": False})
    if not getattr(settings, "companion_runtime_enabled", False):
        return JSONResponse({"handled": False})

    companions = getattr(app_state, "companions", None) or {}
    companion = companions.get("becca")
    if companion is None or not getattr(companion, "started", False):
        return JSONResponse({"handled": False})
    runtime = getattr(companion, "runtime", None)
    if runtime is None:
        return JSONResponse({"handled": False})

    # Resolve the backend — honoring the phone's optional per-turn model pin.
    try:
        backend, model_name = await _resolve_voice_backend(runtime, app_state, body)
    except Exception as exc:  # noqa: BLE001 — degrade to chat fall-through
        log.warning("voice_turn_backend_unavailable", error=str(exc))
        return JSONResponse({"handled": False})
    if backend is None:
        return JSONResponse({"handled": False})

    from augmentum.models.base import (
        InternalChatRequest,
        Message,
        apply_vision_pipeline,
    )
    from augmentum.modes.becca_direct.handler import BeccaDirectHandler

    # The handler composes her own system prompt from the runtime and
    # substitutes it; we only supply the user turn. session_id scopes the
    # referent cache (trail / "play that again") to this conversation.
    images = [i for i in (body.images or []) if isinstance(i, str) and i.strip()]
    req = InternalChatRequest(
        model=model_name or "",
        messages=[Message(role="user", content=transcript, images=images or None)],
        stream=True,
    )
    # Same image handling the chat routes run, so the voice/lock-screen
    # companion sees what it's shown: VL primary reads frames directly; a
    # text-only primary gets the sibling-captioner's text inlined. No-op
    # when no images were attached.
    if images:
        try:
            # reason_on_vision: a held-up image is a "what is this / what do
            # you think?" moment — let her actually reason about it.
            await apply_vision_pipeline(req, app_state, backend, reason_on_vision=True)
        except Exception as exc:  # noqa: BLE001 — never 500 a voice turn over vision
            log.warning("voice_turn_vision_pipeline_failed", error=str(exc))
    handler = BeccaDirectHandler(
        backend=backend,
        app_state=app_state,
        session_id=body.session_id,
        user_id=user_id,
        surface=(body.surface or "assist"),
    )

    from augmentum.training.trace_context import begin_capture, end_capture
    _cap_ctx, _cap_tok = begin_capture(
        user_id=user_id, session_id=body.session_id, mode="voice",
    )
    reply_parts: list[str] = []
    surfaces: list[dict] = []
    fired = False
    try:
        async for chunk in handler.handle_stream(req):
            delta = getattr(chunk, "content_delta", "") or ""
            if delta:
                reply_parts.append(delta)
            aug = getattr(chunk, "augmentum", None)
            if not isinstance(aug, dict):
                continue
            tool_result = aug.get("becca_tool_result")
            if not isinstance(tool_result, dict):
                continue
            # ui_effects carry the drained surface events; reshape each
            # back to the {channel, payload} form the client's
            # intent-action-router routes (open browse, play, sticky, …).
            for eff in tool_result.get("ui_effects") or []:
                channel = (eff or {}).get("kind") or ""
                if channel:
                    fired = True
                    surfaces.append({
                        "channel": channel,
                        "payload": (eff or {}).get("payload") or {},
                    })
    except Exception as exc:  # noqa: BLE001 — loop must never 500 the turn
        end_capture(_cap_ctx, _cap_tok, error=type(exc).__name__)
        log.warning("voice_turn_loop_error", error=str(exc))
        return JSONResponse({"handled": False})
    reply = "".join(reply_parts).strip()
    # Record the assembled spoken reply into the trace before closing it: the
    # native loop emits its final text as events, so the backend-boundary hook
    # captured an empty response. Without this the voice trace has no
    # final_response. Capture-only; never affects the turn.
    try:
        from augmentum.training.trace_context import note_final_response
        note_final_response(reply)
    except Exception:  # noqa: BLE001
        pass
    end_capture(_cap_ctx, _cap_tok)
    # Nothing to say and nothing done — let the client retry via chat
    # rather than speak silence.
    if not reply and not surfaces:
        return JSONResponse({"handled": False})

    # Opt-in capture for the on-device intent dataset (gated; default OFF;
    # refuses anon writes). Records the transcript + whether the model's
    # tools fired a surface action — useful act/converse training signal.
    if getattr(settings, "intent_capture_enabled", False):
        try:
            sm = getattr(app_state, "state_manager", None)
            if sm is not None:
                from augmentum.state.backends.sqlite import SQLiteBackend
                if isinstance(sm.backend, SQLiteBackend):
                    from augmentum.intent.capture_store import record_intent_capture
                    await record_intent_capture(
                        sm.backend.conn,
                        user_id=user_id,
                        session_id=body.session_id,
                        surface="voice_turn",
                        input_text=transcript,
                        active_surface=body.surface,
                        goal="act" if fired else "converse",
                        effective_goal="act" if fired else "converse",
                        addressed=True,  # held-to-talk == explicitly addressed
                        confidence=1.0,
                    )
        except Exception:  # noqa: BLE001 — capture never disrupts the turn
            log.debug("voice_turn_capture_failed", exc_info=True)

    return JSONResponse({
        "handled": True,
        "reply": reply,
        "speak": reply,
        "surfaces": surfaces,
    })


@router.post("/api/voice/turn/stream")
async def voice_turn_stream(
    request: Request, body: VoiceTurnRequest,
) -> StreamingResponse:
    """Streaming sibling of :func:`voice_turn` — emits the companion turn's
    live state as Server-Sent Events so the native overlay's orb can reflect
    each *resolver* stage as it happens instead of a dead spinner.

    Reuses the SAME handler (`BeccaDirectHandler.handle_stream`) — no change to
    the shared `native_loop`; this route just forwards its chunks instead of
    collecting them. Event types (one JSON object per ``data:`` frame):

      * ``{"type":"status","stage":"thinking"}``           turn started
      * ``{"type":"tool_call","tool":"web_search"}``        a resolver is firing
      * ``{"type":"tool_result","tool":...,"ok":true}``     resolver finished
      * ``{"type":"surface","channel":...,"payload":...}``  a UI action fired
      * ``{"type":"metrics","ttft_ms":1234}``               time to first content
      * ``{"type":"delta","text":"..."}``                   answer text
      * ``{"type":"done","handled":true,"reply":...,"surfaces":[...]}``

    Live tokens/sec is a later layer (it needs the loop's final generation to
    stream via ``backend.chat_stream``); today the answer arrives as one chunk,
    so this carries ttft + stage states, not a token rate. Always opens the
    stream (even for the unavailable case → a single ``done handled:false``) so
    the client has a clean fall-through, mirroring the collected route's 200.
    """

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    prepared = await _prepare_turn(request, body)

    async def _events():
        if prepared is None:
            yield _sse({"type": "done", "handled": False})
            return
        handler, req = prepared
        from augmentum.training.trace_context import begin_capture, end_capture
        _user = getattr(request.scope.get("user"), "id", "") if request.scope.get("user") else ""
        _cap_ctx, _cap_tok = begin_capture(
            user_id=_user, session_id=body.session_id, mode="voice",
        )
        t0 = time.monotonic()
        ttft_sent = False
        reply_parts: list[str] = []
        surfaces: list[dict] = []
        yield _sse({"type": "status", "stage": "thinking"})
        try:
            async for chunk in handler.handle_stream(req):
                delta = getattr(chunk, "content_delta", "") or ""
                if delta:
                    if not ttft_sent:
                        ttft_sent = True
                        yield _sse({
                            "type": "metrics",
                            "ttft_ms": int((time.monotonic() - t0) * 1000),
                        })
                    reply_parts.append(delta)
                    yield _sse({"type": "delta", "text": delta})
                aug = getattr(chunk, "augmentum", None)
                if not isinstance(aug, dict):
                    continue
                metrics = aug.get("becca_metrics")
                if isinstance(metrics, dict):
                    out = {"type": "metrics"}
                    out.update(metrics)  # tok_per_s, gen_ms, completion_tokens
                    yield _sse(out)
                call = aug.get("becca_tool_call")
                if isinstance(call, dict):
                    yield _sse({"type": "tool_call", "tool": call.get("tool", "")})
                tr = aug.get("becca_tool_result")
                if isinstance(tr, dict):
                    yield _sse({
                        "type": "tool_result",
                        "tool": tr.get("tool", ""),
                        "ok": bool(tr.get("ok", True)),
                    })
                    for eff in tr.get("ui_effects") or []:
                        channel = (eff or {}).get("kind") or ""
                        if channel:
                            payload = (eff or {}).get("payload") or {}
                            surfaces.append({"channel": channel, "payload": payload})
                            yield _sse({
                                "type": "surface",
                                "channel": channel,
                                "payload": payload,
                            })
        except Exception as exc:  # noqa: BLE001 — a loop error ends the stream cleanly
            end_capture(_cap_ctx, _cap_tok, error=type(exc).__name__)
            log.warning("voice_turn_stream_loop_error", error=str(exc))
            yield _sse({"type": "done", "handled": False})
            return
        end_capture(_cap_ctx, _cap_tok)

        reply = "".join(reply_parts).strip()
        yield _sse({
            "type": "done",
            "handled": bool(reply or surfaces),
            "reply": reply,
            "surfaces": surfaces,
        })

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
