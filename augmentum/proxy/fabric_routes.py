"""Fabric API routes: pairing endpoint + peer WebSocket.

Two surfaces:

  POST /api/fabric/pair
      Admin-only. Accepts a signed PairRequest from a remote peer,
      verifies the signature + fingerprint match, persists to
      fabric_nodes, returns this node's identity so the caller can
      complete its side of the handshake.

  WEBSOCKET /api/fabric/connect
      Peer-to-primary persistent connection. The peer sends ``hello``
      with their signed identity envelope; we look up the pinned
      pubkey for ``sender_node_id`` in fabric_nodes and verify. On
      success we register the socket with the coordinator and run
      the read loop until disconnect.

Every code path here is gated on ``settings.fabric_enabled``. With
the flag off (default), the routes return 503 immediately -- the
existence of the routes is harmless because nothing wires them into
an active fabric.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.fabric.capabilities import serialise
from augmentum.fabric.discovery import (
    derive_subnet_from_host,
    discover_fabric_peers,
)
from augmentum.fabric.pair_client import (
    OutboundPairError,
    initiate_pair_with_remote,
)
from augmentum.fabric.peer_auth import (
    PairRequest,
    PairRequestError,
    persist_pairing,
    verify_pair_request,
)
from augmentum.fabric.protocol import (
    MSG_HELLO,
    FabricEnvelope,
    FabricProtocolError,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/fabric", tags=["fabric"])

# Time budget for the first hello after WS accept. If a peer connects
# but doesn't speak within this window we close -- a stalled "open
# but silent" socket is the most common port-probe behaviour and we
# don't want to keep slots tied up indefinitely.
_HELLO_TIMEOUT_S = 10.0


# ── HTTP: pairing ─────────────────────────────────────────────────


@router.post("/pair")
async def fabric_pair(request: Request) -> dict:
    """Accept a pair request from a remote peer.

    Body shape: the JSON form of :class:`PairRequest`. This is the
    peer-to-peer entrypoint — the caller is ANOTHER augmentum node,
    not a logged-in browser session, so we do NOT gate on
    ``require_admin``. The authentication mechanism is the signed
    request itself:

    * ``timestamp`` (±_PAIR_REQUEST_TTL_S window) blocks replay.
    * ``fingerprint_hint`` must equal this node's own fingerprint;
      misaddressed requests bounce here.
    * The Ed25519 signature over the canonical bytes proves the
      caller holds the private key for ``pubkey_b64``.

    Operator-side trust is the out-of-band fingerprint paste in the
    Fabric UI (the operator on the initiating node types this node's
    fingerprint into their pair form). Gating with ``require_admin``
    here would 401 every legitimate pair (the remote peer has no
    session cookie) — exactly what the LAN report showed when this
    was first hit in production.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    body = await request.json()
    try:
        req = PairRequest(
            sender_node_id=str(body["sender_node_id"]),
            hostname=str(body.get("hostname", "")),
            pubkey_b64=str(body["pubkey_b64"]),
            fingerprint_hint=str(body["fingerprint_hint"]),
            role=str(body.get("role", "peer")),
            timestamp=int(body["timestamp"]),
            signature=str(body["signature"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"malformed pair request: {exc}"
        ) from None

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        raise HTTPException(
            status_code=503, detail="fabric coordinator not initialised"
        )

    own_fp = coordinator._identity.fingerprint
    try:
        verify_pair_request(req, own_fingerprint=own_fp)
    except PairRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend not available")

    addr = body.get("addr") or _client_addr(request)
    try:
        paired = await persist_pairing(conn, req=req, addr=addr)
    except Exception as exc:
        log.warning("fabric_pair_persist_failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"could not persist pairing: {exc}"
        ) from None

    await coordinator.register_paired_peer(paired)

    # Echo back this node's identity so the caller can complete its
    # half of the symmetric pair (it'll INSERT us into ITS
    # fabric_nodes table). The caller already supplied our fingerprint
    # as fingerprint_hint, so this is just a confirmation payload.
    own = coordinator._identity
    return {
        "ok": True,
        # Inbound pairs land PENDING — the receiving operator must approve
        # this peer (POST /api/fabric/peers/{id}/approve) before it can make
        # data-plane requests. Surfaced so the initiating side can tell the
        # difference between "paired and live" and "paired, awaiting approval".
        "pending_approval": not paired.fabric_share_enabled,
        "this_node": {
            "node_id": own.node_id,
            "public_key": own.public_key_b64,
            "fingerprint": own.fingerprint,
        },
        "paired": {
            "node_id": paired.node_id,
            "role": paired.role,
            "addr": paired.addr,
        },
    }


# ── Shared peer-only auth gate ───────────────────────────────────


def _require_fabric_peer(request: Request) -> dict:
    """Gate every /api/fabric/<modality> data-plane endpoint on a
    verified peer envelope + the fabric_enabled operator flag.

    Single source of truth so each new modality endpoint (inference,
    image, knowledge, render, tts, stt, …) stays consistent. Avoids
    five copies of the same branch drifting independently — the
    parallel-class-bug pattern that bit us with the empty user_id
    half-fix in 1c213d1.

    Returns the verified ``fabric_peer`` scope dict so callers can
    log ``sender_node_id`` without re-fetching it. Raises ``HTTPException``
    on either gate failure; the FastAPI route propagates it as the
    response.

    Failure semantics:
      * 403 if scope["fabric_peer"] is missing — the caller is a
        regular user / leaked URL hit / unauthenticated probe. These
        endpoints are peer-only on purpose; letting non-peers in
        would bypass the user-scoped chat handler's orchestration
        (memory, knowledge, narrative, etc.) and become an auth-bypass
        surface.
      * 503 if fabric_enabled is False — operator-disabled fabric.
        Fail-closed when the flag is off.
    """
    fabric_peer = request.scope.get("fabric_peer")
    if not fabric_peer:
        raise HTTPException(
            status_code=403,
            detail="fabric endpoint requires a verified fabric peer envelope",
        )
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    return fabric_peer


# ── HTTP: cross-peer model load coordination ─────────────────────
#
# Pre-load architecture (chosen over implicit auto-load):
#
#   A wants to dispatch a chat to model X on peer B.
#   1. A calls POST /api/fabric/load_model { model_id: "X" }
#      → B returns 202 + {status: "loading"|"ready"|"failed"}.
#        If status="loading", B kicks off manager.start(X) in a
#        background task (manager already coalesces concurrent starts
#        of the same model and swaps from a different one).
#   2. A polls GET /api/fabric/load_status?model_id=X
#      → B returns the current state. "ready" means safe to dispatch.
#        "failed" carries a reason so A can surface it to the operator.
#   3. Once A sees "ready", A POSTs /v1/chat/completions normally.
#
# Why explicit over implicit:
#   - Load failures (OOM, disk-miss) bubble distinctly from chat
#     failures; A's UI can render "[peer] is loading X" vs "X failed
#     to load" vs "chat errored." Pre-fix everything looked like a
#     generic 500.
#   - A doesn't tie up an HTTP slot waiting on a 30s cold load while
#     other requests queue behind it.
#   - Future: load can be prefetched before the user even sends the
#     turn (warm-up on model-pick).
#
# Both endpoints require fabric signed-envelope auth via
# FabricPeerMiddleware — not in _PUBLIC_PATHS, so non-peer callers
# 401. The middleware sets scope["user"] to the per-peer service
# user; handler doesn't touch it (load is a node-level op, no
# per-user state).


@router.post("/load_model")
async def fabric_load_model(request: Request) -> dict:
    """Request the receiver load ``model_id`` into its bundled engine.

    Body: ``{"model_id": "<model identifier>"}``

    Returns ``{"status": "ready" | "loading" | "failed", ...}``. The
    call is non-blocking — when a load is needed, this kicks off the
    background swap and returns immediately so the caller can poll
    ``GET /api/fabric/load_status``. Concurrent requests for the same
    model coalesce on ``LlamaServerManager.start()``'s
    ``_starting_future`` — no duplicate spawns.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    manager = getattr(request.app.state, "llama_manager", None)
    if manager is None:
        # Peer is configured with an external backend (no bundled
        # engine) — no model loading possible from this surface.
        raise HTTPException(
            status_code=503,
            detail="no bundled engine on this peer; nothing to load",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    model_id = str((body or {}).get("model_id", "") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")

    # Already serving this model? Return ready immediately.
    from augmentum.models.llama_server_manager import ProcessState
    if (
        manager.state == ProcessState.READY
        and manager.model_id == model_id
        and manager.check_alive()
    ):
        return {
            "status": "ready",
            "model_id": model_id,
            "current_model": manager.model_id,
        }

    # Resolve the model_id to a disk path. Off-loop because the
    # filesystem walk can be slow on virtiofs / 9p mounts.
    resolved_path = await asyncio.to_thread(manager._resolve_model_path, model_id)
    if not resolved_path:
        log.info(
            "fabric_load_model_not_on_disk",
            sender=request.scope.get("fabric_peer", {}).get("sender_node_id"),
            model_id=model_id,
        )
        raise HTTPException(
            status_code=404,
            detail=f"model {model_id!r} not found on this peer",
        )

    # Kick off the background load. We track the in-flight task on
    # app.state so /load_status can read its exception cleanly on
    # failure (instead of having to scrape llama-server stderr).
    load_tasks: dict = getattr(request.app.state, "_fabric_load_tasks", None)
    if load_tasks is None:
        load_tasks = {}
        request.app.state._fabric_load_tasks = load_tasks

    existing = load_tasks.get(model_id)
    if existing is not None and not existing.done():
        # Same load already in flight — coalesce (manager.start
        # internally awaits the same _starting_future anyway, but
        # tracking it here keeps /load_status accurate).
        log.info(
            "fabric_load_model_coalesced",
            model_id=model_id, current_model=manager.model_id,
        )
        return {
            "status": "loading",
            "model_id": model_id,
            "current_model": manager.model_id,
        }

    # Drop a completed-task reference so the dict doesn't grow
    # unboundedly across many distinct models.
    load_tasks.pop(model_id, None)

    log.info(
        "fabric_load_model_starting",
        model_id=model_id, current_model=manager.model_id, path=resolved_path,
    )
    task = asyncio.create_task(
        manager.start(resolved_path),
        name=f"fabric_load_{model_id}",
    )
    load_tasks[model_id] = task

    return {
        "status": "loading",
        "model_id": model_id,
        "current_model": manager.model_id,
    }


@router.get("/load_status")
async def fabric_load_status(request: Request, model_id: str = "") -> dict:
    """Poll the load state for ``model_id`` on this peer.

    Returns one of:
      - ``{"status": "ready", ...}`` — model is currently loaded
      - ``{"status": "loading", ...}`` — load is in flight
      - ``{"status": "failed", "reason": "..."}`` — most recent load
        attempt failed; the reason comes from the background task's
        exception
      - ``{"status": "unknown", ...}`` — no load tracked for this
        model and the manager isn't currently serving it (caller
        should POST /load_model first)

    The caller polls this with backoff until ``ready`` or ``failed``,
    or until its own timeout fires.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    manager = getattr(request.app.state, "llama_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="no bundled engine on this peer")

    model_id = str(model_id or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id query param required")

    from augmentum.models.llama_server_manager import ProcessState
    from augmentum.models.load_progress import (
        build_load_progress_payload,
        build_prefill_progress_payload,
    )

    def _progress() -> dict:
        """Cross-peer state transparency: embed this peer's live
        model-load + prefill snapshots so the originating peer can drive
        the SAME progress bars a local load would. Shapes match the
        local /api/engine/v2/{load,prefill}_progress endpoints exactly
        (shared builders), so the originator surfaces them through those
        very endpoints with no UI change.
        """
        out: dict = {}
        load = build_load_progress_payload(getattr(manager, "_load_progress", None))
        if load.get("active"):
            out["load_progress"] = load
        prefill = build_prefill_progress_payload(
            getattr(manager, "_prefill_progress", None)
        )
        if prefill.get("active"):
            out["prefill_progress"] = prefill
        return out

    # Currently serving the requested model? Always wins regardless
    # of what the task dict says. (Prefill may still be in flight for a
    # long prompt, so carry progress here too.)
    if (
        manager.state == ProcessState.READY
        and manager.model_id == model_id
        and manager.check_alive()
    ):
        return {
            "status": "ready",
            "model_id": model_id,
            "current_model": manager.model_id,
            **_progress(),
        }

    load_tasks: dict = getattr(request.app.state, "_fabric_load_tasks", None) or {}
    task = load_tasks.get(model_id)
    if task is None:
        return {
            "status": "unknown",
            "model_id": model_id,
            "current_model": manager.model_id,
        }

    if not task.done():
        return {
            "status": "loading",
            "model_id": model_id,
            "current_model": manager.model_id,
            **_progress(),
        }

    # Task completed — surface success/failure.
    exc = task.exception()
    if exc is not None:
        reason = f"{type(exc).__name__}: {str(exc)[:300]}"
        log.warning(
            "fabric_load_status_failed",
            model_id=model_id, reason=reason,
        )
        return {
            "status": "failed",
            "model_id": model_id,
            "current_model": manager.model_id,
            "reason": reason,
        }

    # Task completed without exception. Recheck the manager — usually
    # this means we're now serving the requested model.
    if (
        manager.state == ProcessState.READY
        and manager.model_id == model_id
    ):
        return {
            "status": "ready",
            "model_id": model_id,
            "current_model": manager.model_id,
        }

    # Task succeeded but manager moved on (operator-triggered swap
    # between completion and our read). Treat as ready-then-superseded.
    return {
        "status": "superseded",
        "model_id": model_id,
        "current_model": manager.model_id,
    }


# ── HTTP: cross-peer LLM dispatch (data plane) ────────────────────
#
# Purpose-built fabric inference endpoint. Distinct from /v1/chat/
# completions for one critical reason: the OpenAI chat handler is
# orchestration-heavy (mode classification, session resolution, memory
# recall, knowledge pack eval, narrative state, tools, post-stream
# hooks). All of that depends on subsystems (embedder, pack DB,
# memory store, narrative tables) that the initiating peer ALREADY
# ran on its side before dispatching. Re-running them on the receiver
# is redundant, introduces failure modes that have nothing to do with
# the LLM compute, and crashed cross-peer chats in the 2026-05-23
# incident when the receiver's embedder cache was incomplete.
#
# This endpoint is pure compute:
#   - Resolves the LOCAL backend (recursion guard against A→B→C→…
#     fan-out loops).
#   - Streams the LLM response back as OpenAI-format SSE.
#   - Skips ALL orchestration. The sender's `FabricBackend` posts
#     pre-orchestrated message lists; this endpoint just runs them.
#
# Auth: required scope["fabric_peer"] from FabricPeerMiddleware.
# Non-peer callers 403 (so a leaked URL can't be used as an
# auth-bypass for chat dispatch).
#
# Streaming uses raw ASGI send() instead of Starlette's
# StreamingResponse — its silent-exit behavior on unhandled
# generator exceptions was the source of the 2026-05-23 debug
# nightmare ("ASGI callable returned without completing response"
# with zero traceback). Raw ASGI lets us catch BaseException
# subclasses and emit a final SSE error chunk so the sender always
# sees a typed error instead of a half-stream.


def _internal_chunk_to_openai_sse_dict(chunk, chunk_id: str, fallback_model: str) -> dict:
    """Convert InternalStreamChunk → OpenAI chat.completion.chunk dict.

    Mirrors the shape produced by ``augmentum.proxy.streaming._chunk_to_openai_sse``
    but lives here so this endpoint is self-contained (no import-cycle
    risk with the heavier streaming.py module). Includes thinking as
    ``reasoning_content`` for compatibility with the sender's existing
    SSE parser in fabric_backend.py.
    """
    import time as _time
    delta: dict = {}
    if chunk.role:
        delta["role"] = chunk.role
    if chunk.content_delta:
        delta["content"] = chunk.content_delta
    # InternalStreamChunk's field is ``thinking_delta`` — this
    # previously read ``chunk.thinking`` (always-None getattr), so
    # reasoning never crossed the wire and thinking models routed
    # via fabric showed no reasoning on the sender. Keep the legacy
    # getattr as a fallback for any custom backend that still sets
    # ``thinking``.
    thinking_delta = chunk.thinking_delta or getattr(chunk, "thinking", None)
    if thinking_delta:
        delta["reasoning_content"] = thinking_delta
    # Native tool-call deltas ride in chunk.augmentum (the convention
    # set by OpenAICompatBackend / LlamaCppBackend stream parsers).
    # Forward them as OpenAI ``delta.tool_calls`` so the sender's
    # parser can hand them to its chat_egress accumulator.
    if chunk.augmentum and chunk.augmentum.get("tool_calls"):
        delta["tool_calls"] = chunk.augmentum["tool_calls"]
    result: dict = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(_time.time()),
        "model": chunk.model or fallback_model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": chunk.finish_reason,
        }],
    }
    if chunk.done and chunk.usage:
        result["usage"] = {
            "prompt_tokens": chunk.usage.prompt_tokens,
            "completion_tokens": chunk.usage.completion_tokens,
            "total_tokens": chunk.usage.total_tokens,
        }
    # Cross-peer state transparency (prefill): forward the augmentum
    # stage/status metadata as a top-level ``augmentum`` block, mirroring
    # augmentum.proxy.streaming._chunk_to_openai_sse. The receiver's local
    # backend emits a ``prefill`` stage_start BEFORE it blocks on prompt
    # processing, so this rides the SSE to the sender ahead of the silent
    # prefill wait — letting the sender relay it and start the SAME prefill
    # progress bar + suspend the SAME stall watchdog a local prefill shows.
    # ``tool_calls`` already rode in ``delta.tool_calls`` above; strip it
    # here so it isn't double-forwarded.
    if chunk.augmentum:
        aug = {k: v for k, v in chunk.augmentum.items() if k != "tool_calls"}
        if aug:
            result["augmentum"] = aug
    return result


@router.post("/inference")
async def fabric_inference(request: Request):
    """Cross-peer LLM dispatch — pure compute, no orchestration.

    Body (OpenAI-compat-shaped subset):
      ``{
            "model": "...",
            "messages": [{"role": "user", "content": "...",
                          // optional per-message: tool_call_id,
                          // tool_calls, images, thinking
                         }],
            "stream": true,
            "temperature": 0.7,    // optional — likewise top_p,
                                   // max_tokens, stop, seed,
                                   // frequency_penalty, presence_penalty
            "tools": [...],        // optional native function calling
            "tool_choice": "auto", // optional
            "format": "json",      // optional
            "think": false,        // explicit thinking toggle (always sent)
            "chat_template_kwargs": {...},   // optional
            "preserve_thinking": true,       // optional (Qwen 3.6)
            "reasoning_effort": "high",      // optional (OpenAI family)
            "raw_options": {...},  // optional sampler passthrough
            "continue_last_assistant": true, // optional (Continue button)
            "is_background_task": true,      // optional slot-routing hint
            "session_id": "s_..."  // optional, for KV continuity
        }``

    Returns:
      - ``stream=false``: JSON OpenAI chat completion response.
      - ``stream=true``: SSE event stream of OpenAI chat.completion.chunk
        events terminated by ``data: [DONE]\\n\\n``.

    Errors:
      - 403 if not a verified fabric peer.
      - 503 if fabric is disabled.
      - 400 on malformed body.
      - 404 if model isn't in the local model map (no fan-out to
        other peers — this is the recursion guard).
      - 500 with typed error chunk in the SSE stream on backend
        failure (so the sender sees ``PeerProtocolError`` instead of
        a generic dropped-connection error).
    """
    import json as _json
    import uuid as _uuid

    fabric_peer = _require_fabric_peer(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    model = str(body.get("model", "") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model required")

    messages_raw = body.get("messages") or []
    if not isinstance(messages_raw, list) or not messages_raw:
        raise HTTPException(status_code=400, detail="non-empty messages list required")

    from augmentum.models.base import InternalChatRequest, Message
    messages: list = []
    for m in messages_raw:
        if not isinstance(m, dict):
            continue
        msg_kwargs: dict = {
            "role": str(m.get("role", "user")),
            "content": str(m.get("content", "") or ""),
        }
        tool_call_id = m.get("tool_call_id")
        if tool_call_id:
            msg_kwargs["tool_call_id"] = str(tool_call_id)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            msg_kwargs["tool_calls"] = tool_calls
        images = m.get("images")
        if isinstance(images, list) and images:
            msg_kwargs["images"] = images
        thinking = m.get("thinking")
        if thinking:
            msg_kwargs["thinking"] = str(thinking)
        messages.append(Message(**msg_kwargs))

    # Reconstruct the sender's user-preference surface. Each field is
    # type-gated rather than blindly trusted — the peer is signed, but
    # a version-skewed or buggy sender shouldn't be able to crash this
    # endpoint with a string where a dict belongs. Anything absent or
    # mis-typed falls back to the local default, which matches the
    # pre-forwarding behavior. The local backend applies its own
    # whitelists downstream (e.g. raw_options keys in llama_cpp.py),
    # so this is shape validation, not policy.
    def _opt_dict(key: str) -> dict | None:
        v = body.get(key)
        return v if isinstance(v, dict) and v else None

    def _opt_list(key: str) -> list | None:
        v = body.get(key)
        return v if isinstance(v, list) and v else None

    stop = _opt_list("stop")
    tool_choice = body.get("tool_choice")
    if not isinstance(tool_choice, str | dict):
        tool_choice = None
    fmt = body.get("format")
    reasoning_effort = body.get("reasoning_effort")
    preserve_thinking = body.get("preserve_thinking")

    internal_req = InternalChatRequest(
        model=model,
        messages=messages,
        stream=bool(body.get("stream", False)),
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=body.get("max_tokens"),
        stop=[str(s) for s in stop] if stop else None,
        frequency_penalty=body.get("frequency_penalty"),
        presence_penalty=body.get("presence_penalty"),
        seed=body.get("seed") if isinstance(body.get("seed"), int) else None,
        tools=_opt_list("tools"),
        tool_choice=tool_choice,
        format=str(fmt) if isinstance(fmt, str) and fmt else None,
        # Explicit boolean both ways: think=False must reach the local
        # backend as an explicit ``enable_thinking: false`` for
        # always-on families (Qwen 3.x) — see llama_cpp.py's
        # _chat_template_kwargs.
        think=bool(body.get("think", False)),
        chat_template_kwargs=_opt_dict("chat_template_kwargs"),
        preserve_thinking=(
            bool(preserve_thinking) if preserve_thinking is not None else None
        ),
        reasoning_effort=(
            str(reasoning_effort)
            if isinstance(reasoning_effort, str) and reasoning_effort
            else None
        ),
        raw_options=_opt_dict("raw_options"),
        continue_last_assistant=bool(body.get("continue_last_assistant", False)),
        is_background_task=bool(body.get("is_background_task", False)),
    )

    registry = getattr(request.app.state, "provider_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Provider registry unavailable (server still starting)")
    backend, clean_model = await registry.resolve_backend_for_model(model)
    # ``resolve_backend_for_model`` ALWAYS returns a backend (falls back
    # to default when no map match) — bare ``is None`` never fires. We
    # need the explicit local_known gate to refuse models the receiver
    # doesn't actually have, otherwise the default backend would
    # silently dispatch with the wrong model name.
    #
    # Two-key check (same rationale as provider_registry.py's check —
    # see the long-form comment there): multi-provider models live in
    # the map ONLY under their disambiguated ``model@backend`` keys,
    # so the clean-name lookup alone silently classifies every
    # disambiguated peer-requested model as "not local" even when the
    # @-suffix path correctly resolved a specific backend. Accept the
    # original requested name as proof-of-locality.
    local_known = (
        clean_model in registry._model_map  # noqa: SLF001
        or (model and model in registry._model_map)  # noqa: SLF001
    )
    if backend is None or not local_known:
        log.warning(
            "fabric_inference_model_unavailable",
            model=model,
            peer=fabric_peer.get("sender_node_id", ""),
            local_known=local_known,
        )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"error": {
                "message": f"model {model!r} not available on this peer",
                "type": "model_unavailable",
                "model": model,
            }},
        )
    internal_req.model = clean_model

    # KV-cache continuity: optional session_id lets the sender ride
    # the same engine slot across multiple turns of the same
    # conversation.
    #
    # ISOLATION: the slot key MUST be namespaced by the sender's node
    # id. ``session_id`` is a sender-minted opaque string (e.g.
    # ``s_<uuid>``) — two distinct peers (or a peer and a LOCAL chat)
    # can independently mint the same value, and ``llama_cpp.py``'s
    # ``_session_fingerprint`` uses ``kv_session_key`` verbatim as the
    # save/restore slot affinity key. Without the prefix, peer A's
    # conversation prefix would be restored into peer B's turn — a
    # cross-tenant context leak below the user_id layer (same failure
    # class as the system-message-hash collision removed from
    # ``_session_fingerprint``). The ``fabric:<node>:`` prefix is
    # unforgeable here because ``sender_node_id`` comes from the
    # signature-verified peer envelope, not the request body.
    session_id = str(body.get("session_id", "") or "").strip()
    if session_id:
        sender_node_id = str(fabric_peer.get("sender_node_id", "") or "")
        internal_req.kv_session_key = (
            f"fabric:{sender_node_id}:{session_id}" if sender_node_id
            else f"fabric:{session_id}"
        )
        internal_req.kv_mode = "fabric"

    log.info(
        "fabric_inference_dispatch",
        peer=fabric_peer.get("sender_node_id", ""),
        model=clean_model,
        stream=internal_req.stream,
        msg_count=len(messages),
        backend_type=type(backend).__name__,
        kv_session=session_id or "(none)",
    )

    # ── Non-streaming path ────────────────────────────────────────
    if not internal_req.stream:
        from fastapi.responses import JSONResponse

        from augmentum.models.openai_compat import to_openai_chat_response
        try:
            internal_resp = await backend.chat(internal_req)
            if not internal_resp.model:
                internal_resp.model = clean_model
            return JSONResponse(content=to_openai_chat_response(internal_resp))
        except Exception as exc:
            log.error(
                "fabric_inference_nonstream_failed",
                model=clean_model, error=str(exc)[:300], exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={"error": {
                    "message": str(exc)[:300],
                    "type": "backend_error",
                    "exc_type": type(exc).__name__,
                }},
            )

    # ── Streaming path: raw ASGI ──────────────────────────────────
    # We deliberately don't use Starlette's StreamingResponse here —
    # its silent-exit behavior on unhandled body-iterator exceptions
    # was the source of the 2026-05-23 fabric debug nightmare. Raw
    # ASGI gives us explicit control over every send() and lets us
    # catch BaseException subclasses (CancelledError, GeneratorExit)
    # with a visible log + final error chunk emission.
    chunk_id = f"chatcmpl-{_uuid.uuid4().hex[:12]}"

    async def _asgi_streamer(scope, receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/event-stream"),
                (b"cache-control", b"no-cache, no-store"),
                (b"connection", b"keep-alive"),
                (b"x-accel-buffering", b"no"),
            ],
        })

        chunks_sent = 0
        sent_done = False
        try:
            async for chunk in backend.chat_stream(internal_req):
                sse = _internal_chunk_to_openai_sse_dict(chunk, chunk_id, clean_model)
                payload = f"data: {_json.dumps(sse)}\n\n".encode()
                await send({
                    "type": "http.response.body",
                    "body": payload,
                    "more_body": True,
                })
                chunks_sent += 1
                if chunk.done:
                    await send({
                        "type": "http.response.body",
                        "body": b"data: [DONE]\n\n",
                        "more_body": True,
                    })
                    sent_done = True
                    break
            # Backend returned without a done=True chunk — terminate
            # the SSE stream anyway so the sender's parser exits
            # cleanly.
            if not sent_done:
                await send({
                    "type": "http.response.body",
                    "body": b"data: [DONE]\n\n",
                    "more_body": True,
                })
            # Final terminator (more_body=False is what uvicorn needs
            # to know the response is complete).
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })
            log.info(
                "fabric_inference_complete",
                chunks=chunks_sent, model=clean_model,
            )
        except asyncio.CancelledError:
            # Caller cancelled (sender dropped, peer task cancelled
            # via WS backstop). Best-effort send a done marker so the
            # connection drops cleanly. Don't suppress — let the
            # cancellation propagate up so the middleware's finally
            # block fires.
            log.info(
                "fabric_inference_cancelled",
                chunks_sent=chunks_sent, model=clean_model,
            )
            raise
        except Exception as exc:
            log.error(
                "fabric_inference_stream_error",
                exc_type=type(exc).__name__,
                error=str(exc)[:300],
                chunks_sent=chunks_sent,
                model=clean_model,
                exc_info=True,
            )
            # Best-effort: emit a typed error chunk + done terminator
            # so the sender's SSE parser sees a clean error instead of
            # an EOF that bubbles as a generic "incomplete chunked
            # read" backend error.
            try:
                err_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": clean_model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error",
                    }],
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc)[:300],
                    },
                }
                await send({
                    "type": "http.response.body",
                    "body": f"data: {_json.dumps(err_chunk)}\n\n".encode(),
                    "more_body": True,
                })
                await send({
                    "type": "http.response.body",
                    "body": b"data: [DONE]\n\n",
                    "more_body": True,
                })
                await send({
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                })
            except Exception:
                # Sender already disconnected — nothing to do. The
                # original error is already logged above.
                log.debug("fabric_inference_error_send_failed", exc_info=True)
        except BaseException as exc:
            # Last-resort: BaseException subclasses (SystemExit,
            # GeneratorExit, KeyboardInterrupt). These are normally
            # fatal but logging the type before re-raise is the only
            # way to diagnose silent process-level failures.
            log.warning(
                "fabric_inference_base_exception",
                exc_type=type(exc).__name__,
                chunks_sent=chunks_sent,
            )
            raise

    # Wrap the ASGI streamer in a minimal Response subclass so FastAPI
    # accepts it as a route return value. Starlette's Response.__call__
    # is what FastAPI invokes, so we just override it.
    from starlette.responses import Response

    class _RawASGIResponse(Response):
        """Raw-ASGI streaming response. Bypasses parent ``Response``
        machinery — we manage status/headers/body sends ourselves
        inside ``_asgi_streamer`` so we can catch BaseException and
        emit a final SSE error chunk."""

        def __init__(self) -> None:
            # Skip super().__init__ — we don't have a precomputed body
            # and the parent would try to set Content-Length.
            self.status_code = 200
            self.background = None
            self.body = b""
            self.raw_headers = []

        async def __call__(self, scope, receive, send) -> None:
            await _asgi_streamer(scope, receive, send)

    return _RawASGIResponse()


# ── HTTP: cross-peer knowledge search (data plane) ───────────────


@router.post("/knowledge/search")
async def fabric_knowledge_search(request: Request):
    """Cross-peer knowledge pack search — LOCAL packs only.

    Body:
      ``{"q": "...", "pack_ids": ["pack_a", "pack_b"], "limit": 8}``

    Returns:
      ``{"query": "...", "pack_ids": [...], "results": [...]}``

    Critical recursion guard: this endpoint searches ONLY the local
    pack_manager's installed packs. If a requested pack_id isn't on
    this peer, it's silently skipped — we MUST NOT fan out to another
    peer (which would open A→B→C→A loops with no hop-count protection).
    Subagent-flagged HIGH-severity risk from the architecture review.

    Auth: verified fabric peer only.
    """

    _require_fabric_peer(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    query = str(body.get("q", "") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="q required")

    pack_ids_raw = body.get("pack_ids") or []
    if not isinstance(pack_ids_raw, list):
        raise HTTPException(status_code=400, detail="pack_ids must be a list")
    requested_pack_ids = [str(p) for p in pack_ids_raw if p]

    try:
        limit = int(body.get("limit", 8))
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(50, limit))

    pack_mgr = getattr(request.app.state, "pack_manager", None)
    if pack_mgr is None:
        # No local pack manager → no packs to search. Return empty
        # results rather than 404 so the caller's downstream merge
        # logic stays simple.
        return {"query": query, "pack_ids": requested_pack_ids, "results": []}

    # Filter to packs that actually exist locally. Pre-filter (vs
    # passing all and getting empty per-pack results) so we don't
    # silently waste search time on packs we don't have.
    try:
        # PackManager.installed is a @property (a list), NOT a method —
        # calling it as installed() raised TypeError every time, which the
        # except swallowed to [], so the pre-filter below dropped EVERY
        # requested pack_id and fabric knowledge search always returned 0
        # results regardless of which packs were installed.
        installed = pack_mgr.installed or []
    except Exception:
        installed = []
    local_pack_ids_set = {str(p.get("pack_id", "")) for p in installed if isinstance(p, dict)}
    pack_ids = [pid for pid in requested_pack_ids if pid in local_pack_ids_set]

    if not pack_ids:
        log.info(
            "fabric_knowledge_search_no_local_packs",
            requested=len(requested_pack_ids),
            installed=len(local_pack_ids_set),
        )
        return {"query": query, "pack_ids": requested_pack_ids, "results": []}

    try:
        results = await pack_mgr.search(query, pack_ids=pack_ids, limit=limit)
    except Exception as exc:
        log.warning(
            "fabric_knowledge_search_failed",
            error=str(exc)[:200], exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"local pack search failed: {str(exc)[:200]}",
        ) from None

    # Coerce to JSON-safe shape. Result objects may be dataclasses or
    # dicts; downstream callers expect dicts.
    serialised: list[dict] = []
    for r in results or []:
        if isinstance(r, dict):
            serialised.append(r)
        elif hasattr(r, "__dict__"):
            serialised.append({k: v for k, v in r.__dict__.items() if not k.startswith("_")})

    return {
        "query": query,
        "pack_ids": pack_ids,
        "results": serialised,
    }


# ── HTTP: cross-peer cast/render (data plane) ────────────────────


@router.post("/render")
async def fabric_render(request: Request):
    """Cross-peer render dispatch.

    Body (JSON form of ``RenderJob``):
      ``{"kind": "...", "target_device_id": "...", "payload": {...}}``

    Returns: ``RenderResult`` JSON shape (``{ok, location, node_id,
    output_url, code, message, metadata}``).

    The existing ``/api/cast/render`` endpoint already accepts fabric
    peer requests via the per-peer service user. This dedicated
    fabric endpoint moves the dispatch off the user-facing path so
    operator-side route changes (rate limits, content gates, audit
    logging) don't accidentally affect cross-peer traffic. Render
    is already pure compute on the receiver — no orchestration debt
    to strip.

    Never raises a 5xx for the dispatch itself: ``RenderResult.ok=False``
    is the failure signal, matching the convention in render_client.py.
    """
    fabric_peer = _require_fabric_peer(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    kind = str(body.get("kind", "") or "").strip()
    if not kind:
        raise HTTPException(status_code=400, detail="kind required")

    target_device_id = str(body.get("target_device_id", "") or "")
    payload = body.get("payload") or {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")

    from augmentum.cast.executors import execute_local_render
    from augmentum.cast.render import RenderJob, RenderResult
    job = RenderJob(
        kind=kind,
        target_device_id=target_device_id,
        payload=payload,
    )

    log.info(
        "fabric_render_dispatch",
        peer=fabric_peer.get("sender_node_id", ""),
        kind=kind,
        target=target_device_id,
    )

    # Match cast_routes.render_inbound's render-call signature so this
    # endpoint behaves identically to the regular render path on the
    # receiver. html_renderer + output_store come from lifespan; absent
    # values are OK — execute_local_render falls through to a stubbed
    # result instead of raising.
    html_renderer = getattr(request.app.state, "html_renderer", None)
    output_store = getattr(request.app.state, "render_output_store", None)
    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    own_node_id = ""
    if coordinator is not None:
        own_node_id = getattr(coordinator._identity, "node_id", "")  # noqa: SLF001
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""

    try:
        result = await execute_local_render(
            job,
            node_id=own_node_id,
            user_id=user_id,
            html_renderer=html_renderer,
            output_store=output_store,
        )
    except Exception as exc:
        log.warning("fabric_render_failed", error=str(exc)[:200], exc_info=True)
        result = RenderResult(
            ok=False,
            location="peer",
            node_id=own_node_id,
            code="render_failed",
            message=str(exc)[:300],
        )

    from dataclasses import asdict
    return asdict(result)


# ── HTTP: cross-peer image generation (data plane) ───────────────


@router.post("/image/generate")
async def fabric_image_generate(request: Request):
    """Cross-peer image generation — pure pipeline call.

    Body:
      ``{"model": "...", "prompt": "...", "negative_prompt": "...",
         "width": 1024, "height": 1024, "steps": 30,
         "guidance_scale": 7.5, "seed": null, ...}``

    Returns: ``multipart/mixed`` — JSON metadata header + binary image
    bytes, in a single request/response. Collapses the previous
    two-step (POST generate, GET fetch) into one round-trip.

    Receiver dispatches directly to the local image_pipeline_registry.
    Skips the regular endpoint's prompt enhancement, safety filters,
    library writes — A handles all of that on its side.
    """
    fabric_peer = _require_fabric_peer(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    model = str(body.get("model", "") or "").strip()
    prompt = str(body.get("prompt", "") or "")
    if not model or not prompt:
        raise HTTPException(status_code=400, detail="model and prompt required")

    pipeline_registry = getattr(request.app.state, "image_pipeline_registry", None)
    if pipeline_registry is None:
        raise HTTPException(
            status_code=503,
            detail="no image pipeline registry on this peer",
        )

    log.info(
        "fabric_image_dispatch",
        peer=fabric_peer.get("sender_node_id", ""),
        model=model,
    )

    # Build the GenerateRequest the local pipeline expects. We
    # construct the pydantic model from the body dict so any future
    # schema additions get carried through; on unknown fields the
    # model's ``model_config`` decides whether to error or ignore.
    from augmentum.image.schemas import GenerateRequest
    try:
        gen_request = GenerateRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"invalid GenerateRequest: {str(exc)[:200]}",
        ) from None

    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence is None:
        raise HTTPException(
            status_code=503,
            detail="no image_persistence on this peer (cannot resolve model name)",
        )

    # Explicit capability check BEFORE the call. This peer advertised
    # image.generation in its heartbeat, but a build predating the fabric
    # helper can't serve it — surface that precisely instead of letting a
    # missing method look like an internal crash. Checking by hasattr
    # (rather than catching AttributeError around the call) is the load-
    # bearing fix: a broad ``except AttributeError`` also swallows
    # AttributeErrors raised INSIDE generate_for_fabric and mislabels a
    # real bug as "not wired" → an unactionable 501.
    if not hasattr(pipeline_registry, "generate_for_fabric"):
        log.error(
            "fabric_image_registry_helper_missing",
            peer=fabric_peer.get("sender_node_id", ""),
        )
        raise HTTPException(
            status_code=501,
            detail=(
                "this peer is running an older Augmentum build without "
                "cross-peer image generation — rebuild the peer to enable it"
            ),
        )

    try:
        # Slim path: skip queue, preset application, cache, library
        # write. Returns (image_bytes, metadata_dict). The sender will
        # mint its own local image_id when it writes the bytes to its
        # own library.
        image_bytes, metadata = await pipeline_registry.generate_for_fabric(
            gen_request, persistence=persistence,
        )
    except ValueError as exc:
        # ValueError from generate_for_fabric: model not in this
        # peer's image_models table. Treat as 404 so the sender's
        # capability invalidation path can drop the stale ad.
        log.info(
            "fabric_image_model_unavailable",
            peer=fabric_peer.get("sender_node_id", ""),
            model=model, reason=str(exc)[:200],
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc)[:200],
        ) from None
    except Exception as exc:
        log.warning(
            "fabric_image_generate_failed", error=str(exc)[:200], exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"image generation failed: {str(exc)[:200]}",
        ) from None

    # Single-shot multipart response: JSON metadata + raw bytes.
    # Caller (image_client.py) parses both halves.
    import uuid as _uuid
    boundary = f"--augmentumfabric{_uuid.uuid4().hex}"
    parts: list[bytes] = []

    metadata_json = json.dumps(metadata or {}, separators=(",", ":")).encode("utf-8")
    parts.append(f"--{boundary}\r\n".encode("latin-1"))
    parts.append(b"Content-Type: application/json\r\n")
    parts.append(b"Content-Disposition: form-data; name=\"metadata\"\r\n\r\n")
    parts.append(metadata_json)
    parts.append(b"\r\n")

    parts.append(f"--{boundary}\r\n".encode("latin-1"))
    parts.append(b"Content-Type: application/octet-stream\r\n")
    parts.append(b"Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n\r\n")
    parts.append(image_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("latin-1"))

    body_bytes = b"".join(parts)
    from fastapi.responses import Response as _Response
    return _Response(
        content=body_bytes,
        media_type=f"multipart/form-data; boundary={boundary}",
    )


# ── HTTP: cross-peer TTS (data plane) ────────────────────────────


@router.post("/tts")
async def fabric_tts(request: Request):
    """Cross-peer text-to-speech — local engine only.

    Body (TTSRequest shape):
      ``{"model": "...", "input": "...", "voice": "...",
         "response_format": "mp3", "speed": 1.0, "instructions": "..."}``

    Returns: streaming audio bytes with the matching media_type
    (audio/mpeg / audio/wav / audio/opus etc.).

    Receiver dispatches to the local TTS provider chain (Kokoro,
    Pocket TTS, external HTTPS providers). The fabric-provider branch
    that exists in tts_speech for the user-facing /v1/audio/speech
    path is explicitly REFUSED here — if the receiver's default
    voice resolved to a fabric provider, we 404 instead of recursing
    into another A→B→C dispatch loop.

    Auth: verified fabric peer only.
    """
    fabric_peer = _require_fabric_peer(request)

    try:
        body_raw = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body_raw, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    input_text = str(body_raw.get("input", "") or "").strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="input required")

    # Build the same TTSRequest model the user-facing handler expects
    # so the synth pipeline + format defaults stay consistent.
    from augmentum.proxy.audio_routes import (
        _FABRIC_PROVIDER_PREFIX,
        TTSRequest,
        _get_conn,
        resolve_voice_provider,
        tts_speech,
    )

    tts_body = TTSRequest(
        model=str(body_raw.get("model", "") or ""),
        input=input_text,
        voice=str(body_raw.get("voice", "") or ""),
        response_format=str(body_raw.get("response_format", "mp3") or "mp3"),
        speed=float(body_raw.get("speed", 1.0) or 1.0),
        instructions=body_raw.get("instructions"),
        # Carry the sender's conversation id through to the local provider
        # (re-attached as X-Augmentum-Session by the dispatch below) so a
        # context-aware engine on this peer — e.g. Sesame CSM — conditions
        # prosody on the same conversation's prior turns. See
        # fabric/audio_client.py::tts_stream_via_peer.
        session_id=str(body_raw.get("session_id", "") or ""),
    )

    # Recursion guard: refuse to dispatch if the resolved provider on
    # THIS peer is itself fabric (would create A→B→C fan-out). Same
    # shape as the knowledge endpoint's local-only gate.
    conn = _get_conn(request)
    if conn is not None:
        try:
            provider, _ = await resolve_voice_provider(conn, tts_body.voice or "")
        except Exception:
            provider = None
        if provider and str(provider.get("id", "")).startswith(_FABRIC_PROVIDER_PREFIX):
            log.warning(
                "fabric_tts_recursion_refused",
                peer=fabric_peer.get("sender_node_id", ""),
                provider_id=provider.get("id", ""),
            )
            raise HTTPException(
                status_code=404,
                detail=(
                    f"voice {tts_body.voice!r} resolves to a fabric provider "
                    "on this peer; cross-peer recursion is not supported"
                ),
            )

    log.info(
        "fabric_tts_dispatch",
        peer=fabric_peer.get("sender_node_id", ""),
        voice=tts_body.voice or "(default)",
        format=tts_body.response_format,
        text_len=len(input_text),
    )

    # Delegate to the user-facing handler. It returns a StreamingResponse
    # directly; FastAPI streams its body iterator back to the sender.
    # The recursion guard above ensured we won't take the fabric branch
    # inside tts_speech — provider is local-only at this point.
    return await tts_speech(tts_body, request)


# ── HTTP: cross-peer STT (data plane) ────────────────────────────


@router.post("/stt")
async def fabric_stt(request: Request):
    """Cross-peer speech-to-text — local engine only.

    Accepts ``multipart/form-data`` with a ``file`` field containing
    the audio payload, identical to the user-facing
    ``/v1/audio/transcriptions`` endpoint.

    Returns: ``{"text": "..."}`` JSON.

    Receiver dispatches to the local STT provider (Moonshine,
    Deepgram, etc.). Like the TTS endpoint, this refuses any
    receiver-side provider that would itself fabric-dispatch
    (no A→B→C recursion).
    """
    fabric_peer = _require_fabric_peer(request)

    # Pull the uploaded file out of the multipart form. We do this
    # via Request directly (rather than a typed UploadFile dep) so
    # the auth gate at the top of the handler runs before FastAPI
    # tries to parse the form — gives us clean 403 for non-peer
    # callers without consuming the body.
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid multipart body") from None
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="file field required")

    from augmentum.proxy.audio_routes import (
        _get_conn,
        _get_default_provider,
        stt_transcribe,
    )

    # Recursion guard. STT providers don't have the same "fabric:"
    # prefix as TTS — they're just rows in the audio_providers table.
    # The receiver's _get_default_provider returns the configured
    # default; if it's flagged as a fabric-dispatch provider, refuse.
    conn = _get_conn(request)
    if conn is not None:
        try:
            provider = await _get_default_provider(conn, "stt")
        except Exception:
            provider = None
        if provider:
            base_url = str(provider.get("base_url", ""))
            if "fabric" in str(provider.get("id", "")).lower() or "fabric:" in base_url:
                log.warning(
                    "fabric_stt_recursion_refused",
                    peer=fabric_peer.get("sender_node_id", ""),
                    provider_id=provider.get("id", ""),
                )
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "default STT provider on this peer is fabric-routed; "
                        "cross-peer recursion is not supported"
                    ),
                )

    log.info(
        "fabric_stt_dispatch",
        peer=fabric_peer.get("sender_node_id", ""),
        filename=getattr(upload, "filename", "") or "",
    )

    return await stt_transcribe(request, upload)


@router.post("/voice-clone")
async def fabric_voice_clone(request: Request) -> dict:
    """Receive a voice-clone reference (clip + transcript) from a peer.

    Accepts ``multipart/form-data`` with ``file`` (audio), ``voice_name``,
    and optional ``transcript``. Writes them into THIS node's shared voice
    dir, where its co-located CSM sidecar finds them — bridging the clone
    across the fabric for context-aware engines that can't see the sender's
    ``/voices`` volume. See ``fabric/audio_client.py::clone_upload_via_peer``.

    Auth: verified fabric peer only. The voice name is sanitised before it
    touches the filesystem (no path traversal from a peer-supplied name).
    """
    fabric_peer = _require_fabric_peer(request)

    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid multipart body") from None
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="file field required")
    voice_name = str(form.get("voice_name", "") or "").strip()
    if not voice_name:
        raise HTTPException(status_code=400, detail="voice_name required")
    transcript = str(form.get("transcript", "") or "")

    audio_bytes = await upload.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")
    if len(audio_bytes) > 10 * 1024 * 1024:  # mirror the user-facing 10 MB cap
        raise HTTPException(status_code=413, detail="audio too large (max 10 MB)")

    from augmentum.proxy.audio_routes import save_voice_clone_files

    safe, saved = save_voice_clone_files(
        voice_name, audio_bytes,
        filename=getattr(upload, "filename", "") or "",
        transcript=transcript,
    )
    log.info(
        "fabric_voice_clone_received",
        peer=fabric_peer.get("sender_node_id", ""),
        voice_name=safe,
        file=saved,
        size=len(audio_bytes),
        has_transcript=bool(transcript.strip()),
    )
    return {"status": "ok", "voice_name": safe, "file": saved}


@router.post("/tts/user-context")
async def fabric_user_context(request: Request) -> dict:
    """Receive the USER's spoken turn from a peer and hand it to THIS node's
    CSM sidecar as cross-speaker context (so her next reply reacts to how
    they sounded). Unlike the clone bridge, this context lives in the
    sidecar's RAM, so we forward it to the local sidecar rather than writing
    a file.

    Body: ``multipart/form-data`` with ``file`` (audio), ``session_id``,
    optional ``transcript``. Auth: verified fabric peer only. No-op (200)
    if this node has no CSM sidecar configured — the sender shouldn't have
    routed here, but a clean ack beats a 500 on a best-effort channel."""
    _require_fabric_peer(request)

    from augmentum.config import settings as _settings
    sidecar_url = (_settings.tts_sesame_csm_url or "").rstrip("/")
    if not sidecar_url:
        return {"status": "skipped", "reason": "no csm sidecar on this node"}

    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid multipart body") from None
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="file field required")
    session_id = str(form.get("session_id", "") or "").strip()
    transcript = str(form.get("transcript", "") or "")
    audio_bytes = await upload.read()
    if not session_id or not audio_bytes:
        return {"status": "skipped", "reason": "missing session or audio"}

    # Forward to the local sidecar's cross-speaker endpoint. Plain http on
    # the docker network; the session id rides as the X-Augmentum-Session
    # header the sidecar keys context off.
    fname = getattr(upload, "filename", "") or "user_turn.wav"
    ctype = getattr(upload, "content_type", "") or "audio/wav"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{sidecar_url}/v1/context/user_turn",
                files={"audio": (fname, audio_bytes, ctype)},
                data={"transcript": transcript},
                headers={"X-Augmentum-Session": session_id},
                timeout=15.0,
            )
        ok = resp.status_code == 200
    except Exception as exc:  # noqa: BLE001 — best-effort context channel
        log.warning("fabric_user_context_forward_failed", error=str(exc)[:160])
        return {"status": "error", "reason": "sidecar forward failed"}
    return {"status": "ok" if ok else "skipped", "forwarded": ok}


async def _forward_to_local_csm(method_path: str, *, params: dict | None = None) -> dict:
    """Forward a residency ping to this node's local CSM sidecar. Shared by
    the warmup/unload bridges. No-op (200) if no sidecar is configured."""
    from augmentum.config import settings as _settings
    sidecar_url = (_settings.tts_sesame_csm_url or "").rstrip("/")
    if not sidecar_url:
        return {"status": "skipped", "reason": "no csm sidecar on this node"}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(f"{sidecar_url}{method_path}", params=params or {}, timeout=10.0)
    except Exception as exc:  # noqa: BLE001 — best-effort residency channel
        log.warning("fabric_csm_residency_forward_failed", path=method_path, error=str(exc)[:160])
        return {"status": "error"}
    return {"status": "ok"}


@router.post("/tts/warmup")
async def fabric_tts_warmup(request: Request) -> dict:
    """Pre-load this node's CSM sidecar on behalf of a peer (residency)."""
    _require_fabric_peer(request)
    return await _forward_to_local_csm("/warmup")


@router.post("/tts/unload")
async def fabric_tts_unload(request: Request) -> dict:
    """Unload this node's CSM sidecar + clear a session's context (residency)."""
    _require_fabric_peer(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = str((body or {}).get("session_id", "") or "")
    return await _forward_to_local_csm("/unload", params={"session": session_id} if session_id else None)


# ── HTTP: discovery ───────────────────────────────────────────────


@router.get("/hello")
async def fabric_hello(request: Request) -> dict:
    """Unauthenticated identity announcement.

    Probed by LAN-discovery sweeps to confirm "is this host another
    augmentum instance, and what's its pinned fingerprint?". The
    response is signature-less by design -- a hostile responder on the
    LAN could echo any string, but the operator confirms the
    fingerprint out-of-band before pairing, and the actual pair
    handshake verifies it again. The threat model is "let the user
    find their other augmentum without typing IPs", not "trust
    discovery responses for auth."

    Returns 503 when ``fabric_enabled`` is false -- a disabled node
    doesn't participate in discovery at all, and the absence of a
    response is itself the correct signal.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        raise HTTPException(
            status_code=503, detail="fabric coordinator not initialised"
        )

    import socket as _socket

    from augmentum import __version__ as augmentum_version

    try:
        hostname = _socket.gethostname() or "augmentum"
    except Exception:
        hostname = "augmentum"

    identity = coordinator._identity
    return {
        "service": "augmentum-fabric",
        "node_id": identity.node_id,
        "fingerprint": identity.fingerprint,
        "public_key": identity.public_key_b64,
        "hostname": hostname,
        "version": augmentum_version,
        "role": "peer",
        "icon": settings.local_fabric_icon or "",
    }


@router.post("/discover")
async def fabric_discover(request: Request) -> dict:
    """Operator-triggered LAN sweep. Admin-only.

    Body shape (all optional)::

        {"subnet": "192.168.1.0/24", "timeout_s": 12}

    Returns a partitioned candidate list -- truly new peers, peers
    that responded but match our own fingerprint, and peers that
    responded but are already paired. The UI uses this to pre-fill
    the existing pair form; pairing itself still goes through the
    /pair-with-remote flow so the operator confirms the fingerprint.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        raise HTTPException(
            status_code=503, detail="fabric coordinator not initialised"
        )

    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}

    subnet_raw = body.get("subnet")
    subnet = str(subnet_raw).strip() if subnet_raw else None
    subnet_source = "operator" if subnet else ""

    # No subnet supplied: derive it from the request's Host header before
    # falling back to the hardcoded common-subnet list. The operator is
    # almost certainly browsing augmentum at a LAN IP (e.g. ``Host:
    # 192.168.1.10:6443``), and *that* IP's /24 is what they want to
    # scan — far more likely to contain their other nodes than the three
    # consumer-router defaults. The defaults still apply when the Host
    # header is ``localhost`` / an FQDN / non-RFC1918.
    if not subnet:
        host_header = request.headers.get("host", "") or ""
        derived = derive_subnet_from_host(host_header)
        if derived:
            subnet = derived
            subnet_source = "host_header"
            log.info(
                "fabric_discover_subnet_autodetected",
                subnet=subnet, host=host_header,
            )

    timeout_raw = body.get("timeout_s")
    try:
        timeout_s = float(timeout_raw) if timeout_raw is not None else 12.0
    except (TypeError, ValueError):
        timeout_s = 12.0
    timeout_s = max(2.0, min(60.0, timeout_s))

    result = await discover_fabric_peers(
        subnet=subnet,
        own_fingerprint=coordinator._identity.fingerprint,
        known_node_ids=set(coordinator.known_peer_ids()),
        timeout_s=timeout_s,
    )

    log.info(
        "fabric_discover_complete",
        subnet=subnet or "fallback",
        subnet_source=subnet_source or "fallback",
        hosts_probed=result.hosts_probed,
        peers_found=len(result.peers),
        already_paired=len(result.already_paired),
        self_seen=len(result.self_seen),
        duration_s=round(result.duration_s, 2),
    )

    return {"ok": True, **result.to_dict()}


# ── HTTP: status ──────────────────────────────────────────────────


@router.get("/status")
async def fabric_status(request: Request) -> dict:
    """Lightweight read of fabric state. Admin-only.

    Returned even when fabric is disabled so the operator UI can show
    "fabric is off" without a 503 in the way. The peer list is empty
    until the coordinator initialises.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    from augmentum.fabric.version import get_code_version as _get_code_version

    if not settings.fabric_enabled:
        return {"enabled": False, "peers": {"total": 0, "connected": 0, "offline": 0}}

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        return {"enabled": True, "peers": {"total": 0, "connected": 0, "offline": 0}}

    return {
        "enabled": True,
        "this_node": {
            "node_id": coordinator._identity.node_id,
            "fingerprint": coordinator._identity.fingerprint,
            "icon": settings.local_fabric_icon or "",
            # Git SHA of the code this node is running, for drift visibility
            # ("is this node up to date?"). See scripts/deploy-nodes.sh
            # --status for the cross-node view today; sketch for heartbeat-
            # carried propagation in scripts/FABRIC_UPDATE.md.
            "code_version": _get_code_version(),
        },
        "peers": coordinator.peer_count(),
        "peer_ids": coordinator.known_peer_ids(),
        "connected_peer_ids": coordinator.connected_peer_ids(),
        # Phase 2: aggregate capability counts across this node + all
        # connected peers. Useful for the operator dashboard's "what
        # can my fabric do" summary.
        "capability_summary": coordinator.capability_summary(),
    }


@router.get("/capabilities")
async def fabric_capabilities(request: Request, kind: str | None = None) -> dict:
    """Capability inventory across this node + every connected peer.

    Filter with ``?kind=llm.inference`` to narrow. Admin-only. Returns
    an empty list when fabric is disabled or no peers are connected
    (rather than 503 -- the UI wants to render "fabric is off" cleanly).
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        return {"enabled": False, "local": [], "peers": {}}

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        return {"enabled": True, "local": [], "peers": {}}

    def _filtered(caps):
        if kind is None:
            return [serialise(c) for c in caps]
        return [serialise(c) for c in caps if c.kind == kind]

    local = _filtered(coordinator.local_capabilities())
    peers: dict[str, list[dict]] = {}
    for node_id in coordinator.known_peer_ids():
        state = coordinator.peer_state(node_id)
        if state is None or not state.connected:
            continue
        caps = _filtered(state.capabilities)
        if caps:  # skip peers with nothing matching the filter
            peers[node_id] = caps

    return {
        "enabled": True,
        "this_node_id": coordinator._identity.node_id,
        "local": local,
        "peers": peers,
    }


@router.get("/peers")
async def fabric_peers(request: Request) -> dict:
    """Detailed peer inventory for the operator UI.

    Returns one entry per paired peer with its current connection
    status, advertised capabilities, last-seen, etc. Admin-only.
    Returns ``{"enabled": false, "peers": []}`` cleanly when fabric
    is disabled (the UI shows "fabric is off" instead of erroring).
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        return {"enabled": False, "peers": []}

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        return {"enabled": True, "peers": []}

    peers_out: list[dict] = []
    for node_id in coordinator.known_peer_ids():
        state = coordinator.peer_state(node_id)
        if state is None:
            continue
        paired = state.paired
        peers_out.append({
            "node_id": node_id,
            "hostname": paired.hostname,
            "fingerprint": paired.fingerprint,
            "role": paired.role,
            "addr": paired.addr,
            "tier": paired.tier,
            "icon": paired.icon,
            "fabric_share_enabled": paired.fabric_share_enabled,
            "paired_at": paired.paired_at,
            "last_seen_at": paired.last_seen_at,
            "connected": state.connected,
            "last_seq_received": state.last_seq_received,
            "capability_count": len(state.capabilities),
            "capabilities": [serialise(c) for c in state.capabilities],
        })

    return {
        "enabled": True,
        "this_node_id": coordinator._identity.node_id,
        "peers": peers_out,
    }


@router.post("/peers/{node_id}/approve")
async def fabric_approve_peer(request: Request, node_id: str) -> dict:
    """Approve a PENDING inbound-paired peer. Admin-only.

    Inbound ``/api/fabric/pair`` lands peers with ``fabric_share_enabled = 0``
    (pending) because that endpoint is unauthenticated — anyone who scrapes
    this node's fingerprint from ``/api/fabric/hello`` can mint a valid-looking
    pair request with their own key. This endpoint is the operator's explicit
    consent: it flips the peer to enabled so it can finally authenticate
    data-plane requests (``lookup_peer_pubkey`` only resolves enabled peers).
    Idempotent.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend not available")

    cursor = await conn.execute(
        "SELECT id FROM fabric_nodes WHERE id = ? LIMIT 1", (node_id,),
    )
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="peer not found")

    await conn.execute(
        "UPDATE fabric_nodes SET fabric_share_enabled = 1 WHERE id = ?",
        (node_id,),
    )
    await conn.commit()

    # Seed the now-enabled peer into the coordinator's in-memory view.
    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is not None:
        try:
            from augmentum.fabric.peer_auth import load_paired_peers
            for peer in await load_paired_peers(conn):
                if peer.node_id == node_id:
                    await coordinator.register_paired_peer(peer)
                    break
        except Exception:
            log.warning("fabric_approve_register_failed", node_id=node_id, exc_info=True)

    log.info("fabric_peer_approved", node_id=node_id)
    return {"ok": True, "approved": node_id}


@router.delete("/peers/{node_id}")
async def fabric_unpair(request: Request, node_id: str) -> dict:
    """Remove a paired peer. Admin-only.

    Tears down any active connection + deletes the fabric_nodes row.
    The remote peer's view goes stale until they unpair us symmetrically
    (a re-pair from the same identity would reconnect).
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="fabric coordinator not initialised")

    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend not available")

    # Verify the peer exists before mutating.
    cursor = await conn.execute(
        "SELECT id FROM fabric_nodes WHERE id = ? LIMIT 1", (node_id,),
    )
    if not await cursor.fetchone():
        raise HTTPException(status_code=404, detail="peer not found")

    # Coordinator first: close any active socket, remove from registry.
    # Then SQL delete -- in that order so a request landing on the peer
    # during the unpair sees "no longer connected" rather than "still
    # connected but DB row missing" (which would crash).
    await coordinator.unregister_peer(node_id)
    await conn.execute("DELETE FROM fabric_nodes WHERE id = ?", (node_id,))
    await conn.commit()

    log.info("fabric_peer_unpaired", node_id=node_id)
    return {"ok": True, "unpaired": node_id}


@router.post("/pair-with-remote")
async def fabric_pair_with_remote(request: Request) -> dict:
    """Initiate the pair handshake from THIS node out to a remote peer.

    Operator-driven outbound counterpart to POST /api/fabric/pair.
    The UI submits this when the operator pastes a remote URL +
    fingerprint into the Fabric tab; we build a signed PairRequest
    from our local identity, POST it to the remote's /api/fabric/pair
    endpoint, validate the response identifies them at the expected
    fingerprint (defensive cross-check), and persist them into our
    own fabric_nodes table.

    Request body shape::

        {
          "remote_url": "https://192.168.1.20",
          "expected_fingerprint": "SHA256:abc…",
          "role": "peer",            # optional; this node's role
          "remote_addr": "192.168.1.20:6443"  # optional
        }

    Admin-only. Gated on settings.fabric_enabled.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    if coordinator is None:
        raise HTTPException(
            status_code=503, detail="fabric coordinator not initialised"
        )

    sm = getattr(request.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend not available")

    body = await request.json()
    remote_url = str(body.get("remote_url", "") or "").strip()
    expected_fp = str(body.get("expected_fingerprint", "") or "").strip()
    role = str(body.get("role", "peer") or "peer").strip()
    remote_addr = str(body.get("remote_addr", "") or "").strip()
    # Local-pick icon (Phase 8). Empty is acceptable; UI falls back to 🔗.
    # Trimmed to avoid stray whitespace from copy/paste; capped at 8 chars
    # (any single grapheme cluster fits, including ZWJ-composed emoji).
    icon = str(body.get("icon", "") or "").strip()[:8]

    if not remote_url:
        raise HTTPException(status_code=400, detail="remote_url is required")
    if not expected_fp:
        raise HTTPException(
            status_code=400, detail="expected_fingerprint is required"
        )
    if not remote_addr:
        # Best effort: derive from URL host:port. The user can override
        # via remote_addr when the WS endpoint lives at a different
        # host/port than the HTTPS edge (rare but possible).
        from urllib.parse import urlparse
        parsed = urlparse(remote_url if "://" in remote_url else f"https://{remote_url}")
        host = parsed.hostname or ""
        port = parsed.port or 443
        remote_addr = f"{host}:{port}" if host else ""
    if not remote_addr:
        raise HTTPException(
            status_code=400, detail="could not derive remote_addr from remote_url"
        )

    # Our hostname is best-effort: use the configured hostname setting
    # if present, otherwise socket.gethostname(). The remote stores it
    # for display only; identity is by node_id.
    import socket
    hostname = socket.gethostname() or coordinator._identity.node_id

    # OUR accessible address — what the remote will use to reach back
    # for the persistent peer WS. Best signal we have is the Host header
    # of the operator's pair-with-remote request: that's the URL their
    # browser is hitting us at, which is by definition reachable from
    # at least one LAN device. Strip the operator's port (often the
    # browser hits us via Caddy's 6443) and re-bake the canonical fabric
    # port so the remote knows where the WS edge actually lives.
    own_addr = ""
    host_header = (request.headers.get("host") or "").strip()
    host_only = host_header.split(":", 1)[0] if host_header else ""
    if host_only and host_only.lower() not in ("localhost", "127.0.0.1", "::1"):
        own_addr = f"{host_only}:6443"

    try:
        paired = await initiate_pair_with_remote(
            identity=coordinator._identity, hostname=hostname,
            remote_url=remote_url, expected_fingerprint=expected_fp,
            remote_addr=remote_addr, own_addr=own_addr,
            role=role, icon=icon, db=conn,
        )
    except OutboundPairError as exc:
        # Operator-facing error message; map to 4xx so the UI can
        # render it without retry. 502 = upstream issue (remote
        # rejected or unreachable), distinguishes from our own
        # validation failures.
        raise HTTPException(status_code=502, detail=str(exc)) from None

    # Register with the coordinator so the next reconnect-supervisor
    # pass picks the new peer up for a WS connect.
    await coordinator.register_paired_peer(paired)

    log.info(
        "fabric_outbound_pair_persisted",
        peer_node_id=paired.node_id, addr=paired.addr,
    )
    return {
        "ok": True,
        "paired": {
            "node_id": paired.node_id,
            "fingerprint": paired.fingerprint,
            "addr": paired.addr,
            "role": paired.role,
        },
    }


# ── WebSocket: peer connection ────────────────────────────────────


@router.websocket("/connect")
async def fabric_connect(websocket: WebSocket) -> None:
    """Persistent WebSocket from a paired peer.

    Lifecycle, mirroring the terminal-WS canonical model in
    ``coder_routes.terminal_ws``:

      1. Accept the upgrade.
      2. Wait for the peer's first frame -- must be a ``hello``
         envelope signed by their pinned pubkey. We look the pubkey
         up in fabric_nodes by ``sender_node_id``; an unknown id or
         signature mismatch closes the socket.
      3. Register the socket with the coordinator.
      4. Read loop: verify every envelope, record heartbeats, until
         disconnect.
      5. Detach + close-guard cleanup.

    Auth runs INSIDE the handler rather than at middleware level
    because peer auth is ed25519-challenge-based, not user-token-
    based. The fabric WS is not a user session; it is two server
    processes talking.
    """
    if not settings.fabric_enabled:
        await websocket.close(code=1011, reason="fabric disabled")
        return

    await websocket.accept()

    coordinator = getattr(websocket.app.state, "fabric_coordinator", None)
    if coordinator is None:
        await websocket.close(code=1011, reason="coordinator not initialised")
        return

    sm = getattr(websocket.app.state, "state_manager", None)
    conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
    if conn is None:
        await websocket.close(code=1011, reason="state backend not available")
        return

    # --- Step 1: read + verify the hello ---
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=_HELLO_TIMEOUT_S,
        )
    except TimeoutError:
        await websocket.close(code=4408, reason="hello timeout")
        return
    except WebSocketDisconnect:
        return

    try:
        # We don't yet know the sender's pubkey -- we need to look it
        # up from the envelope's sender_node_id field. Peek the
        # untrusted field first, then verify with the looked-up
        # pubkey. If the lookup fails or signature mismatches, close.
        import json as _json
        peek = _json.loads(raw)
        claimed_sender = str(peek["from"])
    except Exception:
        await websocket.close(code=1003, reason="malformed hello")
        return

    from augmentum.fabric.peer_auth import lookup_peer_pubkey
    pubkey_b64 = await lookup_peer_pubkey(conn, claimed_sender)
    if pubkey_b64 is None:
        log.info(
            "fabric_connect_unknown_peer",
            claimed_node_id=claimed_sender,
        )
        await websocket.close(code=4401, reason="unknown peer")
        return

    try:
        envelope = FabricEnvelope.from_wire(raw, expected_sender_pubkey_b64=pubkey_b64)
    except FabricProtocolError as exc:
        log.warning(
            "fabric_connect_bad_hello",
            claimed_node_id=claimed_sender,
            error=str(exc)[:160],
        )
        await websocket.close(code=4401, reason="bad signature")
        return

    if envelope.msg_type != MSG_HELLO:
        await websocket.close(code=1003, reason="expected hello first")
        return

    # --- Step 2: attach to coordinator ---
    attached = await coordinator.attach_connection(envelope.sender_node_id, websocket)
    if not attached:
        # Race: peer was unregistered between lookup_peer_pubkey() and
        # attach_connection(). Close politely.
        await websocket.close(code=4401, reason="peer not registered")
        return

    coordinator.record_heartbeat(envelope.sender_node_id, envelope.seq)
    log.info(
        "fabric_ws_attached",
        peer_node_id=envelope.sender_node_id,
        hostname=envelope.payload.get("hostname", ""),
    )

    # --- Step 3: read loop ---
    try:
        async for raw in websocket.iter_text():
            try:
                env = FabricEnvelope.from_wire(
                    raw, expected_sender_pubkey_b64=pubkey_b64,
                )
            except FabricProtocolError as exc:
                log.warning(
                    "fabric_ws_bad_envelope",
                    peer_node_id=envelope.sender_node_id,
                    error=str(exc)[:160],
                )
                # Drop the message; don't close. A burst of bad
                # frames can indicate clock skew, version drift, or
                # transient noise, none of which warrants tearing
                # down the connection.
                continue
            # Phase 9.4: delegated to the shared
            # coordinator.handle_inbound_envelope dispatcher.
            # Handles heartbeat seq + capability advertisement +
            # MSG_CANCEL_REQUEST + reserved lifecycle event types.
            # Same logic as the client.py read loop — symmetric
            # topology means either socket can carry any type.
            coordinator.handle_inbound_envelope(env)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception:
        log.warning(
            "fabric_ws_read_error",
            peer_node_id=envelope.sender_node_id,
            exc_info=True,
        )
    finally:
        await coordinator.detach_connection(envelope.sender_node_id)
        # Dual-state close guard (same pattern as terminal_ws): the
        # peer may have already disconnected, in which case calling
        # close() raises a noisy ASGI assertion. Cheap to check.
        if (
            websocket.application_state != WebSocketState.DISCONNECTED
            and websocket.client_state != WebSocketState.DISCONNECTED
        ):
            try:
                await websocket.close()
            except Exception:
                log.debug("fabric_ws_close_failed", exc_info=True)


def _client_addr(request: Request) -> str:
    """Best-effort remote address for fabric_nodes.addr default."""
    if request.client is None:
        return ""
    host = request.client.host or ""
    # No port -- the operator usually wants the URL form (port baked in
    # by Caddy), and we only have raw socket info here.
    return host


# ── HTTP: identity backup / restore (admin) ──────────────────────────
#
# The fabric Ed25519 key is the one piece of state that can't be
# recovered if lost (the fail-closed loader refuses to silently mint a
# replacement). These two admin-only endpoints are the human side of
# that contract: export a 24-word BIP39 phrase to write down, and
# restore from it after a seizure / disk loss / migration.


@router.get("/identity/backup")
async def fabric_identity_backup(request: Request) -> dict:
    """Return the 24-word BIP39 backup phrase for this node's identity.

    Admin-only. The phrase encodes the private key directly — anyone
    holding it can impersonate this instance — so it is shown once for
    the operator to transcribe offline and never stored in plaintext.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")

    identity = getattr(request.app.state, "fabric_identity", None)
    if identity is None:
        raise HTTPException(
            status_code=503, detail="fabric identity not initialised"
        )

    try:
        phrase = identity.mnemonic_backup()
    except Exception as exc:
        log.warning("fabric_identity_backup_failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"could not derive backup phrase: {exc}"
        ) from None

    log.info("fabric_identity_backup_exported", node_id=identity.node_id)
    return {
        "node_id": identity.node_id,
        "did_key": identity.did_key,
        "fingerprint": identity.fingerprint,
        "mnemonic": phrase,
        "word_count": len(phrase.split()),
        "warning": (
            "Write these 24 words down offline and keep them secret. "
            "Anyone with this phrase can impersonate this instance. "
            "Shown once — it is never stored in plaintext."
        ),
    }


@router.post("/identity/restore")
async def fabric_identity_restore(request: Request) -> dict:
    """Restore the fabric identity key from a 24-word BIP39 phrase.

    Admin-only. The manual counterpart to the fail-closed startup halt:
    after the loader refuses a corrupt/missing key, the operator pastes
    their backup phrase here. Validates the BIP39 checksum, re-encrypts
    the key, and persists it under ``fabric.node_private_key``. The
    instance's federated identity (did:key / fingerprint) is preserved
    exactly, because it is fully determined by the key bytes.

    A ``node_id`` may be supplied to also restore the human node id
    after a half-present/torn state; otherwise the existing one is kept.

    Takes effect on the next fabric (re)start — this endpoint writes the
    settings store; it does not hot-swap a running coordinator.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden  # type: ignore[return-value]

    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from augmentum.fabric.didkey import encode_ed25519_did
    from augmentum.fabric.identity import (
        _KEY_NODE_ID,
        _KEY_PRIVATE_KEY,
    )
    from augmentum.fabric.recovery import MnemonicError, mnemonic_to_key
    from augmentum.utils.secrets import encrypt_api_key

    settings_store = getattr(request.app.state, "settings_store", None)
    if settings_store is None:
        raise HTTPException(status_code=503, detail="settings store unavailable")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    phrase = str(body.get("mnemonic", "") or "").strip()
    if not phrase:
        raise HTTPException(status_code=400, detail="mnemonic required")

    try:
        priv_raw = mnemonic_to_key(phrase)
    except MnemonicError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    private_key = Ed25519PrivateKey.from_private_bytes(priv_raw)
    pub_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    did = encode_ed25519_did(pub_raw)

    encrypted = encrypt_api_key(base64.b64encode(priv_raw).decode("ascii"))
    await settings_store.set(_KEY_PRIVATE_KEY, encrypted)

    node_id_in = str(body.get("node_id", "") or "").strip()
    if node_id_in:
        await settings_store.set(_KEY_NODE_ID, node_id_in)
    else:
        existing = await settings_store.get(_KEY_NODE_ID)
        if not existing:
            # Half-present restore with no node_id given: mint one so the
            # next boot has a complete (node_id, key) pair.
            import secrets as _secrets
            node_id_in = _secrets.token_hex(16)
            await settings_store.set(_KEY_NODE_ID, node_id_in)

    log.info("fabric_identity_restored", did_key=did)
    return {
        "ok": True,
        "did_key": did,
        "restart_required": True,
        "detail": (
            "Identity key restored. Restart fabric (or the app) for the "
            "running coordinator to adopt it."
        ),
    }


# ── HTTP: contact-card federation + verification ceremony (P1) ────────
#
# The default federation trust root after D4 removed the directory: a
# user mints a signed contact card (shared as a link/QR), the recipient
# accepts it (TOFU pin, verified=False), and the two humans run the SAS
# /QR ceremony out-of-band to upgrade the pin to verified. These are
# USER-facing routes (a logged-in browser session), not peer endpoints —
# they gate on a real user_id and refuse the anon row.


def _require_user_id(request: Request) -> str:
    """Extract the authenticated user's id or 401. Anon row refused."""
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""
    if not user_id:
        raise HTTPException(status_code=401, detail="authentication required")
    return user_id


def _fabric_db_conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    return getattr(getattr(sm, "backend", None), "conn", None) if sm else None


def _fabric_identity_or_503(request: Request):
    identity = getattr(request.app.state, "fabric_identity", None)
    if identity is None:
        raise HTTPException(
            status_code=503, detail="fabric identity not initialised"
        )
    return identity


@router.post("/contact-card")
async def fabric_mint_contact_card(request: Request) -> dict:
    """Mint a signed contact card for the current user to share.

    Body (all optional): ``{"endpoint": "...", "token": "...",
    "issued_at": <int>}``. Returns the signed card dict plus a
    ``share`` block (link + QR text) the UI renders.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    identity = _fabric_identity_or_503(request)

    from augmentum.fabric.contact_card import mint_card

    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}

    endpoint = str(body.get("endpoint", "") or "")
    if not endpoint:
        # Best-effort self-endpoint from the request host.
        host = request.headers.get("host", "")
        scheme = request.headers.get("x-forwarded-proto", "https")
        endpoint = f"{scheme}://{host}" if host else ""

    issued_at = body.get("issued_at")
    if not isinstance(issued_at, int):
        issued_at = int(time.time())

    host = request.headers.get("host", "") or ""
    handle = f"{user_id}@{host}" if host else user_id
    # P1: author key == instance key (P2 splits per-user author keys via
    # device subkeys). Carried through so the card/SAS format is stable.
    card = mint_card(
        sign=identity.sign,
        instance_did_key=identity.did_key,
        endpoint=endpoint,
        author_did_key=identity.did_key,
        handle=str(body.get("handle", "") or handle),
        token=str(body.get("token", "") or uuid_hex()),
        issued_at=issued_at,
    )

    # Friendly profile (advisory, like a display name): the recipient
    # SEES this so they add a person, not a key. It is NOT a trust input —
    # the key is what's verified — so it rides alongside the signed card,
    # not inside it.
    conn = _fabric_db_conn(request)
    profile = {"display_name": str(body.get("display_name", "") or ""),
               "avatar_ref": "", "status_emoji": ""}
    if conn is not None:
        try:
            from augmentum.connect.profile_store import get_profile
            p = await get_profile(conn, user_id=user_id)
            profile["avatar_ref"] = p.get("avatar_ref", "")
            profile["status_emoji"] = p.get("status_emoji", "")
        except Exception:
            log.warning("fabric_card_profile_lookup_failed", exc_info=True)
    if not profile["display_name"]:
        profile["display_name"] = handle.split("@")[0]

    import json as _json

    card_json = _json.dumps(card, separators=(",", ":"))
    card_b64 = _b64url(card_json.encode("utf-8"))

    # Short, shareable connect code — "scan this, or enter K7P2-9QX4".
    code = ""
    if conn is not None:
        try:
            from augmentum.fabric.connect_codes import create_code, format_code
            raw_code = await create_code(
                conn, user_id=user_id, card={"card": card, "profile": profile},
            )
            code = format_code(raw_code)
        except Exception:
            log.warning("fabric_connect_code_create_failed", exc_info=True)

    code_link = (
        f"{endpoint}/connect/add?code={code.replace('-', '')}"
        if (endpoint and code) else ""
    )
    share_link = f"{endpoint}/connect/add#card={card_b64}" if endpoint else ""
    from augmentum.fabric.presentation import identity_code
    return {
        "card": card,
        "profile": profile,
        "share": {
            "link": code_link or share_link,     # prefer the short link
            "full_link": share_link,             # direct (no server lookup)
            "code": code,                        # human-typable "K7P2-9QX4"
            "qr_url": "/api/fabric/contact-card/qr" + (f"?code={code.replace('-', '')}" if code else ""),
            "qr_text": card_b64,
            "did_key": identity.did_key,
            "safety_code": identity_code(identity.did_key),
        },
    }


@router.get("/contact-card/qr")
async def fabric_contact_card_qr(request: Request, code: str = ""):
    """Return a scannable QR PNG for sharing — 'scan to connect'.

    With ``?code=...`` the QR encodes the short connect link; otherwise it
    encodes a freshly-minted card link for the current user. Image bytes,
    not JSON — the UI drops it straight into an <img>.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    _require_user_id(request)

    host = request.headers.get("host", "") or ""
    scheme = request.headers.get("x-forwarded-proto", "https")
    base = f"{scheme}://{host}" if host else ""
    payload = f"{base}/connect/add?code={code}" if code else base

    try:
        import io

        import qrcode
        img = qrcode.make(payload or "augmentum")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        from fastapi.responses import Response as _Resp
        return _Resp(content=buf.getvalue(), media_type="image/png")
    except Exception as exc:
        log.warning("fabric_qr_render_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"could not render QR: {exc}") from None


@router.get("/connect/{code}")
async def fabric_resolve_connect_code(code: str, request: Request) -> dict:
    """Resolve a short connect code to its contact card + profile.

    Public-ish (any logged-in user) so the recipient can look up the code
    they were given and preview who's inviting them before accepting.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric.connect_codes import resolve_code
    payload = await resolve_code(conn, code=code, now=int(time.time()))
    if payload is None:
        raise HTTPException(status_code=404, detail="that connect code wasn't found or has expired")
    return payload  # {"card": ..., "profile": ...}


# Friendly labels for the admission posture — plain "who can reach me".
_POSTURE_COPY = {
    "private": {"label": "Only my contacts", "detail": "New people can't reach you."},
    "allowlist": {"label": "Only people I approve", "detail": "New people are blocked unless you add them."},
    "knock": {"label": "Let people request", "detail": "New people land in Requests for you to accept."},
    "open": {"label": "Anyone can reach me", "detail": "New people can message you directly."},
}


@router.get("/federation/status")
async def fabric_federation_status(request: Request) -> dict:
    """One call for a polished Connect home: am I set up, my shareable
    identity, and friendly counts. All plain-language — no jargon."""
    if not settings.fabric_enabled:
        return {"enabled": False, "ready": False,
                "message": "Connect isn't turned on yet."}
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    identity = getattr(request.app.state, "fabric_identity", None)

    from augmentum.fabric.presentation import identity_code

    posture = getattr(settings, "fabric_admission_posture", "knock") or "knock"
    out: dict = {
        "enabled": bool(getattr(settings, "fabric_federation_enabled", False)),
        "ready": identity is not None,
        "reach_setting": {"value": posture, **_POSTURE_COPY.get(posture, _POSTURE_COPY["knock"])},
    }
    if identity is not None:
        host = request.headers.get("host", "") or ""
        out["me"] = {
            "name": user_id.split("@")[0] if "@" in user_id else user_id,
            "handle": f"{user_id}@{host}" if host else user_id,
            "safety_code": identity_code(identity.did_key),
        }
    if conn is not None:
        try:
            cur = await conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(verified),0) "
                "FROM fabric_peer_identities WHERE user_id=?", (user_id,),
            )
            total, verified = (await cur.fetchone()) or (0, 0)
            cur2 = await conn.execute(
                "SELECT COUNT(*) FROM fabric_knocks "
                "WHERE to_user_id=? AND status='pending'", (user_id,),
            )
            pending = (await cur2.fetchone() or [0])[0]
            out["contacts"] = {
                "total": int(total), "verified": int(verified),
                "unverified": int(total) - int(verified),
            }
            out["requests"] = int(pending)
        except Exception:
            log.warning("fabric_federation_status_counts_failed", exc_info=True)
    return out


@router.post("/contact-card/accept")
async def fabric_accept_contact_card(request: Request) -> dict:
    """Accept a contact card: verify signature, detect key change, pin.

    Body: ``{"card": {...}}`` OR ``{"card_b64": "<base64url>"}``.
    Returns the pinned peer (verified=False) + any key-change warning.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric.contact_card import ContactCardError, parse_card
    from augmentum.fabric.peer_identity_store import detect_key_change, pin_peer

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")

    # Accept a card directly, a base64 card, OR a short connect code.
    card_data = body.get("card")
    profile = body.get("profile") if isinstance(body.get("profile"), dict) else {}
    if card_data is None and body.get("code"):
        from augmentum.fabric.connect_codes import resolve_code
        wrapper = await resolve_code(conn, code=str(body["code"]), now=int(time.time()))
        if wrapper is None:
            raise HTTPException(
                status_code=404,
                detail="that connect code wasn't found or has expired",
            )
        card_data = wrapper.get("card")
        if isinstance(wrapper.get("profile"), dict):
            profile = wrapper["profile"]
    if card_data is None and body.get("card_b64"):
        import json as _json
        try:
            card_data = _json.loads(_b64url_decode(str(body["card_b64"])))
        except Exception:
            raise HTTPException(status_code=400, detail="invalid card_b64") from None
    if not isinstance(card_data, dict):
        raise HTTPException(status_code=400, detail="card required")

    try:
        card = parse_card(card_data)
    except ContactCardError as exc:
        # Signature/format failure — never pin an unverifiable card.
        from augmentum.fabric.presentation import friendly_error
        raise HTTPException(
            status_code=400, detail=friendly_error(str(exc)),
        ) from None

    # Safety-number-changed detection BEFORE pinning. A known handle now
    # carrying a different key = legitimate rotation OR impersonation;
    # surface it, force re-verification, do not silently overwrite.
    key_change_warning = await detect_key_change(
        conn, user_id=user_id, handle=card.handle, new_did_key=card.instance_did_key,
    )

    pinned = await pin_peer(
        conn,
        user_id=user_id,
        peer_did_key=card.instance_did_key,
        handle=card.handle,
        endpoint=card.endpoint,
        author_did_key=card.author_did_key,
        source="card",
    )

    # Remember the inviter as a PERSON (name + avatar) in the existing
    # contacts model, so they show up with a face — not a key — across
    # Connect. Advisory display data only; the key is what's verified.
    display_name = str(profile.get("display_name", "") or "") or card.handle.split("@")[0]
    try:
        from augmentum.connect.contact_store import remember_peer_display_name
        await remember_peer_display_name(
            conn, user_id=user_id, peer_did=card.handle, display_name=display_name,
        )
    except Exception:
        log.warning("fabric_card_remember_name_failed", exc_info=True)

    changed = bool(key_change_warning)
    peer_dict = _peer_identity_dict(pinned, key_changed=changed)
    peer_dict["display_name"] = display_name
    return {
        "pinned": peer_dict,
        "trust_label": pinned.trust_label,
        "key_change_warning": changed,
        "previous_keys": key_change_warning,
        # Plain-language next step — no "SAS"/"out-of-band" jargon.
        "next_step": (
            f"Added {display_name}. To be sure it's really them, give them a "
            "call and tap Verify — you'll each read a short code that must match."
            if not changed else
            f"Heads up — {display_name}'s security code changed. Verify with them "
            "again before trusting this contact."
        ),
    }


@router.get("/peers/verified")
async def fabric_list_verified_peers(request: Request) -> dict:
    """List the current user's pinned peer identities + trust labels."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    cur = await conn.execute(
        "SELECT id, user_id, peer_did_key, handle, endpoint, author_did_key, "
        "verified, verified_method, verified_at, source "
        "FROM fabric_peer_identities WHERE user_id=? ORDER BY first_pinned_at DESC",
        (user_id,),
    )
    rows = await cur.fetchall()
    from augmentum.fabric.peer_identity_store import _row_to_identity

    return {"peers": [_peer_identity_dict(_row_to_identity(r)) for r in rows]}


@router.post("/peers/ceremony")
async def fabric_compute_ceremony(request: Request) -> dict:
    """Compute the SAS words + safety number for a pinned peer.

    Body: ``{"peer_did_key": "did:key:z..."}``. Returns the values the
    user reads aloud / scans; both sides compute the same thing.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    identity = _fabric_identity_or_503(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric import ceremony as _ceremony
    from augmentum.fabric.peer_identity_store import get_peer

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    peer_did = str((body or {}).get("peer_did_key", "") or "")
    if not peer_did:
        raise HTTPException(status_code=400, detail="peer_did_key required")

    peer = await get_peer(conn, user_id=user_id, peer_did_key=peer_did)
    if peer is None:
        raise HTTPException(status_code=404, detail="peer not pinned")

    self_author = identity.did_key  # P1: author == instance key
    return {
        "sas_words": _ceremony.sas_words(
            identity.did_key, peer.peer_did_key,
            self_author=self_author, peer_author=peer.author_did_key,
        ),
        "safety_number": _ceremony.safety_number(
            identity.did_key, peer.peer_did_key,
            self_author=self_author, peer_author=peer.author_did_key,
        ),
        "instruction": (
            "On a call: read these 4 words to each other — they must match. "
            "In person: compare the safety number (or scan the QR)."
        ),
    }


@router.post("/peers/verify")
async def fabric_mark_peer_verified(request: Request) -> dict:
    """Mark a pinned peer verified after a successful ceremony.

    Body: ``{"peer_did_key": "...", "method": "sas"|"qr"}``. The human
    confirms the SAS/QR matched; this records the upgrade.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric.peer_identity_store import mark_verified

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    peer_did = str((body or {}).get("peer_did_key", "") or "")
    method = str((body or {}).get("method", "") or "")
    if not peer_did:
        raise HTTPException(status_code=400, detail="peer_did_key required")
    if method not in ("sas", "qr"):
        raise HTTPException(status_code=400, detail="method must be 'sas' or 'qr'")

    try:
        peer = await mark_verified(
            conn, user_id=user_id, peer_did_key=peer_did, method=method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"peer": _peer_identity_dict(peer), "trust_label": peer.trust_label}


def _peer_identity_dict(peer, *, key_changed: bool = False) -> dict:
    """Serialize a PeerIdentity for the UI. ALWAYS includes the trust
    state so the frontend can render the verified/unverified chip
    (D1-01) — the name alone is attacker-controllable — PLUS the
    plain-language presentation so every surface shows the same
    professional, non-technical copy."""
    from augmentum.fabric.presentation import (
        connection_presentation,
        identity_code,
    )
    base = {
        "id": peer.id,
        "peer_did_key": peer.peer_did_key,
        "handle": peer.handle,
        "display_name": (peer.handle or "").split("@")[0] or peer.handle,
        "endpoint": peer.endpoint,
        "verified": peer.verified,
        "verified_method": peer.verified_method,
        "trust_label": peer.trust_label,
        "key_changed": key_changed,
        "source": peer.source,
        # Friendly, human-facing presentation (state/label/hint/tone/icon).
        "presentation": connection_presentation(
            {"verified": peer.verified, "key_changed": key_changed}
        ),
        "safety_code": identity_code(peer.peer_did_key),
    }
    return base


# ── HTTP: knock inbox (the deny-by-default stranger queue, P2) ────────


@router.get("/knocks")
async def fabric_list_knocks(request: Request) -> dict:
    """List the current user's pending knocks (intro text withheld)."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")
    from augmentum.fabric.knock import list_pending

    knocks = await list_pending(conn, to_user_id=user_id)
    return {
        "knocks": [
            {
                "id": k.id,
                "from_did_key": k.from_did_key,
                "from_handle": k.from_handle,
                "intro_flagged": k.intro_flagged,
            }
            for k in knocks
        ]
    }


@router.post("/knocks/{knock_id}/accept")
async def fabric_accept_knock(knock_id: str, request: Request) -> dict:
    """Accept a knock: reveal the withheld intro and TOFU-pin the source."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")
    from augmentum.fabric.knock import accept_knock

    try:
        result = await accept_knock(conn, to_user_id=user_id, knock_id=knock_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    return {
        "from_did_key": result["from_did_key"],
        "intro_text": result["intro_text"],
        "pinned": _peer_identity_dict(result["pinned"]),
    }


@router.post("/knocks/{knock_id}/reject")
async def fabric_reject_knock(knock_id: str, request: Request) -> dict:
    """Reject a pending knock."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")
    from augmentum.fabric.knock import reject_knock

    rejected = await reject_knock(conn, to_user_id=user_id, knock_id=knock_id)
    return {"rejected": rejected}


# ── HTTP: per-thread E2E toggle (P3) ─────────────────────────────────


@router.post("/e2e/thread")
async def fabric_set_thread_e2e(request: Request) -> dict:
    """Enable/disable E2E for a thread. Body: ``{"thread_id": "...",
    "enabled": true, "peer_master_did": "did:key:z..."}``."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")
    from augmentum.fabric.thread_e2e import set_e2e

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    thread_id = str((body or {}).get("thread_id", "") or "")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    try:
        state = await set_e2e(
            conn, user_id=user_id, thread_id=thread_id,
            enabled=bool(body.get("enabled", False)),
            peer_master_did=str(body.get("peer_master_did", "") or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"thread_id": thread_id, "enabled": state.enabled,
            "peer_master_did": state.peer_master_did}


@router.get("/e2e/thread/{thread_id}")
async def fabric_get_thread_e2e(thread_id: str, request: Request) -> dict:
    """Return a thread's E2E state for the current user."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")
    from augmentum.fabric.thread_e2e import get_e2e
    state = await get_e2e(conn, user_id=user_id, thread_id=thread_id)
    return {"thread_id": thread_id, "enabled": state.enabled,
            "peer_master_did": state.peer_master_did}


# ── HTTP: E2E device bundles (P2 — public key distribution) ──────────


@router.put("/e2e/device-bundle")
async def fabric_put_device_bundle(request: Request) -> dict:
    """Publish the current user's device bundle (public keys only).

    Body: ``{"master_did": "did:key:z...", "devices": [{subkey_did,
    sealing_pub_b64, binding, label}, ...]}``. The server validates every
    binding chains to ``master_did`` before storing — a forged bundle is
    rejected. Private keys must never be sent here.
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric.device_bundle import DeviceBundleError, put_bundle

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")

    try:
        bundle = await put_bundle(
            conn, user_id=user_id,
            master_did=str(body.get("master_did", "") or ""),
            devices=body.get("devices") or [],
        )
    except DeviceBundleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"published": True, "master_did": bundle.master_did, "devices": len(bundle.devices)}


@router.get("/e2e/device-bundle")
async def fabric_get_own_device_bundle(request: Request) -> dict:
    """Return the current user's own published bundle (or empty)."""
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric.device_bundle import get_bundle
    bundle = await get_bundle(conn, user_id=user_id)
    if bundle is None:
        return {"published": False, "master_did": "", "devices": []}
    return {"published": True, **bundle.as_dict()}


@router.get("/e2e/device-bundle/{peer_user_id}")
async def fabric_get_peer_device_bundle(peer_user_id: str, request: Request) -> dict:
    """Return a local peer's device bundle to seal to them, alongside the
    master the CALLER pinned/verified for that peer (when known) so the
    client can refuse to seal if the published master doesn't match the
    one it verified in the ceremony (closes the host-swap gap).
    """
    if not settings.fabric_enabled:
        raise HTTPException(status_code=503, detail="fabric disabled")
    user_id = _require_user_id(request)
    conn = _fabric_db_conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state backend unavailable")

    from augmentum.fabric.device_bundle import get_bundle
    bundle = await get_bundle(conn, user_id=peer_user_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="that contact hasn't set up encryption yet")

    # The caller's pinned master for this peer (if they ran the ceremony).
    pinned_master = ""
    verified = False
    try:
        cur = await conn.execute(
            "SELECT author_did_key, verified FROM fabric_peer_identities "
            "WHERE user_id=? AND author_did_key=? LIMIT 1",
            (user_id, bundle.master_did),
        )
        row = await cur.fetchone()
        if row is not None:
            pinned_master, verified = row[0], bool(row[1])
    except Exception:
        log.warning("fabric_peer_pinned_master_lookup_failed", exc_info=True)

    return {
        **bundle.as_dict(),
        "pinned_master_matches": bool(pinned_master) and pinned_master == bundle.master_did,
        "verified": verified,
        "warning": (
            "" if (pinned_master and pinned_master == bundle.master_did)
            else "You haven't verified this contact's encryption key — verify before trusting."
        ),
    }


def uuid_hex() -> str:
    import uuid as _uuid
    return _uuid.uuid4().hex


def _b64url(data: bytes) -> str:
    import base64 as _b64
    return _b64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    import base64 as _b64
    pad = "=" * (-len(s) % 4)
    return _b64.urlsafe_b64decode(s + pad)
