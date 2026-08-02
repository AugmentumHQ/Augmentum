"""Anthropic Messages API surface — minimal adapter on top of the
existing OpenAI-compat route.

What's here:
    POST /v1/messages              — chat (stream + non-stream)
    POST /v1/messages/count_tokens — token-budget estimator (CC needs this)

What we deliberately delegate:
    * Auth — middleware already accepts ``x-api-key`` (sk-aug-...) as
      of the change to ``auth/middleware.py::_extract_token``.
    * Orchestration — every Anthropic chat call funnels into
      ``openai_chat`` as a function call. That route already handles
      classifier, fabric routing, memory injection, vision fallback,
      mode hints, lifecycle, telemetry — we get them all for free.
    * Tool execution — Claude Code runs its own tool loop on the user
      side. Augmentum just needs to forward the tool schemas and pass
      back the model's ``tool_use`` blocks.

Faux-streaming
    Even when ``stream=true``, we run the inner request non-streaming
    and emit the full Anthropic event sequence on completion.
    Rationale documented in ``models/anthropic_compat.py``. While the
    inner call runs we emit ``ping`` events every 10s to keep
    intermediaries from idling out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from augmentum.config import settings
from augmentum.models.anthropic_compat import (
    AnthropicMessagesRequest,
    anthropic_request_to_openai,
    build_message_start_event,
    build_ping_event,
    compute_prefix_cache_key,
    count_tokens_estimate,
    internal_response_to_anthropic_message,
    stream_internal_response_as_anthropic_sse,
)
from augmentum.models.base import InternalChatResponse, Message, Usage
from augmentum.models.openai_compat import to_openai_chat_response
from augmentum.models.provider_registry import ModelUnavailableError
from augmentum.proxy.harness import (
    detect_harness,
    detect_project,
    inject_harness_context,
    schedule_harness_capture,
)
from augmentum.proxy.openai_routes import (
    openai_chat as openai_chat_handler,
)
from augmentum.proxy.openai_routes import (
    to_internal_chat_request,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["anthropic"])


# Anthropic streams require these headers — X-Accel-Buffering disables
# nginx response buffering (a buffering reverse proxy silently breaks
# CC), Cache-Control is the conventional SSE no-store.
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# Cadence for the keepalive ping during a faux-stream. 10s is well
# below the typical 30s idle-close on nginx / Cloudflare etc.
_PING_INTERVAL_S = 10.0


def _resolve_claude_alias(model: str) -> tuple[str, str | None]:
    """Map a ``claude-*`` model name to a locally-available equivalent.

    Returns ``(resolved_name, tier_used)`` — tier is one of
    ``"haiku" | "sonnet" | "opus" | "default" | None`` (None when
    no aliasing happened — either model isn't claude-* or no
    alias target is set).

    Why this exists
    ---------------
    Claude Code uses different model names for different jobs:

    * Main loop: whatever ``--model`` selected (e.g. the user's
      local model)
    * Subagents / Explore / Agent tool: hardcoded
      ``claude-haiku-4-5-*`` (cost/speed)
    * Title generation, compaction: also hardcoded haiku

    When CC fires a subagent against Augmentum, the request comes
    in with a hardcoded ``claude-*`` name that Augmentum doesn't
    serve. Without aliasing, every subagent fails with
    "model unavailable" and CC's Agent tool is structurally broken.
    With aliasing, the subagent gets routed to the user's chosen
    local model and just works.

    Lookup order (first non-empty wins):
      1. ``settings.anthropic_alias_haiku|sonnet|opus`` (per-tier)
      2. ``settings.anthropic_alias_default``
      3. ``settings.primary_chat_model`` (auto-tracked by Augmentum)

    Isolated to anthropic_routes — no other surface ever sees
    claude-* names so no other code path needs to know about this.
    """
    if not model or not model.startswith("claude-"):
        return model, None

    tier: str | None = None
    if "haiku" in model:
        tier = "haiku"
    elif "sonnet" in model:
        tier = "sonnet"
    elif "opus" in model:
        tier = "opus"

    target = ""
    if tier:
        target = getattr(settings, f"anthropic_alias_{tier}", "") or ""
    if not target:
        target = getattr(settings, "anthropic_alias_default", "") or ""
    if not target:
        target = getattr(settings, "primary_chat_model", "") or ""

    if target and target != model:
        log.info(
            "anthropic_model_aliased",
            requested=model, mapped=target, tier=tier or "default",
        )
        return target, tier or "default"

    # No alias target available — return unchanged so the downstream
    # ModelUnavailableError gives a clear message rather than us
    # silently routing to nothing.
    return model, None


def _adapt_openai_dict_to_internal(oa_dict: dict) -> InternalChatResponse:
    """Reconstruct an InternalChatResponse from the OpenAI JSON shape
    that ``openai_chat`` returns. We round-trip through this shape
    because the OpenAI route is the orchestration entry point — calling
    it as a function means we don't reimplement classifier / fabric /
    memory / etc.
    """
    choices = oa_dict.get("choices") or [{}]
    choice = choices[0] if choices else {}
    msg = choice.get("message") or {}
    usage = oa_dict.get("usage") or {}

    return InternalChatResponse(
        message=Message(
            role=msg.get("role", "assistant"),
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls"),
            thinking=msg.get("reasoning_content"),
        ),
        model=oa_dict.get("model", ""),
        finish_reason=choice.get("finish_reason"),
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        ),
    )


def _anthropic_error(message: str, *, status: int, error_type: str = "api_error") -> JSONResponse:
    """Render an error in Anthropic's expected shape.

    CC parses ``error.type`` and surfaces ``error.message`` directly to
    the user. Common error_type values: ``invalid_request_error``,
    ``authentication_error``, ``permission_error``, ``not_found_error``,
    ``rate_limit_error``, ``api_error``, ``overloaded_error``."""
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


async def _run_direct_backend_for_tools(
    body: AnthropicMessagesRequest, oa_body, request: Request,
    *, tier: str | None = None,
) -> tuple[dict | None, JSONResponse | None]:
    """Direct-to-backend path for caller-owns-tools requests.

    PassthroughHandler is built around Augmentum's chat UI where the
    server runs tools from its own registry — it filters incoming
    ``tool_calls`` against that registry and overwrites
    ``request.tools`` with Augmentum's own tool set, which drops any
    tools the external caller defined. CC (and Cursor agent mode,
    SillyTavern function-calling, etc.) all expect to run their own
    tool execution — the model's ``tool_calls`` must come back to
    them verbatim.

    This path bypasses PassthroughHandler entirely and calls
    ``backend.chat()`` directly so ``tool_calls`` flow through
    unfiltered. The cost: this path skips Augmentum's
    memory/knowledge/dream enrichment that ``openai_chat`` does.
    That's the right trade for a v1 CC integration where correctness
    of tool execution dominates feature richness. The MCP-server work
    discussed separately is the right place to re-add enrichment via
    explicit tool exposure rather than implicit injection.

    Isolated to anthropic_routes — does NOT modify PassthroughHandler,
    openai_routes, openai_compat, or anything in Augmentum's normal
    chat flow.
    """
    try:
        internal_req = to_internal_chat_request(oa_body)
    except Exception as exc:
        return None, _anthropic_error(
            f"Request translation failed: {exc}", status=400,
            error_type="invalid_request_error",
        )

    registry = getattr(request.app.state, "provider_registry", None)
    if registry is None:
        return None, _anthropic_error(
            "Provider registry unavailable (server still starting)",
            status=503,
            error_type="api_error",
        )
    user = request.scope.get("user")
    uid = user.id if user else ""

    # Pin CC traffic to slots keyed by the actual stable prefix, not a
    # coarse "all CC traffic for this user" bucket. CC sends
    # ``cache_control: ephemeral`` markers at the boundary between
    # stable content (system, tools, environment setup) and per-turn
    # content (current user query) — compute_prefix_cache_key hashes
    # everything up to and including the last marker. Two CC turns that
    # share the same stable prefix collide on the same hash and route
    # to the same slot, hitting llama-server's prefix cache. CC turns
    # in different contexts (different repo, different session) hash
    # differently and warm separate slots in parallel.
    #
    # Falls back to hashing (system + tools) for non-CC clients hitting
    # /v1/messages, so they still get same-system slot affinity without
    # needing to send cache_control.
    #
    # Per-user "uid" prefix keeps users on disjoint slots. "cc" tag
    # keeps CC traffic separate from any Augmentum chat the same user
    # has open in a browser tab (different prefix anyway, but the tag
    # makes the separation explicit in logs).
    if uid:
        prefix_hash = compute_prefix_cache_key(body)
        internal_req.kv_session_key = f"{uid}:cc:{prefix_hash}"

    # Tier-aware thinking control (Qwen team recommendation for
    # Qwen3.6-35B-A3B, which thinks by default). CC's traffic
    # naturally splits into two needs:
    #
    #   haiku tier  → subagents / Explore / Agent / title-gen /
    #                 compaction. Quick utility calls that just need
    #                 to fire a tool or produce a short label. Every
    #                 thinking token here is wasted compute. Disable.
    #
    #   sonnet/opus → main planning loop. Quality of reasoning is
    #                 directly load-bearing. Keep thinking on, and
    #                 enable preserve_thinking (Qwen3.6's agentic
    #                 feature) so prior reasoning carries across
    #                 turns — reduces redundant re-derivation AND
    #                 improves KV cache utilization per Qwen docs.
    #
    # No tier (model wasn't claude-*) means caller used an explicit
    # local model name — let llama-server's CLI defaults / GGUF
    # template apply unchanged.
    #
    # Wired via chat_template_kwargs, which llama-server forwards to
    # the chat template at render time. Models that don't recognize
    # these kwargs ignore them silently — safe for non-Qwen
    # backends that get aliased here too.
    if tier == "haiku":
        internal_req.chat_template_kwargs = {"enable_thinking": False}
    elif tier in ("sonnet", "opus"):
        internal_req.chat_template_kwargs = {
            "enable_thinking": True,
            "preserve_thinking": True,
        }

    # Claude Code (and any /v1/messages harness) gets the same scope-isolated
    # Augmentum-memory briefing as the OpenAI path. Reaching this tools path is
    # itself a strong harness signal, so default to claude_code when the
    # User-Agent doesn't name a tool. Tool-safe + fail-open.
    if uid:
        _harness = detect_harness(request) or "claude_code"
        _project = detect_project(request)
        schedule_harness_capture(
            internal_req, request.app.state, user_id=uid, harness=_harness,
            project=_project,
        )
        await inject_harness_context(
            internal_req, request.app.state, user_id=uid, harness=_harness,
            project=_project,
        )

    try:
        backend, clean_model = await registry.resolve_backend_with_fabric(
            internal_req.model, user_id=uid, session_id="",
        )
    except ModelUnavailableError as exc:
        log.warning(
            "anthropic_direct_model_unavailable",
            model=exc.model, diagnostic=exc.peer_diagnostic,
        )
        return None, _anthropic_error(
            str(exc), status=400, error_type="invalid_request_error",
        )
    except Exception as exc:
        return None, _anthropic_error(
            f"Backend resolution failed: {exc}", status=502,
        )
    internal_req.model = clean_model

    try:
        internal_resp = await backend.chat(internal_req)
    except Exception as exc:
        err_msg = str(exc).strip() or type(exc).__name__
        log.warning(
            "anthropic_direct_backend_failed",
            error=err_msg,
            error_type=type(exc).__name__,
            model=clean_model,
            exc_info=True,
        )
        return None, _anthropic_error(
            f"Backend error: {err_msg}", status=502, error_type="api_error",
        )

    return to_openai_chat_response(internal_resp), None


async def _dispatch_to_backend(
    body: AnthropicMessagesRequest, oa_body, request: Request,
    *, tier: str | None = None,
) -> tuple[dict | None, JSONResponse | None]:
    """Choose path based on whether the caller brought their own tools.

    With tools: direct backend call so ``tool_calls`` round-trip cleanly.
    ``tier`` (haiku/sonnet/opus/default/None) propagates from the
    claude-* alias resolution and shapes per-tier inference hints in
    the direct path.

    Without tools: full ``openai_chat`` orchestration (classifier, fabric,
    memory recall, knowledge pack injection, dream portrait, vision
    fallback, mode hints, KV lifecycle, telemetry — all the moat stuff).
    The no-tools path is for Augmentum's own chat-UI traffic that
    happens to come in via the Anthropic shape; tier hinting doesn't
    apply there because PassthroughHandler owns inference shaping.
    """
    if body.tools:
        return await _run_direct_backend_for_tools(body, oa_body, request, tier=tier)
    return await _run_openai_chat_safely(oa_body, request)


async def _run_openai_chat_safely(
    oa_body, request: Request,
) -> tuple[dict | None, JSONResponse | None]:
    """Call openai_chat as a function, return (oa_dict, error_response)."""
    try:
        oa_resp = await openai_chat_handler(oa_body, request)
    except Exception as exc:
        log.warning("anthropic_openai_chat_raised", error=str(exc), exc_info=True)
        return None, _anthropic_error(
            f"Backend error: {exc}", status=502, error_type="api_error",
        )

    if oa_resp.status_code != 200:
        # The OpenAI route returned a structured error JSON (e.g.
        # ModelUnavailableError surfaces as {"error": {"message", "type"}}).
        # Translate to Anthropic error shape preserving the message.
        try:
            payload = json.loads(oa_resp.body)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        err = payload.get("error") if isinstance(payload, dict) else {}
        msg = (err or {}).get("message") if isinstance(err, dict) else str(payload)
        msg = msg or "Backend returned non-200 status"
        et = "invalid_request_error" if oa_resp.status_code < 500 else "api_error"
        return None, _anthropic_error(msg, status=oa_resp.status_code, error_type=et)

    try:
        oa_dict = json.loads(oa_resp.body)
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("anthropic_openai_body_invalid_json", error=str(exc))
        return None, _anthropic_error(
            "Backend response was not valid JSON", status=502,
        )
    return oa_dict, None


async def _stream_anthropic_messages(
    body: AnthropicMessagesRequest, oa_body, request: Request,
    *, tier: str | None = None,
):
    """Faux-stream generator: emit message_start immediately, ping
    while the inner LLM call runs, then emit the full content sequence
    + message_delta + message_stop on completion."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # 1. Emit message_start NOW — this is the SDK invariant the
    # research called out: streams MUST start with message_start before
    # any other event. We also set X-Accel-Buffering: no on the response
    # so the bytes flush to the client immediately.
    yield build_message_start_event(
        model_name=body.model, message_id=message_id, input_tokens=0,
    )

    # 2. Race the inner LLM call against periodic ping emission.
    task = asyncio.create_task(_dispatch_to_backend(body, oa_body, request, tier=tier))
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_PING_INTERVAL_S)
            except TimeoutError:
                yield build_ping_event()
        oa_dict, error = task.result()
    except asyncio.CancelledError:
        # Client dropped — cancel inner task and re-raise so Starlette
        # cleans up properly.
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise
    except Exception as exc:
        log.warning("anthropic_stream_unexpected_error", error=str(exc), exc_info=True)
        # Emit a graceful closeout — message_delta + message_stop with
        # end_turn rather than leaving the stream half-open. The SDK
        # tolerates empty-content responses.
        yield _emit_empty_close(message_id)
        return

    if error is not None:
        # Read the error body and surface it inside the SSE stream as
        # an end_turn + empty content. CC renders the error from a
        # final marker; we can't easily inject a typed error event
        # inside an already-started message_start.
        try:
            err_payload = json.loads(error.body)
            err_msg = err_payload.get("error", {}).get("message", "")
        except Exception:
            err_msg = ""
        log.info("anthropic_stream_inner_error", message=err_msg[:200])
        # Open a text block with the error message so the user sees it
        # in CC's TUI instead of a silent close.
        if err_msg:
            yield _emit_error_text_block(err_msg)
        yield _emit_empty_close(message_id)
        return

    # 3. Emit content blocks + message_delta + message_stop.
    internal_resp = _adapt_openai_dict_to_internal(oa_dict)
    async for chunk in stream_internal_response_as_anthropic_sse(
        internal_resp, model_name=body.model, emit_message_start=False,
    ):
        yield chunk


def _emit_error_text_block(text: str) -> bytes:
    """Emit a one-shot text block carrying an error message. Useful
    for surfacing backend errors inside an already-started stream."""
    from augmentum.models.anthropic_compat import _sse_event
    out = bytearray()
    out.extend(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    out.extend(_sse_event("content_block_delta", {
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }))
    out.extend(_sse_event("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    }))
    return bytes(out)


def _emit_empty_close(message_id: str) -> bytes:
    """Emit message_delta + message_stop with no content blocks open."""
    from augmentum.models.anthropic_compat import _sse_event
    out = bytearray()
    out.extend(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }))
    out.extend(_sse_event("message_stop", {"type": "message_stop"}))
    return bytes(out)


# ─── Routes ──────────────────────────────────────────────────────────


@router.post("/messages")
async def anthropic_messages(body: AnthropicMessagesRequest, request: Request):
    """Claude Code-compatible /v1/messages endpoint.

    Streams (text/event-stream) when ``body.stream=true``, otherwise
    returns a single JSON message. Tolerates unknown query params
    (``?beta=true``) and Anthropic-specific request headers
    (``anthropic-version``, ``anthropic-beta``) — they don't need
    any code here; FastAPI doesn't reject unknown query/header values.
    """
    # Alias claude-* model names to a locally-available model BEFORE
    # translation so the entire downstream chain (backend resolution,
    # request building, response model echo) sees the resolved name.
    # See ``_resolve_claude_alias`` for the rationale and lookup
    # order. ``requested_model`` is preserved so Anthropic-shape
    # responses can echo what CC actually asked for (CC's TUI
    # surfaces this field in some flows).
    requested_model = body.model
    resolved, tier = _resolve_claude_alias(body.model)
    body.model = resolved

    oa_body = anthropic_request_to_openai(body)

    if body.stream:
        return StreamingResponse(
            _stream_anthropic_messages(body, oa_body, request, tier=tier),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    oa_dict, error = await _dispatch_to_backend(body, oa_body, request, tier=tier)
    if error is not None:
        return error
    internal_resp = _adapt_openai_dict_to_internal(oa_dict)
    return JSONResponse(internal_response_to_anthropic_message(
        internal_resp, model_name=body.model,
    ))


@router.post("/messages/count_tokens")
async def anthropic_count_tokens(body: dict, request: Request):
    """Token-budget estimator. CC consumes this for context-aware
    compaction — returning 404 here causes CC's request-handling state
    to silently degrade. Always return 200 with a best-effort estimate."""
    try:
        messages = body.get("messages") or []
        system = body.get("system")
        tools = body.get("tools")
        n = count_tokens_estimate(messages, system=system, tools=tools)
    except Exception as exc:
        log.warning("anthropic_count_tokens_failed", error=str(exc), exc_info=True)
        n = 0
    return JSONResponse({"input_tokens": int(n)})
