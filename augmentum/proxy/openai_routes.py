"""OpenAI-compatible API routes (/v1/*)."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

from augmentum.classifier.router import MODE_HEADER, Mode, RequestClassifier
from augmentum.config import settings
from augmentum.proxy.ollama_routes import _inject_voice_context, _resolve_request_think
from augmentum.image.schemas import OpenAIImageData, OpenAIImageRequest, OpenAIImageResponse
from augmentum.memory.integration import (
    apply_dream_injection_to_request,
    recall_and_inject,
    schedule_extraction,
)
from augmentum.models.base import (
    InternalChatRequest,
    Message,
    caption_via_router_fallback,
    inject_vision_prompt,
    resolve_chat_image_urls,
)
from augmentum.models.openai_compat import to_openai_chat_response
from augmentum.models.provider_registry import ModelUnavailableError
from augmentum.modes.passthrough.handler import PassthroughHandler, _CHAT_SYNTHESIS_HINT
from augmentum.proxy.handler_factory import (
    _resolve_passthrough_tools,
    apply_prompt_cache_key,
    compute_direct_prefix_cache_key,
    get_handler_for_mode,
    resolve_auto_invoke_tools,
    resolve_session_keys,
)
from augmentum.proxy.harness import (
    detect_harness,
    detect_project,
    inject_harness_context,
    schedule_harness_capture,
)
from augmentum.proxy.status_bus import bind_request_id, make_request_id
from augmentum.proxy.streaming import (
    stream_openai_chat_handler,
    stream_openai_chat_handler_with_extraction,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


# openai_compat raises ``RuntimeError("Backend returned NNN: <body>")`` when
# the upstream provider returns non-200. Same regex pattern is used by
# coder/handler.py for retry classification; kept local here to avoid a
# cross-module import cycle (handler.py imports from this layer indirectly
# via the mode dispatch path).
_BACKEND_STATUS_RE = re.compile(r"Backend returned (\d{3})")

# Substrings that mark a 5xx response as a context-window exhaustion rather
# than a real server-side fault. OpenAI's canonical answer for this is 400 +
# code=context_length_exceeded; some bridges (e.g. chatgpt-bridge) return
# 502 with the same message body, which we remap here.
_CONTEXT_WINDOW_MARKERS = (
    "context window",
    "context length",
    "context_length_exceeded",
    "exceeds the context",
    "maximum context",
)


def _backend_runtime_error_to_response(exc: RuntimeError) -> JSONResponse | None:
    """Map an openai_compat backend RuntimeError to a clean JSONResponse.

    Returns ``None`` when the exception doesn't match the known shape, so
    the caller can re-raise (preserving the existing bubble-up behavior for
    truly unexpected failures).

    Without this translation, a backend 5xx surfaces to the client as a
    generic ASGI 500 with a Rich traceback, instead of a structured error
    the chat UI can render. Direct mode is the most affected path because
    it deliberately skips the fallback-to-passthrough recovery the standard
    chat route does (Direct's contract is verbatim pass-through; see
    augmentum/modes/direct/handler.py module docstring).
    """
    raw = str(exc) or ""
    match = _BACKEND_STATUS_RE.search(raw)
    if not match:
        return None
    upstream_status = int(match.group(1))
    body_lower = raw.lower()

    # 5xx + context-window marker → 400 context_length_exceeded. The caller
    # sent too much input; that's a client problem, not a server problem,
    # and the UI shows it differently (retry vs. shorten conversation).
    if upstream_status >= 500 and any(m in body_lower for m in _CONTEXT_WINDOW_MARKERS):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        "This conversation has exceeded the model's context window. "
                        "Try a model with a larger window, summarise older turns, or "
                        "start a fresh chat."
                    ),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                }
            },
        )

    # Otherwise pass the upstream status through with a structured envelope.
    # We keep the raw body in ``message`` so debugging stays possible — it
    # was already going to surface anyway via the ASGI 500 traceback.
    return JSONResponse(
        status_code=upstream_status,
        content={
            "error": {
                "message": raw,
                "type": "backend_error",
                "code": f"backend_{upstream_status}",
            }
        },
    )


async def _dispatch_direct(
    internal_req: InternalChatRequest,
    body: "OpenAIChatRequest",
    backend,
    app_state,
    *,
    user_id: str,
    session_id: str,
    telemetry_reason: str,
) -> JSONResponse | Any:
    """Dispatch a Direct-mode chat request — no Augmentum injection.

    Skips every injector that the standard route runs: media context,
    memory recall, knowledge packs, dream, vision caption fallback,
    file tokens, SSOS tools, mode inference hints, prompt-cache key,
    lifecycle, dream-scheduler bumps, companion-runtime events. The
    backend sees exactly what the caller sent.

    Telemetry: classifier result was already logged at the route head;
    we add a single ``direct_dispatch`` log line so logs make it easy
    to see how many turns landed on this tier.
    """
    from augmentum.modes.direct.handler import DirectHandler

    log.info(
        "direct_dispatch",
        model=internal_req.model,
        user_id=user_id,
        session_id=session_id,
        stream=bool(body.stream),
        reason=telemetry_reason,
    )

    handler = DirectHandler(backend=backend)
    if body.stream:
        return await stream_openai_chat_handler(internal_req, handler)
    internal_resp = await handler.handle(internal_req)
    return JSONResponse(to_openai_chat_response(internal_resp))


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


# --- Pydantic models ---


class OpenAIMessage(BaseModel):
    role: str
    # None accepted: OpenAI-spec assistant turns that carry tool_calls send
    # ``content: null`` (pi/OpenCode/Cursor/… all do). Rejecting None 400s the
    # SECOND turn of every agentic tool loop.
    content: str | list[dict] | None = ""
    # Reasoning models replay their own reasoning_content on subsequent
    # turns; DeepSeek 400s without it. Optional — non-reasoning clients
    # never set it.
    reasoning_content: str | None = None
    # Agentic harnesses drive multi-turn tool loops: the assistant turn carries
    # ``tool_calls`` (with content: null), and the following tool-result turn
    # carries ``tool_call_id``. Capture + forward both so the model sees its own
    # prior call paired with the result — without this the loop is severed.
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    voice_input: bool = False
    think: bool | None = None
    # Per-request reasoning effort for OpenAI-family models. Valid:
    # "minimal" | "low" | "medium" | "high" | "xhigh". None = let the
    # mode hint apply (coder/agentic=high, narrative/passthrough=
    # minimal, analytical=low). Adapter layer gates transmission so a
    # non-OpenAI provider never sees this field. Accepted on the
    # OpenAI surface as the standard ``reasoning_effort`` field; both
    # the chat composer (UI sends the user's per-turn override) and
    # external API callers can drive it.
    reasoning_effort: str | None = None
    # Per-request chat-template kwargs (e.g. {"enable_thinking": true}).
    # Forwarded verbatim to InternalChatRequest — see ollama_routes.py's
    # twin field for the full routing note. vLLM/SGLang accept the same
    # top-level field, so external /v1 callers can drive it too.
    chat_template_kwargs: dict[str, Any] | None = None
    # Per-request Qwen 3.6 preserve_thinking override. Falls through to the
    # per-user UI setting when None. Other model families ignore the kwarg.
    preserve_thinking: bool | None = None
    # Continue-button signal — last message is a partial assistant turn
    # the model should extend verbatim. See ollama_routes.py for backend
    # mechanics. External /v1/chat/completions callers can drive this too.
    continue_last_assistant: bool = False

    model_config = {"extra": "allow"}

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model name must not be empty")
        return v.strip()

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[OpenAIMessage]) -> list[OpenAIMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


# --- Conversion ---


def _parse_openai_content(content: str | list[dict] | None) -> tuple[str, list[str] | None]:
    """Parse OpenAI message content which may be a string, a content-parts array,
    or None (assistant tool-call turns send content: null).

    Returns (text, images) where images is a list of base64 data-URIs or URLs,
    or None if no images are present.
    """
    if content is None:
        return "", None
    if isinstance(content, str):
        return content, None

    text_parts: list[str] = []
    images: list[str] = []
    for part in content:
        ptype = part.get("type", "")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url:
                images.append(url)
    return "\n".join(text_parts), images or None


def to_internal_chat_request(req: OpenAIChatRequest, *, think: bool | None = None) -> InternalChatRequest:
    """Convert OpenAI chat request to internal format."""
    messages = []
    for m in req.messages:
        text, images = _parse_openai_content(m.content)
        messages.append(Message(
            role=m.role,
            content=text,
            images=images,
            thinking=m.reasoning_content,
            tool_calls=m.tool_calls,
            tool_call_id=m.tool_call_id,
        ))

    stop = req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None)

    fmt = None
    if req.response_format and req.response_format.get("type") == "json_object":
        fmt = "json"

    return InternalChatRequest(
        model=req.model,
        messages=messages,
        stream=req.stream,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        seed=req.seed,
        tools=req.tools,
        format=fmt,
        think=settings.think_enabled if think is None else bool(think),
        chat_template_kwargs=req.chat_template_kwargs,
        reasoning_effort=req.reasoning_effort,
        preserve_thinking=req.preserve_thinking,
        voice_input=req.voice_input,
        continue_last_assistant=bool(getattr(req, "continue_last_assistant", False)),
    )


# --- Route handlers ---


@router.post("/chat/completions")
async def openai_chat(body: OpenAIChatRequest, request: Request) -> JSONResponse:
    """Handle OpenAI /v1/chat/completions requests."""
    # Bind a request id for log correlation + stage event stamping.
    # Not reset in this function: for streaming responses the stream
    # iterates AFTER the route returns and FastAPI runs it in the same
    # asyncio task, so resetting here would clear the binding before
    # the streaming generator emits its first chunk. ContextVars are
    # task-local and the task dies with the request, so leakage is
    # bounded by request lifetime regardless.
    bind_request_id(make_request_id())

    registry = request.app.state.provider_registry
    classifier: RequestClassifier = request.app.state.classifier

    internal_req = to_internal_chat_request(body, think=_resolve_request_think(body.think, request))

    # Voice input compensation — tolerates STT errors
    if internal_req.voice_input:
        _inject_voice_context(internal_req)

    # Classify first (strips a/n/p/ mode prefix from model name).
    # When companion_dispatch_routes_chat is on AND the runtime is up,
    # the chat router consults the companion's dispatcher; otherwise
    # it's a transparent wrapper around the classifier.
    mode_override = request.headers.get(MODE_HEADER)
    uid = _user_id(request)
    # External API clients (authed with an sk-aug key: Open WebUI, Cursor, …)
    # get clean DIRECT passthrough by DEFAULT — no classifier, no memory/persona
    # harness (which is why a bare "hey there" was ballooning to ~11k tokens).
    # They opt INTO a mode explicitly via a model prefix (a/ n/ p/ g/ c/ d/) or
    # the X-Augmentum-Mode header. Augmentum's own UI uses /api/chat, not this
    # endpoint, and browser-session requests here are unaffected.
    if not mode_override and request.scope.get("authed_via_api_key"):
        from augmentum.classifier.router import MODE_PREFIXES
        _model_name = body.model or ""
        if not any(_model_name.startswith(p) for p in MODE_PREFIXES):
            mode_override = "direct"
    from augmentum.companion_runtime.chat_router import resolve_chat_mode
    classification = await resolve_chat_mode(
        request.app.state,
        internal_req,
        classifier=classifier,
        mode_override=mode_override,
        user_id=uid,
    )
    log.info(
        "classified",
        mode=classification.mode.value,
        confidence=round(classification.confidence, 2),
        reason=classification.reason,
    )

    # External IDE coding agents (OpenCode, Claude Code, Cursor, Aider, …):
    # inject a scope-isolated Augmentum-memory briefing + professional working
    # conventions, and route through clean direct passthrough (the `or harness`
    # below) so they get the briefing but NOT the companion/dream/memory
    # injectors meant for the in-app chat. Harness-agnostic (header or
    # User-Agent); fail-open; tool-safe. See augmentum/proxy/harness.py.
    harness = detect_harness(request)
    if harness:
        # Capture explicit teachings from the ORIGINAL user message first, then
        # inject (which prepends the briefing to that message).
        project = detect_project(request)
        schedule_harness_capture(
            internal_req, request.app.state, user_id=uid, harness=harness,
            project=project,
        )
        await inject_harness_context(
            internal_req, request.app.state, user_id=uid, harness=harness,
            project=project,
        )

    # Smart routing — resolve backend from (prefix-stripped) model name.
    # ``resolve_backend_with_fabric`` consults the fabric director when
    # the model isn't in the local map AND a connected peer advertises
    # it; on solo installs (no fabric) it's identical to
    # ``resolve_backend_for_model``. user_id/session_id resolved below
    # for the telemetry call inside the helper.
    workspace_id = request.headers.get("X-Augmentum-Workspace", "")
    coder_workspace = workspace_id if classification.mode == Mode.CODER else ""
    session_id, internal_req.kv_session_key = resolve_session_keys(
        request, internal_req, user_id=uid, workspace_id=coder_workspace,
    )
    # Direct-mode (d/) callers are external harnesses (OpenCode, …) without our
    # session header, so kv_session_key came back empty → no slot save/restore.
    # Pin them to a stable per-conversation key so a UI/companion turn landing on
    # slot 0 between two harness turns doesn't force a full re-prefill. Local
    # engine only; cloud backends ignore kv_session_key. See
    # compute_direct_prefix_cache_key.
    if (classification.mode == Mode.DIRECT or harness) and not internal_req.kv_session_key:
        internal_req.kv_session_key = compute_direct_prefix_cache_key(uid, internal_req)
    internal_req.kv_mode = classification.mode.value
    # Bump activity for coder workspaces so the idle reaper doesn't
    # auto-stop a container the user is actively chatting to. Coder
    # routes hit ``_owns_workspace`` for their own activity bump;
    # chat completions don't go through that gate, so the mark
    # happens here at the dispatch site.
    if coder_workspace:
        _mgr = getattr(request.app.state, "container_manager", None)
        if _mgr is not None:
            try:
                await _mgr.mark_active(coder_workspace)
            except Exception:
                # Failure here means the idle-reaper won't see this
                # activity bump — container may auto-stop mid-conversation
                # if other activity also fails. Worth surfacing at debug
                # since the user-facing path still works.
                log.debug("coder_workspace_mark_active_failed",
                          workspace=coder_workspace, exc_info=True)
    try:
        backend, clean_model = await registry.resolve_backend_with_fabric(
            internal_req.model, user_id=uid, session_id=session_id,
        )
    except ModelUnavailableError as exc:
        # The model isn't on this node and no connected peer has it. Surface
        # the structured diagnostic so the UI can render "peer offline" /
        # "peer dropped the model" rather than a generic upstream error.
        log.warning(
            "openai_chat_model_unavailable",
            model=exc.model, diagnostic=exc.peer_diagnostic,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "model_unavailable",
                    "model": exc.model,
                    "fabric": exc.peer_diagnostic,
                }
            },
        )
    internal_req.model = clean_model
    from augmentum.proxy.primary_model import adopt_primary_chat_model
    await adopt_primary_chat_model(request.app.state, clean_model)

    # Direct mode short-circuit — skip every injector before the request
    # is dispatched. See augmentum/modes/direct/handler.py for the
    # contract. Reached when the caller opts in via ``d/<model>`` prefix or
    # ``X-Augmentum-Mode: direct``, OR when the caller is a detected external
    # harness (clean passthrough + our own briefing already injected above).
    if classification.mode == Mode.DIRECT or harness:
        # Direct mode deliberately skips the fallback-to-passthrough
        # recovery the standard chat path uses (see
        # augmentum/modes/direct/handler.py module docstring — its
        # contract is verbatim pass-through, no degraded mode). That
        # means backend RuntimeErrors here have no in-mode recovery and
        # would otherwise bubble as ASGI 500. We translate the known
        # ``Backend returned NNN: <body>`` shape to a structured error
        # the UI can render; truly unexpected errors still re-raise.
        # Only covers the non-streaming path — streaming errors land
        # mid-SSE after headers are sent and need separate handling.
        # Capture direct/default turns as the grounding "spine" lane. The
        # capture is inert unless training_capture_enabled and (when
        # training_capture_user_id is pinned) the user matches — so external
        # harness traffic from OTHER users never records, and the :D lane is
        # excluded from seed by default at assemble time. Streaming direct is
        # NOT yet captured: its SSE generator outlives this scope, so it needs
        # the begin_capture-inside-the-stream-wrapper pattern (see
        # augmentum/proxy/streaming.py) — tracked as follow-up.
        from augmentum.training.trace_context import begin_capture, end_capture
        _d_ctx, _d_tok = (None, None)
        if not body.stream:
            _d_ctx, _d_tok = begin_capture(
                user_id=uid or "default", session_id=session_id, mode="direct",
            )
        try:
            return await _dispatch_direct(
                internal_req, body, backend, request.app.state,
                user_id=uid, session_id=session_id, telemetry_reason=classification.reason,
            )
        except RuntimeError as exc:
            translated = _backend_runtime_error_to_response(exc)
            if translated is None:
                raise
            log.warning(
                "direct_dispatch_backend_error_translated",
                model=internal_req.model,
                user_id=uid,
                session_id=session_id,
                status=translated.status_code,
            )
            return translated
        finally:
            # Single close point on every exit (success, translated error, or
            # re-raise) — end_capture is a no-op when the scope is inert.
            end_capture(_d_ctx, _d_tok)

    # "Currently listening to X" context from the frontend media player
    # — small 1-sentence system prefix so the LLM grounds naturally in
    # what the user is actively hearing. Companion-scoped (see the injector
    # docstring): no-ops for narrative/passthrough/etc. so it can't pollute
    # RP or break the KV prefix. No-ops when no audio is active.
    from augmentum.proxy.media_context import inject_media_context
    inject_media_context(internal_req, request, mode=classification.mode.value)

    # Resolve passthrough tools from header + config. Pass the latest user
    # message so the companion scheduling substrate is intent-gated (it's
    # force-available otherwise, which knocks plain chat off the streaming
    # fast-path).
    tools_header = request.headers.get("X-Augmentum-Tools")
    _latest_user_msg = next(
        (m.content for m in reversed(internal_req.messages) if m.role == "user"),
        "",
    )
    pt_tools = _resolve_passthrough_tools(
        request.app.state, tools_header, query=_latest_user_msg,
    )
    # Analytical filter is selector-only (no config-default merge).
    from augmentum.proxy.handler_factory import resolve_analytical_tools
    analytical_enabled = resolve_analytical_tools(tools_header)

    # Parse per-message flow tune overrides (Quick Tune panel)
    flow_tune_header = request.headers.get("X-Augmentum-Flow-Tune")
    flow_tune = None
    if flow_tune_header:
        try:
            flow_tune = json.loads(flow_tune_header)
        except (json.JSONDecodeError, TypeError):
            log.warning("invalid_flow_tune_header", raw=flow_tune_header[:200])

    # Standalone complexity hint header — merged into flow_tune for convenience
    complexity_hint = request.headers.get("X-Augmentum-Complexity", "").strip().lower()
    if complexity_hint in ("simple", "moderate", "complex"):
        if flow_tune is None:
            flow_tune = {}
        flow_tune.setdefault("complexity", complexity_hint)

    # Handler resolution context (uid + session already resolved above
    # for the fabric-aware backend resolve).
    mem_user_id = uid or "default"
    coder_strategy = request.headers.get("X-Augmentum-Coder-Strategy", "")
    explicit_flow_id = request.headers.get("X-Augmentum-Flow", "")
    handler = get_handler_for_mode(
        classification.mode, backend, session_id, request.app.state,
        passthrough_tools=pt_tools,
        analytical_enabled_tools=analytical_enabled,
        tool_synthesis_hint=_CHAT_SYNTHESIS_HINT if pt_tools else "",
        workspace_id=workspace_id,
        flow_tune=flow_tune,
        explicit_flow_id=explicit_flow_id,
        user_id=uid,
        coder_strategy=coder_strategy,
    )
    # Per-turn auto-invoke allowlist (button intent): only tools explicitly
    # named in X-Augmentum-Tools may direct-invoke this turn. 'all'/absent →
    # auto-fire nothing, so config-default / blanket tools stay schema-only
    # instead of hijacking every message onto the direct-invoke path.
    if hasattr(handler, "_auto_invoke_tools"):
        handler._auto_invoke_tools = resolve_auto_invoke_tools(
            request.headers.get("X-Augmentum-Tools")
        )

    # Inject relevant memories into context before processing
    # Skip for coder mode — personal memories confuse the coding agent
    memory_query = request.headers.get("X-Augmentum-Memory-Query", "")
    if classification.mode.value != "coder":
        await recall_and_inject(
            internal_req, request.app.state,
            user_id=mem_user_id,
            mode=classification.mode.value, session_id=session_id,
            memory_query=memory_query,
        )

    # Inject pack context — independent of the memory-injection toggles.
    # Per-mode gates live in inject_pack_context; narrative defaults off.
    # Returned dict (when non-None) is stashed on the request so the streaming
    # layer can surface it on the final chunk for the UI chip.
    if session_id:
        from augmentum.knowledge.injection import inject_pack_context
        internal_req.pack_injection = await inject_pack_context(
            internal_req, request.app.state,
            user_id=mem_user_id,
            session_id=session_id,
            mode=classification.mode.value,
        )

    # Inject dream portrait + relevant dream entries (passthrough/analytical/agentic only).
    # The helper handles persona_id, per-user opt-in, and recall toggles.
    if classification.mode.value not in ("narrative", "coder"):
        await apply_dream_injection_to_request(internal_req, request.app.state, mem_user_id)

    # Resolve stored chat image URLs to inline base64 for the LLM backend
    await resolve_chat_image_urls(internal_req, request.app.state)

    # Vision caption fallback — when the resolved backend can't natively
    # read images but the SmolVLM sibling is up, caption each attachment
    # and inline the result so a text-only primary can still respond
    # usefully to image uploads instead of erroring out.
    await caption_via_router_fallback(internal_req, request.app.state, backend)

    # Expand [file:fi_xxx] tokens (from Files panel "Reference in Chat")
    from augmentum.proxy.file_token_resolver import resolve_file_tokens
    await resolve_file_tokens(
        internal_req,
        user_id=uid,
        file_index=getattr(request.app.state, "file_index", None),
        vfs=getattr(request.app.state, "vfs", None),
    )

    # Ensure image attachments have adequate prompting for VL models
    inject_vision_prompt(internal_req)

    # Training data generation overrides
    training_mode = request.headers.get("X-Augmentum-Training", "").lower() == "true"
    if training_mode:
        internal_req.training_mode = True
    system_inject = request.headers.get("X-Augmentum-System-Inject", "")
    if system_inject:
        internal_req.messages.insert(0, Message(role="system", content=system_inject))

    mode_value = classification.mode.value

    # Apply mode-aware inference defaults (temp, top_p, max_tokens,
    # reasoning_effort). Adapter-level gates ensure reasoning_effort
    # only reaches OpenAI-family + xAI targets.
    from augmentum.modes.inference_hints import apply_mode_hints
    apply_mode_hints(internal_req, mode_value)

    # Per-model sampling profiles (OpenWebUI-style layering). Resolve the
    # effective sampling for THIS model and fill whatever's still unset:
    #   per-call (body) ▸ mode hints (above) ▸ per-model (user→seed) ▸ family.
    # ``apply_to_request`` only writes fields that are still None, so an
    # explicit per-call value and the mode hints above both win — this just
    # makes "swap the model, the right sampling follows" actually happen
    # (e.g. Qwen3 → 0.6/0.95/20, Gemma-4 → 1.0/0.95/64) instead of leaving
    # the backend on its own defaults. Never raises (sampling must not break
    # a chat). See augmentum/models/sampling_profiles.py.
    from augmentum.models.sampling_profiles import apply_to_request as _apply_sampling
    await _apply_sampling(
        internal_req,
        getattr(request.app.state, "settings_store", None),
        user_id=uid,
    )

    # Stable prompt-cache key — see apply_prompt_cache_key for the rationale
    # and the (session, user) axis choice. Shared with the Ollama-shaped chat
    # ingress so both surfaces get identical sticky routing.
    apply_prompt_cache_key(internal_req, user_id=uid, session_id=session_id)

    # Track session activity for KV lifecycle management
    lifecycle = getattr(request.app.state, "session_lifecycle", None)
    if lifecycle and session_id:
        model_name = internal_req.model or ""
        lifecycle.touch(session_id, mode_value, model_name)

    # Fetch context window size for utilization display
    ctx_len = 0
    try:
        ctx_len = await backend.get_context_length(internal_req.model)
    except Exception as exc:
        log.debug("context_length_fetch_failed", model=internal_req.model, error=str(exc))

    from augmentum.companion_runtime.bus import emit_safe, maybe_emit_mode_changed
    await maybe_emit_mode_changed(
        request.app.state, session_id, mode_value,
        reason=classification.reason, confidence=classification.confidence,
    )
    await emit_safe(request.app.state, "chat.turn_started", {
        "mode": mode_value,
        "user_id": uid,
        "session_id": session_id,
        "model": internal_req.model,
        "reason": classification.reason,
        "wire_format": "openai",
        "stream": bool(body.stream),
    })

    try:
        if body.stream:
            resp = await stream_openai_chat_handler_with_extraction(
                internal_req, handler, request.app.state, session_id, mode_value,
                context_length=ctx_len, user_id=mem_user_id,
            )
            # Save mode state after stream completes (async, non-blocking)
            if lifecycle and session_id:
                asyncio.ensure_future(lifecycle.on_message_complete(session_id, handler))
            return resp

        from augmentum.training.trace_context import begin_capture, end_capture
        _cap_ctx, _cap_tok = begin_capture(user_id=mem_user_id, session_id=session_id, mode=mode_value)
        try:
            internal_resp = await handler.handle(internal_req)
        except Exception:
            end_capture(_cap_ctx, _cap_tok, error="handler_error")
            raise
        end_capture(_cap_ctx, _cap_tok)
        schedule_extraction(request.app.state, internal_req, internal_resp.message.content, session_id, user_id=mem_user_id, mode=mode_value)
        from augmentum.training.capture import capture_training_trace
        capture_training_trace(internal_req, internal_resp.message.content, session_id, mem_user_id, mode_value)

        # Save mode state after non-stream response
        if lifecycle and session_id:
            await lifecycle.on_message_complete(session_id, handler)

        # Notify dream scheduler of activity (per-user counters)
        dream_scheduler = getattr(request.app.state, "dream_scheduler", None)
        if dream_scheduler:
            dream_scheduler.notify_message(user_id=mem_user_id)
            dream_scheduler.notify_request(user_id=mem_user_id)

        from augmentum.companion_runtime.bus import emit_chat_turn_completed
        # Extract last user turn for the salience pipeline. Non-stream
        # paths reuse the same internal_req we just dispatched.
        _user_text = ""
        for _m in reversed(internal_req.messages or []):
            if getattr(_m, "role", "") == "user":
                _user_text = getattr(_m, "content", "") or ""
                break
        await emit_chat_turn_completed(
            request.app.state,
            mode=mode_value,
            user_id=uid,
            session_id=session_id,
            wire_format="openai",
            stream=False,
            user_text=_user_text,
            assistant_text=internal_resp.message.content,
        )

        return JSONResponse(to_openai_chat_response(internal_resp))
    except Exception:
        if not isinstance(handler, PassthroughHandler):
            log.warning("narrative_handle_failed_falling_back", exc_info=True)
            fallback = PassthroughHandler(
                backend=backend,
                custom_flow_store=getattr(request.app.state, "custom_flow_store", None),
            )
            if body.stream:
                return await stream_openai_chat_handler(internal_req, fallback)
            internal_resp = await fallback.handle(internal_req)
            return JSONResponse(to_openai_chat_response(internal_resp))
        raise


class OpenAIEmbeddingsRequest(BaseModel):
    """OpenAI ``/v1/embeddings`` request shape.

    See https://platform.openai.com/docs/api-reference/embeddings/create.
    ``encoding_format`` and ``dimensions`` and ``user`` are accepted for
    SDK compatibility but currently ignored — the underlying backends
    don't act on them.
    """
    model: str
    input: str | list[str]
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None


async def _internal_embeddings_response(inputs: list) -> JSONResponse | None:
    """Serve /v1/embeddings from the in-process ``EmbeddingService`` —
    the same nomic embedder every internal consumer (memory recall,
    knowledge packs, coder auto-recall) already runs on.

    Every input is embedded with *document* semantics so callers get one
    consistent vector space no matter how they mix queries and documents
    through the same endpoint. The envelope reports the TRUE model name
    so callers are never misled about what produced the vectors.

    Returns None when the fallback can't serve (non-string inputs — the
    OpenAI spec allows token arrays we don't support here — or the
    embedder is unavailable), letting the caller fall through to its
    error response.
    """
    if not inputs or not all(isinstance(s, str) for s in inputs):
        return None
    try:
        from augmentum.memory.embeddings import EmbeddingService
        vectors = await asyncio.to_thread(EmbeddingService.embed, list(inputs))
    except Exception:
        log.warning("openai_embeddings_internal_fallback_failed", exc_info=True)
        return None
    approx_tokens = sum(len(s) // 4 for s in inputs)  # ~4 chars/token
    return JSONResponse({
        "object": "list",
        "model": EmbeddingService.MODEL_NAME,
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens},
    })


@router.post("/embeddings")
async def openai_embeddings(body: OpenAIEmbeddingsRequest, request: Request) -> JSONResponse:
    """OpenAI-shape embeddings endpoint.

    Resolves the requested model to a backend via the standard registry
    (fabric-aware), then delegates to the backend's ``embeddings`` method
    when present. Returns the OpenAI-canonical envelope so SDK clients
    (OpenAI SDK, LangChain, LlamaIndex, Continue, Cursor's indexer)
    work without a shim. Falls back to the Ollama proxy when no backend
    implements embeddings — covers the "user has an Ollama embedding
    model but Augmentum doesn't know about its embedding capability"
    case.
    """
    registry = request.app.state.provider_registry
    uid = _user_id(request)

    inputs = body.input if isinstance(body.input, list) else [body.input]

    # Bound the batch BEFORE any embedding work (including the internal
    # fallback below) so an abusive caller can't pin an embedder with a
    # giant request. Default-on (the rate limiter is opt-in); 0 disables.
    _max_items = int(getattr(settings, "api_embeddings_max_items", 2048))
    if _max_items > 0 and len(inputs) > _max_items:
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": f"Too many inputs: {len(inputs)} (limit {_max_items})",
                "type": "invalid_request_error",
            }},
        )
    _max_chars = int(getattr(settings, "api_embeddings_max_chars", 1_000_000))
    if _max_chars > 0:
        total_chars = sum(len(s) for s in inputs if isinstance(s, str))
        if total_chars > _max_chars:
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": f"Input too large: {total_chars} chars (limit {_max_chars})",
                    "type": "invalid_request_error",
                }},
            )

    try:
        backend, clean_model = await registry.resolve_backend_with_fabric(
            body.model, user_id=uid, session_id="",
        )
    except ModelUnavailableError as exc:
        # Unknown model name (e.g. an SDK default like
        # "text-embedding-3-small") — serve from the internal embedder
        # rather than bouncing the request on a name we were never going
        # to match anyway. The response reports the true model.
        fallback = await _internal_embeddings_response(inputs)
        if fallback is not None:
            return fallback
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": str(exc),
                    "type": "model_unavailable",
                    "model": exc.model,
                }
            },
        )

    # Try the backend's native embeddings method first. llama-server's
    # ``embeddings`` already returns OpenAI-shape — pass through if so.
    raw = None
    embed_fn = getattr(backend, "embeddings", None)
    if callable(embed_fn):
        try:
            raw = await embed_fn(input=inputs, model=clean_model)
        except Exception:
            log.warning("openai_embeddings_backend_failed",
                        model=clean_model, exc_info=True)
            raw = None

    # If the backend returned an OpenAI-shaped envelope (data list of
    # {embedding, index}), use it as-is — that's the happy path for
    # llama-server.
    if isinstance(raw, dict) and isinstance(raw.get("data"), list) and raw["data"]:
        out = dict(raw)
        out.setdefault("object", "list")
        out.setdefault("model", clean_model)
        out.setdefault("usage", {"prompt_tokens": 0, "total_tokens": 0})
        return JSONResponse(out)

    # Fallback: try the Ollama embed endpoint and reshape into OpenAI.
    # Ollama returns ``{embeddings: [[...], [...]], prompt_eval_count: N}``.
    if settings.ollama_base_url:
        try:
            client = request.app.state.http_client
            resp = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": clean_model, "input": inputs},
            )
            if resp.status_code == 200:
                ollama_body = resp.json()
                vectors = ollama_body.get("embeddings") or []
                tokens = int(ollama_body.get("prompt_eval_count") or 0)
                return JSONResponse({
                    "object": "list",
                    "model": clean_model,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": v}
                        for i, v in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                })
        except Exception:
            log.warning("openai_embeddings_ollama_fallback_failed", exc_info=True)

    # Final fallback: the internal embedder. A default install has no
    # llama-server embedding slot and no Ollama, which left this endpoint
    # a guaranteed 501 for the SDK clients it exists for (Cursor's
    # indexer, LangChain, LlamaIndex) — while memory/knowledge/coder
    # recall run a perfectly good nomic embedder in-process.
    fallback = await _internal_embeddings_response(inputs)
    if fallback is not None:
        return fallback

    # Nothing worked — return 501 so callers know Augmentum can't serve
    # embeddings for this model rather than silently returning empty.
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": (
                    f"No embeddings backend available for model {clean_model!r} "
                    "and the internal embedder is unavailable. Configure a "
                    "llama-server with an embedding model loaded, or set "
                    "ollama_base_url with an embedding-capable model."
                ),
                "type": "embeddings_unavailable",
                "model": clean_model,
            }
        },
    )


@router.get("/models")
async def openai_models(request: Request) -> JSONResponse:
    """List models in OpenAI format from all backends.

    By DEFAULT lists each model ONCE by its base (direct) name — external
    clients (Open WebUI, OpenRouter-style pickers) don't want the same model
    repeated for every routing mode. The mode prefixes (a/ analytical, n/
    narrative, p/ passthrough) still WORK when a caller types them; they're
    just not enumerated here (that turned ~400 models into 2k+ near-duplicates).
    Pass ``?include_modes=1`` to get the full prefixed list back. Load balancers
    (lb/) are distinct named entries and always listed.
    """
    registry = request.app.state.provider_registry
    uid = _user_id(request)
    # Opt-in: enumerate a/ n/ p/ variants. Default off — one entry per model.
    include_modes = request.query_params.get("include_modes", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    all_models = []
    seen_names: dict[str, list[str]] = {}

    for key, backend in registry.backends.items():
        # Routing-only backends (e.g. the secondary engine slot) share the
        # primary's GGUF catalog; listing them here would collide every
        # model into name@key. They're reachable via their pin instead.
        if registry.is_listing_excluded(key) is True:
            continue
        # Per-user provider visibility (migration 305): a private provider's
        # models only appear for its owner; shared/global ones for everyone.
        if not registry.provider_visible_to(key, uid):
            continue
        try:
            models = await backend.list_models()
            for m in models:
                all_models.append((key, m))
                seen_names.setdefault(m.name, []).append(key)
        except Exception:
            log.debug("openai_model_list_failed", backend=key, exc_info=True)

    data = []
    for key, m in all_models:
        name = m.name
        if len(seen_names.get(m.name, [])) > 1:
            name = f"{m.name}@{key}"
        # Per-model limits: the model's OWN advertised window (captured in
        # list_models for providers that publish it — e.g. OpenRouter's
        # per-model context_length) wins; else the backend's provider-profile
        # ceiling (provider_profiles.py). Surfaced so external harnesses
        # (OpenCode/Claude Code) size their context compactor from the source
        # of truth instead of a hardcoded guess. Omitted when both are unknown.
        mctx = int(getattr(m, "context_length", 0) or 0)
        mout = int(getattr(m, "max_output", 0) or 0)
        if not mctx or not mout:
            prof = getattr(registry.backends.get(key), "_profile", None)
            if prof is not None:
                mctx = mctx or int(getattr(prof, "max_context", 0) or 0)
                mout = mout or int(getattr(prof, "max_output", 0) or 0)
        # Base (direct) name by default; a/ n/ p/ variants only on opt-in.
        variants = (
            (name, f"a/{name}", f"n/{name}", f"p/{name}")
            if include_modes else (name,)
        )
        for model_id in variants:
            entry = {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": key,
            }
            if mctx:
                entry["max_context"] = mctx
            if mout:
                entry["max_output"] = mout
            data.append(entry)

    # Phase 8 — merge peer-advertised LLM models. See ollama_routes
    # ollama_tags() for the long-form rationale. Bare names (Option A);
    # local-first dedup; peer-peer first-match-wins. The Ollama-compat
    # /api/tags variant carries peer metadata in ``details``; the
    # OpenAI surface uses ``owned_by`` for the peer node_id and adds
    # a small ``augmentum_peer`` extension dict for icon/hostname so
    # the dropdown can decorate.
    try:
        coordinator = getattr(request.app.state, "fabric_coordinator", None)
        if coordinator is not None:
            _peer_seen: set[str] = set()
            for node_id in coordinator.connected_peer_ids():
                state = coordinator.peer_state(node_id)
                if state is None or state.paired is None:
                    continue
                peer_icon = state.paired.icon or ""
                peer_hostname = state.paired.hostname or node_id[:12]
                for cap in state.capabilities:
                    if getattr(cap, "kind", "") != "llm.inference":
                        continue
                    model_id = getattr(cap, "model_id", "") or ""
                    if not model_id:
                        continue
                    if model_id in seen_names or model_id in _peer_seen:
                        continue
                    _peer_seen.add(model_id)
                    peer_ext = {
                        "node_id": node_id,
                        "hostname": peer_hostname,
                        "icon": peer_icon,
                    }
                    data.append({
                        "id": model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": f"peer:{node_id}",
                        "augmentum_peer": peer_ext,
                    })
                    if include_modes:
                        for prefix in ("a/", "n/", "p/"):
                            data.append({
                                "id": f"{prefix}{model_id}",
                                "object": "model",
                                "created": 0,
                                "owned_by": f"peer:{node_id}",
                                "augmentum_peer": peer_ext,
                            })
    except Exception as exc:
        log.debug("openai_peer_merge_failed", error=repr(exc))

    # Load balancers as selectable models
    try:
        store = getattr(request.app.state, "balancer_store", None)
        if store:
            balancers = await store.list_balancers()
            for b in balancers:
                if not b.enabled:
                    continue
                data.append({
                    "id": f"lb/{b.name}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "balancer",
                })
    except Exception:
        log.debug("balancer_list_in_models_failed", exc_info=True)

    return JSONResponse({"object": "list", "data": data})


@router.get("/image-models")
async def openai_image_models(request: Request) -> JSONResponse:
    """List available image generation models in OpenAI format.

    Separate from /v1/models (which lists LLMs only) to avoid image models
    being selected for chat completions.
    """
    data = []
    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if model_mgr:
        for m in model_mgr.list_local_models():
            data.append({
                "id": m["name"],
                "object": "model",
                "created": 0,
                "owned_by": "augmentum-image",
                "pipeline_type": m.get("pipeline_type", ""),
            })
    return JSONResponse({"object": "list", "data": data})


# --- OpenAI Images API ---


def _openai_image_error(
    message: str, status: int, error_type: str = "invalid_request_error",
) -> JSONResponse:
    """Return an error in OpenAI's expected format."""
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "param": None, "code": None}},
        status_code=status,
    )


def _map_openai_image_params(req: OpenAIImageRequest) -> dict:
    """Translate OpenAI image params to GenerationJob fields."""
    # Size — "auto" falls back to server defaults, otherwise parse WxH
    if req.size.lower() == "auto":
        from augmentum.config import settings as _settings
        width = _settings.image_default_width
        height = _settings.image_default_height
    else:
        w_str, h_str = req.size.lower().split("x")
        width, height = int(w_str), int(h_str)

    # Quality → steps and cfg (defaults; overridden by explicit fields below)
    if req.quality in ("hd", "high"):
        steps, cfg_scale = 30, 8.0
    elif req.quality == "medium":
        steps, cfg_scale = 25, 7.5
    elif req.quality == "low":
        steps, cfg_scale = 15, 6.0
    else:
        # "standard" or "auto"
        steps, cfg_scale = 20, 7.0

    # Explicit steps/cfg_scale from request override quality-derived defaults
    if req.steps is not None:
        steps = req.steps
    if req.cfg_scale is not None:
        cfg_scale = req.cfg_scale

    # Style → negative prompt (augmented by explicit negative_prompt if provided)
    negative_prompt = req.negative_prompt
    if req.style == "natural":
        style_neg = "oversaturated, overexposed, vivid colors, neon, harsh contrast"
        negative_prompt = f"{negative_prompt}, {style_neg}" if negative_prompt else style_neg

    # Model — map OpenAI placeholder names to empty (server default)
    model = req.model
    if model in ("dall-e-2", "dall-e-3", "gpt-image-1", "gpt-image-1.5"):
        model = ""

    return {
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
        "model": model,
        "seed": req.seed,
    }


@router.post("/images/generations")
async def openai_image_generate(body: OpenAIImageRequest, request: Request) -> JSONResponse:
    """Handle OpenAI POST /v1/images/generations requests.

    Supports local GPU generation and cloud provider routing via the
    ``provider`` field (e.g. "openai", "stability", "together").
    Set ``enhance_prompt: true`` to run LLM prompt enhancement before generation.
    """
    from augmentum.image.queue import GenerationJob

    params = _map_openai_image_params(body)
    prompt = body.prompt
    negative_prompt = params["negative_prompt"]

    # --- LLM prompt enhancement (optional) ---
    if body.enhance_prompt:
        try:
            from augmentum.image.prompt_condenser import enhance_prompt as _enhance
            registry = request.app.state.provider_registry
            _backend, _model = await registry.resolve_backend_with_fabric(
                settings.image_condense_model or ""
            )
            prompt = await _enhance(prompt, _backend, model=_model)
        except Exception:
            log.warning("openai_image_enhance_failed", exc_info=True)
            # Continue with original prompt

    # --- Cloud provider routing ---
    if body.provider:
        try:
            from augmentum.proxy.cloud_image_routes import generate_cloud_image
            result = await generate_cloud_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=params["model"],
                provider_id=body.provider,
                quality=body.quality,
                width=params["width"],
                height=params["height"],
                n=body.n,
                seed=params["seed"],
                app_state=request.app.state,
            )
            # Convert cloud result to OpenAI response format
            cloud_data = []
            images = result.get("images") or []
            for img in images:
                if body.response_format == "b64_json" and img.get("b64"):
                    cloud_data.append(OpenAIImageData(b64_json=img["b64"], revised_prompt=prompt))
                elif img.get("url"):
                    cloud_data.append(OpenAIImageData(url=img["url"], revised_prompt=prompt))
            if not cloud_data and result.get("url"):
                cloud_data.append(OpenAIImageData(url=result["url"], revised_prompt=prompt))
            return JSONResponse(
                OpenAIImageResponse(created=int(time.time()), data=cloud_data).model_dump(),
            )
        except Exception as exc:
            return _openai_image_error(f"Cloud generation failed: {exc}", 502, "server_error")

    # --- Local GPU generation ---
    queue = getattr(request.app.state, "image_queue", None)
    if not queue:
        return _openai_image_error("Image generation is not enabled", 503, "server_error")

    # Apply default preset if configured
    preset_name = settings.image_default_preset
    preset_manager = getattr(request.app.state, "image_preset_manager", None)
    if preset_name and preset_manager:
        preset = preset_manager.get(preset_name)
        if preset:
            prompt, negative_prompt = preset.apply(prompt, negative_prompt)
            if not params.get("steps"):
                params["steps"] = preset.steps
            if not params.get("cfg_scale"):
                params["cfg_scale"] = preset.cfg_scale

    model = params["model"] or settings.image_default_model
    _oi_user = request.scope.get("user")
    _oi_user_id = _oi_user.id if _oi_user else ""
    if not _oi_user_id:
        return _openai_image_error("Authentication required", 401, "invalid_request_error")
    jobs = []
    for _ in range(body.n):
        job = GenerationJob(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            width=params["width"],
            height=params["height"],
            steps=params["steps"],
            cfg_scale=params["cfg_scale"],
            seed=params["seed"],
            enhance_prompt=False,  # Already enhanced above if requested
            user_id=_oi_user_id,
        )
        try:
            job = await queue.submit(job)
        except RuntimeError as exc:
            # Covers both "queue full" and the per-user fairness cap — surface
            # the specific reason rather than a generic one.
            return _openai_image_error(
                str(exc) or "Image generation queue is full", 429, "rate_limit_error",
            )
        jobs.append(job)

    # Wait for all results
    data: list[OpenAIImageData] = []
    for job in jobs:
        try:
            result = await queue.wait_for_result(job, timeout=settings.image_generation_timeout)
        except TimeoutError:
            return _openai_image_error("Image generation timed out", 504, "server_error")
        except Exception as exc:
            return _openai_image_error(f"Generation failed: {exc}", 500, "server_error")

        image_id = result["image_id"]

        if body.response_format == "b64_json":
            import os
            file_path = result.get("file_path", "")
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                data.append(OpenAIImageData(b64_json=b64, revised_prompt=prompt))
            else:
                return _openai_image_error("Generated image file not found", 500, "server_error")
        else:
            base = str(request.base_url).rstrip("/")
            url = f"{base}/v1/images/{image_id}"
            data.append(OpenAIImageData(url=url, revised_prompt=prompt))

    return JSONResponse(
        OpenAIImageResponse(created=int(time.time()), data=data).model_dump(),
    )


@router.post("/images/edits")
async def openai_image_edit(request: Request):
    """Handle OpenAI POST /v1/images/edits (multipart form) for img2img/inpainting."""
    import os

    from augmentum.image.queue import GenerationJob
    from augmentum.image.schemas import JobType

    queue = getattr(request.app.state, "image_queue", None)
    if not queue:
        return _openai_image_error("Image generation is not enabled", 503, "server_error")

    form = await request.form()
    prompt = form.get("prompt", "")
    if not prompt:
        return _openai_image_error("prompt is required", 400)

    image_file = form.get("image")
    mask_file = form.get("mask")
    model = form.get("model", "")
    n = int(form.get("n", "1"))
    size = form.get("size", "auto")
    response_format = form.get("response_format", "url")

    # Read uploaded images as base64
    import base64 as b64mod
    source_b64 = ""
    if image_file and hasattr(image_file, "read"):
        source_b64 = b64mod.b64encode(await image_file.read()).decode("ascii")

    mask_b64 = ""
    if mask_file and hasattr(mask_file, "read"):
        mask_b64 = b64mod.b64encode(await mask_file.read()).decode("ascii")

    if not source_b64:
        return _openai_image_error("image file is required", 400)

    # Parse size
    width, height = 0, 0
    if size and size.lower() != "auto":
        try:
            w_str, h_str = size.lower().split("x")
            width, height = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            pass

    # Map OpenAI model names
    if model in ("dall-e-2", "dall-e-3", "gpt-image-1", "gpt-image-1.5"):
        model = ""
    model = model or settings.image_default_model

    job_type = JobType.INPAINT if mask_b64 else JobType.IMG2IMG
    strength = float(form.get("strength", "0.75" if not mask_b64 else "1.0"))

    _oe_user = request.scope.get("user")
    _oe_user_id = _oe_user.id if _oe_user else ""
    if not _oe_user_id:
        return _openai_image_error("Authentication required", 401, "invalid_request_error")

    jobs = []
    for _ in range(n):
        job = GenerationJob(
            job_type=job_type,
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            steps=settings.image_default_steps,
            cfg_scale=settings.image_default_cfg,
            source_image=source_b64,
            mask_image=mask_b64,
            strength=strength,
            user_id=_oe_user_id,
        )
        try:
            job = await queue.submit(job)
        except RuntimeError as exc:
            # Covers both "queue full" and the per-user fairness cap — surface
            # the specific reason rather than a generic one.
            return _openai_image_error(
                str(exc) or "Image generation queue is full", 429, "rate_limit_error",
            )
        jobs.append(job)

    data: list[OpenAIImageData] = []
    for job in jobs:
        try:
            result = await queue.wait_for_result(job, timeout=settings.image_generation_timeout)
        except TimeoutError:
            return _openai_image_error("Image editing timed out", 504, "server_error")
        except Exception as exc:
            return _openai_image_error(f"Editing failed: {exc}", 500, "server_error")

        image_id = result["image_id"]

        if response_format == "b64_json":
            file_path = result.get("file_path", "")
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("ascii")
                data.append(OpenAIImageData(b64_json=b64_data, revised_prompt=prompt))
            else:
                return _openai_image_error("Generated image file not found", 500, "server_error")
        else:
            base_url = str(request.base_url).rstrip("/")
            url = f"{base_url}/v1/images/{image_id}"
            data.append(OpenAIImageData(url=url, revised_prompt=prompt))

    return JSONResponse(
        OpenAIImageResponse(created=int(time.time()), data=data).model_dump(),
    )


@router.get("/images/{image_id}")
async def openai_image_serve(image_id: str, request: Request):
    """Serve a generated image by ID (for URL-mode responses)."""
    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        return _openai_image_error("Image persistence not available", 503, "server_error")

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        return _openai_image_error("Unauthorized", 401, "unauthorized")

    gen = await persistence.get_generation(image_id, user_id=uid)
    if not gen:
        return _openai_image_error("Image not found", 404, "not_found")

    import os
    file_path = gen["file_path"]
    if not os.path.exists(file_path):
        return _openai_image_error("Image file not found on disk", 404, "not_found")

    return FileResponse(file_path, media_type="image/png")
