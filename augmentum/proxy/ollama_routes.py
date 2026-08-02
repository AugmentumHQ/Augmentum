"""Ollama-compatible API routes."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from augmentum.classifier.router import MODE_HEADER, Mode, RequestClassifier
from augmentum.config import settings
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
from augmentum.models.provider_registry import ModelUnavailableError
from augmentum.modes.passthrough.handler import _CHAT_SYNTHESIS_HINT, PassthroughHandler
from augmentum.proxy.handler_factory import (
    _resolve_passthrough_tools,
    apply_prompt_cache_key,
    get_handler_for_mode,
    resolve_auto_invoke_tools,
    resolve_session_keys,
)
from augmentum.proxy.status_bus import bind_request_id, make_request_id
from augmentum.utils.logging import get_logger

# Module-level dedup state for ``ollama_tags``'s per-backend probe — see
# ``_probe`` in ``ollama_tags``. Lives at module scope so the dict persists
# across requests (per-process). Maps backend_key → last error class name
# (or "timeout") so a backend that stays in the same failure state doesn't
# re-warn on every /api/tags poll.
_PROBE_STATE_BACKEND_ERR: dict[str, str] = {}

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ollama"])


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


# --- Pydantic models for Ollama API ---


class OllamaMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] = ""
    images: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    # Reasoning models (DeepSeek-R1, Qwen3 thinking, etc.) emit a separate
    # reasoning_content channel during streaming. The UI persists it on
    # the assistant node and replays it on the next turn — DeepSeek 400s
    # without it. Field is optional so non-reasoning clients are unaffected.
    reasoning_content: str | None = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    format: str | None = None
    keep_alive: str | None = None
    tools: list[dict[str, Any]] | None = None
    think: bool | None = None
    # Per-request chat-template kwargs (e.g. {"enable_thinking": true} from
    # the coder composer's thinking toggle). Forwarded verbatim to
    # InternalChatRequest — local engines merge it over the automatic
    # think-mapping (llama_cpp._apply_reasoning_request_options), cloud
    # providers fold enable_thinking into their own toggles
    # (openai_compat._effective_think). Was silently dropped at this
    # boundary until 2026-07-02 — the exact explicit-field-list trap
    # documented in models/base.py:60.
    chat_template_kwargs: dict[str, Any] | None = None
    # Per-request Qwen 3.6 preserve_thinking override. Falls through to the
    # per-user UI setting when None. Other model families ignore the kwarg.
    preserve_thinking: bool | None = None
    # When True, the trailing message is a partial assistant turn the
    # model should continue verbatim (Continue button in the chat UI).
    # Each backend implements differently — DeepSeek's ``prefix: true``,
    # Anthropic's native trailing-assistant, llama-server's
    # ``add_generation_prompt: false`` chat-template kwarg, fallback
    # synthetic-user message for providers without prefix support.
    continue_last_assistant: bool = False
    # Narrative-mode UI lorebook (SillyTavern World Info format).
    # Forwarded to the narrative handler so the full backend LoreEngine
    # (recursion, secondary keys, depth scan, timed effects) runs against
    # the live UI entries — not just whatever was embedded in the card.
    lorebook: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model name must not be empty")
        return v.strip()

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[OllamaMessage]) -> list[OllamaMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        return v


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    options: dict[str, Any] | None = None
    format: str | None = None
    keep_alive: str | None = None
    images: list[str] | None = None
    raw: bool = False
    think: bool | None = None

    model_config = {"extra": "allow"}


class OllamaShowRequest(BaseModel):
    name: str
    verbose: bool = False

    model_config = {"extra": "allow"}


# --- Voice input compensation ---

_VOICE_CONTEXT_NOTE = (
    "[This message was transcribed from voice input via speech-to-text. "
    "It may contain transcription errors: homophones, mishearings, "
    "missing punctuation, or run-on sentences. Interpret the user's "
    "intent generously and do not correct their speech unless asked.]"
)


def _inject_voice_context(req: InternalChatRequest) -> None:
    """Annotate the last user message with a voice-input context note.

    This helps the LLM compensate for STT word error rate (WER) by
    interpreting the input charitably rather than treating transcription
    artifacts as intentional.
    """
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content:
            if isinstance(msg.content, str):
                msg.content = f"{msg.content}\n\n{_VOICE_CONTEXT_NOTE}"
            elif isinstance(msg.content, list):
                # Append to last text part in vision content array
                for part in reversed(msg.content):
                    if part.get("type") == "text":
                        part["text"] = f"{part['text']}\n\n{_VOICE_CONTEXT_NOTE}"
                        break
            break


# --- Conversion helpers ---


def _parse_ollama_content(content: str | list[dict], images: list[str] | None) -> tuple[str, list[str] | None]:
    """Parse message content that may be a string or OpenAI vision content array.

    Merges any images from the content array with the top-level images field.
    """
    if isinstance(content, str):
        return content, images

    # OpenAI vision format: [{type: "text", text: "..."}, {type: "image_url", ...}]
    text_parts: list[str] = []
    content_images: list[str] = list(images) if images else []
    for part in content:
        ptype = part.get("type", "")
        if ptype == "text":
            text_parts.append(part.get("text", ""))
        elif ptype == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url:
                content_images.append(url)
    return "\n".join(text_parts), content_images or None


def _resolve_request_think(explicit: bool | None, request: Request) -> bool:
    """Resolve a per-request thinking override from body/header/defaults."""
    if explicit is not None:
        return bool(explicit)
    header = request.headers.get("X-Augmentum-Think")
    if header is not None:
        return header.strip().lower() in {"1", "true", "yes", "on"}
    return settings.think_enabled


def to_internal_chat_request(req: OllamaChatRequest, *, think: bool | None = None) -> InternalChatRequest:
    """Convert Ollama chat request to internal format."""
    messages = []
    for m in req.messages:
        text, images = _parse_ollama_content(m.content, m.images)
        messages.append(Message(
            role=m.role,
            content=text,
            images=images,
            tool_calls=m.tool_calls,
            thinking=m.reasoning_content,
        ))

    opts = req.options or {}

    return InternalChatRequest(
        model=req.model,
        messages=messages,
        stream=req.stream,
        temperature=opts.get("temperature"),
        top_p=opts.get("top_p"),
        max_tokens=opts.get("num_predict"),
        stop=opts.get("stop"),
        frequency_penalty=opts.get("repeat_penalty"),
        seed=opts.get("seed"),
        tools=req.tools,
        format=req.format,
        keep_alive=req.keep_alive,
        raw_options=opts,
        think=settings.think_enabled if think is None else bool(think),
        chat_template_kwargs=req.chat_template_kwargs,
        preserve_thinking=req.preserve_thinking,
        voice_input=bool(getattr(req, "voice_input", False)),
        lorebook=req.lorebook,
        continue_last_assistant=bool(getattr(req, "continue_last_assistant", False)),
    )


def internal_response_to_ollama(resp, model: str) -> dict:
    """Convert internal response to Ollama chat response format."""
    result: dict = {
        "model": resp.model or model,
        "created_at": "",
        "message": {
            "role": resp.message.role,
            "content": resp.message.content,
        },
        "done": True,
        "done_reason": resp.finish_reason or "stop",
    }

    if resp.message.tool_calls:
        result["message"]["tool_calls"] = resp.message.tool_calls
    if resp.message.thinking:
        result["message"]["thinking"] = resp.message.thinking

    if resp.timing:
        result.update(resp.timing)

    if resp.usage:
        result["prompt_eval_count"] = resp.usage.prompt_tokens
        result["eval_count"] = resp.usage.completion_tokens

    return result


# --- Route handlers ---


@router.post("/chat")
async def ollama_chat(body: OllamaChatRequest, request: Request) -> JSONResponse:
    """Handle Ollama /api/chat requests."""
    # See openai_routes.py::openai_chat for the rationale on why this
    # binding is not reset in finally — streaming responses iterate
    # after the route function returns.
    bind_request_id(make_request_id())

    registry = request.app.state.provider_registry
    classifier: RequestClassifier = request.app.state.classifier

    internal_req = to_internal_chat_request(body, think=_resolve_request_think(body.think, request))

    # Voice input compensation — tag last user message so the LLM knows
    # to be tolerant of speech-to-text errors (homophones, missing
    # punctuation, run-on sentences, mishearings).
    if internal_req.voice_input:
        _inject_voice_context(internal_req)

    # Classify first (strips a/n/p/ mode prefix from model name).
    # When companion_dispatch_routes_chat is on AND the runtime is up,
    # the chat router consults the companion's dispatcher; otherwise
    # it's a transparent wrapper around the classifier — chat works
    # identically whether or not the runtime is enabled.
    mode_override = request.headers.get(MODE_HEADER)
    uid = _user_id(request)
    # External API clients (sk-aug key auth) default to clean DIRECT passthrough
    # — no classifier/memory harness — unless they opt into a mode via a model
    # prefix (a/ n/ p/ g/ c/ d/) or the mode header. Mirrors openai_chat; keeps
    # Augmentum's own /api/chat (browser session) behaviour unchanged.
    if not mode_override and request.scope.get("authed_via_api_key"):
        from augmentum.classifier.router import MODE_PREFIXES
        _m = body.model or ""
        if not any(_m.startswith(p) for p in MODE_PREFIXES):
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

    # Resolve session + user context up-front so we can pass them into
    # the fabric-aware backend resolver below for telemetry.
    workspace_id = request.headers.get("X-Augmentum-Workspace", "")
    coder_workspace = workspace_id if classification.mode == Mode.CODER else ""
    session_id, internal_req.kv_session_key = resolve_session_keys(
        request, internal_req, user_id=uid, workspace_id=coder_workspace,
    )
    internal_req.kv_mode = classification.mode.value
    # Sticky prompt-cache routing for OpenAI-family backends. The in-app
    # UI posts here, so without this the in-app chat surface paid full
    # price on every turn while external API clients did not.
    apply_prompt_cache_key(internal_req, user_id=uid, session_id=session_id)
    # Idle-reaper activity bump — see openai_routes.py for rationale.
    if coder_workspace:
        _mgr = getattr(request.app.state, "container_manager", None)
        if _mgr is not None:
            try:
                await _mgr.mark_active(coder_workspace)
            except Exception:
                # Idle-reaper may auto-stop the workspace if this bump is
                # consistently failing; debug-level so the chat path stays
                # quiet on the happy path.
                log.debug("coder_workspace_mark_active_failed",
                          workspace=coder_workspace, exc_info=True)

    # Smart routing — resolve backend with fabric awareness. On solo
    # installs (no fabric director attached) this is identical to
    # ``resolve_backend_for_model``; with fabric on, a peer-only model
    # returns a FabricBackend automatically.
    try:
        backend, clean_model = await registry.resolve_backend_with_fabric(
            internal_req.model, user_id=uid, session_id=session_id,
        )
    except ModelUnavailableError as exc:
        log.warning(
            "ollama_chat_model_unavailable",
            model=exc.model, diagnostic=exc.peer_diagnostic,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "model": exc.model,
                "fabric": exc.peer_diagnostic,
            },
        )
    internal_req.model = clean_model

    # Adopt this as the user's primary chat model so role resolution
    # ("Auto — use Primary") downstream uses it. Frontend-side push
    # paths are missed by some UI flows; the actual chat request is
    # the only universally-reliable signal of what the user is using.
    from augmentum.proxy.primary_model import adopt_primary_chat_model
    await adopt_primary_chat_model(request.app.state, clean_model)

    # Direct mode short-circuit — skip every injector before the request
    # is dispatched. See augmentum/modes/direct/handler.py for the
    # contract. Reached only when the caller explicitly opts in via
    # ``d/<model>`` prefix or ``X-Augmentum-Mode: direct``.
    if classification.mode == Mode.DIRECT:
        from augmentum.modes.direct.handler import DirectHandler
        from augmentum.proxy.streaming import stream_ollama_chat_handler

        log.info(
            "direct_dispatch",
            model=internal_req.model,
            user_id=uid, session_id=session_id,
            stream=bool(body.stream), reason=classification.reason,
        )
        direct_handler = DirectHandler(backend=backend)
        if body.stream:
            return await stream_ollama_chat_handler(internal_req, direct_handler)
        internal_resp = await direct_handler.handle(internal_req)
        return JSONResponse(internal_response_to_ollama(internal_resp, body.model))

    # "Currently listening to X" context from the frontend media player
    # — small 1-sentence system prefix so the LLM grounds naturally in
    # what the user is actively hearing. Companion-scoped (see the injector
    # docstring): no-ops for narrative/passthrough/etc. so it can't pollute
    # RP or break the KV prefix. No-ops when no audio is active.
    from augmentum.proxy.media_context import inject_media_context
    inject_media_context(internal_req, request, mode=classification.mode.value)

    # Resolve passthrough tools from header + config. Pass the latest user
    # message so the companion scheduling substrate is intent-gated (keeps
    # plain chat on the streaming fast-path).
    tools_header = request.headers.get("X-Augmentum-Tools")
    _latest_user_msg = next(
        (m.content for m in reversed(internal_req.messages) if m.role == "user"),
        "",
    )
    pt_tools = _resolve_passthrough_tools(
        request.app.state, tools_header, query=_latest_user_msg,
    )
    # Analytical filter is selector-only (no config-default merge) — see
    # ``resolve_analytical_tools`` for semantics.
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

    # User's explicitly selected flow (overrides auto-routing in the resolver)
    explicit_flow_id = request.headers.get("X-Augmentum-Flow", "")

    # Group chat id — lets the narrative handler activate GroupTurnManager
    # + per-turn speaker card swap without depending on persisted state
    # having been previously populated.
    group_id_header = request.headers.get("X-Augmentum-Group-Id", "").strip()
    if group_id_header:
        internal_req.group_id = group_id_header
    # One-shot manual speaker pin for group chats — wins over rotation/LLM.
    speaker_header = request.headers.get("X-Augmentum-Speaker", "").strip()
    if speaker_header:
        internal_req.speaker_override = speaker_header
        log.info("speaker_override_received", speaker=speaker_header,
                 session_id=session_id, group_id=group_id_header or "")

    mem_user_id = uid or "default"
    coder_strategy = request.headers.get("X-Augmentum-Coder-Strategy", "")
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
        handler._auto_invoke_tools = resolve_auto_invoke_tools(tools_header)

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

    # Vision caption fallback — see openai_routes.py for rationale.
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

    # Apply mode-aware inference defaults (temp, top_p, max_tokens)
    # Only sets values the user hasn't explicitly configured
    from augmentum.modes.inference_hints import apply_mode_hints
    apply_mode_hints(internal_req, mode_value)

    # Per-model sampling profiles (OpenWebUI-style layering). Fill whatever's
    # still unset for THIS model: per-call (body) ▸ mode hints (above) ▸
    # per-model (user→install seed) ▸ family default. ``apply_to_request``
    # only writes still-None fields, so explicit per-call values and the mode
    # hints both win — this makes "swap the model, the right sampling follows"
    # actually happen (Qwen3 → 0.6/0.95/20, Gemma-4 → 1.0/0.95/64) instead of
    # leaving the backend on its own defaults. Mirrors the same call in
    # openai_routes.py. Never raises. See models/sampling_profiles.py.
    from augmentum.models.sampling_profiles import apply_to_request as _apply_sampling
    await _apply_sampling(
        internal_req,
        getattr(request.app.state, "settings_store", None),
        user_id=uid,
    )

    # Fetch context window size for utilization display (cached by backend)
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
        "wire_format": "ollama",
        "stream": bool(body.stream),
    })

    try:
        if body.stream:
            from augmentum.proxy.streaming import stream_ollama_chat_handler_with_extraction

            return await stream_ollama_chat_handler_with_extraction(
                internal_req, handler, request.app.state, session_id, mode_value,
                context_length=ctx_len, user_id=mem_user_id,
            )

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
        # Notify dream scheduler of message activity (per-user counters)
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
            wire_format="ollama",
            stream=False,
            user_text=_user_text,
            assistant_text=internal_resp.message.content,
        )
        return JSONResponse(internal_response_to_ollama(internal_resp, body.model))
    except Exception:
        if not isinstance(handler, PassthroughHandler):
            log.warning("narrative_handle_failed_falling_back", exc_info=True)
            fallback = PassthroughHandler(
                backend=backend,
                custom_flow_store=getattr(request.app.state, "custom_flow_store", None),
            )
            if body.stream:
                from augmentum.proxy.streaming import stream_ollama_chat_handler

                return await stream_ollama_chat_handler(internal_req, fallback)
            internal_resp = await fallback.handle(internal_req)
            return JSONResponse(internal_response_to_ollama(internal_resp, body.model))
        raise


@router.post("/generate")
async def ollama_generate(body: OllamaGenerateRequest, request: Request) -> JSONResponse:
    """Handle Ollama /api/generate by converting to chat format."""
    registry = request.app.state.provider_registry
    classifier: RequestClassifier = request.app.state.classifier

    # Convert generate to chat format
    messages = []
    if body.system:
        messages.append(Message(role="system", content=body.system))
    messages.append(
        Message(role="user", content=body.prompt, images=body.images)
    )

    internal_req = InternalChatRequest(
        model=body.model,
        messages=messages,
        stream=body.stream,
        format=body.format,
        keep_alive=body.keep_alive,
        raw_options=body.options,
        think=_resolve_request_think(body.think, request),
    )

    # Classify first (strips a/n/p/ mode prefix from model name)
    mode_override = request.headers.get(MODE_HEADER)
    from augmentum.companion_runtime.chat_router import _read_default_mode
    default_mode = await _read_default_mode(
        request.app.state, _user_id(request),
    )
    classification = classifier.classify(
        internal_req, mode_override=mode_override, default_mode=default_mode,
    )
    log.info(
        "classified",
        mode=classification.mode.value,
        confidence=round(classification.confidence, 2),
        reason=classification.reason,
    )

    # Resolve context first so fabric-aware resolve has telemetry IDs.
    gen_uid = _user_id(request)
    workspace_id = request.headers.get("X-Augmentum-Workspace", "")
    coder_workspace = workspace_id if classification.mode == Mode.CODER else ""
    coder_strategy = request.headers.get("X-Augmentum-Coder-Strategy", "")
    session_id, internal_req.kv_session_key = resolve_session_keys(
        request, internal_req, user_id=gen_uid, workspace_id=coder_workspace,
    )
    internal_req.kv_mode = classification.mode.value
    # Sticky prompt-cache routing for OpenAI-family backends. The in-app
    # UI posts here, so without this the in-app chat surface paid full
    # price on every turn while external API clients did not.
    apply_prompt_cache_key(internal_req, user_id=gen_uid, session_id=session_id)
    # Idle-reaper activity bump — see openai_routes.py for rationale.
    if coder_workspace:
        _mgr = getattr(request.app.state, "container_manager", None)
        if _mgr is not None:
            try:
                await _mgr.mark_active(coder_workspace)
            except Exception:
                # Idle-reaper may auto-stop the workspace if this bump is
                # consistently failing; debug-level so the chat path stays
                # quiet on the happy path.
                log.debug("coder_workspace_mark_active_failed",
                          workspace=coder_workspace, exc_info=True)

    # Smart routing — resolve backend with fabric awareness (peer
    # FabricBackend when local doesn't have the model and a peer does).
    try:
        backend, clean_model = await registry.resolve_backend_with_fabric(
            internal_req.model, user_id=gen_uid, session_id=session_id,
        )
    except ModelUnavailableError as exc:
        log.warning(
            "ollama_generate_model_unavailable",
            model=exc.model, diagnostic=exc.peer_diagnostic,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": str(exc),
                "model": exc.model,
                "fabric": exc.peer_diagnostic,
            },
        )
    internal_req.model = clean_model
    from augmentum.proxy.primary_model import adopt_primary_chat_model
    await adopt_primary_chat_model(request.app.state, clean_model)

    handler = get_handler_for_mode(
        classification.mode, backend, session_id, request.app.state,
        workspace_id=workspace_id,
        user_id=gen_uid,
        coder_strategy=coder_strategy,
    )

    # Resolve stored chat image URLs to inline base64 for the LLM backend
    await resolve_chat_image_urls(internal_req, request.app.state)

    # Vision caption fallback — see openai_routes.py for rationale.
    await caption_via_router_fallback(internal_req, request.app.state, backend)

    # Expand [file:fi_xxx] tokens (from Files panel "Reference in Chat")
    from augmentum.proxy.file_token_resolver import resolve_file_tokens
    await resolve_file_tokens(
        internal_req,
        user_id=gen_uid,
        file_index=getattr(request.app.state, "file_index", None),
        vfs=getattr(request.app.state, "vfs", None),
    )

    # Ensure image attachments have adequate prompting for VL models
    inject_vision_prompt(internal_req)

    mode_value = classification.mode.value

    def _generate_response(resp):
        return JSONResponse({
            "model": resp.model or body.model,
            "created_at": "",
            "response": resp.message.content,
            "done": True,
            "done_reason": resp.finish_reason or "stop",
            "context": [],
        })

    try:
        if body.stream:
            from augmentum.proxy.streaming import stream_ollama_generate_handler_with_extraction

            return await stream_ollama_generate_handler_with_extraction(
                internal_req, handler, request.app.state, session_id, mode_value,
                user_id=gen_uid or "default",
            )

        from augmentum.training.trace_context import begin_capture, end_capture
        _cap_ctx, _cap_tok = begin_capture(user_id=gen_uid or "default", session_id=session_id, mode=mode_value)
        try:
            internal_resp = await handler.handle(internal_req)
        except Exception:
            end_capture(_cap_ctx, _cap_tok, error="handler_error")
            raise
        end_capture(_cap_ctx, _cap_tok)
        return _generate_response(internal_resp)
    except Exception:
        if not isinstance(handler, PassthroughHandler):
            log.warning("narrative_handle_failed_falling_back", exc_info=True)
            fallback = PassthroughHandler(
                backend=backend,
                custom_flow_store=getattr(request.app.state, "custom_flow_store", None),
            )
            if body.stream:
                from augmentum.proxy.streaming import stream_ollama_generate_handler

                return await stream_ollama_generate_handler(internal_req, fallback)
            internal_resp = await fallback.handle(internal_req)
            return _generate_response(internal_resp)
        raise


@router.get("/tags")
async def ollama_tags(request: Request) -> JSONResponse:
    """List available models from all backends, including prefixed variants."""
    registry = request.app.state.provider_registry
    await registry.refresh_model_map()
    # Per-user provider visibility (migration 305): private providers only
    # list for their owner. Derive the requester from the ASGI scope.
    _tags_user = request.scope.get("user")
    _tags_uid = getattr(_tags_user, "id", "") if _tags_user else ""
    # Default: list each model ONCE by its base name. The a/ n/ p/ mode
    # variants still work when typed; enumerating them turned ~400 models into
    # 2k+ near-duplicates in external pickers. Opt back in with ?include_modes=1.
    _include_modes = request.query_params.get("include_modes", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    # Collect models from all backends, tracking names for collision detection.
    # Probe backends IN PARALLEL with a per-backend deadline. The previous
    # serial loop meant one slow/unreachable provider (deepseek, openrouter,
    # etc. with a degraded route) blocked the entire response for the sum of
    # their httpx connect timeouts — 20+ seconds in pathological cases, which
    # froze the Settings modal + Model Manager because both prefill from
    # /api/tags on open. Bounded gather: each backend gets a per-backend
    # deadline (local 6s / cloud 15s, via registry.probe_deadline_for) before
    # we give up on it for this request; failures don't poison the response.
    # The cloud leash matters on a COLD restart — deepseek/openrouter's large
    # /models response routinely exceeds 6s, and dropping it here is what made
    # the user's models "not all there" until a second manual refresh.

    # Per-backend last-error coalescing — same pattern as
    # ``provider_registry._probe_failure_state``. /api/tags is hit on every
    # Settings modal open + Model Manager poll, so a single offline
    # backend was warning ~20×/min. Warn on transition (was healthy or
    # had a different failure mode), debug on continued same-state.
    probe_state = _PROBE_STATE_BACKEND_ERR

    async def _probe(key: str, backend) -> tuple[str, list[Any] | None]:
        deadline = registry.probe_deadline_for(key)
        try:
            models = await asyncio.wait_for(
                backend.list_models(), timeout=deadline,
            )
            if probe_state.pop(key, None):
                log.warning("tags_list_recovered", backend=key)
            return key, list(models)
        except TimeoutError:
            err_key = "timeout"
            if probe_state.get(key) != err_key:
                log.warning("tags_list_timeout", backend=key, timeout_s=deadline)
                probe_state[key] = err_key
            else:
                log.debug("tags_list_timeout", backend=key, timeout_s=deadline, repeat=True)
            return key, None
        except Exception as exc:
            err_repr = repr(exc)
            err_key = type(exc).__name__
            if probe_state.get(key) != err_key:
                log.warning("tags_list_failed", backend=key, error=err_repr)
                probe_state[key] = err_key
            else:
                log.debug("tags_list_failed", backend=key, error=err_repr, repeat=True)
            return key, None

    probe_results = await asyncio.gather(
        *(
            _probe(k, b)
            for k, b in registry.backends.items()
            if registry.is_listing_excluded(k) is not True
            and registry.provider_visible_to(k, _tags_uid)
        ),
        return_exceptions=False,
    )

    backend_models: list[tuple[str, Any]] = []  # (backend_key, ModelInfo)
    seen_names: dict[str, list[str]] = {}  # model_name -> [backend_keys]
    for key, models in probe_results:
        if not models:
            continue
        for m in models:
            backend_models.append((key, m))
            seen_names.setdefault(m.name, []).append(key)

    model_list = []
    for key, m in backend_models:
        # Disambiguate if same model name exists in multiple backends
        display_name = m.name
        display_model = m.model
        if len(seen_names.get(m.name, [])) > 1:
            display_name = f"{m.name}@{key}"
            display_model = f"{m.model}@{key}"

        details = {**(m.details or {}), "augmentum_backend": key}

        # ``vision`` carries the multimodal capability flag the backend
        # discovered (e.g., llama-cpp sets it True when a sibling mmproj
        # projector was paired by _find_paired_mmproj). The dropdown's
        # vision-badge gate reads ``supports_vision`` -- expose both
        # keys so legacy consumers also see the truth. ``mtp`` mirrors
        # has_mtp_heads from the GGUF profile so the model card can
        # badge MTP-capable builds (DeepSeek V3/V4, Qwen 3.6, Gemma 4
        # MTP).
        mtp_flag = bool(getattr(m, "mtp", False))
        # Original model
        model_list.append({
            "name": display_name,
            "model": display_model,
            "modified_at": m.modified_at,
            "size": m.size,
            "digest": m.digest,
            "details": details,
            "vision": bool(m.vision),
            "supports_vision": bool(m.vision),
            "mtp": mtp_flag,
        })
        # Prefixed variants for mode selection — opt-in only (see _include_modes).
        if _include_modes:
            for prefix, mode in [("a/", "analytical"), ("n/", "narrative"), ("p/", "passthrough")]:
                model_list.append({
                    "name": f"{prefix}{display_name}",
                    "model": f"{prefix}{display_model}",
                    "modified_at": m.modified_at,
                    "size": m.size,
                    "digest": m.digest,
                    "details": {**details, "augmentum_mode": mode},
                    "vision": bool(m.vision),
                    "supports_vision": bool(m.vision),
                    "mtp": mtp_flag,
                })

    # Phase 8 — merge peer-advertised LLM models. Each connected
    # fabric peer's heartbeat carries its capability list; we surface
    # those models in the dropdown so operators can pick them.
    #
    # Multi-peer-aware naming: every peer-copy of a model gets its own
    # dropdown entry with an ``@fabric:<node_id_short>`` suffix and a
    # peer icon/hostname badge (rendered by the existing UI logic in
    # ``settings.js::renderModelOption`` + ``chat/renderer.js``). LLMs
    # are large and genuinely distinct per machine — unlike Kokoro
    # voices, we don't fight a 50-voice-per-client scaling wall here,
    # so the dropdown can afford to expose the redundancy. Operator
    # can then pin to a specific peer (e.g. "always use the 4090 box")
    # instead of letting the scoring path pick.
    #
    # Pinned ``@fabric:<peer>`` entries are emitted for every peer-copy
    # of every model, INCLUDING models that also exist locally. The
    # original "local-first invariant" suppressed peer entries when a
    # same-named local copy existed, which silently hid working peer
    # copies whenever a local file was present-but-broken (e.g. a
    # Gated Delta Net GGUF that loads but CUDA-OOMs on inference).
    # Bare ``model_id`` (un-pinned) entries still route to local first
    # via the dispatch director, so default behaviour is unchanged —
    # the picker just stops hiding the explicitly-routed alternatives.
    #
    # When 2+ connected peers serve the same model, an additional
    # un-pinned entry (bare ``model_id``) is emitted at the top so
    # operators who don't care which box can still let the director
    # auto-score. Suppressed when only one peer serves it (the
    # pinned entry is the only viable choice in that case — no need
    # for a duplicate auto-route line) OR when a local backend already
    # exposes that bare name (local emission already covers it; emitting
    # again would dup the dropdown row).
    try:
        coordinator = getattr(request.app.state, "fabric_coordinator", None)
        if coordinator is not None:
            # First pass: collect every (node_id, peer_meta, cap) tuple
            # that survives the local-first filter.
            peer_entries: list[tuple[str, dict, Any]] = []
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
                    peer_entries.append((node_id, {
                        "icon": peer_icon,
                        "hostname": peer_hostname,
                    }, cap))

            # Count how many peers serve each model_id so we know when
            # to also emit an un-pinned auto-route entry.
            model_peer_counts: dict[str, int] = {}
            for _, _, cap in peer_entries:
                mid = getattr(cap, "model_id", "")
                model_peer_counts[mid] = model_peer_counts.get(mid, 0) + 1

            # Emit auto-route entries first (one per model_id with 2+
            # serving peers). These carry no pin suffix; the resolver
            # falls through to the director's scoring path.
            for model_id, count in model_peer_counts.items():
                if count < 2 or model_id in seen_names:
                    # <2: pinned entry is the only viable choice.
                    # in seen_names: local backend already emitted this
                    # bare name; a second emission would duplicate.
                    continue
                auto_details = {
                    "augmentum_backend": "fabric",
                    "augmentum_source": "peer-auto",
                    "augmentum_peer_count": count,
                }
                model_list.append({
                    "name": model_id,
                    "model": model_id,
                    "modified_at": "",
                    "size": 0,
                    "digest": "",
                    "details": auto_details,
                    "vision": False,
                    "supports_vision": False,
                })
                if _include_modes:
                    for prefix, mode in [
                        ("a/", "analytical"), ("n/", "narrative"), ("p/", "passthrough"),
                    ]:
                        model_list.append({
                            "name": f"{prefix}{model_id}",
                            "model": f"{prefix}{model_id}",
                            "modified_at": "",
                            "size": 0,
                            "digest": "",
                            "details": {**auto_details, "augmentum_mode": mode},
                            "vision": False,
                            "supports_vision": False,
                        })

            # Emit one pinned entry per peer-copy. Suffix is
            # ``@fabric:<node_id[:12]>`` — short enough to fit a
            # dropdown line, long enough to disambiguate across a
            # household-scale fabric. Resolver prefix-matches against
            # connected peers, so this id doesn't have to be unique
            # globally, only among connected peers at dispatch time.
            for node_id, peer_meta, cap in peer_entries:
                model_id = getattr(cap, "model_id", "")
                pinned_name = f"{model_id}@fabric:{node_id[:12]}"
                peer_details = {
                    "augmentum_backend": "fabric",
                    "augmentum_source": "peer",
                    "augmentum_peer_node_id": node_id,
                    "augmentum_peer_hostname": peer_meta["hostname"],
                    "augmentum_peer_icon": peer_meta["icon"],
                    "family": getattr(cap, "model_family", "") or "",
                    "parameter_size": (
                        f"{getattr(cap, 'params_b', 0)}B"
                        if getattr(cap, "params_b", 0) else ""
                    ),
                }
                model_list.append({
                    "name": pinned_name,
                    "model": pinned_name,
                    "modified_at": "",
                    "size": 0,
                    "digest": "",
                    "details": peer_details,
                    "vision": False,
                    "supports_vision": False,
                })
                if _include_modes:
                    for prefix, mode in [
                        ("a/", "analytical"), ("n/", "narrative"), ("p/", "passthrough"),
                    ]:
                        model_list.append({
                            "name": f"{prefix}{pinned_name}",
                            "model": f"{prefix}{pinned_name}",
                            "modified_at": "",
                            "size": 0,
                            "digest": "",
                            "details": {**peer_details, "augmentum_mode": mode},
                            "vision": False,
                            "supports_vision": False,
                        })
    except Exception as exc:
        # Never let a fabric error break the model list — local
        # models must still be enumerable when fabric coordinator
        # is half-initialised or otherwise unhappy.
        log.warning("fabric_peer_model_merge_failed", error=repr(exc))

    # Append load balancers as selectable "models"
    try:
        store = getattr(request.app.state, "balancer_store", None)
        if store:
            balancers = await store.list_balancers()
            for b in balancers:
                if not b.enabled:
                    continue
                members = await store.list_members(b.id)
                member_names = [m.model_name for m in members if m.enabled]
                model_list.append({
                    "name": f"lb/{b.name}",
                    "model": f"lb/{b.name}",
                    "modified_at": "",
                    "size": 0,
                    "digest": "",
                    "details": {
                        "augmentum_type": "load_balancer",
                        "augmentum_backend": "balancer",
                        "strategy": b.strategy,
                        "members": member_names,
                        "family": "load_balancer",
                    },
                })
    except Exception:
        log.debug("balancer_list_in_tags_failed", exc_info=True)

    return JSONResponse({"models": model_list})


@router.post("/show")
async def ollama_show(body: OllamaShowRequest, request: Request) -> JSONResponse:
    """Show model information."""
    registry = request.app.state.provider_registry
    backend = registry.get_backend("ollama")
    if not backend:
        return JSONResponse({"error": "Ollama backend not configured"}, status_code=404)
    details = await backend.show_model(body.name)

    return JSONResponse(
        {
            "modelfile": details.modelfile,
            "parameters": details.parameters,
            "template": details.template,
            "details": details.details or {},
            "model_info": details.model_info or {},
        }
    )


@router.head("/")
@router.get("/version")
async def ollama_version(request: Request) -> JSONResponse:
    """Return version info (used by Open WebUI).

    Also exposes ``persistence_degraded`` as a top-level boolean so the UI
    can render a banner when the server has fallen back to the in-memory
    backend (rare — the bootstrap fail-fast guard refuses to start in
    that mode unless ``AUGMENTUM_ALLOW_INMEMORY_FALLBACK=1`` is set).
    Kept on this UNAUTHENTICATED endpoint because that's the only signal
    the UI has when auth itself is in the degraded path (every other
    endpoint would 503 with `auth_unavailable_denied`).
    """
    degraded = bool(getattr(request.app.state, "persistence_degraded", False))
    body: dict[str, object] = {"version": "0.1.0"}
    if degraded:
        body["persistence_degraded"] = True
        body["persistence_degraded_reason"] = (
            "Server fell back to in-memory backend. Data is NOT persisted "
            "and will be lost on restart. Check server logs for the root "
            "cause and restart cleanly."
        )
    return JSONResponse(body)


@router.post("/pull")
async def ollama_pull(request: Request):
    """Proxy model pull to upstream Ollama with NDJSON streaming."""
    body = await request.json()
    model_name = body.get("name", body.get("model", "unknown"))
    stream = body.get("stream", True)

    log.info("model_pull_started", model=model_name)

    if not stream:
        # Non-streaming: wait for full response
        client = request.app.state.http_client
        resp = await client.post(
            f"{settings.ollama_base_url}/api/pull",
            json=body,
            timeout=1800.0,  # 30 min for large downloads
        )
        # Invalidate model map so new model appears immediately
        request.app.state.provider_registry.invalidate_model_map()
        return JSONResponse(resp.json(), status_code=resp.status_code)

    # Streaming: relay NDJSON progress updates
    from fastapi.responses import StreamingResponse

    async def _stream_pull():
        try:
            client = request.app.state.http_client
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/pull",
                json=body,
                timeout=1800.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line + "\n"
            log.info("model_pull_completed", model=model_name)
            # Invalidate model map so new model appears immediately
            request.app.state.provider_registry.invalidate_model_map()
        except Exception as exc:
            log.error("pull_stream_error", model=model_name, error=str(exc))
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(
        _stream_pull(),
        media_type="application/x-ndjson",
    )


@router.get("/ps")
async def ollama_ps(request: Request) -> JSONResponse:
    """Show running models (used by Open WebUI)."""
    models = []

    # Ollama running models
    if settings.ollama_base_url:
        client = request.app.state.http_client
        try:
            resp = await client.get(f"{settings.ollama_base_url}/api/ps")
            data = resp.json()
            models.extend(data.get("models", []))
        except Exception:
            log.debug("ps_ollama_failed", exc_info=True)

    # Engine v2 running model
    manager = getattr(request.app.state, "llama_manager", None)
    if manager and manager.model_id:
        from augmentum.models.llama_server_manager import ProcessState
        if manager.state == ProcessState.READY:
            gpu = manager._query_gpu_info()
            engine_model = {
                "name": manager.model_id,
                "model": manager.model_id,
                "size": manager._last_profile.total_size_bytes if manager._last_profile else 0,
                "size_vram": gpu.get("used_bytes", 0) if gpu else 0,
                "details": {"family": "gguf", "format": "gguf"},
                "backend": "engine",
            }
            if manager._last_profile:
                engine_model["details"]["parameter_size"] = f"{manager._last_profile.size_gb}GB"
                engine_model["details"]["quantization_level"] = manager._last_profile.architecture
            models.append(engine_model)

    return JSONResponse({"models": models})


@router.delete("/delete")
async def ollama_delete(request: Request) -> JSONResponse:
    """Proxy model delete to upstream Ollama."""
    body = await request.json()
    client = request.app.state.http_client
    try:
        resp = await client.request(
            "DELETE",
            f"{settings.ollama_base_url}/api/delete",
            json=body,
        )
        return JSONResponse(resp.json() if resp.content else {}, status_code=resp.status_code)
    except Exception:
        log.warning("delete_request_failed", exc_info=True)
        return JSONResponse({"error": "Failed to delete model"}, status_code=502)


@router.post("/copy")
async def ollama_copy(request: Request) -> JSONResponse:
    """Proxy model copy to upstream Ollama."""
    body = await request.json()
    client = request.app.state.http_client
    try:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/copy",
            json=body,
        )
        return JSONResponse(
            resp.json() if resp.content else {"status": "success"},
            status_code=resp.status_code,
        )
    except Exception:
        log.warning("copy_request_failed", exc_info=True)
        return JSONResponse({"error": "Failed to copy model"}, status_code=502)


@router.post("/create")
async def ollama_create(request: Request):
    """Proxy model creation to upstream Ollama with NDJSON streaming."""
    from fastapi.responses import StreamingResponse

    body = await request.json()
    stream = body.get("stream", True)

    if not stream:
        client = request.app.state.http_client
        resp = await client.post(
            f"{settings.ollama_base_url}/api/create",
            json=body,
            timeout=600.0,
        )
        return JSONResponse(resp.json(), status_code=resp.status_code)

    async def _stream_create():
        try:
            client = request.app.state.http_client
            async with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/create",
                json=body,
                timeout=600.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line + "\n"
        except Exception as exc:
            log.error("create_stream_error", error=str(exc))
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(
        _stream_create(),
        media_type="application/x-ndjson",
    )


@router.post("/embeddings")
@router.post("/embed")
async def ollama_embeddings(request: Request) -> JSONResponse:
    """Proxy embedding requests to upstream Ollama."""
    body = await request.json()
    client = request.app.state.http_client
    # Try both endpoints (Ollama uses /api/embed, some clients use /api/embeddings)
    try:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json=body,
        )
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception:
        log.warning("embeddings_request_failed", exc_info=True)
        return JSONResponse({"error": "Embeddings request failed"}, status_code=502)
