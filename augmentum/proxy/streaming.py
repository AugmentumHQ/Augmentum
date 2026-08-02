"""NDJSON/SSE streaming utilities and format translation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
from fastapi.responses import StreamingResponse

from augmentum.models.base import InternalChatRequest, InternalStreamChunk, ModelBackend
from augmentum.modes.base import ModeHandler
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from starlette.datastructures import State

log = get_logger(__name__)

# Headers that prevent proxy/CDN buffering — critical for real-time streaming.
# X-Accel-Buffering disables nginx response buffering per-response.
# no-store prevents Cloudflare and CDN caching.
_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-store",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class StreamPrimeError(Exception):
    """The upstream side of a streaming response failed before any
    chunks could be produced. Carries the original cause so the route
    layer can map it to a clean HTTP status (typically 502).

    The whole point of priming is to surface connection-class failures
    BEFORE Starlette's ``StreamingResponse`` commits ``http.response
    .start`` — without this, the failure inside the generator's first
    ``await`` produces a "Caught handled exception, but response
    already started." RuntimeError and a generic 500 to the client.
    """

    def __init__(self, cause: BaseException) -> None:
        super().__init__(repr(cause))
        self.cause = cause


async def prime_stream(
    gen: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Pull the first chunk of ``gen`` so upstream-connect failures
    raise BEFORE the caller hands the iterator to ``StreamingResponse``.

    Pattern:

        try:
            primed = await prime_stream(_build_upstream_generator(...))
        except StreamPrimeError as exc:
            raise HTTPException(502, f"upstream unreachable: {exc.cause!r}")
        return StreamingResponse(primed, media_type=...)

    Why this works: Starlette's ``stream_response`` sends
    ``http.response.start`` at the top of the body iterator's first
    iteration — by the time the generator's ``async with
    client.stream(...)`` context entered and threw ConnectError,
    headers were already committed. ``await gen.__anext__()`` here
    forces that connection to be made on OUR side of the boundary; if
    it fails, no ``http.response.start`` has been sent yet, so the
    caller can return a clean error response.

    The returned iterator yields the buffered first chunk, then
    forwards from the original generator. Cancellation, cleanup, and
    further-mid-stream errors propagate as before — only the FIRST
    chunk is special.

    On empty streams (generator yields no values and returns cleanly),
    ``StopAsyncIteration`` is wrapped in ``StreamPrimeError`` so the
    caller can decide how to respond — an empty upstream is almost
    always a misconfiguration worth reporting.
    """
    try:
        first = await gen.__anext__()
    except StopAsyncIteration as exc:
        raise StreamPrimeError(exc) from exc
    except BaseException as exc:
        # Best-effort drain so an httpx stream context inside the
        # generator gets a chance to close its connection cleanly
        # before we re-raise. ``aclose()`` is a no-op for generators
        # that don't expose it.
        with contextlib.suppress(Exception):
            await gen.aclose()  # type: ignore[attr-defined]
        if isinstance(exc, Exception):
            raise StreamPrimeError(exc) from exc
        raise  # CancelledError / KeyboardInterrupt — preserve type

    async def _replay() -> AsyncIterator[bytes]:
        yield first
        async for chunk in gen:
            yield chunk

    return _replay()


def _last_user_text(request: InternalChatRequest) -> str:
    """Return the most recent user-role message content from a request.

    Used by the salience synapse (companion runtime) to score the
    just-completed chat turn. Returns empty string when no user-role
    message is present — the salience pipeline gracefully no-ops.
    Tool/system/assistant roles are skipped; only the user's own
    voice counts as the turn she just heard.
    """
    for msg in reversed(request.messages or []):
        if getattr(msg, "role", "") == "user":
            return (getattr(msg, "content", "") or "")
    return ""


# Ceiling for a believable visible single-stream decode rate. Real local
# and cloud backends top out in the low hundreds of tok/s for the visible
# stream; anything past this is a measurement artifact from single-burst
# delivery (see _StreamTimer.stats), so the rate is suppressed rather than
# reported. Deliberately generous — it only needs to catch the impossible.
_MAX_PLAUSIBLE_TPS = 2000.0


class _StreamTimer:
    """Track generation timing and context utilization for UI/perf reporting.

    Accuracy contract:

    * **TTFT** marks ``_first_token`` on the first signal back from the
      model — content OR reasoning. A reasoning model that thinks for
      20s before producing visible content used to show TTFT=20s (which
      reads as "the model was unresponsive"); counting reasoning
      separately reads correctly as "model started replying at 200ms,
      reasoning ran for 19.8s, content followed."

    * **Token counts** prefer upstream Usage when present
      (``prompt_tokens``, ``eval_tokens`` set explicitly). When upstream
      omits usage — OpenAI without ``stream_options.include_usage``,
      Deepseek (which silently drops the flag), any provider that
      doesn't echo back token counts — ``stats()`` falls back to local
      tokenization via ``augmentum.utils.tokenizer.count_tokens`` and
      flags the result via ``prompt_tokens_estimated`` /
      ``eval_tokens_estimated`` so the UI can prefix ``~``. tiktoken's
      cl100k_base is ~15-20% off for non-GPT tokenizers but always
      better than 0 or a chunk-count under-estimate.
    """

    __slots__ = (
        "_start", "_first_token", "eval_tokens", "context_length",
        "prompt_tokens", "_content_chars", "_prompt_estimated",
        "_eval_estimated", "_eval_set_by_usage", "_eval_duration_ms_authoritative",
    )

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._first_token: float | None = None
        self.eval_tokens = 0
        self.context_length = 0   # Model's max context window (set externally)
        self.prompt_tokens = 0    # Prompt token count (set from usage)
        # Accumulated streamed content for end-of-stream tokenization
        # fallback. We store char count + content separately so we can
        # tokenize once at the end rather than per-chunk (cheap CPU).
        self._content_chars: list[str] = []
        self._prompt_estimated = False
        self._eval_estimated = False
        # True once authoritative completion_tokens arrived from upstream
        # usage; turns off the tokenization fallback in stats().
        self._eval_set_by_usage = False
        # Authoritative server-side decode wall-time (ms) from upstream
        # usage (llama-server timings.predicted_ms). 0.0 = not reported;
        # when > 0, stats() computes tok/s from this instead of the
        # proxy's first-to-last-token wall clock, which is the only
        # reliable rate when delivery is single-burst.
        self._eval_duration_ms_authoritative = 0.0

    def on_token(self, n: int = 1, text: str = "") -> None:
        """Record a token signal. ``text`` is the actual chunk text so
        the timer can tokenize it locally if upstream usage never lands.
        """
        if self._first_token is None:
            self._first_token = time.monotonic()
        self.eval_tokens += n
        if text:
            self._content_chars.append(text)

    def on_first_signal(self) -> None:
        """Mark TTFT without bumping the token count. Use when the
        signal is non-content (reasoning_delta) — we want the response
        latency clock to start but not pollute the token count, which
        is meant to reflect visible content only.
        """
        if self._first_token is None:
            self._first_token = time.monotonic()

    def observe(self, chunk: InternalStreamChunk) -> None:
        """Consume one stream chunk: update TTFT, eval_tokens, and
        (on the final chunk) merge authoritative usage from upstream.

        Single source of truth so the 4 streaming wrappers stay in
        sync — earlier they each had their own copy of "if
        content_delta: on_token; if done and usage: assign" which
        drifted easily.
        """
        if chunk.content_delta:
            self.on_token(text=chunk.content_delta)
        elif chunk.thinking_delta:
            # Reasoning models stream thinking_delta before any
            # visible content_delta. Mark TTFT but don't bump eval
            # tokens — the visible token count remains content-only.
            self.on_first_signal()
        if chunk.done and chunk.usage:
            # Upstream usage is authoritative — overrides any local
            # estimate / chunk-count, clears the estimated flags.
            if chunk.usage.completion_tokens:
                self.eval_tokens = chunk.usage.completion_tokens
                self._eval_set_by_usage = True
                self._eval_estimated = False
            if chunk.usage.prompt_tokens:
                self.prompt_tokens = chunk.usage.prompt_tokens
                self._prompt_estimated = False
            # Authoritative decode time, when the backend reports it.
            # Preferred over wall-clock for the tok/s computation below.
            if getattr(chunk.usage, "eval_duration_ms", 0.0):
                self._eval_duration_ms_authoritative = float(chunk.usage.eval_duration_ms)

    def set_prompt_text(self, text: str) -> None:
        """Pre-set prompt token count via local tokenization. Used by
        the streaming wrappers before the model returns so when the
        upstream's usage is missing we can still report something
        accurate. Overridden by an authoritative count if one arrives.
        """
        if not text or self.prompt_tokens:
            return
        from augmentum.utils.tokenizer import count_tokens
        self.prompt_tokens = count_tokens(text)
        self._prompt_estimated = True

    def stats(self) -> dict:
        """Return timing + utilization stats for injection into the final chunk."""
        now = time.monotonic()
        total_s = now - self._start
        ttft_s = (self._first_token - self._start) if self._first_token else 0.0
        eval_s = (now - self._first_token) if self._first_token else 0.0
        # If upstream never sent usage AND we observed content, tokenize
        # the accumulated content as a fallback. eval_tokens at this
        # point is the chunk count (1-per-chunk while streaming), which
        # systematically undercounts for backends that batch tokens per
        # chunk. Local tokenization is closer to truth.
        if self._content_chars and not getattr(self, "_eval_set_by_usage", False):
            from augmentum.utils.tokenizer import count_tokens
            estimated = count_tokens("".join(self._content_chars))
            # Only overwrite if the estimate is larger — a chunk count
            # might be RIGHT for backends that stream one token per
            # chunk (llama-server, Anthropic), and we never want to
            # discard authoritative data.
            if estimated > self.eval_tokens:
                self.eval_tokens = estimated
                self._eval_estimated = True
        # Prefer the backend's authoritative decode time when it reported
        # one (llama-server timings.predicted_ms, plumbed through
        # Usage.eval_duration_ms). The proxy's wall-clock ``eval_s`` is
        # only meaningful when tokens arrive incrementally; a single-burst
        # delivery collapses it to microseconds. The authoritative value
        # is network-free server-side time, so it gives the true decode
        # rate regardless of delivery shape.
        authoritative_eval_s = self._eval_duration_ms_authoritative / 1000.0
        rate_eval_s = authoritative_eval_s if authoritative_eval_s > 0 else eval_s
        tok_s = self.eval_tokens / rate_eval_s if rate_eval_s > 0 else 0.0
        # Guard against single-burst delivery WITHOUT authoritative timing.
        # When a backend returns the whole completion in effectively one
        # chunk — a non-streaming response, or a CPU llama-server that
        # emits all tokens at once after a long prefill — and reports no
        # decode time, ``_first_token`` is stamped microseconds before
        # stats() runs, so ``eval_s`` collapses toward zero and
        # ``eval_tokens / eval_s`` explodes (a real report: 131 tok /
        # ~6µs = 20,091,110 tok/s). No visible single-stream decode rate
        # is real above ~_MAX_PLAUSIBLE_TPS, so treat anything past it as
        # "unmeasurable from the proxy side" and report 0 — the UI hides a
        # zero tok/s rather than showing a garbage number. (The token
        # count itself stays accurate; only the rate is suppressed.) This
        # only triggers when the authoritative path didn't supply a rate.
        if tok_s > _MAX_PLAUSIBLE_TPS:
            tok_s = 0.0
        # Report the authoritative decode duration when we have it; it's
        # the number the rate was computed from and the one the UI should
        # show as "generated in Xms".
        reported_eval_s = authoritative_eval_s if authoritative_eval_s > 0 else eval_s
        result = {
            "total_duration": int(total_s * 1e9),
            "total_duration_ms": int(total_s * 1000),
            "eval_count": self.eval_tokens,
            "eval_duration": int(reported_eval_s * 1e9),
            "eval_duration_ms": int(reported_eval_s * 1000),
            "ttft_ms": int(ttft_s * 1000),
            "prompt_tokens": self.prompt_tokens,
            "tokens_per_second": round(tok_s, 1),
            "prompt_tokens_estimated": self._prompt_estimated,
            "eval_tokens_estimated": self._eval_estimated,
        }
        if self.context_length > 0:
            used = self.prompt_tokens + self.eval_tokens
            result["context_length"] = self.context_length
            result["context_used"] = used
        return result


def _prompt_text_for_timer(request: InternalChatRequest) -> str:
    """Concatenate message content for the timer's local-tokenization
    fallback. Best-effort — used only when upstream skips usage; an
    authoritative count from the upstream's own tokenizer always wins.

    Image attachments and tool-call structured content aren't counted
    (we'd need vision-tokenizer support for the former); that's fine
    for the "≈ how big was this turn" UI signal.
    """
    parts: list[str] = []
    for m in request.messages:
        content = getattr(m, "content", "") or ""
        if content:
            parts.append(content)
    return "\n".join(parts)


def _log_chat_request_perf(
    timer: _StreamTimer,
    request: InternalChatRequest,
    endpoint: str,
) -> None:
    """Emit a ``chat_request_perf`` log event with proxy-observed timings.

    Distinct from ``engine_perf`` (llama_server_manager / llama_cpp), which
    carries SERVER-SIDE timings from llama-server's ``timings`` field.
    This event captures PROXY-SIDE wall-clock numbers for every backend
    — OpenAI, Anthropic, Gemini, Ollama, fabric peers, and yes also
    llama_cpp. That gives dashboards uniform per-provider perf data
    even where we can't see into the upstream.

    For llama_cpp specifically, BOTH events fire per request — the
    server-side ``engine_perf`` is authoritative for prefill/gen ms
    (network-free); the proxy-side ``chat_request_perf`` adds wall-clock
    TTFT (network + queue + our own overhead). The delta between them
    is the proxy/network/queue cost.

    Token-count accuracy note: ``timer.eval_tokens`` is chunk-counted
    while streaming (1 chunk ≈ 1 token; some backends batch). When the
    upstream emits a usage chunk before stream end, the count is
    corrected to ``usage.completion_tokens`` (authoritative). When it
    doesn't (e.g. an OpenAI-compatible provider that ignores
    ``stream_options.include_usage``), the chunk count is approximate.
    """
    stats = timer.stats()
    # Don't log no-op streams (client disconnected before any token, or
    # an upstream error fired before the first chunk). They'd pollute
    # the perf dashboard with zero-token rows.
    if stats["eval_count"] == 0 and stats["prompt_tokens"] == 0:
        return
    from augmentum.proxy.status_bus import request_id_var
    log.info(
        "chat_request_perf",
        endpoint=endpoint,
        model=request.model,
        prompt_tokens=stats["prompt_tokens"],
        gen_tokens=stats["eval_count"],
        ttft_ms=stats["ttft_ms"],
        gen_duration_ms=stats["eval_duration_ms"],
        total_duration_ms=stats["total_duration_ms"],
        gen_tps=stats["tokens_per_second"],
        prompt_tokens_estimated=stats["prompt_tokens_estimated"],
        eval_tokens_estimated=stats["eval_tokens_estimated"],
        request_id=request_id_var.get() or "",
    )


# llama-server (and most OpenAI-compatible backends) phrase a
# prompt-too-long overflow a handful of stable ways. All arrive as an
# HTTP 400 whose body is surfaced verbatim in ``str(exc)`` (see
# OpenAICompatBackend: ``f"Backend returned {status}: {body}"``). Matched
# case-insensitively so a backend that capitalizes differently still routes
# to the context-overflow affordance instead of the generic "backend error".
_CONTEXT_OVERFLOW_PHRASES = (
    "exceeds the available context size",
    "exceeds the context",
    "exceed context",
    "input is too large to process",
    "n_tokens exceeds",
    "exceeds n_ctx",
    "context shift is disabled",
    "requested tokens exceed context window",
    "maximum context length",  # OpenAI-style: "maximum context length is N tokens"
    "reduce the length of the messages",
)


def _is_context_overflow(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in _CONTEXT_OVERFLOW_PHRASES)


def _friendly_backend_error(exc: Exception) -> str:
    """Convert a backend exception to a user-friendly error message."""
    if isinstance(exc, httpx.ReadTimeout):
        return "\n\n[Error: The model backend timed out. Try a shorter prompt or increase the timeout setting.]"
    msg = str(exc)
    if _is_context_overflow(msg):
        return (
            "\n\n[Error: This request is longer than the model's loaded "
            "context window. Load the model with a larger context size "
            "(Model Manager > Load Setup > Context), shorten the prompt, "
            "or trim the conversation history.]"
        )
    # Pre-flight validation: image attached to a text-only model. The
    # message phrase is stable; see LlamaCppBackend._reject_if_images_without_vision.
    if "no vision projector" in msg:
        return (
            "\n\n[Error: This model can't read images. Pair a vision "
            "projector (mmproj) via Model Manager > Pair vision, or "
            "switch to a vision-capable model.]"
        )
    if "429" in msg:
        return "\n\n[Error: The model provider is rate-limiting requests. Please wait a moment and try again.]"
    if "503" in msg or "502" in msg:
        return "\n\n[Error: The model backend is temporarily unavailable. Please try again shortly.]"
    if "401" in msg or "403" in msg:
        return "\n\n[Error: Authentication failed with the model provider. Check your API key.]"
    return "\n\n[Error: The model backend returned an error. Please try again.]"


def _classify_backend_error(exc: Exception) -> str:
    """Classify an exception into a coarse error_kind for the frontend.

    Used so the UI can render different affordances per failure mode
    (rate-limit suggests "wait then retry", auth suggests "check key",
    etc.) without parsing the friendly-text. Stable wire identifiers.
    """
    if isinstance(exc, httpx.ReadTimeout):
        return "timeout"
    msg = str(exc)
    if _is_context_overflow(msg):
        return "context_overflow"
    if "no vision projector" in msg:
        return "no_vision_projector"
    if "429" in msg:
        return "rate_limit"
    if "503" in msg or "502" in msg:
        return "backend_unavailable"
    if "401" in msg or "403" in msg:
        return "auth_failed"
    return "backend_error"


# error_kind values where retrying the same request would just fail again.
# The user has to change something (pair a projector, switch model, fix
# credentials) before another attempt can succeed.
_NON_RETRYABLE_KINDS = frozenset({"no_vision_projector", "auth_failed", "context_overflow"})


def _make_error_chunk(exc: Exception) -> InternalStreamChunk:
    """Build the canonical mid-stream error chunk.

    Combines the legacy inline `[Error: ...]` text (which non-Augmentum
    Ollama clients still rely on to know SOMETHING went wrong) with a
    structured `augmentum.backend_error` field. Augmentum's frontend
    detects the structured field and treats the chunk as a distinct
    error event (toast + interrupted marker + retry affordance) rather
    than rendering the inline text into the response bubble as if the
    model wrote it.
    """
    friendly = _friendly_backend_error(exc)
    kind = _classify_backend_error(exc)
    return InternalStreamChunk(
        content_delta=friendly,
        done=True,
        finish_reason="error",
        augmentum={
            # Strip the bracketed framing so the UI can present the
            # message in its own chrome.
            "backend_error": friendly.strip().lstrip("[").rstrip("]"),
            "retryable": kind not in _NON_RETRYABLE_KINDS,
            "error_kind": kind,
        },
    )


# --- Content accumulation for post-stream extraction ---


def _notify_dream_activity(app_state: State, user_id: str) -> None:
    """Tell the dream scheduler the user just sent a message.

    Streaming chat is the only path real users take, so the scheduler's
    per-user message counter and idle timer have to be incremented here
    or dream cycles never become eligible. Fires unconditionally in the
    finally block — the user's send is the activity signal regardless
    of whether the response completed, errored, or was cancelled.
    """
    scheduler = getattr(app_state, "dream_scheduler", None)
    if scheduler is None:
        return
    try:
        scheduler.notify_message(user_id=user_id)
        scheduler.notify_request(user_id=user_id)
    except Exception as exc:  # noqa: BLE001 — scheduler must never break the stream finalizer
        log.warning("dream_notify_failed", error=str(exc), user_id=user_id)


_HEARTBEAT_INTERVAL_S = 3.0


async def _with_heartbeat(
    stream: AsyncIterator[InternalStreamChunk],
    interval: float = _HEARTBEAT_INTERVAL_S,
) -> AsyncIterator[InternalStreamChunk]:
    """Inject a heartbeat chunk if the upstream is silent for ``interval`` seconds.

    Used so the UI can distinguish "model is thinking quietly" from
    "connection stalled" and keep proxies (nginx/Cloudflare) from idling
    out the response. Heartbeats carry the last known phase + a server
    timestamp; the frontend treats heartbeat-only gaps over ~15s as a
    stall warning.

    Opens with a ``stage_start: chat_dispatch`` so the frontend's
    content-watchdog (4s/15s timers) suspends from byte 0. Before this
    landed, the watchdog would fire its "Stream stalled. Abort &
    retry." banner during normal request prep — including a 30-60s
    model load — because the inner handler can take many seconds
    before yielding its first ``stage_start: model_load``. The
    dispatch stage stays open until the wrapped stream produces its
    first chunk (or terminates), at which point we emit the matching
    stage_complete; any nested stages (model_load, prefill) layer on
    top via the watchdog's set-of-active-stages tracking.
    """
    iterator = stream.__aiter__()
    pending = asyncio.ensure_future(iterator.__anext__())
    last_phase = ""
    last_status = ""
    dispatch_stage_id = f"stg_dispatch_{uuid.uuid4().hex[:10]}"
    dispatch_started_mono = time.monotonic()
    dispatch_closed = False

    def _dispatch_complete_chunk() -> InternalStreamChunk:
        return InternalStreamChunk(
            augmentum={
                "stage_complete": {
                    "id": dispatch_stage_id,
                    "stage": "chat_dispatch",
                    "success": True,
                    "duration_ms": int(
                        (time.monotonic() - dispatch_started_mono) * 1000,
                    ),
                    "detail": "",
                    "error": "",
                    "request_id": "",
                },
            },
        )

    try:
        yield InternalStreamChunk(
            augmentum={
                "stage_start": {
                    "id": dispatch_stage_id,
                    "stage": "chat_dispatch",
                    "label": "Preparing",
                    "detail": "",
                    "started_at": time.time(),
                    "request_id": "",
                },
                # Keep the legacy heartbeat fields too so any consumer
                # that was watching for them still has a wire signal —
                # the stage_start IS the watchdog suppressor, but the
                # heartbeat keys remain for backwards compatibility.
                "heartbeat": True,
                "phase": "starting",
                "phase_status": "starting",
                "t": time.time(),
            },
        )
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if pending in done:
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    if not dispatch_closed:
                        dispatch_closed = True
                        yield _dispatch_complete_chunk()
                    return
                if chunk.augmentum:
                    last_phase = chunk.augmentum.get("phase", last_phase) or last_phase
                    last_status = chunk.augmentum.get(
                        "phase_status", last_status,
                    ) or last_status
                # Close the dispatch stage on the first inner chunk so
                # the frontend's active-stages set transitions cleanly
                # into whatever stage the inner stream yielded (most
                # commonly stage_start: model_load).
                if not dispatch_closed:
                    dispatch_closed = True
                    yield _dispatch_complete_chunk()
                yield chunk
                if chunk.done:
                    return
                pending = asyncio.ensure_future(iterator.__anext__())
            else:
                yield InternalStreamChunk(
                    augmentum={
                        "heartbeat": True,
                        "phase": last_phase,
                        "phase_status": last_status,
                        "t": time.time(),
                    },
                )
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        # Explicitly close the inner generator instead of relying on GC.
        # If the inner stream was suspended at a yield while holding a lock
        # (e.g. narrative's processing_lock around a done=True meta chunk),
        # waiting for GC means the lock can be held long enough to stall
        # the next request on that engine.
        with contextlib.suppress(BaseException):
            await iterator.aclose()


class _ContentAccumulator:
    """Accumulates streamed content for post-stream extraction."""

    def __init__(self) -> None:
        self.content: str = ""

    async def wrap(
        self, stream: AsyncIterator[InternalStreamChunk],
    ) -> AsyncIterator[InternalStreamChunk]:
        """Wrap a stream, accumulating content deltas."""
        async for chunk in stream:
            if chunk.content_delta:
                self.content += chunk.content_delta
            yield chunk


# --- NDJSON helpers (Ollama format) ---


async def _ndjson_chat_generator(
    request: InternalChatRequest, backend: ModelBackend
) -> AsyncIterator[bytes]:
    """Generate NDJSON lines for Ollama /api/chat streaming."""
    try:
        async for chunk in backend.chat_stream(request):
            ndjson = _chunk_to_ollama_chat_ndjson(chunk, request.model)
            yield json.dumps(ndjson).encode() + b"\n"
    except asyncio.CancelledError:
        log.debug("client_disconnected", endpoint="/api/chat")
        raise
    except httpx.ReadTimeout as exc:
        log.error("backend_timeout_during_stream", endpoint="/api/chat")
        error_chunk = _chunk_to_ollama_chat_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        log.error("backend_error_during_stream", endpoint="/api/chat", error=str(exc))
        error_chunk = _chunk_to_ollama_chat_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"


async def _ndjson_generate_generator(
    request: InternalChatRequest, backend: ModelBackend
) -> AsyncIterator[bytes]:
    """Generate NDJSON lines for Ollama /api/generate streaming."""
    try:
        async for chunk in backend.chat_stream(request):
            ndjson = _chunk_to_ollama_generate_ndjson(chunk, request.model)
            yield json.dumps(ndjson).encode() + b"\n"
    except asyncio.CancelledError:
        log.debug("client_disconnected", endpoint="/api/generate")
        raise
    except httpx.ReadTimeout as exc:
        log.error("backend_timeout_during_stream", endpoint="/api/generate")
        error_chunk = _chunk_to_ollama_generate_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        log.error("backend_error_during_stream", endpoint="/api/generate", error=str(exc))
        error_chunk = _chunk_to_ollama_generate_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"


def _chunk_to_ollama_chat_ndjson(
    chunk: InternalStreamChunk, model: str, timing: _StreamTimer | None = None,
) -> dict:
    """Convert internal stream chunk to Ollama /api/chat NDJSON format."""
    result: dict = {
        "model": chunk.model or model,
        "created_at": "",
        "message": {
            "role": chunk.role or "assistant",
            "content": chunk.content_delta,
        },
        "done": chunk.done,
    }
    if chunk.done:
        result["done_reason"] = chunk.finish_reason or "stop"
        if chunk.usage:
            result["prompt_eval_count"] = chunk.usage.prompt_tokens
            result["eval_count"] = chunk.usage.completion_tokens
        # Inject generation timing (tok/s) into the final chunk
        if timing:
            stats = timing.stats()
            result["total_duration"] = stats["total_duration"]
            result["eval_duration"] = stats["eval_duration"]
            if not result.get("prompt_eval_count") and stats["prompt_tokens"] > 0:
                result["prompt_eval_count"] = stats["prompt_tokens"]
            if not result.get("eval_count"):
                result["eval_count"] = stats["eval_count"]
    # Include Augmentum metadata (UARF phase info, mode, model thinking)
    aug = dict(chunk.augmentum) if chunk.augmentum else {}
    if chunk.thinking_delta:
        aug["model_thinking_delta"] = chunk.thinking_delta
    # Always include performance stats in augmentum metadata for the UI.
    if chunk.done and timing:
        stats = timing.stats()
        aug["tokens_per_second"] = stats["tokens_per_second"]
        aug["ttft_ms"] = stats["ttft_ms"]
        aug["total_duration_ms"] = stats["total_duration_ms"]
        aug["eval_duration_ms"] = stats["eval_duration_ms"]
        # When upstream sent authoritative usage, the timer absorbed it
        # in observe(); stats reflect the true values and the
        # _estimated flags are False. When upstream didn't, stats carry
        # tiktoken-based fallback counts with flags True so the UI can
        # mark them ~approximate.
        aug["prompt_tokens"] = stats["prompt_tokens"]
        aug["eval_tokens"] = stats["eval_count"]
        aug["prompt_tokens_estimated"] = stats["prompt_tokens_estimated"]
        aug["eval_tokens_estimated"] = stats["eval_tokens_estimated"]
        if stats.get("context_length"):
            aug["context_length"] = stats["context_length"]
            aug["context_used"] = stats["context_used"]
    if chunk.done and chunk.usage:
        # Cache + reasoning telemetry parity with the OpenAI-shape
        # serializer — see _chunk_to_openai_sse for the rationale.
        if chunk.usage.cache_hit_tokens:
            aug["prompt_tokens_cached"] = chunk.usage.cache_hit_tokens
        if chunk.usage.cache_miss_tokens:
            aug["prompt_tokens_evaluated"] = chunk.usage.cache_miss_tokens
        if chunk.usage.cache_write_tokens:
            aug["prompt_tokens_cache_write"] = chunk.usage.cache_write_tokens
        if chunk.usage.reasoning_tokens:
            aug["reasoning_tokens"] = chunk.usage.reasoning_tokens
    if aug:
        result["augmentum"] = aug
    return result


def _chunk_to_ollama_generate_ndjson(chunk: InternalStreamChunk, model: str) -> dict:
    """Convert internal stream chunk to Ollama /api/generate NDJSON format."""
    result: dict = {
        "model": chunk.model or model,
        "created_at": "",
        "response": chunk.content_delta,
        "done": chunk.done,
    }
    if chunk.done:
        result["done_reason"] = chunk.finish_reason or "stop"
        result["context"] = []
        if chunk.usage:
            result["prompt_eval_count"] = chunk.usage.prompt_tokens
            result["eval_count"] = chunk.usage.completion_tokens
    return result


async def stream_ollama_chat(
    request: InternalChatRequest, backend: ModelBackend
) -> StreamingResponse:
    """Return a StreamingResponse for Ollama /api/chat."""
    return StreamingResponse(
        _ndjson_chat_generator(request, backend),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


async def stream_ollama_generate(
    request: InternalChatRequest, backend: ModelBackend
) -> StreamingResponse:
    """Return a StreamingResponse for Ollama /api/generate."""
    return StreamingResponse(
        _ndjson_generate_generator(request, backend),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


# --- SSE helpers (OpenAI format) ---


async def _sse_generator(
    request: InternalChatRequest, backend: ModelBackend
) -> AsyncIterator[bytes]:
    """Generate SSE events for OpenAI /v1/chat/completions streaming."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    chunk_count = 0
    try:
        async for chunk in backend.chat_stream(request):
            sse_data = _chunk_to_openai_sse(chunk, chunk_id)
            yield f"data: {json.dumps(sse_data)}\n\n".encode()
            chunk_count += 1

            if chunk.done:
                yield b"data: [DONE]\n\n"
                return

        # If the backend doesn't emit a final done chunk, send [DONE] anyway
        yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        log.debug("client_disconnected", endpoint="/v1/chat/completions")
        raise
    except httpx.ReadTimeout as exc:
        log.error("backend_timeout_during_stream", endpoint="/v1/chat/completions")
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(exc),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        log.error("backend_error_during_stream", endpoint="/v1/chat/completions", error=str(exc))
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(exc),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except Exception as exc:
        # Catch-all: anything not in the typed list above was silently
        # propagating up through StreamingResponse, surfacing as
        # ``ASGI callable returned without completing response`` from
        # uvicorn with no Python traceback or exception details visible
        # in the log. That made the 2026-05-23 fabric chat regression
        # nearly undebuggable. Log the type + message + traceback and
        # emit a final error chunk so the client sees *something*
        # instead of a half-stream.
        log.error(
            "backend_unexpected_error_during_stream",
            endpoint="/v1/chat/completions",
            exc_type=type(exc).__name__,
            error=str(exc)[:500],
            exc_info=True,
        )
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(exc),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except BaseException as exc:
        # Last resort: BaseException subclasses (SystemExit, KeyboardInterrupt,
        # GeneratorExit, anything else not caught above). These are
        # normally fatal/structural, but logging the type before re-raise
        # is the only way to distinguish them from CancelledError when
        # debugging silent stream exits. Re-raise to preserve semantics.
        log.warning(
            "sse_generator_base_exception",
            endpoint="/v1/chat/completions",
            exc_type=type(exc).__name__,
            chunks_before=chunk_count,
        )
        raise


def _chunk_to_openai_sse(
    chunk: InternalStreamChunk, chunk_id: str, timing: _StreamTimer | None = None,
) -> dict:
    """Convert internal stream chunk to OpenAI SSE format."""
    delta: dict = {}
    if chunk.role:
        delta["role"] = chunk.role
    if chunk.content_delta:
        delta["content"] = chunk.content_delta

    # Tool-call deltas arrive out-of-band on chunk.augmentum["tool_calls"]
    # (the web UI reads that sidecar). Standard OpenAI clients — the pi /
    # OpenCode / Cursor / Aider coding harnesses and any other /v1 consumer —
    # read choices[].delta.tool_calls, so promote them into the spec location
    # too. Mirrors fabric_routes.py. Additive: the sidecar copy is kept for the
    # UI, and these deltas are already in OpenAI streaming shape (index / id /
    # type / function.{name,arguments}). WITHOUT this, harnesses receive
    # finish_reason="tool_calls" with an EMPTY delta and silently do nothing —
    # every agentic tool turn is dropped while plain-text chat still works.
    if chunk.augmentum and chunk.augmentum.get("tool_calls"):
        delta["tool_calls"] = chunk.augmentum["tool_calls"]

    # Reasoning streams out-of-band on chunk.thinking_delta → the sidecar
    # ``model_thinking_delta`` (below), which only the web UI reads. Standard
    # OpenAI clients read ``delta.reasoning_content`` (DeepSeek convention; pi
    # parses it into thinking blocks), so mirror it there too. Additive: the
    # sidecar copy is kept for the UI. Without this, harnesses never see the
    # model's reasoning even with reasoning enabled.
    if chunk.thinking_delta:
        delta["reasoning_content"] = chunk.thinking_delta

    # OpenAI spec: the final chunk MUST carry a non-null finish_reason.
    # Lenient SDKs (Claude Code's, OpenAI's official) tolerate null and
    # treat the trailing [DONE] sentinel as end-of-stream. Stricter
    # parsers (the `pi` coding agent, some LiteLLM ingest paths) reject
    # the stream with "Stream ended without finish_reason" because
    # neither the chunk NOR a separate marker tells them WHY it ended.
    # Default to "stop" — the same value OpenAI returns for normal
    # completion — so every codepath that emits a done=True chunk
    # produces a spec-compliant final event regardless of whether the
    # handler/backend remembered to set the field.
    finish_reason = chunk.finish_reason
    if chunk.done and not finish_reason:
        finish_reason = "stop"

    result: dict = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": chunk.model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    # Include usage in the final chunk (OpenAI stream_options.include_usage pattern)
    if chunk.done:
        usage = {}
        if chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }
            # Conditional emission for cache + reasoning so providers
            # that don't report (most local + non-DeepSeek cloud) don't
            # grow synthetic zero fields. Cost calculators downstream
            # can distinguish "not reported" from "0 hits".
            if chunk.usage.cache_hit_tokens:
                usage["prompt_cache_hit_tokens"] = chunk.usage.cache_hit_tokens
            if chunk.usage.cache_miss_tokens:
                usage["prompt_cache_miss_tokens"] = chunk.usage.cache_miss_tokens
            if chunk.usage.cache_write_tokens:
                usage["prompt_cache_write_tokens"] = chunk.usage.cache_write_tokens
            if chunk.usage.reasoning_tokens:
                usage["completion_tokens_details"] = {
                    "reasoning_tokens": chunk.usage.reasoning_tokens,
                }
        if timing:
            stats = timing.stats()
            if not usage.get("prompt_tokens") and stats["prompt_tokens"] > 0:
                usage["prompt_tokens"] = stats["prompt_tokens"]
            if not usage.get("completion_tokens"):
                usage["completion_tokens"] = stats["eval_count"]
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + stats["eval_count"]
        if usage:
            result["usage"] = usage
    # Include Augmentum metadata (UARF phase info, mode, model thinking)
    aug = dict(chunk.augmentum) if chunk.augmentum else {}
    if chunk.thinking_delta:
        aug["model_thinking_delta"] = chunk.thinking_delta
    if chunk.done and timing:
        stats = timing.stats()
        aug["tokens_per_second"] = stats["tokens_per_second"]
        aug["ttft_ms"] = stats["ttft_ms"]
        aug["total_duration_ms"] = stats["total_duration_ms"]
        aug["eval_duration_ms"] = stats["eval_duration_ms"]
        aug["prompt_tokens"] = chunk.usage.prompt_tokens if chunk.usage else stats["prompt_tokens"]
        aug["eval_tokens"] = chunk.usage.completion_tokens if chunk.usage else stats["eval_count"]
        if stats.get("context_length"):
            aug["context_length"] = stats["context_length"]
            aug["context_used"] = stats["context_used"]
    if chunk.done and chunk.usage:
        # Mirror cache + reasoning telemetry into the augmentum block so
        # the UI can render "X cached / Y fresh" without re-parsing the
        # OpenAI ``usage`` shape. Aligns with the existing
        # ``prompt_tokens_cached`` channel that llama_cpp populated via
        # the legacy ctx_payload path; the new fields are typed at the
        # Usage layer so every provider participates.
        if chunk.usage.cache_hit_tokens:
            aug["prompt_tokens_cached"] = chunk.usage.cache_hit_tokens
        if chunk.usage.cache_miss_tokens:
            aug["prompt_tokens_evaluated"] = chunk.usage.cache_miss_tokens
        if chunk.usage.cache_write_tokens:
            aug["prompt_tokens_cache_write"] = chunk.usage.cache_write_tokens
        if chunk.usage.reasoning_tokens:
            aug["reasoning_tokens"] = chunk.usage.reasoning_tokens
    if aug:
        result["augmentum"] = aug
    return result


async def stream_openai_chat(
    request: InternalChatRequest, backend: ModelBackend
) -> StreamingResponse:
    """Return a StreamingResponse for OpenAI /v1/chat/completions."""
    return StreamingResponse(
        _sse_generator(request, backend),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


# --- Handler-aware streaming (works with any ModeHandler) ---


async def _handler_ndjson_chat_generator(
    request: InternalChatRequest, handler: ModeHandler,
    context_length: int = 0,
) -> AsyncIterator[bytes]:
    """Generate NDJSON lines for Ollama /api/chat via a ModeHandler.

    Token counting: each content_delta chunk counts as ~1 token (approximation).
    Backends that batch multiple tokens per chunk will undercount until the final
    chunk.usage overrides with the authoritative total.
    """
    timer = _StreamTimer()
    timer.context_length = context_length
    timer.set_prompt_text(_prompt_text_for_timer(request))
    try:
        async for chunk in _with_heartbeat(handler.handle_stream(request)):
            timer.observe(chunk)
            ndjson = _chunk_to_ollama_chat_ndjson(chunk, request.model, timing=timer if chunk.done else None)
            yield json.dumps(ndjson).encode() + b"\n"
    except asyncio.CancelledError:
        log.debug("client_disconnected", endpoint="/api/chat")
        raise
    except httpx.ReadTimeout:
        log.error("backend_timeout_during_stream", endpoint="/api/chat")
        error_chunk = _chunk_to_ollama_chat_ndjson(
            _make_error_chunk(httpx.ReadTimeout("")),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        log.error("backend_error_during_stream", endpoint="/api/chat", error=str(exc))
        error_chunk = _chunk_to_ollama_chat_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    finally:
        _log_chat_request_perf(timer, request, "/api/chat")


async def _handler_ndjson_generate_generator(
    request: InternalChatRequest, handler: ModeHandler
) -> AsyncIterator[bytes]:
    """Generate NDJSON lines for Ollama /api/generate via a ModeHandler."""
    try:
        async for chunk in _with_heartbeat(handler.handle_stream(request)):
            ndjson = _chunk_to_ollama_generate_ndjson(chunk, request.model)
            yield json.dumps(ndjson).encode() + b"\n"
    except asyncio.CancelledError:
        log.debug("client_disconnected", endpoint="/api/generate")
        raise
    except httpx.ReadTimeout:
        log.error("backend_timeout_during_stream", endpoint="/api/generate")
        error_chunk = _chunk_to_ollama_generate_ndjson(
            _make_error_chunk(httpx.ReadTimeout("")),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        log.error("backend_error_during_stream", endpoint="/api/generate", error=str(exc))
        error_chunk = _chunk_to_ollama_generate_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"


async def _handler_sse_generator(
    request: InternalChatRequest, handler: ModeHandler,
    context_length: int = 0,
) -> AsyncIterator[bytes]:
    """Generate SSE events for OpenAI /v1/chat/completions via a ModeHandler."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    timer = _StreamTimer()
    timer.context_length = context_length
    timer.set_prompt_text(_prompt_text_for_timer(request))
    try:
        async for chunk in _with_heartbeat(handler.handle_stream(request)):
            timer.observe(chunk)
            sse_data = _chunk_to_openai_sse(chunk, chunk_id, timing=timer if chunk.done else None)
            yield f"data: {json.dumps(sse_data)}\n\n".encode()

            if chunk.done:
                yield b"data: [DONE]\n\n"
                return

        yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        log.debug("client_disconnected", endpoint="/v1/chat/completions")
        raise
    except httpx.ReadTimeout:
        log.error("backend_timeout_during_stream", endpoint="/v1/chat/completions")
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(httpx.ReadTimeout("")),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        log.error("backend_error_during_stream", endpoint="/v1/chat/completions", error=str(exc))
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(exc),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        _log_chat_request_perf(timer, request, "/v1/chat/completions")


async def stream_ollama_chat_handler(
    request: InternalChatRequest, handler: ModeHandler
) -> StreamingResponse:
    """Return a StreamingResponse for Ollama /api/chat via a ModeHandler."""
    return StreamingResponse(
        _handler_ndjson_chat_generator(request, handler),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


async def stream_ollama_generate_handler(
    request: InternalChatRequest, handler: ModeHandler
) -> StreamingResponse:
    """Return a StreamingResponse for Ollama /api/generate via a ModeHandler."""
    return StreamingResponse(
        _handler_ndjson_generate_generator(request, handler),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


async def stream_openai_chat_handler(
    request: InternalChatRequest, handler: ModeHandler
) -> StreamingResponse:
    """Return a StreamingResponse for OpenAI /v1/chat/completions via a ModeHandler."""
    return StreamingResponse(
        _handler_sse_generator(request, handler),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )


# --- With-extraction variants (fix streaming extraction gap) ---


async def _handler_ndjson_chat_with_extraction(
    request: InternalChatRequest,
    handler: ModeHandler,
    app_state: State,
    session_id: str,
    mode: str,
    context_length: int = 0,
    user_id: str = "default",
) -> AsyncIterator[bytes]:
    """NDJSON chat generator that accumulates content and schedules extraction."""
    from augmentum.training.trace_context import begin_capture, end_capture

    accumulator = _ContentAccumulator()
    timer = _StreamTimer()
    timer.context_length = context_length
    timer.set_prompt_text(_prompt_text_for_timer(request))
    _cap_ctx, _cap_tok = begin_capture(user_id=user_id, session_id=session_id, mode=mode)
    _cap_err = ""
    try:
        async for chunk in accumulator.wrap(handler.handle_stream(request)):
            timer.observe(chunk)
            if chunk.done and request.pack_injection is not None:
                chunk.augmentum = {
                    **(chunk.augmentum or {}),
                    "knowledge_pack": request.pack_injection,
                }
            ndjson = _chunk_to_ollama_chat_ndjson(chunk, request.model, timing=timer if chunk.done else None)
            yield json.dumps(ndjson).encode() + b"\n"
    except asyncio.CancelledError:
        _cap_err = "CancelledError"
        log.debug("client_disconnected", endpoint="/api/chat")
        raise
    except httpx.ReadTimeout:
        _cap_err = "ReadTimeout"
        log.error("backend_timeout_during_stream", endpoint="/api/chat")
        error_chunk = _chunk_to_ollama_chat_ndjson(
            _make_error_chunk(httpx.ReadTimeout("")),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        _cap_err = type(exc).__name__
        log.error("backend_error_during_stream", endpoint="/api/chat", error=str(exc))
        error_chunk = _chunk_to_ollama_chat_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    finally:
        end_capture(_cap_ctx, _cap_tok, error=_cap_err)
        _log_chat_request_perf(timer, request, "/api/chat")
        _notify_dream_activity(app_state, user_id)
        if len(accumulator.content) >= 50:
            from augmentum.memory.integration import schedule_extraction

            schedule_extraction(
                app_state, request, accumulator.content, session_id, user_id=user_id, mode=mode,
            )
        from augmentum.training.capture import capture_training_trace
        capture_training_trace(request, accumulator.content, session_id, user_id, mode)
        from augmentum.companion_runtime.bus import emit_chat_turn_completed
        await emit_chat_turn_completed(
            app_state,
            mode=mode,
            user_id=user_id,
            session_id=session_id,
            wire_format="ollama",
            stream=True,
            user_text=_last_user_text(request),
            assistant_text=accumulator.content,
        )


async def _handler_ndjson_generate_with_extraction(
    request: InternalChatRequest,
    handler: ModeHandler,
    app_state: State,
    session_id: str,
    mode: str,
    user_id: str = "default",
) -> AsyncIterator[bytes]:
    """NDJSON generate generator that accumulates content and schedules extraction."""
    from augmentum.training.trace_context import begin_capture, end_capture

    accumulator = _ContentAccumulator()
    _cap_ctx, _cap_tok = begin_capture(user_id=user_id, session_id=session_id, mode=mode)
    _cap_err = ""
    try:
        async for chunk in accumulator.wrap(handler.handle_stream(request)):
            ndjson = _chunk_to_ollama_generate_ndjson(chunk, request.model)
            yield json.dumps(ndjson).encode() + b"\n"
    except asyncio.CancelledError:
        _cap_err = "CancelledError"
        log.debug("client_disconnected", endpoint="/api/generate")
        raise
    except httpx.ReadTimeout:
        _cap_err = "ReadTimeout"
        log.error("backend_timeout_during_stream", endpoint="/api/generate")
        error_chunk = _chunk_to_ollama_generate_ndjson(
            _make_error_chunk(httpx.ReadTimeout("")),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        _cap_err = type(exc).__name__
        log.error("backend_error_during_stream", endpoint="/api/generate", error=str(exc))
        error_chunk = _chunk_to_ollama_generate_ndjson(
            _make_error_chunk(exc),
            request.model,
        )
        yield json.dumps(error_chunk).encode() + b"\n"
    finally:
        end_capture(_cap_ctx, _cap_tok, error=_cap_err)
        _notify_dream_activity(app_state, user_id)
        if len(accumulator.content) >= 50:
            from augmentum.memory.integration import schedule_extraction

            schedule_extraction(
                app_state, request, accumulator.content, session_id, user_id=user_id, mode=mode,
            )
        from augmentum.training.capture import capture_training_trace
        capture_training_trace(request, accumulator.content, session_id, user_id, mode)


async def _handler_sse_with_extraction(
    request: InternalChatRequest,
    handler: ModeHandler,
    app_state: State,
    session_id: str,
    mode: str,
    context_length: int = 0,
    user_id: str = "default",
) -> AsyncIterator[bytes]:
    """SSE generator that accumulates content and schedules extraction."""
    from augmentum.training.trace_context import begin_capture, end_capture

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    accumulator = _ContentAccumulator()
    timer = _StreamTimer()
    timer.context_length = context_length
    timer.set_prompt_text(_prompt_text_for_timer(request))
    _cap_ctx, _cap_tok = begin_capture(user_id=user_id, session_id=session_id, mode=mode)
    _cap_err = ""
    try:
        async for chunk in accumulator.wrap(handler.handle_stream(request)):
            timer.observe(chunk)
            if chunk.done and request.pack_injection is not None:
                chunk.augmentum = {
                    **(chunk.augmentum or {}),
                    "knowledge_pack": request.pack_injection,
                }
            sse_data = _chunk_to_openai_sse(chunk, chunk_id, timing=timer if chunk.done else None)
            yield f"data: {json.dumps(sse_data)}\n\n".encode()

            if chunk.done:
                yield b"data: [DONE]\n\n"
                return

        yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        _cap_err = "CancelledError"
        log.debug("client_disconnected", endpoint="/v1/chat/completions")
        raise
    except httpx.ReadTimeout:
        _cap_err = "ReadTimeout"
        log.error("backend_timeout_during_stream", endpoint="/v1/chat/completions")
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(httpx.ReadTimeout("")),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except (ValueError, RuntimeError, httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
        _cap_err = type(exc).__name__
        log.error("backend_error_during_stream", endpoint="/v1/chat/completions", error=str(exc))
        error_sse = _chunk_to_openai_sse(
            _make_error_chunk(exc),
            chunk_id,
        )
        yield f"data: {json.dumps(error_sse)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    finally:
        end_capture(_cap_ctx, _cap_tok, error=_cap_err)
        _log_chat_request_perf(timer, request, "/v1/chat/completions")
        _notify_dream_activity(app_state, user_id)
        if len(accumulator.content) >= 50:
            from augmentum.memory.integration import schedule_extraction

            schedule_extraction(
                app_state, request, accumulator.content, session_id, user_id=user_id, mode=mode,
            )
        from augmentum.training.capture import capture_training_trace
        capture_training_trace(request, accumulator.content, session_id, user_id, mode)
        from augmentum.companion_runtime.bus import emit_chat_turn_completed
        await emit_chat_turn_completed(
            app_state,
            mode=mode,
            user_id=user_id,
            session_id=session_id,
            wire_format="openai",
            stream=True,
            user_text=_last_user_text(request),
            assistant_text=accumulator.content,
        )


async def stream_ollama_chat_handler_with_extraction(
    request: InternalChatRequest,
    handler: ModeHandler,
    app_state: State,
    session_id: str,
    mode: str,
    context_length: int = 0,
    user_id: str = "default",
) -> StreamingResponse:
    """Ollama /api/chat streaming with post-stream memory extraction."""
    return StreamingResponse(
        _handler_ndjson_chat_with_extraction(
            request, handler, app_state, session_id, mode,
            context_length=context_length, user_id=user_id,
        ),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


async def stream_ollama_generate_handler_with_extraction(
    request: InternalChatRequest,
    handler: ModeHandler,
    app_state: State,
    session_id: str,
    mode: str,
    user_id: str = "default",
) -> StreamingResponse:
    """Ollama /api/generate streaming with post-stream memory extraction."""
    return StreamingResponse(
        _handler_ndjson_generate_with_extraction(
            request, handler, app_state, session_id, mode, user_id=user_id,
        ),
        media_type="application/x-ndjson",
        headers=_STREAM_HEADERS,
    )


async def stream_openai_chat_handler_with_extraction(
    request: InternalChatRequest,
    handler: ModeHandler,
    app_state: State,
    session_id: str,
    mode: str,
    context_length: int = 0,
    user_id: str = "default",
) -> StreamingResponse:
    """OpenAI /v1/chat/completions streaming with post-stream memory extraction."""
    return StreamingResponse(
        _handler_sse_with_extraction(
            request, handler, app_state, session_id, mode,
            context_length=context_length, user_id=user_id,
        ),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )
