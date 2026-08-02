"""Abstract mode handler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)
from augmentum.utils.datetime_context import get_datetime_context

# Idempotency marker — must be a STABLE substring of the block that
# ``get_datetime_context()`` actually emits. The block is wrapped in a
# ``<current_time>`` tag; the human-readable line inside is "Current date: …
# Current time: …" (NOT "Current date/time:", which an earlier block format
# used — that stale marker silently never matched, leaving this guard dead).
_DT_MARKER = "<current_time>"


class ModeHandler(ABC):
    """Base class for all processing modes (passthrough, analytical, narrative).

    Automatically injects authoritative date/time into the system prompt
    exactly once, regardless of mode.  Subclasses implement ``_handle``
    and ``_handle_stream`` instead of the public methods.
    """

    # Subclasses set False to opt out of real-world datetime injection.
    # Narrative mode uses this to keep its in-world fiction (and to keep
    # the system prefix byte-stable across turns so llama-server's KV
    # cache survives — a fresh "Current time: HH:MM" every turn wrecks
    # the prompt-prefix hash and forces full prefill on every reply).
    _INJECT_DATETIME: bool = True

    # ------------------------------------------------------------------
    # Public API — calls _ensure_datetime then delegates to subclass
    # ------------------------------------------------------------------

    async def handle(self, request: InternalChatRequest) -> InternalChatResponse:
        """Process a non-streaming request (datetime injected automatically)."""
        if self._INJECT_DATETIME:
            self._ensure_datetime(request)
        return await self._handle(request)

    async def handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Process a streaming request (datetime injected automatically)."""
        if self._INJECT_DATETIME:
            self._ensure_datetime(request)
        async for chunk in self._handle_stream(request):
            yield chunk

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def _handle(self, request: InternalChatRequest) -> InternalChatResponse:
        """Process a non-streaming request."""
        ...

    @abstractmethod
    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Process a streaming request, yielding chunks."""
        ...

    # ------------------------------------------------------------------
    # Datetime injection (idempotent — safe to call multiple times)
    # ------------------------------------------------------------------

    def _ensure_datetime(self, request: InternalChatRequest) -> None:
        """Inject authoritative date/time into the system prompt.

        Picks the position based on the backend's caching contract:

        * Backends that support mid-conversation system messages
          (llama-server, Ollama, our own engine — see
          ``ModelBackend.supports_mid_conversation_system``) get the
          datetime as a separate system message inserted just before
          the latest user turn. The minute-precision timestamp then
          sits at the END of the cacheable region, so the long stable
          system + history prefix above it survives the radix prefix
          cache across turns. Without this, ``Current time: HH:MM``
          rotating in position 0 invalidates the cache every minute
          and forces a full re-prefill on every turn (observed
          2026-05-17 on Qwen3.6-35B-A3B: 2021/2025 tokens re-prefilled
          per turn, 9-20 s TTFT on a stable conversation).

        * Backends that don't (cloud OpenAI-compat APIs — many of them
          reject system messages after position 0 with a 400) fall
          back to the legacy behavior: prepend to the leading system
          block. The cache benefit doesn't apply to those backends
          anyway.

        Idempotent — checks every message for the marker so a second
        call within the same request doesn't duplicate the block.
        """
        # Already injected anywhere in the message stack? Skip.
        for msg in request.messages:
            if _DT_MARKER in (msg.content or ""):
                return

        dt_block = get_datetime_context()

        # Backend-aware placement. Default to position 0 if the backend
        # is unknown (no _backend attribute, e.g. tests) or doesn't
        # accept mid-conversation system messages — that's the
        # universally accepted shape.
        backend = getattr(self, "_backend", None)
        mid_ok = getattr(backend, "supports_mid_conversation_system", False)

        if mid_ok and request.messages and request.messages[-1].role == "user":
            # Append as a standalone system message at the absolute END
            # of the payload — strictly AFTER the last user turn, never
            # before it. Placement before the last user message looks
            # equivalent but is not: next turn, the position this block
            # occupied is taken by conversation content, so the byte
            # stream diverges one message EARLIER than the moving
            # suffix and every KV reuse tier (slot LCP, checkpoint
            # restore) is forfeited (observed 2026-07-09: passthrough
            # contract=violated at divergent_index=1, tier
            # cold_no_checkpoint, slot sim 0.299). At the tail, the
            # previous turn's stale block is plain truncated divergence
            # and everything before it stays reusable. The llama.cpp
            # backend converts this trailing system into a user-role
            # carrier for strict templates.
            request.messages.append(Message(role="system", content=dt_block))
            return

        # Fallback for backends that don't accept a mid-conversation system
        # message (the base default; strict cloud OpenAI-compat APIs 400 on a
        # system message after position 0). This USED to prepend the block to
        # the leading system message — which rotates a minute-precision
        # timestamp at the HEAD of the cacheable prefix and invalidates the
        # prefix cache every minute. Harmless for a true cloud API (no prefix
        # cache we control), but WRONG for a self-hosted OpenAI-compat engine —
        # a custom llama.cpp / vLLM / llama-swap backend, or the native engine
        # reached over the OpenAI-compat wire — which DOES keep a radix prefix
        # cache (observed 2026-07-29: passthrough kv_prefix_stability
        # contract=violated, divergent_index=0, stable_pct=0.008 on a custom
        # engine that inherits supports_mid_conversation_system=False).
        #
        # Fix: put the block at the TAIL of the last user message instead of
        # the head. Dynamic tokens then sit at the very end of the payload for
        # EVERY backend — the long stable system+history prefix above stays
        # byte-identical across turns (cache-reusable), there is no post-0
        # system message for a strict cloud API to reject, and it also lets
        # cloud prompt-caching (OpenAI/Anthropic auto prefix cache) hit. Same
        # authoritative content, only relocated, so date/time grounding is
        # unchanged. This mirrors the mid_ok path's tail intent; that path can
        # use a standalone trailing SYSTEM message, whereas here we fold into
        # the user turn because a trailing system message is exactly what these
        # backends reject.
        if request.messages and request.messages[-1].role == "user":
            last = request.messages[-1]
            existing = last.content or ""
            request.messages[-1] = Message(
                role="user",
                content=f"{existing}\n\n{dt_block}" if existing else dt_block,
            )
            return

        # No trailing user turn (unusual shape — e.g. a request ending in an
        # assistant or system message). Preserve the historical
        # prepend-to-leading-system behavior: there's no active user turn whose
        # prefix we'd be protecting, so the cache cost is moot, and this keeps
        # the block on a role every backend accepts.
        if request.messages and request.messages[0].role == "system":
            request.messages[0] = Message(
                role="system",
                content=f"{dt_block}\n\n{request.messages[0].content}",
            )
        else:
            request.messages.insert(0, Message(role="system", content=dt_block))
