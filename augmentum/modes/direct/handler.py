"""Direct mode handler — pure pass-through to the backend.

The route layer (``openai_routes.openai_chat``, ``ollama_routes.ollama_chat``)
short-circuits to this handler immediately after resolving the backend,
bypassing memory recall, knowledge-pack injection, dream context, media
context, vision-caption fallback, file-token expansion, SSOS tools, mode
inference hints, and prompt-cache key pinning.

What this handler still does:
    * Forwards the user's messages verbatim to ``backend.chat`` /
      ``backend.chat_stream``.
    * Yields/returns the backend response exactly as received.

What this handler does NOT do (the explicit list, for reviewer sanity):
    * No datetime injection (``_INJECT_DATETIME = False``).
    * No system-prompt scaffolding.
    * No tool synthesis or tool-loop ceremony.
    * No fallback to PassthroughHandler — direct is a contract, not a
      degraded mode. Failures bubble up as backend errors.

If a future feature is genuinely "must run on every backend call"
(e.g. usage telemetry, billing), wire it at the BACKEND layer, NOT
here. Direct is the one mode that promises the caller "what you send
is what the model sees".

See ``docs/superpowers/specs/2026-06-02-direct-mode-design.md`` if/when
a spec is written; today this docstring is the canonical reference.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    ModelBackend,
)
from augmentum.modes.base import ModeHandler
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class DirectHandler(ModeHandler):
    """Trivial pass-through handler — see module docstring for the contract."""

    # The base class injects an authoritative datetime into the system
    # prompt by default. Direct mode is "what you send is what the model
    # sees" — even a single system message tacked on is a contract
    # violation for callers that are testing raw model behaviour.
    _INJECT_DATETIME = False

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    async def _handle(
        self, request: InternalChatRequest,
    ) -> InternalChatResponse:
        return await self._backend.chat(request)

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        async for chunk in self._backend.chat_stream(request):
            yield chunk
