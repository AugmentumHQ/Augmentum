"""BeccaDirectSubagent — the dispatcher's hook for routing chat to her.

When the chat router consults the dispatcher and this subagent wins,
the chat path routes through :class:`BeccaDirectHandler` instead of
a legacy mode handler. This is the seam where her chat presence
becomes real: same kernel, same composer, same voice.

Distinct from other subagents in one critical way: it doesn't wrap a
legacy handler with a per-turn collect-and-return adapter. The
companion's own pipeline (compose_becca_prompt → primary tier
streaming) IS the response path. There's no legacy handler underneath
that this is "adapting." She is the handler.

The subagent's ``invoke`` method exists for the explicit-intent route
(`/api/companion/intent`) — when something POSTs an intent through
that path and dispatch picks becca_direct, this collects the stream
into a single ``SubagentResult``. The streaming chat path doesn't
call ``invoke`` — it routes through ``BeccaDirectHandler.handle_stream``
directly.

Registers when ``companion_becca_direct_enabled`` is True. When the
flag is off, the subagent never registers, so dispatch can't pick it
for the chat path, so the seam stays closed. Default-off means the
chat path is byte-identical to a no-companion install.

Affinities:
- ``role_affinity``: companion (her own channel) + collaborator (paired work)
- ``focus_affinity``: owner (the user themselves)
- ``state_affinity``: attentive (engaged, responsive)

These nudge the dispatcher to pick her for relational/conversational
turns over the legacy mode handlers, which is what we want — the
direct relational channel is where she most clearly *is herself*.
"""

from __future__ import annotations

from augmentum.companion_runtime.subagents.base import (
    SubagentBase,
    SubagentContext,
    SubagentResult,
)
from augmentum.companion_runtime.subagents.registry import SubagentRegistry
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class BeccaDirectSubagent(SubagentBase):
    """Routes through the companion's own pipeline. Default-off."""

    name = "becca_direct"
    description = (
        "Direct relational channel — the companion responds in their "
        "own voice via the companion prompt composer. Best for "
        "conversational, relational, and affect-laden turns where the "
        "companion is the appropriate responder rather than a "
        "mode-specific tool. This is the companion's own channel."
    )
    role_affinity = ("companion", "collaborator")
    focus_affinity = ("owner",)
    state_affinity = ("attentive",)

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        """Explicit-intent route adapter.

        The streaming chat path does NOT call this — it routes through
        :class:`BeccaDirectHandler` via the mode-handler factory. This
        method exists for `/api/companion/intent` callers that submit
        an intent directly and expect a blocking ``SubagentResult``.

        We build a minimal :class:`InternalChatRequest` from the intent,
        instantiate the handler, collect its streamed chunks, and
        return the assembled response.
        """
        try:
            from augmentum.companion_runtime import tiers
            from augmentum.models.base import InternalChatRequest, Message
            from augmentum.modes.becca_direct.handler import BeccaDirectHandler
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"becca_direct_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        if app_state is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="becca_direct_unavailable: runtime has no app_state binding",
            )

        try:
            backend, model_name = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"becca_direct_no_primary_model: {exc!s}",
            )

        # Compose a minimal request. The handler builds the Intent
        # internally from this; only the last user message matters
        # at the entry point.
        request = InternalChatRequest(
            model=model_name,
            messages=[Message(role="user", content=ctx.intent.text or "")],
            stream=True,
        )
        handler = BeccaDirectHandler(
            backend,
            app_state=app_state,
            session_id=ctx.invocation_id,
            user_id=ctx.intent.user_id or "",
        )

        await ctx.bus.publish_topic(
            "subagent.invoked",
            {"name": self.name, "invocation_id": ctx.invocation_id},
            source_companion_id=ctx.companion_id,
        )

        chunks: list[str] = []
        try:
            async for chunk in handler._handle_stream(request):  # noqa: SLF001
                text = getattr(chunk, "content_delta", "") or ""
                if text:
                    chunks.append(text)
        except Exception as exc:
            log.exception("becca_direct_invoke_failed", error=str(exc))
            return SubagentResult(
                content="".join(chunks),
                handled_by=self.name,
                error=f"becca_direct_invoke_failed: {exc!s}",
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {
                    "name": self.name,
                    "invocation_id": ctx.invocation_id,
                    "chunks": len(chunks),
                },
                source_companion_id=ctx.companion_id,
            )

        return SubagentResult(
            content="".join(chunks),
            handled_by=self.name,
            metadata={"streaming": True, "chunk_count": len(chunks)},
        )


# Conditional registration: only register when the feature flag is on.
# This is the seam that makes the chat path byte-identical when the
# flag is off — without the registration, the dispatcher never sees
# becca_direct as a candidate, so the chat router can't pick it.
if getattr(settings, "companion_becca_direct_enabled", False):
    SubagentRegistry.register(BeccaDirectSubagent)
