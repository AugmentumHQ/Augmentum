"""PassthroughSubagent — wraps ``modes.passthrough.SSOSOrchestrator``.

The passthrough mode is also the validation point for the
heuristic-primary + LLM-tie-breaker dispatch pattern (sprint plan §3,
§4 cite this). The orchestrator already classifies query intent
heuristically and only consults the LLM at synthesis time.

Adapter contract: translate a runtime ``Intent`` into the orchestrator's
``InternalChatRequest`` shape, then return the synthesis seed it
emits. Sprint 2 is inert (registered but registry flag off), so the
actual translation is exercised only in Sprint 3.
"""

from __future__ import annotations

from augmentum.companion_runtime.subagents.base import (
    SubagentBase,
    SubagentContext,
    SubagentResult,
)
from augmentum.companion_runtime.subagents.registry import SubagentRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class PassthroughSubagent(SubagentBase):
    name = "passthrough"
    description = (
        "Direct LLM passthrough with deterministic tool pre-execution. "
        "Best for general questions, web search, calculations — anything "
        "that doesn't need multi-turn planning."
    )
    role_affinity = ("companion", "collaborator")
    focus_affinity = ("owner", "world")
    state_affinity = ("attentive", "idle")

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
            from augmentum.models.base import InternalChatRequest
            from augmentum.companion_runtime import tiers
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"passthrough_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        tool_registry = getattr(app_state, "tool_registry", None) if app_state else None
        if tool_registry is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="passthrough_unavailable: tool_registry not bound to runtime",
                metadata={"note": "Sprint 3 dispatch wires app_state into runtime"},
            )

        try:
            _backend, model_name = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"passthrough_no_primary_model: {exc!s}",
            )

        request = InternalChatRequest(
            model=model_name,
            messages=[{"role": "user", "content": ctx.intent.text}],
        )
        orch = SSOSOrchestrator(
            tool_registry,
            user_id=ctx.intent.user_id,
            app_state=app_state,
        )

        await ctx.bus.publish_topic(
            "subagent.invoked",
            {"name": self.name, "invocation_id": ctx.invocation_id},
            source_companion_id=ctx.companion_id,
        )
        try:
            seeded = await orch.try_orchestrate(request)
        except Exception as exc:
            log.exception("passthrough_orchestrate_failed", error=str(exc))
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"passthrough_orchestrate_failed: {exc!s}",
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {"name": self.name, "invocation_id": ctx.invocation_id},
                source_companion_id=ctx.companion_id,
            )

        if seeded is None:
            return SubagentResult(
                content="", handled_by=self.name,
                metadata={"note": "orchestrator declined to handle"},
            )
        # ``InternalChatRequest.__post_init__`` coerces every message to
        # the ``Message`` dataclass, so attribute access — not dict
        # ``.get`` — is the contract here. The dict form crashed every
        # invocation ('Message' object has no attribute 'get') the
        # moment the headless path started exercising this adapter.
        last = seeded.messages[-1] if seeded.messages else None
        content = getattr(last, "content", "") if last is not None else ""
        return SubagentResult(
            content=str(content or ""),
            handled_by=self.name,
            metadata={"seeded": True},
        )


SubagentRegistry.register(PassthroughSubagent)
