"""NarrativeSubagent — wraps ``modes.narrative.NarrativeEngine``.

In-world fiction renderer: character state, plot tracking, memory
buffers, lore expansion before LLM synthesis. The right subagent when
focus is collaborative storytelling rather than analysis or action.
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


class NarrativeSubagent(SubagentBase):
    name = "narrative"
    description = (
        "Collaborative fiction engine with character/plot/lore state. "
        "Best for in-world dialogue, scene-building, storytelling sessions."
    )
    role_affinity = ("companion", "collaborator")
    focus_affinity = ("owner", "self")
    state_affinity = ("attentive", "working")

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            from augmentum.models.base import InternalChatRequest
            from augmentum.companion_runtime import tiers
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"narrative_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        engine = getattr(app_state, "narrative_engine", None) if app_state else None
        if engine is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="narrative_unavailable: app.state.narrative_engine missing",
            )

        try:
            _backend, model_name = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"narrative_no_primary_model: {exc!s}",
            )

        request = InternalChatRequest(
            model=model_name,
            messages=[{"role": "user", "content": ctx.intent.text}],
        )

        await ctx.bus.publish_topic(
            "subagent.invoked",
            {"name": self.name, "invocation_id": ctx.invocation_id},
            source_companion_id=ctx.companion_id,
        )
        try:
            result = await engine.process(request)
        except Exception as exc:
            log.exception("narrative_process_failed", error=str(exc))
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"narrative_process_failed: {exc!s}",
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {"name": self.name, "invocation_id": ctx.invocation_id},
                source_companion_id=ctx.companion_id,
            )

        content = getattr(result, "content", "") or getattr(result, "text", "") or str(result)
        return SubagentResult(content=content, handled_by=self.name)


SubagentRegistry.register(NarrativeSubagent)
