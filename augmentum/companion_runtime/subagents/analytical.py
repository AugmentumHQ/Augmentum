"""AnalyticalSubagent — wraps ``modes.analytical.AnalyticalEngine``.

The analytical mode runs a multi-phase reasoning pipeline (ASSESS →
IDENTIFY/SEARCH → APPLY → CONCLUDE). It's the right subagent when the
intent calls for stepwise breakdown rather than direct answer.
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


class AnalyticalSubagent(SubagentBase):
    name = "analytical"
    description = (
        "Multi-phase analytical reasoning (assess → identify → apply → "
        "conclude). Best for problems benefiting from stepwise breakdown."
    )
    role_affinity = ("collaborator",)
    focus_affinity = ("owner", "world")
    state_affinity = ("attentive", "working")

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            from augmentum.models.base import InternalChatRequest
            from augmentum.companion_runtime import tiers
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"analytical_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        engine = getattr(app_state, "analytical_engine", None) if app_state else None
        if engine is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="analytical_unavailable: app.state.analytical_engine missing",
            )

        try:
            _backend, model_name = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"analytical_no_primary_model: {exc!s}",
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
            log.exception("analytical_process_failed", error=str(exc))
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"analytical_process_failed: {exc!s}",
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {"name": self.name, "invocation_id": ctx.invocation_id},
                source_companion_id=ctx.companion_id,
            )

        content = getattr(result, "content", "") or getattr(result, "text", "") or str(result)
        return SubagentResult(
            content=content,
            handled_by=self.name,
            metadata={"phase_count": getattr(result, "phase_count", None)},
        )


SubagentRegistry.register(AnalyticalSubagent)
