"""AgenticSubagent — wraps ``modes.agentic.AgenticHandler``.

Multi-step task orchestration with plan generation, checkpoints,
approval gates. Streaming-native — the underlying handler yields chunks.
Sprint 2's adapter collects chunks into a single string so the unified
``SubagentResult`` shape works; Sprint 3 will grow per-token streaming
via the bus.
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


class AgenticSubagent(SubagentBase):
    name = "agentic"
    description = (
        "Multi-step task agent: plans, approves, executes with tool "
        "chain. Best when the intent describes a multi-action goal "
        "(\"do X, then Y, then summarize\")."
    )
    role_affinity = ("collaborator", "host")
    focus_affinity = ("owner", "household", "world")
    state_affinity = ("working",)

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            from augmentum.modes.agentic.handler import AgenticHandler
            from augmentum.models.base import InternalChatRequest
            from augmentum.companion_runtime import tiers
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"agentic_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        if app_state is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="agentic_unavailable: runtime has no app_state binding",
            )
        try:
            backend, model_name = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"agentic_no_primary_model: {exc!s}",
            )

        request = InternalChatRequest(
            model=model_name,
            messages=[{"role": "user", "content": ctx.intent.text}],
        )
        handler = AgenticHandler(
            backend,
            tool_registry=getattr(app_state, "tool_registry", None),
            session_id=ctx.invocation_id,
            user_id=ctx.intent.user_id,
            task_store=getattr(app_state, "task_store", None),
        )

        await ctx.bus.publish_topic(
            "subagent.invoked",
            {"name": self.name, "invocation_id": ctx.invocation_id},
            source_companion_id=ctx.companion_id,
        )
        chunks: list[str] = []
        try:
            async for chunk in handler._handle_stream(request):  # noqa: SLF001
                text = getattr(chunk, "text", None) or getattr(chunk, "content", "") or ""
                if text:
                    chunks.append(text)
                    await ctx.bus.publish_topic(
                        "subagent.chunk",
                        {"name": self.name, "invocation_id": ctx.invocation_id,
                         "delta_len": len(text)},
                        source_companion_id=ctx.companion_id,
                    )
        except Exception as exc:
            log.exception("agentic_stream_failed", error=str(exc))
            return SubagentResult(
                content="".join(chunks), handled_by=self.name,
                error=f"agentic_stream_failed: {exc!s}",
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {"name": self.name, "invocation_id": ctx.invocation_id,
                 "chunks": len(chunks)},
                source_companion_id=ctx.companion_id,
            )

        return SubagentResult(
            content="".join(chunks),
            handled_by=self.name,
            metadata={"streaming": True, "chunk_count": len(chunks)},
        )


SubagentRegistry.register(AgenticSubagent)
