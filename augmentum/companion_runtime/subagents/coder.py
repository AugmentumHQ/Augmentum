"""CoderSubagent — wraps ``modes.coder.CoderHandler``.

Plan/Act software engineering agent with native tool calling and a
workspace container manager. Streaming, like agentic — adapter
collects chunks into one result + per-chunk bus events.
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


class CoderSubagent(SubagentBase):
    name = "coder"
    description = (
        "Software engineering agent (plan/act loop, tool calling, "
        "workspace container). Best when intent is to edit/run code "
        "against a project."
    )
    role_affinity = ("collaborator",)
    focus_affinity = ("owner", "world")
    state_affinity = ("working",)

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            from augmentum.companion_runtime import tiers
            from augmentum.models.base import InternalChatRequest
            from augmentum.modes.coder.handler import CoderHandler
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"coder_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        if app_state is None:
            return SubagentResult(
                content="", handled_by=self.name,
                error="coder_unavailable: runtime has no app_state binding",
            )
        try:
            backend, model_name = await tiers.primary(ctx.runtime)
        except Exception as exc:
            return SubagentResult(
                content="", handled_by=self.name,
                error=f"coder_no_primary_model: {exc!s}",
            )

        request = InternalChatRequest(
            model=model_name,
            messages=[{"role": "user", "content": ctx.intent.text}],
        )
        # NOTE: this SubagentRegistry path is the LEGACY companion dispatcher
        # (runtime.submit_intent, gated by companion_dispatch_enabled=False by
        # default) — not the live companion→coder path. The live path is the
        # `coder.delegate` verb (intent/builtin/coder.py), which resolves a REAL
        # workspace via workspace_resolver (confident-announce / candidate-pick /
        # __new__→create-UI) and enqueues a coder_background_run. Do NOT wire the
        # orchestrator CoderDispatch here: this path has no real workspace
        # (session_id is a random invocation_id), so a dispatch built from it
        # would carry a bogus workspace. The P4 orchestrator contract belongs on
        # the coder.delegate/_enqueue path. (Reverted a mis-wire, 2026-07-27.)
        handler = CoderHandler(
            backend,
            container_manager=getattr(app_state, "container_manager", None),
            # Without these, every coder write (workspace, audit, archive
            # — all user-scoped) landed on the anon row (audit 2026-06-17).
            user_id=ctx.intent.user_id,
            session_id=ctx.invocation_id,
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
            log.exception("coder_stream_failed", error=str(exc))
            return SubagentResult(
                content="".join(chunks), handled_by=self.name,
                error=f"coder_stream_failed: {exc!s}",
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


SubagentRegistry.register(CoderSubagent)
