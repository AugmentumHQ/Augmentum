"""BuildSubagent — placeholder.

The build capability lives in ``augmentum/builds/runtime.py`` as
helper functions; there is no mode orchestrator yet. This adapter
registers a thin subagent so dispatch (Sprint 3) can list "build" as
an option, and routes through the runtime helpers when invoked. If
the helpers don't exist or fail to import, the adapter degrades to a
clean error rather than crashing dispatch.
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


class BuildSubagent(SubagentBase):
    name = "build"
    description = (
        "Builds/composes structured artifacts (manifests, configs, "
        "release bundles). Wraps augmentum.builds runtime helpers."
    )
    role_affinity = ("collaborator", "host")
    focus_affinity = ("owner", "household")
    state_affinity = ("working",)

    async def invoke(self, ctx: SubagentContext) -> SubagentResult:
        try:
            import augmentum.builds.runtime as _builds  # noqa: F401
        except Exception:
            return SubagentResult(
                content="", handled_by=self.name,
                error="build_unavailable: no augmentum.builds.runtime",
                metadata={"note": "Implement build mode orchestrator first"},
            )

        await ctx.bus.publish_topic(
            "subagent.invoked",
            {"name": self.name, "invocation_id": ctx.invocation_id},
            source_companion_id=ctx.companion_id,
        )
        try:
            return SubagentResult(
                content="",
                handled_by=self.name,
                metadata={"note": "Build adapter is a placeholder; "
                          "wire to specific runtime helper in Sprint 3+"},
            )
        finally:
            await ctx.bus.publish_topic(
                "subagent.completed",
                {"name": self.name, "invocation_id": ctx.invocation_id},
                source_companion_id=ctx.companion_id,
            )


SubagentRegistry.register(BuildSubagent)
