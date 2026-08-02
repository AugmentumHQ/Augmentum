"""Reference Resolver tool — surfaces ``resolver.resolve_moments`` to chat / agentic / UARF.

Lets the model resolve **descriptive** references — anything the user
points at without naming directly. The model calls ``resolve_moments``
with the natural-language phrasing; the resolver returns ranked
candidate moments from across the user's indexed content; the model
chooses, confirms, or asks for disambiguation.

The resolver itself is in :mod:`augmentum.resolver`. This module is the
thin tool wrapper that adapts inputs/outputs to the ``Tool`` interface
and threads ``user_id`` per the chain pattern (``_user_id`` or
``_context["user_id"]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.memory import CompanionMemory
    from augmentum.vfs.index import FileIndexService

log = get_logger(__name__)


class ResolveMomentsTool(Tool):
    """Resolve a natural-language reference to ranked moments.

    Constructor takes the per-app singletons (file_index, optional
    CompanionMemory). ``user_id`` is threaded per-request via the
    standard chain convention.
    """

    def __init__(
        self,
        file_index: FileIndexService | None = None,
        memory: CompanionMemory | None = None,
    ) -> None:
        self._file_index = file_index
        self._memory = memory

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "resolve_moments"

    @property
    def description(self) -> str:
        return (
            "Resolve a descriptive reference to ranked candidate items "
            "from the user's indexed content. Use this whenever the user "
            "points at something by description, attribute, time, topic, "
            "or association rather than by exact name (\"the document I "
            "was reading earlier\", \"the picture from the trip\", "
            "\"the note about that idea\"). Searches files and journal "
            "entries via hybrid retrieval (vector + keyword). Returns "
            "structured results that the UI can render as a "
            "disambiguation card when confidence is split."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language reference to resolve.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of moments to return.",
                    "default": 10,
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["file", "journal"]},
                    "description": (
                        "Which sources to search. Default both. Pass "
                        "[\"file\"] to restrict to file_index, "
                        "[\"journal\"] to restrict to journal."
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, error="query is required")

        # User-id resolution per the chain convention. Either slot is
        # acceptable; missing both is an error because the underlying
        # retrieval is user-scoped.
        user_id = kwargs.get("_user_id") or ""
        if not user_id:
            ctx = kwargs.get("_context") or {}
            if isinstance(ctx, dict):
                user_id = str(ctx.get("user_id") or "")
        if not user_id:
            return ToolResult(
                success=False,
                error="user_id missing — tool must be invoked from a user-scoped path",
            )

        limit = int(kwargs.get("limit") or 10)
        # Clamp limit to a sane range so a runaway model can't trigger
        # an unbounded retrieval.
        limit = max(1, min(limit, 50))

        raw_kinds = kwargs.get("kinds") or ["file", "journal"]
        kinds = tuple(k for k in raw_kinds if k in ("file", "journal"))
        if not kinds:
            kinds = ("file", "journal")

        from augmentum.resolver import resolve_moments

        try:
            moments = await resolve_moments(
                query,
                user_id=user_id,
                file_index=self._file_index,
                memory=self._memory,
                limit=limit,
                kinds=kinds,
            )
        except Exception as exc:
            log.exception("resolve_moments_failed", query=query[:80])
            return ToolResult(success=False, error=f"resolver error: {exc}")

        if not moments:
            return ToolResult(
                success=True,
                output="No matching moments found.",
                metadata={"count": 0, "moments": []},
            )

        # Compact human-readable summary for the model to consume.
        # The full structured list lives in metadata so the UI can
        # render the disambiguation card directly.
        lines = []
        for i, m in enumerate(moments, 1):
            label = m.title or m.id
            lines.append(
                f"{i}. [{m.kind}] {label} — {m.snippet}"
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "count": len(moments),
                "moments": [m.to_dict() for m in moments],
            },
        )
