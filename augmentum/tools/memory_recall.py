"""Memory recall tool — lets UARF query the memory system on-demand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore


class MemoryRecallTool(Tool):
    """Recall relevant memories from the cross-session memory store.

    Available in RELEVANT and APPLY phases so UARF can query user context
    on-demand instead of relying on always-on injection.
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._store = memory_store

    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return (
            "Search the user's cross-session memory for relevant context, "
            "preferences, facts, or prior instructions. Use when the user's "
            "question may relate to something they previously told the system."
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
                    "description": "Search query to find relevant memories",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return",
                    "default": 5,
                },
                "memory_type": {
                    "type": "string",
                    "description": "Optional filter: 'fact', 'preference', 'entity', 'narrative', 'analysis'",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        if not query:
            return ToolResult(success=False, error="query is required")

        # Memory is user-scoped: recall() defaults user_id to "default"
        # when omitted, so a tool that forgot to pass it searched the
        # wrong bucket and returned "No relevant memories found" for
        # every real user. Route it through the canonical extractor.
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="No user context — can't reach the user's memory.",
            )

        limit = kwargs.get("limit", 5)
        memory_type_str = kwargs.get("memory_type")

        memory_types = None
        if memory_type_str:
            from augmentum.memory.models import MemoryType

            try:
                memory_types = [MemoryType(memory_type_str)]
            except ValueError:
                return ToolResult(
                    success=False,
                    error=f"Invalid memory_type: {memory_type_str}",
                )

        try:
            memories = await self._store.recall(
                query, user_id=user_id, min_score=0.005, limit=limit,
                memory_types=memory_types,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Memory recall failed: {exc}")

        from augmentum.memory.register import HONEST_EMPTY_NOTE, register_label

        if not memories:
            # Honest gap: the model explicitly reached for memory and found
            # nothing — tell it to say so rather than confabulate (Earned
            # Understanding, the gated S-B honesty floor).
            return ToolResult(
                success=True,
                output=HONEST_EMPTY_NOTE,
                metadata={"count": 0},
            )

        lines = []
        for mem in memories:
            type_tag = f"[{mem.memory_type}]" if isinstance(mem.memory_type, str) else f"[{mem.memory_type.value}]"
            # Calibrated voice: carry the earned-tier confidence cue so the
            # model speaks an unproven hit tentatively, a CORE hit plainly.
            lines.append(f"- {type_tag} [{register_label(mem.tier)}] {mem.content}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"count": len(memories)},
        )
