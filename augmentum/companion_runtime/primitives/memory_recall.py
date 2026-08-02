"""Memory-recall primitive — wraps ``MemoryStore.recall``.

The existing 4-tier memory subsystem already does hybrid vector +
FTS5 retrieval; this primitive surfaces it to subagents through the
runtime registry. Companion-scoped: results are filtered by
``companion_id`` so siblings don't read each other's memories
(Sprint 7 makes that a hard isolation; Sprint 2 is best-effort).
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class MemoryRecallPrimitive(PrimitiveBase):
    name = "memory_recall"
    description = (
        "Hybrid vector + FTS5 recall over Becca's memory subsystem, "
        "companion-scoped."
    )

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        query = kwargs.get("query", "").strip()
        if not query:
            return PrimitiveResult(ok=False, error="memory_recall: empty query")

        runtime = ctx.runtime
        memory = getattr(runtime, "memory", None)
        if memory is None or memory._store is None:  # noqa: SLF001
            return PrimitiveResult(
                ok=False,
                error="memory_recall: runtime.memory has no store attached",
            )

        limit = int(kwargs.get("limit", 5))
        try:
            hits = await memory.recall(
                query=query,
                # Always the caller's identity — never a model-supplied
                # ``user_id`` kwarg. Honouring a model override let the
                # model recall *another* user's memories (audit 2026-06-17).
                user_id=ctx.user_id,
                k=limit,
            )
        except Exception as exc:
            log.exception("memory_recall_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"memory_recall_failed: {exc!s}")

        return PrimitiveResult(
            ok=True,
            payload=hits,
            metadata={"hit_count": len(hits) if hasattr(hits, "__len__") else None},
        )


PrimitiveRegistry.register(MemoryRecallPrimitive)
