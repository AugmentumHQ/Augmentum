"""Memory-consolidate primitive — wraps ``memory.consolidator.try_consolidate``.

LLM-driven merge of related memories in the [0.60, 0.78) similarity
range. The consolidator handles the LLM call; this primitive just
gathers the candidates and forwards.
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


class MemoryConsolidatePrimitive(PrimitiveBase):
    name = "memory_consolidate"
    description = "Merge related memories via the LLM-driven consolidator."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        new_content = kwargs.get("new_content", "").strip()
        candidates = kwargs.get("candidates")  # list[tuple[Memory, float]]
        if not new_content or not candidates:
            return PrimitiveResult(
                ok=False,
                error="memory_consolidate: need new_content + candidates",
            )

        try:
            from augmentum.memory.consolidator import try_consolidate
        except Exception as exc:
            return PrimitiveResult(
                ok=False,
                error=f"memory_consolidate_import_failed: {exc!s}",
            )

        app_state = getattr(ctx.runtime, "_app_state", None)
        provider_registry = getattr(app_state, "provider_registry", None) if app_state else None
        backend = getattr(provider_registry, "default_backend", None) if provider_registry else None
        model = kwargs.get("model", "")

        try:
            merged = await try_consolidate(new_content, candidates, backend, model or None)
        except Exception as exc:
            log.exception("memory_consolidate_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"memory_consolidate_failed: {exc!s}")

        if merged is None:
            return PrimitiveResult(ok=True, payload=None, metadata={"merged": False})
        content, importance = merged
        return PrimitiveResult(
            ok=True,
            payload={"content": content, "importance": importance},
            metadata={"merged": True},
        )


PrimitiveRegistry.register(MemoryConsolidatePrimitive)
