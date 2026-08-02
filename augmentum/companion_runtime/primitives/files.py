"""Files primitive — search & metadata on the user's file index.

Thin wrapper over ``augmentum.tools.search_files``. Sprint 3 may grow
this into a full read/write set; Sprint 2 ships read/search only.
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


class FilesPrimitive(PrimitiveBase):
    name = "files"
    description = "Search the indexed user files by query."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        query = kwargs.get("query", "").strip()
        if not query:
            return PrimitiveResult(ok=False, error="files: empty query")

        try:
            from augmentum.tools.search_files import SearchFilesTool
        except Exception as exc:
            return PrimitiveResult(ok=False, error=f"files_import_failed: {exc!s}")

        app_state = getattr(ctx.runtime, "_app_state", None)
        tool = SearchFilesTool()
        tool_ctx = {
            "user_id": ctx.user_id,
            "app_state": app_state,
        }
        try:
            result = await tool.execute(
                query=query,
                limit=int(kwargs.get("limit", 5)),
                _context=tool_ctx,
            )
        except Exception as exc:
            log.exception("files_search_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"files_search_failed: {exc!s}")

        payload = getattr(result, "result", None) or getattr(result, "content", str(result))
        return PrimitiveResult(ok=True, payload=payload)


PrimitiveRegistry.register(FilesPrimitive)
