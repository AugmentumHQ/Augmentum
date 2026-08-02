"""Browse primitive — wraps web fetch / coder browser.

Two entry points coexist in the codebase: ``augmentum.coder.browser``
(structured snapshot with form metadata) and ``augmentum.tools.web_fetch``
(plain HTTP fetch). This adapter prefers the structured one when a
workspace context is available, falls back to plain fetch otherwise.
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


class BrowsePrimitive(PrimitiveBase):
    name = "browse"
    description = "Fetch a web page and return its structured snapshot."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        url = kwargs.get("url", "").strip()
        if not url:
            return PrimitiveResult(ok=False, error="browse: empty url")
        timeout = float(kwargs.get("timeout", 8.0))

        app_state = getattr(ctx.runtime, "_app_state", None)
        cm = getattr(app_state, "container_manager", None) if app_state else None
        workspace_id = kwargs.get("workspace_id", "")

        if cm is not None and workspace_id:
            try:
                from augmentum.coder.browser import http_snapshot
                snap = await http_snapshot(cm, workspace_id, url, timeout=timeout)
                return PrimitiveResult(ok=True, payload=snap)
            except Exception as exc:
                log.warning("browse_structured_failed", error=str(exc))

        try:
            from augmentum.tools.web_fetch import WebFetchTool
            tool = WebFetchTool()
            result = await tool.execute(url=url)
        except Exception as exc:
            log.exception("browse_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"browse_failed: {exc!s}")

        content = getattr(result, "content", None) or getattr(result, "result", str(result))
        return PrimitiveResult(ok=True, payload=content, metadata={"mode": "plain_fetch"})


PrimitiveRegistry.register(BrowsePrimitive)
