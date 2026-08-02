"""``media_server_detach`` job handler.

Runs the teardown (or its inverse) for a media server whose sharing
state changed. See ``augmentum/media/detach.py`` for what it does and
``state/migrations/323_file_index_detached.sql`` for why.

This is a job rather than inline route work because a shared library is
tens of thousands of ``file_index`` rows, each carrying an FTS trigger
on update, and the shared aiosqlite connection serializes every other
surface behind whatever is queued on it.

Payload::

    {"server_id": str, "owner_user_id": str, "action": "detach"|"reattach"}

``owner_user_id`` is the user whose rows are left alone — the server's
owner. Both actions are idempotent, so a retried job is harmless.
"""

from __future__ import annotations

from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.media.detach import detach_server_rows, reattach_server_rows
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def make_media_server_detach_handler(app):
    """Bind the handler to ``app.state`` services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        backend = getattr(app.state, "backend", None)
        if backend is None or backend.conn is None:
            return {"status": "skipped", "reason": "backend unavailable"}

        payload = ctx.payload or {}
        server_id = str(payload.get("server_id") or "")
        owner_user_id = str(payload.get("owner_user_id") or "")
        action = str(payload.get("action") or "detach")

        if not server_id:
            return {"status": "skipped", "reason": "no server_id"}
        if action not in ("detach", "reattach"):
            return {"status": "skipped", "reason": f"unknown action {action!r}"}

        run = detach_server_rows if action == "detach" else reattach_server_rows
        rows = await run(
            backend.conn, server_id, owner_user_id=owner_user_id,
        )
        log.info(
            "media_server_detach_job_done",
            server_id=server_id, action=action, rows=rows,
        )
        return {"status": "ok", "action": action, "rows": rows}

    return handler
