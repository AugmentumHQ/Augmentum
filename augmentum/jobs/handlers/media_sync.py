"""``media_sync`` job handler.

Runs a user media-server catalog sync in the generic background-jobs queue.
This keeps large Audiobookshelf libraries off the request path while still
reusing the same ``sync_server`` implementation the synchronous route used.
"""

from __future__ import annotations

from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.media.store import MediaServerStore
from augmentum.media.sync import sync_server
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger
from augmentum.vfs.bulk import bulk_index_session

log = get_logger(__name__)


def make_media_sync_handler(app):
    """Build a handler bound to runtime app services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        http_client = getattr(app.state, "http_client", None)
        sm = getattr(app.state, "state_manager", None)
        backend = getattr(sm, "backend", None) if sm else None
        if http_client is None or not isinstance(backend, SQLiteBackend):
            raise RuntimeError("media_sync: http_client or sqlite backend not initialized")

        server_id = str(ctx.payload.get("server_id") or "").strip()
        if not server_id:
            raise RuntimeError("media_sync: malformed payload (missing server_id)")

        store = MediaServerStore(backend.conn)
        # get_visible so non-owners of admin-shared servers (scope='shared')
        # can run a sync. The catalog rows themselves land under
        # ``ctx.user_id`` via ``target_user_id`` below — the shared
        # connection is just credentials, the per-user file_index is not.
        server = await store.get_visible(server_id, user_id=ctx.user_id)
        if server is None:
            log.info(
                "media_sync_server_missing",
                job_id=ctx.job_id, server_id=server_id, user_id=ctx.user_id,
            )
            return {"skipped": "server_missing", "server_id": server_id}

        async def _progress(progress: float, stage: str) -> None:
            await ctx.check_cancel()
            await ctx.update_progress(progress, stage=stage)
            # ``store.update`` filters by user_id, so for a non-owner
            # sync of a shared server this no-ops on the server row
            # (admin's stats stay admin's). The caller still sees
            # progress via the job row, which IS scoped to ctx.user_id.
            await store.update(
                server.id,
                user_id=ctx.user_id,
                status="syncing",
                status_detail=stage,
            )

        await store.update(
            server.id,
            user_id=ctx.user_id,
            status="syncing",
            status_detail="Preparing sync",
        )
        await ctx.update_progress(0.02, stage="Preparing sync")

        # Catalog indexing runs on a dedicated connection with batched
        # commits and a yield between batches. Without this a large
        # library's scan enqueues one commit per item onto the shared
        # aiosqlite worker thread, so every concurrent request — another
        # user's voice turn, a chat completion — waits behind it. See
        # augmentum/vfs/bulk.py. The session degrades to the shared
        # connection (batching kept, isolation lost) rather than failing.
        async with bulk_index_session(backend) as bulk:
            indexed, err = await sync_server(
                server,
                store=store,
                http_client=http_client,
                progress_callback=_progress,
                # Per-user catalog rows (file_index, library_views,
                # comic_series) land under the caller's id even when the
                # server itself is owned by a different admin.
                target_user_id=ctx.user_id,
                bulk=bulk,
            )
        if err:
            raise RuntimeError(err)

        return {
            "server_id": server.id,
            "provider": server.provider,
            "indexed": indexed,
        }

    return handler
