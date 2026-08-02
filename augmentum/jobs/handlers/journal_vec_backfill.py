"""``journal_vec_backfill`` job handler.

One-shot backfill of ``companion_journal_vec`` for journal rows that
exist from before migration 177 (or any row whose vec mirror failed
to write at journal time — best-effort writes can silently miss).

Design for non-stalling — this is the load-bearing piece:

* **Batches of 50** so the connection isn't held for thousands of
  rows at a time.
* **``asyncio.sleep(50ms)`` between batches** yields the event loop
  to ticks, voice, chat, and every other surface running on the
  same aiosqlite connection.
* **Restart-survivable.** If the worker dies mid-walk the next run
  picks up at the highest ``journal_id`` already mirrored — vec
  mirrors use ``journal_id`` as PRIMARY KEY so re-inserts are caught
  by SQLite as duplicates and we skip cleanly.
* **No model load on first iteration.** Journal rows already have
  ``embedding`` BLOBs from when they were written (memory.py
  computes them eagerly with ``embed=True`` default). Backfill just
  copies the existing bytes into the vec0 table — no embedding
  recompute needed for the happy path. The slow path (rows with
  ``embedding IS NULL`` — early entries that failed to embed) is
  handled by ``enrich_pending``, not here.

Payload shape: empty dict — the job walks all rows globally. There's
exactly one companion currently and this is a one-time migration,
so per-companion sharding would be over-engineering.

Returns ``{"status": "ok", "rows_mirrored": int}``. Idempotent so
re-runs on a fully-mirrored DB return ``rows_mirrored=0``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from augmentum.jobs.context import JobContext
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_BATCH_SIZE = 50
_INTER_BATCH_SLEEP_S = 0.05  # 50ms — yield to ticks + chat


def make_journal_vec_backfill_handler(app):
    """Bind the handler to ``app.state`` services.

    Returned coroutine matches the JobRunner dispatch signature.
    """

    async def handler(ctx: JobContext) -> dict[str, Any]:
        backend = getattr(app.state, "backend", None)
        if backend is None or backend.conn is None:
            return {"status": "skipped", "reason": "backend unavailable"}

        # Quick pre-check: does the vec table exist? If sqlite-vec
        # isn't loaded, the table doesn't exist and this whole job
        # is a no-op.
        try:
            cur = await backend.conn.execute(
                "SELECT 1 FROM companion_journal_vec LIMIT 1"
            )
            await cur.fetchone()
            await cur.close()
        except Exception as exc:
            log.info(
                "journal_vec_backfill_skipped_no_vec",
                error=str(exc)[:200],
            )
            return {"status": "skipped", "reason": "vec table unavailable"}

        rows_mirrored = 0
        last_id = 0  # walk forward by id ascending — stable + restart-friendly

        while True:
            # Select rows that have an embedding but no vec mirror.
            # The NOT EXISTS subquery is the idempotency mechanism —
            # already-mirrored rows naturally drop out.
            cur = await backend.conn.execute(
                """
                SELECT j.id, j.embedding
                FROM companion_journal j
                WHERE j.id > ?
                  AND j.embedding IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM companion_journal_vec v
                      WHERE v.journal_id = j.id
                  )
                ORDER BY j.id ASC
                LIMIT ?
                """,
                (last_id, _BATCH_SIZE),
            )
            rows = await cur.fetchall()
            await cur.close()

            if not rows:
                break

            for jid, emb_blob in rows:
                try:
                    await backend.conn.execute(
                        "INSERT INTO companion_journal_vec(journal_id, embedding) "
                        "VALUES (?, ?)",
                        (jid, emb_blob),
                    )
                    rows_mirrored += 1
                except Exception as exc:
                    # A single bad row (e.g. wrong-dim embedding from a
                    # legacy schema) shouldn't poison the whole walk.
                    log.warning(
                        "journal_vec_backfill_row_failed",
                        journal_id=jid, error=str(exc)[:200],
                    )
                last_id = jid

            await backend.conn.commit()
            # Yield the loop so the tick + chat surfaces don't see
            # the connection held for the full walk. The sleep is
            # short but it's the difference between "fine" and
            # "noticeable" on a hot system.
            await asyncio.sleep(_INTER_BATCH_SLEEP_S)

        log.info(
            "journal_vec_backfill_done",
            rows_mirrored=rows_mirrored,
        )
        return {"status": "ok", "rows_mirrored": rows_mirrored}

    return handler
