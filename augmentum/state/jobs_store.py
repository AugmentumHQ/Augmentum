"""Background-jobs persistence layer.

Thin async wrapper around the ``background_jobs`` table. All CRUD is
user-scoped except for the dispatch/crash-recovery helpers, which the
runner calls without a user context (they operate across all users).

See ``augmentum/state/migrations/102_background_jobs.sql`` for schema
and the field-level design notes.

Progress + cancellation runtime state
-------------------------------------
Long-running handlers (``gguf_download`` is the canonical example)
emit progress every few hundred milliseconds. Naively persisting each
one was producing 6-9 commits/sec on the shared aiosqlite connection
during multi-download sessions, which queued every other coroutine
(chat, auth, files) behind the writer lock and surfaced as 19s
event-loop stalls in the audit logs. Every per-chunk ``check_cancel``
also added a SELECT.

The store now keeps a tiny in-memory overlay per running job:

* ``_progress_cache[job_id]`` is the freshest known (progress, stage,
  updated_at). ``update_progress`` writes it on every call and is the
  source of truth for UI reads (``get`` / ``list_for_user`` merge it
  onto the DB row). The DB write is throttled to once every
  ``_PROGRESS_DB_THROTTLE_S`` seconds, plus on stage-class changes and
  on completion (progress >= 1.0). The DB only persists what crash
  recovery actually needs — the live UI sees fresh numbers from cache.

* ``_cancel_events[job_id]`` is an asyncio.Event flipped by
  ``request_cancel``. ``is_cancel_requested`` checks it before falling
  back to a DB SELECT, so per-chunk cancel checks cost a dict lookup
  instead of a query.

Both dicts are entered by ``claim_next_pending`` (single ingress for
"a job is running") and cleaned up by the terminal mark methods. Safe
across retries: ``setdefault`` semantics make registration idempotent
so a retried job reuses any existing event the cancel API created.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Terminal states — a job in any of these won't be picked up by the worker
# and can be cleaned up or archived.
_TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# How often update_progress is allowed to actually hit the DB. Below
# this interval the call updates only the in-memory cache. Five seconds
# is enough for crash recovery to be useful (you lose at most that much
# progress on an unclean shutdown) but rare enough to drop write
# amplification by an order of magnitude during active downloads.
_PROGRESS_DB_THROTTLE_S: float = 5.0


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    d: dict[str, Any] = dict(zip(cols, row))
    # Inflate JSON fields the caller expects as objects.
    for key in ("payload", "result"):
        val = d.get(key)
        if isinstance(val, str) and val:
            try:
                d[key] = json.loads(val)
            except json.JSONDecodeError:
                # Leave as-is; handler can surface a parse error if needed.
                pass
        elif val is None and key == "payload":
            d[key] = {}
    # cancel_requested is stored as 0/1; expose as bool for callers.
    d["cancel_requested"] = bool(d.get("cancel_requested", 0))
    return d


class JobsStore:
    """CRUD for the ``background_jobs`` table."""

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        read_conn: aiosqlite.Connection | None = None,
    ) -> None:
        self._conn = conn
        # Optional second connection routed at hot read paths (the
        # claim-loop's "is there a pending job" SELECT runs on every
        # tick of the runner and contends with media-progress UPDATE
        # and other writers on the main connection's worker thread).
        # Falls back to the main conn so unit tests and bare init
        # keep working.
        self._read_conn = read_conn or conn
        # Per-job runtime overlay. See module docstring.
        self._progress_cache: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        # Active-jobs registry — populated on claim, dropped on
        # terminal status. Lets callers (the resource ledger, in
        # particular) enumerate "what's running right now" without
        # a DB query. Keys: job_id; values: the dict returned by
        # ``claim_next_pending``, kept fresh by the same progress
        # overlay as ``get()``.
        self._active: dict[str, dict[str, Any]] = {}

    def attach_read_conn(self, read_conn: aiosqlite.Connection | None) -> None:
        """Late-bind the read connection.

        ``read_conn`` is opened later in the proxy lifespan than the
        JobsStore is constructed, so callers that want claim-loop
        SELECTs off the main writer thread wire it in after server
        startup finishes. Falls back to the main conn when passed
        None or when the caller never invokes this — keeps tests +
        bare-init paths working.
        """
        self._read_conn = read_conn or self._conn

    # ── Runtime registry (in-memory, no DB) ────────────────────────────

    def list_active(self) -> list[dict[str, Any]]:
        """Snapshot of currently-running jobs, no DB hit.

        Returns a list of job dicts with the latest progress overlay
        applied — same shape as ``get()`` would return. Order is
        insertion order (Python dict guarantee since 3.7); for a
        deterministic UI order callers can sort by ``started_at``.

        Empty when no jobs are running, which is the steady state.
        """
        out: list[dict[str, Any]] = []
        for job_id, job in list(self._active.items()):
            row = dict(job)  # don't mutate the registry entry
            out.append(self._apply_progress_overlay(row))
        return out

    def _ensure_runtime(self, job_id: str) -> None:
        """Create the cancel event for a running job. Idempotent so
        retries (which re-enter ``claim_next_pending``) reuse any event
        a cancel API call may have already populated."""
        if job_id not in self._cancel_events:
            self._cancel_events[job_id] = asyncio.Event()

    def _purge_runtime(self, job_id: str) -> None:
        """Drop in-memory state when a job reaches a terminal status.

        Called from mark_completed / mark_cancelled / mark_failed
        (terminal branch). Safe to call when no entry exists.
        """
        self._cancel_events.pop(job_id, None)
        self._progress_cache.pop(job_id, None)
        self._active.pop(job_id, None)

    # ── Creation ──────────────────────────────────────────────────────

    async def create(
        self,
        *,
        user_id: str,
        job_type: str,
        payload: dict | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> str:
        """Enqueue a new job. Returns the new job_id.

        ``user_id`` is required — there's no legitimate "system-owned"
        background job in Augmentum (every job belongs to the user who
        asked for it, even if some other flow triggers it on their behalf).
        """
        if not user_id:
            raise ValueError("jobs_store.create requires user_id")
        if not job_type:
            raise ValueError("jobs_store.create requires job_type")

        job_id = uuid.uuid4().hex[:16]
        now = int(time.time())
        await self._conn.execute(
            """INSERT INTO background_jobs
               (id, user_id, job_type, payload, priority, max_attempts,
                status, progress, stage, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0.0, '', ?, ?)""",
            (
                job_id,
                user_id,
                job_type,
                json.dumps(payload or {}),
                int(priority),
                int(max_attempts),
                now,
                now,
            ),
        )
        await self._conn.commit()
        return job_id

    # ── Reads ─────────────────────────────────────────────────────────

    def _apply_progress_overlay(self, row: dict[str, Any]) -> dict[str, Any]:
        """Replace progress/stage/updated_at with the in-memory snapshot
        if one exists. The cache is always at least as fresh as the DB
        because ``update_progress`` writes it on every call while the
        DB write is throttled."""
        cached = self._progress_cache.get(row.get("id", ""))
        if cached:
            row["progress"] = cached["progress"]
            row["stage"] = cached["stage"]
            row["updated_at"] = cached["updated_at"]
        return row

    async def get(self, job_id: str, *, user_id: str = "") -> dict | None:
        query = "SELECT * FROM background_jobs WHERE id = ?"
        params: list = [job_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        return self._apply_progress_overlay(_row_to_dict(cursor, row))

    async def list_for_user(
        self,
        *,
        user_id: str,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List a user's jobs, newest first. Filters are optional."""
        query = "SELECT * FROM background_jobs WHERE user_id = ?"
        params: list = [user_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        if job_type:
            query += " AND job_type = ?"
            params.append(job_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._apply_progress_overlay(_row_to_dict(cursor, r)) for r in rows]

    # ── Dispatch (runner-facing, not user-scoped) ─────────────────────

    async def claim_next_pending(self) -> dict | None:
        """Pick the next job to run and flip it to ``status='running'``.

        Highest priority wins; oldest first at the same priority.

        The SELECT runs on the read connection (when available) so a
        loaded main-connection writer queue doesn't stall the runner
        loop. The UPDATE-to-claim runs on the main connection — same
        aiosqlite worker thread that owns every other write, so the
        atomicity guarantee remains: between our SELECT and our UPDATE
        no other writer can have committed against this row because we
        gate the UPDATE on ``status = 'pending'``. If a parallel claimer
        somehow won the race (it can't in current code — single in-
        process runner — but defensive against future fan-out), rowcount
        comes back 0 and we just return None and try again next tick.

        Returns None when the queue is empty.
        """
        cursor = await self._read_conn.execute(
            """SELECT * FROM background_jobs
               WHERE status = 'pending' AND cancel_requested = 0
               ORDER BY priority DESC, created_at ASC
               LIMIT 1""",
        )
        row = await cursor.fetchone()
        if not row:
            return None
        job = _row_to_dict(cursor, row)

        now = int(time.time())
        await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'running',
                      started_at = COALESCE(started_at, ?),
                      attempts = attempts + 1,
                      updated_at = ?
                WHERE id = ? AND status = 'pending'""",
            (now, now, job["id"]),
        )
        await self._conn.commit()
        job["status"] = "running"
        job["started_at"] = job.get("started_at") or now
        job["attempts"] = int(job.get("attempts") or 0) + 1
        # Single ingress point for "this job is now running" — register
        # the cancel event so subsequent is_cancel_requested calls
        # short-circuit on a dict lookup instead of querying the DB.
        self._ensure_runtime(job["id"])
        # Active-jobs registry — pure RAM, surfaced by list_active().
        self._active[job["id"]] = job
        return job

    async def requeue_crashed(self) -> int:
        """Reset rows stuck in ``running`` from a previous boot.

        Called once during server startup. Jobs are re-queued if they
        haven't exhausted ``max_attempts``; otherwise they're marked
        failed so a user never stares at a spinner that will never
        resolve. Returns the number of rows re-queued (not the count
        marked failed).
        """
        now = int(time.time())
        # Requeue anything that still has retries available.
        cursor = await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'pending',
                      stage = 'Recovered after restart',
                      updated_at = ?
                WHERE status = 'running' AND attempts < max_attempts""",
            (now,),
        )
        requeued = cursor.rowcount or 0
        # Hard-fail the rest. If a job crashed us max_attempts times, the
        # handler is broken — human needs to intervene.
        await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'failed',
                      error = COALESCE(error, 'Exceeded max_attempts after restart'),
                      completed_at = ?,
                      updated_at = ?
                WHERE status = 'running'""",
            (now, now),
        )
        await self._conn.commit()
        if requeued:
            log.info("jobs_requeued_after_restart", count=requeued)
        return requeued

    # ── Progress / lifecycle updates ──────────────────────────────────

    async def update_progress(
        self, job_id: str, *, progress: float, stage: str = "",
        force: bool = False,
    ) -> None:
        """Record incremental progress (0.0-1.0) and an optional stage label.

        Cache-first: every call updates the in-memory overlay so UI
        reads of ``get(job_id)`` see fresh data immediately. The DB
        write is throttled to once every ``_PROGRESS_DB_THROTTLE_S``
        unless one of these forces a flush:
          * ``force=True`` — caller explicitly wants persistence now
            (e.g., right before a long blocking operation that may
            outlive the next throttle window).
          * ``progress >= 1.0`` — handler is finishing; the next mark_*
            call will overwrite anyway, but we want the row to reflect
            completion-imminent state in case the handler is killed
            between this call and mark_completed.
          * ``stage_class`` change — the first whitespace-separated
            token of ``stage`` differs from the cached value (e.g.,
            ``downloading`` → ``finalizing``). Captures meaningful
            transitions even within the throttle window.

        Per-byte-count stage updates (``downloading 5 MB`` → ``downloading
        6 MB``) share a stage class and don't trigger writes.
        """
        progress = max(0.0, min(1.0, float(progress)))
        now_mono = time.monotonic()
        now_wall = int(time.time())
        cached = self._progress_cache.get(job_id)

        # First whitespace-separated token. Empty string when stage is
        # empty — both sides compare equal then.
        new_class = stage.split(maxsplit=1)[0] if stage else ""
        old_stage = cached.get("stage", "") if cached else ""
        old_class = old_stage.split(maxsplit=1)[0] if old_stage else ""
        prior_flushed_at = cached.get("flushed_at", 0.0) if cached else 0.0

        write_db = (
            force
            or progress >= 1.0
            or cached is None
            or new_class != old_class
            or (now_mono - prior_flushed_at) >= _PROGRESS_DB_THROTTLE_S
        )

        # Cache always advances, even when we skip the DB. flushed_at
        # only advances when the DB actually got written so the next
        # call's throttle window measures from the last on-disk state.
        self._progress_cache[job_id] = {
            "progress": progress,
            "stage": stage,
            "updated_at": now_wall,
            "flushed_at": now_mono if write_db else prior_flushed_at,
        }

        if not write_db:
            return

        if stage:
            await self._conn.execute(
                """UPDATE background_jobs
                      SET progress = ?, stage = ?, updated_at = ?
                    WHERE id = ?""",
                (progress, stage, now_wall, job_id),
            )
        else:
            await self._conn.execute(
                """UPDATE background_jobs
                      SET progress = ?, updated_at = ?
                    WHERE id = ?""",
                (progress, now_wall, job_id),
            )
        await self._conn.commit()

    async def mark_completed(
        self, job_id: str, *, result: dict | None = None,
    ) -> None:
        now = int(time.time())
        await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'completed',
                      progress = 1.0,
                      result = ?,
                      completed_at = ?,
                      updated_at = ?
                WHERE id = ?""",
            (json.dumps(result) if result is not None else None, now, now, job_id),
        )
        await self._conn.commit()
        self._purge_runtime(job_id)

    async def mark_failed(
        self, job_id: str, *, error: str, retryable: bool = False,
    ) -> None:
        """Mark a job failed.

        When ``retryable=True`` and the job still has attempts left, the
        status reverts to ``pending`` so the runner picks it up again on
        the next tick. When retries are exhausted (or ``retryable=False``)
        the job terminates in the ``failed`` state.
        """
        now = int(time.time())
        if retryable:
            # Check attempts vs max before reverting.
            cur = await self._conn.execute(
                "SELECT attempts, max_attempts FROM background_jobs WHERE id = ?",
                (job_id,),
            )
            row = await cur.fetchone()
            if row and int(row[0]) < int(row[1]):
                await self._conn.execute(
                    """UPDATE background_jobs
                          SET status = 'pending',
                              error = ?,
                              updated_at = ?
                        WHERE id = ?""",
                    (error, now, job_id),
                )
                await self._conn.commit()
                return
        # Terminal failure.
        await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'failed',
                      error = ?,
                      completed_at = ?,
                      updated_at = ?
                WHERE id = ?""",
            (error, now, now, job_id),
        )
        await self._conn.commit()
        self._purge_runtime(job_id)

    async def mark_cancelled(self, job_id: str) -> None:
        now = int(time.time())
        await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'cancelled',
                      completed_at = ?,
                      updated_at = ?
                WHERE id = ?""",
            (now, now, job_id),
        )
        await self._conn.commit()
        self._purge_runtime(job_id)

    # ── Cancellation signalling ───────────────────────────────────────

    async def request_cancel(self, job_id: str, *, user_id: str) -> bool:
        """Flip the cancel flag. Returns False if the job is already terminal.

        Pending jobs are cancelled immediately (worker won't dispatch them).
        Running jobs flip the flag so the handler can observe it.
        """
        cursor = await self._conn.execute(
            "SELECT status FROM background_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        status = row[0]
        if status in _TERMINAL_STATUSES:
            return False

        now = int(time.time())
        if status == "pending":
            # No worker to observe the flag — cancel immediately.
            # claim_next_pending filters cancel_requested=0, so this row
            # won't be picked up.
            await self._conn.execute(
                """UPDATE background_jobs
                      SET status = 'cancelled',
                          cancel_requested = 1,
                          completed_at = ?,
                          updated_at = ?
                    WHERE id = ?""",
                (now, now, job_id),
            )
            await self._conn.commit()
            self._purge_runtime(job_id)
        else:
            # Running — flip the in-memory event INSTANTLY so the
            # handler's next ``check_cancel`` sees the cancel without
            # waiting on a DB read. ``setdefault`` covers the rare race
            # where the cancel arrives between claim_next_pending's
            # commit and ``_ensure_runtime``'s registration.
            event = self._cancel_events.setdefault(job_id, asyncio.Event())
            event.set()
            await self._conn.execute(
                """UPDATE background_jobs
                      SET cancel_requested = 1,
                          updated_at = ?
                    WHERE id = ?""",
                (now, job_id),
            )
            await self._conn.commit()
        return True

    async def is_cancel_requested(self, job_id: str) -> bool:
        """True if the job's cancel signal is set.

        Fast path is a dict lookup + an event check — no DB I/O. Falls
        back to the DB only when the in-memory event is missing, which
        happens for cold-lookup paths (a job started before the runtime
        registry existed, or a future caller bypassing claim_next_pending).
        """
        event = self._cancel_events.get(job_id)
        if event is not None:
            return event.is_set()
        cursor = await self._conn.execute(
            "SELECT cancel_requested FROM background_jobs WHERE id = ?",
            (job_id,),
        )
        row = await cursor.fetchone()
        return bool(row and row[0])

    # ── Cleanup ───────────────────────────────────────────────────────

    async def reset_for_retry(self, job_id: str, *, user_id: str) -> bool:
        """Reset a failed/cancelled job so it can run again from the same row."""
        now = int(time.time())
        cursor = await self._conn.execute(
            """UPDATE background_jobs
                  SET status = 'pending',
                      progress = 0.0,
                      stage = '',
                      result = NULL,
                      error = NULL,
                      attempts = 0,
                      cancel_requested = 0,
                      started_at = NULL,
                      completed_at = NULL,
                      updated_at = ?
                WHERE id = ?
                  AND user_id = ?
                  AND status IN ('failed', 'cancelled')""",
            (now, job_id, user_id),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)

    async def delete_job(self, job_id: str, *, user_id: str) -> bool:
        """Delete a user's job row."""
        cursor = await self._conn.execute(
            "DELETE FROM background_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        )
        await self._conn.commit()
        return bool(cursor.rowcount)

    async def delete_older_than(
        self, *, seconds: int, statuses: tuple[str, ...] = ("completed", "cancelled"),
    ) -> int:
        """Prune terminal rows older than a cutoff. Defaults keep failures.

        Useful for periodic maintenance — retain errors for debugging
        but drop the noise of old successes. Returns the row count.
        """
        cutoff = int(time.time()) - int(seconds)
        placeholders = ",".join(["?"] * len(statuses))
        cursor = await self._conn.execute(
            f"""DELETE FROM background_jobs
                 WHERE status IN ({placeholders})
                   AND completed_at IS NOT NULL
                   AND completed_at < ?""",
            (*statuses, cutoff),
        )
        await self._conn.commit()
        return cursor.rowcount or 0
