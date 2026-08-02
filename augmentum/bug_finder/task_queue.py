"""Per-run task queue — substrate for the Lead Agent.

Replaces the "plan once, scan every chunk" pipeline with a dynamic
queue. The orchestrator (or the lead agent) drains the queue,
dispatches subagents, and lets them ADD MORE tasks based on what
they find. The compound-over-investigation pattern that CC uses.

Mental model: each row is a unit of work the bug-finder will do.
The planner enqueues `detect` tasks (one per chunk it identifies).
Investigators (Phase 2) enqueue `investigate` tasks when a finding
suggests adjacent code is worth examining. The lead agent (Phase 3)
sequences everything by picking highest-priority pending tasks.

This module is the persistence + query layer. The lead's decision
loop lives in ``lead.py``; the investigator's branching logic in
``investigator.py``. Both will be added incrementally — this commit
gives them the substrate to build on without their own complexity.

Same shape as ``PatternStore`` / ``KnowledgeStore``: aiosqlite
connection in, async methods out, multi-tenant via user_id.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskKind(str, Enum):
    """What kind of work this task represents.

    Each kind has its own dispatcher and its own ``target_json`` shape:

    * ``DETECT`` — run the existing detector subagent on a chunk.
      target = ``{file, function, line_start, line_end, rationale,
      suspected_class}``.

    * ``INVESTIGATE`` — spawn an investigator to follow a thread
      (e.g. "same exception pattern as finding X — find other sites").
      target = ``{thread_anchor, scope_hint}``.

    * ``VERIFY`` — run the verifier on a finding to confirm or drop.
      target = ``{finding_id}``.

    * ``FIX`` — attempt a fix for a confirmed finding.
      target = ``{finding_id}``.

    * ``CRITIQUE`` — lead agent self-review of a finding before
      committing to verifier budget. target = ``{finding_id}``.

    * ``COMPREHEND_REFRESH`` — re-comprehend a subsystem when prior
      pass produced shallow output. target = ``{subsystem_path}``.
    """

    DETECT = "detect"
    INVESTIGATE = "investigate"
    VERIFY = "verify"
    FIX = "fix"
    CRITIQUE = "critique"
    COMPREHEND_REFRESH = "comprehend_refresh"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DROPPED = "dropped"          # lead decided this isn't worth pursuing
    FAILED = "failed"            # dispatcher errored out


# Lifecycle: PENDING → IN_PROGRESS → (COMPLETED | DROPPED | FAILED).
# Anything in IN_PROGRESS at run-resume time gets bounced back to
# PENDING so a restart doesn't strand work.
_TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED.value,
    TaskStatus.DROPPED.value,
    TaskStatus.FAILED.value,
})


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BugFinderTask:
    """One row from ``bug_finder_tasks``. Read-only."""

    task_id: str
    user_id: str
    run_id: str
    workspace_id: str
    kind: str
    target: dict[str, Any]       # decoded target_json
    reason: str
    priority: int                # 1 (low) - 10 (high)
    status: str
    parent_task_id: str
    created_by: str
    result_summary: str
    created_at: int
    completed_at: int

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    @property
    def is_pending(self) -> bool:
        return self.status == TaskStatus.PENDING.value


def _task_id(run_id: str, kind: str, target: dict[str, Any]) -> str:
    """Stable hash for idempotent enqueue.

    Two enqueues of the same (run_id, kind, target) collapse to one
    row. The lead agent + investigators can suggest the same task
    repeatedly without spamming the queue.
    """
    # Sort keys for deterministic JSON. Strip any non-essential fields
    # before hashing (priority, reason, parent_task_id) — those don't
    # define the identity of the task.
    canonical = json.dumps(target, sort_keys=True, default=str)
    blob = "|".join((run_id, kind, canonical))
    return "tsk_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _row_to_task(row: tuple) -> BugFinderTask:
    """Map the SELECT row order used by ``_FETCH_COLS``."""
    try:
        target = json.loads(row[5] or "{}")
        if not isinstance(target, dict):
            target = {"value": target}
    except (ValueError, TypeError):
        target = {}
    return BugFinderTask(
        task_id=str(row[0] or ""),
        user_id=str(row[1] or ""),
        run_id=str(row[2] or ""),
        workspace_id=str(row[3] or ""),
        kind=str(row[4] or ""),
        target=target,
        reason=str(row[6] or ""),
        priority=int(row[7] or 0),
        status=str(row[8] or TaskStatus.PENDING.value),
        parent_task_id=str(row[9] or ""),
        created_by=str(row[10] or ""),
        result_summary=str(row[11] or ""),
        created_at=int(row[12] or 0),
        completed_at=int(row[13] or 0),
    )


# Used for every SELECT — keep the SELECT clause + _row_to_task in lockstep.
_FETCH_COLS = (
    "task_id, user_id, run_id, workspace_id, kind, target_json, "
    "reason, priority, status, parent_task_id, created_by, "
    "result_summary, created_at, completed_at"
)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TaskQueue:
    """Read/write access to ``bug_finder_tasks`` scoped to one user.

    The same shape as ``PatternStore`` / ``KnowledgeStore`` — caller
    constructs once with an aiosqlite connection, passes user_id
    explicitly on every method call. The composite primary key
    (user_id, task_id) makes cross-user reads architecturally
    impossible even if user_id ever got dropped in a callsite.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ----- enqueue -----

    async def enqueue(
        self,
        *,
        run_id: str,
        kind: TaskKind | str,
        target: dict[str, Any],
        reason: str = "",
        priority: int = 5,
        user_id: str = "",
        workspace_id: str = "",
        parent_task_id: str = "",
        created_by: str = "planner",
    ) -> BugFinderTask:
        """Append a task to the queue.

        Idempotent: if a task with the same (run_id, kind, target)
        already exists, the existing row is returned unchanged. The
        priority / reason / parent_task_id of repeat enqueues are
        ignored — first writer wins. This is what makes "the
        investigator might suggest the same chunk twice" cheap.
        """
        kind_value = kind.value if isinstance(kind, TaskKind) else str(kind)
        tid = _task_id(run_id, kind_value, target)
        priority = max(1, min(10, int(priority)))
        now = int(time.time())

        async with self._conn.execute(
            f"SELECT {_FETCH_COLS} FROM bug_finder_tasks "
            f"WHERE user_id = ? AND task_id = ?",
            (user_id, tid),
        ) as cur:
            existing = await cur.fetchone()
        if existing is not None:
            return _row_to_task(existing)

        await self._conn.execute(
            "INSERT INTO bug_finder_tasks ("
            "task_id, user_id, run_id, workspace_id, kind, target_json, "
            "reason, priority, status, parent_task_id, created_by, "
            "result_summary, created_at, completed_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tid, user_id, run_id, workspace_id, kind_value,
                json.dumps(target, sort_keys=True, default=str),
                reason, priority, TaskStatus.PENDING.value,
                parent_task_id, created_by, "", now, 0,
            ),
        )
        await self._conn.commit()
        log.info(
            "bug_finder_task_enqueued",
            user_id=user_id, run_id=run_id, task_id=tid,
            kind=kind_value, priority=priority,
            created_by=created_by, parent_task_id=parent_task_id,
        )
        return BugFinderTask(
            task_id=tid, user_id=user_id, run_id=run_id,
            workspace_id=workspace_id, kind=kind_value, target=target,
            reason=reason, priority=priority,
            status=TaskStatus.PENDING.value,
            parent_task_id=parent_task_id, created_by=created_by,
            result_summary="", created_at=now, completed_at=0,
        )

    # ----- take -----

    async def take_next(
        self,
        *,
        run_id: str,
        user_id: str = "",
        kinds: tuple[TaskKind | str, ...] = (),
    ) -> BugFinderTask | None:
        """Return the highest-priority PENDING task, mark it IN_PROGRESS.

        Atomic via the index on (user_id, run_id, status, priority).
        Returns None when no pending tasks remain. Optional ``kinds``
        filter restricts which kinds are considered — useful for
        stage-bounded workers (the detector stage takes only
        DETECT tasks).
        """
        if kinds:
            kind_values = tuple(
                k.value if isinstance(k, TaskKind) else str(k)
                for k in kinds
            )
            placeholders = ",".join("?" for _ in kind_values)
            query = (
                f"SELECT {_FETCH_COLS} FROM bug_finder_tasks "
                f"WHERE user_id = ? AND run_id = ? "
                f"AND status = ? AND kind IN ({placeholders}) "
                f"ORDER BY priority DESC, created_at ASC LIMIT 1"
            )
            params: tuple = (user_id, run_id, TaskStatus.PENDING.value, *kind_values)
        else:
            query = (
                f"SELECT {_FETCH_COLS} FROM bug_finder_tasks "
                f"WHERE user_id = ? AND run_id = ? AND status = ? "
                f"ORDER BY priority DESC, created_at ASC LIMIT 1"
            )
            params = (user_id, run_id, TaskStatus.PENDING.value)

        async with self._conn.execute(query, params) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        task = _row_to_task(row)
        await self._conn.execute(
            "UPDATE bug_finder_tasks SET status = ? "
            "WHERE user_id = ? AND task_id = ? AND status = ?",
            (
                TaskStatus.IN_PROGRESS.value, user_id, task.task_id,
                TaskStatus.PENDING.value,
            ),
        )
        await self._conn.commit()
        return BugFinderTask(
            **{**task.__dict__, "status": TaskStatus.IN_PROGRESS.value},
        )

    # ----- terminal transitions -----

    async def mark_completed(
        self, task_id: str, *,
        user_id: str = "", result_summary: str = "",
    ) -> None:
        await self._terminal(task_id, TaskStatus.COMPLETED, user_id, result_summary)

    async def mark_dropped(
        self, task_id: str, *,
        user_id: str = "", reason: str = "",
    ) -> None:
        await self._terminal(task_id, TaskStatus.DROPPED, user_id, reason)

    async def mark_failed(
        self, task_id: str, *,
        user_id: str = "", reason: str = "",
    ) -> None:
        await self._terminal(task_id, TaskStatus.FAILED, user_id, reason)

    async def _terminal(
        self, task_id: str, status: TaskStatus,
        user_id: str, result_summary: str,
    ) -> None:
        now = int(time.time())
        await self._conn.execute(
            "UPDATE bug_finder_tasks SET "
            "status = ?, result_summary = ?, completed_at = ? "
            "WHERE user_id = ? AND task_id = ?",
            (status.value, result_summary or "", now, user_id, task_id),
        )
        await self._conn.commit()

    async def reset_in_progress(
        self, *, run_id: str, user_id: str = "",
    ) -> int:
        """Bounce IN_PROGRESS tasks back to PENDING — for run resume.

        When a container restart strands tasks mid-dispatch they'd
        otherwise be hidden from ``take_next`` forever. The orchestrator
        calls this at run-resume time so the next ``take_next`` picks
        up the previously-in-flight work. Returns the count bumped.
        """
        async with self._conn.execute(
            "UPDATE bug_finder_tasks SET status = ? "
            "WHERE user_id = ? AND run_id = ? AND status = ?",
            (
                TaskStatus.PENDING.value, user_id, run_id,
                TaskStatus.IN_PROGRESS.value,
            ),
        ) as cur:
            n = cur.rowcount or 0
        await self._conn.commit()
        return n

    # ----- read queries -----

    async def list_tasks(
        self,
        *,
        run_id: str,
        user_id: str = "",
        status: TaskStatus | str | None = None,
        kinds: tuple[TaskKind | str, ...] = (),
    ) -> list[BugFinderTask]:
        """All tasks for one run, optionally filtered by status / kind."""
        where = ["user_id = ?", "run_id = ?"]
        params: list[Any] = [user_id, run_id]
        if status is not None:
            where.append("status = ?")
            params.append(
                status.value if isinstance(status, TaskStatus) else str(status),
            )
        if kinds:
            kind_values = tuple(
                k.value if isinstance(k, TaskKind) else str(k)
                for k in kinds
            )
            placeholders = ",".join("?" for _ in kind_values)
            where.append(f"kind IN ({placeholders})")
            params.extend(kind_values)
        query = (
            f"SELECT {_FETCH_COLS} FROM bug_finder_tasks "
            f"WHERE {' AND '.join(where)} "
            f"ORDER BY priority DESC, created_at ASC"
        )
        async with self._conn.execute(query, tuple(params)) as cur:
            rows = await cur.fetchall()
        return [_row_to_task(r) for r in rows]

    async def counts_by_status(
        self, *, run_id: str, user_id: str = "",
    ) -> dict[str, int]:
        async with self._conn.execute(
            "SELECT status, COUNT(*) FROM bug_finder_tasks "
            "WHERE user_id = ? AND run_id = ? GROUP BY status",
            (user_id, run_id),
        ) as cur:
            rows = await cur.fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    async def pending_count(
        self, *, run_id: str, user_id: str = "",
        kinds: tuple[TaskKind | str, ...] = (),
    ) -> int:
        counts = await self.counts_by_status(run_id=run_id, user_id=user_id)
        if not kinds:
            return counts.get(TaskStatus.PENDING.value, 0)
        # Fallback to explicit list for kind-filtered counts
        rows = await self.list_tasks(
            run_id=run_id, user_id=user_id,
            status=TaskStatus.PENDING, kinds=kinds,
        )
        return len(rows)


# ---------------------------------------------------------------------------
# Prompt-friendly summary renderer (for the lead agent)
# ---------------------------------------------------------------------------


def render_queue_summary(tasks: list[BugFinderTask], *, max_lines: int = 60) -> str:
    """Compact text view of the queue for injection into the lead's
    system prompt. The lead reads this to decide what to dispatch
    next.

    Format prioritizes the pending tasks (what the lead can act on)
    over the completed ones (informational), but includes both so the
    lead sees what's been tried.
    """
    if not tasks:
        return "(queue empty)"

    by_status: dict[str, list[BugFinderTask]] = {}
    for t in tasks:
        by_status.setdefault(t.status, []).append(t)

    lines: list[str] = ["## Task queue"]
    for status_label, status_value in (
        ("PENDING", TaskStatus.PENDING.value),
        ("IN PROGRESS", TaskStatus.IN_PROGRESS.value),
        ("COMPLETED", TaskStatus.COMPLETED.value),
        ("DROPPED", TaskStatus.DROPPED.value),
        ("FAILED", TaskStatus.FAILED.value),
    ):
        items = by_status.get(status_value) or []
        if not items:
            continue
        # Pending shows full detail; everything else is collapsed.
        lines.append(f"\n### {status_label} ({len(items)})")
        if status_value == TaskStatus.PENDING.value:
            for t in items[:max_lines]:
                anchor = (
                    t.target.get("file") or t.target.get("finding_id")
                    or t.target.get("thread_anchor") or "?"
                )
                # Render the actual task_id verbatim so the lead can
                # copy it into a dispatch action. Without this the model
                # hallucinates ids and every dispatch fails with
                # "unknown task" (observed in run bfr_d1c9e7eca09f —
                # 20 iters all wasted on guessed-up ids like
                # `flow_routes_detect`).
                lines.append(
                    f"- `{t.task_id}` [p{t.priority}] **{t.kind}** "
                    f"`{anchor}` — {t.reason or '(no reason)'}"
                )
            if len(items) > max_lines:
                lines.append(f"  ... and {len(items) - max_lines} more")
        else:
            # Compact: just count by kind
            kinds: dict[str, int] = {}
            for t in items:
                kinds[t.kind] = kinds.get(t.kind, 0) + 1
            parts = [f"{n} {k}" for k, n in sorted(kinds.items())]
            lines.append(f"  {' · '.join(parts)}")
    return "\n".join(lines)
