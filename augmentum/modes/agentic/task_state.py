"""Persistent task state for agentic mode (checkpoints in SQLite)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def hash_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    """Deterministic hash for a tool call keyed on (name, canonical args).

    Canonical JSON (sorted keys, default=str for non-serializable values)
    so equivalent calls produce matching hashes across process restarts.
    """
    canonical = json.dumps(args or {}, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(f"{tool_name}\x00{canonical}".encode("utf-8")).hexdigest()


class TaskStatus(str, Enum):
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    APPROVAL_PENDING = "approval_pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """In-memory representation of an agentic task."""

    id: str = ""
    session_id: str = ""
    user_id: str = ""
    flow_id: str = ""
    status: TaskStatus = TaskStatus.PLANNING
    autonomy_level: int = 2
    title: str = ""
    plan_md: str = ""
    current_step: int = 0
    total_steps: int = 0
    step_outputs: dict[int, str] = field(default_factory=dict)
    original_query: str = ""
    tool_calls_made: int = 0
    error: str | None = None
    # Per-slide image candidate pool produced by the Illustrate Slides step.
    # Keyed by 1-based slide index. Each entry is a list of candidate dicts
    # ({candidate_id, query, description, embed_url, thumb_url, source,
    # title}). Populated by ``_execute_illustrate_step`` and grown by the
    # /api/agentic/tasks/{id}/expand endpoint.
    image_candidates: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    # Per-slide user selection — {slide_index: {"primary": candidate_id,
    # "additional": [candidate_id, ...]}}. Tracks the picker's state so
    # re-renders project the same choices after a server restart.
    slide_image_picks: dict[int, dict[str, Any]] = field(default_factory=dict)
    # In-memory only (never persisted). Each entry: {"name", "role"?, "tools"?}.
    # Set by the executor when a flow / chain plan is materialised, then
    # consumed by ``meta_envelope`` to compute the per-step phase array
    # the inspector relies on.
    pipeline: list[dict[str, Any]] = field(default_factory=list)

    def advance_step(self) -> None:
        """Mark current step complete and move to next."""
        self.current_step += 1

    def record_step_output(self, step_index: int, output: str) -> None:
        """Record the output of a completed step."""
        self.step_outputs[step_index] = output

    @property
    def is_complete(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def progress_pct(self) -> float:
        if self.status == TaskStatus.COMPLETED:
            return 100.0
        if self.total_steps == 0:
            return 0.0
        # Use completed step count (steps with recorded output)
        completed = len(self.step_outputs)
        return min(99.0, (completed / self.total_steps) * 100)

    def meta_envelope(
        self,
        *,
        status_override: str | None = None,
        active_step_index: int | None = None,
        active_step_status: str = "running",
    ) -> dict[str, Any]:
        """Canonical state snapshot for inspector stream chunks.

        Every emission helper (the meta heartbeat, per-step events, autonomy
        prompts, chain sub-events) builds its augmentum payload by calling
        this then merging sub-event keys on top. That guarantees the
        inspector receives a consistent snapshot on EVERY chunk — fields
        like ``task_status`` and ``phases`` never silently drop, so the
        renderer never has to guess whether an absent field means "no
        change" or "reset to default".

        ``status_override`` lets callers emit display-only statuses
        (``planning``, ``plan_ready``, ``resuming``, ``cancelled``) that
        aren't in :class:`TaskStatus`. ``active_step_index`` /
        ``active_step_status`` describe which pipeline step is currently
        executing (defaults to ``self.current_step`` running).
        """
        status_value = status_override or (
            self.status.value if hasattr(self.status, "value") else str(self.status)
        )
        idx = self.current_step if active_step_index is None else active_step_index
        envelope: dict[str, Any] = {
            "mode": "agentic",
            "task_id": self.id,
            "task_status": status_value,
            "task_title": self.title,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress": self.progress_pct,
            "plan_md": self.plan_md,
            "autonomy_level": self.autonomy_level,
            # flow_id lets the inspector pick a per-flow renderer (storybook,
            # app builder, doc, …). Omitted for ad-hoc runs where the field
            # is empty so the generic renderer handles them.
            "flow_id": self.flow_id,
        }
        # Phases require a known pipeline. Omit the key entirely (rather
        # than send []) so the renderer's "no change" branch handles
        # pre-pipeline events instead of tearing down the step list.
        if self.pipeline:
            envelope["phases"] = self._compute_phases(
                status_value, idx, active_step_status,
            )
        return envelope

    def _compute_phases(
        self,
        status_value: str,
        active_idx: int,
        active_status: str,
    ) -> list[dict[str, Any]]:
        """Project ``self.pipeline`` into a status-bearing phases array."""
        phases: list[dict[str, Any]] = []
        for i, p in enumerate(self.pipeline):
            base = dict(p)
            if status_value == "completed":
                base["status"] = "complete"
            elif status_value == "failed" and i == active_idx:
                base["status"] = "failed"
            elif i < active_idx:
                base["status"] = "complete"
            elif i == active_idx:
                base["status"] = active_status
            else:
                base["status"] = "pending"
            phases.append(base)
        return phases


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class TaskStore:
    """Persists agentic task state in SQLite for checkpoint/resume."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create(self, state: TaskState, *, user_id: str = "") -> TaskState:
        """Create a new task record.

        ``user_id`` is taken from the argument when provided, otherwise from
        ``state.user_id``. One or the other must be non-empty — anonymous
        tasks are not accepted now that the table is tenant-scoped.
        """
        if not state.id:
            state.id = _new_id()
        if user_id:
            state.user_id = user_id
        if not state.user_id:
            raise ValueError("agentic_tasks insert requires user_id")

        await self._db.execute(
            """INSERT INTO agentic_tasks
               (id, session_id, user_id, flow_id, status, autonomy_level, title,
                plan_md, current_step, total_steps, step_outputs,
                original_query, tool_calls_made)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.id,
                state.session_id,
                state.user_id,
                state.flow_id,
                state.status.value,
                state.autonomy_level,
                state.title,
                state.plan_md,
                state.current_step,
                state.total_steps,
                json.dumps(state.step_outputs),
                state.original_query,
                state.tool_calls_made,
            ),
        )
        await self._db.commit()
        log.info("task_created", task_id=state.id, title=state.title)
        return state

    async def update(self, state: TaskState, *, user_id: str = "") -> None:
        """Update an existing task record (checkpoint).

        ``user_id`` gates the update — a mismatched tenant silently writes
        nothing (rowcount 0) rather than raising, matching the read-side
        filter behaviour.
        """
        uid = user_id or state.user_id
        await self._db.execute(
            """UPDATE agentic_tasks SET
               status = ?, plan_md = ?, current_step = ?, total_steps = ?,
               step_outputs = ?, tool_calls_made = ?,
               updated_at = datetime('now'),
               completed_at = CASE WHEN ? IN ('completed', 'failed')
                              THEN datetime('now') ELSE completed_at END,
               error = ?
               WHERE id = ? AND user_id = ?""",
            (
                state.status.value,
                state.plan_md,
                state.current_step,
                state.total_steps,
                json.dumps(state.step_outputs),
                state.tool_calls_made,
                state.status.value,
                state.error,
                state.id,
                uid,
            ),
        )
        await self._db.commit()

    async def get(self, task_id: str, *, user_id: str = "") -> TaskState | None:
        """Load a task by ID, filtered to ``user_id``."""
        cursor = await self._db.execute(
            "SELECT * FROM agentic_tasks WHERE id = ? AND user_id = ?",
            (task_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_state(cursor, row)

    async def get_incomplete_for_session(
        self, session_id: str, *, user_id: str = "",
    ) -> TaskState | None:
        """Find the most recent incomplete task for a (user, session)."""
        cursor = await self._db.execute(
            """SELECT * FROM agentic_tasks
               WHERE session_id = ? AND user_id = ?
               AND status NOT IN ('completed', 'failed')
               ORDER BY updated_at DESC LIMIT 1""",
            (session_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_state(cursor, row)

    async def list_for_session(
        self, session_id: str, *, user_id: str = "",
    ) -> list[TaskState]:
        """List all tasks for a (user, session)."""
        cursor = await self._db.execute(
            """SELECT * FROM agentic_tasks
               WHERE session_id = ? AND user_id = ?
               ORDER BY created_at DESC""",
            (session_id, user_id),
        )
        rows = await cursor.fetchall()
        return [self._row_to_state(cursor, r) for r in rows]

    def _row_to_state(self, cursor, row) -> TaskState:
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row))
        step_outputs = json.loads(d.get("step_outputs", "{}"))
        # Keys come back as strings from JSON — convert to int
        step_outputs = {int(k): v for k, v in step_outputs.items()}
        image_candidates = _load_int_keyed(d.get("image_candidates"))
        slide_image_picks = _load_int_keyed(d.get("slide_image_picks"))
        return TaskState(
            id=d["id"],
            session_id=d["session_id"],
            user_id=d.get("user_id") or "",
            flow_id=d.get("flow_id", ""),
            status=TaskStatus(d["status"]),
            autonomy_level=d.get("autonomy_level", 2),
            title=d.get("title", ""),
            plan_md=d.get("plan_md", ""),
            current_step=d.get("current_step", 0),
            total_steps=d.get("total_steps", 0),
            step_outputs=step_outputs,
            original_query=d.get("original_query", ""),
            tool_calls_made=d.get("tool_calls_made", 0),
            error=d.get("error"),
            image_candidates=image_candidates,
            slide_image_picks=slide_image_picks,
        )

    async def update_image_candidates(
        self,
        task_id: str,
        candidates: dict[int, list[dict[str, Any]]],
        *,
        user_id: str = "",
    ) -> None:
        """Persist the per-slide candidate pool.

        Stored as JSON with string keys (SQLite text). Caller passes the
        int-keyed dict and we convert here so callers don't need to think
        about JSON serialization edges.
        """
        if not user_id:
            raise ValueError("update_image_candidates requires user_id")
        payload = json.dumps({str(k): v for k, v in candidates.items()})
        await self._db.execute(
            "UPDATE agentic_tasks SET image_candidates = ?, "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            (payload, task_id, user_id),
        )
        await self._db.commit()

    async def update_slide_image_picks(
        self,
        task_id: str,
        picks: dict[int, dict[str, Any]],
        *,
        user_id: str = "",
    ) -> None:
        """Persist the user's per-slide selection (primary + additional)."""
        if not user_id:
            raise ValueError("update_slide_image_picks requires user_id")
        payload = json.dumps({str(k): v for k, v in picks.items()})
        await self._db.execute(
            "UPDATE agentic_tasks SET slide_image_picks = ?, "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            (payload, task_id, user_id),
        )
        await self._db.commit()


def _load_int_keyed(raw: Any) -> dict[int, Any]:
    """Decode a JSON dict whose keys are string-encoded ints."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    out: dict[int, Any] = {}
    for k, v in decoded.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class CachedToolCall:
    """A previously-executed tool call restored from the cache."""

    tool_name: str
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True


class ToolCallCache:
    """SQLite-backed cache of tool execution results.

    Keyed by ``(task_id, step_idx, call_hash)`` where ``call_hash`` is a
    deterministic SHA256 of ``(tool_name, canonical_args_json)``. On agentic
    task resume, the chain executor queries this cache before re-running
    any tool — a hit replays the stored output and metadata verbatim.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def get(
        self, task_id: str, step_idx: int, call_hash: str,
        *, user_id: str = "",
    ) -> CachedToolCall | None:
        cursor = await self._db.execute(
            """SELECT tool_name, output, metadata, success
               FROM agentic_tool_call_cache
               WHERE task_id = ? AND step_idx = ? AND call_hash = ?
               AND user_id = ?""",
            (task_id, step_idx, call_hash, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        tool_name, output, metadata_json, success = row
        try:
            metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        return CachedToolCall(
            tool_name=tool_name,
            output=output or "",
            metadata=metadata,
            success=bool(success),
        )

    async def put(
        self,
        task_id: str,
        step_idx: int,
        call_hash: str,
        *,
        tool_name: str,
        output: str,
        metadata: dict[str, Any] | None = None,
        success: bool = True,
        user_id: str = "",
    ) -> None:
        if not user_id:
            raise ValueError("agentic_tool_call_cache insert requires user_id")
        await self._db.execute(
            """INSERT OR REPLACE INTO agentic_tool_call_cache
               (task_id, step_idx, call_hash, user_id, tool_name, output, metadata, success)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                step_idx,
                call_hash,
                user_id,
                tool_name,
                output or "",
                json.dumps(metadata or {}),
                1 if success else 0,
            ),
        )
        await self._db.commit()

    async def clear_for_task(self, task_id: str, *, user_id: str = "") -> None:
        """Drop all cached calls for a (user, task)."""
        await self._db.execute(
            "DELETE FROM agentic_tool_call_cache WHERE task_id = ? AND user_id = ?",
            (task_id, user_id),
        )
        await self._db.commit()
