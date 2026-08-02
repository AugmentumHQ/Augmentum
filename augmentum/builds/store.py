"""SQLite persistence for user-facing Build Mode runs."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "canceled", "complete", "error", "cancelled"}


def _new_id() -> str:
    return f"build_{uuid.uuid4().hex[:16]}"


def normalize_status(status: str) -> str:
    value = (status or "").strip().lower()
    if value == "complete":
        return "completed"
    if value == "error":
        return "failed"
    if value == "cancelled":
        return "canceled"
    return value or "queued"


def legacy_status(status: str) -> str:
    value = normalize_status(status)
    if value == "completed":
        return "complete"
    if value == "failed":
        return "error"
    if value == "canceled":
        return "cancelled"
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, separators=(",", ":"), default=str)


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


class BuildRunStore:
    """Persists Build Mode runs.

    Rows are strictly user-scoped. Callers that do not have a user_id should
    not create build runs; the app builder artifact path already enforces the
    same rule for saved outputs.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create(
        self,
        *,
        user_id: str,
        session_id: str = "",
        task_id: str = "",
        build_id: str = "",
        kind: str = "application",
        status: str = "running",
        name: str = "",
        request: dict | None = None,
        progress: dict | None = None,
        profile_id: str = "static",
        target: str = "inline",
        capabilities: list | None = None,
        workspace_id: str = "",
    ) -> dict:
        if not user_id:
            raise ValueError("build_runs insert requires user_id")
        bid = build_id or _new_id()
        await self._db.execute(
            """INSERT INTO build_runs
               (id, user_id, session_id, task_id, kind, status, name,
                request_json, progress_json,
                profile_id, target, capabilities_json, workspace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                bid,
                user_id,
                session_id or "",
                task_id or "",
                kind or "application",
                normalize_status(status),
                name or "",
                _json_dumps(request),
                _json_dumps(progress),
                profile_id or "static",
                target or "inline",
                json.dumps(list(capabilities or []), separators=(",", ":")),
                workspace_id or None,
            ),
        )
        await self._db.commit()
        return await self.get(bid, user_id=user_id) or {
            "id": bid,
            "user_id": user_id,
            "session_id": session_id or "",
            "task_id": task_id or "",
            "artifact_id": "",
            "kind": kind or "application",
            "status": normalize_status(status),
            "name": name or "",
            "request": request or {},
            "progress": progress or {},
            "result": {},
            "error": None,
        }

    async def update(
        self,
        build_id: str,
        *,
        user_id: str,
        status: str | None = None,
        name: str | None = None,
        task_id: str | None = None,
        artifact_id: str | None = None,
        progress: dict | None = None,
        result: dict | None = None,
        error: str | None = None,
        profile_id: str | None = None,
        target: str | None = None,
        capabilities: list | None = None,
        workspace_id: str | None = None,
    ) -> None:
        if not user_id:
            raise ValueError("build_runs update requires user_id")
        set_parts = ["updated_at = datetime('now')"]
        params: list[Any] = []

        normalized_status = normalize_status(status) if status is not None else None
        if normalized_status is not None:
            set_parts.append("status = ?")
            params.append(normalized_status)
            if normalized_status in {"completed", "failed", "canceled"}:
                set_parts.append("completed_at = COALESCE(completed_at, datetime('now'))")
        if name is not None:
            set_parts.append("name = ?")
            params.append(name or "")
        if task_id is not None:
            set_parts.append("task_id = ?")
            params.append(task_id or "")
        if artifact_id is not None:
            set_parts.append("artifact_id = ?")
            params.append(artifact_id or "")
        if progress is not None:
            set_parts.append("progress_json = ?")
            params.append(_json_dumps(progress))
        if result is not None:
            set_parts.append("result_json = ?")
            params.append(_json_dumps(result))
        if error is not None:
            set_parts.append("error = ?")
            params.append(error or "")
        if profile_id is not None:
            set_parts.append("profile_id = ?")
            params.append(profile_id or "static")
        if target is not None:
            set_parts.append("target = ?")
            params.append(target or "inline")
        if capabilities is not None:
            set_parts.append("capabilities_json = ?")
            params.append(json.dumps(list(capabilities or []), separators=(",", ":")))
        if workspace_id is not None:
            set_parts.append("workspace_id = ?")
            params.append(workspace_id or None)

        params.extend([build_id, user_id])
        await self._db.execute(
            f"UPDATE build_runs SET {', '.join(set_parts)} WHERE id = ? AND user_id = ?",
            params,
        )
        await self._db.commit()

    async def begin_resume(self, build_id: str, *, user_id: str) -> int:
        """Flip a terminal build back to ``running`` for a resume and bump its
        resume counter. Returns the new resume_count (0 if the row was not
        updated — unknown id / wrong owner).

        Clears the prior error + completed_at so the build reads as active
        again while the continued loop runs. The status guard means a build
        that is *already* running can't be double-resumed.
        """
        if not user_id or not build_id:
            return 0
        cursor = await self._db.execute(
            """UPDATE build_runs
               SET status = 'running',
                   error = NULL,
                   completed_at = NULL,
                   resume_count = resume_count + 1,
                   updated_at = datetime('now')
               WHERE id = ? AND user_id = ?
                 AND status IN ('completed', 'failed', 'canceled', 'paused')""",
            (build_id, user_id),
        )
        await self._db.commit()
        if not cursor.rowcount:
            return 0
        run = await self.get(build_id, user_id=user_id)
        return int((run or {}).get("resume_count") or 0)

    async def mark_running_interrupted(
        self,
        *,
        reason: str,
        older_than_seconds: int = 0,
    ) -> int:
        """Fail any non-terminal build rows that cannot still be running.

        Called at process startup: in-memory build tasks do not survive a
        restart, so persisted ``running`` rows from a previous process should
        become explicit failures instead of looking active forever.
        """
        seconds = max(0, int(older_than_seconds or 0))
        cutoff = f"-{seconds} seconds"
        cursor = await self._db.execute(
            """UPDATE build_runs
               SET status = 'failed',
                   error = CASE
                       WHEN error IS NULL OR error = '' THEN ?
                       ELSE error
                   END,
                   updated_at = datetime('now'),
                   completed_at = COALESCE(completed_at, datetime('now'))
               WHERE status IN ('running', 'queued')
                 AND updated_at <= datetime('now', ?)""",
            (reason or "Build interrupted before completion.", cutoff),
        )
        await self._db.commit()
        return int(cursor.rowcount or 0)

    async def mark_running_stale(
        self,
        build_id: str,
        *,
        user_id: str,
        max_age_seconds: int,
        reason: str,
    ) -> bool:
        """Fail one running build if it has not updated recently."""
        if not user_id:
            return False
        seconds = max(0, int(max_age_seconds or 0))
        cutoff = f"-{seconds} seconds"
        cursor = await self._db.execute(
            """UPDATE build_runs
               SET status = 'failed',
                   error = CASE
                       WHEN error IS NULL OR error = '' THEN ?
                       ELSE error
                   END,
                   updated_at = datetime('now'),
                   completed_at = COALESCE(completed_at, datetime('now'))
               WHERE id = ?
                 AND user_id = ?
                 AND status IN ('running', 'queued')
                 AND updated_at <= datetime('now', ?)""",
            (reason or "Build stopped updating before completion.", build_id, user_id, cutoff),
        )
        await self._db.commit()
        return bool(cursor.rowcount)

    async def get(self, build_id: str, *, user_id: str) -> dict | None:
        if not user_id:
            return None
        cursor = await self._db.execute(
            "SELECT * FROM build_runs WHERE id = ? AND user_id = ?",
            (build_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(cursor, row)

    async def latest_for_session(
        self,
        session_id: str = "",
        *,
        user_id: str,
    ) -> dict | None:
        if not user_id:
            return None
        # Terminal builds that the user has dismissed (acked_at IS NOT NULL)
        # or that have aged out (>24h since last update) are hidden from
        # the persistent monitor. Active builds always surface so a user
        # who switched devices can follow the in-progress run.
        terminal_filter = (
            " AND (status NOT IN ('completed','failed','canceled')"
            "      OR (acked_at IS NULL"
            "          AND updated_at >= datetime('now','-24 hours')))"
        )
        if session_id:
            cursor = await self._db.execute(
                "SELECT * FROM build_runs"
                " WHERE user_id = ? AND session_id = ?"
                + terminal_filter
                + " ORDER BY updated_at DESC LIMIT 1",
                (user_id, session_id),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM build_runs"
                " WHERE user_id = ?"
                + terminal_filter
                + " ORDER BY updated_at DESC LIMIT 1",
                (user_id,),
            )
        row = await cursor.fetchone()
        return self._row_to_dict(cursor, row) if row else None

    async def mark_acked(self, build_id: str, *, user_id: str) -> bool:
        """Stamp a build as user-dismissed so the persistent monitor stops
        resurfacing it on subsequent page loads (cross-device)."""
        if not user_id or not build_id:
            return False
        cursor = await self._db.execute(
            "UPDATE build_runs SET acked_at = datetime('now')"
            " WHERE id = ? AND user_id = ? AND acked_at IS NULL",
            (build_id, user_id),
        )
        await self._db.commit()
        return bool(cursor.rowcount)

    async def list_for_session(
        self,
        session_id: str = "",
        *,
        user_id: str,
        limit: int = 50,
    ) -> list[dict]:
        if not user_id:
            return []
        limit = max(1, min(int(limit or 50), 200))
        if session_id:
            cursor = await self._db.execute(
                """SELECT * FROM build_runs
                   WHERE user_id = ? AND session_id = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, session_id, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT * FROM build_runs
                   WHERE user_id = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        return [self._row_to_dict(cursor, row) for row in rows]

    def _row_to_dict(self, cursor, row) -> dict:
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row))
        return {
            "id": d["id"],
            "user_id": d.get("user_id") or "",
            "session_id": d.get("session_id") or "",
            "task_id": d.get("task_id") or "",
            "artifact_id": d.get("artifact_id") or "",
            "kind": d.get("kind") or "application",
            "status": normalize_status(d.get("status") or ""),
            "name": d.get("name") or "",
            "request": _json_loads(d.get("request_json"), {}),
            "progress": _json_loads(d.get("progress_json"), {}),
            "result": _json_loads(d.get("result_json"), {}),
            "error": d.get("error"),
            "profile_id": d.get("profile_id") or "static",
            "target": d.get("target") or "inline",
            "capabilities": _json_loads(d.get("capabilities_json"), []),
            "workspace_id": d.get("workspace_id") or "",
            "resume_count": int(d.get("resume_count") or 0),
            "created_at": d.get("created_at"),
            "updated_at": d.get("updated_at"),
            "completed_at": d.get("completed_at"),
            "acked_at": d.get("acked_at"),
        }
