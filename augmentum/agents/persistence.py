"""SQLite store for ``coder_subagent_runs``.

One row per ``task_dispatch`` spawn: lifecycle (started_at /
completed_at / stop_reason), budget usage (iterations / tokens /
wallclock_ms), the structured tool-call log (JSON blob), and the final
output text.

User-scoped per CLAUDE.md — every CRUD method takes ``user_id`` and
appends ``AND user_id = ?`` to its WHERE clauses.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from augmentum.agents.loop import SubagentResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class SubagentRunStore:
    """Per-spawn audit log. Wraps a single ``aiosqlite`` connection."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def start_run(
        self,
        *,
        subagent_id: str,
        role: str,
        prompt: str,
        model_spec: str,
        model_resolved: str,
        backend_key: str,
        context_mode: str,
        parent_run_id: str = "",
        parent_turn_id: str = "",
        workspace_id: str = "",
        session_id: str = "",
        user_id: str = "",
        started_at: float | None = None,
    ) -> None:
        """Insert a breadcrumb on spawn. ``complete_run`` upgrades the
        row with the final result fields."""
        ts = int(started_at or time.time())
        async with self._conn.execute(
            """
            INSERT OR REPLACE INTO coder_subagent_runs (
                subagent_id, parent_run_id, parent_turn_id,
                user_id, workspace_id, session_id,
                role, model_spec, model_resolved, backend_key,
                prompt, context_mode,
                started_at, stop_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (
                subagent_id, parent_run_id, parent_turn_id,
                user_id, workspace_id, session_id,
                role, model_spec, model_resolved, backend_key,
                prompt, context_mode,
                ts,
            ),
        ):
            await self._conn.commit()

    async def complete_run(
        self,
        result: SubagentResult,
        *,
        subagent_id: str,
        user_id: str = "",
        completed_at: float | None = None,
    ) -> None:
        """Write the final result fields onto an existing breadcrumb."""
        ts = int(completed_at or time.time())
        tool_call_log = json.dumps(
            [asdict(entry) for entry in result.tool_call_log],
            default=str,
        )
        async with self._conn.execute(
            """
            UPDATE coder_subagent_runs SET
                completed_at  = ?,
                stop_reason   = ?,
                stop_detail   = ?,
                stuck_pattern = ?,
                iterations    = ?,
                tool_calls    = ?,
                tokens_in     = ?,
                tokens_out    = ?,
                wallclock_ms  = ?,
                output_text   = ?,
                tool_call_log = ?,
                model_resolved = COALESCE(NULLIF(?, ''), model_resolved),
                verification  = ?,
                verification_reason = ?
            WHERE subagent_id = ? AND user_id = ?
            """,
            (
                ts,
                result.stop_reason,
                (result.stop_detail or "")[:512],
                result.stuck_pattern or "",
                int(result.iterations),
                int(result.tool_calls),
                int(result.tokens_in),
                int(result.tokens_out),
                int(result.wallclock_ms),
                result.output[:64_000],
                tool_call_log,
                result.model_resolved or "",
                getattr(result, "verification", "unchecked") or "unchecked",
                (getattr(result, "verification_reason", "") or "")[:512],
                subagent_id,
                user_id,
            ),
        ):
            await self._conn.commit()

    async def get_run(self, subagent_id: str, *, user_id: str = "") -> dict[str, Any] | None:
        params: list[Any] = [subagent_id]
        query = (
            "SELECT subagent_id, parent_run_id, parent_turn_id, user_id, "
            "workspace_id, session_id, role, model_spec, model_resolved, "
            "backend_key, prompt, context_mode, started_at, completed_at, "
            "stop_reason, stop_detail, stuck_pattern, iterations, tool_calls, "
            "tokens_in, tokens_out, wallclock_ms, output_text, tool_call_log, "
            "verification, verification_reason "
            "FROM coder_subagent_runs WHERE subagent_id = ?"
        )
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        async with self._conn.execute(query, tuple(params)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    async def list_runs(
        self,
        *,
        user_id: str = "",
        parent_run_id: str = "",
        session_id: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        clauses: list[str] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if parent_run_id:
            clauses.append("parent_run_id = ?")
            params.append(parent_run_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        # Skip the heavy output_text + tool_call_log blobs in list view.
        query = (
            "SELECT subagent_id, parent_run_id, parent_turn_id, user_id, "
            "workspace_id, session_id, role, model_spec, model_resolved, "
            "backend_key, prompt, context_mode, started_at, completed_at, "
            "stop_reason, stop_detail, stuck_pattern, iterations, tool_calls, "
            "tokens_in, tokens_out, wallclock_ms, "
            "substr(output_text, 1, 256) AS output_text, '[]' AS tool_call_log, "
            "verification, verification_reason "
            f"FROM coder_subagent_runs {where} "
            "ORDER BY started_at DESC LIMIT ?"
        )
        params.append(max(1, min(limit, 500)))
        async with self._conn.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


_COLUMNS = (
    "subagent_id", "parent_run_id", "parent_turn_id", "user_id",
    "workspace_id", "session_id", "role", "model_spec", "model_resolved",
    "backend_key", "prompt", "context_mode", "started_at", "completed_at",
    "stop_reason", "stop_detail", "stuck_pattern", "iterations", "tool_calls",
    "tokens_in", "tokens_out", "wallclock_ms", "output_text", "tool_call_log",
    "verification", "verification_reason",
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    out: dict[str, Any] = dict(zip(_COLUMNS, row, strict=False))
    raw = out.get("tool_call_log")
    if raw:
        try:
            out["tool_call_log"] = json.loads(raw)
        except (TypeError, ValueError):
            out["tool_call_log"] = []
    return out
