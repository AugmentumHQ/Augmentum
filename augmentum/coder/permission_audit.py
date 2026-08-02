"""Durable audit trail for coder tool-permission decisions.

Backs the ``coder_permission_audit`` table (migration 260). Before this
existed, an approval lived only in structlog output — once logs rotate
there was no record that the user approved the ``shell_exec`` that ran.

One row per DECISION, not per tool call. ``decided_by`` vocabulary:

- ``user``       — Allow/Deny clicked in the approval modal
- ``timeout``    — modal ignored until the registry timeout fired
- ``disconnect`` — client went away mid-request (denied)
- ``policy``     — a ``permissions.toml`` rule matched

Plan-mode ``auto`` approvals are deliberately NOT recorded — auto mode
approves every mutating call in a turn (hundreds of noise rows), and
``coder_turn_events`` already records each executed tool call. This
table answers "who allowed this", not "what ran".

All writes are best-effort: an audit failure logs a warning and never
blocks the tool dispatch it describes.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Cap the stored tool_input preview. A file_write's content can be
# hundreds of KB; the full call already lives in the turn ledger.
_INPUT_PREVIEW_CHARS = 2000


def _input_preview(tool_input: dict | None) -> str:
    try:
        rendered = json.dumps(tool_input or {}, default=str, sort_keys=True)
    except Exception:
        rendered = "{}"
    if len(rendered) > _INPUT_PREVIEW_CHARS:
        rendered = rendered[:_INPUT_PREVIEW_CHARS] + "…(truncated)"
    return rendered


def resolve_store(app_state: Any) -> PermissionAuditStore | None:
    """Build a store from ``app.state``, or None when the state manager
    isn't wired yet.

    Resolution is deliberately lazy/per-call: the permission registry is
    constructed BEFORE ``app.state.state_manager`` exists in
    ``create_app`` order, so the sink can't capture a connection at
    construction time. Same conn chain as
    ``CoderHandler._resolve_archive_conn``.
    """
    conn = getattr(
        getattr(getattr(app_state, "state_manager", None), "backend", None),
        "conn",
        None,
    )
    return PermissionAuditStore(conn) if conn is not None else None


class PermissionAuditStore:
    """SQLite CRUD for ``coder_permission_audit``."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def record(
        self,
        *,
        tool_name: str,
        decision: str,
        decided_by: str,
        user_id: str = "",
        workspace_id: str = "",
        tool_input: dict | None = None,
    ) -> None:
        """Append one decision row. Best-effort — never raises."""
        try:
            await self._conn.execute(
                """
                INSERT INTO coder_permission_audit
                    (id, user_id, workspace_id, tool_name, tool_input,
                     decision, decided_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    user_id or "",
                    workspace_id or "",
                    tool_name or "",
                    _input_preview(tool_input),
                    decision or "",
                    decided_by or "",
                    time.time(),
                ),
            )
            await self._conn.commit()
        except Exception as exc:
            log.warning(
                "coder.permission_audit_write_failed",
                tool_name=tool_name,
                decision=decision,
                error=str(exc),
            )

    async def list_events(
        self,
        *,
        user_id: str = "",
        workspace_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first decision rows, scoped to the caller.

        Empty ``user_id`` returns all rows (single-tenant dev setups
        without auth — same convention as ``PermissionRegistry
        .pending_for``).
        """
        query = (
            "SELECT id, user_id, workspace_id, tool_name, tool_input, "
            "decision, decided_by, created_at FROM coder_permission_audit"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": r[0],
                "user_id": r[1],
                "workspace_id": r[2],
                "tool_name": r[3],
                "tool_input": r[4],
                "decision": r[5],
                "decided_by": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]
