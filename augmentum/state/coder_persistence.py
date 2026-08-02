"""Persistence helpers for coder-mode session state."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import aiosqlite

from augmentum.coder.state import CoderState
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class CoderPersistence:
    """Read/write coder session rows from ``coder_sessions``."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def load_session_state(
        self,
        session_id: str,
        *,
        user_id: str = "",
    ) -> CoderState | None:
        """Load persisted state for ``session_id`` if it exists."""
        query = "SELECT * FROM coder_sessions WHERE session_id = ?"
        params: list[Any] = [session_id]
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
            query += " ORDER BY CASE WHEN user_id = ? THEN 0 ELSE 1 END"
            params.append(user_id)
        query += " LIMIT 1"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return CoderState.from_row(dict(row))

    async def save_session_state(
        self,
        session_id: str,
        state: CoderState,
        *,
        user_id: str = "",
    ) -> bool:
        """Upsert coder state while preserving any stored conversation.

        Returns True if the row was written, False if an existing row
        owned by a *different* user blocked the upsert (the ON CONFLICT
        guard's WHERE clause). A False here means the caller's state did
        NOT persist — it must not be treated as a silent success.
        """
        if not user_id:
            raise ValueError("coder_sessions insert requires user_id")
        data = state.to_dict()
        data["session_id"] = session_id

        columns = list(data.keys())
        values = [data[col] for col in columns]
        columns.append("user_id")
        values.append(user_id)

        update_columns = [
            col for col in columns
            if col not in {"session_id", "created_at"}
        ]
        update_assignments = ", ".join(
            f"{col} = excluded.{col}" for col in update_columns
        )

        sql = (
            f"INSERT INTO coder_sessions ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)}) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            f"{update_assignments} "
            "WHERE coder_sessions.user_id IS NULL "
            "OR coder_sessions.user_id = excluded.user_id"
        )

        try:
            cursor = await self._conn.execute(sql, values)
        except sqlite3.IntegrityError as exc:
            # FK parents: users(id) and projects(id). A project deleted
            # mid-session SET-NULLs the stored row, but the in-memory
            # state re-asserts the stale id on every subsequent save —
            # without this retry the session would never persist again.
            # A ghost user_id is real corruption and stays a hard error.
            if ("FOREIGN KEY" in str(exc) and data.get("project_id")
                    and "project_id" in columns):
                log.warning(
                    "coder_session_stale_project_ref_dropped",
                    session_id=session_id,
                    project_id=data["project_id"],
                )
                values[columns.index("project_id")] = None
                state.project_id = ""
                cursor = await self._conn.execute(sql, values)
            else:
                raise
        await self._conn.commit()

        # rowcount uniquely identifies the blocked-upsert case: a fresh
        # insert and a permitted update both touch exactly one row, while
        # a conflict whose ownership WHERE fails touches zero. Surface it
        # — otherwise a cross-user session_id collision drops the write
        # and the caller thinks it saved.
        if cursor.rowcount == 0:
            log.warning(
                "coder_session_save_blocked_by_owner",
                session_id=session_id,
                user_id=user_id,
            )
            return False
        return True

    async def load_conversation(
        self,
        workspace_id: str,
        *,
        user_id: str = "",
        reconcile: bool = True,
    ) -> list[dict]:
        """Return the persisted message list for ``workspace_id``.

        When ``reconcile`` is set (the default), any ``role:'tool'`` message
        whose ``result`` never round-tripped into the stored blob is healed
        from the durable ``coder_turn_events`` ledger before returning —
        see ``_reconcile_tool_results``. This is what makes a tool call that
        succeeded live but whose result was lost to a restart / device
        handoff render (and read back into the model) as the success it was,
        rather than a synthetic "interrupted" failure.
        """
        query = "SELECT conversation FROM coder_sessions WHERE session_id = ?"
        params: list[Any] = [workspace_id]
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
            query += " ORDER BY CASE WHEN user_id = ? THEN 0 ELSE 1 END"
            params.append(user_id)
        query += " LIMIT 1"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if row is None or not row[0]:
            return []
        messages = json.loads(row[0])
        if reconcile:
            messages, _ = await self._reconcile_tool_results(
                messages, workspace_id, user_id=user_id,
            )
        return messages

    async def _reconcile_tool_results(
        self,
        messages: list[dict],
        workspace_id: str,
        *,
        user_id: str = "",
    ) -> tuple[list[dict], bool]:
        """Fill missing tool ``result`` fields from the durable event ledger.

        The conversation blob in ``coder_sessions`` is written client-side,
        best-effort, last-writer-wins — so a tool call whose result never got
        POSTed back (server restarting during the debounce window, or the
        turn ran on another device that never saved) is stored as
        ``{role:'tool', ..., result: null}``. The frontend then paints it as
        a failed/interrupted call, and a continuation turn re-feeds that
        false failure to the model.

        The result *did* land durably: ``CoderTurnLedger`` writes every
        ``tool_result`` event to ``coder_turn_events`` as it streams, keyed by
        the same tool-call id the conversation uses. This treats that ledger
        as the source of truth: for each null-result tool message we look up
        the latest recorded result for its id and reconstruct a minimal
        ``{success, output_preview, error}`` object the existing renderers
        already understand.

        Returns ``(messages, changed)``. Best-effort: any ledger read failure
        leaves the messages untouched rather than breaking the load.
        """
        if not isinstance(messages, list) or not messages:
            return messages, False
        # Fast path: only pay for the ledger query when something is actually
        # missing a result. A fully-saved conversation never touches the DB.
        missing_ids = {
            str(m.get("id") or "")
            for m in messages
            if isinstance(m, dict)
            and m.get("role") == "tool"
            and not m.get("result")
            and m.get("id")
        }
        if not missing_ids:
            return messages, False
        try:
            result_map = await self._ledger_tool_results(
                workspace_id, user_id=user_id,
            )
        except Exception as exc:
            log.warning(
                "coder_conversation_reconcile_failed",
                workspace_id=workspace_id,
                error=str(exc),
            )
            return messages, False
        if not result_map:
            return messages, False
        changed = False
        for m in messages:
            if (
                not isinstance(m, dict)
                or m.get("role") != "tool"
                or m.get("result")
            ):
                continue
            recovered = result_map.get(str(m.get("id") or ""))
            if recovered is not None:
                m["result"] = recovered
                changed = True
        return messages, changed

    async def _ledger_tool_results(
        self,
        workspace_id: str,
        *,
        user_id: str = "",
    ) -> dict[str, dict]:
        """Build a ``tool_call_id -> result`` map from ``coder_turn_events``.

        Scoped to the workspace via ``coder_turn_runs.project_id`` (equals the
        checkout/workspace id post-migration 200). Events are read oldest →
        newest so a re-run of the same tool-call id keeps the latest recorded
        outcome. The reconstructed result mirrors the shape the runtime emits
        on completion (``success`` + ``output_preview``; ``error`` carries the
        preview on failure) so ``updateToolResult`` renders it without special
        cases.
        """
        query = (
            "SELECT e.payload FROM coder_turn_events e "
            "JOIN coder_turn_runs r ON e.run_id = r.id "
            "WHERE r.project_id = ? AND e.status = 'tool_result'"
        )
        params: list[Any] = [workspace_id]
        if user_id:
            query += " AND e.user_id = ?"
            params.append(user_id)
        query += " ORDER BY e.timestamp ASC, e.seq ASC"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            try:
                payload = json.loads(row[0] or "{}")
            except Exception:
                continue
            tr = payload.get("tool_result") if isinstance(payload, dict) else None
            if not isinstance(tr, dict):
                continue
            tid = str(tr.get("id") or "")
            if not tid:
                continue
            success = tr.get("success")
            preview = str(tr.get("output_preview") or "")
            out[tid] = {
                "success": bool(success),
                "output_preview": preview,
                "error": "" if success else preview,
                "reconciled": True,
            }
        return out

    async def save_conversation(
        self,
        workspace_id: str,
        messages: list[dict],
        *,
        user_id: str = "",
    ) -> None:
        """Upsert conversation history for a workspace-scoped coder session.

        The incoming messages are reconciled against the durable event
        ledger first (see ``_reconcile_tool_results``): a client snapshot
        that carries ``result: null`` for a tool whose real result the ledger
        already holds is healed before persistence, so a lossy or stale save
        can never overwrite a good result with a null. The stored blob thus
        converges toward complete on every write.
        """
        if not user_id:
            raise ValueError("coder_sessions conversation insert requires user_id")
        try:
            messages, _ = await self._reconcile_tool_results(
                messages, workspace_id, user_id=user_id,
            )
        except Exception as exc:  # never block a save on reconciliation
            log.warning(
                "coder_conversation_save_reconcile_failed",
                workspace_id=workspace_id,
                error=str(exc),
            )
        now = time.time()
        conversation_json = json.dumps(messages)

        await self._conn.execute(
            """
            INSERT INTO coder_sessions
                (session_id, workspace_id, phase, conversation, user_id, created_at, updated_at)
            VALUES (?, ?, 'waiting', ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                conversation = excluded.conversation,
                workspace_id = excluded.workspace_id,
                updated_at = excluded.updated_at
            WHERE coder_sessions.user_id IS NULL
               OR coder_sessions.user_id = excluded.user_id
            """,
            (workspace_id, workspace_id, conversation_json, user_id, now, now),
        )
        await self._conn.commit()

    async def delete_session(
        self,
        session_id: str,
        *,
        user_id: str = "",
    ) -> None:
        """Delete the persisted coder session row entirely."""
        if not user_id:
            raise ValueError("coder_sessions delete requires user_id")
        await self._conn.execute(
            "DELETE FROM coder_sessions "
            "WHERE session_id = ? AND (user_id = ? OR user_id IS NULL)",
            (session_id, user_id),
        )
        await self._conn.commit()
