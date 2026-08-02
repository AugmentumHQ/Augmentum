"""Persistence for Claude Code (external-coder) runs.

Backs the per-workspace run history + native-resume surface. Two tables
(migration 287):

* ``claude_runs``        — one row per run (metadata + verbatim ``raw_jsonl``).
* ``claude_run_events``  — one normalized event per row, appended LIVE during a
  run so the history view survives a mid-run refresh.

All functions take an ``aiosqlite`` connection (the route gets it via
``request.app.state.state_manager.backend.conn``) and are user-scoped: every
read/write carries ``user_id`` so one user can never see or mutate another's
runs (CLAUDE.md data-isolation rule).
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def create_run(
    conn: Any, *, run_id: str, user_id: str, workspace_id: str,
    task: str, permission: str = "auto", resumed_from: str = "",
    session_id: str = "", model: str = "",
) -> None:
    """Insert a freshly-started run (status='running').

    ``model`` is the model the user PINNED at dispatch ("" = account default);
    the real model Claude used is captured from the stream and written at finish.
    """
    await conn.execute(
        """
        INSERT INTO claude_runs
            (id, user_id, workspace_id, session_id, task, permission,
             status, resumed_from, model, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, datetime('now'), datetime('now'))
        """,
        (run_id, user_id, workspace_id, session_id, task[:4000],
         permission, resumed_from, model or ""),
    )
    await conn.commit()


async def add_event(
    conn: Any, *, run_id: str, user_id: str, seq: int,
    kind: str, text: str = "", tool: str = "", path: str = "",
) -> None:
    """Append one normalized event row (cheap insert — called per event)."""
    await conn.execute(
        """
        INSERT INTO claude_run_events
            (run_id, user_id, seq, kind, text, tool, path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, user_id, seq, kind, (text or "")[:4000], tool or "", path or ""),
    )
    await conn.commit()


async def set_session_id(
    conn: Any, *, run_id: str, user_id: str, session_id: str,
) -> None:
    """Persist Claude's session id as soon as it's seen, so a run is resumable
    even if it later crashes mid-flight. No-op if already set to the same value."""
    if not session_id:
        return
    await conn.execute(
        """
        UPDATE claude_runs SET session_id=?, updated_at=datetime('now')
        WHERE id=? AND user_id=? AND session_id != ?
        """,
        (session_id, run_id, user_id, session_id),
    )
    await conn.commit()


async def finish_run(
    conn: Any, *, run_id: str, user_id: str, status: str,
    outcome: str = "", error: str = "", files_changed: list[str] | None = None,
    raw_jsonl: str = "", session_id: str = "", cost_usd: float = 0.0,
    num_turns: int = 0, duration_ms: int = 0, model: str = "",
) -> None:
    """Finalize a run: terminal status + summary + full-fidelity raw transcript.

    ``raw_jsonl`` is written once here (from the run's in-memory buffer) rather
    than appended per-line, so a long run doesn't repeatedly rewrite a growing
    blob. The normalized event rows already persisted live carry the display view.
    """
    await conn.execute(
        """
        UPDATE claude_runs SET
            status=?, outcome=?, error=?, files_changed=?, raw_jsonl=?,
            cost_usd=?, num_turns=?, duration_ms=?,
            session_id=CASE WHEN ?!='' THEN ? ELSE session_id END,
            model=CASE WHEN ?!='' THEN ? ELSE model END,
            updated_at=datetime('now')
        WHERE id=? AND user_id=?
        """,
        (
            status, (outcome or "")[:2000], (error or "")[:2000],
            json.dumps(files_changed or []), raw_jsonl or "",
            float(cost_usd or 0.0), int(num_turns or 0), int(duration_ms or 0),
            session_id, session_id,
            model or "", model or "",
            run_id, user_id,
        ),
    )
    await conn.commit()


async def mark_status(
    conn: Any, *, run_id: str, user_id: str, status: str, error: str = "",
) -> None:
    """Status-only update (used by Stop/cancel). Deliberately does NOT touch
    raw_jsonl / files_changed / events so a cancelled run keeps whatever it had
    persisted live up to the moment it was stopped."""
    await conn.execute(
        "UPDATE claude_runs SET status=?, error=CASE WHEN ?!='' THEN ? ELSE error END, "
        "updated_at=datetime('now') WHERE id=? AND user_id=?",
        (status, error, (error or "")[:2000], run_id, user_id),
    )
    await conn.commit()


def _row_to_meta(row: Any) -> dict:
    return {
        "id": row[0], "workspace_id": row[1], "session_id": row[2],
        "task": row[3], "permission": row[4], "status": row[5],
        "outcome": row[6], "error": row[7],
        "files_changed": json.loads(row[8] or "[]"),
        "cost_usd": row[9], "num_turns": row[10], "duration_ms": row[11],
        "resumed_from": row[12], "created_at": row[13], "updated_at": row[14],
        "model": row[15],
    }


_META_COLS = (
    "id, workspace_id, session_id, task, permission, status, outcome, error, "
    "files_changed, cost_usd, num_turns, duration_ms, resumed_from, "
    "created_at, updated_at, model"
)


async def list_runs(
    conn: Any, *, user_id: str, workspace_id: str, limit: int = 50,
) -> list[dict]:
    """Run metadata for a workspace, newest first (no raw/events — cheap list)."""
    cur = await conn.execute(
        f"SELECT {_META_COLS} FROM claude_runs "
        "WHERE user_id=? AND workspace_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, workspace_id, max(1, min(int(limit or 50), 200))),
    )
    rows = await cur.fetchall()
    return [_row_to_meta(r) for r in rows]


async def get_run(
    conn: Any, *, run_id: str, user_id: str, include_raw: bool = False,
) -> dict | None:
    """One run with its normalized event transcript. ``include_raw`` adds the
    verbatim stream-json (full fidelity) — opt-in since it can be large."""
    cur = await conn.execute(
        f"SELECT {_META_COLS} FROM claude_runs WHERE id=? AND user_id=?",
        (run_id, user_id),
    )
    row = await cur.fetchone()
    if not row:
        return None
    out = _row_to_meta(row)
    ev = await conn.execute(
        "SELECT seq, kind, text, tool, path FROM claude_run_events "
        "WHERE run_id=? AND user_id=? ORDER BY seq ASC",
        (run_id, user_id),
    )
    out["events"] = [
        {"seq": e[0], "kind": e[1], "text": e[2], "tool": e[3], "path": e[4]}
        for e in await ev.fetchall()
    ]
    if include_raw:
        raw = await conn.execute(
            "SELECT raw_jsonl FROM claude_runs WHERE id=? AND user_id=?",
            (run_id, user_id),
        )
        r = await raw.fetchone()
        out["raw_jsonl"] = (r[0] if r else "") or ""
    return out


async def session_for_run(conn: Any, *, run_id: str, user_id: str) -> dict | None:
    """Resolve a prior run's ``{session_id, workspace_id}`` for resume. Returns
    None if the run doesn't exist / isn't this user's."""
    cur = await conn.execute(
        "SELECT session_id, workspace_id FROM claude_runs WHERE id=? AND user_id=?",
        (run_id, user_id),
    )
    row = await cur.fetchone()
    if not row:
        return None
    return {"session_id": row[0] or "", "workspace_id": row[1] or ""}
