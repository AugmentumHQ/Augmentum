"""Persistence for pushed pi (terminal agent) session mirrors.

Backs the /api/coder/external/pi surface (migration 311):

* ``pi_runs``       — one row per pi session (metadata; the host session
  file is the raw record, referenced by path).
* ``pi_run_events`` — normalized events pushed live by the pi host, one
  row each, idempotent on (run_id, seq) so batch retries are safe.

Mirrors ``run_store.py`` (claude_runs) deliberately — same connection
convention (``request.app.state.state_manager.backend.conn``) and the same
user-scoping rule: every read/write carries ``user_id`` (CLAUDE.md
data-isolation rule).
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_META_COLS = (
    "id, project, session_file, title, model, status, outcome, error, "
    "files_changed, num_turns, created_at, updated_at"
)


def _row_to_meta(row: Any) -> dict:
    return {
        "id": row[0], "project": row[1], "session_file": row[2],
        "title": row[3], "model": row[4], "status": row[5],
        "outcome": row[6], "error": row[7],
        "files_changed": json.loads(row[8] or "[]"),
        "num_turns": row[9], "created_at": row[10], "updated_at": row[11],
        "engine": "pi",
    }


async def upsert_run(
    conn: Any, *, run_id: str, user_id: str, project: str = "",
    session_file: str = "", title: str = "", model: str = "",
) -> None:
    """Create or refresh a run row (status flips back to 'running' on
    re-attach — the host is telling us the session is live again)."""
    await conn.execute(
        """
        INSERT INTO pi_runs
            (id, user_id, project, session_file, title, model,
             status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'running', datetime('now'), datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            project=excluded.project,
            session_file=excluded.session_file,
            title=CASE WHEN excluded.title != '' THEN excluded.title ELSE pi_runs.title END,
            model=CASE WHEN excluded.model != '' THEN excluded.model ELSE pi_runs.model END,
            status='running',
            updated_at=datetime('now')
        WHERE pi_runs.user_id = excluded.user_id
        """,
        (run_id, user_id, (project or "")[:200], (session_file or "")[:1000],
         (title or "")[:500], (model or "")[:200]),
    )
    await conn.commit()


async def add_events(
    conn: Any, *, run_id: str, user_id: str, events: list[dict],
) -> int:
    """Batch-append normalized events. Idempotent on (run_id, seq) via the
    unique index + INSERT OR IGNORE, so the host can retry a whole batch
    after a network blip. Returns rows actually inserted.

    The run must already exist in the caller's scope: the (run_id, seq)
    unique index is shared across users, so an unchecked insert would let
    one user squat seq slots on another user's run_id (and strand orphan
    events no listing can reach)."""
    cur = await conn.execute(
        "SELECT 1 FROM pi_runs WHERE id=? AND user_id=?", (run_id, user_id),
    )
    if await cur.fetchone() is None:
        log.warning("pi_run_events_unknown_run", run_id=run_id)
        return 0
    inserted = 0
    for ev in events:
        try:
            seq = int(ev.get("seq", -1))
        except (TypeError, ValueError):
            continue
        if seq < 0:
            continue
        cur = await conn.execute(
            """
            INSERT OR IGNORE INTO pi_run_events
                (run_id, user_id, seq, kind, text, tool, path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, user_id, seq,
                str(ev.get("kind", ""))[:50],
                str(ev.get("text", ""))[:4000],
                str(ev.get("tool", ""))[:200],
                str(ev.get("path", ""))[:1000],
            ),
        )
        inserted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    await conn.execute(
        "UPDATE pi_runs SET updated_at=datetime('now') WHERE id=? AND user_id=?",
        (run_id, user_id),
    )
    await conn.commit()
    return inserted


async def finish_run(
    conn: Any, *, run_id: str, user_id: str, status: str,
    outcome: str = "", error: str = "", files_changed: list[str] | None = None,
    num_turns: int = 0,
) -> None:
    """Terminal (or detach) update from the host."""
    if status not in ("done", "failed", "detached"):
        status = "detached"
    await conn.execute(
        """
        UPDATE pi_runs SET
            status=?, outcome=?, error=?, files_changed=?, num_turns=?,
            updated_at=datetime('now')
        WHERE id=? AND user_id=?
        """,
        (
            status, (outcome or "")[:2000], (error or "")[:2000],
            json.dumps(files_changed or []), int(num_turns or 0),
            run_id, user_id,
        ),
    )
    await conn.commit()


async def list_runs(
    conn: Any, *, user_id: str, project: str = "", limit: int = 50,
) -> list[dict]:
    """Run metadata newest-first, optionally filtered by project."""
    limit = max(1, min(int(limit or 50), 200))
    if project:
        cur = await conn.execute(
            f"SELECT {_META_COLS} FROM pi_runs "
            "WHERE user_id=? AND project=? ORDER BY created_at DESC LIMIT ?",
            (user_id, project, limit),
        )
    else:
        cur = await conn.execute(
            f"SELECT {_META_COLS} FROM pi_runs "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
    rows = await cur.fetchall()
    return [_row_to_meta(r) for r in rows]


async def get_run(
    conn: Any, *, run_id: str, user_id: str, since_seq: int = -1,
) -> dict | None:
    """One run with its event transcript (optionally only events after
    ``since_seq`` — the SSE poller's incremental read)."""
    cur = await conn.execute(
        f"SELECT {_META_COLS} FROM pi_runs WHERE id=? AND user_id=?",
        (run_id, user_id),
    )
    row = await cur.fetchone()
    if not row:
        return None
    out = _row_to_meta(row)
    ev = await conn.execute(
        "SELECT seq, kind, text, tool, path FROM pi_run_events "
        "WHERE run_id=? AND user_id=? AND seq>? ORDER BY seq ASC",
        (run_id, user_id, int(since_seq)),
    )
    out["events"] = [
        {"seq": e[0], "kind": e[1], "text": e[2], "tool": e[3], "path": e[4]}
        for e in await ev.fetchall()
    ]
    return out
