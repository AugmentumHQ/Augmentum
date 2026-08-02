"""Persistence for self-edit attempts — the permanent, never-pruned lineage.

Backs ``self_edit_attempts`` (migration 288). One row per attempt; the detailed
edit transcript lives in the linked Claude run (run_id → claude_runs). By design
there is **no delete function** — the archive is the point (grow by remembering
mistakes; a rollback reverts code but keeps the lesson). User-scoped throughout.
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_COLS = (
    "id, user_id, objective, surface, tier, status, base_ref, candidate_ref, "
    "run_id, gate_passed, gate_verdict, files_changed, outcome, lesson, "
    "promoted_commit, created_at, updated_at, target, source"
)


def _row(r: Any) -> dict:
    return {
        "id": r[0], "user_id": r[1], "objective": r[2], "surface": r[3],
        "tier": r[4], "status": r[5], "base_ref": r[6], "candidate_ref": r[7],
        "run_id": r[8], "gate_passed": bool(r[9]),
        "gate_verdict": json.loads(r[10] or "{}"),
        "files_changed": json.loads(r[11] or "[]"),
        "outcome": r[12], "lesson": r[13], "promoted_commit": r[14],
        "created_at": r[15], "updated_at": r[16],
        # appended after 288 (see growth_db._ADDED_COLUMNS); tolerate older rows
        "target": (r[17] if len(r) > 17 else "") or "",
        "source": (r[18] if len(r) > 18 else "") or "autonomous",
    }


async def create_attempt(
    conn: Any, *, attempt_id: str, user_id: str, objective: str,
    surface: str = "", tier: str = "green", base_ref: str = "", target: str = "",
    source: str = "autonomous",
) -> None:
    await conn.execute(
        """
        INSERT INTO self_edit_attempts
            (id, user_id, objective, surface, tier, target, status, base_ref,
             source, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'proposed', ?, ?, datetime('now'), datetime('now'))
        """,
        (attempt_id, user_id, objective[:4000], surface, tier, target[:120],
         base_ref, source or "autonomous"),
    )
    await conn.commit()


async def ingest_attempt(
    conn: Any, *, attempt_id: str, user_id: str, objective: str, source: str,
    status: str, surface: str = "", tier: str = "green", target: str = "",
    files_changed: list[str] | None = None, outcome: str = "", lesson: str = "",
    base_ref: str = "", promoted_commit: str = "", created_at: str = "",
) -> bool:
    """Record an already-completed unit of work from OUTSIDE the engine's own
    loop (a git commit, an applied coder turn) as one terminal archive row —
    ingest-all-work. Idempotent: callers pass a deterministic ``attempt_id``
    (e.g. ``git:<sha>``) and an existing row is left untouched (returns False).
    ``created_at`` (ISO) preserves the work's real timestamp so the chronological
    activation fold replays history in true order."""
    cur = await conn.execute(
        "SELECT 1 FROM self_edit_attempts WHERE id=?", (attempt_id,))
    if await cur.fetchone():
        return False
    await conn.execute(
        """
        INSERT INTO self_edit_attempts
            (id, user_id, objective, surface, tier, target, status, base_ref,
             files_changed, outcome, lesson, promoted_commit, source,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                COALESCE(NULLIF(?, ''), datetime('now')), datetime('now'))
        """,
        (attempt_id, user_id, objective[:4000], surface, tier, target[:120],
         status, base_ref, json.dumps(files_changed or []), outcome[:2000],
         lesson[:4000], promoted_commit, source or "ingested", created_at),
    )
    await conn.commit()
    return True


async def set_candidate(
    conn: Any, *, attempt_id: str, user_id: str, candidate_ref: str,
    run_id: str = "", base_ref: str = "",
) -> None:
    """Record the isolated candidate (branch) + the editing run; status→editing."""
    await conn.execute(
        """
        UPDATE self_edit_attempts SET
            candidate_ref=?, run_id=CASE WHEN ?!='' THEN ? ELSE run_id END,
            base_ref=CASE WHEN ?!='' THEN ? ELSE base_ref END,
            status='editing', updated_at=datetime('now')
        WHERE id=? AND user_id=?
        """,
        (candidate_ref, run_id, run_id, base_ref, base_ref, attempt_id, user_id),
    )
    await conn.commit()


async def set_gate(
    conn: Any, *, attempt_id: str, user_id: str, passed: bool, verdict: dict,
    files_changed: list[str] | None = None,
) -> None:
    """Record the fitness-gate verdict; status→gated."""
    await conn.execute(
        """
        UPDATE self_edit_attempts SET
            gate_passed=?, gate_verdict=?, files_changed=?,
            status='gated', updated_at=datetime('now')
        WHERE id=? AND user_id=?
        """,
        (1 if passed else 0, json.dumps(verdict), json.dumps(files_changed or []),
         attempt_id, user_id),
    )
    await conn.commit()


async def finalize(
    conn: Any, *, attempt_id: str, user_id: str, status: str,
    outcome: str = "", lesson: str = "", promoted_commit: str = "",
) -> None:
    """Terminal state: promoted | rejected | rolled_back | failed. ``lesson`` is
    the durable takeaway — kept forever even when the code change is reverted."""
    await conn.execute(
        """
        UPDATE self_edit_attempts SET
            status=?, outcome=?, lesson=CASE WHEN ?!='' THEN ? ELSE lesson END,
            promoted_commit=CASE WHEN ?!='' THEN ? ELSE promoted_commit END,
            updated_at=datetime('now')
        WHERE id=? AND user_id=?
        """,
        (status, outcome[:2000], lesson, lesson[:4000],
         promoted_commit, promoted_commit, attempt_id, user_id),
    )
    await conn.commit()


async def list_attempts(
    conn: Any, *, user_id: str, limit: int = 50,
) -> list[dict]:
    cur = await conn.execute(
        f"SELECT {_COLS} FROM self_edit_attempts WHERE user_id=? "
        "ORDER BY created_at DESC LIMIT ?",
        (user_id, max(1, min(int(limit or 50), 500))),
    )
    return [_row(r) for r in await cur.fetchall()]


async def get_attempt(conn: Any, *, attempt_id: str, user_id: str) -> dict | None:
    cur = await conn.execute(
        f"SELECT {_COLS} FROM self_edit_attempts WHERE id=? AND user_id=?",
        (attempt_id, user_id),
    )
    r = await cur.fetchone()
    return _row(r) if r else None
