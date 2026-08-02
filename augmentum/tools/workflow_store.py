"""Persistence for ATP workflows — self-minted soft procedural memory.

A workflow is a natural-language playbook (a ``when_to_use`` trigger +
numbered steps + description) that the model mints when something worked and
refines over time. Retrieval is FTS5 on the trigger text so the right
workflow surfaces for a new-but-similar task. All rows are user-scoped AND
scope-isolated (harness:project), reusing the same isolation as harness
memory. See workflow_tool.py for the tool surface and harness.py for the
briefing injection that auto-surfaces a match before tool-calling.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

MAX_NAME_LEN = 64
MAX_STEPS_CHARS = 4000
MAX_TRIGGER_CHARS = 400
_FTS_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _conn(app_state: Any):
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


def normalize_name(name: str) -> str:
    return (name or "").strip()[:MAX_NAME_LEN]


def _fts_query(text: str) -> str:
    """Build a safe FTS5 MATCH expression: OR the alnum tokens. Returns ''
    when there's nothing searchable (caller should skip the FTS leg)."""
    toks = _FTS_TOKEN.findall(text or "")
    toks = [t for t in toks if len(t) > 2][:24]
    if not toks:
        return ""
    return " OR ".join(toks)


async def save_workflow(
    app_state: Any, *, user_id: str, scope: str, name: str, when_to_use: str,
    steps: str, description: str = "", harness: str = "",
) -> dict[str, Any] | None:
    """Upsert by (user_id, scope, name). On update, bump version and keep
    the accumulated outcome stats. Returns the stored row summary."""
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name or not (when_to_use or "").strip():
        return None
    steps = (steps or "").strip()[:MAX_STEPS_CHARS]
    when_to_use = (when_to_use or "").strip()[:MAX_TRIGGER_CHARS]
    cur = await conn.execute(
        "SELECT id, version FROM atp_workflows WHERE user_id = ? AND scope = ? AND name = ?",
        [user_id, scope, name],
    )
    row = await cur.fetchone()
    if row is not None:
        version = int(row[1] or 1) + 1
        await conn.execute(
            "UPDATE atp_workflows SET when_to_use = ?, description = ?, steps = ?, "
            "version = ?, harness = COALESCE(NULLIF(?, ''), harness), "
            "updated_at = datetime('now') WHERE id = ?",
            [when_to_use, description, steps, version, harness, row[0]],
        )
        wid = row[0]
    else:
        version = 1
        wid = f"wf_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            "INSERT INTO atp_workflows "
            "(id, user_id, scope, name, when_to_use, description, steps, harness) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [wid, user_id, scope, name, when_to_use, description, steps, harness],
        )
    await conn.commit()
    return {"id": wid, "name": name, "scope": scope, "version": version,
            "when_to_use": when_to_use}


async def search_workflows(
    app_state: Any, *, user_id: str, scopes: list[str], query: str, limit: int = 3,
) -> list[dict[str, Any]]:
    """FTS5 retrieval keyed on the trigger text, restricted to the user and
    the given scope(s). Ordered by FTS relevance (bm25)."""
    conn = _conn(app_state)
    if conn is None or not user_id or not scopes:
        return []
    match = _fts_query(query)
    if not match:
        return []
    placeholders = ",".join("?" for _ in scopes)
    sql = (
        "SELECT w.name, w.when_to_use, w.description, w.steps, w.version, "
        "w.times_used, w.times_succeeded, w.scope "
        "FROM atp_workflows_fts f JOIN atp_workflows w ON w.rowid = f.rowid "
        f"WHERE atp_workflows_fts MATCH ? AND w.user_id = ? AND w.scope IN ({placeholders}) "
        "ORDER BY bm25(atp_workflows_fts) LIMIT ?"
    )
    try:
        cur = await conn.execute(sql, [match, user_id, *scopes, int(limit)])
        rows = await cur.fetchall()
    except Exception:
        log.warning("workflow_fts_search_failed", exc_info=True)
        return []
    return [
        {"name": r[0], "when_to_use": r[1], "description": r[2], "steps": r[3],
         "version": r[4], "times_used": r[5], "times_succeeded": r[6], "scope": r[7]}
        for r in rows
    ]


async def list_workflows(
    app_state: Any, *, user_id: str, scopes: list[str],
) -> list[dict[str, Any]]:
    conn = _conn(app_state)
    if conn is None or not user_id or not scopes:
        return []
    placeholders = ",".join("?" for _ in scopes)
    cur = await conn.execute(
        "SELECT name, when_to_use, version, times_used, times_succeeded, scope, updated_at "
        f"FROM atp_workflows WHERE user_id = ? AND scope IN ({placeholders}) "
        "ORDER BY updated_at DESC LIMIT 100",
        [user_id, *scopes],
    )
    return [
        {"name": r[0], "when_to_use": r[1], "version": r[2], "times_used": r[3],
         "times_succeeded": r[4], "scope": r[5], "updated_at": r[6]}
        for r in await cur.fetchall()
    ]


async def get_workflow(
    app_state: Any, *, user_id: str, scope: str, name: str,
) -> dict[str, Any] | None:
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name:
        return None
    cur = await conn.execute(
        "SELECT name, when_to_use, description, steps, version, times_used, "
        "times_succeeded, scope FROM atp_workflows "
        "WHERE user_id = ? AND scope = ? AND name = ?",
        [user_id, scope, name],
    )
    r = await cur.fetchone()
    if r is None:
        return None
    return {"name": r[0], "when_to_use": r[1], "description": r[2], "steps": r[3],
            "version": r[4], "times_used": r[5], "times_succeeded": r[6], "scope": r[7]}


async def delete_workflow(
    app_state: Any, *, user_id: str, scope: str, name: str,
) -> bool:
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name:
        return False
    await conn.execute(
        "DELETE FROM atp_workflows WHERE user_id = ? AND scope = ? AND name = ?",
        [user_id, scope, name],
    )
    await conn.commit()
    cur = await conn.execute(
        "SELECT 1 FROM atp_workflows WHERE user_id = ? AND scope = ? AND name = ?",
        [user_id, scope, name],
    )
    return await cur.fetchone() is None


async def record_outcome(
    app_state: Any, *, user_id: str, scope: str, name: str, success: bool,
) -> bool:
    """Bump usage stats — the curator/model uses these to decide which
    workflows to trust or rewrite."""
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name:
        return False
    await conn.execute(
        "UPDATE atp_workflows SET times_used = times_used + 1, "
        "times_succeeded = times_succeeded + ? WHERE user_id = ? AND scope = ? AND name = ?",
        [1 if success else 0, user_id, scope, name],
    )
    await conn.commit()
    return True
