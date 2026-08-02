"""Persistence for ATP recipes — named, per-user macros over ATP tools.

A recipe is a small JSON program: an ordered list of steps, each a
``{"tool": <atp tool name>, "arguments": {...}}`` where argument strings may
contain ``{{placeholder}}`` tokens filled in at replay time. The store is a
thin CRUD layer over the ``atp_recipes`` table (migration 317); all rows are
user-scoped and every function requires a ``user_id``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

MAX_STEPS = 12
MAX_NAME_LEN = 64


def _conn(app_state: Any):
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


def normalize_name(name: str) -> str:
    return (name or "").strip()[:MAX_NAME_LEN]


def validate_steps(steps: Any) -> tuple[list[dict], str]:
    """Return (clean_steps, error). error is '' on success."""
    if not isinstance(steps, list) or not steps:
        return [], "steps must be a non-empty list"
    if len(steps) > MAX_STEPS:
        return [], f"too many steps (max {MAX_STEPS})"
    clean: list[dict] = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            return [], f"step {i} is not an object"
        tool = (s.get("tool") or "").strip()
        if not tool:
            return [], f"step {i} is missing 'tool'"
        # No recursion: a recipe cannot call the recipe runner.
        if tool == "atp_recipe":
            return [], "a recipe step may not call 'atp_recipe' (no recursion)"
        args = s.get("arguments", {})
        if not isinstance(args, dict):
            return [], f"step {i} 'arguments' must be an object"
        clean.append({"tool": tool, "arguments": args})
    return clean, ""


async def save_recipe(
    app_state: Any, *, user_id: str, name: str, steps: list[dict],
    description: str = "", harness: str = "",
) -> dict[str, Any] | None:
    """Upsert by (user_id, name). Returns the stored recipe row."""
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name:
        return None
    steps_json = json.dumps(steps)
    cur = await conn.execute(
        "SELECT id FROM atp_recipes WHERE user_id = ? AND name = ?",
        [user_id, name],
    )
    row = await cur.fetchone()
    if row is not None:
        await conn.execute(
            "UPDATE atp_recipes SET steps = ?, description = ?, "
            "harness = COALESCE(NULLIF(?, ''), harness), "
            "updated_at = datetime('now') WHERE id = ?",
            [steps_json, description, harness, row[0]],
        )
        recipe_id = row[0]
    else:
        recipe_id = f"rcp_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            "INSERT INTO atp_recipes (id, user_id, name, description, steps, harness) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [recipe_id, user_id, name, description, steps_json, harness],
        )
    await conn.commit()
    return {"id": recipe_id, "name": name, "steps": steps,
            "description": description}


async def get_recipe(
    app_state: Any, *, user_id: str, name: str,
) -> dict[str, Any] | None:
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name:
        return None
    cur = await conn.execute(
        "SELECT id, name, description, steps FROM atp_recipes "
        "WHERE user_id = ? AND name = ?",
        [user_id, name],
    )
    r = await cur.fetchone()
    if r is None:
        return None
    try:
        steps = json.loads(r[3])
    except (ValueError, TypeError):
        steps = []
    return {"id": r[0], "name": r[1], "description": r[2], "steps": steps}


async def list_recipes(app_state: Any, *, user_id: str) -> list[dict[str, Any]]:
    conn = _conn(app_state)
    if conn is None or not user_id:
        return []
    cur = await conn.execute(
        "SELECT name, description, steps, updated_at FROM atp_recipes "
        "WHERE user_id = ? ORDER BY updated_at DESC LIMIT 100",
        [user_id],
    )
    out = []
    for r in await cur.fetchall():
        try:
            n_steps = len(json.loads(r[2]))
        except (ValueError, TypeError):
            n_steps = 0
        out.append({"name": r[0], "description": r[1],
                    "steps": n_steps, "updated_at": r[3]})
    return out


async def delete_recipe(app_state: Any, *, user_id: str, name: str) -> bool:
    conn = _conn(app_state)
    name = normalize_name(name)
    if conn is None or not user_id or not name:
        return False
    await conn.execute(
        "DELETE FROM atp_recipes WHERE user_id = ? AND name = ?",
        [user_id, name],
    )
    await conn.commit()
    cur = await conn.execute(
        "SELECT 1 FROM atp_recipes WHERE user_id = ? AND name = ?",
        [user_id, name],
    )
    return await cur.fetchone() is None
