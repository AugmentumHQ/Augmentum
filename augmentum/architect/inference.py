"""Inference layer — fill missing action args from observation history.

This is the novel UX piece. When the user says "play jazz" the matcher
extracts ``{query: "jazz"}`` but grove.play_matching needs a source +
track_id. Instead of asking, the action's ``arg_inferrer`` queries
``device_play_history`` for jazz-tagged favourites and fills them in.

The inference layer is per-action — each Action that benefits declares
its own ``arg_inferrer`` callable. Common patterns live here as helpers
so individual inferrers stay short:

  - ``query_play_history`` — favourites + recency lookup
  - ``query_image_history`` — last model/sampler/settings used
  - ``query_browse_history`` — recent topics
  - ``pull_referent`` — pull from ReferentCache

Inferrers MUST be user-scoped — they receive ``SessionContext`` with
``user_id`` and pass it into every query. Empty user_id (anon) returns
empty results rather than leaking cross-tenant rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.intent.action import Action, SessionContext

log = get_logger(__name__)


async def infer_args(
    action: "Action",
    partial_args: dict[str, Any],
    session: "SessionContext",
    runtime: Any,
) -> dict[str, Any]:
    """Run the action's arg_inferrer if defined, else return partial.

    ``runtime`` is the CompanionRuntime (or None) — inferrers use it to
    reach the bus, the observation deque, and any companion-specific
    state. None-tolerant: inferrers that don't need runtime ignore it.

    Failures inside an inferrer are caught and logged at WARNING — the
    handler then sees the unmodified partial_args and either fills its
    own defaults or returns a clarifying ActionResult. Inference is
    best-effort enrichment, never a hard requirement.
    """
    if action.arg_inferrer is None:
        return partial_args

    try:
        filled = await action.arg_inferrer(partial_args, session, runtime)
    except Exception as exc:  # noqa: BLE001 — log and degrade gracefully
        log.warning(
            "architect_inference_failed",
            action_id=action.id,
            error=str(exc)[:200],
        )
        return partial_args

    if not isinstance(filled, dict):
        log.warning(
            "architect_inference_bad_return",
            action_id=action.id,
            got_type=type(filled).__name__,
        )
        return partial_args

    return filled


# ---------------------------------------------------------------------------
# Reusable inference helpers — query patterns shared across inferrers.
# Every helper accepts conn (aiosqlite) and user_id; returns concrete rows
# or a sensible empty default. None-safe on conn (no runtime up).
# ---------------------------------------------------------------------------


async def query_play_history(
    conn: Any,
    user_id: str,
    *,
    content_kind: str = "",
    capability_filter: str = "",
    limit: int = 20,
    favourites_first: bool = True,
) -> list[dict[str, Any]]:
    """Pull recent + favourite entries from device_play_history.

    Returns list of {capability_id, file_id, content_key, action,
    is_favorite, created_at, payload} dicts. Empty list on missing conn
    / empty user_id / no rows.

    ``content_kind`` filters by the indexed content_kind column.
    ``capability_filter`` matches capability_id with LIKE %X%.
    ``favourites_first`` orders is_favorite DESC then created_at DESC.
    """
    if conn is None or not user_id:
        return []

    where = ["user_id = ?"]
    params: list[Any] = [user_id]

    if content_kind:
        where.append("content_kind = ?")
        params.append(content_kind)
    if capability_filter:
        where.append("capability_id LIKE ?")
        params.append(f"%{capability_filter}%")

    order = "is_favorite DESC, created_at DESC" if favourites_first else "created_at DESC"

    sql = (
        "SELECT capability_id, file_id, content_key, content_label, "
        "action, is_favorite, created_at, extra "
        "FROM device_play_history "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY {order} LIMIT ?"
    )
    params.append(limit)

    try:
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
    except Exception as exc:  # noqa: BLE001 — table missing or query error
        log.warning("inference_play_history_query_failed", error=str(exc)[:200])
        return []

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "capability_id": r[0],
            "file_id": r[1],
            "content_key": r[2],
            "content_label": r[3],
            "action": r[4],
            "is_favorite": bool(r[5]),
            "created_at": r[6],
            "extra": r[7],
        })
    return out


async def query_image_history(
    conn: Any,
    user_id: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Pull recent image_generations rows — used to infer model + settings
    defaults for "generate a cat" style commands.

    Returns list of {image_id, model, prompt, negative_prompt, width,
    height, steps, cfg_scale, preset, loras, created_at} dicts.
    """
    if conn is None or not user_id:
        return []

    try:
        cursor = await conn.execute(
            "SELECT image_id, model, prompt, negative_prompt, width, "
            "height, steps, cfg_scale, preset, loras, created_at "
            "FROM image_generations "
            "WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        log.warning("inference_image_history_query_failed", error=str(exc)[:200])
        return []

    return [
        {
            "image_id": r[0],
            "model": r[1],
            "prompt": r[2],
            "negative_prompt": r[3],
            "width": r[4],
            "height": r[5],
            "steps": r[6],
            "cfg_scale": r[7],
            "preset": r[8],
            "loras": r[9],
            "created_at": r[10],
        }
        for r in rows
    ]


async def query_browse_history(
    conn: Any,
    user_id: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Pull recent browse_history rows — used to infer topical context
    for "open the article I was just reading" style commands.
    """
    if conn is None or not user_id:
        return []

    try:
        cursor = await conn.execute(
            "SELECT url, domain, visit_count, last_visited "
            "FROM browse_history "
            "WHERE user_id = ? "
            "ORDER BY last_visited DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
    except Exception as exc:  # noqa: BLE001
        log.warning("inference_browse_history_query_failed", error=str(exc)[:200])
        return []

    return [
        {
            "url": r[0],
            "domain": r[1],
            "visit_count": r[2],
            "last_visited": r[3],
        }
        for r in rows
    ]


def pull_referent(session: "SessionContext", field: str) -> str | None:
    """Pull a ReferentCache field if available. Convenience wrapper that
    None-checks the cache so individual inferrers don't have to.

    Common fields: 'last_image_id', 'last_url', 'last_quote',
    'last_file_id', 'last_entity', 'active_note_id'.
    """
    refs = getattr(session, "referents", None)
    if refs is None:
        return None
    return getattr(refs, field, None)
