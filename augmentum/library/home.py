"""Library home payload builder.

Pulls the four sections the Library dashboard renders when nothing is
selected:

* ``pinned``    — artifacts with ``pinned = 1`` (newest first)
* ``recent``    — distinct artifacts touched in :class:`ActivityStore`
* ``continue``  — interacted but not done (30-day window)
* ``collections_summary`` — counts per user-defined collection so the
  sidebar can render badges without a separate roundtrip

Returned as a flat dict so the client only does one fetch on Library open.
"""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from augmentum.library.activity import ActivityStore
from augmentum.library.collections import CollectionStore


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def _decode_tags(row: dict[str, Any]) -> dict[str, Any]:
    """In-place decode of JSON-encoded ``tags`` and ``metadata`` columns.

    Mirrors ``proxy/library_routes._decode_tags``. The library2 surface
    dispatcher reads ``metadata.kind`` to route ROMs / agentic builds
    to their dedicated stages — see ui/scripts/library2/types.js.
    """
    raw_tags = row.get("tags") or "[]"
    if isinstance(raw_tags, str):
        try:
            row["tags"] = json.loads(raw_tags)
        except json.JSONDecodeError:
            row["tags"] = []
    raw_meta = row.get("metadata")
    if isinstance(raw_meta, str):
        try:
            row["metadata"] = json.loads(raw_meta) if raw_meta else {}
        except json.JSONDecodeError:
            row["metadata"] = {}
    elif raw_meta is None:
        row["metadata"] = {}
    return row


# Publication projection into artifact column shape. Mirrors
# ``_ITEMS_UNION_SQL`` in proxy/library_routes.py — keep the two in sync
# (kind↔format, epoch→ISO, real pinned/tags since mig 309, metadata '{}').
_PUB_PROJECTION = """
SELECT id, title AS display_name, entry_point AS filename, kind AS format,
       size_bytes, pinned,
       CASE WHEN last_launched_at IS NULL THEN NULL
            ELSE strftime('%Y-%m-%d %H:%M:%S', last_launched_at, 'unixepoch')
       END AS last_opened_at,
       strftime('%Y-%m-%d %H:%M:%S', created_at, 'unixepoch') AS created_at,
       tags, '{}' AS metadata
FROM library_publications
"""


async def _fetch_artifacts_by_ids(
    conn: aiosqlite.Connection, *, user_id: str, ids: list[str],
) -> list[dict[str, Any]]:
    """Hydrate library items by id across BOTH tables (artifacts +
    publications), preserving the order of ``ids``. Recent/Continue ids can
    now be publications (mig 309 let them log activity)."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    cursor = await conn.execute(
        f"SELECT id, display_name, filename, format, size_bytes, pinned, "
        f"last_opened_at, created_at, tags, metadata FROM ("
        f"  SELECT id, display_name, filename, format, size_bytes, pinned, "
        f"  last_opened_at, created_at, tags, metadata FROM artifacts "
        f"  WHERE user_id = ? AND COALESCE(transient, 0) = 0 "
        f"  UNION ALL {_PUB_PROJECTION} WHERE user_id = ? "
        f") WHERE id IN ({placeholders})",
        (user_id, user_id, *ids),
    )
    rows = await cursor.fetchall()
    by_id = {r[0]: _decode_tags(_row_to_dict(cursor, r)) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


async def _type_counts(
    conn: aiosqlite.Connection, *, user_id: str,
) -> dict[str, int]:
    """Group non-transient artifacts + library publications by their
    visible ``format``. Drives the auto-type virtual collections in the
    sidebar (Apps/Docs/Games/Imports/...).

    Publications contribute their ``kind`` column (game/app/doc/other)
    into the same bucket namespace so a coder save lands in the Games
    bucket alongside imported games. Mirrors the projection used by
    ``/api/library/items`` — see ``_ITEMS_UNION_SQL`` in
    ``proxy/library_routes.py``.
    """
    # Effective format = artifact.format, or metadata.kind when format is
    # empty (emulator ROMs). Mirrors _ITEMS_UNION_SQL so counts match what a
    # type filter actually returns; without it ROMs (empty format) were
    # dropped by the ``if r[0]`` guard and counted nowhere.
    cursor = await conn.execute(
        "SELECT format, COUNT(*) FROM ("
        "  SELECT COALESCE(NULLIF(format, ''), "
        "         json_extract(metadata, '$.kind'), '') AS format "
        "  FROM artifacts "
        "  WHERE user_id = ? AND COALESCE(transient, 0) = 0 "
        "  UNION ALL "
        "  SELECT kind AS format FROM library_publications "
        "  WHERE user_id = ?"
        ") GROUP BY format",
        (user_id, user_id),
    )
    rows = await cursor.fetchall()
    return {str(r[0]): int(r[1]) for r in rows if r[0]}


async def _pinned_artifacts(
    conn: aiosqlite.Connection, *, user_id: str, limit: int = 12,
) -> list[dict[str, Any]]:
    """Pinned items across BOTH tables — publications became pinnable in
    mig 309, so a favorited coder creation shows in the dashboard too."""
    cursor = await conn.execute(
        "SELECT id, display_name, filename, format, size_bytes, pinned, "
        "last_opened_at, created_at, tags, metadata FROM ("
        "  SELECT id, display_name, filename, format, size_bytes, pinned, "
        "  last_opened_at, created_at, tags, metadata FROM artifacts "
        "  WHERE user_id = ? AND pinned = 1 AND COALESCE(transient, 0) = 0 "
        f"  UNION ALL {_PUB_PROJECTION} WHERE user_id = ? AND pinned = 1 "
        ") ORDER BY COALESCE(last_opened_at, created_at) DESC LIMIT ?",
        (user_id, user_id, int(limit)),
    )
    rows = await cursor.fetchall()
    return [_decode_tags(_row_to_dict(cursor, r)) for r in rows]


async def build_home_payload(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    collection_store: CollectionStore | None = None,
    activity_store: ActivityStore | None = None,
) -> dict[str, Any]:
    """Single-fetch dashboard data. Empty arrays where the user has no
    activity yet (a fresh account hits this on first Library open)."""
    if not user_id:
        return {
            "pinned": [], "recent": [], "continue": [],
            "collections_summary": [], "type_counts": {}, "total_count": 0,
        }

    cs = collection_store or CollectionStore(conn)
    acts = activity_store or ActivityStore(conn)

    pinned = await _pinned_artifacts(conn, user_id=user_id)
    recent_ids = await acts.recent_artifact_ids(user_id=user_id)
    recent = await _fetch_artifacts_by_ids(conn, user_id=user_id, ids=recent_ids)
    cont_ids = await acts.continue_artifact_ids(user_id=user_id)
    cont = await _fetch_artifacts_by_ids(conn, user_id=user_id, ids=cont_ids)

    type_counts = await _type_counts(conn, user_id=user_id)

    collections = await cs.list_for_user(user_id=user_id)
    summary = []
    for col in collections:
        count = await cs.count_items(col["id"], user_id=user_id)
        summary.append({
            "id": col["id"],
            "slug": col["slug"],
            "name": col["name"],
            "kind": col["kind"],
            "view_mode": col["view_mode"],
            "cover_url": col["cover_url"],
            "accent_color": col["accent_color"],
            "sort_order": col["sort_order"],
            "count": count,
        })

    return {
        "pinned": pinned,
        "recent": recent,
        "continue": cont,
        "collections_summary": summary,
        "type_counts": type_counts,
        "total_count": sum(type_counts.values()),
    }
