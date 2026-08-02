"""Library collections store — user-defined groupings of artifacts.

Two kinds:

* ``manual`` — explicit (collection_id, artifact_id) rows. Drag-and-drop
  in the sidebar adds; the row is the truth.
* ``dynamic`` — driven by ``filter_json`` rules (tags, types, since).
  Items are computed at query time via :meth:`resolve_dynamic`. No rows
  in ``library_collection_items`` for dynamic collections.

Slugs are url-safe (``[a-z0-9-]``), unique per user. The store auto-derives
one from ``name`` if the caller doesn't supply it.

Every read is user-scoped. Cross-tenant lookups return None; the route
layer translates to 404 so existence never leaks.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from typing import Any, Literal

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_ID_PREFIX = "col_"
_ID_NONCE_BYTES = 8
_SLUG_RE = re.compile(r"[^a-z0-9]+")

CollectionKind = Literal["manual", "dynamic"]
ViewMode = Literal["list", "grid", "cover"]


class SlugCollision(Exception):
    """Raised when a create/rename would collide on (user_id, slug)."""


@dataclass(frozen=True)
class DynamicFilter:
    """Decoded ``filter_json`` for dynamic collections.

    Only fields that have a sensible non-empty default are JSON-serialised
    back; unknown fields are preserved as-is so future fields don't break
    older clients."""

    tags_any: list[str]
    tags_all: list[str]
    types: list[str]      # e.g. ['app', 'game', 'doc']
    since: str            # ISO timestamp; '' = no lower bound
    pinned_only: bool
    raw: dict[str, Any]   # original dict (for forward compat)

    @classmethod
    def from_json(cls, blob: str) -> DynamicFilter:
        try:
            data = json.loads(blob) if blob else {}
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        return cls(
            tags_any=[str(t) for t in data.get("tags_any", []) if t],
            tags_all=[str(t) for t in data.get("tags_all", []) if t],
            types=[str(t) for t in data.get("types", []) if t],
            since=str(data.get("since") or ""),
            pinned_only=bool(data.get("pinned_only")),
            raw=data,
        )


def _new_collection_id() -> str:
    return _ID_PREFIX + secrets.token_hex(_ID_NONCE_BYTES)


def _slugify(name: str) -> str:
    s = _SLUG_RE.sub("-", name.lower()).strip("-")
    return s[:60] or "untitled"


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


class CollectionStore:
    """CRUD over ``library_collections`` + ``library_collection_items``."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Reads ──────────────────────────────────────────────────────────

    async def get(self, collection_id: str, *, user_id: str) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT * FROM library_collections WHERE id = ? AND user_id = ?",
            (collection_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def get_by_slug(self, *, user_id: str, slug: str) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT * FROM library_collections WHERE user_id = ? AND slug = ?",
            (user_id, slug),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_for_user(self, *, user_id: str) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM library_collections WHERE user_id = ? "
            "ORDER BY sort_order, created_at",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def count_items(self, collection_id: str, *, user_id: str) -> int:
        col = await self.get(collection_id, user_id=user_id)
        if not col:
            return 0
        if col["kind"] == "dynamic":
            ids = await self.resolve_dynamic(collection_id, user_id=user_id)
            return len(ids)
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM library_collection_items "
            "WHERE collection_id = ? AND user_id = ?",
            (collection_id, user_id),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def list_items(
        self, collection_id: str, *, user_id: str, limit: int = 500,
    ) -> list[str]:
        """Return artifact_ids in this collection, in display order."""
        col = await self.get(collection_id, user_id=user_id)
        if not col:
            return []
        if col["kind"] == "dynamic":
            ids = await self.resolve_dynamic(collection_id, user_id=user_id)
            return ids[: int(limit)]
        cursor = await self._conn.execute(
            "SELECT artifact_id FROM library_collection_items "
            "WHERE collection_id = ? AND user_id = ? "
            "ORDER BY sort_order, added_at LIMIT ?",
            (collection_id, user_id, int(limit)),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def resolve_dynamic(
        self, collection_id: str, *, user_id: str, limit: int = 500,
    ) -> list[str]:
        """Evaluate a dynamic collection's filter across BOTH backing tables.

        Returns item ids (artifacts + pub_ publications) matching the rules,
        ordered by most-recent-touch DESC so newest matches surface first.
        Tags filter uses JSON_EACH which SQLite handles natively. Empty
        filter = no rows (refuse to return everything by accident).

        Publications reached parity in mig 309, so a dynamic collection like
        "all games" or "type=app" now includes them — matching what the main
        list (/api/library/items) shows for the same filter."""
        col = await self.get(collection_id, user_id=user_id)
        if not col or col["kind"] != "dynamic":
            return []
        rules = DynamicFilter.from_json(col["filter_json"])

        # An empty filter is operator error — return [] rather than
        # exposing every item under a misconfigured collection.
        if not (
            rules.tags_any or rules.tags_all or rules.types
            or rules.since or rules.pinned_only
        ):
            return []

        # ── Artifacts leg. sort_ts is the ISO "recently touched" key. ──
        a_sql = [
            "SELECT id, COALESCE(last_opened_at, created_at) AS sort_ts "
            "FROM artifacts WHERE user_id = ? AND COALESCE(transient,0) = 0"
        ]
        a_params: list[Any] = [user_id]
        if rules.types:
            # Effective format (format, or metadata.kind when empty) so a
            # types=['emulator_rom'] rule matches ROMs — mirrors the main list.
            a_sql.append(
                "AND COALESCE(NULLIF(format, ''), "
                f"json_extract(metadata, '$.kind'), '') IN ({','.join('?' * len(rules.types))})"
            )
            a_params.extend(rules.types)
        if rules.pinned_only:
            a_sql.append("AND pinned = 1")
        if rules.since:
            a_sql.append("AND COALESCE(last_opened_at, created_at) >= ?")
            a_params.append(rules.since)
        if rules.tags_any:
            a_sql.append(
                "AND EXISTS (SELECT 1 FROM json_each(artifacts.tags) "
                f"WHERE json_each.value IN ({','.join('?' * len(rules.tags_any))}))"
            )
            a_params.extend(rules.tags_any)
        for t in rules.tags_all:
            a_sql.append(
                "AND EXISTS (SELECT 1 FROM json_each(artifacts.tags) "
                "WHERE json_each.value = ?)"
            )
            a_params.append(t)

        # ── Publications leg. kind↔format; epoch timestamps → ISO so they
        #    sort and compare against artifacts' text timestamps. ──
        p_ts = ("strftime('%Y-%m-%d %H:%M:%S', "
                "COALESCE(last_launched_at, created_at), 'unixepoch')")
        p_sql = [
            f"SELECT id, {p_ts} AS sort_ts "
            "FROM library_publications WHERE user_id = ?"
        ]
        p_params: list[Any] = [user_id]
        if rules.types:
            p_sql.append(f"AND kind IN ({','.join('?' * len(rules.types))})")
            p_params.extend(rules.types)
        if rules.pinned_only:
            p_sql.append("AND pinned = 1")
        if rules.since:
            p_sql.append(f"AND {p_ts} >= ?")
            p_params.append(rules.since)
        if rules.tags_any:
            p_sql.append(
                "AND EXISTS (SELECT 1 FROM json_each(library_publications.tags) "
                f"WHERE json_each.value IN ({','.join('?' * len(rules.tags_any))}))"
            )
            p_params.extend(rules.tags_any)
        for t in rules.tags_all:
            p_sql.append(
                "AND EXISTS (SELECT 1 FROM json_each(library_publications.tags) "
                "WHERE json_each.value = ?)"
            )
            p_params.append(t)

        # No parens around the individual compound operands — SQLite rejects
        # ``(SELECT ...) UNION ALL (SELECT ...)``. Wrap the whole compound once
        # as a FROM-subquery so the outer ORDER BY/LIMIT applies across both.
        union_sql = (
            f"SELECT id FROM ({' '.join(a_sql)} UNION ALL {' '.join(p_sql)}) "
            "ORDER BY sort_ts DESC LIMIT ?"
        )
        params = [*a_params, *p_params, int(limit)]
        cursor = await self._conn.execute(union_sql, tuple(params))
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    # ── Writes ─────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        kind: CollectionKind = "manual",
        slug: str = "",
        filter_json: dict[str, Any] | None = None,
        cover_url: str = "",
        accent_color: str = "",
        view_mode: ViewMode = "list",
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("create requires user_id")
        if not name.strip():
            raise ValueError("create requires name")
        if kind not in ("manual", "dynamic"):
            raise ValueError(f"invalid kind: {kind}")

        resolved_slug = (slug or _slugify(name)).strip()
        if not resolved_slug:
            resolved_slug = "untitled"

        existing = await self.get_by_slug(user_id=user_id, slug=resolved_slug)
        if existing:
            raise SlugCollision(
                f"slug already in use: {resolved_slug!r}"
            )

        col_id = _new_collection_id()
        filter_blob = json.dumps(filter_json or {})

        # Sort_order: append to end of user's list. Cheap aggregate.
        cursor = await self._conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 "
            "FROM library_collections WHERE user_id = ?",
            (user_id,),
        )
        next_order = (await cursor.fetchone())[0] or 0

        await self._conn.execute(
            "INSERT INTO library_collections "
            "(id, user_id, name, slug, kind, filter_json, cover_url, "
            " accent_color, view_mode, sort_order) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                col_id, user_id, name.strip(), resolved_slug, kind,
                filter_blob, cover_url, accent_color, view_mode, int(next_order),
            ),
        )
        await self._conn.commit()
        return await self.get(col_id, user_id=user_id) or {}

    async def update(
        self,
        collection_id: str,
        *,
        user_id: str,
        name: str | None = None,
        slug: str | None = None,
        filter_json: dict[str, Any] | None = None,
        cover_url: str | None = None,
        accent_color: str | None = None,
        view_mode: ViewMode | None = None,
        sort_order: int | None = None,
    ) -> dict[str, Any] | None:
        existing = await self.get(collection_id, user_id=user_id)
        if not existing:
            return None

        fields: list[str] = []
        params: list[Any] = []
        if name is not None and name.strip():
            fields.append("name = ?")
            params.append(name.strip())
        if slug is not None:
            new_slug = slug.strip() or _slugify(name or existing["name"])
            if new_slug != existing["slug"]:
                conflict = await self.get_by_slug(user_id=user_id, slug=new_slug)
                if conflict and conflict["id"] != collection_id:
                    raise SlugCollision(f"slug already in use: {new_slug!r}")
            fields.append("slug = ?")
            params.append(new_slug)
        if filter_json is not None:
            fields.append("filter_json = ?")
            params.append(json.dumps(filter_json))
        if cover_url is not None:
            fields.append("cover_url = ?")
            params.append(cover_url)
        if accent_color is not None:
            fields.append("accent_color = ?")
            params.append(accent_color)
        if view_mode is not None:
            fields.append("view_mode = ?")
            params.append(view_mode)
        if sort_order is not None:
            fields.append("sort_order = ?")
            params.append(int(sort_order))

        if not fields:
            return existing

        fields.append("updated_at = datetime('now')")
        params.extend([collection_id, user_id])
        await self._conn.execute(
            f"UPDATE library_collections SET {', '.join(fields)} "
            "WHERE id = ? AND user_id = ?",
            tuple(params),
        )
        await self._conn.commit()
        return await self.get(collection_id, user_id=user_id)

    async def delete(self, collection_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM library_collections WHERE id = ? AND user_id = ?",
            (collection_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def add_items(
        self,
        collection_id: str,
        *,
        user_id: str,
        artifact_ids: list[str],
    ) -> int:
        """Add artifacts to a manual collection. Idempotent via INSERT OR
        IGNORE. Returns count of rows actually inserted. Dynamic
        collections reject (their membership is rule-driven)."""
        col = await self.get(collection_id, user_id=user_id)
        if not col:
            return 0
        if col["kind"] != "manual":
            raise ValueError("cannot add items to a dynamic collection")
        if not artifact_ids:
            return 0

        # Reject items the caller doesn't own. Collections hold union ids
        # since mig 309 (artifacts + pub_ publications), so verify ownership
        # against BOTH tables — the FK no longer does it for us.
        placeholders = ",".join("?" * len(artifact_ids))
        cursor = await self._conn.execute(
            f"SELECT id FROM artifacts "
            f"WHERE id IN ({placeholders}) AND user_id = ? "
            f"UNION "
            f"SELECT id FROM library_publications "
            f"WHERE id IN ({placeholders}) AND user_id = ?",
            (*artifact_ids, user_id, *artifact_ids, user_id),
        )
        rows = await cursor.fetchall()
        owned = {r[0] for r in rows}
        if not owned:
            return 0

        # Append at the bottom.
        cursor = await self._conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) + 1 "
            "FROM library_collection_items WHERE collection_id = ?",
            (collection_id,),
        )
        next_order = (await cursor.fetchone())[0] or 0

        inserted = 0
        for aid in artifact_ids:
            if aid not in owned:
                continue
            try:
                cur = await self._conn.execute(
                    "INSERT OR IGNORE INTO library_collection_items "
                    "(collection_id, artifact_id, user_id, sort_order) "
                    "VALUES (?, ?, ?, ?)",
                    (collection_id, aid, user_id, int(next_order)),
                )
                if cur.rowcount:
                    inserted += 1
                    next_order += 1
            except aiosqlite.Error as e:
                log.warning(
                    "collection_add_item_failed",
                    collection_id=collection_id, artifact_id=aid, error=str(e),
                )
        await self._conn.commit()
        return inserted

    async def remove_item(
        self, collection_id: str, artifact_id: str, *, user_id: str,
    ) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM library_collection_items "
            "WHERE collection_id = ? AND artifact_id = ? AND user_id = ?",
            (collection_id, artifact_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0
