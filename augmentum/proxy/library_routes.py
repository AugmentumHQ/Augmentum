"""Library REST routes — Steam-style three-pane Library.

Distinct from ``library_save_routes`` (which owns the save-to-library
write path from coder previews) and from ``cast_routes`` (which serves
the cast-receiver library home). This module exposes the *user-facing*
library: collections, pins, activity, dashboard.

Routes:

* ``GET    /api/library/home``                           — dashboard payload
* ``GET    /api/library/collections``                    — list user collections
* ``POST   /api/library/collections``                    — create
* ``GET    /api/library/collections/{id}``               — one + items
* ``PUT    /api/library/collections/{id}``               — rename / recolor / rules
* ``DELETE /api/library/collections/{id}``
* ``POST   /api/library/collections/{id}/items``         — add artifact_ids
* ``DELETE /api/library/collections/{id}/items/{aid}``   — remove
* ``POST   /api/library/items/{artifact_id}/pin``        — set pinned flag
* ``POST   /api/library/items/{artifact_id}/activity``   — append timeline event
* ``GET    /api/library/items/{artifact_id}/activity``   — fetch timeline
* ``PUT    /api/library/items/{artifact_id}/tags``       — replace tag set

Every route is user-scoped via ``request.scope["user"].id``. Cross-tenant
reads return 404; never leak existence.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import aiosqlite
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from augmentum.library.activity import ActivityStore
from augmentum.library.collections import CollectionStore, SlugCollision
from augmentum.library.home import build_home_payload
from augmentum.library.ids import is_publication_id
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(tags=["library"])


# ── Helpers ───────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _conn(request: Request) -> aiosqlite.Connection | None:
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return getattr(backend, "conn", None)


def _require_auth(request: Request) -> str:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    return uid


def _require_conn(request: Request) -> aiosqlite.Connection:
    conn = _conn(request)
    if conn is None:
        raise HTTPException(status_code=503, detail="state not initialized")
    return conn


def _decode_tags(row: dict[str, Any]) -> dict[str, Any]:
    """In-place decode of JSON-encoded ``tags`` and ``metadata`` columns.

    Both arrive as TEXT in SQLite — tags as a JSON array, metadata as a
    JSON object stamped by the producing surface (titles sources mark
    ROMs with ``{kind: emulator_rom, system: gb}``, agentic builds
    stamp ``{kind: app_build}``, etc.). The library2 surface dispatcher
    (``classifyItem`` / ``_openItem``) reads ``metadata.kind`` to route
    ROMs to the emulator stage and games to the game-surface; without
    decoding here the client would have to JSON.parse on every render.
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


async def _artifact_row(
    conn: aiosqlite.Connection, *, artifact_id: str, user_id: str,
) -> dict[str, Any] | None:
    cursor = await conn.execute(
        "SELECT id, display_name, filename, format, size_bytes, pinned, "
        "last_opened_at, created_at, tags, metadata "
        "FROM artifacts WHERE id = ? AND user_id = ? "
        "  AND COALESCE(transient, 0) = 0",
        (artifact_id, user_id),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cursor.description]
    return _decode_tags(dict(zip(cols, row)))


def _pub_store(request: Request):
    """The PublicationStore, or None if the library subsystem isn't up."""
    return getattr(request.app.state, "publication_store", None)


async def _record_item_activity_safe(
    conn: aiosqlite.Connection, *, user_id: str, item_id: str, action: str,
    surface: str = "desktop", payload: dict[str, Any] | None = None,
) -> str | None:
    """Best-effort activity write for EITHER namespace. Never raises — the
    timeline is a nicety, not a gate. ActivityStore.record validates
    ownership against artifacts OR library_publications (mig 309 dropped the
    artifacts-only FK, and record() is namespace-aware)."""
    try:
        return await ActivityStore(conn).record(
            user_id=user_id, artifact_id=item_id,
            action=action, surface=surface, payload=payload,
        )
    except (PermissionError, ValueError) as e:
        log.warning("activity_record_failed",
                    item_id=item_id, action=action, error=str(e))
        return None


async def _item_exists(
    request: Request, *, item_id: str, user_id: str,
) -> bool:
    """True when ``item_id`` addresses a row the user owns — in EITHER
    namespace (artifacts or library_publications). Used by the per-item
    routes to 404 consistently regardless of backing table."""
    if is_publication_id(item_id):
        store = _pub_store(request)
        if store is None:
            return False
        return bool(await store.get(item_id, user_id=user_id))
    conn = _require_conn(request)
    row = await _artifact_row(conn, artifact_id=item_id, user_id=user_id)
    return row is not None


async def _hydrate_union_by_ids(
    conn: aiosqlite.Connection, *, user_id: str, ids: list[str],
) -> list[dict[str, Any]]:
    """Hydrate library items by id across BOTH backing tables, preserving
    the order of ``ids``. Runs the same UNION projection as
    :func:`list_items` so callers (collection hydration, dashboard
    sections) get a uniform item shape whether the id is an artifact or a
    ``pub_`` publication."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    cursor = await conn.execute(
        f"SELECT id, display_name, filename, format, size_bytes, pinned, "
        f"last_opened_at, created_at, tags, metadata "
        f"FROM ({_ITEMS_UNION_SQL}) WHERE id IN ({placeholders})",
        (user_id, user_id, *ids),
    )
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    by_id = {r[0]: _decode_tags(dict(zip(cols, r))) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


# ── Models ────────────────────────────────────────────────────────────


class CreateCollectionReq(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = ""
    kind: Literal["manual", "dynamic"] = "manual"
    filter_json: dict[str, Any] | None = None
    cover_url: str = ""
    accent_color: str = ""
    view_mode: Literal["list", "grid", "cover"] = "list"


class UpdateCollectionReq(BaseModel):
    name: str | None = None
    slug: str | None = None
    filter_json: dict[str, Any] | None = None
    cover_url: str | None = None
    accent_color: str | None = None
    view_mode: Literal["list", "grid", "cover"] | None = None
    sort_order: int | None = None


class AddItemsReq(BaseModel):
    artifact_ids: list[str] = Field(default_factory=list)


class PinReq(BaseModel):
    pinned: bool


class ActivityReq(BaseModel):
    action: Literal["open", "cast", "edit", "pin", "unpin", "tag"]
    surface: Literal["desktop", "mobile", "tv", "cast"] = "desktop"
    payload: dict[str, Any] | None = None


class TagsReq(BaseModel):
    tags: list[str] = Field(default_factory=list)


# ── Home ──────────────────────────────────────────────────────────────


@router.get("/api/library/home")
async def library_home(request: Request) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    payload = await build_home_payload(conn, user_id=uid)
    return JSONResponse(payload)


# ── Items list (main pane source of truth) ───────────────────────────


# Sort keys reference columns present on BOTH sides of the union below
# (artifacts + library_publications projected into artifact shape).
# Tiebreaker is ``id`` rather than ``rowid`` because rowid isn't
# meaningful across a UNION ALL subquery.
_VALID_SORTS = {
    "recent": "COALESCE(last_opened_at, created_at) DESC, id DESC",
    "name":   "LOWER(display_name) ASC, id DESC",
    "pinned": "pinned DESC, COALESCE(last_opened_at, created_at) DESC, id DESC",
    "size":   "size_bytes DESC, id DESC",
    "oldest": "COALESCE(last_opened_at, created_at) ASC, id ASC",
}


# UNION ALL of (artifacts non-transient, library_publications). The
# publications side projects into the artifact column shape so the
# outer WHERE / ORDER BY in :func:`list_items` and :func:`_type_counts`
# (home.py) don't need to know there are two backing tables. Mapping:
#
#   display_name   <- title
#   filename       <- entry_point
#   format         <- kind  (game / app / doc / other)
#   pinned         <- 0     (publications can't be pinned in v1; the
#                            pin/tags/activity routes operate on
#                            ``artifacts`` and 404 for pub_* ids)
#   tags           <- '[]'
#   metadata       <- '{}'  (publications don't have an arbitrary
#                            metadata JSON; the surface dispatcher in
#                            library2 falls back to ``format`` when
#                            metadata.kind is absent — see
#                            ui/scripts/library2/types.js::classifyItem)
#   last_opened_at <- last_launched_at  (epoch → ISO text)
#   created_at     <- created_at        (epoch → ISO text)
#
# The two epoch-as-REAL columns are converted to ISO TEXT via
# ``strftime(..., 'unixepoch')`` so they sort correctly against
# artifacts' TEXT ``datetime('now')`` values in a mixed UNION (SQLite
# would otherwise apply numeric vs text affinity to each side and the
# ORDER BY would be effectively random).
_ITEMS_UNION_SQL = """
SELECT id, display_name, filename,
       COALESCE(NULLIF(format, ''), json_extract(metadata, '$.kind'), '') AS format,
       size_bytes, pinned,
       last_opened_at, created_at, tags, metadata
FROM artifacts
WHERE user_id = ? AND COALESCE(transient, 0) = 0
UNION ALL
SELECT id,
       title AS display_name,
       entry_point AS filename,
       kind AS format,
       size_bytes,
       pinned,
       CASE WHEN last_launched_at IS NULL THEN NULL
            ELSE strftime('%Y-%m-%d %H:%M:%S', last_launched_at, 'unixepoch')
       END AS last_opened_at,
       strftime('%Y-%m-%d %H:%M:%S', created_at, 'unixepoch') AS created_at,
       tags,
       '{}' AS metadata
FROM library_publications
WHERE user_id = ?
"""
# ``pinned`` + ``tags`` are real columns since migration 309 (publications
# reached parity with artifacts). ``metadata`` stays '{}' — publications carry
# no arbitrary metadata JSON; the cover renderer builds their screenshot URL
# from the pub_ id + the assets route, and classifyItem falls back to
# ``format`` when metadata.kind is absent.


@router.get("/api/library/items")
async def list_items(
    request: Request,
    types: str = "",
    q: str = "",
    pinned: int = 0,
    sort: str = "recent",
    limit: int = 60,
    offset: int = 0,
) -> JSONResponse:
    """List the user's non-transient artifacts + library publications
    with optional filters.

    Params:
      types   comma-separated format list (e.g. "html,htm,zip,game")
      q       LIKE-search on display_name / filename / tags
      pinned  1 = restrict to pinned only (publications excluded —
              their projected ``pinned`` is always 0)
      sort    one of ``_VALID_SORTS`` keys
      limit   max rows (capped at 200)
      offset  pagination offset

    Returns ``{items, total, has_more}``. ``total`` is the unbounded
    match count (so paginators can show "60 of 247"); ``has_more`` is
    a cheap boolean for infinite-scroll callers that don't want to
    track offset arithmetic themselves.

    Publications (kind=game/app/doc/other from ``library_publications``)
    are merged into the same listing — see ``_ITEMS_UNION_SQL`` above
    for the projection contract.
    """
    uid = _require_auth(request)
    conn = _require_conn(request)

    sort_key = sort if sort in _VALID_SORTS else "recent"
    limit_n = max(1, min(200, int(limit)))
    offset_n = max(0, int(offset))

    # Outer filters apply to the unioned view, not the underlying tables,
    # so the union subquery doesn't need to be duplicated per predicate.
    outer_where: list[str] = []
    outer_params: list[Any] = []

    formats = [f.strip().lower() for f in types.split(",") if f.strip()]
    if formats:
        placeholders = ",".join("?" * len(formats))
        outer_where.append(f"format IN ({placeholders})")
        outer_params.extend(formats)

    if pinned:
        outer_where.append("pinned = 1")

    qs = q.strip()
    if qs:
        # LIKE-search over display_name + filename + tags JSON blob. Cheap
        # and good enough until artifacts get an FTS5 mirror. Wildcarded
        # both sides so partial matches in the middle still hit.
        like = f"%{qs}%"
        outer_where.append(
            "(display_name LIKE ? OR filename LIKE ? OR tags LIKE ?)"
        )
        outer_params.extend([like, like, like])

    outer_where_sql = (
        " WHERE " + " AND ".join(outer_where) if outer_where else ""
    )

    # uid appears twice — once per side of the UNION ALL.
    union_params = (uid, uid)

    cursor = await conn.execute(
        f"SELECT COUNT(*) FROM ({_ITEMS_UNION_SQL}){outer_where_sql}",
        (*union_params, *outer_params),
    )
    total = int((await cursor.fetchone())[0])

    cursor = await conn.execute(
        f"SELECT id, display_name, filename, format, size_bytes, pinned, "
        f"last_opened_at, created_at, tags, metadata "
        f"FROM ({_ITEMS_UNION_SQL}){outer_where_sql} "
        f"ORDER BY {_VALID_SORTS[sort_key]} LIMIT ? OFFSET ?",
        (*union_params, *outer_params, limit_n, offset_n),
    )
    rows = await cursor.fetchall()
    cols_meta = [d[0] for d in cursor.description]
    items = [
        _decode_tags(dict(zip(cols_meta, r))) for r in rows
    ]
    return JSONResponse({
        "items": items,
        "total": total,
        "has_more": (offset_n + len(items)) < total,
        "offset": offset_n,
        "limit": limit_n,
    })


# ── Collections CRUD ──────────────────────────────────────────────────


@router.get("/api/library/collections")
async def list_collections(request: Request) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    rows = await store.list_for_user(user_id=uid)
    out: list[dict[str, Any]] = []
    for col in rows:
        out.append({
            **col,
            "filter_json": json.loads(col.get("filter_json") or "{}"),
            "count": await store.count_items(col["id"], user_id=uid),
        })
    return JSONResponse({"collections": out})


@router.post("/api/library/collections")
async def create_collection(
    req: CreateCollectionReq, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    try:
        col = await store.create(
            user_id=uid,
            name=req.name,
            kind=req.kind,
            slug=req.slug,
            filter_json=req.filter_json,
            cover_url=req.cover_url,
            accent_color=req.accent_color,
            view_mode=req.view_mode,
        )
    except SlugCollision as e:
        raise HTTPException(status_code=409, detail=sanitize_error_detail(str(e)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_detail(str(e)))
    col["filter_json"] = json.loads(col.get("filter_json") or "{}")
    return JSONResponse(col, status_code=201)


@router.get("/api/library/collections/{collection_id}")
async def get_collection(
    collection_id: str, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    col = await store.get(collection_id, user_id=uid)
    if not col:
        raise HTTPException(status_code=404, detail="collection not found")
    item_ids = await store.list_items(collection_id, user_id=uid)
    # Collections can hold BOTH artifacts and publications since mig 309;
    # hydrate across the union so pub_ members aren't silently dropped.
    items = await _hydrate_union_by_ids(conn, user_id=uid, ids=item_ids)
    col["filter_json"] = json.loads(col.get("filter_json") or "{}")
    col["items"] = items
    return JSONResponse(col)


@router.put("/api/library/collections/{collection_id}")
async def update_collection(
    collection_id: str, req: UpdateCollectionReq, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    try:
        col = await store.update(
            collection_id,
            user_id=uid,
            name=req.name,
            slug=req.slug,
            filter_json=req.filter_json,
            cover_url=req.cover_url,
            accent_color=req.accent_color,
            view_mode=req.view_mode,
            sort_order=req.sort_order,
        )
    except SlugCollision as e:
        raise HTTPException(status_code=409, detail=sanitize_error_detail(str(e)))
    if not col:
        raise HTTPException(status_code=404, detail="collection not found")
    col["filter_json"] = json.loads(col.get("filter_json") or "{}")
    return JSONResponse(col)


@router.delete("/api/library/collections/{collection_id}")
async def delete_collection(
    collection_id: str, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    ok = await store.delete(collection_id, user_id=uid)
    if not ok:
        raise HTTPException(status_code=404, detail="collection not found")
    return JSONResponse({"ok": True})


# ── Collection items ──────────────────────────────────────────────────


@router.post("/api/library/collections/{collection_id}/items")
async def add_collection_items(
    collection_id: str, req: AddItemsReq, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    try:
        added = await store.add_items(
            collection_id, user_id=uid, artifact_ids=req.artifact_ids,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_detail(str(e)))
    return JSONResponse({"added": added})


@router.delete(
    "/api/library/collections/{collection_id}/items/{artifact_id}"
)
async def remove_collection_item(
    collection_id: str, artifact_id: str, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    store = CollectionStore(conn)
    ok = await store.remove_item(collection_id, artifact_id, user_id=uid)
    if not ok:
        raise HTTPException(status_code=404, detail="item not in collection")
    return JSONResponse({"ok": True})


# ── Per-item ──────────────────────────────────────────────────────────


@router.post("/api/library/items/{artifact_id}/pin")
async def set_pin(
    artifact_id: str, req: PinReq, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    pin_val = 1 if req.pinned else 0

    # Publications live in their own table (mig 309 gave them a pinned column).
    if is_publication_id(artifact_id):
        store = _pub_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail="library not initialized")
        row = await store.set_pinned(artifact_id, user_id=uid, pinned=req.pinned)
        if not row:
            raise HTTPException(status_code=404, detail="publication not found")
        await _record_item_activity_safe(
            conn, user_id=uid, item_id=artifact_id,
            action="pin" if pin_val else "unpin",
        )
        return JSONResponse({"id": artifact_id, "pinned": bool(pin_val)})

    row = await _artifact_row(conn, artifact_id=artifact_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="artifact not found")
    await conn.execute(
        "UPDATE artifacts SET pinned = ? WHERE id = ? AND user_id = ?",
        (pin_val, artifact_id, uid),
    )
    await conn.commit()
    await _record_item_activity_safe(
        conn, user_id=uid, item_id=artifact_id,
        action="pin" if pin_val else "unpin",
    )
    return JSONResponse({"id": artifact_id, "pinned": bool(pin_val)})


@router.post("/api/library/items/{artifact_id}/activity")
async def record_activity(
    artifact_id: str, req: ActivityReq, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    # `open` and `cast` also bump the "recently touched" timestamp so sort
    # orders (and the Recent collection) stay consistent. Publications track
    # this via last_launched_at (record_launch); artifacts via last_opened_at.
    if req.action in ("open", "cast"):
        if is_publication_id(artifact_id):
            store = _pub_store(request)
            if store is not None:
                await store.record_launch(artifact_id, user_id=uid)
        else:
            await conn.execute(
                "UPDATE artifacts SET last_opened_at = datetime('now') "
                "WHERE id = ? AND user_id = ?",
                (artifact_id, uid),
            )
            await conn.commit()
    try:
        aid = await ActivityStore(conn).record(
            user_id=uid, artifact_id=artifact_id,
            action=req.action, surface=req.surface, payload=req.payload,
        )
    except PermissionError:
        raise HTTPException(status_code=404, detail="item not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_detail(str(e)))
    return JSONResponse({"id": aid}, status_code=201)


@router.get("/api/library/items/{artifact_id}/activity")
async def list_activity(
    artifact_id: str, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    if not await _item_exists(request, item_id=artifact_id, user_id=uid):
        raise HTTPException(status_code=404, detail="item not found")
    events = await ActivityStore(conn).list_for_artifact(
        artifact_id, user_id=uid,
    )
    return JSONResponse({"events": events})


@router.put("/api/library/items/{artifact_id}/tags")
async def set_tags(
    artifact_id: str, req: TagsReq, request: Request,
) -> JSONResponse:
    uid = _require_auth(request)
    conn = _require_conn(request)
    # Dedupe + normalize (preserve order from first occurrence).
    seen: set[str] = set()
    normalized: list[str] = []
    for t in req.tags:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        normalized.append(s)

    # Publications store tags in their own column (mig 309).
    if is_publication_id(artifact_id):
        store = _pub_store(request)
        if store is None:
            raise HTTPException(status_code=503, detail="library not initialized")
        row = await store.set_tags(artifact_id, user_id=uid, tags=normalized)
        if not row:
            raise HTTPException(status_code=404, detail="publication not found")
        return JSONResponse({"id": artifact_id, "tags": normalized})

    row = await _artifact_row(conn, artifact_id=artifact_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="artifact not found")
    await conn.execute(
        "UPDATE artifacts SET tags = ? WHERE id = ? AND user_id = ?",
        (json.dumps(normalized), artifact_id, uid),
    )
    await conn.commit()
    return JSONResponse({"id": artifact_id, "tags": normalized})
