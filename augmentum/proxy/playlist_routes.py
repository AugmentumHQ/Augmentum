"""Playlist routes — user-scoped media queues spanning YouTube + library files.

Items are typed pointers, not audio bytes:
  - {type: 'youtube', videoId, title, channel?, thumbnail?}
  - {type: 'file',    fileId,  name,  kind: 'audio'|'video', thumbnail?}

Stored as a JSON array on the row so v1 is a single round-trip per playlist.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/playlists", tags=["playlists"])

_MAX_NAME_LEN = 80
_MAX_ITEMS_PER_PLAYLIST = 500


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _conn(request: Request):
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        raise RuntimeError("Backend not initialized")
    return sm.backend.conn


def _make_id() -> str:
    ts = int(time.time() * 1000)
    h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]
    return f"pl_{ts}_{h}"


async def _emit_playlist_evidence(request: Request, *, user_id: str, name: str) -> None:
    """Feed the playlist name into the Evidence Bus (Earned Understanding P2).
    Best-effort and isolated — a learning hiccup must never break a save."""
    try:
        memory_store = getattr(request.app.state, "memory_store", None)
        evidence_store = getattr(request.app.state, "evidence_store", None)
        if memory_store is None or evidence_store is None:
            return
        from augmentum.memory.evidence_emitters import emit_playlist_evidence
        await emit_playlist_evidence(
            memory_store, evidence_store, user_id=user_id, playlist_name=name,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("playlist_evidence_emit_failed", error=str(exc)[:200])


def _sanitize_item(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or "").strip().lower()
    if kind == "youtube":
        vid = str(raw.get("videoId") or "").strip()
        if not vid:
            return None
        return {
            "type": "youtube",
            "videoId": vid,
            "title": str(raw.get("title") or "").strip()[:300],
            "channel": str(raw.get("channel") or "").strip()[:120],
            "thumbnail": str(raw.get("thumbnail") or "").strip()[:500],
        }
    if kind == "file":
        fid = str(raw.get("fileId") or "").strip()
        if not fid:
            return None
        file_kind = str(raw.get("kind") or "").strip().lower()
        if file_kind not in ("audio", "video"):
            return None
        return {
            "type": "file",
            "fileId": fid,
            "name": str(raw.get("name") or "").strip()[:300],
            "kind": file_kind,
            # Content category (movie/series/episode/music_video/book/
            # podcast/music/comic/live_program). Preserved so the playlist
            # boundary groups by FAMILY (watch/music/spoken/comics) rather
            # than the coarse audio/video kind. Optional — older items
            # without it fall back to kind-based family inference.
            "entityKind": str(raw.get("entityKind") or "").strip().lower()[:40],
            "thumbnail": str(raw.get("thumbnail") or "").strip()[:500],
        }
    return None


def _sanitize_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw[:_MAX_ITEMS_PER_PLAYLIST]:
        item = _sanitize_item(entry)
        if item:
            out.append(item)
    return out


def _row_to_dict(row) -> dict[str, Any]:
    try:
        items = json.loads(row[3] or "[]")
    except (json.JSONDecodeError, TypeError):
        items = []
    return {
        "id": row[0],
        "name": row[2],
        "items": items if isinstance(items, list) else [],
        "created_at": row[4],
        "updated_at": row[5],
        # Provenance (migration 262): '' = user-created, 'companion'
        # = created by the companion verb. Read-only on the wire.
        "origin": row[6] if len(row) > 6 else "",
    }


@router.get("")
async def list_playlists(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    cursor = await conn.execute(
        "SELECT id, user_id, name, items_json, created_at, updated_at, origin "
        "FROM playlists WHERE user_id = ? ORDER BY updated_at DESC",
        (uid,),
    )
    rows = await cursor.fetchall()
    return JSONResponse({"playlists": [_row_to_dict(r) for r in rows]})


@router.post("")
async def create_playlist(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    name = str(body.get("name") or "").strip()[:_MAX_NAME_LEN]
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    items = _sanitize_items(body.get("items"))
    pl_id = _make_id()
    conn = _conn(request)
    await conn.execute(
        "INSERT INTO playlists (id, user_id, name, items_json) "
        "VALUES (?, ?, ?, ?)",
        (pl_id, uid, name, json.dumps(items)),
    )
    await conn.commit()
    cursor = await conn.execute(
        "SELECT id, user_id, name, items_json, created_at, updated_at, origin "
        "FROM playlists WHERE id = ? AND user_id = ?",
        (pl_id, uid),
    )
    row = await cursor.fetchone()
    if not row:
        return JSONResponse({"error": "Insert failed"}, status_code=500)
    await _emit_playlist_evidence(request, user_id=uid, name=name)
    return JSONResponse({"playlist": _row_to_dict(row)})


@router.put("/{playlist_id}")
async def update_playlist(playlist_id: str, request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    conn = _conn(request)
    cursor = await conn.execute(
        "SELECT id, user_id, name, items_json, created_at, updated_at, origin "
        "FROM playlists WHERE id = ? AND user_id = ?",
        (playlist_id, uid),
    )
    row = await cursor.fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)

    old_name = row[2]
    name = old_name
    if "name" in body:
        n = str(body.get("name") or "").strip()[:_MAX_NAME_LEN]
        if n:
            name = n

    if "items" in body:
        items = _sanitize_items(body.get("items"))
    else:
        items = json.loads(row[3] or "[]")

    await conn.execute(
        "UPDATE playlists SET name = ?, items_json = ?, "
        "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
        (name, json.dumps(items), playlist_id, uid),
    )
    await conn.commit()

    cursor = await conn.execute(
        "SELECT id, user_id, name, items_json, created_at, updated_at, origin "
        "FROM playlists WHERE id = ? AND user_id = ?",
        (playlist_id, uid),
    )
    updated = await cursor.fetchone()
    if name != old_name:
        await _emit_playlist_evidence(request, user_id=uid, name=name)
    return JSONResponse({"playlist": _row_to_dict(updated)})


@router.delete("/{playlist_id}")
async def delete_playlist(playlist_id: str, request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    conn = _conn(request)
    cursor = await conn.execute(
        "DELETE FROM playlists WHERE id = ? AND user_id = ?",
        (playlist_id, uid),
    )
    await conn.commit()
    return JSONResponse({"deleted": cursor.rowcount or 0})
