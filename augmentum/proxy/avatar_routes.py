"""Avatar API routes — list, upload, serve, delete, assign, for-session."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from augmentum.avatar.store import AvatarStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/avatar", tags=["avatar"])

def _get_avatar_dir() -> str:
    """Resolve avatar storage directory using the project's standard data dir."""
    env_override = os.environ.get("AUGMENTUM_AVATAR_DIR", "")
    if env_override:
        return env_override
    from augmentum.config import settings
    base = getattr(settings, "data_dir", "/data")
    return os.path.join(base, "avatars")
VRM_MAX_SIZE = 100 * 1024 * 1024  # 100MB

# Thumbnail upload cap. We size for ~256×256 PNGs (a roomy ceiling for
# anything the client realistically captures), and the client already
# downscales before upload — anything past this is either malformed or
# malicious.
THUMBNAIL_MAX_SIZE = 256 * 1024  # 256KB


def _user_thumb_path(avatar_id: str, user_id: str) -> Path:
    """Per-user override path for an avatar's thumbnail.

    Lives under ``<avatar_dir>/.user_thumbnails/<user_id>/<id>.png``.
    Used in two cases:
      - The avatar is bundled (shared, read-only), so the user's
        captured snapshot can't overwrite the bundle.
      - The avatar IS user-owned but we want to keep captures
        per-user-per-avatar without colliding with the canonical
        ``<avatar_dir>/<id>/thumbnail.png`` that the avatar's own
        upload flow already manages.
    """
    return (
        Path(_get_avatar_dir())
        / ".user_thumbnails"
        / user_id
        / f"{avatar_id}.png"
    )


def _get_store(request: Request) -> AvatarStore:
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        raise RuntimeError("Backend not initialized")
    return AvatarStore(sm.backend.conn)


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


def _safe_avatar_id(avatar_id: str) -> str | None:
    """Return avatar_id only if it contains no path separators, else None."""
    if "/" in avatar_id or "\\" in avatar_id or ".." in avatar_id:
        return None
    return avatar_id


def _avatar_vrm_url(avatar_id: str) -> str:
    return f"/api/avatar/{avatar_id}.vrm"


def _avatar_thumbnail_url(avatar_id: str) -> str:
    return f"/api/avatar/{avatar_id}/thumbnail"


def _avatar_to_response(avatar: dict) -> dict:
    """Enrich an avatar dict with computed URL fields."""
    avatar_id = avatar["id"]
    try:
        mannerisms = json.loads(avatar.get("mannerisms") or "{}")
    except Exception:
        mannerisms = {}
    avatar_type = avatar.get("type", "vrm")

    # Look up display name for bundled avatars
    name = None
    if avatar.get("is_bundled"):
        from augmentum.avatar.bundled import BUNDLED_AVATARS
        for ba in BUNDLED_AVATARS:
            if ba["id"] == avatar_id:
                name = ba["name"]
                break

    return {
        "id": avatar_id,
        "type": avatar_type,
        "name": name,
        "character_id": avatar.get("character_id"),
        "persona_id": avatar.get("persona_id"),
        "vrm_url": _avatar_vrm_url(avatar_id) if avatar_type == "vrm" else None,
        "portrait_url": f"/api/avatar/{avatar_id}/portrait" if avatar_type == "portrait" else None,
        "thumbnail_url": _avatar_thumbnail_url(avatar_id),
        "mannerisms": mannerisms,
        "is_bundled": bool(avatar.get("is_bundled")),
        "created_at": avatar.get("created_at"),
        "updated_at": avatar.get("updated_at"),
    }


# ── List ──────────────────────────────────────────────────────────────────

@router.get("/lipsync-capabilities")
async def lipsync_capabilities():
    """Report available lip sync backends. Frontend uses this to decide
    whether to upgrade volume-gated visemes to Audio2Face blendshapes."""
    a2f_path = Path(_get_avatar_dir()).parent / "models" / "audio2face" / "network.onnx"
    a2f_available = a2f_path.exists()
    return JSONResponse({
        "audio2face": a2f_available,
        "version": "2.3" if a2f_available else None,
    })


@router.get("/list")
async def list_avatars(request: Request):
    """Return avatars visible to the authenticated user (owned + bundled)."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _get_store(request)
    avatars = await store.list_all(user_id=uid)
    return JSONResponse({"avatars": [_avatar_to_response(a) for a in avatars]})


@router.get("/bundled")
async def list_bundled_avatars(request: Request):
    """Return bundled avatars only."""
    store = _get_store(request)
    avatars = await store.list_bundled()
    return JSONResponse({"avatars": [_avatar_to_response(a) for a in avatars]})



# ── For-session resolution ────────────────────────────────────────────────

@router.get("/for-session")
async def avatar_for_session(request: Request):
    """Resolve which avatar to display for a given session/mode.

    Query params:
      - session_id: str
      - mode: str  (narrative | passthrough | agentic | ...)
      - avatar_id: str  (optional direct override from in-call picker)
      - character_id: str  (optional shortcut — skip session lookup)

    Resolution chain:
      1. If character_id provided → look up avatar by character
      2. If mode == narrative → look up session's active character → avatar
      3. Else → look up persona avatar
      4. Fallback → first bundled avatar
    """
    session_id = request.query_params.get("session_id", "")
    mode = request.query_params.get("mode", "")
    avatar_id_param = request.query_params.get("avatar_id", "")
    character_id = request.query_params.get("character_id", "")

    uid = _user_id(request)
    store = _get_store(request)
    avatar = None
    source = "none"

    # Direct in-call picker override. If the caller asked for a specific
    # avatar, fail clearly instead of silently falling back to another one.
    if avatar_id_param:
        if not _safe_avatar_id(avatar_id_param):
            return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)
        avatar = await store.get(avatar_id_param, user_id=uid)
        if not avatar:
            return JSONResponse({"error": "Avatar not found"}, status_code=404)
        source = "avatar"

    # Fast path: character_id provided directly
    if not avatar and character_id:
        avatar = await store.get_by_character(character_id, user_id=uid)
        if avatar:
            source = "character"

    # Check the caller's default avatar (per-user via Stage D settings
    # split), falling back to the install-wide default only if they've
    # never set one.
    if not avatar:
        settings_store = getattr(request.app.state, "settings_store", None)
        if settings_store:
            default_id = await settings_store.get_user_or_global(
                uid, "user_default_avatar_id",
            )
            if default_id:
                avatar = await store.get(default_id, user_id=uid)
                if avatar:
                    source = "user_default"

    # Fallback to first bundled avatar
    if not avatar:
        bundled = await store.list_bundled()
        if bundled:
            avatar = bundled[0]
            source = "bundled"

    # Fallback to any uploaded avatar (user uploaded one but hasn't assigned it)
    if not avatar:
        all_avatars = await store.list_all(user_id=uid)
        if all_avatars:
            avatar = all_avatars[0]
            source = "uploaded"

    if not avatar:
        return JSONResponse({"avatar_id": None, "source": "none"})

    avatar_id = avatar["id"]
    try:
        mannerisms = json.loads(avatar.get("mannerisms") or "{}")
    except Exception:
        mannerisms = {}

    avatar_type = avatar.get("type", "vrm")

    return JSONResponse({
        "avatar_id": avatar_id,
        "vrm_url": _avatar_vrm_url(avatar_id) if avatar_type == "vrm" else None,
        "thumbnail_url": _avatar_thumbnail_url(avatar_id),
        "mannerisms": mannerisms,
        "source": source,
        "type": avatar_type,
        "portrait_url": f"/api/avatar/{avatar_id}/portrait" if avatar_type == "portrait" else None,
        "segmentation_data": avatar.get("segmentation_data") if avatar_type == "portrait" else None,
    })


# ── Upload ────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_avatar(request: Request, file: UploadFile = File(...)):
    """Upload a custom VRM file and register it in the avatars table."""
    filename = file.filename or ""
    if not filename.lower().endswith(".vrm"):
        return JSONResponse({"error": "Only .vrm files are accepted"}, status_code=400)

    # Read file content and enforce size limit
    content = await file.read()
    if len(content) > VRM_MAX_SIZE:
        return JSONResponse(
            {"error": f"File too large (max {VRM_MAX_SIZE // 1024 // 1024}MB)"},
            status_code=400,
        )

    # Create record first to get the avatar_id
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _get_store(request)
    # Placeholder paths — will be updated after we know the avatar_id
    avatar = await store.create(
        vrm_path="",
        is_bundled=False,
        user_id=uid,
    )
    avatar_id = avatar["id"]

    # Save to disk
    avatar_dir = Path(_get_avatar_dir()) / avatar_id
    try:
        avatar_dir.mkdir(parents=True, exist_ok=True)
        vrm_path = avatar_dir / f"{avatar_id}.vrm"
        vrm_path.write_bytes(content)
    except OSError as exc:
        log.warning("avatar_upload_write_failed", avatar_id=avatar_id, error=str(exc))
        await store.delete(avatar_id, user_id=uid)
        return JSONResponse({"error": "Failed to save VRM file"}, status_code=500)

    # Update the vrm_path in the DB (scoped to the creating user)
    try:
        await store._conn.execute(
            "UPDATE avatars SET vrm_path = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (str(vrm_path), avatar_id, uid),
        )
        await store._conn.commit()
    except Exception as exc:
        log.warning("avatar_upload_db_update_failed", avatar_id=avatar_id, error=str(exc))

    log.info("avatar_uploaded", avatar_id=avatar_id, size=len(content), user_id=uid)
    return JSONResponse({
        "ok": True,
        "avatar_id": avatar_id,
        "vrm_url": _avatar_vrm_url(avatar_id),
        "thumbnail_url": _avatar_thumbnail_url(avatar_id),
    })


@router.post("/upload-portrait")
async def upload_portrait(request: Request, file: UploadFile = File(...)):
    """Upload a portrait image for 2D avatar."""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        return JSONResponse({"error": "Only PNG/JPG/WebP images accepted"}, status_code=400)

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        return JSONResponse({"error": "File too large (max 10MB)"}, status_code=400)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _get_store(request)
    avatar = await store.create(
        vrm_path="", avatar_type="portrait", is_bundled=False, user_id=uid,
    )
    avatar_id = avatar["id"]

    avatar_dir = Path(_get_avatar_dir()) / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    portrait_path = avatar_dir / f"portrait{ext}"
    portrait_path.write_bytes(content)

    # Update path (scoped to the creating user)
    try:
        await store._conn.execute(
            "UPDATE avatars SET vrm_path = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (str(portrait_path), avatar_id, uid),
        )
        await store._conn.commit()
    except Exception as exc:
        log.warning("portrait_upload_db_update_failed", avatar_id=avatar_id, error=str(exc))

    log.info("portrait_uploaded", avatar_id=avatar_id, size=len(content), user_id=uid)
    return JSONResponse({
        "ok": True,
        "avatar_id": avatar_id,
        "type": "portrait",
        "portrait_url": f"/api/avatar/{avatar_id}/portrait",
    })


# ── Select default avatar ─────────────────────────────────────────────────

@router.post("/select")
async def select_default_avatar(request: Request):
    """Set the user's default avatar for non-character voice calls."""
    body = await request.json()
    avatar_id = body.get("avatar_id")
    if not avatar_id:
        return JSONResponse({"error": "avatar_id required"}, status_code=400)

    # Verify avatar exists
    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    # Save to per-user settings so each tenant has their own default.
    settings_store = getattr(request.app.state, "settings_store", None)
    if settings_store:
        if not uid:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        await settings_store.set_user(uid, "user_default_avatar_id", avatar_id)

    return JSONResponse({"ok": True, "avatar_id": avatar_id})


# ── Serve VRM ─────────────────────────────────────────────────────────────

@router.get("/{avatar_id}.vrm")
async def serve_vrm(avatar_id: str, request: Request):
    """Serve a VRM file by avatar ID."""
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)

    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    vrm_path = avatar.get("vrm_path") or ""
    if not vrm_path:
        # Try the conventional path
        vrm_path = str(Path(_get_avatar_dir()) / avatar_id / f"{avatar_id}.vrm")

    if not Path(vrm_path).is_file():
        return JSONResponse({"error": "VRM file not found on disk"}, status_code=404)

    return FileResponse(vrm_path, media_type="application/octet-stream",
                        filename=f"{avatar_id}.vrm")


# ── Serve thumbnail ───────────────────────────────────────────────────────

@router.get("/{avatar_id}/thumbnail")
async def serve_thumbnail(avatar_id: str, request: Request):
    """Serve the thumbnail PNG for an avatar.

    Lookup order:
      1. Per-user override (captured from the live canvas) at
         ``<avatar_dir>/.user_thumbnails/<user_id>/<avatar_id>.png``
      2. Avatar's stored ``thumbnail_path`` (shipped or upload-flow)
      3. Conventional path ``<avatar_dir>/<id>/thumbnail.png``
      4. 1×1 transparent placeholder + ``X-Avatar-Thumbnail-Placeholder``
         header so the client knows to capture-and-upload.
    """
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)

    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    # 1) Per-user override first — the user's own canvas snapshot
    # takes priority over any shipped/bundled asset.
    if uid:
        override = _user_thumb_path(avatar_id, uid)
        if override.is_file():
            return FileResponse(str(override), media_type="image/png")

    # 2/3) Avatar's stored or conventional path.
    thumbnail_path = avatar.get("thumbnail_path") or ""
    if not thumbnail_path:
        thumbnail_path = str(Path(_get_avatar_dir()) / avatar_id / "thumbnail.png")

    if not Path(thumbnail_path).is_file():
        # 4) Placeholder. Header lets the client distinguish this from
        # a real PNG so it knows to trigger the capture-and-upload
        # path rather than displaying transparency.
        from starlette.responses import Response
        _TRANSPARENT_PNG = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return Response(
            content=_TRANSPARENT_PNG,
            media_type="image/png",
            headers={"X-Avatar-Thumbnail-Placeholder": "1"},
        )

    return FileResponse(thumbnail_path, media_type="image/png")


@router.post("/{avatar_id}/thumbnail")
async def upload_thumbnail_file(
    avatar_id: str, request: Request, file: UploadFile = File(...),
):
    """Receive a captured thumbnail PNG (typically a downscaled canvas
    snapshot from the active avatar renderer) and persist it as the
    per-user override for this avatar.

    Always writes to the per-user path so a bundled-avatar snapshot
    from User A doesn't leak into User B's pip. For non-bundled
    avatars the canonical path is still authoritative on render —
    this just gives the user their own captured version on top.
    """
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    # Cap read size to avoid an oversized upload eating memory while
    # we're streaming it in.
    data = await file.read(THUMBNAIL_MAX_SIZE + 1)
    if not data:
        return JSONResponse({"error": "Empty body"}, status_code=400)
    if len(data) > THUMBNAIL_MAX_SIZE:
        return JSONResponse(
            {"error": f"Thumbnail exceeds {THUMBNAIL_MAX_SIZE} bytes"},
            status_code=413,
        )
    # Cheap PNG signature check — first 8 bytes are the magic header.
    # We don't validate the rest; the renderer is trusted-source and
    # an invalid PNG just won't display.
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return JSONResponse(
            {"error": "Body is not a PNG"}, status_code=415,
        )

    out_path = _user_thumb_path(avatar_id, uid)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    try:
        out_path.chmod(0o600)
    except OSError:
        pass
    log.info(
        "avatar_thumbnail_uploaded",
        avatar_id=avatar_id, user_id=uid, bytes=len(data),
    )
    return JSONResponse({"ok": True, "bytes": len(data)})


# ── Serve portrait ────────────────────────────────────────────────────────

@router.get("/{avatar_id}/portrait")
async def serve_portrait(avatar_id: str, request: Request):
    """Serve the portrait image for a 2D avatar."""
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)
    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    portrait_path = avatar.get("vrm_path") or ""
    if not portrait_path or not Path(portrait_path).is_file():
        avatar_dir = Path(_get_avatar_dir()) / avatar_id
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = avatar_dir / f"portrait{ext}"
            if candidate.is_file():
                portrait_path = str(candidate)
                break

    if not Path(portrait_path).is_file():
        return JSONResponse({"error": "Portrait not found"}, status_code=404)

    return FileResponse(portrait_path)


# ── Upload rendered thumbnail ─────────────────────────────────────────────

@router.put("/{avatar_id}/thumbnail")
async def upload_thumbnail(avatar_id: str, request: Request):
    """Save a client-rendered thumbnail PNG for an avatar."""
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)

    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    body = await request.body()
    if len(body) > 512 * 1024:
        return JSONResponse({"error": "Too large"}, status_code=413)
    if not body.startswith(b"\x89PNG"):
        return JSONResponse({"error": "Not a valid PNG"}, status_code=400)

    avatar_dir = Path(_get_avatar_dir()) / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = avatar_dir / "thumbnail.png"
    thumb_path.write_bytes(body)

    # Only the owner (or admin via user_id="") can update the thumbnail path.
    upd_query = (
        "UPDATE avatars SET thumbnail_path = ?, updated_at = datetime('now') "
        "WHERE id = ?"
    )
    upd_params: list = [str(thumb_path), avatar_id]
    if uid:
        upd_query += " AND (user_id = ? OR user_id IS NULL)"
        upd_params.append(uid)
    await store._conn.execute(upd_query, upd_params)
    await store._conn.commit()
    log.info("avatar_thumbnail_saved", avatar_id=avatar_id, size=len(body))
    return JSONResponse({"ok": True})


# ── Segmentation cache ────────────────────────────────────────────────────

@router.put("/{avatar_id}/segmentation")
async def save_segmentation(avatar_id: str, request: Request):
    """Cache segmentation data for a portrait avatar (owner only)."""
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid"}, status_code=400)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    store = _get_store(request)
    await store._conn.execute(
        "UPDATE avatars SET segmentation_data = ?, updated_at = datetime('now') "
        "WHERE id = ? AND user_id = ?",
        (json.dumps(body), avatar_id, uid),
    )
    await store._conn.commit()
    return JSONResponse({"status": "ok"})


# ── Delete ────────────────────────────────────────────────────────────────

@router.delete("/{avatar_id}")
async def delete_avatar(avatar_id: str, request: Request):
    """Delete an avatar record and its files from disk. Bundled avatars cannot be deleted."""
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)

    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    if avatar.get("is_bundled"):
        return JSONResponse({"error": "Bundled avatars cannot be deleted"}, status_code=403)

    # Delete from DB — scoped so tenants can't nuke each other's avatars.
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    await store.delete(avatar_id, user_id=uid)

    # Delete files from disk
    avatar_dir = Path(_get_avatar_dir()) / avatar_id
    if avatar_dir.is_dir():
        try:
            shutil.rmtree(avatar_dir)
        except OSError as exc:
            log.warning("avatar_delete_disk_failed", avatar_id=avatar_id, error=str(exc))

    log.info("avatar_deleted", avatar_id=avatar_id)
    return JSONResponse({"ok": True})


# ── Update mannerisms ─────────────────────────────────────────────────────

@router.patch("/{avatar_id}/mannerisms")
async def update_mannerisms(avatar_id: str, request: Request):
    """Update the mannerism JSON for an avatar."""
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)

    uid = _user_id(request)
    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    await store.update_mannerisms(avatar_id, body, user_id=uid)
    log.info("avatar_mannerisms_updated", avatar_id=avatar_id, user_id=uid)
    return JSONResponse({"ok": True})


# ── Pair / unpair avatar with a character ────────────────────────────────

@router.patch("/{avatar_id}/character")
async def update_avatar_character(avatar_id: str, request: Request):
    """Pair an avatar with a character card, or unpair it.

    Body: ``{"character_id": "<char_id>"}`` to pair, or
    ``{"character_id": null}`` / ``{"character_id": ""}`` to unpair.

    Multi-tenant note: ``store.assign_to_character`` allows bundled
    avatars (``user_id IS NULL``) to be claimed by any caller, and the
    underlying ``avatars.character_id`` column is install-wide rather
    than per-user. That means if two tenants pair the same bundled VRM
    with their own characters, last-write-wins. The legacy comment in
    ``store.py`` calls this out and accepts the tradeoff — it's the
    cost of bundled avatars being shared assets. Don't rely on the
    paired-with hint surfaced in the UI as a hard exclusion.
    """
    if not _safe_avatar_id(avatar_id):
        return JSONResponse({"error": "Invalid avatar_id"}, status_code=400)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, status_code=400)

    raw = body.get("character_id")
    # Treat null and empty string identically: unpair.
    character_id = "" if raw is None else str(raw).strip()
    if len(character_id) > 256:
        return JSONResponse({"error": "character_id too long"}, status_code=400)

    store = _get_store(request)
    avatar = await store.get(avatar_id, user_id=uid)
    if not avatar:
        return JSONResponse({"error": "Avatar not found"}, status_code=404)
    if avatar.get("type") == "portrait":
        return JSONResponse({
            "error": "Only VRM avatars can be paired with characters",
        }, status_code=400)

    await store.assign_to_character(avatar_id, character_id, user_id=uid)
    log.info(
        "avatar_character_pairing_updated",
        avatar_id=avatar_id,
        character_id=character_id or None,
        user_id=uid,
    )
    return JSONResponse({"ok": True, "avatar_id": avatar_id, "character_id": character_id or None})
