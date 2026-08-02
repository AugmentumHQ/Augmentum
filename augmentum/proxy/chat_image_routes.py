"""Chat image attachment storage — persists VL images across reloads."""

from __future__ import annotations

import base64
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/chat-images", tags=["chat-images"])

# Max image payload: 20 MB (base64-encoded, so ~15 MB raw)
_MAX_PAYLOAD = 20 * 1024 * 1024


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


@router.post("")
async def upload_chat_image(request: Request) -> JSONResponse:
    """Store a base64 data-URL image and return a stable URL.

    Request body: {"data_url": "data:image/png;base64,...", "session_id": "..."}
    Response: {"id": "<uuid>", "url": "/api/chat-images/<uuid>"}
    """
    body = await request.json()
    data_url: str = body.get("data_url", "")
    session_id: str = body.get("session_id", "")

    if not data_url or not data_url.startswith("data:"):
        return JSONResponse({"error": "Missing or invalid data_url"}, status_code=400)

    # Parse data URL: data:<mime>;base64,<payload>
    try:
        header, b64_data = data_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        raw_bytes = base64.b64decode(b64_data)
    except Exception:
        return JSONResponse({"error": "Invalid data URL format"}, status_code=400)

    if len(raw_bytes) > _MAX_PAYLOAD:
        return JSONResponse({"error": "Image too large (max 20MB)"}, status_code=413)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    image_id = uuid.uuid4().hex[:16]

    backend = getattr(request.app.state, "state_manager", None)
    conn = getattr(backend, "_backend", None) if backend else None
    if not conn:
        # Fallback: try direct sqlite backend
        conn = getattr(request.app.state, "sqlite_backend", None)

    if not conn:
        return JSONResponse({"error": "Storage unavailable"}, status_code=503)

    db = getattr(conn, "_conn", conn)
    await db.execute(
        "INSERT INTO chat_images (id, mime_type, data, session_id, user_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [image_id, mime_type, raw_bytes, session_id or None, uid],
    )
    await db.commit()

    from augmentum.vfs import register_file
    ext = mime_type.split("/")[-1] if mime_type else "png"
    await register_file(
        user_id=uid, source="chat_images", source_id=image_id,
        name=f"chat_{image_id[:8]}.{ext}", mime_type=mime_type,
        size_bytes=len(raw_bytes),
        source_metadata={"session_id": session_id},
    )

    url = f"/api/chat-images/{image_id}"
    log.info("chat_image_stored", id=image_id, mime=mime_type, size=len(raw_bytes))
    return JSONResponse({"id": image_id, "url": url})


@router.get("/{image_id}")
async def get_chat_image(image_id: str, request: Request) -> Response:
    """Serve a stored chat image by ID."""
    uid = _user_id(request)
    if not uid:
        return Response(status_code=401)

    backend = getattr(request.app.state, "state_manager", None)
    conn = getattr(backend, "_backend", None) if backend else None
    if not conn:
        conn = getattr(request.app.state, "sqlite_backend", None)

    if not conn:
        return Response(status_code=503)

    db = getattr(conn, "_conn", conn)
    async with db.execute(
        "SELECT mime_type, data FROM chat_images WHERE id = ? AND user_id = ?",
        [image_id, uid],
    ) as cursor:
        row = await cursor.fetchone()

    if not row:
        return Response(status_code=404)

    mime_type, data = row
    return Response(
        content=data,
        media_type=mime_type,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@router.delete("/{image_id}")
async def delete_chat_image(image_id: str, request: Request) -> JSONResponse:
    """Delete a stored chat image."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    backend = getattr(request.app.state, "state_manager", None)
    conn = getattr(backend, "_backend", None) if backend else None
    if not conn:
        conn = getattr(request.app.state, "sqlite_backend", None)

    if not conn:
        return JSONResponse({"error": "Storage unavailable"}, status_code=503)

    db = getattr(conn, "_conn", conn)
    await db.execute(
        "DELETE FROM chat_images WHERE id = ? AND user_id = ?",
        [image_id, uid],
    )
    await db.commit()

    # Cascade into file_index so the files panel doesn't strand the row
    from augmentum.vfs import unregister_file
    await unregister_file("chat_images", image_id, user_id=uid)

    return JSONResponse({"ok": True})
