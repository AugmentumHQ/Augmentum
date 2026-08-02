"""Document RAG API routes — upload, list, delete, search."""

from __future__ import annotations

from pathlib import PurePosixPath

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 50 MB upload limit
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

ALLOWED_MIME_TYPES = frozenset({
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "application/pdf",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
})

# Also allow by extension for cases where mime is generic
ALLOWED_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".json", ".log", ".pdf", ".docx", ".pptx", ".xlsx",
    ".html", ".htm", ".py", ".js", ".ts", ".yaml", ".yml", ".toml", ".xml", ".rst",
})


def _get_store(request: Request):
    """Get DocumentStore from app state."""
    store = getattr(request.app.state, "document_store", None)
    if store is None:
        return None
    return store


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


@router.post("")
async def upload_document(request: Request, file: UploadFile) -> JSONResponse:
    """Upload and ingest a document for RAG."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Document store not enabled"}, status_code=503)

    # Validate file — sanitize filename to prevent path traversal
    raw_name = file.filename or "unknown"
    filename = PurePosixPath(raw_name).name or "unknown"
    mime = file.content_type or "application/octet-stream"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if mime not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            {"error": f"Unsupported file type: {mime} ({ext})"},
            status_code=400,
        )

    # Read file in chunks to enforce size limit without buffering oversized files
    _CHUNK = 64 * 1024  # 64 KB
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            return JSONResponse(
                {"error": f"File too large (max {MAX_UPLOAD_SIZE} bytes)"},
                status_code=413,
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        return JSONResponse({"error": "Empty file"}, status_code=400)

    # Infer better mime from extension when generic
    if mime == "application/octet-stream":
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".json": "application/json",
            ".html": "text/html",
            ".htm": "text/html",
        }
        mime = mime_map.get(ext, "text/plain")

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    try:
        result = await store.ingest(data, filename, mime, user_id=uid)
        return JSONResponse(result, status_code=201)
    except ValueError as exc:
        return JSONResponse(
            {"error": sanitize_error_detail(str(exc))}, status_code=400,
        )
    except Exception:
        log.error("document_ingest_failed", filename=filename, exc_info=True)
        return JSONResponse({"error": "Failed to process document"}, status_code=500)


@router.get("")
async def list_documents(request: Request) -> JSONResponse:
    """List all uploaded documents."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Document store not enabled"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    docs = await store.list_documents(user_id=uid)
    return JSONResponse({"documents": docs})


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> JSONResponse:
    """Delete a document and all its chunks."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Document store not enabled"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    deleted = await store.delete_document(doc_id, user_id=uid)
    if not deleted:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"deleted": True})


@router.get("/{doc_id}")
async def get_document(doc_id: str, request: Request) -> JSONResponse:
    """Get document metadata."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Document store not enabled"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    doc = await store.get_document(doc_id, user_id=uid)
    if not doc:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse(doc)


@router.post("/search")
async def search_documents(request: Request) -> JSONResponse:
    """Search document chunks by query."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Document store not enabled"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    limit = min(int(body.get("limit", 5)), 20)
    doc_id = body.get("document_id")

    results = await store.search(query, user_id=uid, limit=limit, document_id=doc_id)
    return JSONResponse({"results": results})


def _get_conn(request: Request):
    """Get database connection from app state."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    if sm and isinstance(sm.backend, SQLiteBackend):
        return sm.backend.conn
    return None


# ------------------------------------------------------------------
# Session-document bindings
# ------------------------------------------------------------------

@router.get("/session/{session_id}")
async def get_session_documents(session_id: str, request: Request) -> JSONResponse:
    """List documents bound to a session with their inject mode."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"bindings": []})

    uid = _user_id(request)
    try:
        query = (
            "SELECT sd.document_id, sd.inject_mode, d.filename, d.chunk_count, d.file_size "
            "FROM session_documents sd "
            "JOIN documents d ON d.id = sd.document_id "
            "WHERE sd.session_id = ?"
        )
        params: list[object] = [session_id]
        if uid:
            query += " AND sd.user_id = ?"
            params.append(uid)
        query += " ORDER BY sd.created_at"
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return JSONResponse({
            "bindings": [dict(r) for r in rows],
        })
    except Exception:
        return JSONResponse({"bindings": []})


@router.post("/session/{session_id}")
async def bind_document_to_session(session_id: str, request: Request) -> JSONResponse:
    """Bind a document to a session. Body: {document_id, inject_mode?}."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    body = await request.json()
    doc_id = body.get("document_id", "").strip()
    inject_mode = body.get("inject_mode", "search").strip()
    if not doc_id:
        return JSONResponse({"error": "document_id required"}, status_code=400)
    if inject_mode not in ("search", "full"):
        return JSONResponse({"error": "inject_mode must be 'search' or 'full'"}, status_code=400)

    uid = _user_id(request)
    try:
        # Validate document ownership BEFORE inserting the binding.
        # Without this guard, a user can POST any doc_id and pollute
        # their session_documents table with references to documents
        # they don't own (or non-existent documents). The read-side
        # check in documents.store.get_full_content() currently
        # prevents the actual content from leaking — but relying
        # solely on that is fragile (one regression in get_full_content
        # turns this into a real cross-user data leak). Fail loudly at
        # write time instead. Only enforce when auth is active (uid
        # truthy) to preserve the unauthenticated dev-mode path.
        # 404 (not 403) avoids revealing whether the doc exists for
        # another user — same response either way.
        if uid:
            check = await conn.execute(
                "SELECT 1 FROM documents WHERE id = ? AND user_id = ?",
                (doc_id, uid),
            )
            if not await check.fetchone():
                return JSONResponse({"error": "Document not found"}, status_code=404)

        cols = "session_id, document_id, inject_mode"
        placeholders = "?, ?, ?"
        vals: list[object] = [session_id, doc_id, inject_mode]
        if uid:
            cols += ", user_id"
            placeholders += ", ?"
            vals.append(uid)
        await conn.execute(
            f"INSERT OR REPLACE INTO session_documents ({cols}) VALUES ({placeholders})",
            vals,
        )
        await conn.commit()
        return JSONResponse({"bound": True, "inject_mode": inject_mode})
    except Exception:
        log.warning("bind_document_failed", session_id=session_id, doc_id=doc_id, exc_info=True)
        return JSONResponse({"error": "Failed to bind document"}, status_code=500)


@router.put("/session/{session_id}/{doc_id}/mode")
async def update_inject_mode(session_id: str, doc_id: str, request: Request) -> JSONResponse:
    """Update the inject mode for a bound document. Body: {inject_mode}."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    body = await request.json()
    inject_mode = body.get("inject_mode", "search").strip()
    if inject_mode not in ("search", "full"):
        return JSONResponse({"error": "inject_mode must be 'search' or 'full'"}, status_code=400)

    uid = _user_id(request)
    try:
        upd_query = "UPDATE session_documents SET inject_mode = ? WHERE session_id = ? AND document_id = ?"
        upd_params: list[object] = [inject_mode, session_id, doc_id]
        if uid:
            upd_query += " AND user_id = ?"
            upd_params.append(uid)
        cursor = await conn.execute(upd_query, upd_params)
        await conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse({"error": "Binding not found"}, status_code=404)
        return JSONResponse({"updated": True, "inject_mode": inject_mode})
    except Exception:
        log.debug("update_inject_mode_failed", exc_info=True)
        return JSONResponse({"error": "Failed to update mode"}, status_code=500)


@router.delete("/session/{session_id}/{doc_id}")
async def unbind_document_from_session(session_id: str, doc_id: str, request: Request) -> JSONResponse:
    """Remove a document binding from a session."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    uid = _user_id(request)
    try:
        del_query = "DELETE FROM session_documents WHERE session_id = ? AND document_id = ?"
        del_params: list[object] = [session_id, doc_id]
        if uid:
            del_query += " AND user_id = ?"
            del_params.append(uid)
        cursor = await conn.execute(del_query, del_params)
        await conn.commit()
        return JSONResponse({"unbound": True, "deleted": cursor.rowcount > 0})
    except Exception:
        log.debug("unbind_document_failed", exc_info=True)
        return JSONResponse({"error": "Failed to unbind document"}, status_code=500)


# ------------------------------------------------------------------
# Session-knowledge-pack bindings
# ------------------------------------------------------------------

@router.get("/session/{session_id}/packs")
async def get_session_packs(session_id: str, request: Request) -> JSONResponse:
    """List knowledge packs bound to one of the authenticated user's sessions."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"packs": []})

    uid = _user_id(request)
    pack_mgr = getattr(request.app.state, "pack_manager", None)
    installed = {p["pack_id"]: p for p in (pack_mgr.installed if pack_mgr else [])}

    try:
        query = (
            "SELECT pack_id, created_at FROM session_knowledge_packs "
            "WHERE session_id = ?"
        )
        params: list = [session_id]
        if uid:
            query += " AND user_id = ?"
            params.append(uid)
        query += " ORDER BY created_at"
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        packs = []
        for row in rows:
            pid = row[0]
            info = installed.get(pid, {})
            packs.append({
                "pack_id": pid,
                "name": info.get("name", pid),
                "chunk_count": info.get("chunk_count", 0),
                "type": info.get("type", "augpack"),
                "created_at": row[1],
            })
        return JSONResponse({"packs": packs})
    except Exception:
        log.warning("get_session_packs_failed", session_id=session_id, exc_info=True)
        return JSONResponse({"packs": []})


@router.post("/session/{session_id}/packs")
async def bind_pack_to_session(session_id: str, request: Request) -> JSONResponse:
    """Bind a knowledge pack to a session the caller owns."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body = await request.json()
    pack_id = body.get("pack_id", "").strip()
    if not pack_id:
        return JSONResponse({"error": "pack_id required"}, status_code=400)

    try:
        await conn.execute(
            "INSERT OR IGNORE INTO session_knowledge_packs "
            "(session_id, pack_id, user_id) VALUES (?, ?, ?)",
            (session_id, pack_id, uid),
        )
        await conn.commit()
        return JSONResponse({"bound": True, "pack_id": pack_id})
    except Exception:
        log.warning("bind_pack_failed", session_id=session_id, pack_id=pack_id, exc_info=True)
        return JSONResponse({"error": "Failed to bind pack"}, status_code=500)


@router.delete("/session/{session_id}/packs/{pack_id}")
async def unbind_pack_from_session(session_id: str, pack_id: str, request: Request) -> JSONResponse:
    """Remove a knowledge pack binding from a session the caller owns."""
    conn = _get_conn(request)
    if not conn:
        return JSONResponse({"error": "Database not available"}, status_code=503)

    uid = _user_id(request)
    try:
        query = (
            "DELETE FROM session_knowledge_packs "
            "WHERE session_id = ? AND pack_id = ?"
        )
        params: list = [session_id, pack_id]
        if uid:
            query += " AND user_id = ?"
            params.append(uid)
        cursor = await conn.execute(query, params)
        await conn.commit()
        return JSONResponse({"unbound": True, "deleted": cursor.rowcount > 0})
    except Exception:
        log.warning("unbind_pack_failed", session_id=session_id, pack_id=pack_id, exc_info=True)
        return JSONResponse({"error": "Failed to unbind pack"}, status_code=500)
