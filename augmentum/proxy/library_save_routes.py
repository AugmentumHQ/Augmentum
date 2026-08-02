"""Save-to-Library REST routes.

Anchored to the coder preview pane: if the user can see a static
artifact running in the preview, one click here saves it to their
Library. Dynamic previews (uvicorn/node/etc.) are rejected with a
clear hint pointing at v2 ephemeral-container launch.

Routes:

* ``POST /api/library/save/preflight``   — query preview state for UI gating
* ``POST /api/library/save``             — workhorse; snapshot + catalog row
* ``GET  /api/library/publications``     — list user's saves
* ``GET  /api/library/publications/{id}`` — one save's metadata
* ``PATCH /api/library/publications/{id}`` — rename / edit description
* ``DELETE /api/library/publications/{id}`` — drop row + storage
* ``GET  /api/library/play/{id}``         — sandboxed iframe launcher
* ``GET  /api/library/publications/{id}/assets/{path:path}`` — serve content
* ``POST /api/library/publications/{id}/launch`` — bump launch counter
* ``POST /api/library/publications/{id}/open-in-coder`` — clone source project into a fresh checkout
* ``GET  /api/library/publications/{id}/download`` — bundle.zip

Every route is user-scoped via ``request.scope["user"].id``. Cross-tenant
reads return 404; never leak existence.
"""

from __future__ import annotations

import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from augmentum.config import settings
from augmentum.library.coder_bridge import (
    PreviewSnapshot,
    find_entry_point,
    gather_preview_state,
    snapshot_container_path,
    write_bundle_zip,
)
from augmentum.library.publications import (
    PublicationStore,
    SizeBudgetExceeded,
    TitleCollision,
)
from augmentum.proxy.content_isolation_routes import (
    _isolated_origin,
    check_content_isolated_auth,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["library"])


# ── Request models ────────────────────────────────────────────────────


class PreflightRequest(BaseModel):
    workspace_id: str
    proposed_title: str = ""   # optional — preflight checks for title collision


class SaveRequest(BaseModel):
    workspace_id: str
    title: str = Field(min_length=1, max_length=160)
    description: str = ""
    on_collision: Literal["overwrite", "abort"] = "abort"
    # Reserved for future "force static save on a dynamic-looking
    # preview" power-user toggle; not honored in v1 (the static check
    # is server-side authoritative).
    force_static: bool = False


class PatchRequest(BaseModel):
    title: str | None = None
    description: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request) -> PublicationStore | None:
    return getattr(request.app.state, "publication_store", None)


async def _owns_workspace(request: Request, workspace_id: str, *, user_id: str) -> bool:
    """Same gating as coder_routes — workspace ownership check via DB."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    conn = getattr(backend, "conn", None)
    if conn is None or not user_id:
        return False
    try:
        cursor = await conn.execute(
            "SELECT 1 FROM project_checkouts WHERE id = ? AND user_id = ? LIMIT 1",
            (workspace_id, user_id),
        )
        return await cursor.fetchone() is not None
    except Exception:
        return False


def _publication_to_view(row: dict[str, Any]) -> dict[str, Any]:
    """Strip server-only fields (absolute storage_path) before returning
    a publication row to the client. The client never needs the on-disk
    path; the asset / launch URLs are sufficient."""
    return {
        "id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "description": row["description"],
        "screenshot_url": (
            f"/api/library/publications/{row['id']}/assets/__screenshot.png"
            if row.get("screenshot_path") else ""
        ),
        "entry_point": row["entry_point"],
        "storage_kind": row["storage_kind"],
        "size_bytes": row["size_bytes"],
        "version": row["version"],
        "shared": bool(row.get("shared")),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_launched_at": row.get("last_launched_at"),
        "launch_count": row.get("launch_count", 0),
        "launch_url": f"/api/library/play/{row['id']}",
        "download_url": f"/api/library/publications/{row['id']}/download",
    }


# ── Preflight ─────────────────────────────────────────────────────────


@router.post("/api/library/save/preflight")
async def save_preflight(req: PreflightRequest, request: Request) -> JSONResponse:
    """Inspect the active preview without saving. UI calls this on save-button
    hover / title-input change to decide whether to enable the save action."""
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    if not await _owns_workspace(request, req.workspace_id, user_id=uid):
        raise HTTPException(status_code=404, detail="workspace not found")

    snap = await gather_preview_state(
        request=request, workspace_id=req.workspace_id, user_id=uid,
    )
    body = snap.to_dict()

    # Title collision is checked here so the UI can show the
    # overwrite/rename prompt before the user even clicks save.
    body["title_collision"] = False
    body["existing_publication_id"] = None
    if req.proposed_title.strip():
        store = _get_store(request)
        if store is not None:
            existing = await store.get_by_title(
                user_id=uid, title=req.proposed_title.strip(),
            )
            if existing:
                body["title_collision"] = True
                body["existing_publication_id"] = existing["id"]
                body["existing_updated_at"] = existing["updated_at"]
                body["existing_version"] = existing["version"]
    return JSONResponse(body)


# ── Save ──────────────────────────────────────────────────────────────


@router.post("/api/library/save")
async def save_to_library(req: SaveRequest, request: Request) -> JSONResponse:
    """Snapshot the workspace's static preview into the user's Library."""
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    if not await _owns_workspace(request, req.workspace_id, user_id=uid):
        raise HTTPException(status_code=404, detail="workspace not found")

    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")

    snap = await gather_preview_state(
        request=request, workspace_id=req.workspace_id, user_id=uid,
    )
    if not snap.saveable:
        return _refusal_for_snap(snap)

    assert snap.served_dir is not None  # narrowed by snap.saveable check

    # Extract the served dir from the container into a host temp dir,
    # then hand the resolved path to the store. The TemporaryDirectory
    # cleans up regardless of save success.
    with tempfile.TemporaryDirectory(prefix="aug-libsave-") as tmpdir:
        host_extracted: Path
        try:
            host_extracted = await snapshot_container_path(
                request=request,
                workspace_id=req.workspace_id,
                container_path=snap.served_dir,
                host_dest_dir=Path(tmpdir),
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=409,
                detail=f"served path {snap.served_dir!r} not found in container",
            ) from None
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        entry_point = find_entry_point(host_extracted)
        kind = _infer_kind_from_snapshot(host_extracted, entry_point)

        try:
            row = await store.create_or_overwrite(
                user_id=uid,
                workspace_id=req.workspace_id,
                title=req.title.strip(),
                description=req.description.strip(),
                kind=kind,
                source_path=host_extracted,
                entry_point=entry_point,
                on_collision=req.on_collision,
                max_bytes=int(settings.library_publication_max_bytes),
                user_budget_bytes=int(settings.library_publication_user_budget_bytes),
            )
        except TitleCollision as exc:
            return JSONResponse(
                {
                    "error": "title_collision",
                    "existing_publication_id": exc.existing["id"],
                    "existing_updated_at": exc.existing["updated_at"],
                    "existing_version": exc.existing["version"],
                    "hint": "Pass on_collision='overwrite' or pick a different title.",
                },
                status_code=409,
            )
        except SizeBudgetExceeded as exc:
            return JSONResponse(
                {
                    "error": "size_budget_exceeded",
                    "scope": exc.scope,
                    "attempted_bytes": exc.attempted_bytes,
                    "limit_bytes": exc.limit_bytes,
                    "hint": _budget_hint(exc.scope),
                },
                status_code=413,
            )

    log.info(
        "library_save_completed",
        publication_id=row["id"],
        user_id=uid,
        kind=row["kind"],
        size_bytes=row["size_bytes"],
        action=row.get("_action"),
        workspace_id=req.workspace_id,
    )
    return JSONResponse({
        "publication_id": row["id"],
        "kind": row["kind"],
        "title": row["title"],
        "version": row["version"],
        "size_bytes": row["size_bytes"],
        "action": row.get("_action", "created"),
        "launch_url": f"/api/library/play/{row['id']}",
        "library_url": "/library?from=coder",
    })


def _refusal_for_snap(snap: PreviewSnapshot) -> JSONResponse:
    """Translate a non-saveable preview into a typed 409 error."""
    if snap.preview_kind == "dynamic":
        return JSONResponse(
            {
                "error": "dynamic_preview",
                "preview_kind": "dynamic",
                "hint": (
                    "Save-to-Library v1 only supports static services. "
                    "Switch to a static build (e.g. `npm run build && npx serve dist`) "
                    "and try again. Dynamic-runtime saves land in v2."
                ),
            },
            status_code=409,
        )
    if snap.preview_kind == "unknown":
        return JSONResponse(
            {
                "error": "preview_unclassified",
                "preview_kind": "unknown",
                "hint": "Couldn't probe the preview. Check the dev server is healthy and try again.",
            },
            status_code=409,
        )
    return JSONResponse(
        {
            "error": "no_preview",
            "preview_kind": "none",
            "hint": (
                "Start a static service first (service_start with python http.server, "
                "npx serve, etc.) and confirm it's reachable in the preview pane."
            ),
        },
        status_code=409,
    )


def _infer_kind_from_snapshot(extracted: Path, entry_point: str) -> Literal["game", "app", "doc", "other"]:
    """Defaults to ``game`` for HTML/JS bundles since that's the dominant
    v1 case. Future heuristic could inspect the entry HTML for canvas
    + game-loop patterns to disambiguate from ``app``; not worth the
    code for v1."""
    if (extracted / entry_point).is_file():
        return "game"
    return "other"


def _budget_hint(scope: str) -> str:
    if scope == "per_publication":
        cap_mb = int(settings.library_publication_max_bytes) / (1024 * 1024)
        return (
            f"This save exceeds the per-publication cap (~{cap_mb:.0f} MB). "
            "Slim the bundle (drop sourcemaps, prune assets) or raise "
            "the cap in settings."
        )
    cap_gb = int(settings.library_publication_user_budget_bytes) / (1024 ** 3)
    return (
        f"You're over your Library storage budget (~{cap_gb:.1f} GB). "
        "Delete older saves from the Library or raise the budget in settings."
    )


# ── Read / patch / delete ─────────────────────────────────────────────


@router.get("/api/library/publications")
async def list_publications(
    request: Request, kind: str = "", limit: int = 200,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    kind_filter = kind.strip() or None
    rows = await store.list_for_user(
        user_id=uid, kind=kind_filter, limit=max(1, min(int(limit), 500)),
    )
    return JSONResponse({"publications": [_publication_to_view(r) for r in rows]})


@router.get("/api/library/publications/{publication_id}")
async def get_publication(publication_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    row = await store.get(publication_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")
    return JSONResponse(_publication_to_view(row))


@router.patch("/api/library/publications/{publication_id}")
async def patch_publication(
    publication_id: str, req: PatchRequest, request: Request,
) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    try:
        row = await store.patch(
            publication_id,
            user_id=uid,
            title=(req.title.strip() if req.title is not None else None),
            description=req.description,
        )
    except TitleCollision as exc:
        return JSONResponse(
            {
                "error": "title_collision",
                "existing_publication_id": exc.existing["id"],
                "hint": "Another publication already uses this title.",
            },
            status_code=409,
        )
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")
    return JSONResponse(_publication_to_view(row))


@router.delete("/api/library/publications/{publication_id}")
async def delete_publication(publication_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    ok = await store.delete(publication_id, user_id=uid)
    if not ok:
        raise HTTPException(status_code=404, detail="publication not found")
    return JSONResponse({"deleted": True, "publication_id": publication_id})


# ── Play / assets / launch / download ─────────────────────────────────


_PLAY_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<style>
  html, body {{ margin: 0; padding: 0; height: 100%; background: #000; color: #ddd; font-family: system-ui, sans-serif; }}
  iframe {{ width: 100vw; height: 100vh; border: 0; display: block; }}
</style>
</head>
<body>
<iframe id="play" src="{src}" sandbox="{sandbox}" allowfullscreen></iframe>
<script>
fetch({launch_url!r}, {{method: "POST", credentials: "same-origin"}}).catch(() => {{}});
</script>
</body>
</html>
"""


# Sandbox flags for the played-bundle iframe.
#   * Same-origin fallback (isolation off): NO allow-same-origin — the
#     bundle gets a null origin so it can't reach the user's main
#     session cookies. The tradeoff is that ES-module + localStorage
#     apps break (their fetches are CORS-blocked / storage throws).
#   * Isolated origin (isolation on): the iframe loads from :6444, a
#     genuinely different origin, so allow-same-origin is SAFE — the
#     content gets a real (but foreign-to-Augmentum) origin where
#     modules and localStorage work, with no cookie crossover.
_PLAY_SANDBOX_BASE = (
    "allow-scripts allow-pointer-lock allow-popups "
    "allow-popups-to-escape-sandbox allow-forms"
)
_PLAY_SANDBOX_ISOLATED = _PLAY_SANDBOX_BASE + " allow-same-origin"


@router.get("/api/library/play/{publication_id}", response_class=HTMLResponse)
async def play_publication(publication_id: str, request: Request) -> HTMLResponse:
    """Sandboxed iframe wrapper for a played publication.

    When content-iframe isolation is enabled the inner iframe loads the
    bundle from the isolated origin (:6444) via a one-time token, so
    ES-module + localStorage apps run correctly without exposing the
    user's main session. When disabled it falls back to the same-origin
    null-origin sandbox (today's behaviour).
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    row = await store.get(publication_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")

    entry_path = (
        f"/api/library/publications/{publication_id}/assets/"
        f"{row['entry_point']}"
    )
    # Default: same-origin, null-origin sandbox.
    entry_url = entry_path
    sandbox = _PLAY_SANDBOX_BASE

    # Upgrade to the isolated origin when enabled + reachable. Best-effort:
    # any failure (disabled, store missing, origin underivable) degrades
    # to the same-origin path so play never hard-fails on a config gap.
    if getattr(settings, "content_iframe_isolation_enabled", False):
        token_store = getattr(request.app.state, "preview_token_store", None)
        isolated = _isolated_origin(request)
        if token_store is not None and isolated:
            try:
                token, _expires = token_store.mint(
                    user_id=uid,
                    workspace_id=publication_id,
                    ttl_s=float(getattr(settings, "coder_preview_token_ttl_seconds", 60)),
                    kind="publication",
                )
                entry_url = f"{isolated}{entry_path}?_pvt={token}"
                sandbox = _PLAY_SANDBOX_ISOLATED
            except Exception:
                log.warning("publication_play_isolated_mint_failed",
                            publication_id=publication_id, exc_info=True)

    launch_url = f"/api/library/publications/{publication_id}/launch"
    safe_title = (row["title"] or "Library Publication").replace("<", "&lt;").replace(">", "&gt;")
    html = _PLAY_TEMPLATE.format(
        title=safe_title, src=entry_url, launch_url=launch_url, sandbox=sandbox,
    )
    return HTMLResponse(html)


# A reserved sub-path within the asset namespace for the saved
# screenshot. Maps to {storage_path}/screenshot.png. Kept separate from
# user-supplied paths so a bundle file named "__screenshot.png" can't
# shadow the catalog field.
_SCREENSHOT_SENTINEL = "__screenshot.png"


@router.get("/api/library/publications/{publication_id}/assets/{path:path}")
async def get_publication_asset(
    publication_id: str, path: str, request: Request,
) -> Response:
    """Serve a file from the publication's ``content/`` dir. Traversal-safe
    via :meth:`LibraryStorage.asset_path`. Cross-tenant returns 404.

    On the isolated preview origin the request carries no main session
    cookie, so it's authenticated via the one-time token / preview
    session and the user is resolved from the redeemed record.
    """
    if request.scope.get("augmentum_preview_isolated"):
        auth = await check_content_isolated_auth(request, "publication", publication_id)
        if auth is not None:
            return auth
        uid = request.scope.get("augmentum_preview_user_id", "")
    else:
        uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    row = await store.get(publication_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")

    storage = store.storage

    # Reserved screenshot path is served from the publication root,
    # not from content/.
    if path == _SCREENSHOT_SENTINEL:
        screenshot_rel = row.get("screenshot_path", "")
        if not screenshot_rel:
            raise HTTPException(status_code=404, detail="no screenshot")
        candidate = storage.publication_dir(uid, publication_id) / screenshot_rel
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="screenshot missing")
        return FileResponse(str(candidate), media_type="image/png")

    fs_path = storage.asset_path(
        user_id=uid, publication_id=publication_id, rel_path=path,
    )
    if fs_path is None:
        raise HTTPException(status_code=404, detail="asset not found")

    media_type, _ = mimetypes.guess_type(fs_path.name)
    return FileResponse(str(fs_path), media_type=media_type or "application/octet-stream")


@router.post("/api/library/publications/{publication_id}/launch")
async def record_launch(publication_id: str, request: Request) -> JSONResponse:
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    row = await store.get(publication_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")
    await store.record_launch(publication_id, user_id=uid)
    return JSONResponse({"recorded": True})


@router.post("/api/library/publications/{publication_id}/open-in-coder")
async def open_publication_in_coder(
    publication_id: str, request: Request,
) -> JSONResponse:
    """Phase 1 / PR-1.3: clone the publication's source Project into a
    fresh checkout and return the new workspace id.

    Closes the "Library Play is a read-only dead end" gap from the
    integrated coding nervous system spec. The publication points at
    a ``project_refs`` row (kind=publication) via ``project_ref_id``;
    that ref points at a Project; the Project's bare repo gets cloned
    into a new container.

    Resolution order:
      1. ``project_ref_id`` (Phase 2+ publications)
      2. ``workspace_id`` -> ``project_checkouts.project_id`` (legacy
         publications saved before PR-1.3; the workspace may be gone
         but the Project survives if PR-1.2 backfilled it)

    Returns 410 with a clear error when neither lookup yields a Project.
    """
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")

    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    row = await store.get(publication_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")

    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    conn = getattr(backend, "conn", None)
    cm = getattr(request.app.state, "container_manager", None)
    if conn is None or cm is None:
        raise HTTPException(status_code=503, detail="coder not initialized")

    project_id = ""
    ref_id = row.get("project_ref_id") or ""
    if ref_id:
        cursor = await conn.execute(
            "SELECT project_id FROM project_refs WHERE id = ?",
            (ref_id,),
        )
        r = await cursor.fetchone()
        if r:
            project_id = r[0] or ""

    if not project_id and row.get("workspace_id"):
        cursor = await conn.execute(
            "SELECT project_id FROM project_checkouts "
            "WHERE id = ? AND user_id = ?",
            (row["workspace_id"], uid),
        )
        r = await cursor.fetchone()
        if r:
            project_id = r[0] or ""

    if not project_id:
        raise HTTPException(
            status_code=410,
            detail=(
                "no source project found for this publication "
                "(legacy publication saved before the Project substrate)"
            ),
        )

    new_name = (row.get("title") or "Edit").strip()[:60] or "Edit"
    try:
        info = await cm.create_workspace(
            name=new_name,
            user_id=uid,
            project_id=project_id,
        )
    except Exception as exc:
        log.warning(
            "open_publication_in_coder_failed",
            publication_id=publication_id, project_id=project_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"failed to open in coder: {exc}",
        ) from exc

    return JSONResponse({
        "workspace_id": info.id,
        "project_id": project_id,
        "name": new_name,
    })


@router.get("/api/library/publications/{publication_id}/download")
async def download_publication(publication_id: str, request: Request) -> Response:
    """Stream a zip of the publication's content/ dir. Built on demand —
    we don't keep zip files on disk because users typically download
    each save at most a handful of times, and a stale zip after an
    overwrite would mislead."""
    uid = _user_id(request)
    if not uid:
        raise HTTPException(status_code=401, detail="auth required")
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="library not initialized")
    row = await store.get(publication_id, user_id=uid)
    if not row:
        raise HTTPException(status_code=404, detail="publication not found")

    pub_dir = store.storage.publication_dir(uid, publication_id)
    content_dir = pub_dir / "content"
    if not content_dir.is_dir():
        raise HTTPException(status_code=410, detail="publication storage missing")

    # Build into a tempfile that auto-cleans after the response is sent.
    with tempfile.NamedTemporaryFile(
        prefix=f"libdl-{publication_id}-", suffix=".zip", delete=False,
    ) as tmp:
        out_path = Path(tmp.name)
    try:
        await write_bundle_zip(source_dir=content_dir, output_path=out_path)
    except Exception:
        out_path.unlink(missing_ok=True)
        raise

    safe_name = (row["title"] or publication_id).replace("/", "_").replace("\\", "_")
    headers = {
        "Content-Disposition": f'attachment; filename="{safe_name}.zip"',
    }
    return FileResponse(
        str(out_path),
        media_type="application/zip",
        headers=headers,
        # Schedule cleanup once the response finishes streaming.
        background=_DeleteAfterResponse(out_path),
    )


# Tiny BackgroundTask shim — FileResponse accepts a Starlette
# BackgroundTask, and we want the temp zip cleaned up after the
# response is sent so we don't leak across requests.
from starlette.background import BackgroundTask


def _DeleteAfterResponse(path: Path) -> BackgroundTask:  # noqa: N802
    def _cleanup() -> None:
        try:
            path.unlink()
        except OSError:
            pass
    return BackgroundTask(_cleanup)
