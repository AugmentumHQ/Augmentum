"""Session Canvas routes.

The Canvas is a side-docked panel beside chat that holds one artifact and
stays anchored while the user keeps chatting (the foundation other rich-
content surfaces dock into). This module owns the *binding* — which artifact
is pinned to a given session's canvas — persisted server-side so the canvas
survives refresh / restart / device switch.

Binding storage + queries live on the shared ``ArtifactStore``
(``request.app.state.artifact_store``); see ``session_canvas`` (migration
267) and ``artifact_storage.py::{get,set,clear}_canvas_binding``. All access
is user-scoped — the canvas never resolves another tenant's artifact.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


def _user_id(request: Request) -> str:
    """Extract user_id from the authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


def _get_store(request: Request):
    store = getattr(request.app.state, "artifact_store", None)
    if not store:
        raise HTTPException(status_code=503, detail="Artifact storage not available")
    return store


def _app_files(artifact: dict) -> list[dict]:
    """Return the editable file manifest for an application artifact, or [].

    The canvas edit write-path only supports app bundles (HTML/JS/CSS) —
    their ``source_json`` carries ``type:"application"`` + a ``files`` list
    (the same shape the builder + revert paths use). Other formats (PDF,
    DOCX, images, single-file uploads) return [] and are not editable.
    """
    raw = artifact.get("source_json")
    if not raw:
        return []
    try:
        source = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return []
    if not isinstance(source, dict) or source.get("type") != "application":
        return []
    files = source.get("files")
    return files if isinstance(files, list) and files else []


async def _resolve_canvas_artifact(store, session_id: str, *, user_id: str) -> tuple[dict | None, bool]:
    """Resolve the artifact a session's canvas shows.

    Prefers an explicit pin; self-heals a stale pin (artifact deleted) by
    clearing the binding; falls back to the session's most recent artifact.
    Returns ``(artifact, pinned)`` or ``(None, False)`` when the session has
    nothing to show. Shared by the GET resolver and the edit write-path so
    both agree on what "the canvas artifact" is.
    """
    pinned_id = await store.get_canvas_binding(session_id, user_id=user_id)
    if pinned_id:
        artifact = await store.get(pinned_id, user_id=user_id)
        if artifact:
            return artifact, True
        # Stale pin — drop it and fall through to latest.
        await store.clear_canvas_binding(session_id, user_id=user_id)
    artifacts = await store.list_for_session(session_id, user_id=user_id)
    if artifacts:
        return artifacts[0], False
    return None, False


async def _summarize(store, artifact: dict, *, user_id: str, pinned: bool) -> dict:
    """Build the lightweight canvas payload for an artifact row."""
    artifact_id = artifact["id"]
    try:
        versions = await store.list_versions(artifact_id, user_id=user_id)
    except Exception as exc:  # noqa: BLE001 — versions are best-effort decoration
        log.warning("canvas.versions_failed", artifact_id=artifact_id, error=str(exc))
        versions = []
    # Newest-first from the store. The dock's version stepper needs the ids;
    # cap the payload so a long edit history doesn't bloat every canvas poll.
    version_list = [
        {
            "id": v.get("id"),
            "version_index": v.get("version_index"),
            "label": v.get("label", ""),
        }
        for v in versions[:50]
    ]
    return {
        "artifact_id": artifact_id,
        "display_name": artifact.get("display_name") or artifact.get("filename") or "Artifact",
        "format": artifact.get("format", ""),
        "version_count": len(versions),
        "versions": version_list,
        "pinned": pinned,
        # The dock shows the "Ask for a change" composer only when an
        # artifact is an editable app bundle (see _app_files).
        "editable": bool(_app_files(artifact)),
        "preview_url": f"/api/artifacts/{artifact_id}/preview",
    }


@router.get("/{session_id}")
async def get_canvas(session_id: str, request: Request):
    """Resolve the artifact shown in this session's canvas.

    Prefers an explicit pin; falls back to the session's most recent
    artifact (so opening the canvas on a session that produced an artifact
    "just works" before anything is pinned). Returns ``artifact_id: null``
    when the session has no artifact to show.
    """
    store = _get_store(request)
    user_id = _user_id(request)
    if not user_id:
        return JSONResponse({"artifact_id": None})

    artifact, pinned = await _resolve_canvas_artifact(store, session_id, user_id=user_id)
    if not artifact:
        return JSONResponse({"artifact_id": None})
    return JSONResponse(await _summarize(store, artifact, user_id=user_id, pinned=pinned))


@router.put("/{session_id}")
async def set_canvas(session_id: str, request: Request):
    """Pin a specific artifact to this session's canvas."""
    store = _get_store(request)
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        body = await request.json()
    except Exception:
        body = {}
    artifact_id = (body or {}).get("artifact_id", "") if isinstance(body, dict) else ""
    if not artifact_id:
        raise HTTPException(status_code=400, detail="artifact_id is required")

    # Verify ownership — never pin another tenant's (or a non-existent) artifact.
    artifact = await store.get(artifact_id, user_id=user_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    await store.set_canvas_binding(session_id, artifact_id, user_id=user_id)
    return JSONResponse(await _summarize(store, artifact, user_id=user_id, pinned=True))


@router.post("/{session_id}/edit")
async def edit_canvas(session_id: str, request: Request):
    """Apply a natural-language change to the canvas artifact, in place.

    Reuses the application builder's quick-edit (SEARCH/REPLACE grammar +
    fuzzy apply — the same machinery as ``POST /api/artifacts/fix``),
    snapshots a version so the edit is revertable, then rewrites the
    artifact's bundle + ``source_json`` (the same rewrite the revert path
    uses) so the live preview reflects the change on reload.

    Only application artifacts (HTML/JS/CSS bundles) are editable; other
    formats return 400 and never show the composer (see ``editable`` in the
    summary). User-scoped throughout — never touches another tenant's
    artifact.
    """
    store = _get_store(request)
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    description = str(body.get("description") or "").strip()
    model = str(body.get("model") or "")
    if not description:
        raise HTTPException(status_code=400, detail="description is required")

    artifact, pinned = await _resolve_canvas_artifact(store, session_id, user_id=user_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="No artifact on this canvas")

    files = _app_files(artifact)
    if not files:
        raise HTTPException(status_code=400, detail="This artifact can't be edited from the canvas")

    tool_registry = getattr(request.app.state, "tool_registry", None)
    builder = tool_registry.get("build_application") if tool_registry else None
    if not builder:
        raise HTTPException(status_code=503, detail="Application builder not available")

    artifact_id = artifact["id"]

    # Seed history with the pre-edit state on the FIRST edit so the user can
    # always step back to the original (mirrors revert's "make it reversible").
    try:
        if not await store.list_versions(artifact_id, user_id=user_id):
            await store.save_version(artifact_id, files, user_id=user_id, label="Original")
    except Exception as exc:  # noqa: BLE001 — seeding is best-effort
        log.warning("canvas.seed_version_failed", artifact_id=artifact_id, error=str(exc))

    # Builder owns the LLM quick-edit. Lazy import keeps canvas routes
    # import-light and avoids a load-order coupling to artifact_routes.
    from augmentum.proxy.artifact_routes import (
        _build_quick_edit_prompt,
        _summarize_patch_details,
    )
    from augmentum.tools.application_scaffolds import GRAMMAR_SEARCH_REPLACE

    messages = _build_quick_edit_prompt(files, description)
    max_tokens = max(getattr(builder, "_max_tokens", 8192) // 2, 2048)
    try:
        response = await builder._call_llm(
            messages, max_tokens=max_tokens, model=model, grammar=GRAMMAR_SEARCH_REPLACE,
        )
    except Exception as exc:  # noqa: BLE001 — surface a clean error to the dock
        log.warning("canvas.edit_llm_failed", artifact_id=artifact_id, error=str(exc))
        raise HTTPException(status_code=502, detail="The model couldn't make that change") from exc

    # Snapshot pre-edit content so we can report what changed — the apply
    # mutates `files` in place.
    pre_edit = [{"path": f.get("path", ""), "content": f.get("content", "")} for f in files]
    patches = builder._apply_file_patches(files, response)
    if patches <= 0:
        raise HTTPException(
            status_code=422, detail="No changes were applied — try rephrasing the request"
        )

    # Compact "what changed" summary for the dock chip — applied edits grouped
    # by file. Best-effort: a parse miss just yields the count, never an error.
    changed_files: list[dict] = []
    try:
        details = _summarize_patch_details(response, pre_edit, files)
        by_file: dict[str, int] = {}
        for d in details:
            if d.get("applied"):
                by_file[d.get("file", "")] = by_file.get(d.get("file", ""), 0) + 1
        changed_files = [{"path": p or "file", "count": c} for p, c in by_file.items()]
    except Exception as exc:  # noqa: BLE001 — the chip is decoration
        log.warning("canvas.patch_details_failed", artifact_id=artifact_id, error=str(exc))

    # Persist: snapshot the new state, then rebuild the bundle + source so
    # the live preview reflects the edit (identical to the revert rewrite).
    await store.save_version(artifact_id, files, user_id=user_id, label=description[:200])
    import io
    import zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.get("content"):
                zf.writestr(f.get("path", ""), f["content"])
    new_source = json.dumps({
        "type": "application",
        "name": artifact.get("display_name", ""),
        "files": files,
    })
    await store.update_file(artifact_id, zip_buf.getvalue(), user_id=user_id)
    await store.update_source(artifact_id, new_source, user_id=user_id)

    log.info("canvas.edited", artifact_id=artifact_id, patches=patches)

    # Re-fetch so the summary carries the bumped version_count (drives the
    # dock's cache-bust + version badge).
    refreshed = await store.get(artifact_id, user_id=user_id) or artifact
    summary = await _summarize(store, refreshed, user_id=user_id, pinned=pinned)
    summary["patches_applied"] = patches
    summary["changed_files"] = changed_files
    return JSONResponse(summary)


@router.get("/{session_id}/version/{version_id}/preview")
async def preview_canvas_version(session_id: str, version_id: str, request: Request):
    """Render a past version of the canvas artifact, read-only.

    Lets the dock's version stepper *browse* history without mutating the
    artifact (that's what restore/revert is for). Serves a self-contained
    assembled HTML so the snapshot runs in the sandboxed iframe without the
    sibling-file route. The version must belong to this session's canvas
    artifact and this user.
    """
    store = _get_store(request)
    user_id = _user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    artifact, _pinned = await _resolve_canvas_artifact(store, session_id, user_id=user_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="No artifact on this canvas")

    version = await store.get_version(version_id, user_id=user_id)
    if not version or version.get("artifact_id") != artifact["id"]:
        raise HTTPException(status_code=404, detail="Version not found")

    from augmentum.tools.artifact_application import assemble_application_html

    html = assemble_application_html(version.get("files") or [])
    if not html:
        raise HTTPException(status_code=422, detail="Could not render this version")
    # no-store: a snapshot is immutable, but the same version_id should never
    # be cached against a different artifact after a session rebind.
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})
