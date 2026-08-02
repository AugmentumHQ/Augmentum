"""Artifact Studio API — source retrieval, save, re-render."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/studio", tags=["studio"])

# ---------------------------------------------------------------------------
# Studio Tool Palette substrate — Phase 1
# ---------------------------------------------------------------------------
# Staging registry for the Image-tool Generate tab. Generations land in the
# user's library on creation (image_generation always persists), and Studio
# tracks the image_id here so an unacted-upon generation gets swept after the
# TTL window. Pro-tool UX promise: nothing silently rots in the library.
#
# Shape: app.state.studio_staging[image_id] = {
#     "user_id", "artifact_id", "prompt", "created_at" (epoch seconds)
# }
# In-memory only; process restart = treat all staging entries as abandoned
# (the sweep on next startup catches them and deletes the underlying images).

_STAGING_TTL_SECONDS = 30 * 60       # un-acted generations swept after 30 min
_STAGING_SWEEP_INTERVAL = 5 * 60     # sweep cadence


def _ensure_staging_registry(request: Request) -> dict:
    """Lazy-init the staging registry on app.state."""
    state = request.app.state
    reg = getattr(state, "studio_staging", None)
    if reg is None:
        reg = {}
        state.studio_staging = reg
    return reg


def _get_store(request: Request):
    return getattr(request.app.state, "artifact_store", None)


async def _snapshot_studio_version(
    store, artifact_id: str, source_json: str, user_id: str,
) -> None:
    """Wrap Studio source as a single pseudo-file and snapshot a version.

    Failure is non-fatal: the user's save has already persisted; losing
    a history row is recoverable, losing the edit is not. Logged warning
    so prod degradation is visible.
    """
    if not source_json:
        return
    try:
        await store.save_version(
            artifact_id,
            [{"path": "source.json", "role": "source", "content": source_json}],
            user_id=user_id,
        )
    except Exception as exc:
        log.warning(
            "studio_version_snapshot_failed",
            artifact_id=artifact_id, error=str(exc),
        )


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


# ---------------------------------------------------------------------------
# Studio Design block — lazy migration + render-side projection
# ---------------------------------------------------------------------------
# `source.design` is the new canonical home for theme + typography + density.
# Old artifacts in the field use `source.theme` (string or {preset}) and, for
# ebooks, `source.reading` ({font, size, leading}). We synthesize a design
# block on every load so renderers can always read from one shape. The save
# path writes design back; we don't mutate the original source here so old
# fields remain in place for one more release cycle as belt-and-braces.

_READING_FONT_TO_FAMILY = {"serif": "serif", "sans": "sans", "dyslexic": "dyslexic"}
_READING_SIZE_TO_SCALE = {"xs": 0.85, "sm": 0.92, "md": 1.0, "lg": 1.15, "xl": 1.45}
_READING_LEADING_TO_LH = {"compact": "tight", "normal": "comfortable", "relaxed": "airy"}


def _extract_theme_name(source: dict) -> str:
    """Pull the theme preset name from any of the legacy shapes."""
    theme = source.get("theme")
    if isinstance(theme, dict):
        return str(theme.get("preset") or "")
    if isinstance(theme, str):
        return theme
    return ""


def _resolve_design(source: dict) -> dict:
    """Return a fully-populated design block for the given source.

    Precedence: explicit `source.design` > derived from `source.theme` +
    `source.reading` > defaults. Always returns every design key so renderers
    can index directly.
    """
    from augmentum.tools.artifact_theme import DEFAULT_DESIGN, normalize_design

    existing = source.get("design")
    if isinstance(existing, dict):
        return normalize_design(existing, fallback_theme=_extract_theme_name(source))

    base = dict(DEFAULT_DESIGN)
    base["theme"] = _extract_theme_name(source)

    reading = source.get("reading")
    if isinstance(reading, dict):
        font = str(reading.get("font") or "").strip().lower()
        if font in _READING_FONT_TO_FAMILY:
            base["font_family"] = _READING_FONT_TO_FAMILY[font]
        size = str(reading.get("size") or "").strip().lower()
        if size in _READING_SIZE_TO_SCALE:
            base["font_size_scale"] = _READING_SIZE_TO_SCALE[size]
        leading = str(reading.get("leading") or "").strip().lower()
        if leading in _READING_LEADING_TO_LH:
            base["line_height"] = _READING_LEADING_TO_LH[leading]

    return normalize_design(base, fallback_theme=base["theme"])


def _project_design_to_reading(design: dict) -> dict:
    """Project a design block back into the legacy `reading` shape.

    Used by the EPUB renderer, which already knows how to consume
    `reading.{font,size,leading}`. Avoids needing a second EPUB rewrite.
    """
    out: dict = {}
    ff = design.get("font_family", "system")
    if ff in ("serif", "sans", "dyslexic"):
        out["font"] = ff
    elif ff == "mono":
        # mono has no EPUB equivalent — sans is the safer fallback than
        # serif (mono and sans share geometric/upright character).
        out["font"] = "sans"

    scale = float(design.get("font_size_scale", 1.0))
    if scale < 0.9:
        out["size"] = "xs"
    elif scale < 0.97:
        out["size"] = "sm"
    elif scale < 1.1:
        out["size"] = "md"
    elif scale < 1.3:
        out["size"] = "lg"
    else:
        out["size"] = "xl"

    lh = design.get("line_height", "comfortable")
    out["leading"] = {"tight": "compact", "comfortable": "normal", "airy": "relaxed"}.get(lh, "normal")
    return out


@router.get("/themes/list")
async def list_themes() -> JSONResponse:
    """Return available artifact themes."""
    from augmentum.tools.artifact_theme import THEMES

    result = []
    for name, theme in THEMES.items():
        result.append({
            "name": name,
            "accent": theme.accent,
            "accent_light": theme.accent_light,
            "accent_dark": theme.accent_dark,
            "text": theme.text,
            "text_secondary": theme.text_secondary,
            "text_muted": theme.text_muted,
            "background": theme.background,
            "surface": theme.surface,
            "border": theme.border,
        })
    return JSONResponse({"themes": result})


@router.get("/{artifact_id}")
async def get_artifact_source(artifact_id: str, request: Request) -> JSONResponse:
    """Get artifact metadata + source JSON for editing."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    info = await store.get(artifact_id, user_id=uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    source = info.get("source_json")
    if source and isinstance(source, str):
        try:
            source = json.loads(source)
        except json.JSONDecodeError:
            source = None

    return JSONResponse({
        "id": info["id"],
        "filename": info["filename"],
        "display_name": info.get("display_name", info["filename"]),
        "format": info["format"],
        "size_bytes": info.get("size_bytes", 0),
        "source": source,
        "metadata": info.get("metadata", {}),
        "download_url": info.get("download_url", ""),
    })


@router.post("/{artifact_id}/save")
async def save_artifact_source(artifact_id: str, request: Request) -> JSONResponse:
    """Save updated source JSON and re-render the artifact binary."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    info = await store.get(artifact_id, user_id=uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    source = body.get("source")
    if not source or not isinstance(source, dict):
        return JSONResponse({"error": "source object required"}, status_code=400)

    source_type = source.get("type", "")
    # Manual saves snapshot a version; autosaves (every-2s debounced) don't —
    # otherwise the history table balloons. Frontend sets is_autosave=true
    # for the debounced path; absence defaults to False (manual).
    is_autosave = bool(body.get("is_autosave"))

    # For charts, prefer client-rendered PNG over server-side matplotlib
    rendered_png = body.get("rendered_png")
    if rendered_png and source_type == "chart":
        import base64
        try:
            data = base64.b64decode(rendered_png)
            # Skip server-side rendering — use client PNG
            source_str = json.dumps(source, ensure_ascii=False)
            await store.update_source(artifact_id, source_str, user_id=uid)
            await store.update_file(artifact_id, data, user_id=uid)
            if not is_autosave:
                await _snapshot_studio_version(store, artifact_id, source_str, uid)
            return JSONResponse({"ok": True, "size_bytes": len(data)})
        except Exception:
            log.debug("client_png_decode_failed", exc_info=True)
            # Fall through to server-side rendering

    # Re-render based on type. Renderers may surface non-fatal warnings
    # (e.g. PDF Unicode font fallback degrading "café" → "cafe" silently).
    # We collect those and return them in the response so the UI can show
    # a non-error toast — silent degradation was the original bug.
    warnings: list[str] = []
    try:
        if source_type == "document":
            data, doc_warnings = await _render_document(source, request)
            warnings.extend(doc_warnings)
        elif source_type == "presentation":
            data = await _render_presentation(source, request)
        elif source_type == "spreadsheet":
            data = await _render_spreadsheet(source)
        elif source_type == "chart":
            data = await _render_chart_from_source(source)
        elif source_type == "ebook":
            data = await _render_ebook(source, request)
        else:
            return JSONResponse({"error": f"Unknown source type: {source_type}"}, status_code=400)
    except Exception as exc:
        log.error("studio_render_failed", artifact_id=artifact_id, error=str(exc), exc_info=True)
        return JSONResponse({"error": "Render failed. Check server logs for details."}, status_code=500)

    # Update source JSON + re-rendered binary
    source_str = json.dumps(source, ensure_ascii=False)
    await store.update_source(artifact_id, source_str, user_id=uid)
    await store.update_file(artifact_id, data, user_id=uid)

    if not is_autosave:
        await _snapshot_studio_version(store, artifact_id, source_str, uid)

    return JSONResponse({
        "ok": True,
        "size_bytes": len(data),
        "warnings": warnings,
    })


@router.post("/{artifact_id}/preview")
async def preview_artifact(artifact_id: str, request: Request) -> Response:
    """Re-render the artifact from source and return the binary for preview.

    Unlike /save, this does NOT persist changes. It renders from the posted
    source and returns the binary directly (e.g., PDF bytes for iframe display).
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    info = await store.get(artifact_id, user_id=uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    source = body.get("source")
    if not source:
        return JSONResponse({"error": "source required"}, status_code=400)

    source_type = source.get("type", "")

    try:
        if source_type == "document":
            # Preview always renders as PDF for browser display
            preview_source = {**source, "format": "pdf"}
            data, _warnings = await _render_document(preview_source, request)
        elif source_type == "presentation":
            data = await _render_presentation(source, request)
        elif source_type == "spreadsheet":
            data = await _render_spreadsheet(source)
        elif source_type == "chart":
            data = await _render_chart_from_source(source)
        elif source_type == "ebook":
            data = await _render_ebook(source, request)
        else:
            return JSONResponse({"error": "Unknown type"}, status_code=400)
    except Exception as exc:
        log.error("studio_preview_failed", error=str(exc), exc_info=True)
        return JSONResponse({"error": "Preview render failed"}, status_code=500)

    # Return the binary with appropriate content type
    content_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "png": "image/png",
        "epub": "application/epub+zip",
    }
    # For preview, always render documents as PDF (browser can display PDF in iframe)
    # The actual save still uses the correct format
    fmt = source.get("format", info.get("format", "pdf"))
    preview_fmt = fmt
    if source.get("type") == "document":
        preview_fmt = "pdf"
    ct = content_types.get(preview_fmt, "application/octet-stream")

    return Response(content=data, media_type=ct, headers={
        "Cache-Control": "no-store",
    })


# ---------------------------------------------------------------------------
# Version history — list + restore
# ---------------------------------------------------------------------------
# Version snapshots are written by save_artifact_source when is_autosave
# is False (manual save only — see _snapshot_studio_version). The list
# endpoint mirrors the shared /api/artifacts/{id}/versions route but lives
# under /api/studio so the Studio drawer doesn't have to know about two
# route prefixes. Restoring is Studio-specific because the version stores
# source.json (Studio shape), not a file bundle like the app-builder
# revert path expects.


@router.get("/{artifact_id}/versions")
async def list_studio_versions(artifact_id: str, request: Request) -> JSONResponse:
    """List manual-save versions for a Studio artifact, newest first."""
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    info = await store.get(artifact_id, user_id=uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)
    versions = await store.list_versions(artifact_id, user_id=uid)
    return JSONResponse({"artifact_id": artifact_id, "versions": versions})


@router.post("/{artifact_id}/restore-version/{version_id}")
async def restore_studio_version(
    artifact_id: str, version_id: str, request: Request,
) -> JSONResponse:
    """Restore the artifact's source from a saved version.

    Snapshots the current state first so the restore is itself reversible —
    the user can click the auto-snapshot in the drawer to undo. Re-renders
    the binary from the restored source so download / preview reflect the
    rollback immediately.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    info = await store.get(artifact_id, user_id=uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    version = await store.get_version(version_id, user_id=uid)
    if not version:
        return JSONResponse({"error": "Version not found"}, status_code=404)
    if version.get("artifact_id") != artifact_id:
        return JSONResponse(
            {"error": "Version does not belong to this artifact"}, status_code=400,
        )

    # Extract the source.json content from the version's files_json.
    files = version.get("files") or []
    source_file = next(
        (f for f in files if f.get("path") == "source.json"), None,
    )
    if not source_file or not source_file.get("content"):
        return JSONResponse(
            {"error": "Version does not contain a Studio source snapshot"},
            status_code=400,
        )

    try:
        source = json.loads(source_file["content"])
    except (TypeError, ValueError) as exc:
        log.warning(
            "studio_restore_source_decode_failed",
            artifact_id=artifact_id, version_id=version_id, error=str(exc),
        )
        return JSONResponse(
            {"error": "Version source is corrupted"}, status_code=400,
        )
    if not isinstance(source, dict):
        return JSONResponse(
            {"error": "Version source has unexpected shape"}, status_code=400,
        )

    # Snapshot current state BEFORE overwrite so user can undo the restore.
    # Failure here is non-fatal; the user's main concern is that the restore
    # itself succeeds.
    current_source_json = info.get("source_json") or ""
    if current_source_json:
        await _snapshot_studio_version(store, artifact_id, current_source_json, uid)

    # Re-render the binary from the restored source so download / preview
    # reflect the rollback immediately. Mirrors the dispatch in save.
    source_type = source.get("type", "")
    try:
        if source_type == "document":
            data, _warnings = await _render_document(source, request)
        elif source_type == "presentation":
            data = await _render_presentation(source, request)
        elif source_type == "spreadsheet":
            data = await _render_spreadsheet(source)
        elif source_type == "chart":
            data = await _render_chart_from_source(source)
        elif source_type == "ebook":
            data = await _render_ebook(source, request)
        else:
            return JSONResponse(
                {"error": f"Unknown source type in version: {source_type}"},
                status_code=400,
            )
    except Exception as exc:
        log.error(
            "studio_restore_render_failed",
            artifact_id=artifact_id, version_id=version_id, error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            {"error": "Render of restored version failed"}, status_code=500,
        )

    source_str = json.dumps(source, ensure_ascii=False)
    await store.update_source(artifact_id, source_str, user_id=uid)
    await store.update_file(artifact_id, data, user_id=uid)

    return JSONResponse({
        "restored": True,
        "artifact_id": artifact_id,
        "version_index": version.get("version_index"),
        "source": source,
        "size_bytes": len(data),
    })


async def _render_document(source: dict, request: Request) -> tuple[bytes, list[str]]:
    """Re-render a document from source JSON.

    Returns (bytes, warnings). Warnings are non-fatal — e.g. PDF Unicode
    fallback dropping emoji / accented characters when DejaVu Sans isn't
    available. The Studio shows these as a non-error toast so users don't
    silently ship a degraded PDF (the original bug — emoji vanished without
    any indication to the user).
    """
    from augmentum.tools.artifact_document import (
        _render_pdf, _render_docx, pdf_render_will_downgrade_unicode,
    )

    title = source.get("title", "Document")
    author = source.get("author", "")
    sections = source.get("sections", [])
    fmt = source.get("format", "pdf")
    design = _resolve_design(source)
    theme_name = design["theme"]

    # Resolve images (reuse the tool's logic)
    artifact_store = getattr(request.app.state, "artifact_store", None)
    if artifact_store:
        from augmentum.tools.artifact_document import DocumentTool
        temp_tool = DocumentTool(artifact_store)
        sections = await temp_tool._resolve_images(sections)

    warnings: list[str] = []
    if fmt == "docx":
        data = await asyncio.to_thread(_render_docx, title, author, sections, theme_name, design)
        return data, warnings

    # PDF: pre-flight the Unicode fallback so the user gets a non-error
    # warning when DejaVu Sans can't render their content (emoji, rare
    # accents). Pre-flight is read-only; the actual render still happens
    # below and sanitizes silently.
    if pdf_render_will_downgrade_unicode(sections):
        warnings.append(
            "Some characters in this document can't be embedded in the PDF "
            "with the available fonts and were simplified (e.g. emoji or "
            "rare accented letters). Switch to DOCX for full Unicode support."
        )

    data = await asyncio.to_thread(_render_pdf, title, author, sections, theme_name, design)
    return data, warnings


async def _render_presentation(source: dict, request: Request) -> bytes:
    """Re-render a presentation from source JSON."""
    from augmentum.tools.artifact_presentation import _render_pptx

    title = source.get("title", "Presentation")
    subtitle = source.get("subtitle", "")
    author = source.get("author", "")
    slides = source.get("slides", [])
    design = _resolve_design(source)
    theme_name = design["theme"]

    artifact_store = getattr(request.app.state, "artifact_store", None)
    if artifact_store:
        from augmentum.tools.artifact_presentation import PresentationTool
        temp_tool = PresentationTool(artifact_store)
        slides = await temp_tool._resolve_images(slides)

    return await asyncio.to_thread(_render_pptx, title, subtitle, author, slides, theme_name, design)


async def _render_spreadsheet(source: dict) -> bytes:
    """Re-render a spreadsheet from source JSON."""
    from augmentum.tools.artifact_spreadsheet import _render_xlsx
    design = _resolve_design(source)
    return await asyncio.to_thread(_render_xlsx, source.get("sheets", []), design["theme"], design)


async def _render_chart_from_source(source: dict) -> bytes:
    """Re-render a chart from source JSON."""
    from augmentum.tools.artifact_chart import _render_chart as _do_render
    design = _resolve_design(source)
    return await asyncio.to_thread(
        _do_render,
        title=source.get("title", "Chart"),
        chart_type=source.get("chart_type", "bar"),
        x_label=source.get("x_label", ""),
        y_label=source.get("y_label", ""),
        labels=source.get("labels", []),
        datasets=source.get("datasets", []),
        show_values=source.get("show_values", False),
        theme_name=design["theme"],
        design=design,
        # Presentation fields added in the pro-charts pass — keep server-side
        # re-renders (theme change, re-render button) consistent with the
        # Studio preview and the original tool render.
        value_format=source.get("value_format", "auto"),
        sort=source.get("sort", "none"),
        subtitle=source.get("subtitle", ""),
        caption=source.get("caption", ""),
    )


async def _render_ebook(source: dict, request: Request) -> bytes:
    """Re-render an EPUB from Studio source JSON."""
    from augmentum.tools.artifact_ebook import EbookTool, _render_epub

    title = source.get("title") or "Ebook"
    author = source.get("author") or ""
    chapters = source.get("chapters") if isinstance(source.get("chapters"), list) else []
    cover_url = source.get("cover_image_url") or source.get("cover_url") or ""
    design = _resolve_design(source)
    theme_name = design["theme"]
    # Project design back into the legacy reading shape so _build_epub_css
    # can stay as-is. Pure additive — old artifacts that only have `reading`
    # still work because _resolve_design synthesizes design FROM reading.
    reading = _project_design_to_reading(design)
    artifact_store = getattr(request.app.state, "artifact_store", None)
    cover_path = None

    if artifact_store:
        temp_tool = EbookTool(artifact_store)
        uid = _user_id(request)
        chapters = await temp_tool._resolve_images(chapters, user_id=uid)
        if cover_url:
            cover_path = await temp_tool._resolve_image_path(cover_url, user_id=uid)

    return await asyncio.to_thread(
        _render_epub,
        title,
        author,
        chapters,
        cover_image_path=cover_path,
        theme_name=theme_name or "",
        reading=reading,
    )


# ---------------------------------------------------------------------------
# Image-tool endpoints (Studio palette Phase 1)
# ---------------------------------------------------------------------------


async def _resolve_image_search_tool(request: Request):
    registry = getattr(request.app.state, "tool_registry", None)
    return registry.resolve("image_search") if registry else None


async def _resolve_image_generation_tool(request: Request):
    registry = getattr(request.app.state, "tool_registry", None)
    return registry.resolve("image_generation") if registry else None


def _check_artifact_owner(store, artifact_id: str, uid: str):
    """Same auth pattern as /save and /preview — task-scope these tools so a
    crafted POST can't drive image search/gen as another user's artifact."""

    async def _check():
        info = await store.get(artifact_id, user_id=uid)
        return info
    return _check()


@router.post("/{artifact_id}/search-images")
async def search_images(artifact_id: str, request: Request) -> JSONResponse:
    """Run image_search and return Studio-compatible candidates.

    Body: {"query": str, "count": int=4, "prefer_charts": bool=false}.
    Response shape matches the agentic picker's /candidates so the Studio
    UI can render the same gallery component either way.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    info = await _check_artifact_owner(store, artifact_id, uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    count = max(1, min(int(body.get("count") or 4), 6))
    prefer_charts = bool(body.get("prefer_charts"))

    tool = await _resolve_image_search_tool(request)
    if not tool:
        return JSONResponse({"error": "image_search tool not registered"}, status_code=503)

    try:
        result = await tool.execute(
            query=query,
            count=count,
            prefer_charts=prefer_charts,
            task_id="",  # studio-scope, no agentic task
            session_id="",
            _user_id=uid,
        )
    except Exception as exc:
        log.warning("studio_search_images_failed",
                    artifact_id=artifact_id, error=str(exc))
        return JSONResponse(
            {"error": "image search failed", "detail": str(exc)},
            status_code=502,
        )
    if not result.success or not result.metadata:
        return JSONResponse(
            {"error": result.error or "no results", "candidates": []},
            status_code=200,
        )

    images = result.metadata.get("images") or []
    candidates: list[dict] = []
    for img in images:
        if not isinstance(img, dict):
            continue
        embed_url = img.get("embed_url") or img.get("url") or ""
        if not embed_url:
            continue
        candidates.append({
            "candidate_id": uuid.uuid4().hex[:12],
            "query": query,
            "embed_url": embed_url,
            "thumb_url": img.get("thumb_url") or embed_url,
            "source": img.get("source", ""),
            "title": img.get("title", ""),
        })

    return JSONResponse({"candidates": candidates, "query": query})


@router.post("/{artifact_id}/generate-image")
async def generate_image(artifact_id: str, request: Request) -> JSONResponse:
    """Generate an image via image_generation and stage it for the picker.

    Body: {"prompt": str, "style": str?, "aspect": str?}.

    The generated image lands in the user's library immediately (image_gen
    persists on creation), but the gen_id is recorded in the in-memory
    staging registry so the sweep deletes it after the TTL if the user
    never clicks Use / Save.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    info = await _check_artifact_owner(store, artifact_id, uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        body = {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)
    style = (body.get("style") or "").strip()
    aspect = (body.get("aspect") or "square").strip()

    tool = await _resolve_image_generation_tool(request)
    if not tool:
        return JSONResponse({"error": "image_generation tool not registered"}, status_code=503)

    try:
        result = await tool.execute(
            prompt=prompt,
            style=style,
            aspect=aspect,
            session_id="",
            _user_id=uid,
        )
    except Exception as exc:
        log.warning("studio_generate_image_failed",
                    artifact_id=artifact_id, error=str(exc))
        return JSONResponse(
            {"error": "image generation failed", "detail": str(exc)},
            status_code=502,
        )
    if not result.success or not result.metadata:
        return JSONResponse(
            {"error": result.error or "generation failed"},
            status_code=502,
        )

    image_id = result.metadata.get("image_id") or ""
    url = result.metadata.get("url") or (f"/api/image/{image_id}" if image_id else "")
    if not image_id:
        return JSONResponse(
            {"error": "image_generation returned no id"}, status_code=502,
        )

    # Stage for sweep — until Use / Save fires, this entry will be deleted
    # after TTL_SECONDS. Use of `app.state.studio_staging` mirrors the
    # picker's lazy-init pattern; pure in-memory by design.
    registry = _ensure_staging_registry(request)
    registry[image_id] = {
        "user_id": uid,
        "artifact_id": artifact_id,
        "prompt": prompt,
        "style": style,
        "aspect": aspect,
        "created_at": time.time(),
    }

    return JSONResponse({
        "gen_id": image_id,
        "embed_url": url,
        "thumb_url": url,
        "prompt_used": result.metadata.get("prompt", prompt),
        "staged_until": registry[image_id]["created_at"] + _STAGING_TTL_SECONDS,
    })


@router.post("/{artifact_id}/staging/{gen_id}/commit")
async def commit_staged_image(
    artifact_id: str, gen_id: str, request: Request,
) -> JSONResponse:
    """Promote a staged generation out of the sweep window.

    Called by both 'Use it' and 'Save to library'. The image is already
    persisted in image_generations; this just removes it from the staging
    registry so the sweep won't delete it.

    Returns the canonical URL so the UI can write it into a slot.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    info = await _check_artifact_owner(store, artifact_id, uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    registry = _ensure_staging_registry(request)
    entry = registry.get(gen_id)
    if not entry or entry.get("user_id") != uid:
        # Either already committed, swept, or never existed. We treat that
        # as success on the user-facing side — they clicked Use after we
        # already removed it (idempotent), so just confirm and return the URL.
        return JSONResponse({"committed": True, "url": f"/api/image/{gen_id}"})

    registry.pop(gen_id, None)
    return JSONResponse({"committed": True, "url": f"/api/image/{gen_id}"})


@router.delete("/{artifact_id}/staging/{gen_id}")
async def discard_staged_image(
    artifact_id: str, gen_id: str, request: Request,
) -> JSONResponse:
    """Delete a staged generation immediately (Regenerate / explicit Discard).

    Removes the row from image_generations + the file from disk + the
    staging registry entry. Idempotent — re-deleting a gone gen_id
    succeeds quietly.
    """
    store = _get_store(request)
    if not store:
        return JSONResponse({"error": "Artifact store not available"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    info = await _check_artifact_owner(store, artifact_id, uid)
    if not info:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)

    registry = _ensure_staging_registry(request)
    entry = registry.get(gen_id)
    # Guard: only delete the image if THIS user owns the staging entry —
    # without that check a crafted gen_id of another user's image could be
    # nuked via cross-tenant request. The persistence layer's user_id
    # filter would also catch it, but defence-in-depth.
    if entry and entry.get("user_id") != uid:
        return JSONResponse({"error": "Not found"}, status_code=404)

    deleted = await _delete_staged_image(request, gen_id, uid)
    registry.pop(gen_id, None)
    return JSONResponse({"deleted": deleted, "gen_id": gen_id})


async def _delete_staged_image(request: Request, image_id: str, user_id: str) -> bool:
    """Best-effort delete: DB row + disk file. Returns True if row deleted."""
    state = request.app.state
    image_store = getattr(state, "image_persistence", None) or getattr(
        state, "image_store", None,
    )
    if image_store is None:
        log.warning("studio_staging_no_image_store", image_id=image_id)
        return False
    try:
        file_path = await image_store.delete_generation(image_id, user_id=user_id)
    except Exception as exc:
        log.warning("studio_staging_delete_failed",
                    image_id=image_id, error=str(exc))
        return False
    if file_path:
        try:
            import os
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as exc:
            log.warning("studio_staging_file_unlink_failed",
                        image_id=image_id, error=str(exc))
    return True


async def studio_staging_sweep(app, *, now: float | None = None) -> int:
    """Background sweep — delete un-acted staging entries past the TTL.

    Module-level so the server lifespan startup hook can spawn the loop
    without importing route-specific symbols. Returns the number of
    entries swept (deleted from registry + image library).
    """
    registry = getattr(app.state, "studio_staging", None)
    if not registry:
        return 0
    cutoff = (now or time.time()) - _STAGING_TTL_SECONDS
    expired: list[tuple[str, str]] = []
    for gen_id, entry in list(registry.items()):
        if (entry.get("created_at") or 0) < cutoff:
            expired.append((gen_id, entry.get("user_id", "")))
    if not expired:
        return 0
    image_store = getattr(app.state, "image_persistence", None) or getattr(
        app.state, "image_store", None,
    )
    swept = 0
    for gen_id, uid in expired:
        registry.pop(gen_id, None)
        if not image_store or not uid:
            continue
        try:
            file_path = await image_store.delete_generation(gen_id, user_id=uid)
        except Exception as exc:
            log.warning("studio_staging_sweep_delete_failed",
                        image_id=gen_id, error=str(exc))
            continue
        if file_path:
            try:
                import os
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        swept += 1
    if swept:
        log.info("studio_staging_swept", count=swept)
    return swept


async def studio_staging_sweep_loop(app) -> None:
    """Long-running loop — sweeps every _STAGING_SWEEP_INTERVAL seconds.

    Spawned at app startup. Cancelled on shutdown. Catches its own
    exceptions so a transient DB failure doesn't kill the loop.
    """
    while True:
        try:
            await asyncio.sleep(_STAGING_SWEEP_INTERVAL)
            await studio_staging_sweep(app)
        except asyncio.CancelledError:
            break
        except Exception:
            log.warning("studio_staging_sweep_loop_iteration_failed",
                        exc_info=True)
            # Soft backoff before retrying
            await asyncio.sleep(30)
