"""User-uploaded animation API routes — list, upload, update, delete, serve.

The widget atlas merges the rows surfaced here alongside the
code-defined ATLAS in ``ui/scripts/anim-atlas.js`` so the conductor's
selection population grows with each upload.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from augmentum.animations.store import UserAnimationStore

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/animations", tags=["animations"])


# 50MB ceiling per upload. VRMA files are typically <5MB; the cap is
# generous for BVH files which can be larger when uncompressed but
# bounded enough that a malicious or buggy client can't fill /data.
ANIMATION_MAX_SIZE = 50 * 1024 * 1024
_ALLOWED_EXTS = (".vrma", ".bvh")


def _get_animations_dir() -> str:
    env_override = os.environ.get("AUGMENTUM_ANIMATIONS_DIR", "")
    if env_override:
        return env_override
    from augmentum.config import settings
    base = getattr(settings, "data_dir", "/data")
    return os.path.join(base, "user_animations")


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _conn(request: Request):
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        raise RuntimeError("Backend not initialized")
    return sm.backend.conn


def _store(request: Request) -> UserAnimationStore:
    return UserAnimationStore(_conn(request))


def _safe_animation_id(animation_id: str) -> bool:
    """Reject path-traversal attempts. Animation ids are
    ``user:<ts>_<hex>`` — letters, digits, colon, underscore only."""
    if not animation_id:
        return False
    if "/" in animation_id or "\\" in animation_id or ".." in animation_id:
        return False
    return True


def _safe_filename_label(filename: str) -> str:
    """Strip path components + extension from the original filename to
    produce a human-readable default label."""
    base = Path(filename).stem if filename else "animation"
    cleaned = "".join(ch if (ch.isalnum() or ch in " -_") else " "
                      for ch in base)
    cleaned = " ".join(cleaned.split())
    return cleaned[:60] or "animation"


def _row_to_atlas_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Shape a DB row into the atlas-compatible JS object the conductor
    expects. The widget's anim-atlas.js merges these alongside ATLAS.

    Notable transforms:
      - ``source_path`` → ``source`` as the public URL
      - ``loop_flag`` → ``loop`` (boolean)
      - ``explicit_only`` → ``explicitOnly``
      - ``duration_sec`` → ``duration``
      - ``cooldown_sec`` → ``cooldown``
      - ``trim_start`` / ``trim_end`` stay as-is (the atlas uses snake_
        and camel- variants — these are the snake form)
    """
    return {
        "id": row["id"],
        "type": row.get("type") or "vrma",
        "source": f"/api/animations/{row['id']}/file",
        "roles": row.get("roles") or [],
        "emotion": row.get("emotion") or {},
        "modes": row.get("modes") or [],
        "cost": row.get("cost") or 0.5,
        "duration": row.get("duration_sec") or 0,
        "cooldown": row.get("cooldown_sec") or 300,
        "framing": row.get("framing"),
        "trimStart": row.get("trim_start"),
        "trimEnd": row.get("trim_end"),
        "speed": row.get("speed"),
        "loop": bool(row.get("loop_flag")),
        "explicitOnly": bool(row.get("explicit_only")),
        "label": row.get("label"),
        "notes": row.get("notes"),
        "userOwned": True,
    }


def _parse_metadata_form(value: str | None) -> dict[str, Any]:
    """Optional JSON metadata blob sent alongside the file upload.
    Tolerates absence + parse errors — the upload still lands with
    safe defaults if the client doesn't send anything."""
    if not value:
        return {}
    try:
        data = json.loads(value)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        log.warning("animation_upload_metadata_parse_failed", error=str(exc))
    return {}


# ── List ─────────────────────────────────────────────────────────────


@router.get("/list")
async def list_animations(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rows = await _store(request).list_for_user(user_id=uid)
    return JSONResponse({
        "animations": [_row_to_atlas_entry(r) for r in rows],
    })


# ── Upload ───────────────────────────────────────────────────────────


@router.post("/upload")
async def upload_animation(
    request: Request,
    file: UploadFile = File(...),
    metadata: str | None = Form(default=None),
):
    """Upload a VRMA or BVH file. Optional ``metadata`` form field
    accepts a JSON object with any of: label, roles, emotion, modes,
    cost, duration_sec, cooldown_sec, framing, trim_start, trim_end,
    speed, loop_flag, explicit_only, notes.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        return JSONResponse(
            {"error": "Only .vrma and .bvh files are accepted"},
            status_code=400,
        )
    content = await file.read()
    if len(content) > ANIMATION_MAX_SIZE:
        return JSONResponse(
            {
                "error": (
                    f"File too large (max "
                    f"{ANIMATION_MAX_SIZE // 1024 // 1024}MB)"
                ),
            },
            status_code=400,
        )
    if len(content) == 0:
        return JSONResponse({"error": "Empty upload"}, status_code=400)

    meta = _parse_metadata_form(metadata)
    label = str(meta.get("label") or _safe_filename_label(filename))
    animation_type = ext.lstrip(".")
    user_dir = Path(_get_animations_dir()) / uid
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.warning("animation_upload_mkdir_failed", user_id=uid)
        return JSONResponse(
            {"error": "Failed to prepare upload directory"},
            status_code=500,
        )

    store = _store(request)
    try:
        row = await store.create(
            animation_type=animation_type,
            source_path="",  # filled in after write
            label=label,
            roles=meta.get("roles") if isinstance(meta.get("roles"), list)
                  else None,
            emotion=meta.get("emotion") if isinstance(meta.get("emotion"), dict)
                    else None,
            modes=meta.get("modes") if isinstance(meta.get("modes"), list)
                  else None,
            cost=float(meta.get("cost") or 0.5),
            duration_sec=float(meta.get("duration_sec") or 0),
            cooldown_sec=float(meta.get("cooldown_sec") or 300),
            framing=meta.get("framing"),
            trim_start=meta.get("trim_start"),
            trim_end=meta.get("trim_end"),
            speed=meta.get("speed"),
            loop_flag=bool(meta.get("loop_flag")),
            explicit_only=bool(meta.get("explicit_only")),
            notes=meta.get("notes"),
            user_id=uid,
        )
    except ValueError:
        return JSONResponse(
            {"error": "Invalid animation metadata"}, status_code=400,
        )

    anim_id = row["id"]
    # Storage path under the user's animation dir. anim_id has the form
    # 'user:<ts>_<hex>' — replace ':' with '_' to keep the on-disk name
    # safe across filesystems.
    safe_name = anim_id.replace(":", "_")
    target_path = user_dir / f"{safe_name}{ext}"
    try:
        target_path.write_bytes(content)
    except OSError as exc:
        log.warning(
            "animation_upload_write_failed", animation_id=anim_id,
            error=str(exc),
        )
        await store.delete(anim_id, user_id=uid)
        return JSONResponse(
            {"error": "Failed to save animation file"}, status_code=500,
        )

    # Persist the final path on the row so future serves can find it.
    await store.update(
        anim_id, {"label": label}, user_id=uid,  # bump updated_at via no-op
    )
    # Direct update of source_path bypassing the update() whitelist —
    # source_path is intentionally NOT user-editable but the upload
    # handler is the one place where it gets set.
    await _conn(request).execute(
        "UPDATE user_animations SET source_path = ? "
        "WHERE id = ? AND user_id = ?",
        (str(target_path), anim_id, uid),
    )
    await _conn(request).commit()
    log.info(
        "animation_uploaded", animation_id=anim_id,
        size=len(content), user_id=uid, type=animation_type,
    )
    final = await store.get(anim_id, user_id=uid)
    return JSONResponse({
        "ok": True,
        "animation": _row_to_atlas_entry(final),
    })


# ── Role vocabulary snapshot ─────────────────────────────────────────
#
# The merged role vocabulary (bundled atlas + uploads + override
# patches) only exists client-side — the bundled registry is code in
# anim-atlas.js. The widget pushes a snapshot here after every registry
# refresh so server-side consumers (the future LLM-facing ``gesture``
# verb, tool schemas, diagnostics) can read the user's live vocabulary
# without mirroring the atlas in Python. Client-authoritative by
# design: a stale snapshot self-heals on the next widget mount.

_ROLES_SETTING_KEY = "companion.animation_roles"
_ROLES_MAX_COUNT = 300
_ROLES_MAX_LEN = 48


def _settings_store(request: Request):
    return getattr(request.app.state, "settings_store", None)


@router.get("/roles")
async def get_roles_vocabulary(request: Request):
    """The user's last-pushed role vocabulary (sorted, deduped)."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _settings_store(request)
    if store is None:
        return JSONResponse({"error": "Settings store unavailable"}, status_code=503)
    raw = await store.get_user(uid, _ROLES_SETTING_KEY)
    roles: list[str] = []
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                roles = [str(r) for r in parsed]
        except json.JSONDecodeError:
            log.warning("animation_roles_snapshot_unparseable", user_id=uid)
    return JSONResponse({"roles": roles})


@router.put("/roles-snapshot")
async def put_roles_snapshot(request: Request):
    """Replace the stored vocabulary with the client's current merge.

    Body: ``{"roles": ["greet", "celebrate", ...]}``. Normalized to
    sorted unique lowercase strings, bounded so a buggy client can't
    bloat the settings row.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    store = _settings_store(request)
    if store is None:
        return JSONResponse({"error": "Settings store unavailable"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    raw_roles = body.get("roles") if isinstance(body, dict) else None
    if not isinstance(raw_roles, list):
        return JSONResponse({"error": "roles must be a list"}, status_code=400)
    cleaned = sorted({
        str(r).strip().lower()[:_ROLES_MAX_LEN]
        for r in raw_roles
        if str(r).strip()
    })[:_ROLES_MAX_COUNT]
    await store.set_user(uid, _ROLES_SETTING_KEY, json.dumps(cleaned))
    return JSONResponse({"ok": True, "count": len(cleaned)})


# ── Update tags ──────────────────────────────────────────────────────


@router.put("/{animation_id}")
async def update_animation(request: Request, animation_id: str):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_animation_id(animation_id):
        return JSONResponse({"error": "Invalid id"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be an object"}, status_code=400)
    store = _store(request)
    try:
        updated = await store.update(animation_id, body, user_id=uid)
    except ValueError:
        return JSONResponse(
            {"error": "Invalid update"}, status_code=400,
        )
    if not updated:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"animation": _row_to_atlas_entry(updated)})


# ── Delete ───────────────────────────────────────────────────────────


@router.delete("/{animation_id}")
async def delete_animation(request: Request, animation_id: str):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_animation_id(animation_id):
        return JSONResponse({"error": "Invalid id"}, status_code=400)
    store = _store(request)
    deleted = await store.delete(animation_id, user_id=uid)
    if not deleted:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Best-effort file cleanup. A missing file isn't fatal — the row
    # is already gone, so future GETs will 404 cleanly anyway.
    src = deleted.get("source_path") or ""
    if src:
        try:
            Path(src).unlink(missing_ok=True)
        except OSError as exc:
            log.warning(
                "animation_delete_unlink_failed",
                animation_id=animation_id, error=str(exc),
            )
    return JSONResponse({"ok": True})


# ── Serve ────────────────────────────────────────────────────────────


@router.get("/{animation_id}/file")
async def serve_animation(request: Request, animation_id: str):
    """Serve the raw VRMA/BVH bytes. Uses the on-disk path recorded at
    upload time — never user-supplied path components."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_animation_id(animation_id):
        return JSONResponse({"error": "Invalid id"}, status_code=400)
    row = await _store(request).get(animation_id, user_id=uid)
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    src = row.get("source_path") or ""
    if not src or not Path(src).exists():
        return JSONResponse({"error": "File missing"}, status_code=404)
    media_type = (
        "model/vrm" if (row.get("type") == "vrma") else "application/octet-stream"
    )
    return FileResponse(src, media_type=media_type)


# ── Bundled-atlas overrides ──────────────────────────────────────────
#
# Per-user disable/re-tag of the BUNDLED atlas entries (anim-atlas.js).
# The widget fetches these alongside /list and merges them over the
# code-defined registry; see registerAtlasOverrides() in anim-atlas.js.


@router.get("/overrides")
async def list_atlas_overrides(request: Request):
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    rows = await _store(request).list_overrides(user_id=uid)
    return JSONResponse({"overrides": rows})


@router.put("/overrides/{atlas_id}")
async def put_atlas_override(request: Request, atlas_id: str):
    """Upsert an override: ``{"disabled": bool?, "patch": {...}?}``.

    Omitted halves keep their stored value, so a disable toggle doesn't
    clobber an existing re-tag and vice versa. Patch keys are
    whitelisted in the store (atlas metadata only — never ``source``).
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_animation_id(atlas_id):
        return JSONResponse({"error": "Invalid id"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be an object"}, status_code=400)
    disabled = body.get("disabled")
    patch = body.get("patch")
    if disabled is not None and not isinstance(disabled, bool):
        return JSONResponse({"error": "disabled must be a boolean"}, status_code=400)
    if patch is not None and not isinstance(patch, dict):
        return JSONResponse({"error": "patch must be an object"}, status_code=400)
    try:
        row = await _store(request).set_override(
            atlas_id, disabled=disabled, patch=patch, user_id=uid,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"override": row})


@router.delete("/overrides/{atlas_id}")
async def delete_atlas_override(request: Request, atlas_id: str):
    """Clear an override — the bundled entry returns to its defaults."""
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not _safe_animation_id(atlas_id):
        return JSONResponse({"error": "Invalid id"}, status_code=400)
    removed = await _store(request).clear_override(atlas_id, user_id=uid)
    if not removed:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"ok": True})
