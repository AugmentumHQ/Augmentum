"""Artifact storage — manages generated files (documents, presentations, etc.)."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# Provenance choke point (wiring program Phase 7). The companion's
# tool-execution path sets this to "companion" for the duration of a
# tool call; ArtifactStore.save() reads it so EVERY artifact-producing
# tool (create_document/.../image_search/remove_background) stamps
# origin without per-tool wiring. Async-safe: ContextVars are
# task-local. '' = user-created.
ARTIFACT_ORIGIN: ContextVar[str] = ContextVar("artifact_origin", default="")



def _mime_for_fmt(fmt: str) -> str:
    return {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "html": "text/html", "epub": "application/epub+zip", "png": "image/png",
    }.get(fmt, "application/octet-stream")


def _id() -> str:
    return uuid.uuid4().hex[:16]


def _safe_filename(name: str, *, fallback: str = "file") -> str:
    """Reduce an arbitrary (possibly attacker-supplied) filename to a safe
    basename with no directory components.

    ``save``/``save_from_path`` join this onto the artifact dir as
    ``target_dir / filename``. Without sanitising, an upload whose
    ``filename`` is ``../../../x`` or ``/etc/passwd`` would write OUTSIDE
    the artifact store (pathlib lets an absolute or ``..`` segment escape).
    ``os.path.basename`` after normalising separators strips every path
    component; a name that was pure traversal collapses to the fallback.
    """
    raw = str(name or "").replace("\\", "/")
    base = os.path.basename(raw).strip()
    # A name that was only separators / dots (``..``, ``.``, ``/``) is unsafe.
    if not base or set(base) <= {"."}:
        return fallback
    # Defensive: drop null bytes and any residual separator.
    base = base.replace("\x00", "").replace("/", "_")
    return base or fallback


class ArtifactStore:
    """Manages artifact file persistence and metadata in SQLite."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        configured = getattr(settings, "agentic_artifact_dir", "data/artifacts")
        artifact_path = Path(configured)
        # If the configured path is relative, resolve it under data_dir
        # so it works inside Docker (/data/artifacts) and locally (.data/artifacts)
        if not artifact_path.is_absolute():
            artifact_path = Path(settings.data_dir) / artifact_path
        self._base_dir = artifact_path

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    async def save(
        self,
        data: bytes,
        filename: str,
        fmt: str,
        *,
        task_id: str = "",
        session_id: str = "",
        display_name: str = "",
        metadata: dict | None = None,
        source_json: str | None = None,
        user_id: str,
        transient: bool = False,
    ) -> dict:
        """Write artifact bytes to disk and record in SQLite.

        Returns a dict with id, filename, path, size_bytes, and download_url.

        transient=True marks an ephemeral cache entry (e.g. image_search
        thumbnails): the row is stamped transient=1, VFS registration is
        skipped so it stays out of the Files browser, and the eviction
        sweep is free to purge it by age/size. The download URL still
        works, so chat galleries render normally.
        """
        if not user_id:
            raise ValueError("ArtifactStore.save requires a user_id")
        # Neutralise path traversal in caller-supplied names (import uploads
        # forward the raw multipart filename here) before it's joined onto
        # the artifact dir below.
        filename = _safe_filename(filename)
        artifact_id = _id()
        sub_dir = "_transient" if transient else (task_id or "standalone")
        target_dir = self._base_dir / sub_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Ensure unique filename within directory
        target_path = target_dir / filename
        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            filename = target_path.name

        target_path.write_bytes(data)
        size_bytes = len(data)
        rel_path = f"{sub_dir}/{filename}"

        meta_json = json.dumps(metadata or {})

        origin = ARTIFACT_ORIGIN.get() or ""
        await self._db.execute(
            "INSERT INTO artifacts "
            "(id, task_id, session_id, filename, display_name, format, "
            " size_bytes, path, metadata, source_json, user_id, transient, "
            " origin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                artifact_id, task_id, session_id, filename,
                display_name or filename, fmt, size_bytes, rel_path,
                meta_json, source_json, user_id, 1 if transient else 0,
                origin,
            ],
        )
        await self._db.commit()

        # Register in file index — skipped for transient entries so they
        # don't pollute the user's Files browser.
        if not transient:
            from augmentum.vfs import register_file
            await register_file(
                user_id=user_id, source="artifacts", source_id=artifact_id,
                name=filename, mime_type=_mime_for_fmt(fmt),
                size_bytes=len(data), real_path=str(target_path),
                description=display_name or filename,
                source_metadata={"task_id": task_id, "format": fmt, "display_name": display_name},
            )

        log.info(
            "artifact_saved",
            id=artifact_id,
            filename=filename,
            format=fmt,
            size_bytes=size_bytes,
        )

        return {
            "id": artifact_id,
            "filename": filename,
            "display_name": display_name or filename,
            "format": fmt,
            "size_bytes": size_bytes,
            "path": rel_path,
            "download_url": f"/api/artifacts/{artifact_id}/download",
            "source_json": source_json,
        }

    async def save_from_path(
        self,
        src_path: str,
        filename: str,
        fmt: str,
        *,
        task_id: str = "",
        session_id: str = "",
        display_name: str = "",
        metadata: dict | None = None,
        user_id: str,
        transient: bool = False,
    ) -> dict:
        """Move an already-written file into the artifact store + record it.

        Like :meth:`save` but takes a path instead of bytes — so a big
        already-on-disk file (e.g. a book-length narration WAV/MP3) is
        *moved*, never round-tripped through process memory. ``src_path``
        is consumed (moved); if it's on a different filesystem it falls
        back to copy+unlink.

        ``transient=True`` marks a derived, regenerable cache entry (e.g.
        per-page comic-narration audio — 30+ artifacts per chapter):
        stamped transient=1 so listings/library exclude it, and file-index/
        VFS registration is skipped so it never clutters the Files surface.
        Still fetchable by id (players stream via /download). Default False
        — most save_from_path callers produce real deliverables.
        """
        if not user_id:
            raise ValueError("ArtifactStore.save_from_path requires a user_id")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(src_path)
        filename = _safe_filename(filename)
        artifact_id = _id()
        sub_dir = "_transient" if transient else (task_id or "standalone")
        target_dir = self._base_dir / sub_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / filename
        if target_path.exists():
            stem, suffix = target_path.stem, target_path.suffix
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            filename = target_path.name

        size_bytes = os.path.getsize(src_path)
        try:
            shutil.move(src_path, str(target_path))
        except OSError:
            shutil.copy2(src_path, str(target_path))
            try:
                os.remove(src_path)
            except OSError:
                pass
        rel_path = f"{sub_dir}/{filename}"
        meta_json = json.dumps(metadata or {})

        await self._db.execute(
            "INSERT INTO artifacts "
            "(id, task_id, session_id, filename, display_name, format, "
            " size_bytes, path, metadata, source_json, user_id, transient) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                artifact_id, task_id, session_id, filename,
                display_name or filename, fmt, size_bytes, rel_path,
                meta_json, None, user_id, 1 if transient else 0,
            ],
        )
        await self._db.commit()

        # Mirror save(): transient cache entries never register in the
        # file index / VFS — they're playback cache, not user files.
        if not transient:
            from augmentum.vfs import register_file
            await register_file(
                user_id=user_id, source="artifacts", source_id=artifact_id,
                name=filename, mime_type=_mime_for_fmt(fmt),
                size_bytes=size_bytes, real_path=str(target_path),
                description=display_name or filename,
                source_metadata={"task_id": task_id, "format": fmt, "display_name": display_name},
            )
        log.info("artifact_saved_from_path", id=artifact_id, filename=filename, format=fmt, size_bytes=size_bytes)
        return {
            "id": artifact_id,
            "filename": filename,
            "display_name": display_name or filename,
            "format": fmt,
            "size_bytes": size_bytes,
            "path": rel_path,
            "download_url": f"/api/artifacts/{artifact_id}/download",
        }

    async def save_checkpoint(
        self,
        build_id: str,
        session_id: str,
        project_name: str,
        files: list[dict],
        planned_files: list[dict],
        scaffold: str = "static",
        *,
        user_id: str,
    ) -> str:
        """Save or update a build checkpoint.

        Checkpoints are hidden from list_all() by default via the
        is_checkpoint metadata flag.  They persist to SQLite so they
        survive server restarts — unlike the in-memory ACTIVE_BUILDS dict.
        Uses UPSERT on display_name so each build has at most one checkpoint.
        """
        if not user_id:
            raise ValueError("save_checkpoint requires a user_id")
        import io
        import zipfile
        # Pack files into a zip (same as deliver pass)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                if f.get("content"):
                    zf.writestr(f["path"], f["content"])
        checkpoint_name = f"checkpoint-{build_id}"

        # Check for existing checkpoint for this build
        cursor = await self._db.execute(
            "SELECT id FROM artifacts WHERE display_name = ? "
            "AND json_extract(metadata, '$.is_checkpoint') = 1 "
            "AND user_id = ?",
            [checkpoint_name, user_id],
        )
        existing = await cursor.fetchone()
        if existing:
            # Update existing checkpoint
            await self.update_file(existing[0], zip_buf.getvalue(), user_id=user_id)
            meta = json.dumps({
                "is_checkpoint": True,
                "build_id": build_id,
                "scaffold": scaffold,
                "file_count": len(files),
                "planned_files": planned_files,
                "completed_files": [f["path"] for f in files if f.get("content")],
                "files_meta": [{"path": f["path"], "role": f.get("role", ""), "content": f.get("content", "")} for f in files],
            })
            await self._db.execute(
                "UPDATE artifacts SET metadata = ? WHERE id = ?",
                (meta, existing[0]),
            )
            await self._db.commit()
            return existing[0]

        # Create new checkpoint
        return (await self.save(
            data=zip_buf.getvalue(),
            filename=f"{project_name.lower().replace(' ', '-')}-checkpoint.zip",
            fmt="zip",
            session_id=session_id,
            display_name=checkpoint_name,
            user_id=user_id,
            metadata={
                "is_checkpoint": True,
                "build_id": build_id,
                "scaffold": scaffold,
                "file_count": len(files),
                "planned_files": planned_files,
                "completed_files": [f["path"] for f in files if f.get("content")],
                "files_meta": [{"path": f["path"], "role": f.get("role", ""), "content": f.get("content", "")} for f in files],
            },
        ))["id"]

    async def get_checkpoint(self, build_id: str, *, user_id: str = "") -> dict | None:
        """Retrieve checkpoint data for a build, if any."""
        if not user_id:
            log.warning("get_checkpoint.empty_user_id", build_id=build_id)
            return None
        cursor = await self._db.execute(
            "SELECT * FROM artifacts WHERE display_name = ? "
            "AND json_extract(metadata, '$.is_checkpoint') = 1 "
            "AND user_id = ?",
            [f"checkpoint-{build_id}", user_id],
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row, strict=False))
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        return d

    async def delete_checkpoint(self, build_id: str, *, user_id: str) -> bool:
        """Remove a checkpoint after successful build completion."""
        if not user_id:
            raise ValueError("delete_checkpoint requires a user_id")
        cursor = await self._db.execute(
            "SELECT id FROM artifacts WHERE display_name = ? "
            "AND json_extract(metadata, '$.is_checkpoint') = 1 "
            "AND user_id = ?",
            [f"checkpoint-{build_id}", user_id],
        )
        row = await cursor.fetchone()
        if row:
            return await self.delete(row[0], user_id=user_id)
        return False

    # ------------------------------------------------------------------
    # Version history (artifact_versions table)
    # ------------------------------------------------------------------

    async def save_version(
        self,
        artifact_id: str,
        files: list[dict],
        *,
        user_id: str,
        label: str = "",
        score: float | None = None,
    ) -> dict:
        """Append a new version snapshot for ``artifact_id``.

        ``version_index`` auto-increments per artifact (1, 2, 3, …) so
        the UI can show "v3 of 5" without doing math. Files are stored
        inline as JSON — see migration 137 for the rationale.
        """
        if not user_id:
            raise ValueError("save_version requires a user_id")
        cursor = await self._db.execute(
            "SELECT COALESCE(MAX(version_index), 0) FROM artifact_versions "
            "WHERE artifact_id = ? AND user_id = ?",
            [artifact_id, user_id],
        )
        row = await cursor.fetchone()
        next_index = int(row[0] or 0) + 1
        # Strip transient fields that don't belong in a snapshot.
        snapshot = [
            {
                "path": f.get("path", ""),
                "role": f.get("role", ""),
                "lang": f.get("lang", ""),
                "content": f.get("content", ""),
            }
            for f in files
            if f.get("path")
        ]
        version_id = _id()
        await self._db.execute(
            "INSERT INTO artifact_versions "
            "(id, artifact_id, version_index, label, files_json, file_count, "
            "score, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                version_id,
                artifact_id,
                next_index,
                (label or "")[:200],
                json.dumps(snapshot),
                len(snapshot),
                score,
                user_id,
            ],
        )
        await self._db.commit()
        return {
            "id": version_id,
            "artifact_id": artifact_id,
            "version_index": next_index,
            "label": label,
            "file_count": len(snapshot),
            "score": score,
        }

    async def list_versions(self, artifact_id: str, *, user_id: str) -> list[dict]:
        """Return every version of ``artifact_id`` newest-first.

        Files JSON is omitted from the list payload to keep it cheap to
        render — call ``get_version`` for the actual content.
        """
        if not user_id:
            raise ValueError("list_versions requires a user_id")
        cursor = await self._db.execute(
            "SELECT id, version_index, label, file_count, score, created_at "
            "FROM artifact_versions WHERE artifact_id = ? AND user_id = ? "
            "ORDER BY version_index DESC",
            [artifact_id, user_id],
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "version_index": r[1],
                "label": r[2] or "",
                "file_count": r[3],
                "score": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    async def get_version(self, version_id: str, *, user_id: str) -> dict | None:
        """Fetch a single version including its files."""
        if not user_id:
            raise ValueError("get_version requires a user_id")
        cursor = await self._db.execute(
            "SELECT id, artifact_id, version_index, label, files_json, "
            "file_count, score, created_at FROM artifact_versions "
            "WHERE id = ? AND user_id = ?",
            [version_id, user_id],
        )
        row = await cursor.fetchone()
        if not row:
            return None
        try:
            files = json.loads(row[4]) if row[4] else []
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("artifact_versions.invalid_files_json",
                        version_id=version_id, error=str(exc))
            files = []
        return {
            "id": row[0],
            "artifact_id": row[1],
            "version_index": row[2],
            "label": row[3] or "",
            "files": files,
            "file_count": row[5],
            "score": row[6],
            "created_at": row[7],
        }

    async def update_source(self, artifact_id: str, source_json: str, *, user_id: str) -> bool:
        """Update the source JSON for an artifact."""
        if not user_id:
            raise ValueError("update_source requires a user_id")
        cursor = await self._db.execute(
            "UPDATE artifacts SET source_json = ? WHERE id = ? AND user_id = ?",
            [source_json, artifact_id, user_id],
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def update_file(self, artifact_id: str, data: bytes, *, user_id: str) -> bool:
        """Replace the artifact binary file and update size."""
        if not user_id:
            raise ValueError("update_file requires a user_id")
        info = await self.get(artifact_id, user_id=user_id)
        if not info:
            return False
        file_path = self.get_file_path(info["path"])
        if not file_path:
            return False
        file_path.write_bytes(data)
        cursor = await self._db.execute(
            "UPDATE artifacts SET size_bytes = ? WHERE id = ? AND user_id = ?",
            [len(data), artifact_id, user_id],
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, artifact_id: str, *, user_id: str = "") -> dict | None:
        """Get artifact metadata by ID.

        Read paths keep user_id optional — a missing uid logs a warning
        and returns None so internal helpers (e.g., _resolve_image_path
        in document tools) degrade gracefully. Writes are strict; see
        save/update_*/delete which raise on empty user_id.
        """
        if not user_id:
            log.warning("artifact_store.get.empty_user_id", artifact_id=artifact_id)
            return None
        cursor = await self._db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            [artifact_id, user_id],
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row, strict=False))
        d["metadata"] = json.loads(d.get("metadata", "{}"))
        d["download_url"] = f"/api/artifacts/{d['id']}/download"
        return d

    async def list_for_session(self, session_id: str, *, user_id: str = "") -> list[dict]:
        """List all artifacts for a session."""
        if not user_id:
            log.warning("list_for_session.empty_user_id", session_id=session_id)
            return []
        cursor = await self._db.execute(
            "SELECT * FROM artifacts WHERE session_id = ? AND user_id = ? "
            "ORDER BY created_at DESC",
            [session_id, user_id],
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        results = []
        for row in rows:
            d = dict(zip(cols, row, strict=False))
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            d["download_url"] = f"/api/artifacts/{d['id']}/download"
            results.append(d)
        return results

    # --- Session Canvas binding -------------------------------------------
    # Which artifact is pinned to a session's side-docked canvas (migration
    # 267 + canvas_routes.py). One row per session, user-scoped.

    async def get_canvas_binding(self, session_id: str, *, user_id: str = "") -> str | None:
        """Return the artifact_id pinned to this session's canvas, or None."""
        if not user_id:
            log.warning("artifact_store.get_canvas_binding.empty_user_id", session_id=session_id)
            return None
        cursor = await self._db.execute(
            "SELECT artifact_id FROM session_canvas WHERE session_id = ? AND user_id = ?",
            [session_id, user_id],
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set_canvas_binding(
        self, session_id: str, artifact_id: str, *, user_id: str = ""
    ) -> None:
        """Pin ``artifact_id`` to this session's canvas (upsert)."""
        if not user_id:
            raise ValueError("user_id is required to pin a canvas artifact")
        await self._db.execute(
            "INSERT INTO session_canvas (session_id, artifact_id, user_id, updated_at) "
            "VALUES (?, ?, ?, datetime('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "artifact_id = excluded.artifact_id, "
            "user_id = excluded.user_id, "
            "updated_at = excluded.updated_at",
            [session_id, artifact_id, user_id],
        )
        await self._db.commit()

    async def clear_canvas_binding(self, session_id: str, *, user_id: str = "") -> None:
        """Unpin the canvas for this session."""
        if not user_id:
            raise ValueError("user_id is required to clear a canvas binding")
        await self._db.execute(
            "DELETE FROM session_canvas WHERE session_id = ? AND user_id = ?",
            [session_id, user_id],
        )
        await self._db.commit()

    async def list_for_task(self, task_id: str, *, user_id: str = "") -> list[dict]:
        """List all artifacts for a specific task."""
        if not user_id:
            log.warning("list_for_task.empty_user_id", task_id=task_id)
            return []
        cursor = await self._db.execute(
            "SELECT * FROM artifacts WHERE task_id = ? AND user_id = ? "
            "ORDER BY created_at ASC",
            [task_id, user_id],
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        results = []
        for row in rows:
            d = dict(zip(cols, row, strict=False))
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            d["download_url"] = f"/api/artifacts/{d['id']}/download"
            results.append(d)
        return results

    async def list_all(self, limit: int = 200, include_checkpoints: bool = False, *, user_id: str = "") -> list[dict]:
        """List all artifacts across all sessions, newest first.

        Excludes source_json to keep the response lightweight.
        Use get() to fetch source_json for a specific artifact.
        Checkpoints (intermediate build saves) are hidden by default.
        """
        if not user_id:
            log.warning("list_all.empty_user_id")
            return []
        # Transient rows (e.g. image_search thumbnails) never appear in
        # listings — they're cache, not authored artifacts.
        conditions: list[str] = ["user_id = ?", "transient = 0"]
        params: list = [user_id]
        if not include_checkpoints:
            conditions.append("json_extract(metadata, '$.is_checkpoint') IS NOT 1")
        where = " WHERE " + " AND ".join(conditions)
        params.append(limit)
        cursor = await self._db.execute(
            "SELECT id, task_id, session_id, filename, display_name, format,"
            " size_bytes, path, created_at, metadata, pinned, last_opened_at"
            f" FROM artifacts{where} ORDER BY pinned DESC, created_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        results = []
        for row in rows:
            d = dict(zip(cols, row, strict=False))
            d["metadata"] = json.loads(d.get("metadata", "{}"))
            d["download_url"] = f"/api/artifacts/{d['id']}/download"
            results.append(d)
        return results

    async def set_pinned(self, artifact_id: str, pinned: bool, *, user_id: str) -> None:
        """Toggle pinned status for an artifact."""
        if not user_id:
            raise ValueError("set_pinned requires a user_id")
        await self._db.execute(
            "UPDATE artifacts SET pinned = ? WHERE id = ? AND user_id = ?",
            [1 if pinned else 0, artifact_id, user_id],
        )
        await self._db.commit()

    async def touch_opened(self, artifact_id: str, *, user_id: str) -> None:
        """Update last_opened_at timestamp."""
        if not user_id:
            raise ValueError("touch_opened requires a user_id")
        await self._db.execute(
            "UPDATE artifacts SET last_opened_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            [artifact_id, user_id],
        )
        await self._db.commit()

    # ------------------------------------------------------------------
    # File access
    # ------------------------------------------------------------------

    def get_file_path(self, rel_path: str) -> Path | None:
        """Resolve a relative artifact path to an absolute filesystem path.

        Returns None if the file doesn't exist or escapes the base directory.
        """
        full = (self._base_dir / rel_path).resolve()
        # Prevent path traversal
        if not str(full).startswith(str(self._base_dir.resolve())):
            log.warning("artifact_path_traversal_blocked", path=rel_path)
            return None
        if not full.is_file():
            return None
        return full

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, artifact_id: str, *, user_id: str) -> bool:
        """Delete an artifact record and its file."""
        if not user_id:
            raise ValueError("ArtifactStore.delete requires a user_id")
        info = await self.get(artifact_id, user_id=user_id)
        if not info:
            return False

        # Delete file
        file_path = self.get_file_path(info["path"])
        if file_path and file_path.is_file():
            try:
                os.remove(file_path)
            except OSError:
                log.warning("artifact_file_delete_failed", path=str(file_path))

        # Delete record
        await self._db.execute(
            "DELETE FROM artifacts WHERE id = ? AND user_id = ?",
            [artifact_id, user_id],
        )
        # Manual cascade: migration 309 dropped the artifacts(id) FK (with its
        # ON DELETE CASCADE) from library_activity + library_collection_items so
        # union ids (artifacts + pub_ publications) could share those tables.
        # The cleanup the FK used to do now lives here. Cheap no-op when the
        # artifact was never in a collection / had no activity.
        await self._db.execute(
            "DELETE FROM library_activity WHERE artifact_id = ? AND user_id = ?",
            [artifact_id, user_id],
        )
        await self._db.execute(
            "DELETE FROM library_collection_items WHERE artifact_id = ? AND user_id = ?",
            [artifact_id, user_id],
        )
        await self._db.commit()
        log.info("artifact_deleted", id=artifact_id)

        # Cascade into file_index so the files panel doesn't strand the row
        from augmentum.vfs import unregister_file
        await unregister_file("artifacts", artifact_id, user_id=user_id)

        return True

    async def prune_transient(
        self, *, max_age_days: int = 7, max_total_mb: int = 200,
    ) -> dict:
        """Evict transient artifacts by age, then by total-size cap.

        Runs across all users — transient entries are cache, not user data,
        so global policy is fine. Age cutoff fires first; if what remains
        still exceeds the size cap, the oldest surviving rows are deleted
        until we're under the cap.

        Returns ``{"by_age": int, "by_size": int, "bytes_freed": int}``.
        """
        import time
        cutoff = int(time.time()) - max_age_days * 86400

        # Age sweep — created_at stored as ISO datetime text in this table.
        cursor = await self._db.execute(
            "SELECT id, path, size_bytes FROM artifacts "
            "WHERE transient = 1 AND "
            "strftime('%s', created_at) < ?",
            [str(cutoff)],
        )
        aged = await cursor.fetchall()

        by_age = 0
        bytes_freed = 0
        for art_id, rel_path, size_bytes in aged:
            if await self._delete_transient_row(art_id, rel_path):
                by_age += 1
                bytes_freed += int(size_bytes or 0)

        # Size cap — measure what remains, trim oldest until under cap.
        max_bytes = max_total_mb * 1024 * 1024
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM artifacts WHERE transient = 1",
        )
        (total,) = await cursor.fetchone()
        total = int(total or 0)

        by_size = 0
        if total > max_bytes:
            over = total - max_bytes
            cursor = await self._db.execute(
                "SELECT id, path, size_bytes FROM artifacts "
                "WHERE transient = 1 ORDER BY created_at ASC",
            )
            rows = await cursor.fetchall()
            for art_id, rel_path, size_bytes in rows:
                if over <= 0:
                    break
                if await self._delete_transient_row(art_id, rel_path):
                    by_size += 1
                    freed = int(size_bytes or 0)
                    bytes_freed += freed
                    over -= freed

        if by_age or by_size:
            log.info(
                "transient_artifacts_pruned",
                by_age=by_age, by_size=by_size, bytes_freed=bytes_freed,
            )
        return {"by_age": by_age, "by_size": by_size, "bytes_freed": bytes_freed}

    async def _delete_transient_row(self, art_id: str, rel_path: str) -> bool:
        """Remove one transient row + its file. No VFS cascade (never registered)."""
        file_path = self.get_file_path(rel_path)
        if file_path and file_path.is_file():
            try:
                os.remove(file_path)
            except OSError:
                log.warning("transient_file_delete_failed", path=str(file_path))
        cursor = await self._db.execute(
            "DELETE FROM artifacts WHERE id = ? AND transient = 1",
            [art_id],
        )
        await self._db.commit()
        return (cursor.rowcount or 0) > 0

    async def delete_for_task(self, task_id: str, *, user_id: str) -> int:
        """Delete all artifacts for a task. Returns count deleted."""
        if not user_id:
            raise ValueError("delete_for_task requires a user_id")
        artifacts = await self.list_for_task(task_id, user_id=user_id)
        for a in artifacts:
            file_path = self.get_file_path(a["path"])
            if file_path and file_path.is_file():
                try:
                    os.remove(file_path)
                except OSError:
                    pass

        cursor = await self._db.execute(
            "DELETE FROM artifacts WHERE task_id = ? AND user_id = ?",
            [task_id, user_id],
        )
        await self._db.commit()

        # Cascade into file_index
        if artifacts:
            from augmentum.vfs import unregister_file
            for a in artifacts:
                if a.get("id"):
                    await unregister_file("artifacts", a["id"], user_id=user_id)

        return cursor.rowcount or 0
