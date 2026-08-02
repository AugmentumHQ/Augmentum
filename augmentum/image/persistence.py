"""SQLite persistence for image generation history and model metadata."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.image.schemas import (
    HistoryEntry,
    LoraInfo,
    ModelInfo,
    PipelineType,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class ImagePersistence:
    """Async SQLite persistence layer for image generation data."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        # In-memory cache of the full image_models row set. The fabric
        # heartbeat loop calls list_models() every 5s to build the
        # advertised capability list; on a shared aiosqlite connection
        # contended by writers (media-progress UPDATE, audio_providers
        # SELECT, …) that SELECT was blocking 3+ seconds and driving
        # event-loop stalls. The list mutates only through save_model /
        # delete_model, so a tiny invalidation cache eliminates the
        # heartbeat's contribution to connection contention entirely.
        # ``None`` = unpopulated; empty list = "we checked, no models".
        self._models_cache: list[ModelInfo] | None = None

    # --- Generation history ---

    async def save_generation(
        self,
        *,
        image_id: str,
        session_id: str,
        prompt: str,
        negative_prompt: str,
        model: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        preset: str,
        loras: list[dict],
        file_path: str,
        job_type: str = "txt2img",
        strength: float = 1.0,
        source_image_id: str = "",
        user_id: str,
        origin: str = "",
    ) -> bool:
        """Persist the generation row and index it in the VFS.

        Returns ``True`` when VFS registration succeeded OR when no
        file index is configured (both are acceptable end states).
        Returns ``False`` only when registration was attempted and
        raised — that's the orphan case where the image is on disk
        and in the DB but invisible to the file browser, and callers
        should surface a warning to the user.
        """
        if not user_id:
            raise ValueError("save_generation requires a user_id")
        await self._conn.execute(
            "INSERT INTO image_generations ("
            "image_id, session_id, prompt, negative_prompt, model, seed,"
            " width, height, steps, cfg_scale, preset, loras, file_path,"
            " job_type, strength, source_image_id, user_id, origin"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                image_id, session_id, prompt, negative_prompt, model, seed,
                width, height, steps, cfg_scale, preset, json.dumps(loras),
                file_path, job_type, strength, source_image_id, user_id,
                # Provenance, not silos: 'companion' when she generated
                # it (tool call / architect dispatch); '' = user.
                origin,
            ],
        )
        await self._conn.commit()

        # Index the generation. Stat the file so file_index has the real
        # byte count — save_generation is called after bytes hit disk, so
        # os.path.getsize is reliable. Without this the row starts at 0 B
        # and only the startup repair sweep can fix it later.
        import os
        try:
            size_bytes = os.path.getsize(file_path) if file_path else 0
        except OSError:
            size_bytes = 0

        from augmentum.vfs import file_index_is_configured, register_file

        vfs_expected = file_index_is_configured()
        vfs_id = await register_file(
            user_id=user_id, source="images", source_id=image_id,
            name=f"{image_id}.png", mime_type="image/png",
            size_bytes=size_bytes,
            real_path=file_path, description=prompt[:500] if prompt else "",
            tags=["generated", job_type],
            source_metadata={"prompt": prompt, "model": model, "seed": seed,
                             "width": width, "height": height},
        )
        # Registration failed only when an index was configured but the
        # register call returned no id (register_file logs the exception).
        return not (vfs_expected and vfs_id is None)

    async def get_generation(self, image_id: str, *, user_id: str = "") -> dict | None:
        if not user_id:
            log.warning("get_generation.empty_user_id", image_id=image_id)
            return None
        query = "SELECT * FROM image_generations WHERE image_id = ? AND user_id = ?"
        params: list[object] = [image_id, user_id]
        async with self._conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row, strict=False))

    async def list_generations(
        self,
        limit: int = 50,
        offset: int = 0,
        q: str = "",
        model: str = "",
        preset: str = "",
        sort: str = "newest",
        origin: str = "",
        private: bool | None = None,
        background: bool | None = None,
        *,
        user_id: str = "",
    ) -> list[HistoryEntry]:
        if not user_id:
            log.warning("list_generations.empty_user_id")
            return []
        where_clauses: list[str] = ["user_id = ?"]
        params: list[object] = [user_id]
        if q:
            where_clauses.append("prompt LIKE ?")
            params.append(f"%{q}%")
        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if preset:
            where_clauses.append("preset = ?")
            params.append(preset)
        if private is not None:
            where_clauses.append("is_private = ?")
            params.append(1 if private else 0)
        if background is not None:
            where_clauses.append("is_background = ?")
            params.append(1 if background else 0)
        if origin:
            where_clauses.append("origin = ?")
            params.append(origin)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        order = "ASC" if sort == "oldest" else "DESC"

        query = f"""SELECT image_id, prompt, negative_prompt, model, seed, width,
                           height, steps, cfg_scale, preset, loras, created_at,
                           job_type, strength, source_image_id, is_private,
                           is_background, origin
                    FROM image_generations{where_sql}
                    ORDER BY created_at {order}
                    LIMIT ? OFFSET ?"""
        params.extend([limit, offset])

        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            HistoryEntry(
                image_id=r[0],
                prompt=r[1],
                negative_prompt=r[2],
                model=r[3],
                seed=r[4],
                width=r[5],
                height=r[6],
                steps=r[7],
                cfg_scale=r[8],
                preset=r[9] or "",
                loras=json.loads(r[10]) if r[10] else [],
                created_at=r[11],
                job_type=r[12] if len(r) > 12 else "txt2img",
                strength=r[13] if len(r) > 13 else 1.0,
                source_image_id=r[14] if len(r) > 14 else "",
                is_private=bool(r[15]) if len(r) > 15 else False,
                is_background=bool(r[16]) if len(r) > 16 else False,
                origin=(r[17] or "") if len(r) > 17 else "",
            )
            for r in rows
        ]

    async def count_generations(
        self,
        q: str = "",
        model: str = "",
        preset: str = "",
        origin: str = "",
        private: bool | None = None,
        background: bool | None = None,
        *,
        user_id: str = "",
    ) -> int:
        if not user_id:
            log.warning("count_generations.empty_user_id")
            return 0
        where_clauses: list[str] = ["user_id = ?"]
        params: list[object] = [user_id]
        if q:
            where_clauses.append("prompt LIKE ?")
            params.append(f"%{q}%")
        if model:
            where_clauses.append("model = ?")
            params.append(model)
        if preset:
            where_clauses.append("preset = ?")
            params.append(preset)
        if private is not None:
            where_clauses.append("is_private = ?")
            params.append(1 if private else 0)
        if background is not None:
            where_clauses.append("is_background = ?")
            params.append(1 if background else 0)
        if origin:
            where_clauses.append("origin = ?")
            params.append(origin)

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query = f"SELECT COUNT(*) FROM image_generations{where_sql}"

        async with self._conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def set_private(self, image_id: str, is_private: bool, *, user_id: str) -> bool:
        """Toggle private flag on an image. Returns True if the image existed."""
        if not user_id:
            raise ValueError("set_private requires a user_id")
        cursor = await self._conn.execute(
            "UPDATE image_generations SET is_private = ? "
            "WHERE image_id = ? AND user_id = ?",
            [1 if is_private else 0, image_id, user_id],
        )
        await self._conn.commit()
        if cursor.rowcount <= 0:
            return False
        await self._sync_privacy_to_index([image_id], is_private, user_id=user_id)
        return True

    async def set_private_batch(self, image_ids: list[str], is_private: bool, *, user_id: str) -> int:
        """Toggle private flag on multiple images. Returns count of updated rows."""
        if not user_id:
            raise ValueError("set_private_batch requires a user_id")
        if not image_ids:
            return 0
        placeholders = ",".join("?" for _ in image_ids)
        cursor = await self._conn.execute(
            f"UPDATE image_generations SET is_private = ? "
            f"WHERE image_id IN ({placeholders}) AND user_id = ?",
            [1 if is_private else 0, *image_ids, user_id],
        )
        await self._conn.commit()
        count = cursor.rowcount or 0
        if count > 0:
            await self._sync_privacy_to_index(image_ids, is_private, user_id=user_id)
        return count

    async def _sync_privacy_to_index(
        self, image_ids: list[str], is_private: bool, *, user_id: str,
    ) -> None:
        """Keep file_index in sync with privacy state.

        Private images are unregistered so the Files panel and global search
        can't surface them; un-marking private re-registers from the
        image_generations row. Without this the gallery hides the image but
        the Files tab still lists it — the bug this method fixes.
        """
        import os

        from augmentum.vfs import register_file, unregister_file

        if is_private:
            for image_id in image_ids:
                await unregister_file("images", image_id, user_id=user_id)
            return

        placeholders = ",".join("?" for _ in image_ids)
        async with self._conn.execute(
            "SELECT image_id, file_path, prompt, model, seed, width, height, "
            f"job_type FROM image_generations WHERE image_id IN ({placeholders}) "
            "AND user_id = ?",
            [*image_ids, user_id],
        ) as cursor:
            rows = await cursor.fetchall()

        for image_id, file_path, prompt, model, seed, width, height, job_type in rows:
            try:
                size_bytes = os.path.getsize(file_path) if file_path else 0
            except OSError:
                size_bytes = 0
            await register_file(
                user_id=user_id, source="images", source_id=image_id,
                name=f"{image_id}.png", mime_type="image/png",
                size_bytes=size_bytes,
                real_path=file_path, description=(prompt or "")[:500],
                tags=["generated", job_type or "txt2img"],
                source_metadata={"prompt": prompt or "", "model": model or "",
                                 "seed": seed, "width": width, "height": height},
            )

    async def set_background(self, image_id: str, is_background: bool, *, user_id: str) -> bool:
        """Toggle background collection flag on an image."""
        if not user_id:
            raise ValueError("set_background requires a user_id")
        cursor = await self._conn.execute(
            "UPDATE image_generations SET is_background = ? "
            "WHERE image_id = ? AND user_id = ?",
            [1 if is_background else 0, image_id, user_id],
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def set_background_batch(self, image_ids: list[str], is_background: bool, *, user_id: str) -> int:
        """Toggle background flag on multiple images."""
        if not user_id:
            raise ValueError("set_background_batch requires a user_id")
        if not image_ids:
            return 0
        placeholders = ",".join("?" for _ in image_ids)
        cursor = await self._conn.execute(
            f"UPDATE image_generations SET is_background = ? "
            f"WHERE image_id IN ({placeholders}) AND user_id = ?",
            [1 if is_background else 0, *image_ids, user_id],
        )
        await self._conn.commit()
        return cursor.rowcount

    async def list_backgrounds(self, *, user_id: str = "") -> list[HistoryEntry]:
        """List all images tagged as backgrounds."""
        if not user_id:
            log.warning("list_backgrounds.empty_user_id")
            return []
        query = """SELECT image_id, prompt, negative_prompt, model, seed, width,
                          height, steps, cfg_scale, preset, loras, created_at,
                          job_type, strength, source_image_id, is_private,
                          is_background
                   FROM image_generations WHERE is_background = 1 AND user_id = ?
                   ORDER BY created_at DESC"""
        params: list[object] = [user_id]
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            HistoryEntry(
                image_id=r[0], prompt=r[1], negative_prompt=r[2], model=r[3],
                seed=r[4], width=r[5], height=r[6], steps=r[7], cfg_scale=r[8],
                preset=r[9] or "", loras=json.loads(r[10]) if r[10] else [],
                created_at=r[11], job_type=r[12] if len(r) > 12 else "txt2img",
                strength=r[13] if len(r) > 13 else 1.0,
                source_image_id=r[14] if len(r) > 14 else "",
                is_private=bool(r[15]) if len(r) > 15 else False,
                is_background=bool(r[16]) if len(r) > 16 else False,
            )
            for r in rows
        ]

    async def count_backgrounds(self, *, user_id: str = "") -> int:
        """Count images in the background collection."""
        if not user_id:
            log.warning("count_backgrounds.empty_user_id")
            return 0
        async with self._conn.execute(
            "SELECT COUNT(*) FROM image_generations "
            "WHERE is_background = 1 AND user_id = ?",
            [user_id],
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def delete_generation(self, image_id: str, *, user_id: str) -> str | None:
        """Delete a generation record and its cache entry. Returns the file_path for disk cleanup."""
        if not user_id:
            raise ValueError("delete_generation requires a user_id")
        async with self._conn.execute(
            "SELECT file_path, user_id FROM image_generations "
            "WHERE image_id = ? AND user_id = ?",
            [image_id, user_id],
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        file_path = row[0]
        owner_uid = row[1] or user_id
        await self._conn.execute(
            "DELETE FROM image_cache WHERE image_id = ?", (image_id,),
        )
        await self._conn.execute(
            "DELETE FROM image_generations WHERE image_id = ? AND user_id = ?",
            [image_id, user_id],
        )
        await self._conn.commit()

        # Cascade into file_index so the files panel doesn't strand the row
        if owner_uid:
            from augmentum.vfs import unregister_file
            await unregister_file("images", image_id, user_id=owner_uid)

        return file_path

    # --- Model metadata ---

    async def save_model(
        self,
        *,
        name: str,
        pipeline_type: str,
        path: str,
        source: str = "huggingface",
        size_bytes: int = 0,
        metadata: dict | None = None,
    ) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO image_models
               (name, pipeline_type, path, source, size_bytes, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, pipeline_type, path, source, size_bytes, json.dumps(metadata or {})),
        )
        await self._conn.commit()
        self._models_cache = None  # invalidate list_models cache

    async def delete_model(self, name: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM image_models WHERE name = ?", (name,)
        )
        await self._conn.commit()
        self._models_cache = None  # invalidate list_models cache
        return cursor.rowcount > 0

    async def list_models(self) -> list[ModelInfo]:
        # Cached after first hit, busted on save_model / delete_model.
        # Fabric heartbeat hits this every 5s and was the proximate
        # cause of event-loop stalls when the shared aiosqlite
        # connection was held by writers — see __init__ docstring.
        if self._models_cache is not None:
            return list(self._models_cache)
        async with self._conn.execute(
            "SELECT name, pipeline_type, path, source, size_bytes FROM image_models"
        ) as cursor:
            rows = await cursor.fetchall()
        models = [
            ModelInfo(
                name=r[0],
                pipeline_type=PipelineType(r[1]),
                path=r[2],
                source=r[3],
                size_bytes=r[4],
            )
            for r in rows
        ]
        self._models_cache = models
        return list(models)

    async def get_model(self, name: str) -> ModelInfo | None:
        # Serve from the list cache when available — same invalidation
        # contract; spares a single-row SELECT on the hot heartbeat
        # path when something asks "is X installed?" between full
        # listings.
        if self._models_cache is not None:
            for m in self._models_cache:
                if m.name == name:
                    return m
            return None
        async with self._conn.execute(
            "SELECT name, pipeline_type, path, source, size_bytes FROM image_models WHERE name = ?",
            (name,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return ModelInfo(
            name=row[0],
            pipeline_type=PipelineType(row[1]),
            path=row[2],
            source=row[3],
            size_bytes=row[4],
        )

    # --- LoRA metadata ---

    async def save_lora(
        self,
        *,
        name: str,
        path: str,
        trigger_words: list[str] | None = None,
        size_bytes: int = 0,
    ) -> None:
        await self._conn.execute(
            """INSERT OR REPLACE INTO image_loras
               (name, path, trigger_words, size_bytes)
               VALUES (?, ?, ?, ?)""",
            (name, path, json.dumps(trigger_words or []), size_bytes),
        )
        await self._conn.commit()

    async def list_loras(self) -> list[LoraInfo]:
        async with self._conn.execute(
            "SELECT name, path, trigger_words, size_bytes FROM image_loras"
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            LoraInfo(
                name=r[0],
                path=r[1],
                trigger_words=json.loads(r[2]) if r[2] else [],
                size_bytes=r[3],
            )
            for r in rows
        ]

    # --- Cache ---

    async def save_cache_entry(self, cache_key: str, image_id: str, *, user_id: str) -> None:
        if not user_id:
            raise ValueError("save_cache_entry requires a user_id")
        await self._conn.execute(
            "INSERT OR REPLACE INTO image_cache (cache_key, image_id, user_id) "
            "VALUES (?, ?, ?)",
            [cache_key, image_id, user_id],
        )
        await self._conn.commit()

    async def get_cache_entry(self, cache_key: str, *, user_id: str = "") -> str | None:
        if not user_id:
            log.warning("get_cache_entry.empty_user_id")
            return None
        async with self._conn.execute(
            "SELECT image_id FROM image_cache WHERE cache_key = ? AND user_id = ?",
            [cache_key, user_id],
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    async def cleanup_orphaned(self, output_dir: str, *, user_id: str = "", force: bool = False) -> int:
        """Remove DB rows whose image files no longer exist on disk.

        Maintenance sweep. user_id is intentionally optional: empty means
        "sweep all users" and is how server startup calls this. Pass a
        real user_id when narrowing to one user's rows.

        Safety gate: if more than 10% of total rows would be deleted, the
        sweep is skipped with a warning. This catches the case where the
        image volume is missing or unmounted (which would otherwise
        destroy the entire gallery). Pass ``force=True`` to bypass.

        Returns the number of purged entries (0 when skipped).
        """
        import os

        query = "SELECT image_id, file_path, user_id FROM image_generations"
        params: list[object] = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        total_rows = len(rows)
        if total_rows == 0:
            return 0

        orphaned: list[tuple[str, str]] = []  # (image_id, owner_uid)
        for image_id, file_path, owner_uid in rows:
            if file_path and os.path.exists(file_path):
                continue
            # Also check the output dir fallback path
            fallback = os.path.join(output_dir, f"{image_id}.png")
            if os.path.exists(fallback):
                continue
            orphaned.append((image_id, owner_uid or user_id))

        if not orphaned:
            return 0

        # SAFETY GATE: if more than 10% of rows would be purged, this is
        # likely a volume-mount problem, not real orphans. Log a warning
        # and skip — the caller can re-invoke with force=True.
        orphan_ratio = len(orphaned) / total_rows
        if not force and orphan_ratio > 0.10:
            log.warning(
                "image_orphan_cleanup_blocked",
                orphans=len(orphaned),
                total=total_rows,
                ratio=round(orphan_ratio, 3),
                msg="Too many orphaned images — volume may be missing. "
                    "Run with force=True to purge anyway.",
            )
            return 0

        # Delete in batches of 500 to avoid huge SQL statements
        total = 0
        for i in range(0, len(orphaned), 500):
            batch = orphaned[i : i + 500]
            ids = [oid for oid, _ in batch]
            placeholders = ",".join("?" for _ in ids)
            await self._conn.execute(
                f"DELETE FROM image_cache WHERE image_id IN ({placeholders})",
                ids,
            )
            cursor = await self._conn.execute(
                f"DELETE FROM image_generations WHERE image_id IN ({placeholders})",
                ids,
            )
            # Cascade into file_index per row (needs user_id)
            from augmentum.vfs import unregister_file
            for oid, uid in batch:
                if uid:
                    await unregister_file("images", oid, user_id=uid)
            total += cursor.rowcount
        await self._conn.commit()

        log.info("orphaned_images_cleaned", purged=total)
        return total
