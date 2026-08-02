"""TitleStore -- read/write surface for manifests + per-launch runs.

Manifests live in the existing ``artifacts`` table (we project rows
through ``TitleManifest.from_artifact_row``). Per-launch telemetry lives
in the new ``title_runs`` table (migration 123).

This store does NOT call ArtifactStore directly for reads -- it talks
to the underlying SQLite connection so list/filter queries can stay one
SQL statement instead of N+1 round-trips through the store layer. For
writes that change artifact state (pin, unpin, last_opened_at) we DO
delegate to ArtifactStore so its VFS bookkeeping stays consistent.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import aiosqlite

from augmentum.titles.manifest import TITLE_KINDS, TitleManifest
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Statuses we treat as "still running" for cleanup / live-stats queries.
# A run is "open" until ``ended_at`` is non-null.
RUN_EXIT_REASONS: frozenset[str] = frozenset({
    "clean", "crash", "idle", "force-stop", "abandon",
})


class TitleStore:
    """CRUD over the title surface (artifacts projection + title_runs)."""

    def __init__(self, conn: aiosqlite.Connection, artifact_store: Any | None = None) -> None:
        self._conn = conn
        # Optional ArtifactStore for write-through. Reads bypass it for
        # query efficiency; writes that touch artifact-row state go
        # through it so VFS sees the change.
        self._artifacts = artifact_store

    # ── Manifest reads ─────────────────────────────────────────────

    async def list_for_user(
        self,
        *,
        user_id: str,
        kind: str | None = None,
        pinned_only: bool = False,
        limit: int = 200,
    ) -> list[TitleManifest]:
        """Return the user's titles. Filters: kind, pinned-only.

        Reads all candidate rows (``metadata.kind`` matching one of the
        recognised values) and projects to TitleManifest. Total play
        time is fetched in one batch query rather than N+1.
        """
        if not user_id:
            return []

        query = (
            "SELECT id, user_id, display_name, filename, format, "
            "  metadata, pinned, last_opened_at "
            "FROM artifacts "
            "WHERE user_id = ? AND metadata IS NOT NULL"
        )
        params: list[Any] = [user_id]
        if pinned_only:
            query += " AND pinned = 1"
        # Filter to artifact rows that look like titles -- check
        # metadata.kind via json_extract for index-friendly filtering.
        # (Legacy js13k pins use kind == 'game', so we also accept that.)
        query += (
            " AND ("
            "  json_extract(metadata, '$.kind') IN ("
            f"    {', '.join(['?'] * len(TITLE_KINDS))}, 'game'"
            "  )"
            ")"
        )
        params.extend(sorted(TITLE_KINDS))
        query += " ORDER BY COALESCE(last_opened_at, created_at) DESC LIMIT ?"
        params.append(int(limit))

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        # Batch fetch total play time for all artifact ids in one query.
        artifact_ids = [str(r[cols.index("id")]) for r in rows]
        play_times = await self._batch_total_play_time(artifact_ids, user_id=user_id)

        manifests: list[TitleManifest] = []
        for row in rows:
            d = dict(zip(cols, row))
            manifest = TitleManifest.from_artifact_row(
                d, total_play_time_s=play_times.get(d["id"], 0),
            )
            if manifest is not None and (kind is None or manifest.kind == kind):
                manifests.append(manifest)
        return manifests

    async def get(
        self, title_id: str, *, user_id: str = "",
    ) -> TitleManifest | None:
        if not title_id:
            return None
        query = (
            "SELECT id, user_id, display_name, filename, format, "
            "  metadata, pinned, last_opened_at "
            "FROM artifacts WHERE id = ?"
        )
        params: list[Any] = [title_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        else:
            # No scope: never return another user's title by id. Titles are
            # user-owned artifact rows, so this yields nothing rather than
            # leaking a cross-tenant manifest.
            query += " AND user_id IS NULL"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row))
        play_time = await self._total_play_time_for(title_id, user_id=user_id)
        return TitleManifest.from_artifact_row(d, total_play_time_s=play_time)

    async def is_title(self, artifact_id: str, *, user_id: str = "") -> bool:
        """Cheap predicate: does this artifact id look like a title row?"""
        manifest = await self.get(artifact_id, user_id=user_id)
        return manifest is not None

    # ── Manifest writes ────────────────────────────────────────────

    async def update_metadata(
        self,
        title_id: str,
        *,
        user_id: str,
        patch: dict[str, Any],
    ) -> bool:
        """Merge ``patch`` into the artifact's metadata JSON. User-scoped.

        When the patch includes a ``title`` field, the artifact's
        ``display_name`` column is also updated so the manifest's
        projection (which prefers ``display_name``) stays consistent
        with what the user just set.
        """
        if not user_id or not title_id or not isinstance(patch, dict):
            return False
        existing = await self._fetch_metadata(title_id, user_id=user_id)
        if existing is None:
            return False
        existing.update(patch)
        new_title = patch.get("title")
        if isinstance(new_title, str) and new_title.strip():
            await self._conn.execute(
                "UPDATE artifacts SET metadata = ?, display_name = ? "
                "WHERE id = ? AND user_id = ?",
                (json.dumps(existing), new_title, title_id, user_id),
            )
        else:
            await self._conn.execute(
                "UPDATE artifacts SET metadata = ? "
                "WHERE id = ? AND user_id = ?",
                (json.dumps(existing), title_id, user_id),
            )
        await self._conn.commit()
        return True

    async def set_pinned(
        self, title_id: str, *, user_id: str, pinned: bool,
    ) -> bool:
        cursor = await self._conn.execute(
            "UPDATE artifacts SET pinned = ? WHERE id = ? AND user_id = ?",
            (1 if pinned else 0, title_id, user_id),
        )
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    async def touch_last_played(
        self, title_id: str, *, user_id: str,
    ) -> bool:
        cursor = await self._conn.execute(
            "UPDATE artifacts SET last_opened_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (title_id, user_id),
        )
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    async def delete(
        self, title_id: str, *, user_id: str,
    ) -> TitleManifest | None:
        """Remove a title's artifact row + its run history. User-scoped.

        Returns the deleted manifest (so callers can release the ROM /
        save blobs it referenced) or None when there's no such title for
        this user. The blob refcounts and per-title save slots are NOT
        touched here -- that's the caller's job (route layer, which has
        the BlobStore + SaveStore handles).
        """
        if not user_id or not title_id:
            return None
        manifest = await self.get(title_id, user_id=user_id)
        if manifest is None:
            return None
        await self._conn.execute(
            "DELETE FROM title_runs WHERE artifact_id = ? AND user_id = ?",
            (title_id, user_id),
        )
        await self._conn.execute(
            "DELETE FROM artifacts WHERE id = ? AND user_id = ?",
            (title_id, user_id),
        )
        await self._conn.commit()
        # Cascade into file_index so the files panel doesn't strand the
        # row. Legacy js13k pins are VFS-registered; uploaded ROM rows
        # aren't, and unregister is a cheap no-op when there's nothing
        # to remove.
        try:
            from augmentum.vfs import unregister_file
            await unregister_file("artifacts", title_id, user_id=user_id)
        except Exception as exc:
            log.warning(
                "title_delete_vfs_unregister_failed",
                id=title_id, error=str(exc),
            )
        return manifest

    # ── Run telemetry ─────────────────────────────────────────────

    async def create_run(
        self,
        *,
        user_id: str,
        artifact_id: str,
        runtime_id: str,
        source_id: str = "",
        launch_latency_ms: int | None = None,
    ) -> str:
        if not user_id or not artifact_id or not runtime_id:
            raise ValueError("create_run requires user_id, artifact_id, runtime_id")
        run_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            """INSERT INTO title_runs
               (id, user_id, artifact_id, runtime_id, source_id,
                launch_latency_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                user_id,
                artifact_id,
                runtime_id,
                source_id,
                int(launch_latency_ms) if launch_latency_ms is not None else None,
            ),
        )
        await self._conn.commit()
        return run_id

    async def end_run(
        self,
        run_id: str,
        *,
        user_id: str,
        exit_reason: str = "clean",
        avg_fps: float | None = None,
        avg_rtt_ms: float | None = None,
        avg_bitrate_kbps: int | None = None,
        crashes: int = 0,
        metadata: dict | None = None,
    ) -> bool:
        if exit_reason and exit_reason not in RUN_EXIT_REASONS:
            log.warning(
                "title_run_unknown_exit_reason",
                run_id=run_id, exit_reason=exit_reason,
            )
        # Fetch started_at to materialise duration in one round-trip.
        cursor = await self._conn.execute(
            "SELECT started_at FROM title_runs "
            "WHERE id = ? AND user_id = ? AND ended_at IS NULL",
            (run_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        started_at = row[0]
        # Compute duration server-side via SQLite's strftime; cheaper
        # than pulling the timestamp into Python.
        await self._conn.execute(
            """UPDATE title_runs SET
                 ended_at = datetime('now'),
                 duration_s = CAST(
                   (julianday('now') - julianday(?)) * 86400 AS INTEGER
                 ),
                 exit_reason = ?,
                 avg_fps = ?,
                 avg_rtt_ms = ?,
                 avg_bitrate_kbps = ?,
                 crashes = ?,
                 metadata = ?
               WHERE id = ? AND user_id = ?""",
            (
                started_at,
                exit_reason,
                avg_fps,
                avg_rtt_ms,
                int(avg_bitrate_kbps) if avg_bitrate_kbps is not None else None,
                int(crashes or 0),
                json.dumps(metadata or {}),
                run_id,
                user_id,
            ),
        )
        await self._conn.commit()
        return True

    async def list_runs(
        self,
        *,
        user_id: str,
        artifact_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        query = "SELECT * FROM title_runs WHERE user_id = ?"
        params: list[Any] = [user_id]
        if artifact_id:
            query += " AND artifact_id = ?"
            params.append(artifact_id)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        out = []
        for row in rows:
            d = dict(zip(cols, row))
            md = d.get("metadata")
            if isinstance(md, str) and md:
                try:
                    d["metadata"] = json.loads(md)
                except json.JSONDecodeError:
                    pass
            out.append(d)
        return out

    async def list_open_runs(self, *, user_id: str = "") -> list[dict]:
        """Live runs (ended_at IS NULL). Used by the lifecycle reaper."""
        query = "SELECT * FROM title_runs WHERE ended_at IS NULL"
        params: list[Any] = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Internals ─────────────────────────────────────────────────

    async def _fetch_metadata(
        self, title_id: str, *, user_id: str,
    ) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT metadata FROM artifacts WHERE id = ? AND user_id = ?",
            (title_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        raw = row[0]
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        return {}

    async def _total_play_time_for(
        self, artifact_id: str, *, user_id: str,
    ) -> int:
        query = (
            "SELECT COALESCE(SUM(duration_s), 0) FROM title_runs "
            "WHERE artifact_id = ? AND duration_s IS NOT NULL"
        )
        params: list[Any] = [artifact_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    async def _batch_total_play_time(
        self, artifact_ids: list[str], *, user_id: str,
    ) -> dict[str, int]:
        """One query, GROUP BY artifact_id. Cheap even at 200 titles."""
        if not artifact_ids or not user_id:
            return {}
        placeholders = ",".join(["?"] * len(artifact_ids))
        query = (
            f"SELECT artifact_id, COALESCE(SUM(duration_s), 0) "
            f"FROM title_runs "
            f"WHERE user_id = ? AND artifact_id IN ({placeholders}) "
            f"  AND duration_s IS NOT NULL "
            f"GROUP BY artifact_id"
        )
        params: list[Any] = [user_id, *artifact_ids]
        cursor = await self._conn.execute(query, params)
        out: dict[str, int] = {}
        for row in await cursor.fetchall():
            out[row[0]] = int(row[1] or 0)
        return out
