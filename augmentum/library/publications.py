"""Library publications store + storage manager.

``PublicationStore`` is the SQLite layer over ``library_publications``.
``LibraryStorage`` owns the on-disk per-publication directory:
``{root}/{user_id}/{publication_id}/`` with a ``content/`` subdir plus
optional ``screenshot.png`` and an always-present ``meta.json`` mirror.

The two classes coordinate through :meth:`create_or_overwrite` which is
the only entrypoint the routes will need. It performs an atomic snapshot
on disk (sibling temp dir + rename), then writes the row in a single
SQL UPSERT. If either side fails, the other rolls back so a publication
never half-exists.

Title is the soft natural key — ``(user_id, title)`` uniqueness is
enforced at the Python layer, not by a SQL UNIQUE, so the
overwrite-vs-rename UI can introspect the existing row's metadata
before deciding. Last-write-wins on overwrite: the previous bytes are
gone, ``version`` increments, ``updated_at`` bumps.
"""

from __future__ import annotations

import json
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Constants ──────────────────────────────────────────────────────────

# Publication IDs are prefixed so a stray ID in logs / URLs is obviously
# a library publication, not e.g. a chat message id.
_ID_PREFIX = "pub_"
_ID_NONCE_BYTES = 12  # 96 bits → ~16 char hex; sufficient anti-collision
                      # for a single-user catalog that will never exceed
                      # millions of rows.

# Storage kind discriminator written to the row + meta.json.
StorageKind = Literal["single", "bundle"]

# Kind values surfaced to the UI. v1 supports the first two; doc/other
# are reserved so a future migration doesn't have to expand the enum.
PublicationKind = Literal["game", "app", "doc", "other"]


# ── Exceptions ─────────────────────────────────────────────────────────


class TitleCollision(Exception):
    """Raised when a save would overwrite an existing publication and the
    caller asked for ``on_collision="abort"``. Carries the existing row
    so the route can surface its ID and last-modified time to the UI."""

    def __init__(self, existing: dict[str, Any]) -> None:
        super().__init__(
            f"publication with title {existing.get('title')!r} already exists"
        )
        self.existing = existing


class SizeBudgetExceeded(Exception):
    """Raised when a save would push the user over their configured
    per-publication or cumulative budget. Carries both numbers so the
    route can format a helpful error."""

    def __init__(
        self,
        *,
        attempted_bytes: int,
        limit_bytes: int,
        scope: Literal["per_publication", "user_total"],
    ) -> None:
        super().__init__(
            f"library budget exceeded: {attempted_bytes} > {limit_bytes} ({scope})"
        )
        self.attempted_bytes = attempted_bytes
        self.limit_bytes = limit_bytes
        self.scope = scope


# ── Helpers ────────────────────────────────────────────────────────────


def _new_publication_id() -> str:
    return _ID_PREFIX + secrets.token_hex(_ID_NONCE_BYTES)


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def _dir_size_bytes(path: Path) -> int:
    """Sum of all files under ``path``. Symlinks not followed."""
    if not path.is_dir():
        if path.is_file():
            return path.stat().st_size
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
    return total


# ── Storage ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SavedBundle:
    """Result of :meth:`LibraryStorage.write_bundle`. Fields are what the
    PublicationStore needs to populate a row."""

    storage_path: str   # absolute path of the publication dir
    storage_kind: StorageKind
    entry_point: str    # relative path inside content/
    size_bytes: int     # content/ size only (screenshot added later)


class LibraryStorage:
    """Owns the ``{root}/{user_id}/{publication_id}/`` tree.

    All write paths are atomic in the sense that a failure mid-write
    leaves no partial publication visible — the destination dir does
    not exist until ``os.replace`` swaps a fully-populated temp dir
    into place.

    The store is intentionally synchronous. File-tree copies block the
    event loop briefly (Phase 1 caps at 50MB so this is sub-second),
    and routes can wrap in ``asyncio.to_thread`` if a publication
    grows large enough to matter.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _user_dir(self, user_id: str) -> Path:
        if not user_id:
            raise ValueError("LibraryStorage requires non-empty user_id")
        return self._root / user_id

    def publication_dir(self, user_id: str, publication_id: str) -> Path:
        return self._user_dir(user_id) / publication_id

    def write_bundle(
        self,
        *,
        user_id: str,
        publication_id: str,
        source_path: str | Path,
        entry_point: str,
    ) -> SavedBundle:
        """Snapshot ``source_path`` into the publication's content/ dir.

        ``source_path`` may be a directory (bundle) or a single file
        (single). ``entry_point`` is the relative path the launcher
        will open and must exist inside the snapshot. The method
        verifies that, raising ``FileNotFoundError`` early if not.

        Atomicity: writes go to ``{pub_dir}.tmp-<token>``, then
        ``os.replace`` swaps it into place. If an old publication dir
        exists (overwrite path), it's moved aside to ``{pub_dir}.old-<token>``
        and only deleted after the swap succeeds.
        """
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"source_path does not exist: {src}")

        # Validate entry point inside the source before we copy anything.
        if src.is_file():
            storage_kind: StorageKind = "single"
            if entry_point != src.name:
                # For single-file saves the entry point IS the file. Let
                # callers be lazy and supply either; surface a clear
                # error if they supplied something inconsistent.
                raise ValueError(
                    f"entry_point {entry_point!r} doesn't match single file {src.name!r}"
                )
        else:
            storage_kind = "bundle"
            entry_candidate = (src / entry_point).resolve()
            if not entry_candidate.is_file():
                raise FileNotFoundError(
                    f"entry_point {entry_point!r} not found under {src}"
                )
            if not str(entry_candidate).startswith(str(src) + ("\\" if "\\" in str(src) else "/")):
                # Defensive: entry_point traversed out of source.
                raise ValueError(f"entry_point escapes source: {entry_point!r}")

        dest = self.publication_dir(user_id, publication_id)
        dest.parent.mkdir(parents=True, exist_ok=True)

        token = secrets.token_hex(4)
        tmp_dir = dest.parent / f"{dest.name}.tmp-{token}"
        old_dir = dest.parent / f"{dest.name}.old-{token}" if dest.exists() else None

        try:
            tmp_dir.mkdir(parents=True)
            content_dir = tmp_dir / "content"

            if storage_kind == "single":
                content_dir.mkdir(parents=True)
                shutil.copy2(src, content_dir / src.name)
            else:
                # copytree creates the destination — let it own content/.
                shutil.copytree(
                    src,
                    content_dir,
                    symlinks=False,
                    copy_function=shutil.copy2,
                    dirs_exist_ok=False,
                )

            size = _dir_size_bytes(content_dir)

            # Move old aside first (if any), then swap tmp into place.
            if old_dir is not None:
                dest.rename(old_dir)
            tmp_dir.rename(dest)

            if old_dir is not None and old_dir.exists():
                shutil.rmtree(old_dir, ignore_errors=True)

            return SavedBundle(
                storage_path=str(dest),
                storage_kind=storage_kind,
                entry_point=entry_point if storage_kind == "bundle" else src.name,
                size_bytes=size,
            )
        except Exception:
            # Cleanup partial state.
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            # If we moved old aside but never swapped tmp, restore it.
            if old_dir is not None and old_dir.exists() and not dest.exists():
                old_dir.rename(dest)
            raise

    def write_screenshot(
        self, *, user_id: str, publication_id: str, png_bytes: bytes
    ) -> str:
        """Write screenshot.png. Returns relative path stored on the row."""
        pub_dir = self.publication_dir(user_id, publication_id)
        if not pub_dir.is_dir():
            raise FileNotFoundError(f"publication dir missing: {pub_dir}")
        target = pub_dir / "screenshot.png"
        target.write_bytes(png_bytes)
        return "screenshot.png"

    def write_meta_json(
        self, *, user_id: str, publication_id: str, meta: dict[str, Any]
    ) -> None:
        pub_dir = self.publication_dir(user_id, publication_id)
        if not pub_dir.is_dir():
            raise FileNotFoundError(f"publication dir missing: {pub_dir}")
        (pub_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def asset_path(
        self, *, user_id: str, publication_id: str, rel_path: str
    ) -> Path | None:
        """Resolve a relative content/ path safely. Returns None on
        traversal or missing file. Used by the asset-serve route."""
        pub_dir = self.publication_dir(user_id, publication_id)
        content_dir = pub_dir / "content"
        if not content_dir.is_dir():
            return None
        # Reject any segment that looks like an absolute path or traversal.
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            return None
        candidate = (content_dir / rel_path).resolve()
        # Ensure candidate stays inside content_dir.
        try:
            candidate.relative_to(content_dir.resolve())
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def delete_publication(self, *, user_id: str, publication_id: str) -> bool:
        """Remove the on-disk publication dir. Idempotent."""
        pub_dir = self.publication_dir(user_id, publication_id)
        if not pub_dir.exists():
            return False
        shutil.rmtree(pub_dir)
        # Best-effort cleanup of empty user dir.
        user_dir = self._user_dir(user_id)
        try:
            if user_dir.is_dir() and not any(user_dir.iterdir()):
                user_dir.rmdir()
        except OSError:
            pass
        return True


# ── Store ──────────────────────────────────────────────────────────────


class PublicationStore:
    """CRUD over ``library_publications``. user_id-scoped on every read.

    The store coordinates with :class:`LibraryStorage` through
    :meth:`create_or_overwrite`, which is the only write path the
    application uses. Other writes (PATCH for rename/description,
    DELETE) operate on the row only — caller is responsible for any
    storage cleanup via :class:`LibraryStorage`.
    """

    def __init__(self, conn: aiosqlite.Connection, storage: LibraryStorage) -> None:
        self._conn = conn
        self._storage = storage

    @property
    def storage(self) -> LibraryStorage:
        return self._storage

    # ── Reads ──────────────────────────────────────────────────────────

    async def get(self, publication_id: str, *, user_id: str) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT * FROM library_publications WHERE id = ? AND user_id = ?",
            (publication_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def get_by_title(
        self, *, user_id: str, title: str
    ) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT * FROM library_publications "
            "WHERE user_id = ? AND title = ? "
            "ORDER BY version DESC LIMIT 1",
            (user_id, title),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_for_user(
        self,
        *,
        user_id: str,
        kind: PublicationKind | None = None,
        limit: int = 200,
    ) -> list[dict]:
        if kind:
            cursor = await self._conn.execute(
                "SELECT * FROM library_publications "
                "WHERE user_id = ? AND kind = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, kind, int(limit)),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM library_publications "
                "WHERE user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, int(limit)),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def total_bytes_for_user(self, *, user_id: str) -> int:
        cursor = await self._conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM library_publications "
            "WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    # ── Writes ─────────────────────────────────────────────────────────

    async def create_or_overwrite(
        self,
        *,
        user_id: str,
        workspace_id: str,
        title: str,
        description: str,
        kind: PublicationKind,
        source_path: str | Path,
        entry_point: str,
        screenshot_bytes: bytes | None = None,
        on_collision: Literal["overwrite", "abort"] = "abort",
        max_bytes: int,
        user_budget_bytes: int,
    ) -> dict[str, Any]:
        """Atomic create-or-overwrite by (user_id, title).

        Flow:
        1. Check title collision. If existing + abort → raise.
        2. Snapshot to disk via LibraryStorage (atomic temp+rename).
        3. Verify size budgets (per-publication, then per-user cumulative
           with the OLD row's bytes deducted on overwrite).
        4. Write screenshot if provided.
        5. UPSERT row in one transaction.
        6. Write meta.json mirror.

        On any failure after step 2, the on-disk snapshot is rolled
        back so the catalog stays consistent.
        """
        if not user_id:
            raise ValueError("create_or_overwrite requires user_id")
        if not title.strip():
            raise ValueError("create_or_overwrite requires title")

        existing = await self.get_by_title(user_id=user_id, title=title)
        if existing and on_collision == "abort":
            raise TitleCollision(existing)

        action: Literal["created", "overwritten"] = (
            "overwritten" if existing else "created"
        )
        publication_id = existing["id"] if existing else _new_publication_id()
        new_version = (existing["version"] + 1) if existing else 1

        # Snapshot first so we know the actual byte size before committing.
        bundle = self._storage.write_bundle(
            user_id=user_id,
            publication_id=publication_id,
            source_path=source_path,
            entry_point=entry_point,
        )

        try:
            # Per-publication cap.
            if bundle.size_bytes > max_bytes:
                raise SizeBudgetExceeded(
                    attempted_bytes=bundle.size_bytes,
                    limit_bytes=max_bytes,
                    scope="per_publication",
                )

            # Per-user cumulative cap. On overwrite, the OLD row's bytes
            # are about to be replaced — subtract them before adding new.
            current_total = await self.total_bytes_for_user(user_id=user_id)
            old_bytes = int(existing["size_bytes"]) if existing else 0
            projected = current_total - old_bytes + bundle.size_bytes
            if projected > user_budget_bytes:
                raise SizeBudgetExceeded(
                    attempted_bytes=projected,
                    limit_bytes=user_budget_bytes,
                    scope="user_total",
                )

            screenshot_rel = ""
            if screenshot_bytes:
                screenshot_rel = self._storage.write_screenshot(
                    user_id=user_id,
                    publication_id=publication_id,
                    png_bytes=screenshot_bytes,
                )

            now = time.time()
            created_at = float(existing["created_at"]) if existing else now

            if existing:
                await self._conn.execute(
                    """UPDATE library_publications SET
                        workspace_id   = ?,
                        kind           = ?,
                        description    = ?,
                        screenshot_path = CASE WHEN ? != '' THEN ? ELSE screenshot_path END,
                        entry_point    = ?,
                        storage_path   = ?,
                        storage_kind   = ?,
                        size_bytes     = ?,
                        version        = ?,
                        updated_at     = ?
                       WHERE id = ? AND user_id = ?""",
                    (
                        workspace_id,
                        kind,
                        description,
                        screenshot_rel, screenshot_rel,
                        bundle.entry_point,
                        bundle.storage_path,
                        bundle.storage_kind,
                        bundle.size_bytes,
                        new_version,
                        now,
                        publication_id,
                        user_id,
                    ),
                )
            else:
                await self._conn.execute(
                    """INSERT INTO library_publications (
                        id, user_id, workspace_id, kind, title, description,
                        screenshot_path, entry_point, storage_path, storage_kind,
                        size_bytes, version, shared, created_at, updated_at,
                        last_launched_at, launch_count
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, 0)""",
                    (
                        publication_id, user_id, workspace_id, kind, title, description,
                        screenshot_rel, bundle.entry_point, bundle.storage_path,
                        bundle.storage_kind, bundle.size_bytes, new_version,
                        created_at, now,
                    ),
                )
            await self._conn.commit()

            row = await self.get(publication_id, user_id=user_id)
            assert row is not None  # we just wrote it

            # Mirror to disk last — non-authoritative; failure here is
            # logged but doesn't roll back. The catalog row IS the truth.
            try:
                self._storage.write_meta_json(
                    user_id=user_id,
                    publication_id=publication_id,
                    meta={**row, "_action": action},
                )
            except OSError as exc:
                log.warning(
                    "library_meta_write_failed",
                    publication_id=publication_id,
                    user_id=user_id,
                    error=str(exc),
                )

            row["_action"] = action
            return row

        except Exception:
            # Roll back the on-disk snapshot so we don't leak storage
            # without a row pointing at it.
            self._storage.delete_publication(
                user_id=user_id, publication_id=publication_id
            )
            # If we were overwriting and the old bytes were already swapped
            # away by write_bundle, the old data is unrecoverable — by
            # design (last-write-wins). The DB row is still the old one
            # because we never committed. The next save will rebuild.
            raise

    async def patch(
        self,
        publication_id: str,
        *,
        user_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> dict | None:
        """Rename or edit description. Returns the updated row or None
        if not found. Title rename respects (user_id, title) uniqueness
        — raises TitleCollision if the new title is already taken."""
        existing = await self.get(publication_id, user_id=user_id)
        if not existing:
            return None

        if title is not None and title != existing["title"]:
            collide = await self.get_by_title(user_id=user_id, title=title)
            if collide and collide["id"] != publication_id:
                raise TitleCollision(collide)

        new_title = title if title is not None else existing["title"]
        new_desc = description if description is not None else existing["description"]
        now = time.time()
        await self._conn.execute(
            "UPDATE library_publications "
            "SET title = ?, description = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (new_title, new_desc, now, publication_id, user_id),
        )
        await self._conn.commit()
        return await self.get(publication_id, user_id=user_id)

    async def delete(self, publication_id: str, *, user_id: str) -> bool:
        existing = await self.get(publication_id, user_id=user_id)
        if not existing:
            return False
        await self._conn.execute(
            "DELETE FROM library_publications WHERE id = ? AND user_id = ?",
            (publication_id, user_id),
        )
        # Manual cascade: migration 309 dropped the artifacts(id) FK from
        # library_activity + library_collection_items so publications could
        # join them. Publications never had that FK's ON DELETE CASCADE (their
        # ids aren't in artifacts), so this cleanup is the only thing that
        # purges a deleted publication's activity + collection rows.
        await self._conn.execute(
            "DELETE FROM library_activity WHERE artifact_id = ? AND user_id = ?",
            (publication_id, user_id),
        )
        await self._conn.execute(
            "DELETE FROM library_collection_items WHERE artifact_id = ? AND user_id = ?",
            (publication_id, user_id),
        )
        await self._conn.commit()
        self._storage.delete_publication(
            user_id=user_id, publication_id=publication_id
        )
        return True

    async def set_pinned(
        self, publication_id: str, *, user_id: str, pinned: bool
    ) -> dict | None:
        """Toggle the pinned/favorite flag. Returns the updated row or None
        if not found. Parity with artifacts.pinned (mig 057); the
        /api/library/items UNION reads this column since migration 309."""
        existing = await self.get(publication_id, user_id=user_id)
        if not existing:
            return None
        await self._conn.execute(
            "UPDATE library_publications SET pinned = ? WHERE id = ? AND user_id = ?",
            (1 if pinned else 0, publication_id, user_id),
        )
        await self._conn.commit()
        return await self.get(publication_id, user_id=user_id)

    async def set_tags(
        self, publication_id: str, *, user_id: str, tags: list[str]
    ) -> dict | None:
        """Replace the tag set (JSON array). Returns the updated row or None
        if not found. Parity with artifacts.tags (mig 235)."""
        existing = await self.get(publication_id, user_id=user_id)
        if not existing:
            return None
        await self._conn.execute(
            "UPDATE library_publications SET tags = ? WHERE id = ? AND user_id = ?",
            (json.dumps(list(tags)), publication_id, user_id),
        )
        await self._conn.commit()
        return await self.get(publication_id, user_id=user_id)

    async def record_launch(
        self, publication_id: str, *, user_id: str
    ) -> None:
        """Bump last_launched_at + launch_count. No-op if row missing."""
        now = time.time()
        await self._conn.execute(
            "UPDATE library_publications "
            "SET last_launched_at = ?, launch_count = launch_count + 1 "
            "WHERE id = ? AND user_id = ?",
            (now, publication_id, user_id),
        )
        await self._conn.commit()
