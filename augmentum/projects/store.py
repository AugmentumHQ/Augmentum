"""Project entity store + on-disk bare-repo manager.

``ProjectStore`` is the SQLite layer over the three tables introduced in
migration 199: ``projects``, ``project_repos``, ``project_refs``.

``ProjectRepoStorage`` owns the per-user directory of bare git
repositories at ``{data_dir}/projects/{user_id}/{project_id}.git/``. A
bare repo is the durable source of truth for a Project's history — it
survives container recycle, can be cloned into a workspace, and is the
target every Coder ``git_checkpoint`` pushes to.

Phase 1 / PR-1.1 ships these two classes plus the substrate wiring.
PR-1.2 retargets ``containers.py`` onto them; PR-1.3 wires Library
publications onto ``project_refs``.
"""

from __future__ import annotations

import asyncio
import re
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Constants ──────────────────────────────────────────────────────────

# Project IDs are prefixed so a stray ID in logs / URLs is obviously a
# Project, not a chat message or publication. 12 hex chars ≈ 48 bits of
# entropy — sufficient for a per-user space that will never exceed
# thousands of rows.
_ID_PREFIX = "prj_"
_ID_NONCE_BYTES = 6

ProjectKind = Literal["scratchpad", "app", "coder"]
"""User-facing project kind. v1 set; new kinds can be added without a
migration since the column is an open TEXT."""

ProjectRefKind = Literal[
    "branch",
    "tag",
    "savepoint",
    "publication",
    "share",
]
"""Kind of git ref. ``branch`` and ``savepoint`` are Phase 1 use cases;
``publication`` is wired by PR-1.3; ``share`` lights up in Phase 5+."""

# Slug generation: lowercase ASCII alnum + dashes, max 64 chars.
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_SLUG_MAX = 64

# Bare-repo dirname format. ``.git`` suffix is conventional and lets a
# user mistakenly `cd` into the dir without confusion.
_REPO_SUFFIX = ".git"


def _new_project_id() -> str:
    return _ID_PREFIX + secrets.token_hex(_ID_NONCE_BYTES)


def _new_ref_id() -> str:
    # Refs use their own prefix so log/audit messages disambiguate.
    return "ref_" + secrets.token_hex(_ID_NONCE_BYTES)


def _slugify(name: str) -> str:
    s = _SLUG_STRIP.sub("-", (name or "").lower()).strip("-")
    return s[:_SLUG_MAX] or "project"


def _now() -> float:
    # REAL epoch seconds — matches created_at column types in 197/198.
    return time.time()


# ── Exceptions ─────────────────────────────────────────────────────────


class SlugCollision(Exception):
    """Raised when an explicit slug collides with an existing project
    for the same user. The auto-suffix path (``create()`` with no
    ``slug=``) never raises this; only callers passing ``slug=`` do."""

    def __init__(self, slug: str, existing_id: str) -> None:
        super().__init__(f"slug {slug!r} already used by project {existing_id}")
        self.slug = slug
        self.existing_id = existing_id


# ── Bare-repo storage ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BareRepoHandle:
    """Result of :meth:`ProjectRepoStorage.init_bare`."""

    repo_path: str  # absolute on host
    head_ref: str   # 'refs/heads/<default_branch>'


class ProjectRepoStorage:
    """Owns ``{root}/{user_id}/{project_id}.git/``.

    The class is intentionally synchronous. Git operations are bounded
    (init takes <100ms even on a slow disk) so wrapping in
    ``asyncio.to_thread`` at the caller is fine. ``ProjectStore``
    already does this on the hot paths.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _user_dir(self, user_id: str) -> Path:
        if not user_id:
            raise ValueError("ProjectRepoStorage requires non-empty user_id")
        return self._root / user_id

    def bare_repo_path(self, user_id: str, project_id: str) -> Path:
        return self._user_dir(user_id) / f"{project_id}{_REPO_SUFFIX}"

    def init_bare(
        self,
        *,
        user_id: str,
        project_id: str,
        default_branch: str = "main",
    ) -> BareRepoHandle:
        """Create the bare repo if it doesn't exist. Idempotent.

        Uses ``git init --bare`` rather than hand-rolling the dir
        structure — keeps us in step with whatever shape upstream git
        considers canonical (HEAD format, default config, hooks dir).
        """
        repo_dir = self.bare_repo_path(user_id, project_id)
        head_ref = f"refs/heads/{default_branch}"

        if (repo_dir / "HEAD").exists():
            return BareRepoHandle(repo_path=str(repo_dir), head_ref=head_ref)

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        # ``--initial-branch`` is git ≥ 2.28; the supported versions in
        # both host and container are well past that.
        subprocess.run(
            [
                "git",
                "init",
                "--bare",
                f"--initial-branch={default_branch}",
                str(repo_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return BareRepoHandle(repo_path=str(repo_dir), head_ref=head_ref)

    def delete_bare(self, *, user_id: str, project_id: str) -> bool:
        """Remove a single project's bare repo. Idempotent."""
        repo_dir = self.bare_repo_path(user_id, project_id)
        if not repo_dir.exists():
            return False
        shutil.rmtree(repo_dir, ignore_errors=True)
        # Best-effort cleanup of an empty user dir.
        user_dir = self._user_dir(user_id)
        try:
            if user_dir.is_dir() and not any(user_dir.iterdir()):
                user_dir.rmdir()
        except OSError:
            pass
        return True

    def delete_user_dir(self, user_id: str) -> bool:
        """Remove the entire ``{root}/{user_id}/`` tree.

        Wired into ``delete_user()``. The DB cascade handles row removal
        but the on-disk bare repos live outside any table, so a
        dedicated rmtree is mandatory — flagged High-likelihood in the
        spec's risk register.
        """
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return False
        shutil.rmtree(user_dir, ignore_errors=True)
        return True

    def dir_size_bytes(self, user_id: str, project_id: str) -> int:
        repo_dir = self.bare_repo_path(user_id, project_id)
        if not repo_dir.is_dir():
            return 0
        total = 0
        for entry in repo_dir.rglob("*"):
            try:
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:
                continue
        return total


# ── DB row helpers ─────────────────────────────────────────────────────


def _row_to_dict(cursor: aiosqlite.Cursor, row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    cols = [c[0] for c in cursor.description]
    return dict(zip(cols, row, strict=False))


# ── Store ──────────────────────────────────────────────────────────────


class ProjectStore:
    """CRUD over ``projects`` + ``project_repos`` + ``project_refs``.

    Every read is user_id-scoped; writes are too, except for the
    cascaded row removals delete_user() handles by table-scan. The
    storage layer is injected so tests can point at a tmp dir.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        storage: ProjectRepoStorage,
    ) -> None:
        self._conn = conn
        self._storage = storage

    @property
    def storage(self) -> ProjectRepoStorage:
        return self._storage

    # ── Reads ──────────────────────────────────────────────────────────

    async def get(
        self, project_id: str, *, user_id: str,
    ) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            "SELECT * FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def get_by_slug(
        self, *, user_id: str, slug: str,
    ) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            "SELECT * FROM projects WHERE user_id = ? AND slug = ?",
            (user_id, slug),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_for_user(
        self,
        *,
        user_id: str,
        kind: ProjectKind | None = None,
        include_archived: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM projects WHERE user_id = ?"
        params: list[Any] = [user_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        if not include_archived:
            sql += " AND archived_at IS NULL"
        sql += " ORDER BY last_activity_at DESC LIMIT ?"
        params.append(int(limit))
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    async def get_repo(
        self, project_id: str,
    ) -> dict[str, Any] | None:
        cursor = await self._conn.execute(
            "SELECT * FROM project_repos WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        return _row_to_dict(cursor, row) if row else None

    async def list_refs(
        self,
        project_id: str,
        *,
        kind: ProjectRefKind | None = None,
    ) -> list[dict[str, Any]]:
        if kind is not None:
            cursor = await self._conn.execute(
                "SELECT * FROM project_refs WHERE project_id = ? AND kind = ? "
                "ORDER BY created_at DESC",
                (project_id, kind),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM project_refs WHERE project_id = ? "
                "ORDER BY created_at DESC",
                (project_id,),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]

    # ── Writes ─────────────────────────────────────────────────────────

    async def create(
        self,
        *,
        user_id: str,
        name: str,
        kind: ProjectKind,
        origin: str = "manual",
        slug: str | None = None,
        description: str = "",
        default_branch: str = "main",
    ) -> dict[str, Any]:
        """Insert a row. If ``slug`` is None, auto-derive from ``name``
        and suffix on collision (``foo``, ``foo-2``, ``foo-3``, ...).
        If ``slug`` is provided and collides, raise :class:`SlugCollision`.
        """
        if not user_id:
            raise ValueError("create requires non-empty user_id")
        if not name.strip():
            raise ValueError("create requires non-empty name")

        explicit_slug = slug is not None
        base_slug = _slugify(slug if explicit_slug else name)
        final_slug = await self._resolve_slug(
            user_id=user_id, base_slug=base_slug, allow_suffix=not explicit_slug,
        )

        now = _now()
        project_id = _new_project_id()
        await self._conn.execute(
            """INSERT INTO projects
               (id, user_id, slug, name, description, kind, origin,
                default_branch, created_at, updated_at, last_activity_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                user_id,
                final_slug,
                name.strip(),
                description,
                kind,
                origin,
                default_branch,
                now,
                now,
                now,
            ),
        )
        await self._conn.commit()
        log.info(
            "project_created",
            project_id=project_id,
            user_id=user_id,
            slug=final_slug,
            kind=kind,
            origin=origin,
        )
        row = await self.get(project_id, user_id=user_id)
        assert row is not None  # we just inserted it
        return row

    async def _resolve_slug(
        self,
        *,
        user_id: str,
        base_slug: str,
        allow_suffix: bool,
    ) -> str:
        existing = await self.get_by_slug(user_id=user_id, slug=base_slug)
        if existing is None:
            return base_slug
        if not allow_suffix:
            raise SlugCollision(base_slug, existing["id"])
        # Find the next free suffix. Bounded retry — astronomically
        # unlikely to need more than a handful for one user.
        for n in range(2, 1000):
            candidate = f"{base_slug}-{n}"[:_SLUG_MAX]
            if await self.get_by_slug(user_id=user_id, slug=candidate) is None:
                return candidate
        raise SlugCollision(base_slug, existing["id"])

    async def update_activity(
        self, project_id: str, *, user_id: str,
    ) -> bool:
        now = _now()
        cursor = await self._conn.execute(
            "UPDATE projects SET last_activity_at = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (now, now, project_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def archive(self, project_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "UPDATE projects SET archived_at = ? "
            "WHERE id = ? AND user_id = ? AND archived_at IS NULL",
            (_now(), project_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def unarchive(self, project_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "UPDATE projects SET archived_at = NULL "
            "WHERE id = ? AND user_id = ? AND archived_at IS NOT NULL",
            (project_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def delete(self, project_id: str, *, user_id: str) -> bool:
        """Delete a project + its bare repo. Idempotent.

        DB cascades take care of ``project_repos`` and ``project_refs``
        rows; we explicitly rmtree the on-disk bare repo since it isn't
        in any table.
        """
        # Snapshot before the row goes away so we know where the repo lives.
        row = await self.get(project_id, user_id=user_id)
        if row is None:
            return False
        cursor = await self._conn.execute(
            "DELETE FROM projects WHERE id = ? AND user_id = ?",
            (project_id, user_id),
        )
        await self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            # rmtree may block briefly; offload so the event loop stays free.
            await asyncio.to_thread(
                self._storage.delete_bare,
                user_id=user_id,
                project_id=project_id,
            )
            log.info(
                "project_deleted",
                project_id=project_id,
                user_id=user_id,
            )
        return deleted

    # ── Bare-repo lifecycle ────────────────────────────────────────────

    async def ensure_bare_repo(
        self, project_id: str, *, user_id: str,
    ) -> dict[str, Any]:
        """Lazily create the bare repo + ``project_repos`` row. Idempotent.

        Returns the ``project_repos`` row as a dict. Safe to call from
        any path that needs a repo path on disk (e.g. PR-1.2's
        ``containers.create_workspace``).
        """
        project = await self.get(project_id, user_id=user_id)
        if project is None:
            raise LookupError(f"project not found: {project_id}")

        existing = await self.get_repo(project_id)
        if existing is not None and Path(existing["repo_path"]).exists():
            return existing

        handle = await asyncio.to_thread(
            self._storage.init_bare,
            user_id=user_id,
            project_id=project_id,
            default_branch=project.get("default_branch") or "main",
        )
        now = _now()
        if existing is None:
            await self._conn.execute(
                """INSERT INTO project_repos
                   (project_id, repo_path, head_ref, sha_count, size_bytes,
                    created_at, updated_at, user_id)
                   VALUES (?, ?, ?, 0, 0, ?, ?, ?)""",
                (project_id, handle.repo_path, handle.head_ref, now, now,
                 user_id),
            )
        else:
            # Row exists but the on-disk dir was missing (data-dir
            # restore from backup, etc.). Update the path in case it
            # changed and bump updated_at.
            await self._conn.execute(
                "UPDATE project_repos SET repo_path = ?, head_ref = ?, "
                "updated_at = ? WHERE project_id = ?",
                (handle.repo_path, handle.head_ref, now, project_id),
            )
        await self._conn.commit()
        repo_row = await self.get_repo(project_id)
        assert repo_row is not None
        return repo_row

    # ── Refs ───────────────────────────────────────────────────────────

    async def record_ref(
        self,
        *,
        project_id: str,
        kind: ProjectRefKind,
        ref_name: str,
        sha: str,
        label: str = "",
        message_id: str | None = None,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Insert a ``project_refs`` row. On (project_id, ref_name)
        conflict (unique index from migration 199), update the sha +
        label in place — refs are mutable pointers in git's model, so
        last-write-wins on a name is correct.
        """
        ref_id = _new_ref_id()
        now = _now()
        await self._conn.execute(
            """INSERT INTO project_refs
               (id, project_id, kind, ref_name, sha, label,
                created_at, created_by_message_id, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id, ref_name) DO UPDATE SET
                   sha = excluded.sha,
                   label = excluded.label""",
            (
                ref_id,
                project_id,
                kind,
                ref_name,
                sha,
                label,
                now,
                message_id,
                user_id,
            ),
        )
        await self._conn.commit()
        cursor = await self._conn.execute(
            "SELECT * FROM project_refs WHERE project_id = ? AND ref_name = ?",
            (project_id, ref_name),
        )
        row = await cursor.fetchone()
        assert row is not None
        return _row_to_dict(cursor, row)

    async def delete_ref(
        self, ref_id: str, *, project_id: str,
    ) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM project_refs WHERE id = ? AND project_id = ?",
            (ref_id, project_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0


__all__ = [
    "BareRepoHandle",
    "ProjectKind",
    "ProjectRefKind",
    "ProjectRepoStorage",
    "ProjectStore",
    "SlugCollision",
]
