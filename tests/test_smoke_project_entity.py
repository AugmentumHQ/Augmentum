"""Phase 1 smoke tests for the Project entity lifecycle.

Covers the substrate added by the Integrated Coding Nervous System
spec — migration 199-201, ProjectStore CRUD, ProjectRepoStorage
on-disk lifecycle, and the migration-200 backfill that links
existing coder_workspaces rows to fresh Project rows. These tests
are fast (in-memory SQLite + tmp dir) and live in the offline
default subset.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def backend(tmp_path):
    """Real SQLiteBackend, in-memory DB, all migrations applied."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    yield backend
    _run(backend.close())


@pytest.fixture
def repo_root(tmp_path):
    """A fresh tmp dir for ProjectRepoStorage to root its bare repos."""
    root = tmp_path / "projects"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# ProjectRepoStorage — host-side bare-repo lifecycle (no DB)
# ---------------------------------------------------------------------------


class TestProjectRepoStorage:
    def test_init_bare_creates_repo(self, repo_root):
        from augmentum.projects import ProjectRepoStorage
        storage = ProjectRepoStorage(repo_root)
        handle = storage.init_bare(user_id="u1", project_id="prj_aaa111")
        assert (Path(handle.repo_path) / "HEAD").exists()
        # HEAD points at the default branch
        head_text = (Path(handle.repo_path) / "HEAD").read_text()
        assert "refs/heads/main" in head_text

    def test_init_bare_is_idempotent(self, repo_root):
        from augmentum.projects import ProjectRepoStorage
        storage = ProjectRepoStorage(repo_root)
        h1 = storage.init_bare(user_id="u1", project_id="prj_bbb222")
        h2 = storage.init_bare(user_id="u1", project_id="prj_bbb222")
        assert h1.repo_path == h2.repo_path

    def test_delete_bare_cleans_up_empty_user_dir(self, repo_root):
        from augmentum.projects import ProjectRepoStorage
        storage = ProjectRepoStorage(repo_root)
        storage.init_bare(user_id="u1", project_id="prj_ccc333")
        user_dir = repo_root / "u1"
        assert user_dir.exists()
        assert storage.delete_bare(user_id="u1", project_id="prj_ccc333")
        # Best-effort: empty user dir gets removed too
        assert not user_dir.exists()

    def test_delete_user_dir_wipes_all_projects(self, repo_root):
        from augmentum.projects import ProjectRepoStorage
        storage = ProjectRepoStorage(repo_root)
        storage.init_bare(user_id="u2", project_id="prj_a")
        storage.init_bare(user_id="u2", project_id="prj_b")
        assert storage.delete_user_dir("u2")
        assert not (repo_root / "u2").exists()


# ---------------------------------------------------------------------------
# Migrations 199-201 — apply cleanly to a fresh DB
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_phase1_tables_exist_after_connect(self, backend):
        async def _check() -> set[str]:
            async with backend.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('projects', 'project_repos', 'project_refs', "
                "             'project_checkouts')"
            ) as cur:
                return {row[0] for row in await cur.fetchall()}
        tables = _run(_check())
        assert tables == {
            "projects", "project_repos", "project_refs", "project_checkouts",
        }

    def test_coder_workspaces_was_renamed(self, backend):
        """Migration 200 renamed the table. The old name should not
        survive — anything else is a half-applied migration footgun."""
        async def _check() -> bool:
            async with backend.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='coder_workspaces'"
            ) as cur:
                return (await cur.fetchone()) is not None
        assert not _run(_check()), (
            "coder_workspaces must be renamed to project_checkouts (migration 200)"
        )

    def test_coder_turn_runs_column_renamed(self, backend):
        """Migration 200 renamed workspace_id -> project_id."""
        async def _columns() -> set[str]:
            async with backend.conn.execute(
                "PRAGMA table_info(coder_turn_runs)"
            ) as cur:
                return {row[1] for row in await cur.fetchall()}
        cols = _run(_columns())
        assert "project_id" in cols
        assert "workspace_id" not in cols

    def test_library_publications_has_project_ref_id(self, backend):
        async def _columns() -> set[str]:
            async with backend.conn.execute(
                "PRAGMA table_info(library_publications)"
            ) as cur:
                return {row[1] for row in await cur.fetchall()}
        assert "project_ref_id" in _run(_columns())


# ---------------------------------------------------------------------------
# ProjectStore — DB CRUD + bare-repo coordination
# ---------------------------------------------------------------------------


class TestProjectStore:
    def _seed_user(self, conn, user_id: str = "u1") -> None:
        async def _do():
            await conn.execute(
                "INSERT INTO users (id, username, display_name, password_hash, role) "
                "VALUES (?, ?, ?, ?, 'user')",
                (user_id, user_id, user_id, "pw"),
            )
            await conn.commit()
        _run(_do())

    def _make_store(self, backend, repo_root):
        from augmentum.projects import ProjectRepoStorage, ProjectStore
        storage = ProjectRepoStorage(repo_root)
        return ProjectStore(backend.conn, storage), storage

    def test_create_and_get(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        proj = _run(store.create(user_id="u1", name="My App", kind="coder"))
        assert proj["id"].startswith("prj_")
        assert proj["user_id"] == "u1"
        assert proj["slug"] == "my-app"
        assert proj["kind"] == "coder"
        # Round-trip via get()
        fetched = _run(store.get(proj["id"], user_id="u1"))
        assert fetched is not None
        assert fetched["id"] == proj["id"]

    def test_create_auto_suffixes_slug_on_collision(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        a = _run(store.create(user_id="u1", name="dup", kind="coder"))
        b = _run(store.create(user_id="u1", name="dup", kind="coder"))
        c = _run(store.create(user_id="u1", name="dup", kind="coder"))
        slugs = {a["slug"], b["slug"], c["slug"]}
        assert slugs == {"dup", "dup-2", "dup-3"}

    def test_explicit_slug_collision_raises(self, backend, repo_root):
        from augmentum.projects import SlugCollision
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        _run(store.create(
            user_id="u1", name="A", kind="coder", slug="taken",
        ))
        with pytest.raises(SlugCollision):
            _run(store.create(
                user_id="u1", name="B", kind="coder", slug="taken",
            ))

    def test_list_for_user_orders_by_activity(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        a = _run(store.create(user_id="u1", name="A", kind="coder"))
        b = _run(store.create(user_id="u1", name="B", kind="coder"))
        # Touch B last so it sorts first
        _run(store.update_activity(b["id"], user_id="u1"))
        rows = _run(store.list_for_user(user_id="u1"))
        assert [r["id"] for r in rows[:2]] == [b["id"], a["id"]]

    def test_list_for_user_is_isolated_by_tenant(self, backend, repo_root):
        self._seed_user(backend.conn, "alice")
        self._seed_user(backend.conn, "bob")
        store, _ = self._make_store(backend, repo_root)
        _run(store.create(user_id="alice", name="A", kind="coder"))
        _run(store.create(user_id="bob", name="B", kind="coder"))
        alice = _run(store.list_for_user(user_id="alice"))
        bob = _run(store.list_for_user(user_id="bob"))
        assert len(alice) == 1
        assert len(bob) == 1
        assert alice[0]["user_id"] == "alice"
        assert bob[0]["user_id"] == "bob"

    def test_archive_hides_from_default_list(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        a = _run(store.create(user_id="u1", name="A", kind="coder"))
        assert _run(store.archive(a["id"], user_id="u1"))
        live = _run(store.list_for_user(user_id="u1"))
        assert a["id"] not in {r["id"] for r in live}
        all_rows = _run(store.list_for_user(
            user_id="u1", include_archived=True,
        ))
        assert a["id"] in {r["id"] for r in all_rows}

    def test_ensure_bare_repo_creates_dir_and_row(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, storage = self._make_store(backend, repo_root)
        proj = _run(store.create(user_id="u1", name="App", kind="coder"))
        repo_row = _run(store.ensure_bare_repo(proj["id"], user_id="u1"))
        bare = Path(repo_row["repo_path"])
        assert bare.exists()
        assert (bare / "HEAD").exists()
        assert repo_row["head_ref"] == "refs/heads/main"

    def test_ensure_bare_repo_is_idempotent(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        proj = _run(store.create(user_id="u1", name="App", kind="coder"))
        r1 = _run(store.ensure_bare_repo(proj["id"], user_id="u1"))
        r2 = _run(store.ensure_bare_repo(proj["id"], user_id="u1"))
        assert r1["repo_path"] == r2["repo_path"]

    def test_delete_removes_row_and_on_disk_dir(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        proj = _run(store.create(user_id="u1", name="App", kind="coder"))
        repo_row = _run(store.ensure_bare_repo(proj["id"], user_id="u1"))
        bare = Path(repo_row["repo_path"])
        assert bare.exists()
        assert _run(store.delete(proj["id"], user_id="u1"))
        assert _run(store.get(proj["id"], user_id="u1")) is None
        assert not bare.exists()

    def test_record_ref_is_upsert(self, backend, repo_root):
        self._seed_user(backend.conn)
        store, _ = self._make_store(backend, repo_root)
        proj = _run(store.create(user_id="u1", name="App", kind="coder"))
        ref1 = _run(store.record_ref(
            project_id=proj["id"],
            kind="publication",
            ref_name="refs/published/v1",
            sha="aaaa",
            label="v1",
        ))
        ref2 = _run(store.record_ref(
            project_id=proj["id"],
            kind="publication",
            ref_name="refs/published/v1",
            sha="bbbb",
            label="v1.1",
        ))
        # Same row updated in place — unique index on (project_id, ref_name)
        assert ref2["sha"] == "bbbb"
        assert ref2["label"] == "v1.1"
        all_refs = _run(store.list_refs(proj["id"], kind="publication"))
        assert len(all_refs) == 1
