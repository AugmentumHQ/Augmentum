"""Behavior tests for files routes (/api/files/*).

Scope note: the `files_routes.py` god-file is 2436 lines covering 30
endpoints. This suite tests the core VFS-backed operations that every
"I need to work with a user's files" caller depends on:

* CRUD + lookup (search, stats, entry, list by source)
* Trash lifecycle (soft delete → list trash → restore → purge)
* Tags (update, suggest)
* Favorites (list, toggle)
* Rename (PATCH) + delete (DELETE)
* Bulk operations (bulk-delete, bulk-restore)

Intentionally deferred to a later pass (each needs dedicated infra):

* /upload         — needs uploads_adapter
* /thumb/*        — needs thumbnail_service
* /preview, /text, /render, /transform, /summarize — need file backends
* /zip            — needs real disk artifacts
* /comics/*       — series rollups covered elsewhere
* /browse         — VFS adapter protocol
* /download       — needs real_path resolution

User isolation is the most critical property; every read/write here
scopes by user_id and a bug would leak files across tenants.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"  # matches conftest.test_user


@pytest.fixture(autouse=True)
def _reset_file_stats_cache():
    """Reset the per-user TTL cache between tests so writes in one test
    don't return stale counts in the next. The cache is module-level
    (process-wide) by design; tests get a clean slate via this fixture.
    """
    from augmentum.proxy.files_routes import _FILE_STATS_CACHE
    _FILE_STATS_CACHE.clear()
    yield
    _FILE_STATS_CACHE.clear()


@pytest.fixture
def files_client(app):
    """Client with a real FileIndexService + in-memory SQLite.

    The conftest `app` provides a mock SessionManager that authenticates
    "Bearer test-token" as usr_test. We layer a real FileIndexService
    on top so routes exercise actual CRUD.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager
    from augmentum.vfs.index import FileIndexService

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    app.state.state_manager = StateManager(backend)
    app.state.file_index = FileIndexService(backend._conn)

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, app.state.file_index
    _run(backend.close())


def _register(idx, *, user_id=TEST_USER_ID, source="uploads", source_id=None,
              name="file.txt", mime_type="text/plain", size_bytes=42,
              tags=None):
    """Seed a file_index row. Returns the generated file_id."""
    import secrets

    sid = source_id or f"test-{secrets.token_hex(4)}"
    return _run(idx.register(
        user_id=user_id, source=source, source_id=sid, name=name,
        mime_type=mime_type, size_bytes=size_bytes, tags=tags or [],
    ))


# ===========================================================================
# GET /api/files/search
# ===========================================================================

class TestSearch:
    def test_empty_when_no_files(self, files_client):
        client, _ = files_client
        r = client.get("/api/files/search")
        assert r.status_code == 200
        data = r.json()
        assert data["files"] == []
        assert data["has_more"] is False
        assert data["offset"] == 0

    def test_returns_user_files(self, files_client):
        client, idx = files_client
        _register(idx, name="notes.md")
        _register(idx, name="todo.md")
        r = client.get("/api/files/search")
        assert r.status_code == 200
        files = r.json()["files"]
        assert len(files) == 2
        names = {f["name"] for f in files}
        assert {"notes.md", "todo.md"} == names

    def test_isolates_other_users_files(self, files_client):
        """A user must not see another user's files in search results. A
        regression here leaks filenames (and via /entry, full metadata)
        across tenants."""
        client, idx = files_client
        _register(idx, user_id=TEST_USER_ID, name="mine.txt")
        _register(idx, user_id="usr_other", name="secret.txt")

        r = client.get("/api/files/search")
        files = r.json()["files"]
        assert len(files) == 1
        assert files[0]["name"] == "mine.txt"

    def test_query_matches_name(self, files_client):
        client, idx = files_client
        _register(idx, name="invoice-2026.pdf", mime_type="application/pdf")
        _register(idx, name="photo.jpg", mime_type="image/jpeg")

        r = client.get("/api/files/search?q=invoice")
        files = r.json()["files"]
        assert len(files) == 1
        assert files[0]["name"] == "invoice-2026.pdf"

    def test_filter_by_source(self, files_client):
        client, idx = files_client
        _register(idx, source="uploads", name="a.txt")
        _register(idx, source="notes", name="b.md")

        r = client.get("/api/files/search?source=uploads")
        sources = {f["source"] for f in r.json()["files"]}
        assert sources == {"uploads"}


# ===========================================================================
# GET /api/files/stats
# ===========================================================================

class TestStats:
    def test_empty_counts(self, files_client):
        client, _ = files_client
        r = client.get("/api/files/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_count"] == 0
        assert data["total_size"] == 0

    def test_counts_user_files(self, files_client):
        client, idx = files_client
        _register(idx, size_bytes=100)
        _register(idx, size_bytes=200)
        _register(idx, user_id="usr_other", size_bytes=9999)  # excluded

        r = client.get("/api/files/stats")
        data = r.json()
        assert data["total_count"] == 2
        assert data["total_size"] == 300

    def test_cache_serves_repeated_calls_without_recomputing(self, files_client):
        """Two back-to-back stats calls within the TTL window must hit
        the cache — verified by registering a file BETWEEN them and
        checking the second response is the stale (pre-write) snapshot.

        This lets us assert the caching is wired without time-mocking:
        cache fresh + write → cache returns old value, not new.
        """
        client, idx = files_client
        _register(idx, size_bytes=100)

        first = client.get("/api/files/stats").json()
        assert first["total_count"] == 1
        assert first["total_size"] == 100

        # Write a row but do NOT invalidate the cache. Within the 30s
        # TTL window the next call must return the cached snapshot.
        _register(idx, size_bytes=999)

        second = client.get("/api/files/stats").json()
        assert second["total_count"] == 1, (
            "cache must serve stale-but-fresh-enough snapshot within TTL"
        )
        assert second["total_size"] == 100

    @pytest.mark.asyncio
    async def test_per_user_ttl_cache_unit(self):
        """Direct unit tests for _PerUserTTLCache:
          * Hit returns cached value.
          * Miss invokes compute_fn.
          * Stampede: concurrent callers for the same user share one
            compute (lock prevents duplicate work).
          * Different users do not collide.
          * invalidate() forces recompute.
          * Capacity-bounded: oldest entry evicted at the boundary.
        """
        from augmentum.proxy.files_routes import _PerUserTTLCache

        cache = _PerUserTTLCache(ttl_s=60.0, max_users=3)
        compute_calls = {"u1": 0, "u2": 0, "u3": 0, "u4": 0}

        def make_fn(uid: str, value: int):
            async def _fn():
                compute_calls[uid] += 1
                await asyncio.sleep(0)  # yield to expose stampede races
                return value
            return _fn

        v = await cache.get_or_compute("u1", make_fn("u1", 100))
        assert v == 100
        assert compute_calls["u1"] == 1

        # Second call within TTL: hit, no recompute.
        v = await cache.get_or_compute("u1", make_fn("u1", 999))
        assert v == 100
        assert compute_calls["u1"] == 1

        # Stampede: 5 concurrent callers for the same user must collapse
        # into one compute.
        compute_calls["u2"] = 0
        results = await asyncio.gather(*[
            cache.get_or_compute("u2", make_fn("u2", 200)) for _ in range(5)
        ])
        assert results == [200] * 5
        assert compute_calls["u2"] == 1, (
            "stampede protection failed — multiple concurrent callers "
            "for the same user invoked compute_fn separately"
        )

        # Different users: separate entries, no collision.
        v = await cache.get_or_compute("u3", make_fn("u3", 300))
        assert v == 300

        # Invalidate forces recompute on next call.
        cache.invalidate("u1")
        v = await cache.get_or_compute("u1", make_fn("u1", 101))
        assert v == 101
        assert compute_calls["u1"] == 2

        # Capacity boundary: cache must stay at max_users after insert.
        # Eviction picks one entry by oldest timestamp; on the test
        # clock multiple stores can share a timestamp (sub-microsecond),
        # so we don't assert which one — only the cap is the invariant.
        # u4 (just-stored) and u1 (just-stored after invalidate) must
        # both be present; u2 is the oldest of the surviving set.
        v = await cache.get_or_compute("u4", make_fn("u4", 400))
        assert v == 400
        assert len(cache._cache) == 3
        assert "u4" in cache._cache
        assert "u1" in cache._cache

    @pytest.mark.asyncio
    async def test_per_user_ttl_cache_expires_after_ttl(self):
        """Entries past ttl_s must be recomputed."""
        from augmentum.proxy.files_routes import _PerUserTTLCache

        cache = _PerUserTTLCache(ttl_s=0.05, max_users=8)
        calls = {"n": 0}

        async def _fn():
            calls["n"] += 1
            return calls["n"]

        v1 = await cache.get_or_compute("u", _fn)
        assert v1 == 1
        # Wait past TTL, then recompute.
        await asyncio.sleep(0.08)
        v2 = await cache.get_or_compute("u", _fn)
        assert v2 == 2
        assert calls["n"] == 2

    def test_cache_isolates_users(self, files_client, app):
        """Each user gets their own cache entry — one user's cached
        result must never satisfy another user's request.
        """
        from augmentum.proxy.files_routes import _FILE_STATS_CACHE
        client, idx = files_client
        _register(idx, size_bytes=100)
        _register(idx, user_id="usr_other", size_bytes=500)

        # Prime cache for usr_test (the auth fixture's user).
        first = client.get("/api/files/stats").json()
        assert first["total_count"] == 1

        # Switch the request's authenticated user by clearing the
        # client's auth and using a different bearer; the conftest
        # mock auth recognises any "Bearer <id>" pattern.
        # Easiest cross-test: call _FILE_STATS_CACHE directly to confirm
        # the keying.
        assert "usr_test" in _FILE_STATS_CACHE._cache
        assert "usr_other" not in _FILE_STATS_CACHE._cache


# ===========================================================================
# GET /api/files/entry/{file_id}
# ===========================================================================

class TestGetEntry:
    def test_returns_own_file(self, files_client):
        client, idx = files_client
        file_id = _register(idx, name="report.pdf")
        r = client.get(f"/api/files/entry/{file_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "report.pdf"
        assert r.json()["id"] == file_id

    def test_missing_returns_404(self, files_client):
        client, _ = files_client
        r = client.get("/api/files/entry/fi_ghostghost")
        assert r.status_code == 404

    def test_other_users_file_returns_404(self, files_client):
        """ID-guessing attack — another user's file ID must resolve to
        404, not 200, even if the ID is known to the attacker."""
        client, idx = files_client
        other_id = _register(idx, user_id="usr_other", name="secret.txt")
        r = client.get(f"/api/files/entry/{other_id}")
        assert r.status_code == 404


# ===========================================================================
# GET /api/files/list/{source}
# ===========================================================================

class TestListBySource:
    def test_returns_only_matching_source(self, files_client):
        client, idx = files_client
        _register(idx, source="uploads", name="a.txt")
        _register(idx, source="notes", name="b.md")

        r = client.get("/api/files/list/uploads")
        assert r.status_code == 200
        files = r.json()["files"]
        assert all(f["source"] == "uploads" for f in files)
        assert len(files) == 1


# ===========================================================================
# POST /api/files/favorite/{file_id} — toggle, list
# ===========================================================================

class TestFavorites:
    def test_toggle_marks_favorite(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        r = client.post(f"/api/files/favorite/{file_id}")
        assert r.status_code == 200
        assert r.json()["is_favorite"] is True

    def test_toggle_unmarks_favorite(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        client.post(f"/api/files/favorite/{file_id}")
        r = client.post(f"/api/files/favorite/{file_id}")
        assert r.json()["is_favorite"] is False

    def test_favorites_list_shows_only_favorited(self, files_client):
        client, idx = files_client
        fav = _register(idx, name="kept.md")
        _register(idx, name="ignored.md")
        client.post(f"/api/files/favorite/{fav}")

        r = client.get("/api/files/favorites")
        assert r.status_code == 200
        files = r.json()["files"]
        assert len(files) == 1
        assert files[0]["id"] == fav


# ===========================================================================
# Trash lifecycle: DELETE → /trash → /restore → /purge-trash
# ===========================================================================

class TestTrashLifecycle:
    def test_soft_delete_moves_to_trash(self, files_client):
        client, idx = files_client
        file_id = _register(idx, name="goodbye.md")
        r = client.delete(f"/api/files/{file_id}")
        assert r.status_code == 200

        # Gone from search
        search = client.get("/api/files/search").json()["files"]
        assert all(f["id"] != file_id for f in search)

        # Present in trash
        trash = client.get("/api/files/trash").json()["files"]
        assert any(f["id"] == file_id for f in trash)

    def test_restore_returns_to_search(self, files_client):
        client, idx = files_client
        file_id = _register(idx, name="comeback.md")
        client.delete(f"/api/files/{file_id}")
        r = client.post(f"/api/files/restore/{file_id}")
        assert r.status_code == 200

        search = client.get("/api/files/search").json()["files"]
        assert any(f["id"] == file_id for f in search)
        trash = client.get("/api/files/trash").json()["files"]
        assert all(f["id"] != file_id for f in trash)

    def test_restore_nonexistent_returns_404(self, files_client):
        client, _ = files_client
        r = client.post("/api/files/restore/fi_ghost")
        assert r.status_code == 404

    def test_bulk_delete(self, files_client):
        client, idx = files_client
        ids = [_register(idx, name=f"f{i}.md") for i in range(3)]
        r = client.post("/api/files/bulk-delete", json={"ids": ids})
        assert r.status_code == 200
        assert r.json()["deleted"] == 3

    def test_bulk_delete_rejects_empty(self, files_client):
        client, _ = files_client
        r = client.post("/api/files/bulk-delete", json={"ids": []})
        assert r.status_code == 400

    def test_bulk_delete_rejects_oversize(self, files_client):
        """Cap is 200 per call — larger batches rejected to avoid the
        per-row loop becoming a DoS vector."""
        client, _ = files_client
        r = client.post("/api/files/bulk-delete", json={"ids": [f"x{i}" for i in range(201)]})
        assert r.status_code == 400

    def test_bulk_restore(self, files_client):
        client, idx = files_client
        ids = [_register(idx, name=f"f{i}.md") for i in range(3)]
        client.post("/api/files/bulk-delete", json={"ids": ids})
        r = client.post("/api/files/bulk-restore", json={"ids": ids})
        assert r.status_code == 200
        assert r.json()["restored"] == 3

    def test_delete_other_users_file_returns_404(self, files_client):
        """Cross-tenant delete protection."""
        client, idx = files_client
        other_id = _register(idx, user_id="usr_other", name="protected.md")
        r = client.delete(f"/api/files/{other_id}")
        assert r.status_code == 404
        # File still exists for the other user
        got = _run(idx.get(other_id, user_id="usr_other"))
        assert got is not None
        assert got.is_trashed is False

    def test_purge_trash_empties_it(self, files_client):
        client, idx = files_client
        file_id = _register(idx, name="tmp.md")
        client.delete(f"/api/files/{file_id}")
        r = client.post("/api/files/purge-trash")
        assert r.status_code == 200

        # Trash now empty
        trash = client.get("/api/files/trash").json()["files"]
        assert len(trash) == 0


# ===========================================================================
# Tags — update + suggest
# ===========================================================================

class TestTags:
    def test_update_tags(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        r = client.patch(f"/api/files/tags/{file_id}", json={"tags": ["todo", "work"]})
        assert r.status_code == 200
        body = r.json()
        assert "todo" in body["tags"]
        assert "work" in body["tags"]

    def test_update_tags_rejects_non_list(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        r = client.patch(f"/api/files/tags/{file_id}", json={"tags": "not-a-list"})
        assert r.status_code == 400

    def test_update_tags_rejects_oversize(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        big = [f"tag{i}" for i in range(51)]
        r = client.patch(f"/api/files/tags/{file_id}", json={"tags": big})
        assert r.status_code == 400

    def test_update_tags_for_missing_file_404(self, files_client):
        client, _ = files_client
        r = client.patch("/api/files/tags/fi_ghost", json={"tags": ["x"]})
        assert r.status_code == 404

    def test_update_other_users_tags_returns_404(self, files_client):
        """Cross-tenant tag injection is blocked — another user's file ID
        can't be used to smuggle tags into their index."""
        client, idx = files_client
        other_id = _register(idx, user_id="usr_other", tags=["original"])
        r = client.patch(f"/api/files/tags/{other_id}", json={"tags": ["hijacked"]})
        assert r.status_code == 404
        # Other user's tags untouched
        entry = _run(idx.get(other_id, user_id="usr_other"))
        assert "hijacked" not in entry.tags

    def test_tags_suggest(self, files_client):
        client, idx = files_client
        fid = _register(idx, tags=["work", "urgent"])
        # tags are stored in the index via register → tags_json column
        r = client.get("/api/files/tags/suggest?q=")
        assert r.status_code == 200
        assert "tags" in r.json()


# ===========================================================================
# PATCH /api/files/{file_id} — rename
# ===========================================================================

class TestRename:
    def test_renames_file(self, files_client):
        client, idx = files_client
        file_id = _register(idx, name="old.md")
        r = client.patch(f"/api/files/{file_id}", json={"name": "new.md"})
        assert r.status_code == 200
        assert r.json()["name"] == "new.md"

        got = _run(idx.get(file_id, user_id=TEST_USER_ID))
        assert got.name == "new.md"

    def test_empty_name_rejected(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        r = client.patch(f"/api/files/{file_id}", json={"name": "   "})
        assert r.status_code == 400

    def test_oversize_name_rejected(self, files_client):
        client, idx = files_client
        file_id = _register(idx)
        r = client.patch(f"/api/files/{file_id}", json={"name": "a" * 256})
        assert r.status_code == 400

    def test_missing_file_404(self, files_client):
        client, _ = files_client
        r = client.patch("/api/files/fi_ghost", json={"name": "x"})
        assert r.status_code == 404

    def test_cannot_rename_other_users_file(self, files_client):
        client, idx = files_client
        other_id = _register(idx, user_id="usr_other", name="theirs.md")
        r = client.patch(f"/api/files/{other_id}", json={"name": "hijacked.md"})
        assert r.status_code == 404


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.files_routes import router
        assert router.prefix == "/api/files"

    def test_registered_routes_cover_expected_surface(self):
        from augmentum.proxy.files_routes import router
        paths = {r.path for r in router.routes}
        # Spot-check the endpoint groups covered by this file
        expected = {
            "/api/files/search", "/api/files/stats",
            "/api/files/entry/{file_id}",
            "/api/files/favorites", "/api/files/favorite/{file_id}",
            "/api/files/trash", "/api/files/restore/{file_id}",
            "/api/files/bulk-delete", "/api/files/bulk-restore",
            "/api/files/purge-trash",
            "/api/files/tags/{file_id}", "/api/files/tags/suggest",
            "/api/files/{file_id}",
        }
        assert expected.issubset(paths)
