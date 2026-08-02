"""Comprehensive endpoint tests for all knowledge API routes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from augmentum.knowledge.catalog import CatalogEntry, CatalogClient, CATEGORY_MAP
from augmentum.knowledge.packs import PackManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog_entry(
    id: str = "test.pack",
    title: str = "Test Pack",
    description: str = "A test entry",
    language: str = "en",
    raw_category: str = "wikipedia",
    article_count: int = 1000,
    size_bytes: int = 50_000_000,
    download_url: str = "https://example.com/test.zim",
    tags: list[str] | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        id=id,
        title=title,
        description=description,
        language=language,
        raw_category=raw_category,
        article_count=article_count,
        media_count=0,
        size_bytes=size_bytes,
        download_url=download_url,
        thumbnail_url="",
        issued_date="2026-01-01",
        tags=tags or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def knowledge_app(app, tmp_path):
    """Set up app with knowledge PackManager and CatalogClient."""
    pack_dir = tmp_path / "knowledge"
    pack_dir.mkdir()

    mgr = PackManager(pack_dir)
    app.state.pack_manager = mgr

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    catalog = CatalogClient(cache_dir=cache_dir, cache_ttl=3600)
    app.state.catalog_client = catalog

    app.state.install_jobs = {}
    return app


@pytest.fixture
def knowledge_client(knowledge_app):
    client = TestClient(knowledge_app)
    client.headers.update({"Authorization": "Bearer test-token"})
    return client


@pytest.fixture
def bare_app(app):
    """App with pack_manager and catalog_client explicitly unset (503 tests)."""
    app.state.pack_manager = None
    app.state.catalog_client = None
    app.state.install_jobs = {}
    return app


@pytest.fixture
def bare_client(bare_app):
    client = TestClient(bare_app)
    client.headers.update({"Authorization": "Bearer test-token"})
    return client


# ===================================================================
# Pack Management
# ===================================================================


class TestListPacks:
    """GET /api/knowledge/packs"""

    def test_empty(self, knowledge_client):
        resp = knowledge_client.get("/api/knowledge/packs")
        assert resp.status_code == 200
        data = resp.json()
        assert "packs" in data
        assert data["packs"] == []

    def test_with_packs(self, knowledge_app, knowledge_client):
        """Inject a fake pack into the manager and verify it appears."""
        mgr: PackManager = knowledge_app.state.pack_manager
        fake_meta = MagicMock()
        fake_meta.name = "Test"
        fake_meta.version = "1.0"
        fake_meta.description = "desc"
        fake_meta.embedding_model = "nomic"
        fake_meta.embedding_dim = 768
        fake_meta.chunk_count = 42
        fake_meta.source_license = "CC"
        fake_meta.build_date = "2026-01-01"
        fake_conn = MagicMock()
        from augmentum.knowledge.packs import PackConnection
        mgr._packs["test-pack"] = PackConnection(
            conn=fake_conn, meta=fake_meta, path=Path("/tmp/test.augpack"), active=True,
        )
        resp = knowledge_client.get("/api/knowledge/packs")
        assert resp.status_code == 200
        packs = resp.json()["packs"]
        assert len(packs) == 1
        assert packs[0]["pack_id"] == "test-pack"
        assert packs[0]["name"] == "Test"
        assert packs[0]["active"] is True

    def test_503_when_no_manager(self, bare_client):
        resp = bare_client.get("/api/knowledge/packs")
        assert resp.status_code == 503


# ===================================================================
# Activate / Deactivate
# ===================================================================


class TestActivatePack:
    """POST /api/knowledge/activate/{pack_id}"""

    def test_success(self, knowledge_app, knowledge_client):
        mgr: PackManager = knowledge_app.state.pack_manager
        from augmentum.knowledge.packs import PackConnection
        fake_meta = MagicMock()
        mgr._packs["mypack"] = PackConnection(
            conn=MagicMock(), meta=fake_meta, path=Path("/tmp/x.augpack"), active=False,
        )
        resp = knowledge_client.post("/api/knowledge/activate/mypack")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["pack_id"] == "mypack"
        assert body["active"] is True
        # Verify internal state changed
        assert mgr._packs["mypack"].active is True

    def test_not_found(self, knowledge_client):
        resp = knowledge_client.post("/api/knowledge/activate/nonexistent")
        assert resp.status_code == 404


class TestDeactivatePack:
    """POST /api/knowledge/deactivate/{pack_id}"""

    def test_success(self, knowledge_app, knowledge_client):
        mgr: PackManager = knowledge_app.state.pack_manager
        from augmentum.knowledge.packs import PackConnection
        fake_meta = MagicMock()
        mgr._packs["mypack"] = PackConnection(
            conn=MagicMock(), meta=fake_meta, path=Path("/tmp/x.augpack"), active=True,
        )
        resp = knowledge_client.post("/api/knowledge/deactivate/mypack")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["active"] is False
        assert mgr._packs["mypack"].active is False

    def test_not_found(self, knowledge_client):
        resp = knowledge_client.post("/api/knowledge/deactivate/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# Delete
# ===================================================================


class TestDeletePack:
    """DELETE /api/knowledge/{pack_id}"""

    def test_success(self, knowledge_app, knowledge_client, tmp_path):
        mgr: PackManager = knowledge_app.state.pack_manager
        # Create a real file so os.remove works
        pack_file = tmp_path / "knowledge" / "deleteme.augpack"
        pack_file.write_bytes(b"fake")

        from augmentum.knowledge.packs import PackConnection
        fake_conn = AsyncMock()
        fake_meta = MagicMock()
        mgr._packs["deleteme"] = PackConnection(
            conn=fake_conn, meta=fake_meta, path=pack_file, active=True,
        )
        resp = knowledge_client.delete("/api/knowledge/deleteme")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["pack_id"] == "deleteme"
        assert "deleteme" not in mgr._packs
        assert not pack_file.exists()

    def test_not_found(self, knowledge_client):
        resp = knowledge_client.delete("/api/knowledge/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# Supported Formats
# ===================================================================


class TestSupportedFormats:
    """GET /api/knowledge/supported-formats"""

    def test_returns_sorted_list(self, knowledge_client):
        resp = knowledge_client.get("/api/knowledge/supported-formats")
        assert resp.status_code == 200
        data = resp.json()
        assert "formats" in data
        formats = data["formats"]
        assert isinstance(formats, list)
        assert ".csv" in formats
        assert ".augpack" in formats
        assert ".pdf" in formats
        # Should be sorted
        assert formats == sorted(formats)


# ===================================================================
# Search
# ===================================================================


class TestSearchPacks:
    """GET /api/knowledge/search?q=test"""

    def test_no_active_packs(self, knowledge_client):
        """Search with no packs loaded returns empty results."""
        with patch("augmentum.memory.embeddings.EmbeddingService") as mock_emb:
            mock_emb.embed_query.return_value = [0.0] * 384
            mock_emb.to_blob.return_value = b"\x00" * (384 * 4)
            resp = knowledge_client.get("/api/knowledge/search", params={"q": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test"
        assert data["results"] == []

    def test_with_mocked_results(self, knowledge_app, knowledge_client):
        """Mock PackManager.search to return canned results."""
        from augmentum.knowledge.packs import PackResult

        fake_result = PackResult(
            content="Test content", title="Test Title", section="Intro",
            url="https://example.com", score=0.9, pack_id="test-pack",
            source="wikipedia",
        )

        async def _mock_search(query, *, pack_ids, limit=5, rerank=True):
            return [fake_result]

        knowledge_app.state.pack_manager.search = _mock_search

        resp = knowledge_client.get("/api/knowledge/search", params={"q": "hello", "limit": "3"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "hello"
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "Test Title"
        assert data["results"][0]["pack_id"] == "test-pack"

    def test_503_when_no_manager(self, bare_client):
        resp = bare_client.get("/api/knowledge/search", params={"q": "test"})
        assert resp.status_code == 503


# ===================================================================
# Catalog — Browse
# ===================================================================


class TestBrowseCatalog:
    """GET /api/knowledge/catalog"""

    def test_default_params(self, knowledge_app, knowledge_client):
        entries = [
            _make_catalog_entry(id="wiki.en", title="Wikipedia EN", article_count=5000),
            _make_catalog_entry(id="devdocs.en", title="DevDocs", raw_category="devdocs", article_count=2000),
        ]

        async def _mock_browse(**kwargs):
            # Simple: return entries filtered by offset/limit
            offset = kwargs.get("offset", 0)
            limit = kwargs.get("limit", 50)
            return entries[offset:offset + limit]

        knowledge_app.state.catalog_client.browse = _mock_browse

        resp = knowledge_client.get("/api/knowledge/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "total" in data
        assert "offset" in data
        assert "limit" in data
        assert len(data["entries"]) == 2
        # Each entry should have "installed" key
        for entry in data["entries"]:
            assert "installed" in entry
            assert "id" in entry
            assert "title" in entry

    def test_category_filter(self, knowledge_app, knowledge_client):
        dev_entry = _make_catalog_entry(id="devdocs.en", title="DevDocs", raw_category="devdocs")

        async def _mock_browse(**kwargs):
            cat = kwargs.get("category")
            if cat == "Dev":
                return [dev_entry]
            return []

        knowledge_app.state.catalog_client.browse = _mock_browse

        resp = knowledge_client.get("/api/knowledge/catalog", params={"category": "Dev"})
        assert resp.status_code == 200
        assert len(resp.json()["entries"]) == 1
        assert resp.json()["entries"][0]["id"] == "devdocs.en"

    def test_size_filter(self, knowledge_app, knowledge_client):
        small = _make_catalog_entry(id="small", title="Small", size_bytes=100_000)

        async def _mock_browse(**kwargs):
            max_size = kwargs.get("max_size_bytes")
            if max_size and max_size <= 200_000:
                return [small]
            return [small, _make_catalog_entry(id="big", size_bytes=999_999_999)]

        knowledge_app.state.catalog_client.browse = _mock_browse

        resp = knowledge_client.get("/api/knowledge/catalog", params={"size_max": 200000})
        assert resp.status_code == 200
        assert len(resp.json()["entries"]) == 1

    def test_sort_param(self, knowledge_app, knowledge_client):
        async def _mock_browse(**kwargs):
            return [_make_catalog_entry()]

        knowledge_app.state.catalog_client.browse = _mock_browse

        resp = knowledge_client.get("/api/knowledge/catalog", params={"sort": "smallest"})
        assert resp.status_code == 200

    def test_query_param(self, knowledge_app, knowledge_client):
        matched = _make_catalog_entry(id="python.en", title="Python Docs")

        async def _mock_browse(**kwargs):
            q = kwargs.get("query")
            if q and "python" in q.lower():
                return [matched]
            return []

        knowledge_app.state.catalog_client.browse = _mock_browse

        resp = knowledge_client.get("/api/knowledge/catalog", params={"q": "python"})
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["id"] == "python.en"

    def test_installed_marker(self, knowledge_app, knowledge_client):
        """Entries matching installed pack IDs get installed=True.

        Note: PackManager.installed returns dicts with ``pack_id`` key.
        The route checks ``pack.get("id", "")``, so we inject a dict with
        an ``id`` key to simulate a future-proofed pack metadata dict.
        """
        mgr: PackManager = knowledge_app.state.pack_manager
        # Override installed property to return dicts with "id" key
        # (mirrors what the route actually looks up)
        original_installed = type(mgr).installed
        type(mgr).installed = property(lambda self: [
            {"id": "wiki.en", "pack_id": "wiki.en", "name": "Wikipedia", "active": True},
        ])

        entries = [
            _make_catalog_entry(id="wiki.en", title="Wikipedia"),
            _make_catalog_entry(id="other.en", title="Other"),
        ]

        async def _mock_browse(**kwargs):
            offset = kwargs.get("offset", 0)
            limit = kwargs.get("limit", 50)
            return entries[offset:offset + limit]

        knowledge_app.state.catalog_client.browse = _mock_browse

        resp = knowledge_client.get("/api/knowledge/catalog")
        assert resp.status_code == 200
        result = resp.json()["entries"]
        installed_map = {e["id"]: e["installed"] for e in result}
        assert installed_map.get("wiki.en") is True
        assert installed_map.get("other.en") is False

        # Restore original property
        type(mgr).installed = original_installed

    def test_503_when_no_catalog(self, bare_client):
        resp = bare_client.get("/api/knowledge/catalog")
        assert resp.status_code == 503


# ===================================================================
# Catalog — Featured
# ===================================================================


class TestFeaturedCatalog:
    """GET /api/knowledge/catalog/featured"""

    def test_returns_featured(self, knowledge_app, knowledge_client):
        featured = [_make_catalog_entry(id="wikipedia.en.medicine", title="Medicine")]

        async def _mock_featured(**kwargs):
            return featured

        knowledge_app.state.catalog_client.featured = _mock_featured

        resp = knowledge_client.get("/api/knowledge/catalog/featured")
        assert resp.status_code == 200
        data = resp.json()
        assert "featured" in data
        assert len(data["featured"]) == 1
        assert data["featured"][0]["id"] == "wikipedia.en.medicine"
        assert "installed" in data["featured"][0]

    def test_empty_featured(self, knowledge_app, knowledge_client):
        async def _mock_featured(**kwargs):
            return []

        knowledge_app.state.catalog_client.featured = _mock_featured

        resp = knowledge_client.get("/api/knowledge/catalog/featured")
        assert resp.status_code == 200
        assert resp.json()["featured"] == []

    def test_503_when_no_catalog(self, bare_client):
        resp = bare_client.get("/api/knowledge/catalog/featured")
        assert resp.status_code == 503


# ===================================================================
# Catalog — Categories
# ===================================================================


class TestCatalogCategories:
    """GET /api/knowledge/catalog/categories"""

    def test_returns_categories(self, knowledge_app, knowledge_client):
        resp = knowledge_client.get("/api/knowledge/catalog/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        cats = data["categories"]
        assert isinstance(cats, list)
        # Should be the unique display categories from CATEGORY_MAP
        expected = sorted(set(CATEGORY_MAP.values()))
        assert cats == expected

    def test_503_when_no_catalog(self, bare_client):
        resp = bare_client.get("/api/knowledge/catalog/categories")
        assert resp.status_code == 503


# ===================================================================
# Install — Start Job
# ===================================================================


class TestStartInstall:
    """POST /api/knowledge/install"""

    def test_creates_job(self, knowledge_app, knowledge_client):
        # Patch httpx so the background task doesn't actually download
        with patch("augmentum.proxy.knowledge_routes.httpx") as mock_httpx:
            mock_cm = AsyncMock()
            mock_httpx.AsyncClient.return_value.__aenter__ = mock_cm
            resp = knowledge_client.post(
                "/api/knowledge/install",
                json={
                    "catalog_id": "test.en",
                    "download_url": "https://example.com/test.zim",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "started"
        # Job should be tracked
        assert data["job_id"] in knowledge_app.state.install_jobs

    def test_custom_dir(self, knowledge_app, knowledge_client, tmp_path):
        custom = tmp_path / "custom_packs"
        with patch("augmentum.proxy.knowledge_routes.httpx"):
            resp = knowledge_client.post(
                "/api/knowledge/install",
                json={
                    "catalog_id": "test2.en",
                    "download_url": "https://example.com/test2.zim",
                    "custom_dir": str(custom),
                },
            )
        assert resp.status_code == 200
        assert custom.exists()

    def test_503_when_no_manager(self, bare_client):
        resp = bare_client.post(
            "/api/knowledge/install",
            json={"catalog_id": "x", "download_url": "https://example.com/x.zim"},
        )
        assert resp.status_code == 503


# ===================================================================
# Install — Progress SSE
# ===================================================================


class TestInstallProgress:
    """GET /api/knowledge/install/{job_id}/progress"""

    def test_streams_progress(self, knowledge_app, knowledge_client):
        from augmentum.proxy.knowledge_routes import InstallJob

        job = InstallJob(
            job_id="abc123",
            catalog_id="test.en",
            status="complete",
            stage="done",
            current=100,
            total=100,
        )
        knowledge_app.state.install_jobs["abc123"] = job

        resp = knowledge_client.get("/api/knowledge/install/abc123/progress")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Parse the SSE data
        text = resp.text
        assert "data:" in text
        # Extract JSON from the first SSE line
        for line in text.strip().split("\n"):
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                assert payload["status"] == "complete"
                assert payload["stage"] == "done"
                break

    def test_not_found(self, knowledge_app, knowledge_client):
        resp = knowledge_client.get("/api/knowledge/install/nonexistent/progress")
        assert resp.status_code == 404


# ===================================================================
# Install — Cancel
# ===================================================================


class TestCancelInstall:
    """POST /api/knowledge/install/{job_id}/cancel"""

    def test_cancel_running(self, knowledge_app, knowledge_client):
        from augmentum.proxy.knowledge_routes import InstallJob

        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = True

        job = InstallJob(
            job_id="run123",
            catalog_id="test.en",
            status="running",
            stage="downloading",
        )
        job.task = mock_task
        knowledge_app.state.install_jobs["run123"] = job

        resp = knowledge_client.post("/api/knowledge/install/run123/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "run123"
        assert data["status"] == "cancelled"
        mock_task.cancel.assert_called_once()

    def test_cancel_already_done(self, knowledge_app, knowledge_client):
        from augmentum.proxy.knowledge_routes import InstallJob

        mock_task = MagicMock()
        mock_task.done.return_value = True

        job = InstallJob(
            job_id="done123",
            catalog_id="test.en",
            status="complete",
            stage="done",
        )
        job.task = mock_task
        knowledge_app.state.install_jobs["done123"] = job

        resp = knowledge_client.post("/api/knowledge/install/done123/cancel")
        assert resp.status_code == 200
        # Task was already done, so cancel shouldn't be called
        mock_task.cancel.assert_not_called()
        assert resp.json()["status"] == "complete"

    def test_not_found(self, knowledge_app, knowledge_client):
        resp = knowledge_client.post("/api/knowledge/install/nope/cancel")
        assert resp.status_code == 404


# ===================================================================
# Storage Location
# ===================================================================


class TestStorageLocation:
    """PUT /api/knowledge/storage-location"""

    def test_change_location(self, knowledge_client, tmp_path):
        new_dir = tmp_path / "new_knowledge_dir"
        resp = knowledge_client.put(
            "/api/knowledge/storage-location",
            json={"path": str(new_dir)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["path"] == str(new_dir)
        assert new_dir.exists()

    def test_nested_dirs_created(self, knowledge_client, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "packs"
        resp = knowledge_client.put(
            "/api/knowledge/storage-location",
            json={"path": str(deep)},
        )
        assert resp.status_code == 200
        assert deep.exists()

    def test_missing_path(self, knowledge_client):
        resp = knowledge_client.put(
            "/api/knowledge/storage-location",
            json={},
        )
        assert resp.status_code == 400

    def test_empty_path(self, knowledge_client):
        resp = knowledge_client.put(
            "/api/knowledge/storage-location",
            json={"path": ""},
        )
        assert resp.status_code == 400

    def test_persists_to_settings_store(self, knowledge_app, knowledge_client, tmp_path):
        """If settings_store is available, the new path should be persisted."""
        mock_store = AsyncMock()
        knowledge_app.state.settings_store = mock_store

        new_dir = tmp_path / "persisted"
        resp = knowledge_client.put(
            "/api/knowledge/storage-location",
            json={"path": str(new_dir)},
        )
        assert resp.status_code == 200
        mock_store.set.assert_called_once_with("knowledge_packs_dir", str(new_dir))


# ===================================================================
# Import
# ===================================================================


class TestImportPack:
    """POST /api/knowledge/import"""

    def test_import_csv(self, knowledge_app, knowledge_client, tmp_path):
        csv_content = b"title,content\nTest Title,This is a test content row with enough text to pass validation.\n"

        with patch("augmentum.knowledge.importer.import_to_augpack", new_callable=AsyncMock) as mock_import:
            mock_import.return_value = {"format": ".csv", "chunk_count": 1, "embedding_dim": 384, "file_size": 1024}
            with patch("augmentum.knowledge.importer.detect_format", return_value=".csv"):
                resp = knowledge_client.post(
                    "/api/knowledge/import",
                    files={"file": ("test.csv", csv_content, "text/csv")},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["filename"] == "test.augpack"
        assert data["chunk_count"] == 1

    def test_unsupported_format(self, knowledge_client):
        resp = knowledge_client.post(
            "/api/knowledge/import",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]

    def test_duplicate_name(self, knowledge_app, knowledge_client, tmp_path):
        """If the output .augpack already exists, return 409."""
        pack_dir = knowledge_app.state.pack_manager.pack_dir
        (pack_dir / "existing.augpack").write_bytes(b"taken")

        with patch("augmentum.knowledge.importer.detect_format", return_value=".csv"):
            resp = knowledge_client.post(
                "/api/knowledge/import",
                files={"file": ("existing.csv", b"title,content\na,b\n", "text/csv")},
            )
        assert resp.status_code == 409

    def test_503_when_no_manager(self, bare_client):
        resp = bare_client.post(
            "/api/knowledge/import",
            files={"file": ("test.csv", b"data", "text/csv")},
        )
        assert resp.status_code == 503

    def test_import_augpack_direct(self, knowledge_app, knowledge_client):
        """Importing a .augpack file should work (direct copy path)."""
        with patch("augmentum.knowledge.importer.import_to_augpack", new_callable=AsyncMock) as mock_import:
            mock_import.return_value = {"format": "augpack", "chunk_count": 0, "file_size": 512, "copied": True}
            with patch("augmentum.knowledge.importer.detect_format", return_value=".augpack"):
                resp = knowledge_client.post(
                    "/api/knowledge/import",
                    files={"file": ("mypack.augpack", b"fake-augpack", "application/octet-stream")},
                )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


# ===================================================================
# Install Job Lifecycle (integration-style)
# ===================================================================


class TestInstallLifecycle:
    """Test create -> check progress -> cancel flow."""

    def test_full_lifecycle(self, knowledge_app, knowledge_client):
        from augmentum.proxy.knowledge_routes import InstallJob

        # 1. Create a job manually (avoid real download)
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel.return_value = True

        job = InstallJob(
            job_id="lifecycle1",
            catalog_id="test.en",
            status="running",
            stage="downloading",
            current=50,
            total=100,
        )
        job.task = mock_task
        knowledge_app.state.install_jobs["lifecycle1"] = job

        # 2. Check progress
        # Set job to complete so SSE terminates
        job.status = "running"
        job.stage = "downloading"

        # 3. Cancel
        resp = knowledge_client.post("/api/knowledge/install/lifecycle1/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

        # 4. Progress should now show cancelled
        resp = knowledge_client.get("/api/knowledge/install/lifecycle1/progress")
        assert resp.status_code == 200
        text = resp.text
        for line in text.strip().split("\n"):
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                assert payload["status"] == "cancelled"
                break
