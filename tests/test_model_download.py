"""Tests for unified model download — Ollama + llama.cpp GGUF."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.model_manager import ModelManager

# --- Unit tests for ModelManager GGUF methods ---


class TestListGgufFiles:
    """Tests for ModelManager.list_gguf_files()."""

    @staticmethod
    async def _passthrough_sizes(repo_id, files, limit=None):
        return files

    @pytest.mark.asyncio
    async def test_list_gguf_files_returns_gguf_only(self):
        """Should return only .gguf files from repo siblings."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        mock_sibling_gguf = MagicMock()
        mock_sibling_gguf.rfilename = "model-Q4_K_M.gguf"
        mock_sibling_gguf.size = 5_000_000_000

        mock_sibling_txt = MagicMock()
        mock_sibling_txt.rfilename = "README.md"
        mock_sibling_txt.size = 1000

        mock_sibling_gguf2 = MagicMock()
        mock_sibling_gguf2.rfilename = "model-Q8_0.gguf"
        mock_sibling_gguf2.size = 8_500_000_000

        mock_repo = MagicMock()
        mock_repo.siblings = [mock_sibling_gguf, mock_sibling_txt, mock_sibling_gguf2]

        with (
            patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
            patch("augmentum.models.model_manager.asyncio.to_thread", new=AsyncMock(return_value=mock_repo)),
            patch.object(mgr, "fill_missing_hf_file_sizes", new=AsyncMock(side_effect=self._passthrough_sizes)),
        ):
            files = await mgr.list_gguf_files("bartowski/test-GGUF")

        assert len(files) == 2
        assert files[0]["filename"] == "model-Q4_K_M.gguf"
        assert files[0]["size"] == 5_000_000_000
        assert files[1]["filename"] == "model-Q8_0.gguf"

    @pytest.mark.asyncio
    async def test_list_gguf_files_empty_repo(self):
        """Should return empty list if no .gguf files."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        mock_repo = MagicMock()
        mock_repo.siblings = [MagicMock(rfilename="README.md", size=100)]

        with (
            patch.dict("sys.modules", {"huggingface_hub": MagicMock()}),
            patch("augmentum.models.model_manager.asyncio.to_thread", new=AsyncMock(return_value=mock_repo)),
            patch.object(mgr, "fill_missing_hf_file_sizes", new=AsyncMock(side_effect=self._passthrough_sizes)),
        ):
            files = await mgr.list_gguf_files("some/repo")

        assert files == []

    @pytest.mark.asyncio
    async def test_list_gguf_files_import_error(self):
        """Should raise ImportError if huggingface_hub is not installed."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        with (
            patch.dict("sys.modules", {"huggingface_hub": None}),
            pytest.raises(ImportError, match="huggingface_hub"),
        ):
            await mgr.list_gguf_files("some/repo")

    @pytest.mark.asyncio
    async def test_list_gguf_files_bad_repo(self):
        """Should raise ValueError for invalid repo."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        mock_hf = MagicMock()
        mock_hf.HfApi.return_value = MagicMock()

        with (
            patch.dict("sys.modules", {"huggingface_hub": mock_hf}),
            patch("augmentum.models.model_manager.asyncio.to_thread", new=AsyncMock(side_effect=Exception("404"))),
            pytest.raises(ValueError, match="Could not fetch repo info"),
        ):
            await mgr.list_gguf_files("nonexistent/repo")

    @pytest.mark.asyncio
    async def test_fill_missing_hf_file_sizes_enriches_zero_values(self):
        registry = MagicMock()
        mgr = ModelManager(registry)
        mgr.resolve_hf_file_size = AsyncMock(return_value=1234)

        files = [
            {"filename": "model-Q4_K_M.gguf", "size": 0},
            {"filename": "model-Q8_0.gguf", "size": 5678},
        ]

        enriched = await mgr.fill_missing_hf_file_sizes("bartowski/test-GGUF", files)

        assert enriched[0]["size"] == 1234
        assert enriched[1]["size"] == 5678


class TestListLocalGguf:
    """Tests for ModelManager.list_local_gguf()."""

    def test_list_local_gguf_finds_files(self):
        """Should list .gguf files in directory."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf1 = os.path.join(tmpdir, "model-Q4.gguf")
            gguf2 = os.path.join(tmpdir, "model-Q8.gguf")
            txt = os.path.join(tmpdir, "README.md")
            for f in [gguf1, gguf2, txt]:
                with open(f, "w") as fh:
                    fh.write("test")

            files = mgr.list_local_gguf(tmpdir)

        assert len(files) == 2
        names = [f["filename"] for f in files]
        assert "model-Q4.gguf" in names
        assert "model-Q8.gguf" in names
        assert all(f["size"] > 0 for f in files)

    def test_list_local_gguf_subdir(self):
        """Should list .gguf files in subdirectories."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "bartowski")
            os.makedirs(subdir)
            gguf = os.path.join(subdir, "model.gguf")
            with open(gguf, "w") as fh:
                fh.write("test")

            files = mgr.list_local_gguf(tmpdir)

        assert len(files) == 1
        assert files[0]["filename"] == "bartowski/model.gguf"

    def test_list_local_gguf_nonexistent_dir(self):
        """Should return empty list for nonexistent directory."""
        registry = MagicMock()
        mgr = ModelManager(registry)
        files = mgr.list_local_gguf("/nonexistent/path/12345")
        assert files == []

    def test_list_local_gguf_empty_dir(self):
        """Should return empty list for empty directory."""
        registry = MagicMock()
        mgr = ModelManager(registry)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = mgr.list_local_gguf(tmpdir)

        assert files == []


# Multi-part download handler tests live in tests/test_gguf_download.py.
# The legacy single-stream pull_gguf was removed — see jobs/handlers/gguf_download.py.


# --- API route tests ---


class TestModelPullEndpoint:
    """Tests for POST /api/models/pull."""

    def test_pull_ollama_routes_correctly(self, client):
        """Ollama pull should stream back NDJSON from Ollama backend."""
        async def mock_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "success"}

        client.app.state.model_manager.pull_model = mock_pull

        resp = client.post(
            "/api/models/pull",
            json={"backend": "ollama", "name": "llama3.1:8b"},
        )
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]

        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        statuses = [entry.get("status") for entry in lines]
        assert "pulling manifest" in statuses
        assert "success" in statuses

    def test_pull_llamacpp_enqueues_background_job(self, client):
        """llama.cpp pull should enqueue a gguf_download job and return its id."""
        client.app.state.model_manager.resolve_hf_file_size = AsyncMock(return_value=1000)

        jobs_store = MagicMock()
        jobs_store.list_for_user = AsyncMock(return_value=[])
        jobs_store.create = AsyncMock(return_value="job-123")
        client.app.state.jobs_store = jobs_store
        client.app.state.job_runner = MagicMock()

        resp = client.post(
            "/api/models/pull",
            json={"backend": "llamacpp", "name": "bartowski/test-GGUF", "filename": "model.gguf"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "queued"
        assert data["existing"] is False
        client.app.state.job_runner.wake.assert_called_once()

    def test_pull_llamacpp_requires_filename(self, client):
        """llama.cpp pull without filename should return error."""
        resp = client.post(
            "/api/models/pull",
            json={"backend": "llamacpp", "name": "bartowski/test-GGUF"},
        )
        assert resp.status_code == 400

    def test_pull_missing_name(self, client):
        """Pull without name should return error."""
        resp = client.post("/api/models/pull", json={"backend": "ollama"})
        assert resp.status_code == 400

    def test_pull_default_backend_is_ollama(self, client):
        """Default backend should be ollama when not specified."""
        async def mock_pull(name):
            yield {"status": "success", "backend": "ollama"}

        client.app.state.model_manager.pull_model = mock_pull

        resp = client.post("/api/models/pull", json={"name": "test-model"})
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.strip().split("\n") if line.strip()]
        assert lines[0].get("backend") == "ollama"


class TestGgufListEndpoint:
    """Tests for GET /api/models/gguf/list."""

    def test_list_gguf_files_success(self, client):
        """Should return list of .gguf files."""
        client.app.state.model_manager.list_gguf_files = AsyncMock(return_value=[
            {"filename": "model-Q4_K_M.gguf", "size": 5_000_000_000},
            {"filename": "model-Q8_0.gguf", "size": 8_500_000_000},
        ])

        resp = client.get("/api/models/gguf/list?repo=bartowski/test-GGUF")
        assert resp.status_code == 200
        data = resp.json()
        assert data["repo"] == "bartowski/test-GGUF"
        assert len(data["files"]) == 2

    def test_list_gguf_files_import_error(self, client):
        """Should return error field when huggingface_hub missing."""
        client.app.state.model_manager.list_gguf_files = AsyncMock(
            side_effect=ImportError("huggingface_hub not installed")
        )

        resp = client.get("/api/models/gguf/list?repo=bartowski/test-GGUF")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert data["files"] == []

    def test_list_gguf_files_bad_repo(self, client):
        """Should return error for invalid repo."""
        client.app.state.model_manager.list_gguf_files = AsyncMock(
            side_effect=ValueError("Could not fetch repo info")
        )

        resp = client.get("/api/models/gguf/list?repo=nonexistent/repo")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_list_gguf_files_requires_repo(self, client):
        """Should fail without repo query param."""
        resp = client.get("/api/models/gguf/list")
        assert resp.status_code == 422


class TestGgufLocalEndpoint:
    """Tests for GET /api/models/gguf/local."""

    def test_list_local_gguf(self, client):
        """Should return locally downloaded GGUF files."""
        client.app.state.model_manager.list_local_gguf = MagicMock(return_value=[
            {"filename": "model.gguf", "size": 5_000_000_000, "modified": 1709571200.0},
        ])

        resp = client.get("/api/models/gguf/local")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["files"]) == 1
        assert data["files"][0]["filename"] == "model.gguf"

    def test_list_local_gguf_empty(self, client):
        """Should return empty list when no files exist."""
        client.app.state.model_manager.list_local_gguf = MagicMock(return_value=[])

        resp = client.get("/api/models/gguf/local")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == []


# --- Config test ---


class TestConfig:
    """Test that llamacpp_model_dir config exists."""

    def test_llamacpp_model_dir_default(self):
        """Config should have llamacpp_model_dir with default."""
        from augmentum.config import Settings

        s = Settings()
        assert s.llamacpp_model_dir == "/data/llama_models"

    def test_llamacpp_model_dir_env_override(self):
        """Config should be overridable via environment."""
        with patch.dict(os.environ, {"AUGMENTUM_LLAMACPP_MODEL_DIR": "/custom/path"}):
            from augmentum.config import Settings

            s = Settings()
            assert s.llamacpp_model_dir == "/custom/path"
