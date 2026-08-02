"""Tests for model_routes.py — model management API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# GET /api/models/status
# ---------------------------------------------------------------------------


class TestModelsStatus:
    def test_status_returns_model_list(self, client):
        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 1
        model = data["models"][0]
        assert "name" in model
        assert "size" in model

    def test_status_model_shape(self, client):
        resp = client.get("/api/models/status")
        data = resp.json()
        for m in data["models"]:
            assert "name" in m
            assert "size" in m
            assert "modified_at" in m


# ---------------------------------------------------------------------------
# GET /api/models/running
# ---------------------------------------------------------------------------


class TestRunningModels:
    def test_running_returns_empty_list(self, client):
        resp = client.get("/api/models/running")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)


# ---------------------------------------------------------------------------
# GET /api/models/{model_name}/info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_model_info_returns_details(self, client):
        # The mock model_manager.get_model_status returns a MagicMock
        mock_status = MagicMock()
        mock_status.name = "llama3.1:8b"
        mock_status.available = True
        mock_status.backend = "ollama"
        mock_status.quantization = "q4_0"
        mock_status.parameter_count = "8B"
        mock_status.loaded = True
        client.app.state.model_manager.get_model_status = AsyncMock(return_value=mock_status)

        resp = client.get("/api/models/llama3.1:8b/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "llama3.1:8b"
        assert data["available"] is True
        assert "backend" in data


# ---------------------------------------------------------------------------
# POST /api/models/{model_name}/load
# ---------------------------------------------------------------------------


class TestModelLoad:
    def test_load_model_success(self, client):
        resp = client.post("/api/models/llama3.1:8b/load")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["model"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# POST /api/models/{model_name}/unload
# ---------------------------------------------------------------------------


class TestModelUnload:
    def test_unload_model_success(self, client):
        resp = client.post("/api/models/llama3.1:8b/unload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["model"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# POST /api/models/pull
# ---------------------------------------------------------------------------


class TestModelPull:
    def test_pull_missing_name_returns_400(self, client):
        resp = client.post("/api/models/pull", json={"name": ""})
        assert resp.status_code == 400

    def test_pull_llamacpp_missing_filename_returns_400(self, client):
        resp = client.post("/api/models/pull", json={
            "name": "some-repo/model",
            "backend": "llamacpp",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/models/gguf/local
# ---------------------------------------------------------------------------


class TestGGUFLocal:
    def test_local_gguf_returns_list(self, client):
        mock_manager = client.app.state.model_manager
        mock_manager.list_local_gguf = MagicMock(return_value=[])
        resp = client.get("/api/models/gguf/local")
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        assert "model_dir" in data


# ---------------------------------------------------------------------------
# Download activity routes
# ---------------------------------------------------------------------------


class TestDownloadActivity:
    def test_retry_failed_download_reuses_same_job(self, app, client):
        jobs_store = MagicMock()
        jobs_store.get = AsyncMock(return_value={
            "id": "job_dl_1",
            "job_type": "gguf_download",
            "status": "failed",
            "payload": {"filename": "foo.gguf", "model_dir": "/tmp"},
        })
        jobs_store.reset_for_retry = AsyncMock(return_value=True)
        app.state.jobs_store = jobs_store
        app.state.job_runner = MagicMock()

        resp = client.post("/api/models/downloads/job_dl_1/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        jobs_store.reset_for_retry.assert_awaited_once_with("job_dl_1", user_id="usr_test")
        app.state.job_runner.wake.assert_called_once()

    def test_delete_terminal_download_can_remove_partial_file(self, app, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        partial = model_dir / "foo.gguf.part"
        partial.write_bytes(b"partial-bytes")

        jobs_store = MagicMock()
        jobs_store.get = AsyncMock(return_value={
            "id": "job_dl_2",
            "job_type": "gguf_download",
            "status": "failed",
            "payload": {"filename": "foo.gguf", "model_dir": str(model_dir)},
        })
        jobs_store.delete_job = AsyncMock(return_value=True)
        app.state.jobs_store = jobs_store

        llama_manager = MagicMock()
        llama_manager.model_dirs = [str(model_dir)]
        app.state.llama_manager = llama_manager

        resp = client.delete("/api/models/downloads/job_dl_2?delete_partial=true")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["partial_deleted"] is True
        assert not partial.exists()
        jobs_store.delete_job.assert_awaited_once_with("job_dl_2", user_id="usr_test")

    def test_delete_active_download_is_rejected(self, app, client):
        jobs_store = MagicMock()
        jobs_store.get = AsyncMock(return_value={
            "id": "job_dl_3",
            "job_type": "gguf_download",
            "status": "running",
            "payload": {"filename": "foo.gguf", "model_dir": "/tmp"},
        })
        app.state.jobs_store = jobs_store

        resp = client.delete("/api/models/downloads/job_dl_3")
        assert resp.status_code == 409

    def test_delete_missing_download_is_idempotent(self, app, client):
        jobs_store = MagicMock()
        jobs_store.get = AsyncMock(return_value=None)
        app.state.jobs_store = jobs_store

        resp = client.delete("/api/models/downloads/job_missing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["already_gone"] is True
        assert data["deleted"] is False

    def test_cleanup_downloads_bulk_removes_history_and_partials(self, app, client, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        partial = model_dir / "stale.gguf.part"
        partial.write_bytes(b"partial-bytes")

        jobs_store = MagicMock()
        jobs_store.list_for_user = AsyncMock(return_value=[
            {
                "id": "job_done",
                "job_type": "gguf_download",
                "status": "completed",
                "payload": {"filename": "done.gguf", "model_dir": str(model_dir)},
            },
            {
                "id": "job_partial",
                "job_type": "gguf_download",
                "status": "failed",
                "payload": {"filename": "stale.gguf", "model_dir": str(model_dir)},
            },
        ])
        jobs_store.delete_job = AsyncMock(return_value=True)
        app.state.jobs_store = jobs_store

        llama_manager = MagicMock()
        llama_manager.model_dirs = [str(model_dir)]
        app.state.llama_manager = llama_manager

        resp = client.post("/api/models/downloads/cleanup", json={
            "statuses": ["completed", "failed"],
            "require_partial": True,
            "delete_partial": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["removed"] == 1
        assert data["partial_deleted"] == 1
        assert not partial.exists()
        jobs_store.delete_job.assert_awaited_once_with("job_partial", user_id="usr_test")


# ---------------------------------------------------------------------------
# llama.cpp routes — 404 when no backend
# ---------------------------------------------------------------------------


class TestLlamaCppRoutes:
    def test_llamacpp_status_no_backend_returns_404(self, client):
        client.app.state.provider_registry.get_backend = MagicMock(return_value=None)
        resp = client.get("/api/llamacpp/status")
        assert resp.status_code == 404

    def test_llamacpp_slots_no_backend_returns_404(self, client):
        client.app.state.provider_registry.get_backend = MagicMock(return_value=None)
        resp = client.get("/api/llamacpp/slots")
        assert resp.status_code == 404

    def test_llamacpp_slots_falls_back_to_managed_engine_backend(self, client):
        backend = MagicMock()
        backend.health = AsyncMock()
        backend.get_props = AsyncMock()
        backend.get_slots = AsyncMock(return_value=[{"id": 0, "n_past": 12}])
        backend.tokenize = AsyncMock()
        backend.detokenize = AsyncMock()

        def get_backend(name: str):
            if name == "engine":
                return backend
            return None

        client.app.state.provider_registry.get_backend = MagicMock(side_effect=get_backend)
        resp = client.get("/api/llamacpp/slots")
        assert resp.status_code == 200
        assert resp.json()["slots"] == [{"id": 0, "n_past": 12}]
        backend.get_slots.assert_awaited_once()


# ---------------------------------------------------------------------------
# /api/engine/v2/* — the managed llama-server path (Engine v2)
# ---------------------------------------------------------------------------

def _mock_llama_manager():
    """Minimal LlamaServerManager mock with the attributes and methods the
    v2 routes reach for. Tests override specific behaviors per-case."""
    from augmentum.models.llama_server_manager import ProcessState

    mgr = MagicMock()
    mgr.state = ProcessState.IDLE
    mgr.model_id = ""
    mgr.model_dirs = []
    mgr.draft_model = ""
    mgr.draft_max = 5
    mgr.status = MagicMock(return_value={"state": "stopped", "model_id": ""})
    mgr.discover_models = AsyncMock(return_value=[])
    mgr.build_load_plan = MagicMock(return_value={
        "model_id": "fake-id",
        "model_path": "/tmp/model.gguf",
        "profile": {"context_length": 8192},
        "applied": {"ctx_size": 8192, "gpu_layers_mode": "auto", "gpu_layers": 32},
        "memory": {"estimated_vram_mb": 4096, "estimated_ram_mb": 2048},
        "warnings": [],
    })
    mgr.start = AsyncMock()
    mgr.swap = AsyncMock()
    mgr.stop = AsyncMock()
    mgr.persist_load_options = AsyncMock()
    mgr._resolve_model_path = MagicMock(return_value=None)
    mgr.list_pinned_sessions = MagicMock(return_value=[])
    mgr.pin_session = MagicMock(return_value=True)
    mgr.unpin_session = MagicMock(return_value=True)
    mgr.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return mgr


class TestEngineV2Enablement:
    """The 404 gate when v2 isn't wired catches degraded startups; every
    downstream test implicitly depends on it."""

    def test_status_404_when_manager_missing(self, client):
        """/v2/status uses _get_llama_manager which raises 404. Without the
        gate, every v2 call would AttributeError."""
        if hasattr(client.app.state, "llama_manager"):
            delattr(client.app.state, "llama_manager")
        r = client.get("/api/engine/v2/status")
        assert r.status_code == 404


class TestEngineV2Status:
    def test_status_returns_manager_state(self, app, client):
        mgr = _mock_llama_manager()
        mgr.status = MagicMock(return_value={"state": "ready", "model_id": "llama-3"})
        app.state.llama_manager = mgr

        r = client.get("/api/engine/v2/status")
        assert r.status_code == 200
        assert r.json()["state"] == "ready"
        assert r.json()["model_id"] == "llama-3"

    def test_status_error_surfaces_as_error_state(self, app, client):
        """If the manager raises, the route doesn't 500 — it returns a
        well-formed error payload so the UI can keep polling."""
        mgr = _mock_llama_manager()
        mgr.status = MagicMock(side_effect=RuntimeError("engine crashed"))
        app.state.llama_manager = mgr

        r = client.get("/api/engine/v2/status")
        assert r.status_code == 200
        assert r.json()["state"] == "error"
        assert "engine crashed" in r.json()["error"]


class TestEngineV2LoadModel:
    def test_load_missing_path_returns_400(self, app, client):
        app.state.llama_manager = _mock_llama_manager()
        r = client.post("/api/engine/v2/models/load", json={})
        assert r.status_code == 400

    def test_load_unresolved_relative_path_returns_404(self, app, client):
        """A relative path that can't be resolved by the manager must 404,
        not crash or start loading nonsense."""
        mgr = _mock_llama_manager()
        mgr._resolve_model_path = MagicMock(return_value=None)
        app.state.llama_manager = mgr

        r = client.post("/api/engine/v2/models/load",
                        json={"model_path": "not-a-real-model.gguf"})
        assert r.status_code == 404

    def test_load_absolute_path_calls_start(self, app, client, tmp_path):
        """When the server is stopped, load triggers start() with the
        resolved path and reports `loaded` on success."""
        from augmentum.models.llama_server_manager import ProcessState

        # Write a tiny real file so the absolute-path guard passes
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake")

        mgr = _mock_llama_manager()
        mgr.state = ProcessState.IDLE
        mgr.model_id = "fake-id"
        app.state.llama_manager = mgr

        r = client.post("/api/engine/v2/models/load",
                        json={"model_path": str(model_file)})
        assert r.status_code == 200
        assert r.json()["status"] == "loaded"
        mgr.start.assert_awaited_once()

    def test_load_timeout_returns_504(self, app, client, tmp_path):
        model_file = tmp_path / "m.gguf"
        model_file.write_bytes(b"x")
        mgr = _mock_llama_manager()
        mgr.start = AsyncMock(side_effect=TimeoutError("slow"))
        app.state.llama_manager = mgr

        r = client.post("/api/engine/v2/models/load",
                        json={"model_path": str(model_file)})
        assert r.status_code == 504

    def test_load_forwards_load_options(self, app, client, tmp_path):
        from augmentum.models.llama_server_manager import ProcessState

        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake")

        mgr = _mock_llama_manager()
        mgr.state = ProcessState.IDLE
        app.state.llama_manager = mgr

        r = client.post("/api/engine/v2/models/load", json={
            "model_path": str(model_file),
            "ctx_size": 16384,
            "gpu_layers_mode": "custom",
            "gpu_layers": 24,
            "batch_size": 1024,
            "kv_cache_type": "q8_0",
            "flash_attn": False,
            "idle_timeout": 900,
        })
        assert r.status_code == 200
        mgr.start.assert_awaited_once()
        kwargs = mgr.start.await_args.kwargs
        assert kwargs["load_options"]["ctx_size"] == 16384
        assert kwargs["load_options"]["gpu_layers_mode"] == "custom"
        assert kwargs["load_options"]["gpu_layers"] == 24
        assert kwargs["load_options"]["kv_cache_type"] == "q8_0"


class TestEngineV2LoadPlan:
    def test_plan_returns_manager_preview(self, app, client, tmp_path):
        model_file = tmp_path / "plan.gguf"
        model_file.write_bytes(b"fake")

        mgr = _mock_llama_manager()
        app.state.llama_manager = mgr

        r = client.post("/api/engine/v2/models/plan", json={
            "model_path": str(model_file),
            "ctx_size": 16384,
        })
        assert r.status_code == 200
        mgr.build_load_plan.assert_called_once()
        assert r.json()["memory"]["estimated_vram_mb"] == 4096


class TestEngineV2UnloadModel:
    def test_unload_calls_stop(self, app, client):
        mgr = _mock_llama_manager()
        app.state.llama_manager = mgr
        r = client.post("/api/engine/v2/models/unload")
        assert r.status_code == 200
        assert r.json()["status"] == "unloaded"
        mgr.stop.assert_awaited_once()

    def test_unload_swallows_stop_errors(self, app, client):
        """Stop errors can't block future loads — the endpoint must succeed
        even if the process is already dead."""
        mgr = _mock_llama_manager()
        mgr.stop = AsyncMock(side_effect=RuntimeError("already stopped"))
        app.state.llama_manager = mgr
        r = client.post("/api/engine/v2/models/unload")
        assert r.status_code == 200


class TestEngineV2SessionPinning:
    def test_pin_session(self, app, client):
        mgr = _mock_llama_manager()
        app.state.llama_manager = mgr
        r = client.post("/api/engine/v2/sessions/pin",
                        json={"session_id": "sess-1"})
        assert r.status_code == 200
        mgr.pin_session.assert_called_once()

    def test_unpin_session(self, app, client):
        mgr = _mock_llama_manager()
        app.state.llama_manager = mgr
        r = client.post("/api/engine/v2/sessions/unpin",
                        json={"session_id": "sess-1"})
        assert r.status_code == 200
        mgr.unpin_session.assert_called_once()

    def test_list_pinned(self, app, client):
        mgr = _mock_llama_manager()
        mgr._pinned_sessions = {"sess-A", "sess-B"}
        app.state.llama_manager = mgr
        r = client.get("/api/engine/v2/sessions/pinned")
        assert r.status_code == 200
        assert set(r.json()["pinned"]) == {"sess-A", "sess-B"}


# ---------------------------------------------------------------------------
# Router registration sanity
# ---------------------------------------------------------------------------

class TestRouterShape:
    def test_three_routers_registered(self):
        """Three routers ship in this file — regressions that drop one would
        silently break the UI's model management surface."""
        from augmentum.proxy.model_routes import engine_router, llamacpp_router, router
        assert router.prefix == "/api/models"
        assert llamacpp_router.prefix == "/api/llamacpp"
        assert engine_router.prefix == "/api/engine"

    def test_engine_v2_endpoints_registered(self):
        from augmentum.proxy.model_routes import engine_router
        paths = {r.path for r in engine_router.routes}
        expected = {
            "/api/engine/v2/status", "/api/engine/v2/models",
            "/api/engine/v2/models/plan",
            "/api/engine/v2/models/load", "/api/engine/v2/models/unload",
            "/api/engine/v2/sessions/pin", "/api/engine/v2/sessions/unpin",
            "/api/engine/v2/sessions/pinned",
            "/api/engine/v2/embeddings", "/api/engine/v2/generate",
            "/api/engine/v2/browse", "/api/engine/v2/cache/stats",
        }
        assert expected.issubset(paths)


# ---------------------------------------------------------------------------
# GET /api/models/vision/captioner-options
# ---------------------------------------------------------------------------


class TestVisionCaptionerOptions:
    """The captioner (vision sibling) picker — surfaces installed VL base+
    projector pairs so users choose from a dropdown instead of typing paths."""

    def test_no_manager_returns_unavailable(self, client):
        # conftest doesn't wire a llama_manager; endpoint must degrade cleanly
        if hasattr(client.app.state, "llama_manager"):
            delattr(client.app.state, "llama_manager")
        resp = client.get("/api/models/vision/captioner-options")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["options"] == []
        assert "model_path" in data["current"]
        assert "mmproj_path" in data["current"]

    def test_filters_to_compatible_bases(self, client):
        mgr = MagicMock()
        mgr.discover_gguf_files.return_value = [
            {"filename": "SmolVLM-256M-Instruct-Q8_0.gguf", "path": "/m/smol.gguf"},
            {"filename": "BigText-70B-Q4.gguf", "path": "/m/big.gguf"},
            {"filename": "mmproj-SmolVLM-256M.gguf", "path": "/m/mmproj-smol.gguf"},
        ]
        mgr.profile_cache = MagicMock()
        mgr.profile_cache.get.return_value = None

        def _candidates(base_path, profile):
            compatible = base_path == "/m/smol.gguf"
            return [{
                "path": "/m/mmproj-smol.gguf",
                "filename": "mmproj-SmolVLM-256M.gguf",
                "compatible": compatible,
                "reason": "" if compatible else "projection dim mismatch",
                "projector_type": "smolvlm",
                "projection_dim": 768,
                "is_current": compatible,
            }]
        mgr.suggest_mmproj_candidates.side_effect = _candidates
        client.app.state.llama_manager = mgr

        resp = client.get("/api/models/vision/captioner-options")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True

        names = [o["base_name"] for o in data["options"]]
        # SmolVLM has a compatible projector → included.
        assert "SmolVLM-256M-Instruct-Q8_0" in names
        # Big text model: projector present but dim-incompatible → excluded.
        assert all("BigText" not in n for n in names)
        # An mmproj file is never itself offered as a base.
        assert all("mmproj" not in n.lower() for n in names)

        smol = next(o for o in data["options"] if o["base_name"].startswith("SmolVLM"))
        assert smol["base_path"] == "/m/smol.gguf"
        assert smol["projectors"][0]["compatible"] is True
