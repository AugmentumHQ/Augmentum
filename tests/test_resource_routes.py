"""Tests for resource_routes.py — resource monitoring endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


def _mock_ledger():
    snap = MagicMock(
        gpu_name="GPU-B", gpu_total_mb=24576, gpu_used_mb=8192,
        gpu_free_mb=16384, ram_total_mb=65536, ram_used_mb=32768,
        ram_free_mb=32768, models=[], gpu_processes=[],
        unattributed_vram_mb=0,
        # Sprint A additions — empty lists so JSON encoder is happy.
        disk_destinations=[], active_jobs=[], inventory=[],
        inventory_etag="",
    )
    ledger = MagicMock()
    ledger.collect = AsyncMock(return_value=snap)
    ledger.last_snapshot = snap
    ledger.list_profiles = AsyncMock(return_value=[])
    ledger.get_history = AsyncMock(return_value=[])
    ledger.can_fit_model = AsyncMock(return_value=(True, 4096))
    return ledger


class TestResourceStatus:
    def test_status_no_ledger(self, client):
        resp = client.get("/api/resources/status")
        assert resp.status_code == 503

    def test_status_success(self, app, client):
        app.state.resource_ledger = _mock_ledger()
        resp = client.get("/api/resources/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu"]["name"] == "GPU-B"
        assert data["gpu"]["scope"] in {"host", "unknown"}
        assert data["ram"]["total_mb"] == 65536
        assert data["ram"]["scope"] in {"host", "runtime"}
        assert data["cpu_scope"] in {"host", "runtime"}
        assert isinstance(data["models"], list)

    def test_status_marks_runtime_scope_in_container(self, app, client):
        app.state.resource_ledger = _mock_ledger()
        with patch("augmentum.proxy.resource_routes.os.path.exists", return_value=True):
            resp = client.get("/api/resources/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ram"]["scope"] == "runtime"
        assert data["cpu_scope"] == "runtime"

    def test_status_models_carry_confidence_and_as_of(self, app, client):
        from datetime import datetime

        from augmentum.resource.ledger import ResourceSnapshot, TrackedModel
        snap = ResourceSnapshot(
            timestamp=datetime(2026, 6, 19, 12, 0, 0),
            gpu_total_mb=24576, gpu_used_mb=8192, gpu_free_mb=16384,
            models=[
                TrackedModel(name="m-measured", subsystem="llm", backend="engine",
                             device="gpu", vram_mb=6000, confidence="measured"),
                TrackedModel(name="m-declared", subsystem="embeddings",
                             backend="fastembed", device="cpu", ram_mb=45,
                             confidence="declared"),
            ],
        )
        ledger = MagicMock()
        ledger.collect = AsyncMock(return_value=snap)
        app.state.resource_ledger = ledger
        resp = client.get("/api/resources/status")
        assert resp.status_code == 200
        models = {m["name"]: m for m in resp.json()["models"]}
        assert models["m-measured"]["confidence"] == "measured"
        assert models["m-declared"]["confidence"] == "declared"
        # as_of is the snapshot's collect time, surfaced to every ledger model.
        assert models["m-measured"]["as_of"] > 0
        assert "cpu_pct" in models["m-measured"]

    def test_unattributed_vram_subtracts_measured_sidecars_on_wsl2(self, app, client):
        # WSL2: gpu_processes is empty (per-process VRAM opaque), so a sidecar's
        # measured VRAM is sitting inside the residual — subtract it back out.
        from datetime import datetime

        from augmentum.resource.ledger import ResourceSnapshot
        snap = ResourceSnapshot(
            timestamp=datetime(2026, 6, 19, 12, 0, 0),
            gpu_total_mb=24576, gpu_used_mb=8000, gpu_free_mb=16576,
            models=[], gpu_processes=[], unattributed_vram_mb=8000,
        )
        ledger = MagicMock()
        ledger.collect = AsyncMock(return_value=snap)
        app.state.resource_ledger = ledger

        sidecars = [{
            "name": "Classifier", "subsystem": "llm", "backend": "container",
            "device": "gpu", "vram_mb": 2000, "ram_mb": 600, "cpu_pct": 5.0,
            "status": "ready", "container": "augmentum-classifier-1",
            "controllable": True, "kind": "sidecar", "confidence": "measured",
            "as_of": 0.0,
        }]
        with patch("augmentum.resource.container_probe.probe_sidecar_containers",
                   new=AsyncMock(return_value=sidecars)):
            resp = client.get("/api/resources/status")
        assert resp.status_code == 200
        data = resp.json()
        # 8000 device-used − 2000 attributed sidecar = 6000 truly unattributed.
        assert data["unattributed_vram_mb"] == 6000
        assert any(m["name"] == "Classifier" and m["vram_mb"] == 2000
                   for m in data["models"])

    def test_status_host_unavailable_by_default(self, app, client):
        app.state.resource_ledger = _mock_ledger()
        resp = client.get("/api/resources/status")
        assert resp.status_code == 200
        assert resp.json()["host"] == {"available": False}

    def test_status_includes_host_block_when_agent_up(self, app, client):
        from augmentum.resource.host_probe import HostStats
        app.state.resource_ledger = _mock_ledger()
        fake = HostStats(
            ram_total_mb=65536, ram_used_mb=40000, ram_free_mb=25000,
            cpu_pct=31.4, cpu_count=24, os_name="Windows", hostname="DESKTOP-ABC",
        )
        with patch("augmentum.proxy.resource_routes.probe_host_stats",
                   new=AsyncMock(return_value=fake)):
            resp = client.get("/api/resources/status")
        assert resp.status_code == 200
        host = resp.json()["host"]
        assert host["available"] is True
        assert host["source"] == "agent"
        assert host["os"] == "Windows"
        assert host["hostname"] == "DESKTOP-ABC"
        assert host["ram"]["total_mb"] == 65536
        assert host["cpu_pct"] == 31.4


class TestModelProfiles:
    def test_profiles_no_ledger(self, client):
        resp = client.get("/api/resources/profiles")
        assert resp.status_code == 200
        assert resp.json()["profiles"] == []

    def test_profiles_success(self, app, client):
        app.state.resource_ledger = _mock_ledger()
        resp = client.get("/api/resources/profiles")
        assert resp.status_code == 200


class TestResourceHistory:
    def test_history_no_ledger(self, client):
        resp = client.get("/api/resources/history")
        assert resp.status_code == 200
        assert resp.json()["snapshots"] == []


class TestCheckModelFit:
    def test_check_no_ledger(self, client):
        resp = client.get("/api/resources/check/llama3.1:8b")
        assert resp.status_code == 200
        assert resp.json()["can_fit"] is True

    def test_check_success(self, app, client):
        app.state.resource_ledger = _mock_ledger()
        resp = client.get("/api/resources/check/llama3.1:8b")
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_fit"] is True
        assert data["estimated_vram_mb"] == 4096


class TestUnloadModel:
    def test_unload_missing_name(self, client):
        resp = client.post("/api/resources/unload", json={"name": ""})
        assert resp.status_code == 400

    def test_unload_no_manager(self, client):
        resp = client.post(
            "/api/resources/unload",
            json={"name": "llama3.1:8b", "backend": "ollama"},
        )
        assert resp.status_code == 200  # model_manager is already mocked in conftest

    def test_unload_engine_uses_manager(self, app, client):
        app.state.llama_manager = MagicMock()
        app.state.llama_manager.stop = AsyncMock()

        resp = client.post(
            "/api/resources/unload",
            json={"name": "Qwen3.6-35B-A3B-UD-Q8_K_XL", "backend": "engine"},
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        app.state.llama_manager.stop.assert_awaited_once()
        app.state.provider_registry.invalidate_model_map.assert_called()

    def test_unload_engine_backend_fallback_passes_name(self, app, client):
        if hasattr(app.state, "llama_manager"):
            delattr(app.state, "llama_manager")
        engine_backend = MagicMock()
        engine_backend.unload_model = AsyncMock(return_value=True)
        app.state.provider_registry.get_backend.return_value = engine_backend

        resp = client.post(
            "/api/resources/unload",
            json={"name": "Qwen3.6-35B-A3B-UD-Q8_K_XL", "backend": "engine"},
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        engine_backend.unload_model.assert_awaited_once_with("Qwen3.6-35B-A3B-UD-Q8_K_XL")
