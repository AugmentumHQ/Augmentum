"""Tests for image generation API routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from augmentum.image.hardware import HardwareProfile, ModelTier
from augmentum.image.presets import PresetManager
from augmentum.image.queue import GenerationQueue


@pytest.fixture
def mock_app_state():
    """Create a mock app state with image subsystem components."""
    state = MagicMock()

    # Hardware
    state.image_hardware = HardwareProfile(
        device="cuda",
        device_name="NVIDIA RTX 3080",
        vram_total_mb=10240,
        vram_free_mb=8192,
        tier=ModelTier.MEDIUM,
        recommended_pipeline="sdxl",
        recommended_model="stabilityai/stable-diffusion-xl-base-1.0",
    )

    # Queue
    state.image_queue = MagicMock(spec=GenerationQueue)
    state.image_queue.queue_size = 0

    # Preset manager
    state.image_preset_manager = PresetManager()

    # Pipeline registry
    state.image_pipeline_registry = MagicMock()
    state.image_pipeline_registry.is_loaded = False
    state.image_pipeline_registry.current_model = ""

    # Model manager
    state.image_model_manager = MagicMock()
    state.image_model_manager.list_local_models.return_value = [
        {
            "name": "sd-v1-5",
            "path": "/data/image_models/sd-v1-5",
            "pipeline_type": "sd15",
            "size_bytes": 4_000_000_000,
            "source": "local",
        }
    ]

    # LoRA manager
    state.image_lora_manager = MagicMock()
    state.image_lora_manager.discover.return_value = []

    # Persistence
    state.image_persistence = AsyncMock()
    state.image_persistence.list_generations = AsyncMock(return_value=[])
    state.image_persistence.count_generations = AsyncMock(return_value=0)
    state.image_persistence.get_generation = AsyncMock(return_value=None)
    state.image_persistence.delete_generation = AsyncMock(return_value=None)

    # Cache
    state.image_cache = AsyncMock()
    state.image_cache.get = AsyncMock(return_value=None)

    return state


@pytest.fixture
def client(mock_app_state):
    """Create a test client with mocked image subsystem."""
    from fastapi import FastAPI

    from augmentum.proxy.image_routes import router

    app = FastAPI()
    app.include_router(router)

    # Inject mock state
    app.state = mock_app_state

    return TestClient(app)


class TestHardwareEndpoint:
    def test_get_hardware(self, client):
        resp = client.get("/api/image/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert data["device"] == "cuda"
        assert data["tier"] == "medium"
        assert "RTX 3080" in data["device_name"]


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/api/image/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "sd-v1-5"

    def test_recommended_models(self, client):
        resp = client.get("/api/image/models/recommended")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "medium"
        assert data["recommended_pipeline"] == "sdxl"


class TestPresetsEndpoint:
    def test_list_presets(self, client):
        resp = client.get("/api/image/presets")
        assert resp.status_code == 200
        data = resp.json()
        names = [p["name"] for p in data]
        assert "fantasy_rpg" in names
        assert "anime" in names

    def test_get_preset(self, client):
        resp = client.get("/api/image/presets/fantasy_rpg")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "fantasy_rpg"
        assert "fantasy" in data["positive_tags"].lower()

    def test_get_preset_not_found(self, client):
        resp = client.get("/api/image/presets/nonexistent")
        assert resp.status_code == 404


class TestConfigEndpoint:
    def test_get_config(self, client):
        resp = client.get("/api/image/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "default_steps" in data
        assert "device" in data


class TestHistoryEndpoint:
    def test_empty_history(self, client):
        resp = client.get("/api/image/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["entries"] == []

    def test_history_returns_total(self, client, mock_app_state):
        from augmentum.image.schemas import HistoryEntry

        entries = [
            HistoryEntry(image_id="img1", prompt="sunset"),
            HistoryEntry(image_id="img2", prompt="mountain"),
        ]
        mock_app_state.image_persistence.list_generations = AsyncMock(return_value=entries)
        mock_app_state.image_persistence.count_generations = AsyncMock(return_value=5)

        resp = client.get("/api/image/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["entries"]) == 2

    def test_history_search(self, client, mock_app_state):
        from augmentum.image.schemas import HistoryEntry

        entries = [HistoryEntry(image_id="img1", prompt="sunset over ocean")]
        mock_app_state.image_persistence.list_generations = AsyncMock(return_value=entries)
        mock_app_state.image_persistence.count_generations = AsyncMock(return_value=1)

        resp = client.get("/api/image/history?q=sunset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["entries"][0]["prompt"] == "sunset over ocean"
        # Verify the search param was passed through
        mock_app_state.image_persistence.list_generations.assert_called_once()
        call_kwargs = mock_app_state.image_persistence.list_generations.call_args
        assert call_kwargs.kwargs.get("q") == "sunset" or (call_kwargs[1].get("q") == "sunset")

    def test_history_filter_model(self, client, mock_app_state):
        from augmentum.image.schemas import HistoryEntry

        entries = [HistoryEntry(image_id="img1", prompt="test", model="sd-v1-5")]
        mock_app_state.image_persistence.list_generations = AsyncMock(return_value=entries)
        mock_app_state.image_persistence.count_generations = AsyncMock(return_value=1)

        resp = client.get("/api/image/history?model=sd-v1-5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entries"]) == 1

    def test_history_pagination(self, client, mock_app_state):
        from augmentum.image.schemas import HistoryEntry

        entries = [HistoryEntry(image_id="img3", prompt="page2")]
        mock_app_state.image_persistence.list_generations = AsyncMock(return_value=entries)
        mock_app_state.image_persistence.count_generations = AsyncMock(return_value=100)

        resp = client.get("/api/image/history?limit=48&offset=48")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 100
        call_kwargs = mock_app_state.image_persistence.list_generations.call_args
        assert call_kwargs.kwargs.get("offset") == 48 or (call_kwargs[1].get("offset") == 48)


class TestDeleteEndpoint:
    def test_delete_image(self, client, mock_app_state):
        mock_app_state.image_persistence.delete_generation = AsyncMock(
            return_value="/data/images/test.png"
        )

        with patch("os.path.exists", return_value=False):
            resp = client.delete("/api/image/test-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

    def test_delete_image_not_found(self, client, mock_app_state):
        mock_app_state.image_persistence.delete_generation = AsyncMock(return_value=None)

        resp = client.delete("/api/image/nonexistent-id")
        assert resp.status_code == 404

    def test_batch_delete(self, client, mock_app_state):
        # First call returns a path, second returns a path too
        mock_app_state.image_persistence.delete_generation = AsyncMock(
            side_effect=["/data/images/a.png", "/data/images/b.png"]
        )

        with patch("os.path.exists", return_value=False):
            resp = client.request(
                "DELETE",
                "/api/image/batch",
                json={"image_ids": ["id-a", "id-b"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "id-a" in data["deleted"]
        assert "id-b" in data["deleted"]
        assert len(data["failed"]) == 0

    def test_batch_delete_partial(self, client, mock_app_state):
        mock_app_state.image_persistence.delete_generation = AsyncMock(
            side_effect=["/data/images/a.png", None]
        )

        with patch("os.path.exists", return_value=False):
            resp = client.request(
                "DELETE",
                "/api/image/batch",
                json={"image_ids": ["id-a", "id-missing"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "id-a" in data["deleted"]
        assert "id-missing" in data["failed"]


class TestLoraEndpoint:
    def test_list_loras(self, client):
        resp = client.get("/api/image/lora/list")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_load_lora_no_model(self, client):
        resp = client.post("/api/image/lora/load", json={"name": "test", "weight": 1.0})
        assert resp.status_code == 409  # No model loaded


class TestCatalogEndpoint:
    def test_catalog_returns_models(self, client):
        resp = client.get("/api/image/models/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0

        # Each entry should have required fields
        first = data[0]
        assert "repo_id" in first
        assert "name" in first
        assert "description" in first
        assert "pipeline_type" in first
        assert "size_gb" in first
        assert "min_vram_mb" in first
        assert "min_tier" in first
        assert "cpu_friendly" in first
        assert "compatible" in first
        assert "installed" in first
        assert "speed_note" in first

    def test_catalog_compatibility_annotation(self, client):
        """With MEDIUM tier (10GB), CPU/LOW/MEDIUM models should be compatible."""
        resp = client.get("/api/image/models/catalog")
        data = resp.json()

        for m in data:
            if m["min_tier"] in ("cpu", "low", "medium"):
                assert m["compatible"] is True, f"{m['name']} should be compatible"
            elif m["min_tier"] == "high":
                assert m["compatible"] is False, f"{m['name']} should be incompatible"

    def test_catalog_has_cpu_friendly(self, client):
        """Catalog should contain at least one CPU-friendly model."""
        resp = client.get("/api/image/models/catalog")
        data = resp.json()
        cpu_models = [m for m in data if m["cpu_friendly"]]
        assert len(cpu_models) >= 1

    def test_catalog_compatible_first(self, client):
        """Compatible models should appear before incompatible ones."""
        resp = client.get("/api/image/models/catalog")
        data = resp.json()
        seen_incompatible = False
        for m in data:
            if not m["compatible"]:
                seen_incompatible = True
            elif seen_incompatible:
                pytest.fail(f"Compatible model {m['name']} appeared after incompatible models")

    def test_catalog_works_without_image_subsystem(self):
        """Catalog should work even when image generation is not enabled."""
        from unittest.mock import MagicMock

        from fastapi import FastAPI

        from augmentum.proxy.image_routes import router

        app = FastAPI()
        app.include_router(router)
        # Minimal state — no image_queue, no image_hardware
        app.state = MagicMock(spec=[])

        bare_client = TestClient(app)
        resp = bare_client.get("/api/image/models/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        assert all("repo_id" in m for m in data)
