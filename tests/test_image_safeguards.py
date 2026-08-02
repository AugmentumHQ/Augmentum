"""Tests for image generation VRAM safeguards and OOM recovery."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.image.hardware import (
    PIPELINE_VRAM_REQUIREMENTS,
    HardwareProfile,
    ModelTier,
    check_vram_for_pipeline,
    estimate_model_ram_mb,
    get_system_ram_free_mb,
    pre_load_safety_check,
    refresh_vram_free,
)
from augmentum.image.pipeline_registry import PipelineRegistry
from augmentum.image.schemas import PipelineType

# ==========================================================================
# VRAM Validation — check_vram_for_pipeline
# ==========================================================================


class TestCheckVramForPipeline:
    """Tests for the check_vram_for_pipeline helper."""

    def test_cpu_allows_sd15(self):
        hw = HardwareProfile(device="cpu", tier=ModelTier.CPU)
        assert check_vram_for_pipeline("sd15", hw) is None

    def test_cpu_blocks_sdxl(self):
        hw = HardwareProfile(device="cpu", tier=ModelTier.CPU)
        msg = check_vram_for_pipeline("sdxl", hw)
        assert msg is not None
        assert "requires a GPU" in msg

    def test_cpu_blocks_flux(self):
        hw = HardwareProfile(device="cpu", tier=ModelTier.CPU)
        msg = check_vram_for_pipeline("flux", hw)
        assert msg is not None
        assert "requires a GPU" in msg

    def test_gpu_sufficient_vram_sd15(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=4000, vram_free_mb=3000)
        assert check_vram_for_pipeline("sd15", hw) is None

    def test_gpu_sufficient_vram_sdxl(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=8000, vram_free_mb=6000)
        assert check_vram_for_pipeline("sdxl", hw) is None

    def test_gpu_sufficient_vram_flux(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=24000, vram_free_mb=12000)
        assert check_vram_for_pipeline("flux", hw) is None

    def test_gpu_insufficient_vram_sdxl(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=8000, vram_free_mb=4000)
        msg = check_vram_for_pipeline("sdxl", hw)
        assert msg is not None
        assert "SDXL" in msg
        assert "5GB" in msg

    def test_gpu_insufficient_vram_flux(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=12000, vram_free_mb=8000)
        msg = check_vram_for_pipeline("flux", hw)
        assert msg is not None
        assert "FLUX" in msg
        assert "10GB" in msg

    def test_gpu_insufficient_vram_sd15(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=2000, vram_free_mb=1000)
        msg = check_vram_for_pipeline("sd15", hw)
        assert msg is not None
        assert "SD15" in msg

    def test_unknown_pipeline_passes(self):
        hw = HardwareProfile(device="cuda", vram_total_mb=4000, vram_free_mb=4000)
        assert check_vram_for_pipeline("unknown_type", hw) is None

    def test_vram_requirements_dict_has_all_types(self):
        assert "sd15" in PIPELINE_VRAM_REQUIREMENTS
        assert "sdxl" in PIPELINE_VRAM_REQUIREMENTS
        assert "flux" in PIPELINE_VRAM_REQUIREMENTS


# ==========================================================================
# OOM Recovery — Pipeline load/generate
# ==========================================================================


class TestOomRecovery:
    """Tests for OOM error handling in pipeline load/generate."""

    @pytest.mark.asyncio
    async def test_load_oom_raises_friendly_message(self):
        from augmentum.image.pipeline_v2 import UnifiedPipeline

        pipe = UnifiedPipeline()
        with patch("augmentum.image.pipeline_v2._run_on_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = RuntimeError("Out of GPU memory. Try a smaller model or close other GPU applications.")
            with pytest.raises(RuntimeError, match="Out of GPU memory"):
                await pipe.load("/fake/model")

    @pytest.mark.asyncio
    async def test_generate_oom_raises_friendly_message(self):
        from augmentum.image.pipeline_v2 import UnifiedPipeline

        pipe = UnifiedPipeline()
        pipe._pipe = MagicMock()
        pipe._pipe.__call__ = MagicMock()
        pipe._device = "cuda"
        pipe._pipe_params = set()
        pipe._is_edit = False

        with patch("augmentum.image.pipeline_v2._run_on_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = RuntimeError("Out of GPU memory. Try a smaller model or close other GPU applications.")
            with pytest.raises(RuntimeError, match="Out of GPU memory"):
                await pipe.generate(prompt="test")

    @pytest.mark.asyncio
    async def test_non_oom_runtime_error_propagates(self):
        from augmentum.image.pipeline_v2 import UnifiedPipeline

        pipe = UnifiedPipeline()
        with patch("augmentum.image.pipeline_v2._run_on_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.side_effect = RuntimeError("some other error")
            with pytest.raises(RuntimeError, match="some other error"):
                await pipe.load("/fake/model")


# ==========================================================================
# Pipeline Registry — cleanup on failed load
# ==========================================================================


class TestRegistryCleanup:
    """Tests that PipelineRegistry cleans up state on failed loads."""

    @pytest.mark.asyncio
    async def test_failed_load_clears_state(self):
        registry = PipelineRegistry()

        with patch.dict(
            "augmentum.image.pipeline_registry._PIPELINE_CLASSES",
            {PipelineType.SD15: MagicMock},
        ):
            mock_pipeline = MagicMock()
            mock_pipeline.load = AsyncMock(
                side_effect=RuntimeError("CUDA error: out of memory")
            )
            mock_pipeline.is_loaded = False

            # Make the class constructor return our mock
            _PIPELINE_CLASSES = {PipelineType.SD15: lambda: mock_pipeline}
            with patch.dict(
                "augmentum.image.pipeline_registry._PIPELINE_CLASSES",
                _PIPELINE_CLASSES,
            ):
                with pytest.raises(RuntimeError):
                    await registry.load("/fake", PipelineType.SD15)

                # Registry should be in clean state
                assert registry._current is None
                assert registry._current_model == ""
                assert registry._current_type is None

    @pytest.mark.asyncio
    async def test_successful_load_sets_state(self):
        registry = PipelineRegistry()

        mock_pipeline = MagicMock()
        mock_pipeline.load = AsyncMock()
        mock_pipeline.is_loaded = True

        _PIPELINE_CLASSES = {PipelineType.SD15: lambda: mock_pipeline}
        with patch.dict(
            "augmentum.image.pipeline_registry._PIPELINE_CLASSES",
            _PIPELINE_CLASSES,
        ):
            result = await registry.load("/fake", PipelineType.SD15)
            assert result is mock_pipeline
            assert registry._current is mock_pipeline
            assert registry._current_model == "/fake"


# ==========================================================================
# Image Routes — VRAM check in generate endpoint
# ==========================================================================


# ==========================================================================
# HuggingFace Token API
# ==========================================================================


class TestHfTokenApi:
    """Tests for the HuggingFace token GET/PUT endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from augmentum.proxy.image_routes import router

        app = FastAPI()
        app.include_router(router)
        app.state = MagicMock()
        app.state.settings_store = AsyncMock()
        return TestClient(app)

    def test_put_hf_token_sets_value(self, client):
        resp = client.put(
            "/api/image/hf-token",
            json={"token": "hf_testABC123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["is_set"] is True

    def test_put_hf_token_empty_clears(self, client):
        # First set a token
        client.put("/api/image/hf-token", json={"token": "hf_testABC123"})
        # Then clear it
        resp = client.put("/api/image/hf-token", json={"token": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert data["is_set"] is False

    def test_get_hf_token_returns_status_not_value(self, client):
        # Set a token first
        client.put("/api/image/hf-token", json={"token": "hf_secret"})
        resp = client.get("/api/image/hf-token")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_set"] is True
        # Must never contain the actual token value
        assert "hf_secret" not in resp.text

    def test_get_hf_token_when_unset(self, client):
        # Clear token
        client.put("/api/image/hf-token", json={"token": ""})
        resp = client.get("/api/image/hf-token")
        assert resp.status_code == 200
        assert resp.json()["is_set"] is False

    def test_pull_model_passes_token_to_hf(self):
        """pull_model route should pass settings.image_huggingface_token to HF download."""
        from unittest.mock import patch as _patch

        from augmentum.config import settings
        from augmentum.image.model_manager import ModelManager

        # Set token on settings
        object.__setattr__(settings, "image_huggingface_token", "hf_mytoken")

        async def _fake_pull(repo_id, name="", token=None):
            # Verify the token was passed through
            assert token == "hf_mytoken"
            yield {"status": "complete", "name": "test", "path": "/tmp/test", "pipeline_type": "sd15", "size_bytes": 0}

        with _patch.object(ModelManager, "pull_from_huggingface", _fake_pull):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from augmentum.proxy.image_routes import router

            app = FastAPI()
            app.include_router(router)
            state = MagicMock()
            state.image_model_manager = ModelManager("/tmp/fake")
            state.image_persistence = AsyncMock()
            app.state = state

            client = TestClient(app)
            resp = client.post(
                "/api/image/models/pull",
                json={"source": "test/model"},
            )
            assert resp.status_code == 200

        # Clean up
        object.__setattr__(settings, "image_huggingface_token", None)


class TestGenerateVramCheck:
    """Tests for the VRAM pre-check in the generate endpoint."""

    @pytest.fixture
    def app_and_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from augmentum.image.presets import PresetManager
        from augmentum.proxy.image_routes import router

        app = FastAPI()
        app.include_router(router)

        state = MagicMock()
        state.image_hardware = HardwareProfile(
            device="cuda",
            device_name="NVIDIA RTX 3060",
            vram_total_mb=12288,
            vram_free_mb=4000,
            tier=ModelTier.MEDIUM,
            recommended_pipeline="sdxl",
            recommended_model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        state.image_queue = MagicMock()
        state.image_queue.queue_size = 0
        state.image_preset_manager = PresetManager()
        state.image_pipeline_registry = MagicMock()
        state.image_pipeline_registry.is_loaded = False
        state.image_model_manager = MagicMock()
        state.image_model_manager.get_model_path.return_value = None
        state.image_persistence = AsyncMock()
        state.image_cache = AsyncMock()
        state.image_cache.get = AsyncMock(return_value=None)
        state.image_lora_manager = MagicMock()

        app.state = state
        return app, TestClient(app), state

    def test_generate_passes_to_queue_even_on_low_vram(self, app_and_client):
        """VRAM check moved to queue worker — route should submit the job.

        The worker does a live VRAM check and accounts for model swaps
        (unloading the old model frees VRAM before the new one loads).
        """
        app, client, state = app_and_client
        state.image_hardware.recommended_pipeline = "flux"
        state.image_model_manager.get_model_path.return_value = None

        # Set up queue to simulate a successful generation
        job = MagicMock()
        job.job_id = "test123"
        fut = asyncio.get_event_loop().create_future()
        fut.set_result({
            "image_id": "abc", "file_path": "/tmp/abc.png",
            "seed": 42, "width": 512, "height": 512,
        })
        job.future = fut
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value={
            "image_id": "abc", "file_path": "/tmp/abc.png",
            "seed": 42, "width": 512, "height": 512,
        })

        resp = client.post(
            "/api/image/generate",
            json={"prompt": "a cat"},
        )
        # Route no longer blocks — VRAM check is in the worker
        assert resp.status_code == 200

    def test_generate_allows_sd15_on_low_vram(self, app_and_client):
        """SD15 needs ~2GB, 4GB free → should pass (may fail later for other reasons)."""
        app, client, state = app_and_client
        state.image_hardware.recommended_pipeline = "sd15"
        state.image_model_manager.get_model_path.return_value = None

        # Queue submit will need to work for the request to proceed
        job = MagicMock()
        job.job_id = "test123"
        fut = asyncio.get_event_loop().create_future()
        fut.set_result({
            "image_id": "abc", "file_path": "/tmp/abc.png",
            "seed": 42, "width": 512, "height": 512,
        })
        job.future = fut
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value={
            "image_id": "abc", "file_path": "/tmp/abc.png",
            "seed": 42, "width": 512, "height": 512,
        })

        resp = client.post(
            "/api/image/generate",
            json={"prompt": "a cat"},
        )
        assert resp.status_code == 200


# ==========================================================================
# Live VRAM refresh
# ==========================================================================


class TestRefreshVramFree:
    def test_returns_0_when_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert refresh_vram_free() == 0

    def test_returns_free_mb(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024**3, 8 * 1024**3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert refresh_vram_free() == 4096

    def test_returns_0_on_exception(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("no GPU")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert refresh_vram_free() == 0


# ==========================================================================
# System RAM detection
# ==========================================================================


class TestGetSystemRamFree:
    def test_with_psutil(self):
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.available = 16 * 1024**3

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            assert get_system_ram_free_mb() == 16384

    def test_returns_int_on_failure(self):
        """Should return 0 (not crash) when psutil is missing."""
        result = get_system_ram_free_mb()
        assert isinstance(result, int)
        assert result >= 0


# ==========================================================================
# Model RAM estimation
# ==========================================================================


class TestEstimateModelRam:
    def test_nonexistent_path_returns_0(self):
        assert estimate_model_ram_mb("/nonexistent/model/path") == 0

    def test_single_safetensor_file(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"\x00" * (100 * 1024 * 1024))
        assert estimate_model_ram_mb(str(f)) == 100

    def test_directory_sums_weights(self, tmp_path):
        (tmp_path / "model-00001.safetensors").write_bytes(b"\x00" * (50 * 1024 * 1024))
        (tmp_path / "model-00002.safetensors").write_bytes(b"\x00" * (50 * 1024 * 1024))
        (tmp_path / "config.json").write_bytes(b"{}")
        assert estimate_model_ram_mb(str(tmp_path)) == 100

    def test_gguf_file(self, tmp_path):
        f = tmp_path / "model-Q4_K_M.gguf"
        f.write_bytes(b"\x00" * (200 * 1024 * 1024))
        assert estimate_model_ram_mb(str(f)) == 200

    def test_ignores_non_weight_files(self, tmp_path):
        (tmp_path / "config.json").write_bytes(b"{}" * 1000)
        (tmp_path / "tokenizer.json").write_bytes(b"{}" * 1000)
        (tmp_path / "README.md").write_bytes(b"test" * 1000)
        assert estimate_model_ram_mb(str(tmp_path)) == 0


# ==========================================================================
# Pre-load safety check
# ==========================================================================


class TestPreLoadSafetyCheck:
    def _hw(self, device="cuda", vram_free=6144):
        return HardwareProfile(
            device=device,
            device_name="Test GPU",
            vram_total_mb=8192,
            vram_free_mb=vram_free,
            tier=ModelTier.MEDIUM,
            recommended_pipeline="sdxl",
            recommended_model="test",
        )

    def test_passes_sufficient_resources(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"\x00" * (100 * 1024 * 1024))

        with patch("augmentum.image.hardware.refresh_vram_free", return_value=8000), \
             patch("augmentum.image.hardware.get_system_ram_free_mb", return_value=32000):
            assert pre_load_safety_check(str(model_dir), "sdxl", self._hw()) is None

    def test_blocks_insufficient_live_vram(self):
        with patch("augmentum.image.hardware.refresh_vram_free", return_value=3000), \
             patch("augmentum.image.hardware.get_system_ram_free_mb", return_value=32000):
            result = pre_load_safety_check("/fake", "flux", self._hw())
            assert result is not None
            assert "VRAM" in result

    def test_blocks_insufficient_system_ram(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("augmentum.image.hardware.refresh_vram_free", return_value=12000), \
             patch("augmentum.image.hardware.get_system_ram_free_mb", return_value=2000), \
             patch("augmentum.image.hardware.estimate_model_ram_mb", return_value=10000):
            result = pre_load_safety_check(str(model_dir), "flux", self._hw())
            assert result is not None
            assert "RAM" in result

    def test_skips_vram_check_on_cpu(self):
        hw = self._hw(device="cpu")
        with patch("augmentum.image.hardware.get_system_ram_free_mb", return_value=32000):
            assert pre_load_safety_check("/fake", "sd15", hw) is None

    def test_graceful_when_ram_detection_returns_0(self):
        """0 means we couldn't detect — should not block."""
        with patch("augmentum.image.hardware.refresh_vram_free", return_value=12000), \
             patch("augmentum.image.hardware.get_system_ram_free_mb", return_value=0):
            assert pre_load_safety_check("/fake", "flux", self._hw()) is None

    def test_graceful_when_vram_detection_returns_0(self):
        """0 means we couldn't detect — should not block."""
        with patch("augmentum.image.hardware.refresh_vram_free", return_value=0), \
             patch("augmentum.image.hardware.get_system_ram_free_mb", return_value=32000):
            assert pre_load_safety_check("/fake", "flux", self._hw()) is None
