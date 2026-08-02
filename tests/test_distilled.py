"""Tests for distilled model detection (10.6) and negative prompt defaults (10.7)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.image.defaults import PIPELINE_NEGATIVE_DEFAULTS, resolve_negative_prompt
from augmentum.image.distilled import (
    DISTILLED_PATTERNS,
    apply_distilled_defaults,
    detect_distilled_type,
    get_distilled_defaults,
)

# ---------------------------------------------------------------------------
# 10.6 — Distilled model detection
# ---------------------------------------------------------------------------


class TestDetectDistilledType:
    """detect_distilled_type should identify distilled model variants."""

    def test_turbo_model(self):
        assert detect_distilled_type("DreamShaper-XL-Turbo") == "turbo"

    def test_lightning_model(self):
        assert detect_distilled_type("sdxl-lightning-4step") == "lightning"

    def test_lcm_model(self):
        assert detect_distilled_type("lcm-dreamshaper-v7") == "lcm"

    def test_hyper_sd_model(self):
        assert detect_distilled_type("Hyper-SD-v1") == "hyper"

    def test_hyper_sd_underscore(self):
        assert detect_distilled_type("hyper_sd_v1") == "hyper"

    def test_hypersd_no_separator(self):
        assert detect_distilled_type("my-hypersd-model") == "hyper"

    def test_non_distilled_model(self):
        assert detect_distilled_type("stable-diffusion-xl-base-1.0") is None

    def test_case_insensitivity(self):
        assert detect_distilled_type("SDXL-TURBO") == "turbo"
        assert detect_distilled_type("Sdxl-Lightning") == "lightning"
        assert detect_distilled_type("LCM-Lora") == "lcm"
        assert detect_distilled_type("HYPER-SD-xl") == "hyper"

    def test_path_style_names(self):
        assert detect_distilled_type("models/lcm-lora-sdxl") == "lcm"
        assert detect_distilled_type("/data/models/sdxl-turbo-v2") == "turbo"
        assert detect_distilled_type("C:\\models\\lightning-sdxl") == "lightning"

    def test_empty_string(self):
        assert detect_distilled_type("") is None

    def test_plain_model_name(self):
        assert detect_distilled_type("runwayml--stable-diffusion-v1-5") is None

    def test_flux_not_distilled(self):
        assert detect_distilled_type("FLUX.1-schnell") is None


class TestGetDistilledDefaults:
    """get_distilled_defaults should return appropriate step/cfg values."""

    def test_turbo_defaults(self):
        defaults = get_distilled_defaults("turbo")
        assert defaults == {"steps": 4, "cfg_scale": 2.0}

    def test_lightning_defaults(self):
        defaults = get_distilled_defaults("lightning")
        assert defaults == {"steps": 4, "cfg_scale": 2.0}

    def test_lcm_defaults(self):
        defaults = get_distilled_defaults("lcm")
        assert defaults == {"steps": 8, "cfg_scale": 1.5}

    def test_hyper_defaults(self):
        defaults = get_distilled_defaults("hyper")
        assert defaults == {"steps": 4, "cfg_scale": 1.5}

    def test_none_returns_empty(self):
        assert get_distilled_defaults(None) == {}

    def test_unknown_returns_empty(self):
        assert get_distilled_defaults("unknown_type") == {}


class TestApplyDistilledDefaults:
    """apply_distilled_defaults should merge user overrides with auto-detection."""

    def test_distilled_no_overrides(self):
        steps, cfg = apply_distilled_defaults("sdxl-turbo-v2", None, None)
        assert steps == 4
        assert cfg == 2.0

    def test_distilled_user_steps_override(self):
        steps, cfg = apply_distilled_defaults("sdxl-turbo-v2", 6, None)
        assert steps == 6
        assert cfg == 2.0  # distilled default

    def test_distilled_user_cfg_override(self):
        steps, cfg = apply_distilled_defaults("sdxl-turbo-v2", None, 3.5)
        assert steps == 4  # distilled default
        assert cfg == 3.5

    def test_distilled_both_overrides(self):
        steps, cfg = apply_distilled_defaults("lightning-sdxl", 10, 5.0)
        assert steps == 10
        assert cfg == 5.0

    def test_non_distilled_no_overrides(self):
        steps, cfg = apply_distilled_defaults("stable-diffusion-v1-5", None, None)
        assert steps == 20
        assert cfg == 7.0

    def test_non_distilled_with_overrides(self):
        steps, cfg = apply_distilled_defaults("stable-diffusion-v1-5", 30, 9.0)
        assert steps == 30
        assert cfg == 9.0

    def test_lcm_defaults_applied(self):
        steps, cfg = apply_distilled_defaults("lcm-dreamshaper-v7", None, None)
        assert steps == 8
        assert cfg == 1.5

    def test_hyper_defaults_applied(self):
        steps, cfg = apply_distilled_defaults("Hyper-SD-v1", None, None)
        assert steps == 4
        assert cfg == 1.5


# ---------------------------------------------------------------------------
# 10.7 — Negative prompt defaults
# ---------------------------------------------------------------------------


class TestResolveNegativePrompt:
    """resolve_negative_prompt should follow explicit > config > pipeline priority."""

    def test_explicit_prompt_returned_unchanged(self):
        result = resolve_negative_prompt("my custom negative", "sd15")
        assert result == "my custom negative"

    def test_config_default_used_when_no_explicit(self):
        result = resolve_negative_prompt("", "sd15", config_default="config negatives")
        assert result == "config negatives"

    def test_pipeline_default_when_no_explicit_no_config(self):
        result = resolve_negative_prompt("", "sd15")
        assert result == PIPELINE_NEGATIVE_DEFAULTS["sd15"]
        assert "worst quality" in result

    def test_flux_returns_empty(self):
        result = resolve_negative_prompt("", "flux")
        assert result == ""

    def test_sd15_returns_quality_defaults(self):
        result = resolve_negative_prompt("", "sd15")
        assert "worst quality" in result
        assert "blurry" in result
        assert "watermark" in result

    def test_sdxl_returns_quality_defaults(self):
        result = resolve_negative_prompt("", "sdxl")
        assert "worst quality" in result
        assert "blurry" in result

    def test_explicit_overrides_config(self):
        result = resolve_negative_prompt(
            "explicit negative", "sd15", config_default="config negative",
        )
        assert result == "explicit negative"

    def test_config_overrides_pipeline(self):
        result = resolve_negative_prompt("", "sd15", config_default="just bad quality")
        assert result == "just bad quality"

    def test_unknown_pipeline_empty(self):
        result = resolve_negative_prompt("", "unknown_pipe")
        assert result == ""

    def test_explicit_takes_priority_even_for_flux(self):
        """Even FLUX should respect an explicit negative prompt."""
        result = resolve_negative_prompt("low quality", "flux")
        assert result == "low quality"


# ---------------------------------------------------------------------------
# 10.6 — Distilled type in model listing
# ---------------------------------------------------------------------------


class TestDistilledInModelList:
    """list_local_models should include a distilled_type field."""

    def test_model_list_includes_distilled_type(self):
        from augmentum.image.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake model directory with a minimal indicator file
            model_dir = os.path.join(tmpdir, "sdxl-turbo-v2")
            os.makedirs(model_dir)
            # Write a dummy safetensors file so the scanner picks it up
            with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
                f.write(b"\x00" * 16)

            mgr = ModelManager(tmpdir)
            models = mgr.list_local_models()
            assert len(models) == 1
            assert models[0]["distilled_type"] == "turbo"

    def test_non_distilled_model_empty_string(self):
        from augmentum.image.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "stable-diffusion-v1-5")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
                f.write(b"\x00" * 16)

            mgr = ModelManager(tmpdir)
            models = mgr.list_local_models()
            assert len(models) == 1
            assert models[0]["distilled_type"] == ""

    def test_lightning_model_detected(self):
        from augmentum.image.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "sdxl-lightning-4step")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "model.safetensors"), "wb") as f:
                f.write(b"\x00" * 16)

            mgr = ModelManager(tmpdir)
            models = mgr.list_local_models()
            assert len(models) == 1
            assert models[0]["distilled_type"] == "lightning"


# ---------------------------------------------------------------------------
# 10.6 + 10.7 — Route integration
# ---------------------------------------------------------------------------


class TestDistilledInRoutes:
    """Integration: /api/image/generate should use distilled defaults."""

    @pytest.fixture
    def image_client(self):
        """Create a test client with mocked image subsystem."""
        from augmentum.image.hardware import HardwareProfile, ModelTier
        from augmentum.image.presets import PresetManager
        from augmentum.image.queue import GenerationQueue
        from augmentum.proxy.image_routes import router

        app = FastAPI()
        app.include_router(router)

        state = MagicMock()
        state.image_hardware = HardwareProfile(
            device="cuda",
            device_name="Test GPU",
            vram_total_mb=8192,
            vram_free_mb=6144,
            tier=ModelTier.MEDIUM,
            recommended_pipeline="sdxl",
            recommended_model="stabilityai/stable-diffusion-xl-base-1.0",
        )
        state.image_queue = MagicMock(spec=GenerationQueue)
        state.image_queue.queue_size = 0
        state.image_preset_manager = PresetManager()
        state.image_pipeline_registry = MagicMock()
        state.image_pipeline_registry.is_loaded = False
        state.image_pipeline_registry.current_model = ""
        state.image_model_manager = MagicMock()
        state.image_model_manager.list_local_models.return_value = []
        state.image_lora_manager = MagicMock()
        state.image_persistence = AsyncMock()
        state.image_cache = AsyncMock()
        state.image_cache.get = AsyncMock(return_value=None)

        app.state = state
        return TestClient(app), state

    def test_turbo_model_gets_distilled_defaults(self, image_client):
        """POST /api/image/generate with a turbo model uses 4 steps and CFG 2.0."""
        client, state = image_client

        # Capture the submitted job
        captured_job = None

        async def mock_submit(job):
            nonlocal captured_job
            captured_job = job
            return job

        async def mock_wait(job, timeout=None):
            return {"image_id": "test-img", "seed": 42}

        state.image_queue.submit = AsyncMock(side_effect=mock_submit)
        state.image_queue.wait_for_result = AsyncMock(side_effect=mock_wait)

        resp = client.post("/api/image/generate", json={
            "prompt": "a beautiful landscape",
            "model": "sdxl-turbo-v2",
        })
        assert resp.status_code == 200

        assert captured_job is not None
        assert captured_job.steps == 4
        assert captured_job.cfg_scale == 2.0

    def test_non_distilled_model_uses_standard_defaults(self, image_client):
        """Non-distilled models should use 20 steps / 7.0 CFG."""
        client, state = image_client

        captured_job = None

        async def mock_submit(job):
            nonlocal captured_job
            captured_job = job
            return job

        async def mock_wait(job, timeout=None):
            return {"image_id": "test-img", "seed": 42}

        state.image_queue.submit = AsyncMock(side_effect=mock_submit)
        state.image_queue.wait_for_result = AsyncMock(side_effect=mock_wait)

        resp = client.post("/api/image/generate", json={
            "prompt": "a beautiful landscape",
            "model": "stable-diffusion-v1-5",
        })
        assert resp.status_code == 200

        assert captured_job is not None
        assert captured_job.steps == 20
        assert captured_job.cfg_scale == 7.0

    def test_user_override_respected_for_distilled(self, image_client):
        """User-specified steps/cfg should override distilled defaults."""
        client, state = image_client

        captured_job = None

        async def mock_submit(job):
            nonlocal captured_job
            captured_job = job
            return job

        async def mock_wait(job, timeout=None):
            return {"image_id": "test-img", "seed": 42}

        state.image_queue.submit = AsyncMock(side_effect=mock_submit)
        state.image_queue.wait_for_result = AsyncMock(side_effect=mock_wait)

        resp = client.post("/api/image/generate", json={
            "prompt": "a beautiful landscape",
            "model": "sdxl-turbo-v2",
            "steps": 10,
            "cfg_scale": 5.0,
        })
        assert resp.status_code == 200

        assert captured_job is not None
        assert captured_job.steps == 10
        assert captured_job.cfg_scale == 5.0

    def test_negative_prompt_pipeline_default_applied(self, image_client):
        """Empty negative prompt should get pipeline default for SDXL."""
        client, state = image_client

        captured_job = None

        async def mock_submit(job):
            nonlocal captured_job
            captured_job = job
            return job

        async def mock_wait(job, timeout=None):
            return {"image_id": "test-img", "seed": 42}

        state.image_queue.submit = AsyncMock(side_effect=mock_submit)
        state.image_queue.wait_for_result = AsyncMock(side_effect=mock_wait)

        resp = client.post("/api/image/generate", json={
            "prompt": "a beautiful landscape",
            "model": "stable-diffusion-v1-5",
        })
        assert resp.status_code == 200

        assert captured_job is not None
        assert "worst quality" in captured_job.negative_prompt

    def test_explicit_negative_prompt_not_overridden(self, image_client):
        """User-provided negative prompt should not be overridden."""
        client, state = image_client

        captured_job = None

        async def mock_submit(job):
            nonlocal captured_job
            captured_job = job
            return job

        async def mock_wait(job, timeout=None):
            return {"image_id": "test-img", "seed": 42}

        state.image_queue.submit = AsyncMock(side_effect=mock_submit)
        state.image_queue.wait_for_result = AsyncMock(side_effect=mock_wait)

        resp = client.post("/api/image/generate", json={
            "prompt": "a beautiful landscape",
            "model": "stable-diffusion-v1-5",
            "negative_prompt": "my custom negatives only",
        })
        assert resp.status_code == 200

        assert captured_job is not None
        assert captured_job.negative_prompt == "my custom negatives only"

    def test_models_list_includes_distilled_type(self, image_client):
        """GET /api/image/models should include distilled_type field."""
        client, state = image_client

        state.image_model_manager.list_local_models.return_value = [
            {
                "name": "sdxl-turbo-v2",
                "path": "/data/models/sdxl-turbo-v2",
                "pipeline_type": "sdxl",
                "size_bytes": 6_000_000_000,
                "source": "local",
                "distilled_type": "turbo",
            },
            {
                "name": "sd-v1-5",
                "path": "/data/models/sd-v1-5",
                "pipeline_type": "sd15",
                "size_bytes": 4_000_000_000,
                "source": "local",
                "distilled_type": "",
            },
        ]

        resp = client.get("/api/image/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["distilled_type"] == "turbo"
        assert data[1]["distilled_type"] == ""


# ---------------------------------------------------------------------------
# Pattern coverage
# ---------------------------------------------------------------------------


class TestDistilledPatterns:
    """Verify the DISTILLED_PATTERNS dict is well-formed."""

    def test_all_types_have_required_keys(self):
        for name, info in DISTILLED_PATTERNS.items():
            assert "steps" in info, f"{name} missing steps"
            assert "cfg_scale" in info, f"{name} missing cfg_scale"
            assert "patterns" in info, f"{name} missing patterns"
            assert isinstance(info["patterns"], list)
            assert len(info["patterns"]) > 0

    def test_all_steps_are_positive(self):
        for name, info in DISTILLED_PATTERNS.items():
            assert info["steps"] > 0, f"{name} has non-positive steps"

    def test_all_cfg_scales_are_non_negative(self):
        for name, info in DISTILLED_PATTERNS.items():
            assert info["cfg_scale"] >= 0, f"{name} has negative cfg_scale"
