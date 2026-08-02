"""Tests for sampler/scheduler selection across the image generation stack."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.image.presets import BUILTIN_PRESETS, GenrePreset, PresetManager
from augmentum.image.queue import GenerationJob
from augmentum.image.schedulers import (
    SAMPLER_ALIASES,
    SAMPLER_MAP,
    _resolve_alias,
    apply_sampler,
    get_available_samplers,
)
from augmentum.image.schemas import GenerateRequest, OpenAIImageRequest, SamplerInfo

# ---------------------------------------------------------------------------
# A) Scheduler mapping (schedulers.py)
# ---------------------------------------------------------------------------


class TestSchedulerMapping:
    """Verify SAMPLER_MAP entries and alias resolution."""

    def test_all_canonical_names_are_strings(self):
        for name, (cls_name, kwargs) in SAMPLER_MAP.items():
            assert isinstance(name, str)
            assert isinstance(cls_name, str)
            assert isinstance(kwargs, dict)

    def test_sampler_map_has_expected_entries(self):
        expected = {
            "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras",
            "dpm++_2m_sde", "dpm++_2m_sde_karras", "ddim", "lms",
            "unipc", "lcm", "heun", "deis", "pndm",
        }
        assert expected == set(SAMPLER_MAP.keys())

    def test_karras_variants_have_use_karras_sigmas(self):
        for name in ("dpm++_2m_karras", "dpm++_2m_sde_karras"):
            _, kwargs = SAMPLER_MAP[name]
            assert kwargs.get("use_karras_sigmas") is True

    def test_sde_variants_have_algorithm_type(self):
        for name in ("dpm++_2m_sde", "dpm++_2m_sde_karras"):
            _, kwargs = SAMPLER_MAP[name]
            assert "algorithm_type" in kwargs

    def test_alias_resolution_euler_ancestral(self):
        assert _resolve_alias("euler_ancestral") == "euler_a"

    def test_alias_resolution_dpmpp(self):
        assert _resolve_alias("dpmpp_2m") == "dpm++_2m"

    def test_alias_resolution_k_dpm(self):
        assert _resolve_alias("k_dpm_2m") == "dpm++_2m"

    def test_alias_resolution_case_insensitive(self):
        assert _resolve_alias("EULER_ANCESTRAL") == "euler_a"
        assert _resolve_alias("Dpmpp_2m") == "dpm++_2m"

    def test_alias_resolution_passthrough(self):
        """Canonical names should pass through unchanged."""
        assert _resolve_alias("euler") == "euler"
        assert _resolve_alias("ddim") == "ddim"

    def test_all_aliases_resolve_to_valid_canonical(self):
        for alias, canonical in SAMPLER_ALIASES.items():
            assert canonical in SAMPLER_MAP, (
                f"Alias '{alias}' -> '{canonical}' not in SAMPLER_MAP"
            )

    def test_apply_sampler_with_each_name(self):
        """apply_sampler should call diffusers import + from_config for each sampler."""
        for name in SAMPLER_MAP:
            mock_pipe = MagicMock()
            scheduler_config = {"beta_start": 0.0001}
            mock_pipe.scheduler.config = scheduler_config

            cls_name, kwargs = SAMPLER_MAP[name]
            mock_scheduler_cls = MagicMock()
            mock_scheduler_instance = MagicMock()
            mock_scheduler_cls.from_config.return_value = mock_scheduler_instance

            mock_diffusers = MagicMock()
            setattr(mock_diffusers, cls_name, mock_scheduler_cls)

            with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
                apply_sampler(mock_pipe, name)

            mock_scheduler_cls.from_config.assert_called_once_with(
                scheduler_config, **kwargs,
            )
            assert mock_pipe.scheduler == mock_scheduler_instance

    def test_apply_sampler_unknown_raises_valueerror(self):
        mock_pipe = MagicMock()
        with pytest.raises(ValueError, match="Unknown sampler"):
            apply_sampler(mock_pipe, "nonexistent_sampler_xyz")

    def test_apply_sampler_resolves_alias(self):
        """Passing an alias should resolve to the canonical sampler."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler.config = {}

        cls_name, kwargs = SAMPLER_MAP["euler_a"]
        mock_scheduler_cls = MagicMock()
        mock_diffusers = MagicMock()
        setattr(mock_diffusers, cls_name, mock_scheduler_cls)

        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            apply_sampler(mock_pipe, "euler_ancestral")

        mock_scheduler_cls.from_config.assert_called_once()

    def test_apply_sampler_whitespace_stripped(self):
        mock_pipe = MagicMock()
        mock_pipe.scheduler.config = {}

        mock_scheduler_cls = MagicMock()
        mock_diffusers = MagicMock()
        mock_diffusers.EulerDiscreteScheduler = mock_scheduler_cls

        with patch.dict("sys.modules", {"diffusers": mock_diffusers}):
            apply_sampler(mock_pipe, "  euler  ")

        mock_scheduler_cls.from_config.assert_called_once()


# ---------------------------------------------------------------------------
# B) Sampler list (get_available_samplers)
# ---------------------------------------------------------------------------


class TestSamplerList:
    def test_returns_list(self):
        result = get_available_samplers()
        assert isinstance(result, list)

    def test_returns_non_empty(self):
        result = get_available_samplers()
        assert len(result) == len(SAMPLER_MAP)

    def test_each_entry_has_required_keys(self):
        for entry in get_available_samplers():
            assert "name" in entry
            assert "display_name" in entry
            assert "aliases" in entry
            assert isinstance(entry["aliases"], list)

    def test_names_match_sampler_map(self):
        names = {e["name"] for e in get_available_samplers()}
        assert names == set(SAMPLER_MAP.keys())

    def test_euler_a_has_aliases(self):
        for entry in get_available_samplers():
            if entry["name"] == "euler_a":
                assert "euler_ancestral" in entry["aliases"]
                break
        else:
            pytest.fail("euler_a not found in available samplers")

    def test_display_names_are_human_readable(self):
        for entry in get_available_samplers():
            # Display name should not be empty
            assert entry["display_name"]
            # Display name should be title-case-ish (at least first char upper)
            # or an acronym like DDIM/LMS/PNDM
            assert entry["display_name"][0].isupper() or entry["display_name"].isupper()


# ---------------------------------------------------------------------------
# C) Schema fields
# ---------------------------------------------------------------------------


class TestSamplerInSchemas:
    def test_generate_request_accepts_sampler(self):
        req = GenerateRequest(prompt="test", sampler="euler_a")
        assert req.sampler == "euler_a"

    def test_generate_request_accepts_scheduler(self):
        req = GenerateRequest(prompt="test", scheduler="ddim")
        assert req.scheduler == "ddim"

    def test_generate_request_sampler_default_none(self):
        req = GenerateRequest(prompt="test")
        assert req.sampler is None
        assert req.scheduler is None

    def test_openai_image_request_accepts_sampler(self):
        req = OpenAIImageRequest(prompt="test", sampler="euler")
        assert req.sampler == "euler"

    def test_openai_image_request_accepts_scheduler(self):
        req = OpenAIImageRequest(prompt="test", scheduler="lms")
        assert req.scheduler == "lms"

    def test_openai_image_request_default_none(self):
        req = OpenAIImageRequest(prompt="test")
        assert req.sampler is None
        assert req.scheduler is None

    def test_sampler_info_model(self):
        info = SamplerInfo(name="euler", display_name="Euler", aliases=["e"])
        assert info.name == "euler"
        assert info.display_name == "Euler"
        assert info.aliases == ["e"]

    def test_sampler_info_default_aliases(self):
        info = SamplerInfo(name="ddim", display_name="DDIM")
        assert info.aliases == []


# ---------------------------------------------------------------------------
# D) Queue fields
# ---------------------------------------------------------------------------


class TestSamplerInQueue:
    def test_generation_job_has_sampler(self):
        job = GenerationJob(sampler="euler_a")
        assert job.sampler == "euler_a"

    def test_generation_job_has_scheduler(self):
        job = GenerationJob(scheduler="ddim")
        assert job.scheduler == "ddim"

    def test_generation_job_defaults_empty(self):
        job = GenerationJob()
        assert job.sampler == ""
        assert job.scheduler == ""

    def test_generation_job_sampler_preserved_through_creation(self):
        job = GenerationJob(
            prompt="test",
            sampler="dpm++_2m_karras",
            scheduler="dpm++_2m_karras",
        )
        assert job.sampler == "dpm++_2m_karras"
        assert job.scheduler == "dpm++_2m_karras"


# ---------------------------------------------------------------------------
# E) Endpoint (GET /api/image/samplers)
# ---------------------------------------------------------------------------


class TestSamplerEndpoint:
    @pytest.fixture
    def client(self):
        from augmentum.proxy.image_routes import router

        app = FastAPI()
        app.include_router(router)

        # Minimal state that satisfies _require_image guard
        state = MagicMock()
        state.image_queue = MagicMock()
        state.image_preset_manager = PresetManager()
        state.image_pipeline_registry = MagicMock()
        state.image_pipeline_registry.is_loaded = False
        state.image_model_manager = MagicMock()
        state.image_model_manager.list_local_models.return_value = []
        state.image_lora_manager = MagicMock()
        state.image_lora_manager.discover.return_value = []
        state.image_persistence = AsyncMock()
        state.image_cache = AsyncMock()
        state.image_hardware = MagicMock()
        state.image_hardware.device = "cpu"
        state.image_hardware.tier = MagicMock()
        state.image_hardware.tier.value = "cpu"
        app.state = state

        return TestClient(app)

    def test_list_samplers_returns_200(self, client):
        resp = client.get("/api/image/samplers")
        assert resp.status_code == 200

    def test_list_samplers_non_empty(self, client):
        resp = client.get("/api/image/samplers")
        data = resp.json()
        assert len(data) > 0

    def test_list_samplers_has_euler(self, client):
        resp = client.get("/api/image/samplers")
        data = resp.json()
        names = [s["name"] for s in data]
        assert "euler" in names

    def test_list_samplers_structure(self, client):
        resp = client.get("/api/image/samplers")
        data = resp.json()
        for entry in data:
            assert "name" in entry
            assert "display_name" in entry
            assert "aliases" in entry


# ---------------------------------------------------------------------------
# F) Presets
# ---------------------------------------------------------------------------


class TestSamplerInPresets:
    def test_genre_preset_has_sampler_field(self):
        preset = GenrePreset(name="test", display_name="Test")
        assert hasattr(preset, "sampler")
        assert preset.sampler == ""

    def test_genre_preset_has_scheduler_field(self):
        preset = GenrePreset(name="test", display_name="Test")
        assert hasattr(preset, "scheduler")
        assert preset.scheduler == ""

    def test_genre_preset_accepts_sampler(self):
        preset = GenrePreset(name="test", display_name="Test", sampler="euler_a")
        assert preset.sampler == "euler_a"

    def test_builtin_realism_has_sampler(self):
        preset = BUILTIN_PRESETS["realism"]
        assert preset.sampler == "dpm++_2m_karras"

    def test_builtin_anime_has_sampler(self):
        preset = BUILTIN_PRESETS["anime"]
        assert preset.sampler == "euler_a"

    def test_builtin_fantasy_rpg_has_sampler(self):
        preset = BUILTIN_PRESETS["fantasy_rpg"]
        assert preset.sampler == "dpm++_2m_karras"

    def test_builtin_scifi_has_sampler(self):
        preset = BUILTIN_PRESETS["scifi"]
        assert preset.sampler == "dpm++_2m_sde_karras"

    def test_builtin_horror_has_sampler(self):
        preset = BUILTIN_PRESETS["horror"]
        assert preset.sampler == "dpm++_2m_sde_karras"

    def test_all_builtin_presets_have_sampler(self):
        for name, preset in BUILTIN_PRESETS.items():
            assert preset.sampler, f"Preset '{name}' has no sampler"

    def test_preset_apply_does_not_affect_sampler(self):
        """apply() only modifies prompt/negative_prompt, not sampler."""
        preset = GenrePreset(
            name="test", display_name="Test",
            positive_tags="hq", sampler="euler",
        )
        aug_prompt, aug_neg = preset.apply("landscape")
        assert "hq" in aug_prompt
        # Sampler is still the same — not mixed into the prompt
        assert preset.sampler == "euler"


# ---------------------------------------------------------------------------
# G) Flowthrough: request -> job
# ---------------------------------------------------------------------------


class TestSamplerFlowthrough:
    """Verify that sampler flows from GenerateRequest through to GenerationJob."""

    @pytest.fixture
    def client(self):
        from augmentum.proxy.image_routes import router

        app = FastAPI()
        app.include_router(router)

        state = MagicMock()
        state.image_queue = MagicMock(spec=["submit", "wait_for_result", "queue_size"])
        state.image_queue.queue_size = 0

        # Make submit capture the job for inspection
        self._submitted_job = None

        async def _capture_submit(job):
            self._submitted_job = job
            return job

        state.image_queue.submit = AsyncMock(side_effect=_capture_submit)
        state.image_queue.wait_for_result = AsyncMock(return_value={
            "image_id": "test123",
            "file_path": "/data/test.png",
            "seed": 42,
            "width": 512,
            "height": 512,
        })

        state.image_preset_manager = PresetManager()
        state.image_cache = AsyncMock()
        state.image_cache.get = AsyncMock(return_value=None)
        state.image_persistence = AsyncMock()
        state.image_hardware = MagicMock()
        app.state = state

        return TestClient(app)

    def test_sampler_flows_to_job(self, client):
        resp = client.post("/api/image/generate", json={
            "prompt": "test image",
            "sampler": "euler_a",
        })
        assert resp.status_code == 200
        assert self._submitted_job is not None
        assert self._submitted_job.sampler == "euler_a"

    def test_scheduler_flows_to_job(self, client):
        resp = client.post("/api/image/generate", json={
            "prompt": "test image",
            "scheduler": "ddim",
        })
        assert resp.status_code == 200
        assert self._submitted_job is not None
        assert self._submitted_job.scheduler == "ddim"

    def test_sampler_takes_precedence_over_scheduler(self, client):
        """When both sampler and scheduler are given, sampler wins (it's checked first)."""
        resp = client.post("/api/image/generate", json={
            "prompt": "test image",
            "sampler": "euler",
            "scheduler": "ddim",
        })
        assert resp.status_code == 200
        assert self._submitted_job.sampler == "euler"

    def test_no_sampler_defaults_empty(self, client):
        resp = client.post("/api/image/generate", json={
            "prompt": "test image",
        })
        assert resp.status_code == 200
        assert self._submitted_job.sampler == ""

    def test_preset_sampler_used_as_default(self, client):
        """When a preset has a sampler and user doesn't specify one, use preset's."""
        resp = client.post("/api/image/generate", json={
            "prompt": "fantasy warrior",
            "preset": "anime",
        })
        assert resp.status_code == 200
        assert self._submitted_job.sampler == "euler_a"

    def test_explicit_sampler_overrides_preset(self, client):
        """User-specified sampler should override the preset's default."""
        resp = client.post("/api/image/generate", json={
            "prompt": "fantasy warrior",
            "preset": "anime",
            "sampler": "ddim",
        })
        assert resp.status_code == 200
        assert self._submitted_job.sampler == "ddim"
