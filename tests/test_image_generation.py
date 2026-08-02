"""Tests for image generation pipeline, queue, presets, and cache."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.image.cache import ImageCache, _build_cache_key
from augmentum.image.hardware import (
    RECOMMENDED_MODELS,
    ModelTier,
    _classify_tier,
    detect_hardware,
    get_catalog_for_tier,
)
from augmentum.image.pipeline_registry import PipelineRegistry
from augmentum.image.presets import BUILTIN_PRESETS, GenrePreset, PresetManager
from augmentum.image.queue import GenerationJob, GenerationQueue
from augmentum.image.schemas import JobStatus
from augmentum.tools.base import ToolCategory
from augmentum.tools.image_generation import ImageGenerationTool

# --- Hardware Detection ---


class TestHardwareDetection:
    def test_classify_tier_high(self):
        tier, pipeline, model = _classify_tier(16_000)
        assert tier == ModelTier.HIGH
        assert pipeline == "flux"

    def test_classify_tier_medium(self):
        tier, pipeline, model = _classify_tier(8_000)
        assert tier == ModelTier.MEDIUM
        assert pipeline == "sdxl"

    def test_classify_tier_low(self):
        tier, pipeline, model = _classify_tier(4_000)
        assert tier == ModelTier.LOW
        assert pipeline == "sd15"

    def test_classify_tier_cpu(self):
        tier, pipeline, model = _classify_tier(2_000)
        assert tier == ModelTier.CPU
        assert pipeline == "sd15"

    def test_detect_hardware_fallback(self):
        """Without CUDA, should fall back to CPU."""
        profile = detect_hardware()
        assert profile.device in ("cuda", "cpu")
        assert isinstance(profile.tier, ModelTier)


# --- Model Catalog ---


class TestModelCatalog:
    def test_catalog_has_models(self):
        assert len(RECOMMENDED_MODELS) >= 5

    def test_catalog_has_cpu_models(self):
        cpu_models = [m for m in RECOMMENDED_MODELS if m.cpu_friendly]
        assert len(cpu_models) >= 1

    def test_catalog_tiers_cover_all(self):
        tiers = {m.min_tier for m in RECOMMENDED_MODELS}
        assert ModelTier.CPU in tiers
        assert ModelTier.LOW in tiers
        assert ModelTier.MEDIUM in tiers
        assert ModelTier.HIGH in tiers

    def test_catalog_models_have_vram_info(self):
        for m in RECOMMENDED_MODELS:
            assert m.size_gb > 0, f"{m.name} missing size_gb"
            assert m.speed_note, f"{m.name} missing speed_note"
            assert m.description, f"{m.name} missing description"

    def test_get_catalog_for_tier_sorts_compatible_first(self):
        result = get_catalog_for_tier(ModelTier.MEDIUM)
        seen_incompatible = False
        tier_order = {ModelTier.CPU: 0, ModelTier.LOW: 1, ModelTier.MEDIUM: 2, ModelTier.HIGH: 3}
        for m in result:
            compatible = tier_order[m.min_tier] <= tier_order[ModelTier.MEDIUM]
            if not compatible:
                seen_incompatible = True
            elif seen_incompatible:
                raise AssertionError(f"Compatible model {m.name} after incompatible")

    def test_get_catalog_for_cpu(self):
        result = get_catalog_for_tier(ModelTier.CPU)
        # First models should be CPU-compatible
        assert result[0].min_tier == ModelTier.CPU


# --- Presets ---


class TestPresets:
    def test_builtin_presets_exist(self):
        assert "fantasy_rpg" in BUILTIN_PRESETS
        assert "anime" in BUILTIN_PRESETS
        assert "scifi" in BUILTIN_PRESETS
        assert "horror" in BUILTIN_PRESETS
        assert "realism" in BUILTIN_PRESETS

    def test_preset_apply(self):
        preset = BUILTIN_PRESETS["fantasy_rpg"]
        prompt, neg = preset.apply("a castle", "modern")
        assert "castle" in prompt
        assert "fantasy" in prompt.lower()
        assert "modern" in neg

    def test_preset_apply_empty_negative(self):
        preset = BUILTIN_PRESETS["anime"]
        prompt, neg = preset.apply("a girl")
        assert "girl" in prompt
        assert neg  # Should have preset negative tags

    def test_preset_manager(self):
        mgr = PresetManager()
        assert mgr.get("fantasy_rpg") is not None
        assert mgr.get("nonexistent") is None
        assert len(mgr.list_presets()) >= 5

    def test_preset_manager_add_remove(self):
        mgr = PresetManager()
        custom = GenrePreset(name="test_preset", display_name="Test")
        mgr.add(custom)
        assert mgr.get("test_preset") is not None
        assert mgr.remove("test_preset") is True
        assert mgr.get("test_preset") is None

    def test_cannot_remove_builtin(self):
        mgr = PresetManager()
        assert mgr.remove("fantasy_rpg") is False


# --- Cache ---


class TestCache:
    def test_cache_key_deterministic(self):
        key1 = _build_cache_key("hello", "", "model", 42, 512, 512, 20, 7.0)
        key2 = _build_cache_key("hello", "", "model", 42, 512, 512, 20, 7.0)
        assert key1 == key2

    def test_cache_key_different(self):
        key1 = _build_cache_key("hello", "", "model", 42, 512, 512, 20, 7.0)
        key2 = _build_cache_key("world", "", "model", 42, 512, 512, 20, 7.0)
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        cache = ImageCache()
        result = await cache.get("prompt", "", "model", 42, 512, 512, 20, 7.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_put_get(self):
        cache = ImageCache()
        await cache.put("prompt", "", "model", 42, 512, 512, 20, 7.0, "img_123")
        result = await cache.get("prompt", "", "model", 42, 512, 512, 20, 7.0)
        assert result == "img_123"

    @pytest.mark.asyncio
    async def test_cache_skips_random_seed(self):
        cache = ImageCache()
        await cache.put("prompt", "", "model", -1, 512, 512, 20, 7.0, "img_123")
        result = await cache.get("prompt", "", "model", -1, 512, 512, 20, 7.0)
        assert result is None  # Random seeds are not cached


# --- Queue ---


class TestGenerationQueue:
    @pytest.mark.asyncio
    async def test_submit_and_complete(self):
        results = {"image_id": "test_123", "seed": 42, "width": 512, "height": 512}

        async def mock_generate(job):
            return results

        queue = GenerationQueue(max_size=5)
        queue.start(mock_generate)

        job = GenerationJob(prompt="test prompt")
        job = await queue.submit(job)
        result = await queue.wait_for_result(job, timeout=5.0)

        assert result["image_id"] == "test_123"
        assert job.status == JobStatus.COMPLETED

        await queue.stop()

    @pytest.mark.asyncio
    async def test_submit_failure(self):
        async def mock_generate(job):
            raise RuntimeError("GPU OOM")

        queue = GenerationQueue(max_size=5)
        queue.start(mock_generate)

        job = GenerationJob(prompt="test")
        job = await queue.submit(job)

        with pytest.raises(RuntimeError, match="GPU OOM"):
            await queue.wait_for_result(job, timeout=5.0)

        assert job.status == JobStatus.FAILED

        await queue.stop()

    @pytest.mark.asyncio
    async def test_queue_full(self):
        async def slow_generate(job):
            await asyncio.sleep(10)
            return {}

        queue = GenerationQueue(max_size=2)
        queue.start(slow_generate)

        # Fill the queue
        await queue.submit(GenerationJob(prompt="1"))
        await queue.submit(GenerationJob(prompt="2"))

        # Third should fail
        with pytest.raises(RuntimeError, match="full"):
            await queue.submit(GenerationJob(prompt="3"))

        await queue.stop()

    def test_job_id_generated(self):
        job = GenerationJob(prompt="test")
        assert len(job.job_id) > 0


# --- Tool ---


class TestImageGenerationTool:
    def test_tool_properties(self):
        queue = MagicMock()
        tool = ImageGenerationTool(queue=queue)

        assert tool.name == "image_generation"
        assert tool.category == ToolCategory.IMAGE
        assert "prompt" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["prompt"]

    def test_validate_input(self):
        queue = MagicMock()
        tool = ImageGenerationTool(queue=queue)

        assert tool.validate_input(prompt="hello") is True
        assert tool.validate_input(prompt="") is False
        assert tool.validate_input(prompt="  ") is False

    @pytest.mark.asyncio
    async def test_execute_success(self):
        future = asyncio.get_running_loop().create_future()
        future.set_result({
            "image_id": "abc123",
            "seed": 42,
            "width": 512,
            "height": 512,
        })

        mock_job = GenerationJob(prompt="test")
        mock_job.future = future

        queue = AsyncMock()
        queue.submit = AsyncMock(return_value=mock_job)
        queue.wait_for_result = AsyncMock(return_value={
            "image_id": "abc123",
            "seed": 42,
            "width": 512,
            "height": 512,
        })

        tool = ImageGenerationTool(queue=queue)

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.image_default_model = ""
            mock_settings.image_default_steps = 20
            mock_settings.image_default_cfg = 7.0
            mock_settings.image_default_width = 512
            mock_settings.image_default_height = 512

            result = await tool.execute(prompt="a sunset over mountains")

        assert result.success is True
        assert "gallery" in result.output
        assert result.metadata["image_id"] == "abc123"
        assert result.metadata["url"] == "/api/image/abc123"


# --- Pipeline Registry ---


class TestImagePersistence:
    """Tests for persistence layer search/filter/count/delete functions."""

    @pytest.fixture
    async def persistence(self):
        import aiosqlite

        from augmentum.image.persistence import ImagePersistence

        conn = await aiosqlite.connect(":memory:")
        # Schema mirrors the production migrations — user_id is required
        # because save_generation always writes it (even empty) so rows are
        # never NULL-owned (see persistence.py docstring).
        await conn.executescript("""
            CREATE TABLE image_generations (
                image_id TEXT PRIMARY KEY,
                session_id TEXT DEFAULT '',
                prompt TEXT NOT NULL,
                negative_prompt TEXT DEFAULT '',
                model TEXT NOT NULL,
                seed INTEGER DEFAULT -1,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                steps INTEGER NOT NULL,
                cfg_scale REAL NOT NULL DEFAULT 7.0,
                preset TEXT DEFAULT '',
                loras TEXT DEFAULT '[]',
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                job_type TEXT NOT NULL DEFAULT 'txt2img',
                strength REAL NOT NULL DEFAULT 1.0,
                source_image_id TEXT DEFAULT '',
                is_private INTEGER NOT NULL DEFAULT 0,
                is_background INTEGER NOT NULL DEFAULT 0,
                user_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE image_cache (
                cache_key TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        p = ImagePersistence(conn)
        yield p
        await conn.close()

    @pytest.mark.asyncio
    async def test_count_generations_empty(self, persistence):
        count = await persistence.count_generations(user_id="u1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_generations(self, persistence):
        await persistence.save_generation(
            image_id="img1", session_id="s1", prompt="sunset",
            negative_prompt="", model="sd-v1-5", seed=42,
            width=512, height=512, steps=20, cfg_scale=7.0,
            preset="", loras=[], file_path="/tmp/img1.png", user_id="u1",
        )
        await persistence.save_generation(
            image_id="img2", session_id="s1", prompt="mountain",
            negative_prompt="", model="sdxl", seed=43,
            width=1024, height=1024, steps=30, cfg_scale=7.0,
            preset="realism", loras=[], file_path="/tmp/img2.png", user_id="u1",
        )
        count = await persistence.count_generations(user_id="u1")
        assert count == 2

        count_filtered = await persistence.count_generations(q="sunset", user_id="u1")
        assert count_filtered == 1

        count_model = await persistence.count_generations(model="sdxl", user_id="u1")
        assert count_model == 1

    @pytest.mark.asyncio
    async def test_delete_generation(self, persistence):
        await persistence.save_generation(
            image_id="del1", session_id="s1", prompt="to delete",
            negative_prompt="", model="sd-v1-5", seed=1,
            width=512, height=512, steps=20, cfg_scale=7.0,
            preset="", loras=[], file_path="/tmp/del1.png", user_id="u1",
        )
        file_path = await persistence.delete_generation("del1", user_id="u1")
        assert file_path == "/tmp/del1.png"

        # Should be gone
        gen = await persistence.get_generation("del1", user_id="u1")
        assert gen is None

    @pytest.mark.asyncio
    async def test_delete_generation_not_found(self, persistence):
        result = await persistence.delete_generation("nonexistent", user_id="u1")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_generation_returns_true_when_vfs_absent(self, persistence):
        """No file index configured → nothing to register, not a failure.
        save_generation must return True so the tool doesn't warn the user
        about a missing index the user never configured.
        """
        from augmentum import vfs
        # Ensure no index is configured for this test
        prior = vfs._file_index
        vfs._file_index = None
        try:
            ok = await persistence.save_generation(
                image_id="v1", session_id="s", prompt="p", negative_prompt="",
                model="m", seed=1, width=8, height=8, steps=1, cfg_scale=1.0,
                preset="", loras=[], file_path="/tmp/v1.png", user_id="u1",
            )
            assert ok is True
        finally:
            vfs._file_index = prior

    @pytest.mark.asyncio
    async def test_save_generation_returns_false_on_vfs_failure(self, persistence):
        """Index configured, registration raises → return False so the
        caller can surface an orphan-file warning to the user."""
        from unittest.mock import AsyncMock, MagicMock

        from augmentum import vfs
        fake_index = MagicMock()
        fake_index.register = AsyncMock(side_effect=RuntimeError("index down"))
        prior = vfs._file_index
        vfs._file_index = fake_index
        try:
            ok = await persistence.save_generation(
                image_id="v2", session_id="s", prompt="p", negative_prompt="",
                model="m", seed=1, width=8, height=8, steps=1, cfg_scale=1.0,
                preset="", loras=[], file_path="/tmp/v2.png", user_id="u1",
            )
            assert ok is False
        finally:
            vfs._file_index = prior

    @pytest.mark.asyncio
    async def test_save_generation_returns_true_on_vfs_success(self, persistence):
        from unittest.mock import AsyncMock, MagicMock

        from augmentum import vfs
        fake_index = MagicMock()
        fake_index.register = AsyncMock(return_value="file-abc")
        prior = vfs._file_index
        vfs._file_index = fake_index
        try:
            ok = await persistence.save_generation(
                image_id="v3", session_id="s", prompt="p", negative_prompt="",
                model="m", seed=1, width=8, height=8, steps=1, cfg_scale=1.0,
                preset="", loras=[], file_path="/tmp/v3.png", user_id="u1",
            )
            assert ok is True
        finally:
            vfs._file_index = prior

    @pytest.mark.asyncio
    async def test_list_generations_search(self, persistence):
        await persistence.save_generation(
            image_id="s1", session_id="s1", prompt="sunset over ocean",
            negative_prompt="", model="sd-v1-5", seed=1,
            width=512, height=512, steps=20, cfg_scale=7.0,
            preset="", loras=[], file_path="/tmp/s1.png", user_id="u1",
        )
        await persistence.save_generation(
            image_id="s2", session_id="s1", prompt="mountain landscape",
            negative_prompt="", model="sd-v1-5", seed=2,
            width=512, height=512, steps=20, cfg_scale=7.0,
            preset="", loras=[], file_path="/tmp/s2.png", user_id="u1",
        )

        results = await persistence.list_generations(q="sunset", user_id="u1")
        assert len(results) == 1
        assert results[0].image_id == "s1"

    @pytest.mark.asyncio
    async def test_list_generations_filter_model(self, persistence):
        await persistence.save_generation(
            image_id="m1", session_id="s1", prompt="test",
            negative_prompt="", model="sd-v1-5", seed=1,
            width=512, height=512, steps=20, cfg_scale=7.0,
            preset="", loras=[], file_path="/tmp/m1.png", user_id="u1",
        )
        await persistence.save_generation(
            image_id="m2", session_id="s1", prompt="test",
            negative_prompt="", model="sdxl", seed=2,
            width=1024, height=1024, steps=30, cfg_scale=7.0,
            preset="", loras=[], file_path="/tmp/m2.png", user_id="u1",
        )

        results = await persistence.list_generations(model="sdxl", user_id="u1")
        assert len(results) == 1
        assert results[0].model == "sdxl"

    @pytest.mark.asyncio
    async def test_list_generations_includes_preset_loras(self, persistence):
        await persistence.save_generation(
            image_id="pl1", session_id="s1", prompt="test",
            negative_prompt="", model="sd-v1-5", seed=1,
            width=512, height=512, steps=20, cfg_scale=7.0,
            preset="fantasy_rpg", loras=[{"name": "lora1", "weight": 0.8}],
            file_path="/tmp/pl1.png", user_id="u1",
        )

        results = await persistence.list_generations(user_id="u1")
        assert len(results) == 1
        assert results[0].preset == "fantasy_rpg"
        assert len(results[0].loras) == 1
        assert results[0].loras[0]["name"] == "lora1"


class TestPipelineRegistry:
    def test_initial_state(self):
        reg = PipelineRegistry()
        assert reg.current is None
        assert reg.is_loaded is False
        assert reg.current_model == ""
