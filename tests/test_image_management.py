"""Tests for image management -- model_manager, lora_manager, presets, schedulers, persistence."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.image.lora_manager import LoraManager
from augmentum.image.model_manager import ModelManager
from augmentum.image.presets import BUILTIN_PRESETS, GenrePreset, PresetManager
from augmentum.image.schedulers import (
    SAMPLER_ALIASES,
    SAMPLER_MAP,
    _resolve_alias,
    get_available_samplers,
)


class TestModelManager:
    def test_construction(self, tmp_path):
        mgr = ModelManager(str(tmp_path / "models"))
        assert os.path.isdir(mgr.model_dir)

    def test_list_local_models_empty(self, tmp_path):
        mgr = ModelManager(str(tmp_path / "models"))
        models = mgr.list_local_models()
        assert models == []

    def test_get_model_path_missing(self, tmp_path):
        mgr = ModelManager(str(tmp_path / "models"))
        assert mgr.get_model_path("nonexistent") is None

    def test_safe_model_path_blocks_traversal(self, tmp_path):
        mgr = ModelManager(str(tmp_path / "models"))
        result = mgr._safe_model_path("../../etc/passwd")
        assert result is None

    def test_has_model_files_with_safetensors(self, tmp_path):
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake")
        assert ModelManager._has_model_files(str(model_dir)) is True

    def test_has_model_files_empty_dir(self, tmp_path):
        model_dir = tmp_path / "empty"
        model_dir.mkdir()
        assert ModelManager._has_model_files(str(model_dir)) is False


class TestLoraManager:
    def test_construction(self, tmp_path):
        mgr = LoraManager(str(tmp_path))
        assert os.path.isdir(mgr.lora_dir)

    def test_discover_empty(self, tmp_path):
        mgr = LoraManager(str(tmp_path))
        assert mgr.discover() == []

    def test_discover_finds_safetensors(self, tmp_path):
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir()
        (lora_dir / "my_lora.safetensors").write_bytes(b"fake_lora_data")
        mgr = LoraManager(str(tmp_path))
        loras = mgr.discover()
        assert len(loras) == 1
        assert loras[0].name == "my_lora"

    def test_get_path_missing(self, tmp_path):
        mgr = LoraManager(str(tmp_path))
        assert mgr.get_path("nonexistent") is None

    def test_match_character_none(self, tmp_path):
        mgr = LoraManager(str(tmp_path))
        assert mgr.match_character("Bob") is None


class TestPresetManager:
    def test_builtin_presets_exist(self):
        assert len(BUILTIN_PRESETS) >= 5

    def test_list_presets(self):
        mgr = PresetManager()
        presets = mgr.list_presets()
        assert len(presets) >= 5

    def test_get_preset(self):
        mgr = PresetManager()
        preset = mgr.get("fantasy_rpg")
        assert preset is not None
        assert preset.display_name == "Fantasy RPG"

    def test_add_custom_preset(self):
        mgr = PresetManager()
        custom = GenrePreset(name="custom_test", display_name="Custom Test")
        mgr.add(custom)
        assert mgr.get("custom_test") is not None

    def test_remove_custom_preset(self):
        mgr = PresetManager()
        custom = GenrePreset(name="temp_preset", display_name="Temp")
        mgr.add(custom)
        assert mgr.remove("temp_preset") is True
        assert mgr.get("temp_preset") is None

    def test_cannot_remove_builtin(self):
        mgr = PresetManager()
        assert mgr.remove("fantasy_rpg") is False

    def test_preset_apply(self):
        preset = BUILTIN_PRESETS["anime"]
        pos, neg = preset.apply("a warrior", "blurry")
        assert "warrior" in pos
        assert "anime" in pos
        assert "blurry" in neg


class TestSchedulers:
    def test_sampler_map_has_entries(self):
        assert len(SAMPLER_MAP) >= 10

    def test_each_entry_has_class_and_kwargs(self):
        for name, (cls_name, kwargs) in SAMPLER_MAP.items():
            assert isinstance(cls_name, str)
            assert isinstance(kwargs, dict)

    def test_resolve_alias(self):
        assert _resolve_alias("euler_ancestral") == "euler_a"
        assert _resolve_alias("dpmpp_2m_karras") == "dpm++_2m_karras"

    def test_resolve_alias_passthrough(self):
        assert _resolve_alias("euler") == "euler"

    def test_get_available_samplers(self):
        samplers = get_available_samplers()
        assert len(samplers) >= 10
        names = {s["name"] for s in samplers}
        assert "euler" in names
        assert "ddim" in names


class TestImagePersistenceRoundTrip:
    @pytest.mark.asyncio
    async def test_save_and_load_generation(self, tmp_path):
        import aiosqlite
        from augmentum.image.persistence import ImagePersistence

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE image_generations (
                    image_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    prompt TEXT,
                    negative_prompt TEXT,
                    model TEXT,
                    seed INTEGER,
                    width INTEGER,
                    height INTEGER,
                    steps INTEGER,
                    cfg_scale REAL,
                    preset TEXT,
                    loras TEXT,
                    file_path TEXT,
                    job_type TEXT DEFAULT 'txt2img',
                    strength REAL DEFAULT 1.0,
                    source_image_id TEXT DEFAULT '',
                    is_private INTEGER DEFAULT 0,
                    is_background INTEGER DEFAULT 0,
                    user_id TEXT NOT NULL DEFAULT '',
                    origin TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                CREATE TABLE image_cache (
                    cache_key TEXT PRIMARY KEY,
                    image_id TEXT,
                    user_id TEXT NOT NULL DEFAULT ''
                )
            """)
            await conn.commit()

            persistence = ImagePersistence(conn)
            await persistence.save_generation(
                image_id="test_img_001",
                session_id="sess_001",
                prompt="a beautiful sunset",
                negative_prompt="ugly",
                model="sd15",
                seed=42,
                width=512,
                height=512,
                steps=20,
                cfg_scale=7.0,
                preset="realism",
                loras=[],
                file_path="/tmp/test.png",
                user_id="u1",
            )

            result = await persistence.get_generation("test_img_001", user_id="u1")
            assert result is not None
            assert result["prompt"] == "a beautiful sunset"
            assert result["seed"] == 42

    @pytest.mark.asyncio
    async def test_list_generations(self, tmp_path):
        import aiosqlite
        from augmentum.image.persistence import ImagePersistence

        db_path = str(tmp_path / "test2.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE image_generations (
                    image_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    prompt TEXT,
                    negative_prompt TEXT,
                    model TEXT,
                    seed INTEGER,
                    width INTEGER,
                    height INTEGER,
                    steps INTEGER,
                    cfg_scale REAL,
                    preset TEXT,
                    loras TEXT,
                    file_path TEXT,
                    job_type TEXT DEFAULT 'txt2img',
                    strength REAL DEFAULT 1.0,
                    source_image_id TEXT DEFAULT '',
                    is_private INTEGER DEFAULT 0,
                    is_background INTEGER DEFAULT 0,
                    user_id TEXT NOT NULL DEFAULT '',
                    origin TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute(
                "CREATE TABLE image_cache (cache_key TEXT PRIMARY KEY, image_id TEXT, user_id TEXT NOT NULL DEFAULT '')"
            )
            await conn.commit()

            persistence = ImagePersistence(conn)
            await persistence.save_generation(
                image_id="img1", session_id="s1", prompt="cat",
                negative_prompt="", model="sd15", seed=1,
                width=512, height=512, steps=20, cfg_scale=7.0,
                preset="", loras=[], file_path="/tmp/1.png", user_id="u1",
            )
            await persistence.save_generation(
                image_id="img2", session_id="s1", prompt="dog",
                negative_prompt="", model="sd15", seed=2,
                width=512, height=512, steps=20, cfg_scale=7.0,
                preset="", loras=[], file_path="/tmp/2.png", user_id="u1",
            )

            entries = await persistence.list_generations(limit=10, user_id="u1")
            assert len(entries) == 2

            count = await persistence.count_generations(user_id="u1")
            assert count == 2


class TestImageModelCache:
    """list_models() cache invalidates correctly.

    Bug it prevents: fabric heartbeat hits list_models() every 5s on
    the shared aiosqlite connection. Without caching, a contended
    connection (media-progress UPDATE etc.) blocked the SELECT for
    seconds and drove event-loop stalls.
    """

    @pytest.mark.asyncio
    async def test_list_models_caches_and_invalidates(self, tmp_path):
        import aiosqlite
        from augmentum.image.persistence import ImagePersistence

        db_path = str(tmp_path / "models.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE image_models (
                    name TEXT PRIMARY KEY,
                    pipeline_type TEXT,
                    path TEXT,
                    source TEXT,
                    size_bytes INTEGER,
                    metadata TEXT
                )
            """)
            await conn.commit()

            persistence = ImagePersistence(conn)

            # First call populates the cache.
            assert await persistence.list_models() == []
            assert persistence._models_cache == []

            # save_model invalidates → next list returns the new row.
            await persistence.save_model(
                name="sd-1.5", pipeline_type="sd15", path="/m/sd15",
            )
            assert persistence._models_cache is None
            listed = await persistence.list_models()
            assert len(listed) == 1
            assert listed[0].name == "sd-1.5"

            # get_model served from cache when populated — no extra
            # SQL round-trip, hot path stays off the connection.
            hit = await persistence.get_model("sd-1.5")
            assert hit is not None
            miss = await persistence.get_model("not-there")
            assert miss is None

            # delete_model invalidates too.
            assert await persistence.delete_model("sd-1.5") is True
            assert persistence._models_cache is None
            assert await persistence.list_models() == []
