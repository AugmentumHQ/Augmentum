"""Smoke tests -- verify every image module imports and primary classes construct."""

from __future__ import annotations

import importlib


class TestImageModuleImports:
    """Every module under augmentum/image/ must import without error."""

    def test_import_pipeline(self):
        mod = importlib.import_module("augmentum.image.pipeline")
        assert mod is not None

    def test_import_pipeline_v2(self):
        mod = importlib.import_module("augmentum.image.pipeline_v2")
        assert mod is not None

    def test_import_pipeline_registry(self):
        mod = importlib.import_module("augmentum.image.pipeline_registry")
        assert mod is not None

    def test_import_model_manager(self):
        mod = importlib.import_module("augmentum.image.model_manager")
        assert hasattr(mod, "ModelManager")

    def test_import_distiller(self):
        mod = importlib.import_module("augmentum.image.distiller")
        assert mod is not None

    def test_import_distilled(self):
        mod = importlib.import_module("augmentum.image.distilled")
        assert mod is not None

    def test_import_cache(self):
        mod = importlib.import_module("augmentum.image.cache")
        assert mod is not None

    def test_import_queue(self):
        mod = importlib.import_module("augmentum.image.queue")
        assert mod is not None

    def test_import_lora_manager(self):
        mod = importlib.import_module("augmentum.image.lora_manager")
        assert hasattr(mod, "LoraManager")

    def test_import_presets(self):
        mod = importlib.import_module("augmentum.image.presets")
        assert hasattr(mod, "PresetManager")
        assert hasattr(mod, "BUILTIN_PRESETS")

    def test_import_defaults(self):
        mod = importlib.import_module("augmentum.image.defaults")
        assert mod is not None

    def test_import_schedulers(self):
        mod = importlib.import_module("augmentum.image.schedulers")
        assert hasattr(mod, "SAMPLER_MAP")
        assert hasattr(mod, "get_available_samplers")

    def test_import_hardware(self):
        mod = importlib.import_module("augmentum.image.hardware")
        assert mod is not None

    def test_import_persistence(self):
        mod = importlib.import_module("augmentum.image.persistence")
        assert hasattr(mod, "ImagePersistence")

    def test_import_schemas(self):
        mod = importlib.import_module("augmentum.image.schemas")
        assert hasattr(mod, "GenerateRequest")
        assert hasattr(mod, "GenerateResponse")

    def test_import_vram(self):
        mod = importlib.import_module("augmentum.image.vram")
        assert hasattr(mod, "release_pipeline")
        assert hasattr(mod, "flush_cuda_cache")

    def test_import_prompt_condenser(self):
        mod = importlib.import_module("augmentum.image.prompt_condenser")
        assert mod is not None
