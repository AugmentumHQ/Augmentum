"""Tests for capability extractors.

Each extractor is tested with a fake of its data source. Pins:

  - llm extractor surfaces the engine model + provider_registry models
  - llm extractor handles missing managers (no llama, no registry)
    gracefully -- returns empty list, doesn't raise
  - llm extractor: engine reported with `loaded=True` only when state is ready
  - image extractor handles missing registry
  - knowledge extractor reads pack_manager.installed() dicts
  - all extractors absorb individual data-source errors

These tests intentionally don't touch the real LlamaServerManager /
ImagePipelineRegistry / PackManager classes; fakes are simpler and
the extractors don't need to know the real classes' implementation.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.fabric.capabilities import (
    KIND_IMAGE_GENERATION,
    KIND_KNOWLEDGE_SEARCH,
    KIND_LLM_INFERENCE,
)
from augmentum.fabric.extractors import (
    ImageCapabilityExtractor,
    KnowledgeSearchExtractor,
    LLMCapabilityExtractor,
)

# ── LLM extractor ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_extractor_returns_empty_when_no_sources():
    extractor = LLMCapabilityExtractor()
    caps = await extractor.collect()
    assert caps == []


@pytest.mark.asyncio
async def test_llm_extractor_surfaces_engine_model():
    """A live llama_manager.status() returning a ready model produces
    one LLMInferenceCapability with detailed fields.
    """
    fake_mgr = MagicMock()
    fake_mgr.status.return_value = {
        "model_id": "Qwen3.5-72B-A10B-q4",
        "state": "ready",
        "ctx_size": 32768,
        "free_slots": 2,
        "profile": {"architecture": "qwen3", "size_gb": 72.0},
        "gpu": {
            "name": "GPU-A",
            "vram_free_mib": 8200,
            "vram_total_mib": 24576,
        },
    }
    extractor = LLMCapabilityExtractor(llama_manager=fake_mgr)
    caps = await extractor.collect()
    assert len(caps) == 1
    cap = caps[0]
    assert cap.kind == KIND_LLM_INFERENCE
    assert cap.backend == "engine"
    assert cap.model_id == "Qwen3.5-72B-A10B-q4"
    assert cap.loaded is True
    assert cap.free_slots == 2
    assert cap.device["vram_free_mb"] == 8200


@pytest.mark.asyncio
async def test_llm_extractor_skips_engine_when_no_model_loaded():
    fake_mgr = MagicMock()
    fake_mgr.status.return_value = {
        "model_id": "",  # no model loaded
        "state": "idle",
    }
    extractor = LLMCapabilityExtractor(llama_manager=fake_mgr)
    caps = await extractor.collect()
    assert caps == []


@pytest.mark.asyncio
async def test_llm_extractor_surfaces_provider_registry_models():
    """Non-engine backends (anthropic, openai, etc.) come from the
    refresh_model_map() listing.
    """
    fake_registry = MagicMock()
    fake_registry.refresh_model_map = AsyncMock(return_value={
        "claude-opus-4-7": "anthropic",
        "gpt-5": "openai",
        # The "engine" model should be skipped -- engine extractor
        # path emits a richer version of it.
        "Qwen3.5-72B-A10B-q4": "engine",
    })
    extractor = LLMCapabilityExtractor(provider_registry=fake_registry)
    caps = await extractor.collect()
    # Two non-engine backends.
    assert len(caps) == 2
    backends = {c.backend for c in caps}
    assert backends == {"anthropic", "openai"}


@pytest.mark.asyncio
async def test_llm_extractor_absorbs_engine_status_error():
    """A broken status() doesn't crash the extractor."""
    fake_mgr = MagicMock()
    fake_mgr.status.side_effect = RuntimeError("engine corrupted state")
    extractor = LLMCapabilityExtractor(llama_manager=fake_mgr)
    caps = await extractor.collect()
    assert caps == []  # silently empty, not raised


@pytest.mark.asyncio
async def test_llm_extractor_absorbs_registry_error():
    fake_registry = MagicMock()
    fake_registry.refresh_model_map = AsyncMock(side_effect=RuntimeError("net down"))
    extractor = LLMCapabilityExtractor(provider_registry=fake_registry)
    caps = await extractor.collect()
    assert caps == []


# ── Image extractor ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_image_extractor_returns_empty_when_no_registry():
    extractor = ImageCapabilityExtractor()
    caps = await extractor.collect()
    assert caps == []


@pytest.mark.asyncio
async def test_image_extractor_reads_persistence():
    """Persistence's async list_models() seeds the cap list."""
    fake_persistence = MagicMock()
    fake_pt_flux = MagicMock()
    fake_pt_flux.value = "flux"
    fake_pt_sdxl = MagicMock()
    fake_pt_sdxl.value = "sdxl"
    fake_persistence.list_models = AsyncMock(return_value=[
        MagicMock(name="ignored", pipeline_type=fake_pt_flux),
        MagicMock(name="ignored2", pipeline_type=fake_pt_sdxl),
    ])
    # MagicMock(name=...) collides with the .name kwarg used by the
    # Mock itself; set .name explicitly via the configure path.
    fake_persistence.list_models.return_value[0].name = "flux-schnell-fp8"
    fake_persistence.list_models.return_value[1].name = "sdxl-base-1.0"
    extractor = ImageCapabilityExtractor(persistence=fake_persistence)
    caps = await extractor.collect()
    assert len(caps) == 2
    assert all(c.kind == KIND_IMAGE_GENERATION for c in caps)
    ids = {c.model_id for c in caps}
    assert ids == {"flux-schnell-fp8", "sdxl-base-1.0"}


@pytest.mark.asyncio
async def test_image_extractor_absorbs_persistence_error():
    fake_persistence = MagicMock()
    fake_persistence.list_models = AsyncMock(side_effect=RuntimeError("bad"))
    extractor = ImageCapabilityExtractor(persistence=fake_persistence)
    caps = await extractor.collect()
    assert caps == []


@pytest.mark.asyncio
async def test_image_extractor_unions_disk_scan_with_persistence():
    """Models on disk but missing from the SQLite ``image_models`` table
    must still get advertised. Baked-in system models (e.g. DreamShaper
    in the GPU image) and hand-dropped folders never call save_model().
    Pre-fix the extractor only saw persistence, so these were invisible
    across the fabric -- the dropdown showed them locally but cross-peer
    routing thought nobody could serve them.
    """
    fake_persistence = MagicMock()
    fake_persistence.list_models = AsyncMock(return_value=[])  # empty SQLite

    fake_model_mgr = MagicMock()
    fake_pt = MagicMock()
    fake_pt.value = "sd_1.5"
    fake_model_mgr.list_local_models.return_value = [
        {"name": "dreamshaper_8", "pipeline_type": fake_pt, "path": "/baked"},
        {"name": "hand_dropped_model", "pipeline_type": fake_pt, "path": "/data"},
    ]

    extractor = ImageCapabilityExtractor(
        persistence=fake_persistence,
        model_manager=fake_model_mgr,
    )
    caps = await extractor.collect()
    ids = {c.model_id for c in caps}
    assert ids == {"dreamshaper_8", "hand_dropped_model"}


@pytest.mark.asyncio
async def test_image_extractor_disk_scan_wins_on_name_collision():
    """When the same model is in both sources, the disk scan's family
    wins -- it's the source of truth for what's actually on disk
    right now.
    """
    fake_persistence = MagicMock()
    persistence_pt = MagicMock()
    persistence_pt.value = "stale-family"
    persistence_model = MagicMock(pipeline_type=persistence_pt)
    persistence_model.name = "shared_model"
    fake_persistence.list_models = AsyncMock(return_value=[persistence_model])

    fake_model_mgr = MagicMock()
    disk_pt = MagicMock()
    disk_pt.value = "sdxl"
    fake_model_mgr.list_local_models.return_value = [
        {"name": "shared_model", "pipeline_type": disk_pt, "path": "/data"},
    ]

    extractor = ImageCapabilityExtractor(
        persistence=fake_persistence,
        model_manager=fake_model_mgr,
    )
    caps = await extractor.collect()
    assert len(caps) == 1
    assert caps[0].model_id == "shared_model"
    assert caps[0].family == "sdxl"


# ── Knowledge extractor ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_knowledge_extractor_returns_empty_when_no_pack_manager():
    extractor = KnowledgeSearchExtractor()
    caps = await extractor.collect()
    assert caps == []


@pytest.mark.asyncio
async def test_knowledge_extractor_reads_installed():
    fake_manager = MagicMock()
    fake_manager.installed.return_value = [
        {
            "pack_id": "wikipedia_en_simple_2026-02",
            "name": "Wikipedia (Simple English)",
            "chunk_count": 12345,
            "embedding_dim": 768,
            "active": True,
        },
        {
            "pack_id": "devdocs_en_javascript_2026-04",
            "name": "DevDocs JavaScript",
            "chunk_count": 0,
            "embedding_dim": 0,
            "active": True,
            "main_entry_path": "/A/index.html",  # → zim format
        },
    ]
    extractor = KnowledgeSearchExtractor(pack_manager=fake_manager)
    caps = await extractor.collect()
    assert len(caps) == 2
    assert all(c.kind == KIND_KNOWLEDGE_SEARCH for c in caps)
    wiki = next(c for c in caps if c.pack_id.startswith("wikipedia"))
    assert wiki.chunk_count == 12345
    devdocs = next(c for c in caps if c.pack_id.startswith("devdocs"))
    assert devdocs.pack_format == "zim"


@pytest.mark.asyncio
async def test_knowledge_extractor_skips_rows_without_pack_id():
    fake_manager = MagicMock()
    fake_manager.installed.return_value = [
        {"pack_id": "valid_pack", "name": "Valid"},
        {"name": "No ID"},  # dropped
    ]
    extractor = KnowledgeSearchExtractor(pack_manager=fake_manager)
    caps = await extractor.collect()
    assert len(caps) == 1
    assert caps[0].pack_id == "valid_pack"


@pytest.mark.asyncio
async def test_knowledge_extractor_absorbs_installed_error():
    fake_manager = MagicMock()
    fake_manager.installed.side_effect = RuntimeError("packs corrupted")
    extractor = KnowledgeSearchExtractor(pack_manager=fake_manager)
    caps = await extractor.collect()
    assert caps == []
