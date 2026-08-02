"""Tests for provider catalog models and loading."""
from __future__ import annotations

from augmentum.providers.models import (
    GpuRequirements,
    ManagedService,
    ServiceCategory,
    ServiceDefinition,
    ServiceStatus,
)


def test_service_definition_defaults():
    sd = ServiceDefinition(
        id="test",
        name="Test",
        description="A test service",
        category=ServiceCategory.TTS,
        image="test:latest",
        internal_port=8080,
        host_port=6500,
    )
    assert sd.gpu.required is False
    assert sd.env == {}
    assert sd.volumes == {}
    assert sd.is_custom is False
    assert sd.health_check is None
    assert sd.command is None


def test_service_category_values():
    assert ServiceCategory.LLM.value == "llm"
    assert ServiceCategory.TTS.value == "tts"
    assert ServiceCategory.STT.value == "stt"
    assert ServiceCategory.IMAGE.value == "image"


def test_service_status_values():
    assert ServiceStatus.STOPPED.value == "stopped"
    assert ServiceStatus.RUNNING.value == "running"
    assert ServiceStatus.PULLING.value == "pulling"


def test_managed_service_defaults():
    ms = ManagedService(
        id="test",
        definition_id="test-def",
        name="Test",
        category="tts",
        image="test:latest",
    )
    assert ms.enabled is False
    assert ms.status == "stopped"
    assert ms.container_id is None
    assert ms.config_json == "{}"


# -- Catalog loading tests --

from augmentum.providers.catalog import ProviderCatalog


def test_catalog_loads():
    catalog = ProviderCatalog()
    entries = catalog.list_all()
    assert len(entries) > 0
    for entry in entries:
        assert entry.id
        assert entry.name
        assert entry.image
        assert entry.internal_port > 0
        assert entry.host_port > 0
        assert entry.category in ServiceCategory


def test_catalog_filter_by_category():
    catalog = ProviderCatalog()
    tts = catalog.list_by_category(ServiceCategory.TTS)
    assert len(tts) > 0
    for entry in tts:
        assert entry.category == ServiceCategory.TTS


def test_catalog_get_by_id():
    # LLM runners (ollama/llama.cpp) were removed from the marketplace catalog
    # — Augmentum ships its own engine + auto-detects external OpenAI-compat
    # endpoints. The catalog is TTS/STT-focused now.
    catalog = ProviderCatalog()
    chatterbox = catalog.get("chatterbox-turbo")
    assert chatterbox is not None
    assert chatterbox.name == "Chatterbox Turbo"
    assert chatterbox.gpu.required is True


def test_catalog_get_nonexistent():
    catalog = ProviderCatalog()
    assert catalog.get("nonexistent") is None
