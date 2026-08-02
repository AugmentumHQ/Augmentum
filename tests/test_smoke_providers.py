"""Smoke tests — import and construct every module in augmentum/providers/."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestProviderModels:
    """Verify provider data models construct correctly."""

    def test_service_category_enum(self):
        from augmentum.providers.models import ServiceCategory

        assert ServiceCategory.LLM.value == "llm"
        assert ServiceCategory.TTS.value == "tts"
        assert ServiceCategory.STT.value == "stt"
        assert ServiceCategory.IMAGE.value == "image"

    def test_service_status_enum(self):
        from augmentum.providers.models import ServiceStatus

        assert ServiceStatus.STOPPED.value == "stopped"
        assert ServiceStatus.RUNNING.value == "running"
        assert ServiceStatus.ERROR.value == "error"

    def test_construct_health_check(self):
        from augmentum.providers.models import HealthCheck

        hc = HealthCheck(test=["CMD", "curl", "-f", "http://localhost/health"])
        assert hc.interval_s == 10
        assert hc.retries == 5

    def test_construct_gpu_requirements(self):
        from augmentum.providers.models import GpuRequirements

        gpu = GpuRequirements()
        assert gpu.required is False
        assert gpu.driver == "nvidia"

    def test_construct_service_definition(self):
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        sd = ServiceDefinition(
            id="test-svc",
            name="Test Service",
            description="A test",
            category=ServiceCategory.LLM,
            image="test:latest",
            internal_port=8080,
            host_port=8080,
        )
        assert sd.id == "test-svc"
        assert sd.is_custom is False

    def test_construct_managed_service(self):
        from augmentum.providers.models import ManagedService

        ms = ManagedService(
            id="svc1",
            definition_id="ollama",
            name="Ollama",
            category="llm",
            image="ollama/ollama:latest",
        )
        assert ms.enabled is False
        assert ms.status == "stopped"
        assert ms.container_id is None


class TestProviderCatalog:
    """Verify ProviderCatalog loads from the bundled catalog.json."""

    def test_construct_default(self):
        from augmentum.providers.catalog import ProviderCatalog

        catalog = ProviderCatalog()
        entries = catalog.list_all()
        assert isinstance(entries, list)

    def test_list_all_returns_service_definitions(self):
        from augmentum.providers.catalog import ProviderCatalog
        from augmentum.providers.models import ServiceDefinition

        catalog = ProviderCatalog()
        entries = catalog.list_all()
        if entries:
            assert isinstance(entries[0], ServiceDefinition)

    def test_get_returns_none_for_unknown(self):
        from augmentum.providers.catalog import ProviderCatalog

        catalog = ProviderCatalog()
        assert catalog.get("nonexistent-service-xyz") is None

    def test_list_by_category(self):
        from augmentum.providers.catalog import ProviderCatalog
        from augmentum.providers.models import ServiceCategory

        catalog = ProviderCatalog()
        llm_services = catalog.list_by_category(ServiceCategory.LLM)
        assert isinstance(llm_services, list)

    def test_construct_with_missing_path(self, tmp_path):
        from augmentum.providers.catalog import ProviderCatalog

        catalog = ProviderCatalog(path=tmp_path / "nonexistent.json")
        assert catalog.list_all() == []


class TestServiceManager:
    """Verify ServiceManager can be constructed and has expected interface."""

    def test_construct(self, mock_docker):
        from augmentum.providers.manager import ServiceManager

        sm = ServiceManager(docker=mock_docker, db=None)
        assert sm._docker is mock_docker
        assert sm.catalog is not None

    def test_construct_with_custom_catalog(self, mock_docker):
        from augmentum.providers.catalog import ProviderCatalog
        from augmentum.providers.manager import ServiceManager

        catalog = ProviderCatalog()
        sm = ServiceManager(docker=mock_docker, db=None, catalog=catalog)
        assert sm.catalog is catalog

    def test_build_container_config(self, mock_docker):
        from augmentum.providers.manager import ServiceManager
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        sd = ServiceDefinition(
            id="test-svc",
            name="Test",
            description="test",
            category=ServiceCategory.LLM,
            image="test:latest",
            internal_port=11434,
            host_port=11434,
            env={"KEY": "value"},
        )
        config = ServiceManager._build_container_config(sd, "augmentum_default")
        assert config["Image"] == "test:latest"
        assert "KEY=value" in config["Env"]
        assert "11434/tcp" in config["ExposedPorts"]

    def test_build_container_config_with_gpu(self):
        from augmentum.providers.manager import ServiceManager
        from augmentum.providers.models import (
            GpuRequirements,
            ServiceCategory,
            ServiceDefinition,
        )

        sd = ServiceDefinition(
            id="gpu-svc",
            name="GPU Test",
            description="gpu",
            category=ServiceCategory.IMAGE,
            image="gpu:latest",
            internal_port=8080,
            host_port=8080,
            gpu=GpuRequirements(required=True, vram_mb=8000),
        )
        config = ServiceManager._build_container_config(sd, "augmentum_default")
        assert "DeviceRequests" in config["HostConfig"]

    def test_build_container_config_with_volumes(self):
        from augmentum.providers.manager import ServiceManager
        from augmentum.providers.models import ServiceCategory, ServiceDefinition

        sd = ServiceDefinition(
            id="vol-svc",
            name="Vol Test",
            description="vol",
            category=ServiceCategory.LLM,
            image="test:latest",
            internal_port=8080,
            host_port=8080,
            volumes={"my_vol": "/data"},
        )
        config = ServiceManager._build_container_config(sd, "augmentum_default")
        assert "my_vol:/data" in config["HostConfig"]["Binds"]


class TestNetwork:
    """Verify network module imports and functions exist."""

    def test_import_ensure_network(self):
        from augmentum.providers.network import ensure_network

        assert callable(ensure_network)

    def test_import_connect_container(self):
        from augmentum.providers.network import connect_container

        assert callable(connect_container)

    @pytest.mark.asyncio
    async def test_ensure_network_creates_when_missing(self, mock_docker):
        from augmentum.providers.network import ensure_network

        mock_docker.networks.list = AsyncMock(return_value=[])
        name = await ensure_network(mock_docker)
        assert "augmentum" in name
