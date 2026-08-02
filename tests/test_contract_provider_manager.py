"""Contract tests for ServiceManager — verify Docker container config, lifecycle, and status."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.providers.manager import ServiceManager, _parse_size
from augmentum.providers.models import (
    GpuRequirements,
    HealthCheck,
    ServiceCategory,
    ServiceDefinition,
    ServiceStatus,
)


def _make_service_def(**overrides) -> ServiceDefinition:
    defaults = {
        "id": "test-ollama",
        "name": "Ollama",
        "description": "Local LLM server",
        "category": ServiceCategory.LLM,
        "image": "ollama/ollama:latest",
        "internal_port": 11434,
        "host_port": 11434,
    }
    defaults.update(overrides)
    return ServiceDefinition(**defaults)


class TestBuildContainerConfig:
    """Verify Docker container config is built correctly."""

    def test_basic_config_shape(self):
        sd = _make_service_def()
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert config["Image"] == "ollama/ollama:latest"
        assert "11434/tcp" in config["ExposedPorts"]
        assert "HostConfig" in config
        assert "Labels" in config
        assert config["Labels"]["augmentum.managed"] == "true"
        assert config["Labels"]["augmentum.service.id"] == "test-ollama"

    def test_port_bindings(self):
        sd = _make_service_def(internal_port=8080, host_port=9090)
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        bindings = config["HostConfig"]["PortBindings"]
        assert "8080/tcp" in bindings
        assert bindings["8080/tcp"][0]["HostPort"] == "9090"

    def test_env_vars(self):
        sd = _make_service_def(env={"CUDA_VISIBLE_DEVICES": "0", "OLLAMA_HOST": "0.0.0.0"})
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert "CUDA_VISIBLE_DEVICES=0" in config["Env"]
        assert "OLLAMA_HOST=0.0.0.0" in config["Env"]

    def test_gpu_device_requests(self):
        sd = _make_service_def(
            gpu=GpuRequirements(required=True, vram_mb=8000, driver="nvidia")
        )
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        devices = config["HostConfig"]["DeviceRequests"]
        assert len(devices) == 1
        assert devices[0]["Driver"] == "nvidia"
        assert devices[0]["Count"] == -1

    def test_no_gpu_when_not_required(self):
        sd = _make_service_def(gpu=GpuRequirements(required=False))
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert "DeviceRequests" not in config["HostConfig"]

    def test_volume_binds(self):
        sd = _make_service_def(volumes={"ollama_data": "/root/.ollama"})
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert "ollama_data:/root/.ollama" in config["HostConfig"]["Binds"]

    def test_health_check(self):
        hc = HealthCheck(
            test=["CMD", "curl", "-f", "http://localhost:11434"],
            interval_s=10,
            timeout_s=5,
            retries=3,
            start_period_s=30,
        )
        sd = _make_service_def(health_check=hc)
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert "Healthcheck" in config
        assert config["Healthcheck"]["Test"] == ["CMD", "curl", "-f", "http://localhost:11434"]
        assert config["Healthcheck"]["Retries"] == 3

    def test_custom_command(self):
        sd = _make_service_def(command=["serve", "--host", "0.0.0.0"])
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert config["Cmd"] == ["serve", "--host", "0.0.0.0"]

    def test_network_config(self):
        sd = _make_service_def()
        config = ServiceManager._build_container_config(sd, "my_network")

        endpoints = config["NetworkingConfig"]["EndpointsConfig"]
        assert "my_network" in endpoints
        assert sd.id in endpoints["my_network"]["Aliases"]

    def test_restart_policy(self):
        sd = _make_service_def()
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert config["HostConfig"]["RestartPolicy"]["Name"] == "unless-stopped"

    def test_shm_size(self):
        sd = _make_service_def(shm_size="2gb")
        config = ServiceManager._build_container_config(sd, "augmentum_default")

        assert config["HostConfig"]["ShmSize"] == 2 * 1024 * 1024 * 1024


class TestGetStatus:
    """Verify get_status returns correct ServiceStatus."""

    @pytest.mark.asyncio
    async def test_stopped_when_no_container(self, mock_docker):
        mock_docker.containers.list = AsyncMock(return_value=[])
        sm = ServiceManager(docker=mock_docker, db=None)

        status = await sm.get_status("test-svc")
        assert status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_running_when_healthy(self, mock_docker):
        container = MagicMock()
        container.show = AsyncMock(return_value={
            "State": {
                "Running": True,
                "Health": {"Status": "healthy"},
            }
        })
        mock_docker.containers.list = AsyncMock(return_value=[container])
        sm = ServiceManager(docker=mock_docker, db=None)

        status = await sm.get_status("test-svc")
        assert status == ServiceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_unhealthy_status(self, mock_docker):
        container = MagicMock()
        container.show = AsyncMock(return_value={
            "State": {
                "Running": True,
                "Health": {"Status": "unhealthy"},
            }
        })
        mock_docker.containers.list = AsyncMock(return_value=[container])
        sm = ServiceManager(docker=mock_docker, db=None)

        status = await sm.get_status("test-svc")
        assert status == ServiceStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_starting_when_running_no_health(self, mock_docker):
        container = MagicMock()
        container.show = AsyncMock(return_value={
            "State": {"Running": True, "Health": {}},
        })
        mock_docker.containers.list = AsyncMock(return_value=[container])
        sm = ServiceManager(docker=mock_docker, db=None)

        status = await sm.get_status("test-svc")
        assert status == ServiceStatus.STARTING

    @pytest.mark.asyncio
    async def test_error_on_exception(self, mock_docker):
        container = MagicMock()
        container.show = AsyncMock(side_effect=Exception("Docker error"))
        mock_docker.containers.list = AsyncMock(return_value=[container])
        sm = ServiceManager(docker=mock_docker, db=None)

        status = await sm.get_status("test-svc")
        assert status == ServiceStatus.ERROR


class TestDisableService:
    """Verify disable_service stops and removes container."""

    @pytest.mark.asyncio
    async def test_disable_stops_and_deletes(self, mock_docker):
        container = MagicMock()
        container.stop = AsyncMock()
        container.delete = AsyncMock()
        mock_docker.containers.list = AsyncMock(return_value=[container])

        sm = ServiceManager(docker=mock_docker, db=None)
        await sm.disable_service("test-svc")

        container.stop.assert_called_once()
        container.delete.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_disable_no_container(self, mock_docker):
        mock_docker.containers.list = AsyncMock(return_value=[])
        sm = ServiceManager(docker=mock_docker, db=None)

        # Should not raise
        await sm.disable_service("nonexistent")


class TestParseSize:
    """Verify Docker size string parsing."""

    def test_parse_gb(self):
        assert _parse_size("2gb") == 2 * 1024 * 1024 * 1024

    def test_parse_mb(self):
        assert _parse_size("512mb") == 512 * 1024 * 1024

    def test_parse_g(self):
        assert _parse_size("4g") == 4 * 1024 * 1024 * 1024

    def test_parse_raw_bytes(self):
        assert _parse_size("1024") == 1024
