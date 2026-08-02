"""Live test fixtures — skip when services unavailable."""

from __future__ import annotations

import pytest
import httpx


def _probe_service(url: str, timeout: float = 2.0) -> bool:
    """Check if a service is reachable."""
    try:
        resp = httpx.get(url, timeout=timeout, verify=False)
        return resp.status_code < 500
    except Exception:
        return False


@pytest.fixture
def live_ollama():
    """Async httpx client for live Ollama instance."""
    base = "http://localhost:11434"
    if not _probe_service(f"{base}/"):
        pytest.skip("Ollama not available at localhost:11434")
    client = httpx.AsyncClient(base_url=base, timeout=30.0)
    yield client
    # cleanup handled by pytest-asyncio


@pytest.fixture
def live_searxng():
    """Async httpx client for live SearXNG instance."""
    base = "http://localhost:8080"
    if not _probe_service(f"{base}/"):
        pytest.skip("SearXNG not available at localhost:8080")
    client = httpx.AsyncClient(base_url=base, timeout=15.0)
    yield client


@pytest.fixture
def live_augmentum():
    """Async httpx client for live Augmentum instance."""
    base = "http://localhost:6100"
    if not _probe_service(f"{base}/"):
        pytest.skip("Augmentum not available at localhost:6100")
    client = httpx.AsyncClient(base_url=base, timeout=30.0)
    yield client


@pytest.fixture
def live_kokoro():
    """Async httpx client for live Kokoro TTS instance."""
    base = "http://localhost:8880"
    if not _probe_service(f"{base}/"):
        pytest.skip("Kokoro TTS not available at localhost:8880")
    client = httpx.AsyncClient(base_url=base, timeout=30.0)
    yield client


@pytest.fixture
def live_executor():
    """Async httpx client for live code executor instance."""
    base = "http://localhost:5000"
    if not _probe_service(f"{base}/"):
        pytest.skip("Executor not available at localhost:5000")
    client = httpx.AsyncClient(base_url=base, timeout=30.0)
    yield client


@pytest.fixture
def live_docker():
    """Live aiodocker client — skip if Docker not available."""
    try:
        import aiodocker
    except ImportError:
        pytest.skip("aiodocker not installed")
    import asyncio
    try:
        docker = asyncio.get_event_loop().run_until_complete(
            aiodocker.Docker().__aenter__()
        )
        yield docker
        asyncio.get_event_loop().run_until_complete(docker.close())
    except Exception:
        pytest.skip("Docker not available")
