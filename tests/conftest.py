"""Shared test fixtures."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from augmentum.models.base import (
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)


class MockOllamaBackend(ModelBackend):
    """Mock Ollama backend that returns canned responses."""

    async def chat(self, request) -> InternalChatResponse:
        return InternalChatResponse(
            message=Message(role="assistant", content="Hello from mock Ollama!"),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def chat_stream(self, request) -> AsyncIterator[InternalStreamChunk]:
        words = ["Hello", " from", " mock", " Ollama", "!"]
        for i, word in enumerate(words):
            yield InternalStreamChunk(
                content_delta=word,
                role="assistant" if i == 0 else None,
                model=request.model,
                done=False,
            )
        yield InternalStreamChunk(
            content_delta="",
            model=request.model,
            done=True,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                name="llama3.1:8b",
                model="llama3.1:8b",
                size=4_000_000_000,
                digest="abc123",
                modified_at="2024-01-01T00:00:00Z",
                details={"family": "llama", "parameter_size": "8B"},
            ),
        ]

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(
            modelfile="FROM llama3.1:8b",
            parameters="temperature 0.7",
            template="",
            details={"family": "llama"},
        )


@pytest.fixture(autouse=True)
def _enable_image_for_tests():
    """Ensure image_enabled is True for all tests (default is False for minimal startup)."""
    from augmentum.config import settings

    orig = settings.image_enabled
    object.__setattr__(settings, "image_enabled", True)
    yield
    object.__setattr__(settings, "image_enabled", orig)


def pytest_configure(config):
    """Set PYTHONIOENCODING early so structlog box-drawing chars don't crash on Windows cp1252.

    This must run before test collection to influence any streams opened during import.
    """
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"


@pytest.fixture
def mock_backend():
    """Provide a mock Ollama backend."""
    return MockOllamaBackend()


@pytest.fixture
def test_user():
    """The default User injected into route requests by the `client` fixture.

    Role is ``admin`` so existing tests that touch shared-infrastructure
    endpoints (providers, balancers, MCP, knowledge packs, managed
    services) don't need per-test role setup. Non-admin behaviors are
    verified by tests that override this fixture with ``role="user"``.
    """
    from augmentum.auth.models import User

    return User(
        id="usr_test",
        username="tester",
        display_name="Test User",
        role="admin",
        is_active=True,
    )


@pytest.fixture
def test_nonadmin_user():
    """A non-admin User for tests that exercise admin-gated routes' 403 path."""
    from augmentum.auth.models import User

    return User(
        id="usr_nonadmin",
        username="rando",
        display_name="Non-Admin",
        role="user",
        is_active=True,
    )


@pytest.fixture
def app(mock_backend, test_user):
    """Create a test FastAPI app with mocked backend."""
    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.proxy.server import create_app
    from augmentum.state.backends.memory import MemoryBackend
    from augmentum.state.manager import StateManager

    application = create_app()

    from augmentum.classifier.router import RequestClassifier

    # Auth: install a mock session_manager that accepts the fixture bearer
    # token. Without this, AuthMiddleware fails closed with 503 on every
    # non-public route. The `client` fixture sends the matching Authorization
    # header so route tests can exercise their handlers.
    mock_sm = MagicMock()
    mock_sm.validate_token = AsyncMock(return_value=test_user)
    mock_sm.get_user_by_id = AsyncMock(return_value=test_user)
    mock_sm.validate_ws_ticket = MagicMock(return_value=test_user.id)
    application.state.session_manager = mock_sm

    # Override lifespan resources manually
    application.state.http_client = MagicMock()
    application.state.provider_registry = MagicMock(spec=ProviderRegistry)
    application.state.provider_registry.get_backend.return_value = mock_backend
    application.state.provider_registry.default_backend = mock_backend
    application.state.provider_registry.resolve_backend_for_model = AsyncMock(return_value=(mock_backend, "llama3.1:8b"))
    application.state.provider_registry.refresh_model_map = AsyncMock(return_value={})
    application.state.provider_registry.register_backend = MagicMock()
    application.state.provider_registry.unregister_backend = MagicMock()
    application.state.provider_registry.backends = {"ollama": mock_backend}
    # Per-backend probe deadline (provider_registry.probe_deadline_for) must be
    # a real number — /api/tags passes it straight to asyncio.wait_for(timeout=).
    application.state.provider_registry.probe_deadline_for = MagicMock(return_value=6.0)
    application.state.state_manager = StateManager(MemoryBackend())
    application.state.classifier = RequestClassifier()
    application.state.narrative_engines = {}

    from augmentum.cache.dedup import RequestDeduplicator
    from augmentum.cache.prefix_cache import PrefixCache
    from augmentum.cache.prompt_cache import PromptCache
    from augmentum.tools.registry import ToolRegistry

    application.state.tool_registry = ToolRegistry()
    application.state.prompt_cache = PromptCache()
    application.state.prefix_cache = PrefixCache()
    application.state.request_deduplicator = RequestDeduplicator()

    # Model manager (uses mock provider registry)
    mock_manager = MagicMock()
    mock_manager.list_all_models = AsyncMock(return_value=[
        ModelInfo(name="llama3.1:8b", model="llama3.1:8b", size=4_000_000_000,
                  digest="abc123", modified_at="2024-01-01T00:00:00Z"),
    ])
    mock_manager.get_running_models = AsyncMock(return_value=[])
    mock_manager.get_model_status = AsyncMock()
    mock_manager.load_model = AsyncMock(return_value=True)
    mock_manager.unload_model = AsyncMock(return_value=True)
    application.state.model_manager = mock_manager

    return application


@pytest.fixture
def client(app):
    """Create a TestClient that auto-sends the test bearer token.

    The mock session_manager installed in `app` accepts any token string, so
    the exact value here only matters for consistency.
    """
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    return tc


# ---------------------------------------------------------------------------
# Shared fixtures for comprehensive test suite
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESPONSES_DIR = FIXTURES_DIR / "responses"
AUDIO_DIR = FIXTURES_DIR / "audio_samples"


def _load_fixture(name: str) -> dict:
    """Load a JSON fixture file from tests/fixtures/responses/."""
    return json.loads((RESPONSES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def load_fixture():
    """Fixture that returns a helper to load canned JSON responses."""
    return _load_fixture


@pytest.fixture
def sqlite_client(app):
    """Client with real SQLite backend for stateful tests.

    Fresh in-memory DB per test — no cleanup needed.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())

    async def _seed_auth_users() -> None:
        # The mock session manager authenticates as usr_test/usr_nonadmin
        # without ever inserting them — but user-scoped tables FK
        # users(id), and FK enforcement is real since the post-migration
        # pragma re-apply (2026-07-18). Seed both so route writes work.
        for uid in ("usr_test", "usr_nonadmin"):
            await backend.conn.execute(
                "INSERT OR IGNORE INTO users "
                "(id, username, display_name, password_hash, role) "
                "VALUES (?, ?, ?, 'pw', 'user')",
                (uid, uid, uid),
            )
        await backend.conn.commit()

    asyncio.get_event_loop().run_until_complete(_seed_auth_users())
    app.state.state_manager = StateManager(backend)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc
    asyncio.get_event_loop().run_until_complete(backend.close())


@pytest.fixture
def mock_docker():
    """Mock aiodocker.Docker client for container/network tests."""
    docker = MagicMock()
    # Containers
    docker.containers.run = AsyncMock(return_value=MagicMock(id="cnt_test123"))
    docker.containers.get = AsyncMock(return_value=MagicMock(
        id="cnt_test123",
        show=AsyncMock(return_value={"State": {"Status": "running", "Running": True}}),
        stop=AsyncMock(),
        delete=AsyncMock(),
        log=AsyncMock(return_value=["log line 1", "log line 2"]),
    ))
    docker.containers.list = AsyncMock(return_value=[])
    # Images
    docker.images.inspect = AsyncMock(return_value={"Id": "sha256:abc123"})
    docker.images.pull = AsyncMock()
    # Networks
    docker.networks.list = AsyncMock(return_value=[])
    docker.networks.create = AsyncMock(return_value=MagicMock(id="net_test"))
    docker.networks.get = AsyncMock(return_value=MagicMock(
        connect=AsyncMock(),
        disconnect=AsyncMock(),
    ))
    return docker


@pytest.fixture
def audio_silence():
    """1 second of 16-bit PCM silence at 16kHz (32000 bytes)."""
    return b"\x00" * 32000


@pytest.fixture
def audio_tone():
    """1 second of 440Hz sine wave at 16kHz, 16-bit PCM."""
    import math
    import struct
    samples = []
    for i in range(16000):
        val = int(16000 * math.sin(2 * math.pi * 440 * i / 16000))
        samples.append(struct.pack("<h", val))
    return b"".join(samples)


# ---------------------------------------------------------------------------
# Live test support
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Add --run-live flag for live integration tests."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run live integration tests requiring external services",
    )


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless --run-live is passed."""
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="need --run-live option to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
