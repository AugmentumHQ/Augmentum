"""Tests for runtime provider configuration and smart model routing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from augmentum.models.base import (
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockBackendA(ModelBackend):
    """Mock backend returning a specific model list."""

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self._models = models or [
            ModelInfo(name="modelA", model="modelA", size=1000, digest="aaa", modified_at="")
        ]

    async def chat(self, request) -> InternalChatResponse:
        return InternalChatResponse(
            message=Message(role="assistant", content="Response from A"),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def chat_stream(self, request) -> AsyncIterator[InternalStreamChunk]:
        yield InternalStreamChunk(content_delta="A", model=request.model, done=True)

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(details={"backend": "A"})


class MockBackendB(ModelBackend):
    """Second mock backend."""

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self._models = models or [
            ModelInfo(name="modelB", model="modelB", size=2000, digest="bbb", modified_at="")
        ]

    async def chat(self, request) -> InternalChatResponse:
        return InternalChatResponse(
            message=Message(role="assistant", content="Response from B"),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def chat_stream(self, request) -> AsyncIterator[InternalStreamChunk]:
        yield InternalStreamChunk(content_delta="B", model=request.model, done=True)

    async def list_models(self) -> list[ModelInfo]:
        return self._models

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(details={"backend": "B"})


class FlakyBackend(ModelBackend):
    """Backend whose ``list_models`` raises when ``fail`` is set — models a
    cloud provider whose /models probe times out or errors on a given round."""

    def __init__(self, models: list[ModelInfo] | None = None) -> None:
        self._models = models or [
            ModelInfo(name="flakyModel", model="flakyModel", size=300, digest="ccc", modified_at="")
        ]
        self.fail = False

    async def chat(self, request) -> InternalChatResponse:
        return InternalChatResponse(
            message=Message(role="assistant", content="Response from flaky"),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def chat_stream(self, request) -> AsyncIterator[InternalStreamChunk]:
        yield InternalStreamChunk(content_delta="F", model=request.model, done=True)

    async def list_models(self) -> list[ModelInfo]:
        if self.fail:
            raise RuntimeError("simulated probe failure")
        return self._models

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(details={"backend": "flaky"})


# ---------------------------------------------------------------------------
# ProviderStore CRUD tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def store_conn():
    """Create an in-memory SQLite database with the providers schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row

    # Create schema_version table and providers table
    await conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now')),
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS providers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            api_key TEXT,
            provider_type TEXT NOT NULL DEFAULT 'openai',
            is_enabled INTEGER NOT NULL DEFAULT 1,
            is_default INTEGER NOT NULL DEFAULT 0,
            profile_id TEXT NOT NULL DEFAULT '',
            owner_user_id TEXT DEFAULT '',
            shared INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    await conn.commit()
    yield conn
    await conn.close()


class TestProviderStore:
    async def test_create_and_get(self, store_conn):
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        config = ProviderConfig(
            id="lmstudio",
            name="LM Studio",
            base_url="http://localhost:1234/v1",
            api_key="sk-test",
        )
        created = await store.create_provider(config)
        assert created.id == "lmstudio"
        assert created.name == "LM Studio"
        assert created.base_url == "http://localhost:1234/v1"
        assert created.api_key == "sk-test"
        assert created.is_enabled is True

        got = await store.get_provider("lmstudio")
        assert got is not None
        assert got.name == "LM Studio"

    async def test_list_providers(self, store_conn):
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        await store.create_provider(ProviderConfig(id="a", name="A", base_url="http://a"))
        await store.create_provider(ProviderConfig(id="b", name="B", base_url="http://b", is_enabled=False))

        all_providers = await store.list_providers()
        assert len(all_providers) == 2

        enabled = await store.list_providers(enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].id == "a"

    async def test_update_provider(self, store_conn):
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        await store.create_provider(ProviderConfig(id="x", name="X", base_url="http://x"))

        updated = await store.update_provider("x", name="Updated X", base_url="http://new-x")
        assert updated is not None
        assert updated.name == "Updated X"
        assert updated.base_url == "http://new-x"

    async def test_update_nonexistent(self, store_conn):
        from augmentum.state.provider_store import ProviderStore

        store = ProviderStore(store_conn)
        result = await store.update_provider("nope", name="Foo")
        assert result is None

    async def test_delete_provider(self, store_conn):
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        await store.create_provider(ProviderConfig(id="del", name="Del", base_url="http://del"))

        assert await store.delete_provider("del") is True
        assert await store.get_provider("del") is None

    async def test_delete_nonexistent(self, store_conn):
        from augmentum.state.provider_store import ProviderStore

        store = ProviderStore(store_conn)
        assert await store.delete_provider("nope") is False

    async def test_profile_id_persists(self, store_conn):
        """Profile selection must survive a create + read round trip.

        Regression: providers used to discard profile_id at the DB layer,
        so NVIDIA/DeepSeek-bound providers lost their post-processing
        rules across restarts and any narrative-mode request 400'd.
        """
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        await store.create_provider(ProviderConfig(
            id="my-nvidia",
            name="My NVIDIA",
            base_url="https://integrate.api.nvidia.com/v1",
            profile_id="nvidia",
        ))
        got = await store.get_provider("my-nvidia")
        assert got is not None
        assert got.profile_id == "nvidia"

    async def test_profile_id_default_empty(self, store_conn):
        """Providers created without a profile expose profile_id=''."""
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        await store.create_provider(ProviderConfig(
            id="custom",
            name="Custom",
            base_url="http://localhost:8000",
        ))
        got = await store.get_provider("custom")
        assert got is not None
        assert got.profile_id == ""

    async def test_update_changes_profile_id(self, store_conn):
        """update_provider can promote an existing provider to a profile."""
        from augmentum.state.provider_store import ProviderConfig, ProviderStore

        store = ProviderStore(store_conn)
        await store.create_provider(ProviderConfig(
            id="upgrade",
            name="Upgrade",
            base_url="https://integrate.api.nvidia.com/v1",
        ))
        updated = await store.update_provider("upgrade", profile_id="nvidia")
        assert updated is not None
        assert updated.profile_id == "nvidia"


# ---------------------------------------------------------------------------
# Profile resolution tests
# ---------------------------------------------------------------------------


class TestProfileResolution:
    """get_profile_for_url drives the legacy fallback path on boot.

    Without it, providers added before migration 112 (or by users who
    skipped the dropdown) wouldn't get post-processing applied — exactly
    the failure mode that produced "System message must be at the
    beginning" 400s from NVIDIA in narrative mode.
    """

    def test_matches_nvidia_by_host(self):
        from augmentum.models.provider_profiles import get_profile_for_url

        profile = get_profile_for_url("https://integrate.api.nvidia.com/v1")
        assert profile is not None
        assert profile.id == "nvidia"
        assert profile.post_process == "semi"

    def test_matches_ignoring_path_variations(self):
        from augmentum.models.provider_profiles import get_profile_for_url

        # Trailing slash, different path — host is the only stable token.
        for url in (
            "https://integrate.api.nvidia.com/",
            "https://integrate.api.nvidia.com/v1/",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ):
            profile = get_profile_for_url(url)
            assert profile is not None and profile.id == "nvidia", url

    def test_matches_deepseek(self):
        from augmentum.models.provider_profiles import get_profile_for_url

        profile = get_profile_for_url("https://api.deepseek.com/beta")
        assert profile is not None
        assert profile.id == "deepseek"

    def test_unknown_host_returns_none(self):
        from augmentum.models.provider_profiles import get_profile_for_url

        assert get_profile_for_url("https://my-private-proxy.example.com/v1") is None

    def test_empty_url_returns_none(self):
        from augmentum.models.provider_profiles import get_profile_for_url

        assert get_profile_for_url("") is None

    def test_azure_empty_base_url_does_not_match_anything(self):
        """Profiles with empty base_url (Azure) must not match unrelated URLs."""
        from augmentum.models.provider_profiles import get_profile_for_url

        # Azure's profile has base_url="" — guard against false positives.
        assert get_profile_for_url("https://anything.example.com/") is None


# ---------------------------------------------------------------------------
# ProviderRegistry runtime tests
# ---------------------------------------------------------------------------


class TestProviderRegistryRuntime:
    @patch("augmentum.models.provider_registry.settings")
    def test_register_and_unregister(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        backend_b = MockBackendB()
        registry.register_backend("custom", backend_b)

        assert "custom" in registry.backends
        assert registry.get_backend("custom") is backend_b

        registry.unregister_backend("custom")
        assert "custom" not in registry.backends

    @patch("augmentum.models.provider_registry.settings")
    def test_cannot_unregister_default(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        with pytest.raises(ValueError, match="Cannot unregister"):
            registry.unregister_backend("ollama")

    @patch("augmentum.models.provider_registry.settings")
    async def test_refresh_model_map(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        # Replace ollama with a mock
        registry._backends["ollama"] = MockBackendA()
        registry.register_backend("custom", MockBackendB())

        model_map = await registry.refresh_model_map(force=True)

        assert "modelA" in model_map
        assert model_map["modelA"] == "ollama"
        assert "modelB" in model_map
        assert model_map["modelB"] == "custom"

    @patch("augmentum.models.provider_registry.settings")
    async def test_model_map_collision_disambiguation(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        # Both backends serve "shared-model"
        shared = ModelInfo(name="shared-model", model="shared-model", size=100, digest="x", modified_at="")
        registry._backends["ollama"] = MockBackendA(models=[shared])
        registry.register_backend("custom", MockBackendB(models=[shared]))

        model_map = await registry.refresh_model_map(force=True)

        # Should disambiguate with @backend suffix
        assert "shared-model@ollama" in model_map
        assert "shared-model@custom" in model_map
        assert "shared-model" not in model_map

    @patch("augmentum.models.provider_registry.settings")
    async def test_model_map_carries_forward_failed_probe(self, mock_settings):
        """A transient probe failure must NOT wipe a backend's models from the
        map — regression for the cold-start-after-restart staleness where a
        cloud provider's models vanished and the companion fell through to
        ``model_not_in_map_using_default``."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        registry._backends["ollama"] = MockBackendA()
        flaky = FlakyBackend()
        registry.register_backend("cloud", flaky)

        # Healthy round — both present and VERIFIED.
        model_map = await registry.refresh_model_map(force=True)
        assert model_map.get("modelA") == "ollama"
        assert model_map.get("flakyModel") == "cloud"
        assert registry.model_freshness("flakyModel") == "verified"
        assert registry.is_model_verified("modelA") is True

        # Degraded round — cloud probe raises. Its model must persist but flip
        # to the UNVERIFIED tier; the healthy backend stays verified.
        flaky.fail = True
        model_map = await registry.refresh_model_map(force=True)
        assert model_map.get("modelA") == "ollama"
        assert model_map.get("flakyModel") == "cloud", (
            "failed-probe backend's last-known-good catalog should be retained"
        )
        assert registry.model_freshness("flakyModel") == "unverified"
        assert registry.model_freshness("modelA") == "verified"

        # Recovery round — cloud healthy again, present and VERIFIED again.
        flaky.fail = False
        model_map = await registry.refresh_model_map(force=True)
        assert model_map.get("flakyModel") == "cloud"
        assert registry.model_freshness("flakyModel") == "verified"

        # A model nobody ever advertised is the absent (unknown) tier.
        assert registry.model_freshness("never-existed") == "unknown"
        assert registry.is_model_verified("never-existed") is False

    @patch("augmentum.models.provider_registry.settings")
    async def test_boot_catalog_survives_startup_refresh_race(self, mock_settings):
        """Regression: cloud/runtime backends register SECONDS after the on-disk
        model-map cache loads at startup. A ``refresh_model_map`` that runs in
        that window must NOT drop their cached models — and must not persist a
        degraded map that would poison the cache across every future restart
        (the reported "cloud model not in map after restart" class). The model
        is carried forward from the immutable boot snapshot, UNVERIFIED, until
        the backend registers and a probe verifies it."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        registry._backends["ollama"] = MockBackendA()

        # Simulate the on-disk cache having loaded a cloud backend's catalog at
        # startup, BEFORE that backend registered.
        registry._boot_catalog = {"modelA": "ollama", "flakyModel": "cloud"}
        registry._model_map = dict(registry._boot_catalog)
        registry._model_unverified = set(registry._boot_catalog)

        # Early refresh races the cloud backend's registration: only ollama is
        # registered. The cloud model must SURVIVE (carried from the boot
        # snapshot) and stay routable, marked unverified.
        m = await registry.refresh_model_map(force=True)
        assert m.get("flakyModel") == "cloud", (
            "startup-race refresh must NOT drop an unregistered backend's "
            "cached model (else the degraded map gets persisted and poisons "
            "the cache across restarts)"
        )
        assert registry.model_freshness("flakyModel") == "unverified"

        # Once the backend registers and its probe responds, it verifies.
        registry.register_backend("cloud", FlakyBackend())
        m = await registry.refresh_model_map(force=True)
        assert m.get("flakyModel") == "cloud"
        assert registry.model_freshness("flakyModel") == "verified"
        backend, clean = await registry.resolve_backend_for_model("flakyModel")
        assert clean == "flakyModel"
        assert isinstance(backend, FlakyBackend)

    @patch("augmentum.models.provider_registry.settings")
    async def test_model_map_degraded_round_shortens_ttl(self, mock_settings):
        """A degraded round backdates the cache timestamp so the next caller
        re-probes within the short retry window instead of the full TTL."""
        import time

        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import (
            _MODEL_MAP_DEGRADED_RETRY_S,
            _MODEL_MAP_TTL,
            ProviderRegistry,
        )

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        registry._backends["ollama"] = MockBackendA()
        flaky = FlakyBackend()
        flaky.fail = True
        registry.register_backend("cloud", flaky)

        await registry.refresh_model_map(force=True)

        # ts was backdated by ~(TTL - retry); cache is already near-expiry.
        age = time.monotonic() - registry._model_map_ts
        assert age >= (_MODEL_MAP_TTL - _MODEL_MAP_DEGRADED_RETRY_S) - 1, (
            "degraded round should backdate _model_map_ts for a fast retry"
        )

    @patch("augmentum.models.provider_registry.settings")
    async def test_is_cloud_backend_classifies_by_base_url(self, mock_settings):
        """Fix #2/#3 helper: locality comes from the backend's _base_url —
        no base_url or a loopback/RFC-1918/docker host is local; an external
        host is cloud."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        registry._backends["local_nobaseurl"] = MockBackendA()  # no _base_url
        loopback = FlakyBackend()
        loopback._base_url = "http://127.0.0.1:8091/v1"
        registry._backends["loopback"] = loopback
        cloud = FlakyBackend()
        cloud._base_url = "https://openrouter.ai/api/v1"
        registry._backends["cloud"] = cloud

        assert registry._is_cloud_backend("local_nobaseurl") is False
        assert registry._is_cloud_backend("loopback") is False
        assert registry._is_cloud_backend("cloud") is True
        # Unknown key (no backend) is local by default — never starves the probe.
        assert registry._is_cloud_backend("ghost") is False

    @patch("augmentum.models.provider_registry.settings")
    async def test_cloud_failure_fast_retry_then_settles(self, mock_settings):
        """Fix #3 (cold-start-aware): the FIRST cloud-only degraded round fast-
        retries so a cold catalog (deepseek/openrouter) recovers in ~5s, but a
        REPEATED identical failure (chronic) backs off to the full TTL — no 5s
        re-probe storm against a permanently-slow cloud /models."""
        import time

        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""
        mock_settings.data_dir = "/nonexistent-test-dir"  # persist is best-effort

        from augmentum.models.provider_registry import (
            _MODEL_MAP_DEGRADED_RETRY_S,
            _MODEL_MAP_TTL,
            ProviderRegistry,
        )

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        # Isolate from the keyless backends auto-registered at init (they all
        # fail probing and would pollute the failed-set assertion).
        registry._backends.clear()
        registry._backends["ollama"] = MockBackendA()  # local, succeeds
        cloud = FlakyBackend()
        cloud.fail = True
        cloud._base_url = "https://openrouter.ai/api/v1"
        registry.register_backend("cloud", cloud)

        # Round 1: brand-new failure → fast retry (backdated ts).
        await registry.refresh_model_map(force=True)
        age1 = time.monotonic() - registry._model_map_ts
        assert age1 >= (_MODEL_MAP_TTL - _MODEL_MAP_DEGRADED_RETRY_S) - 1, (
            "first cloud failure should fast-retry to recover a cold catalog"
        )
        assert registry._degraded_round_logged == frozenset({"cloud"})

        # Round 2: same failure set (chronic) → ride the full TTL, no storm.
        await registry.refresh_model_map(force=True)
        age2 = time.monotonic() - registry._model_map_ts
        assert age2 < 2, "chronic cloud failure must NOT keep fast-retrying"

    @patch("augmentum.models.provider_registry.settings")
    async def test_model_map_persists_and_reloads(self, mock_settings, tmp_path):
        """Fix A: a verified map is written to disk and reloaded at next
        construction as the UNVERIFIED last-known-good tier — so the full
        catalog is resolvable immediately after a restart (no cold-start gap)."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""
        mock_settings.data_dir = str(tmp_path)

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        registry._backends.clear()
        registry._backends["ollama"] = MockBackendA(
            models=[ModelInfo(name="deepseek-v4-pro", model="deepseek-v4-pro",
                              size=1, digest="d", modified_at="")]
        )
        registry._backends["ollama"]._base_url = "http://localhost:11434"

        m = await registry.refresh_model_map(force=True)
        assert "deepseek-v4-pro" in m
        # The cache file was written.
        cache = tmp_path / "model_map_cache.json"
        assert cache.exists(), "verified map should be persisted to disk"

        # A FRESH registry (simulating a restart) loads it immediately, marked
        # unverified, resolvable before any probe runs.
        registry2 = ProviderRegistry(client)
        assert registry2._model_map.get("deepseek-v4-pro") == "ollama"
        assert registry2.model_freshness("deepseek-v4-pro") == "unverified"
        assert registry2._model_map_ts == 0, "must still re-probe to confirm"

    @patch("augmentum.models.provider_registry.settings")
    async def test_models_changed_event_published_on_change(self, mock_settings, tmp_path):
        """Fix C: the catalog filling in / verifying publishes ``models.changed``
        so connected UIs refetch without a manual double-refresh."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""
        mock_settings.data_dir = str(tmp_path)

        from augmentum.models import provider_registry as pr

        client = MagicMock(spec=httpx.AsyncClient)
        registry = pr.ProviderRegistry(client)
        registry._backends.clear()
        registry._backends["ollama"] = MockBackendA()

        with patch("augmentum.proxy.system_events.publish") as mock_pub:
            await registry.refresh_model_map(force=True)  # cold → map appears
            topics = [c.args[0] for c in mock_pub.call_args_list if c.args]
            assert "models.changed" in topics
            # No change second round → no duplicate publish.
            mock_pub.reset_mock()
            await registry.refresh_model_map(force=True)
            topics2 = [c.args[0] for c in mock_pub.call_args_list if c.args]
            assert "models.changed" not in topics2

    @patch("augmentum.models.provider_registry.settings")
    async def test_degraded_round_warning_dedups(self, mock_settings):
        """Fix #1: an unchanged unhealthy set warns once then drops to debug,
        so a chronically slow backend stops machine-gunning the log."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models import provider_registry as pr

        client = MagicMock(spec=httpx.AsyncClient)
        registry = pr.ProviderRegistry(client)
        registry._backends.clear()  # isolate from keyless auto-registered backends
        registry._backends["ollama"] = MockBackendA()
        cloud = FlakyBackend()
        cloud.fail = True
        cloud._base_url = "https://openrouter.ai/api/v1"
        registry.register_backend("cloud", cloud)

        def _count(mock_log, level):
            return sum(
                1
                for c in getattr(mock_log, level).call_args_list
                if c.args and c.args[0] == "model_map_degraded_round"
            )

        with patch.object(pr, "log") as mock_log:
            await registry.refresh_model_map(force=True)  # round 1: warn
            await registry.refresh_model_map(force=True)  # round 2: same set -> debug
            assert _count(mock_log, "warning") == 1, "degraded round should warn exactly once"
            assert _count(mock_log, "debug") >= 1, "repeat round should log at debug"

            # Recovery resets the dedup so a future degradation warns afresh.
            cloud.fail = False
            await registry.refresh_model_map(force=True)
            assert registry._degraded_round_logged is None
            cloud.fail = True
            await registry.refresh_model_map(force=True)
            assert _count(mock_log, "warning") == 2, "warns again after recovery"

    @patch("augmentum.models.provider_registry.settings")
    async def test_model_map_drops_unregistered_backend_on_failure(self, mock_settings):
        """A DELIBERATELY-removed provider's stale models must not linger. The
        real removal path is ``unregister_backend``, which purges the backend
        from both the live map and the boot snapshot so carry-forward stops
        re-seeding it (vs. a not-yet-registered backend during startup, which
        IS carried — see test_boot_catalog_survives_startup_refresh_race)."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        registry._backends["ollama"] = MockBackendA()
        flaky = FlakyBackend()
        registry.register_backend("cloud", flaky)

        await registry.refresh_model_map(force=True)  # seed last-known-good

        # Deliberately remove the cloud backend via the real unregister path,
        # AND make a different backend fail so the degraded round runs; the
        # removed backend's model must NOT be carried forward.
        registry.unregister_backend("cloud")
        registry._backends["ollama"] = FlakyBackend(
            models=[ModelInfo(name="modelA", model="modelA", size=1, digest="a", modified_at="")]
        )
        registry._backends["ollama"].fail = True

        model_map = await registry.refresh_model_map(force=True)
        assert "flakyModel" not in model_map

    @patch("augmentum.models.provider_registry.settings")
    async def test_resolver_traces_unverified_once(self, mock_settings):
        """Resolving a stale (carried-forward) model emits a coalesced trace —
        once while unverified, reset on recovery — for diagnosability without
        per-turn log spam."""
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        registry._backends["ollama"] = MockBackendA()
        flaky = FlakyBackend()
        registry.register_backend("cloud", flaky)

        await registry.refresh_model_map(force=True)  # verified baseline
        assert not registry._unverified_served_logged

        # Degrade: cloud unresponsive → flakyModel becomes unverified.
        flaky.fail = True
        await registry.refresh_model_map(force=True)
        assert registry.model_freshness("flakyModel") == "unverified"

        # First resolve traces + records the dedup marker.
        _, name = await registry.resolve_backend_for_model("flakyModel")
        assert name == "flakyModel"
        assert "flakyModel" in registry._unverified_served_logged

        # Second resolve is silent (dedup set unchanged — no spam).
        await registry.resolve_backend_for_model("flakyModel")
        assert registry._unverified_served_logged == {"flakyModel"}

        # Recovery clears the marker so a future degradation traces afresh.
        flaky.fail = False
        await registry.refresh_model_map(force=True)
        assert "flakyModel" not in registry._unverified_served_logged

    @patch("augmentum.models.provider_registry.settings")
    async def test_resolve_backend_with_at_suffix(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        backend_b = MockBackendB()
        registry.register_backend("custom", backend_b)

        backend, clean_name = await registry.resolve_backend_for_model("mymodel@custom")
        assert backend is backend_b
        assert clean_name == "mymodel"

    @patch("augmentum.models.provider_registry.settings")
    async def test_resolve_fallback_to_default(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        backend, clean_name = await registry.resolve_backend_for_model("unknown-model")
        assert backend == registry.default_backend
        assert clean_name == "unknown-model"

    @patch("augmentum.models.provider_registry.settings")
    async def test_resolve_via_model_map(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        backend_b = MockBackendB()
        registry._backends["ollama"] = MockBackendA()
        registry.register_backend("custom", backend_b)
        await registry.refresh_model_map(force=True)

        backend, clean_name = await registry.resolve_backend_for_model("modelB")
        assert backend is backend_b
        assert clean_name == "modelB"

    @patch("augmentum.models.provider_registry.settings")
    async def test_load_runtime_providers(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.state.provider_store import ProviderConfig

        mock_store = MagicMock()
        mock_store.list_providers = AsyncMock(return_value=[
            ProviderConfig(id="rt1", name="Runtime1", base_url="http://rt1/v1"),
        ])

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)

        await registry.load_runtime_providers(mock_store)
        assert "rt1" in registry.backends

    @patch("augmentum.models.provider_registry.settings")
    async def test_model_map_ttl_cache(self, mock_settings):
        import httpx

        mock_settings.default_backend = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"
        mock_settings.openai_api_key = ""
        mock_settings.llamacpp_base_url = ""

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock(spec=httpx.AsyncClient)
        registry = ProviderRegistry(client)
        registry._backends["ollama"] = MockBackendA()

        # First call should probe
        await registry.refresh_model_map(force=True)
        ts1 = registry._model_map_ts

        # Second call (within TTL) should use cache
        await registry.refresh_model_map()
        assert registry._model_map_ts == ts1  # Same timestamp = cache hit


# ---------------------------------------------------------------------------
# Provider API endpoint tests
# ---------------------------------------------------------------------------


class TestProviderAPIEndpoints:
    def test_list_providers(self, client):
        resp = client.get("/api/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        # Should include the mocked ollama backend
        names = [p["id"] for p in data["providers"]]
        assert "ollama" in names

    def test_probe_connection(self, client, app):
        """Probe should create a temporary backend and test it."""
        # We need to mock the OpenAIBackend.list_models + bypass SSRF check for test
        with patch("augmentum.proxy.provider_routes.OpenAIBackend") as mock_cls, \
             patch("augmentum.proxy.provider_routes.check_ssrf"):
            mock_instance = MagicMock()
            mock_instance.list_models = AsyncMock(return_value=[
                ModelInfo(name="test-model", model="test-model", size=100, digest="x", modified_at=""),
            ])
            mock_cls.return_value = mock_instance

            resp = client.post("/api/providers/probe", json={
                "base_url": "http://example.com:9999/v1",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert len(data["models"]) == 1
            assert data["models"][0]["name"] == "test-model"

    def test_probe_private_url_blocked(self, client):
        """Probe blocks private/loopback URLs by default (SSRF protection)."""
        resp = client.post("/api/providers/probe", json={
            "base_url": "http://127.0.0.1:9999/v1",
        })
        assert resp.status_code == 403
        assert "SSRF" in resp.json()["error"]

    def test_probe_private_url_allowed_via_allowlist(self, client):
        """Probe allows private URLs when they appear in the SSRF allowlist."""
        with patch("augmentum.proxy.provider_routes.settings") as mock_settings:
            mock_settings.ssrf_allowlist = "127.0.0.0/8"
            resp = client.post("/api/providers/probe", json={
                "base_url": "http://127.0.0.1:9999/v1",
            })
            # Connection refused → 502, but not 403 (allowlisted)
            assert resp.status_code == 502

    def test_create_reserved_id_rejected(self, client):
        resp = client.post("/api/providers", json={
            "id": "ollama",
            "name": "Ollama Duplicate",
            "base_url": "http://localhost:11434",
        })
        assert resp.status_code == 400
        assert "reserved" in resp.json()["error"]

    def test_delete_builtin_rejected(self, client):
        resp = client.delete("/api/providers/ollama")
        assert resp.status_code == 400
        assert "built-in" in resp.json()["error"]

    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/providers/nonexistent")
        # With MemoryBackend, store is None -> 503
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Unified /api/tags tests
# ---------------------------------------------------------------------------


class TestUnifiedTags:
    def test_tags_include_backend_info(self, client, app):
        """Models should include augmentum_backend in details."""
        # The mock backend is already set up in conftest
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        models = data["models"]

        # Should have base models + prefixed variants
        base_models = [m for m in models if not m["name"].startswith(("a/", "n/", "p/"))]
        assert len(base_models) >= 1

    def test_tags_prefixed_variants(self, client, app):
        """Each model should generate a/, n/, p/ variants."""
        resp = client.get("/api/tags")
        data = resp.json()
        models = data["models"]
        names = [m["name"] for m in models]

        # Since conftest mock has one model, we should see the base + 3 prefixed = 4 total
        base_names = [n for n in names if not n.startswith(("a/", "n/", "p/"))]
        for base in base_names:
            assert f"a/{base}" in names
            assert f"n/{base}" in names
            assert f"p/{base}" in names


# ---------------------------------------------------------------------------
# Smart routing tests
# ---------------------------------------------------------------------------


class TestSmartRouting:
    def test_chat_uses_resolve_backend(self, client, app):
        """POST /api/chat should call resolve_backend_for_model."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        assert resp.status_code == 200
        app.state.provider_registry.resolve_backend_for_model.assert_called()

    def test_generate_uses_resolve_backend(self, client, app):
        """POST /api/generate should call resolve_backend_for_model."""
        resp = client.post("/api/generate", json={
            "model": "llama3.1:8b",
            "prompt": "hi",
            "stream": False,
        })
        assert resp.status_code == 200
        app.state.provider_registry.resolve_backend_for_model.assert_called()

    def test_openai_chat_uses_resolve_backend(self, client, app):
        """POST /v1/chat/completions should call resolve_backend_for_model."""
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        app.state.provider_registry.resolve_backend_for_model.assert_called()


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_ollama_only_setup_works(self, client):
        """With only the default ollama mock, everything should work as before."""
        # Chat
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"]["content"] == "Hello from mock Ollama!"

    def test_tags_work(self, client):
        resp = client.get("/api/tags")
        assert resp.status_code == 200

    def test_openai_chat_works(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 200

    def test_openai_models_works(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
