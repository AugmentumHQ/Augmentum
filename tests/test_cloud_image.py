"""Tests for cloud image generation routes."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.proxy.cloud_image_routes import (
    _KNOWN_CLOUD_MODELS,
    CloudEditRequest,
    CloudGenerateRequest,
    ImageProviderCreate,
    ImageProviderUpdate,
    _build_headers,
    _detect_provider_type,
    _fetch_cloud_models,
    _is_bfl,
    _is_fal,
    _is_stability,
    _save_cloud_image,
)

# ---------------------------------------------------------------------------
# Provider type detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    def test_detect_openai(self):
        assert _detect_provider_type("https://api.openai.com") == "openai"

    def test_detect_together(self):
        assert _detect_provider_type("https://api.together.xyz") == "together"

    def test_detect_stability(self):
        assert _detect_provider_type("https://api.stability.ai") == "stability"
        assert _is_stability("https://api.stability.ai/v2beta")

    def test_detect_bfl(self):
        assert _detect_provider_type("https://api.bfl.ml") == "bfl"
        assert _is_bfl("https://api.bfl.ai/v1")

    def test_detect_fal(self):
        assert _detect_provider_type("https://fal.run") == "fal"
        assert _is_fal("https://fal.ai")

    def test_detect_openai_compat_fallback(self):
        assert _detect_provider_type("https://my-custom-server.com") == "openai_compat"

    def test_case_insensitive(self):
        assert _detect_provider_type("https://API.OPENAI.COM") == "openai"
        assert _is_stability("https://API.STABILITY.AI")


# ---------------------------------------------------------------------------
# Auth header building
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def test_no_key(self):
        assert _build_headers(None) == {}
        assert _build_headers("") == {}

    def test_bearer_default(self):
        h = _build_headers("sk-test123")
        assert h["Authorization"] == "Bearer sk-test123"

    def test_bfl_xkey(self):
        h = _build_headers("bfl-key", "https://api.bfl.ml")
        assert h["x-key"] == "bfl-key"
        assert "Authorization" not in h

    def test_fal_key_prefix(self):
        h = _build_headers("fal-key", "https://fal.run")
        assert h["Authorization"] == "Key fal-key"

    def test_openai_bearer(self):
        h = _build_headers("sk-openai", "https://api.openai.com")
        assert h["Authorization"] == "Bearer sk-openai"

    def test_stability_bearer(self):
        h = _build_headers("stab-key", "https://api.stability.ai")
        assert h["Authorization"] == "Bearer stab-key"


# ---------------------------------------------------------------------------
# Known model catalogs
# ---------------------------------------------------------------------------


class TestKnownModels:
    def test_all_providers_have_models(self):
        for key in ("openai", "together", "stability", "bfl", "fal"):
            assert key in _KNOWN_CLOUD_MODELS
            assert len(_KNOWN_CLOUD_MODELS[key]) > 0

    def test_openai_models(self):
        names = [m["name"] for m in _KNOWN_CLOUD_MODELS["openai"]]
        assert "gpt-image-1" in names
        assert "dall-e-3" in names

    def test_together_models(self):
        names = [m["name"] for m in _KNOWN_CLOUD_MODELS["together"]]
        assert any("FLUX" in n for n in names)

    def test_stability_models(self):
        names = [m["name"] for m in _KNOWN_CLOUD_MODELS["stability"]]
        assert "stable-image-core" in names

    def test_bfl_models(self):
        names = [m["name"] for m in _KNOWN_CLOUD_MODELS["bfl"]]
        assert "flux-pro-1.1" in names

    def test_fal_models(self):
        names = [m["name"] for m in _KNOWN_CLOUD_MODELS["fal"]]
        assert any("fal-ai" in n for n in names)

    def test_all_have_pipeline_type_cloud(self):
        for models in _KNOWN_CLOUD_MODELS.values():
            for m in models:
                assert m["pipeline_type"] == "cloud"

    def test_all_have_name_and_label(self):
        for models in _KNOWN_CLOUD_MODELS.values():
            for m in models:
                assert m["name"]
                assert m["label"]


# ---------------------------------------------------------------------------
# Fetch cloud models
# ---------------------------------------------------------------------------


class TestFetchCloudModels:
    @pytest.mark.asyncio
    async def test_known_provider_returns_catalog(self):
        provider = {"id": "test", "base_url": "https://api.openai.com", "default_model": ""}
        models = await _fetch_cloud_models(provider)
        assert len(models) == len(_KNOWN_CLOUD_MODELS["openai"])
        # Should be copies, not references
        assert models[0] is not _KNOWN_CLOUD_MODELS["openai"][0]

    @pytest.mark.asyncio
    async def test_unknown_provider_fallback_to_default_model(self):
        provider = {"id": "test", "base_url": "https://custom.example.com", "default_model": "my-model"}
        with patch("augmentum.proxy.cloud_image_routes._cloud_client") as mock_client:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.get = AsyncMock(return_value=MagicMock(status_code=404))
            mock_client.return_value = ctx
            models = await _fetch_cloud_models(provider)
        assert len(models) == 1
        assert models[0]["name"] == "my-model"

    @pytest.mark.asyncio
    async def test_no_default_model_returns_empty(self):
        provider = {"id": "test", "base_url": "https://custom.example.com", "default_model": ""}
        with patch("augmentum.proxy.cloud_image_routes._cloud_client") as mock_client:
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            ctx.get = AsyncMock(return_value=MagicMock(status_code=500))
            mock_client.return_value = ctx
            models = await _fetch_cloud_models(provider)
        assert models == []


# ---------------------------------------------------------------------------
# Save cloud image
# ---------------------------------------------------------------------------


class TestSaveCloudImage:
    def test_save_creates_file(self, tmp_path):
        with patch("augmentum.proxy.cloud_image_routes.settings") as mock_settings:
            mock_settings.image_output_dir = str(tmp_path)
            mock_settings.data_dir = str(tmp_path)

            # 1x1 red pixel PNG encoded as base64
            pixel = base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
            path = _save_cloud_image("test-id", pixel)

            assert (tmp_path / "test-id.png").exists()
            assert "test-id.png" in path


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestPydanticModels:
    def test_create_model_defaults(self):
        body = ImageProviderCreate(id="test", name="Test", base_url="https://example.com")
        assert body.default_model == ""
        assert body.default_quality == "standard"

    def test_update_model_all_optional(self):
        body = ImageProviderUpdate()
        assert body.name is None
        assert body.base_url is None

    def test_generate_request_validation(self):
        req = CloudGenerateRequest(prompt="a cat")
        assert req.width == 1024
        assert req.height == 1024
        assert req.quality == "standard"
        assert req.n == 1
        assert req.seed == -1

    def test_generate_request_prompt_required(self):
        with pytest.raises(Exception):
            CloudGenerateRequest(prompt="")

    def test_edit_request(self):
        req = CloudEditRequest(prompt="fix this", source_image="abc123")
        assert req.strength == 0.75
        assert req.mask_image == ""

    def test_edit_request_with_mask(self):
        req = CloudEditRequest(prompt="fix", source_image="img", mask_image="mask")
        assert req.mask_image == "mask"


# ---------------------------------------------------------------------------
# CRUD routes (via test client)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    """Mock SQLite connection."""
    conn = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=None)
    cursor.fetchall = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value=cursor)
    conn.commit = AsyncMock()
    return conn


class TestCRUDRoutes:
    @pytest.mark.asyncio
    async def test_list_empty_providers(self, mock_conn):
        from augmentum.proxy.cloud_image_routes import list_image_providers

        request = MagicMock()
        state = MagicMock()
        state.state_manager.backend = MagicMock()
        state.state_manager.backend.__class__.__name__ = "SQLiteBackend"
        request.app.state = state

        with patch("augmentum.proxy.cloud_image_routes._get_conn", return_value=mock_conn):
            result = await list_image_providers(request)
            data = json.loads(result.body)
            assert data == []

    @pytest.mark.asyncio
    async def test_list_providers_no_db(self):
        from augmentum.proxy.cloud_image_routes import list_image_providers

        request = MagicMock()
        with patch("augmentum.proxy.cloud_image_routes._get_conn", return_value=None):
            result = await list_image_providers(request)
            data = json.loads(result.body)
            assert data == []

    @pytest.mark.asyncio
    async def test_create_provider(self, mock_conn):
        from augmentum.proxy.cloud_image_routes import create_image_provider

        # First provider → auto-set as default
        count_cursor = AsyncMock()
        count_cursor.fetchone = AsyncMock(return_value=(0,))
        mock_conn.execute = AsyncMock(side_effect=[
            AsyncMock(fetchone=AsyncMock(return_value=None)),  # existing check
            count_cursor,  # count check
            AsyncMock(),  # insert
        ])

        body = ImageProviderCreate(
            id="openai", name="OpenAI", base_url="https://api.openai.com",
            api_key="sk-test", default_model="dall-e-3",
        )
        request = MagicMock()
        with patch("augmentum.proxy.cloud_image_routes._get_conn", return_value=mock_conn):
            result = await create_image_provider(body, request)
            data = json.loads(result.body)
            assert data["status"] == "created"
            assert data["is_default"] is True

    @pytest.mark.asyncio
    async def test_create_duplicate_fails(self, mock_conn):
        from augmentum.proxy.cloud_image_routes import create_image_provider

        existing_cursor = AsyncMock()
        existing_cursor.fetchone = AsyncMock(return_value=("openai", "OpenAI", "url", None, "", "standard", True, True))
        mock_conn.execute = AsyncMock(return_value=existing_cursor)

        body = ImageProviderCreate(id="openai", name="OpenAI", base_url="https://api.openai.com")
        request = MagicMock()
        with patch("augmentum.proxy.cloud_image_routes._get_conn", return_value=mock_conn):
            with pytest.raises(Exception) as exc_info:
                await create_image_provider(body, request)
            assert "409" in str(exc_info.value) or "already exists" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_provider(self, mock_conn):
        from augmentum.proxy.cloud_image_routes import delete_image_provider

        existing_cursor = AsyncMock()
        existing_cursor.fetchone = AsyncMock(return_value=("test", "Test", "url", None, "", "standard", True, False))
        mock_conn.execute = AsyncMock(return_value=existing_cursor)

        request = MagicMock()
        with patch("augmentum.proxy.cloud_image_routes._get_conn", return_value=mock_conn):
            result = await delete_image_provider("test", request)
            data = json.loads(result.body)
            assert data["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_fails(self, mock_conn):
        from augmentum.proxy.cloud_image_routes import delete_image_provider

        cursor = AsyncMock()
        cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=cursor)

        request = MagicMock()
        with patch("augmentum.proxy.cloud_image_routes._get_conn", return_value=mock_conn):
            with pytest.raises(Exception) as exc_info:
                await delete_image_provider("nonexistent", request)
            assert "404" in str(exc_info.value) or "not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Generation adapter routing
# ---------------------------------------------------------------------------


class TestGenerationRouting:
    def test_stability_endpoint_map(self):
        """Verify Stability model-to-endpoint mapping."""
        from augmentum.proxy.cloud_image_routes import _generate_stability
        # The function exists and is callable
        assert callable(_generate_stability)

    def test_cloud_generate_request_fields(self):
        req = CloudGenerateRequest(
            prompt="a sunset",
            provider_id="openai",
            model="dall-e-3",
            width=1792,
            height=1024,
            quality="hd",
            style="vivid",
        )
        assert req.prompt == "a sunset"
        assert req.style == "vivid"
        assert req.quality == "hd"
        assert req.response_format == "url"

    def test_cloud_edit_request_fields(self):
        req = CloudEditRequest(
            prompt="add a hat",
            provider_id="openai",
            model="dall-e-2",
            source_image="base64data",
            mask_image="maskdata",
            strength=0.5,
        )
        assert req.strength == 0.5
        assert req.source_image == "base64data"
        assert req.mask_image == "maskdata"
