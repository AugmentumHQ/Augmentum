"""Tests for OpenAI Images API (/v1/images/generations, /v1/images/{id})."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.image.schemas import OpenAIImageRequest
from augmentum.proxy.openai_routes import _map_openai_image_params, router

# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture
def app_and_client():
    """Create a FastAPI app with openai_routes and mocked image state."""
    app = FastAPI()
    app.include_router(router)

    state = MagicMock()
    state.image_queue = MagicMock()
    state.image_queue.submit = AsyncMock()
    state.image_queue.wait_for_result = AsyncMock()
    state.image_preset_manager = MagicMock()
    state.image_preset_manager.get.return_value = None
    state.image_hardware = None
    state.image_model_manager = None
    state.image_persistence = AsyncMock()
    app.state = state

    return app, TestClient(app), state


# ==========================================================================
# OpenAIImageRequest Pydantic Validation
# ==========================================================================


class TestOpenAIImageRequest:
    """Tests for OpenAIImageRequest Pydantic validation."""

    def test_valid_minimal_request(self):
        req = OpenAIImageRequest(prompt="a cat")
        assert req.prompt == "a cat"
        assert req.n == 1
        assert req.size == "1024x1024"
        assert req.quality == "standard"
        assert req.style == "vivid"
        assert req.response_format == "url"

    def test_valid_full_request(self):
        req = OpenAIImageRequest(
            prompt="a dog",
            model="dall-e-3",
            n=2,
            size="512x768",
            quality="hd",
            style="natural",
            response_format="b64_json",
            user="test-user",
        )
        assert req.n == 2
        assert req.size == "512x768"
        assert req.quality == "hd"
        assert req.style == "natural"

    def test_valid_sizes(self):
        for size in ("256x256", "512x512", "1024x1024", "1024x1792", "2048x2048", "768x512"):
            req = OpenAIImageRequest(prompt="test", size=size)
            assert req.size == size

    def test_invalid_size_format(self):
        with pytest.raises(ValueError, match="Invalid size format"):
            OpenAIImageRequest(prompt="test", size="big")

    def test_invalid_size_too_small(self):
        with pytest.raises(ValueError, match="between 256 and 2048"):
            OpenAIImageRequest(prompt="test", size="128x128")

    def test_invalid_size_too_large(self):
        with pytest.raises(ValueError, match="between 256 and 2048"):
            OpenAIImageRequest(prompt="test", size="4096x4096")

    def test_invalid_size_not_multiple_of_8(self):
        with pytest.raises(ValueError, match="multiples of 8"):
            OpenAIImageRequest(prompt="test", size="513x512")

    def test_invalid_quality(self):
        with pytest.raises(ValueError, match="quality must be"):
            OpenAIImageRequest(prompt="test", quality="ultra")

    def test_invalid_style(self):
        with pytest.raises(ValueError, match="style must be"):
            OpenAIImageRequest(prompt="test", style="abstract")

    def test_invalid_response_format(self):
        with pytest.raises(ValueError, match="response_format must be"):
            OpenAIImageRequest(prompt="test", response_format="png")

    def test_n_too_high(self):
        with pytest.raises(ValueError):
            OpenAIImageRequest(prompt="test", n=5)

    def test_n_too_low(self):
        with pytest.raises(ValueError):
            OpenAIImageRequest(prompt="test", n=0)

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValueError):
            OpenAIImageRequest(prompt="")

    def test_prompt_too_long(self):
        with pytest.raises(ValueError):
            OpenAIImageRequest(prompt="x" * 4001)

    def test_size_auto_accepted(self):
        req = OpenAIImageRequest(prompt="test", size="auto")
        assert req.size == "auto"

    def test_gpt_image_quality_values(self):
        for q in ("low", "medium", "high", "auto"):
            req = OpenAIImageRequest(prompt="test", quality=q)
            assert req.quality == q

    def test_sd_extension_fields(self):
        req = OpenAIImageRequest(
            prompt="test",
            negative_prompt="ugly, blurry",
            steps=35,
            cfg_scale=9.5,
            seed=12345,
        )
        assert req.negative_prompt == "ugly, blurry"
        assert req.steps == 35
        assert req.cfg_scale == 9.5
        assert req.seed == 12345

    def test_extra_fields_accepted(self):
        """Unknown fields (gpt-image-1 params, IMAGES_OPENAI_API_PARAMS) must not 422."""
        req = OpenAIImageRequest(
            prompt="test",
            background="transparent",
            output_format="webp",
            moderation="low",
        )
        assert req.prompt == "test"

    def test_gpt_image_1_5_model_accepted(self):
        req = OpenAIImageRequest(prompt="test", model="gpt-image-1.5")
        assert req.model == "gpt-image-1.5"


# ==========================================================================
# _map_openai_image_params
# ==========================================================================


class TestParamMapping:
    """Tests for _map_openai_image_params helper."""

    def test_size_parsing(self):
        req = OpenAIImageRequest(prompt="test", size="768x512")
        params = _map_openai_image_params(req)
        assert params["width"] == 768
        assert params["height"] == 512

    def test_size_auto_uses_defaults(self):
        req = OpenAIImageRequest(prompt="test", size="auto")
        params = _map_openai_image_params(req)
        # Should use server defaults (512x512 from settings)
        assert isinstance(params["width"], int)
        assert isinstance(params["height"], int)
        assert params["width"] > 0
        assert params["height"] > 0

    def test_quality_standard(self):
        req = OpenAIImageRequest(prompt="test", quality="standard")
        params = _map_openai_image_params(req)
        assert params["steps"] == 20
        assert params["cfg_scale"] == 7.0

    def test_quality_hd(self):
        req = OpenAIImageRequest(prompt="test", quality="hd")
        params = _map_openai_image_params(req)
        assert params["steps"] == 30
        assert params["cfg_scale"] == 8.0

    def test_quality_high_same_as_hd(self):
        req = OpenAIImageRequest(prompt="test", quality="high")
        params = _map_openai_image_params(req)
        assert params["steps"] == 30
        assert params["cfg_scale"] == 8.0

    def test_quality_medium(self):
        req = OpenAIImageRequest(prompt="test", quality="medium")
        params = _map_openai_image_params(req)
        assert params["steps"] == 25
        assert params["cfg_scale"] == 7.5

    def test_quality_low(self):
        req = OpenAIImageRequest(prompt="test", quality="low")
        params = _map_openai_image_params(req)
        assert params["steps"] == 15
        assert params["cfg_scale"] == 6.0

    def test_quality_auto_uses_standard_defaults(self):
        req = OpenAIImageRequest(prompt="test", quality="auto")
        params = _map_openai_image_params(req)
        assert params["steps"] == 20
        assert params["cfg_scale"] == 7.0

    def test_explicit_steps_override_quality(self):
        req = OpenAIImageRequest(prompt="test", quality="hd", steps=50)
        params = _map_openai_image_params(req)
        assert params["steps"] == 50
        assert params["cfg_scale"] == 8.0  # cfg_scale still from quality

    def test_explicit_cfg_override_quality(self):
        req = OpenAIImageRequest(prompt="test", quality="hd", cfg_scale=5.0)
        params = _map_openai_image_params(req)
        assert params["steps"] == 30  # steps still from quality
        assert params["cfg_scale"] == 5.0

    def test_style_vivid_no_negative(self):
        req = OpenAIImageRequest(prompt="test", style="vivid")
        params = _map_openai_image_params(req)
        assert params["negative_prompt"] == ""

    def test_style_natural_adds_negative(self):
        req = OpenAIImageRequest(prompt="test", style="natural")
        params = _map_openai_image_params(req)
        assert "oversaturated" in params["negative_prompt"]

    def test_style_natural_appends_to_explicit_negative(self):
        req = OpenAIImageRequest(prompt="test", style="natural", negative_prompt="ugly, blurry")
        params = _map_openai_image_params(req)
        assert "ugly, blurry" in params["negative_prompt"]
        assert "oversaturated" in params["negative_prompt"]

    def test_explicit_negative_prompt_passthrough(self):
        req = OpenAIImageRequest(prompt="test", negative_prompt="deformed hands")
        params = _map_openai_image_params(req)
        assert params["negative_prompt"] == "deformed hands"

    def test_dalle_model_maps_to_empty(self):
        for model_name in ("dall-e-2", "dall-e-3", "gpt-image-1", "gpt-image-1.5"):
            req = OpenAIImageRequest(prompt="test", model=model_name)
            params = _map_openai_image_params(req)
            assert params["model"] == ""

    def test_custom_model_passthrough(self):
        req = OpenAIImageRequest(prompt="test", model="my-custom-sd")
        params = _map_openai_image_params(req)
        assert params["model"] == "my-custom-sd"

    def test_empty_model_stays_empty(self):
        req = OpenAIImageRequest(prompt="test")
        params = _map_openai_image_params(req)
        assert params["model"] == ""

    def test_seed_passthrough(self):
        req = OpenAIImageRequest(prompt="test", seed=42)
        params = _map_openai_image_params(req)
        assert params["seed"] == 42

    def test_seed_default_is_minus_1(self):
        req = OpenAIImageRequest(prompt="test")
        params = _map_openai_image_params(req)
        assert params["seed"] == -1


# ==========================================================================
# POST /v1/images/generations
# ==========================================================================


class TestOpenAIImageGenerate:
    """Tests for the /v1/images/generations endpoint."""

    def _make_result(self, image_id="img-abc123", file_path=None):
        if file_path is None:
            file_path = "/tmp/fake.png"
        return {
            "image_id": image_id,
            "file_path": file_path,
            "seed": 42,
            "width": 1024,
            "height": 1024,
        }

    def test_basic_url_mode(self, app_and_client):
        app, client, state = app_and_client

        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value=self._make_result())

        resp = client.post("/v1/images/generations", json={"prompt": "a cat"})
        assert resp.status_code == 200
        data = resp.json()
        assert "created" in data
        assert isinstance(data["created"], int)
        assert len(data["data"]) == 1
        assert data["data"][0]["url"] is not None
        assert "img-abc123" in data["data"][0]["url"]
        assert data["data"][0]["revised_prompt"] == "a cat"
        assert data["data"][0]["b64_json"] is None

    def test_basic_b64_mode(self, app_and_client):
        app, client, state = app_and_client

        # Create a real temp file for b64 reading
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fakepngdata")
            tmp_path = f.name

        try:
            job = MagicMock()
            state.image_queue.submit = AsyncMock(return_value=job)
            state.image_queue.wait_for_result = AsyncMock(
                return_value=self._make_result(file_path=tmp_path),
            )

            resp = client.post(
                "/v1/images/generations",
                json={"prompt": "a dog", "response_format": "b64_json"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["b64_json"] is not None
            assert data["data"][0]["url"] is None

            # Verify b64 decodes to original content
            import base64
            decoded = base64.b64decode(data["data"][0]["b64_json"])
            assert decoded == b"fakepngdata"
        finally:
            os.unlink(tmp_path)

    def test_n_greater_than_1(self, app_and_client):
        app, client, state = app_and_client

        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        results = [self._make_result(f"img-{i}") for i in range(3)]
        state.image_queue.wait_for_result = AsyncMock(side_effect=results)

        resp = client.post("/v1/images/generations", json={"prompt": "cats", "n": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        # Each should have a unique URL
        urls = [d["url"] for d in data["data"]]
        assert len(set(urls)) == 3

    def test_subsystem_disabled_returns_503(self, app_and_client):
        app, client, state = app_and_client
        state.image_queue = None

        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        assert resp.status_code == 503
        err = resp.json()
        assert "error" in err
        assert err["error"]["type"] == "server_error"
        assert "not enabled" in err["error"]["message"]

    def test_queue_full_returns_429(self, app_and_client):
        app, client, state = app_and_client
        state.image_queue.submit = AsyncMock(side_effect=RuntimeError("queue full"))

        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        assert resp.status_code == 429
        err = resp.json()
        assert err["error"]["type"] == "rate_limit_error"

    def test_timeout_returns_504(self, app_and_client):
        app, client, state = app_and_client
        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(side_effect=TimeoutError())

        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        assert resp.status_code == 504
        err = resp.json()
        assert err["error"]["type"] == "server_error"
        assert "timed out" in err["error"]["message"]

    def test_failure_returns_500(self, app_and_client):
        app, client, state = app_and_client
        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(
            side_effect=Exception("GPU exploded"),
        )

        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        assert resp.status_code == 500
        err = resp.json()
        assert err["error"]["type"] == "server_error"
        assert "GPU exploded" in err["error"]["message"]

    def test_openai_error_format_shape(self, app_and_client):
        """All errors must have {error: {message, type, param, code}} shape."""
        app, client, state = app_and_client
        state.image_queue = None

        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        err = resp.json()["error"]
        assert "message" in err
        assert "type" in err
        assert "param" in err
        assert "code" in err

    def test_url_construction_includes_base(self, app_and_client):
        app, client, state = app_and_client

        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value=self._make_result())

        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        url = resp.json()["data"][0]["url"]
        assert url.startswith("http")
        assert "/v1/images/img-abc123" in url

    def test_created_timestamp_is_recent(self, app_and_client):
        import time

        app, client, state = app_and_client
        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value=self._make_result())

        before = int(time.time())
        resp = client.post("/v1/images/generations", json={"prompt": "test"})
        after = int(time.time())

        created = resp.json()["created"]
        assert before <= created <= after + 1

    def test_extra_fields_dont_cause_422(self, app_and_client):
        """Open WebUI's IMAGES_OPENAI_API_PARAMS can inject arbitrary fields."""
        app, client, state = app_and_client
        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value=self._make_result())

        resp = client.post("/v1/images/generations", json={
            "prompt": "test",
            "background": "transparent",
            "output_format": "webp",
            "moderation": "low",
        })
        assert resp.status_code == 200

    def test_sd_params_passed_to_job(self, app_and_client):
        """Steps, cfg_scale, negative_prompt, seed should reach the GenerationJob."""
        app, client, state = app_and_client
        job = MagicMock()
        state.image_queue.submit = AsyncMock(return_value=job)
        state.image_queue.wait_for_result = AsyncMock(return_value=self._make_result())

        resp = client.post("/v1/images/generations", json={
            "prompt": "a landscape",
            "steps": 40,
            "cfg_scale": 9.0,
            "negative_prompt": "ugly",
            "seed": 999,
        })
        assert resp.status_code == 200
        # Verify the submit call used our params
        submitted_job = state.image_queue.submit.call_args[0][0]
        assert submitted_job.steps == 40
        assert submitted_job.cfg_scale == 9.0
        assert "ugly" in submitted_job.negative_prompt
        assert submitted_job.seed == 999


# ==========================================================================
# GET /v1/images/{image_id}
# ==========================================================================


class TestOpenAIImageServe:
    """Tests for the GET /v1/images/{image_id} endpoint."""

    def test_serve_existing_image(self, app_and_client):
        app, client, state = app_and_client

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG fake image data")
            tmp_path = f.name

        try:
            state.image_persistence.get_generation = AsyncMock(
                return_value={"file_path": tmp_path},
            )

            resp = client.get("/v1/images/test-image-id")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "image/png"
            assert resp.content == b"\x89PNG fake image data"
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_image_returns_404(self, app_and_client):
        app, client, state = app_and_client
        state.image_persistence.get_generation = AsyncMock(return_value=None)

        resp = client.get("/v1/images/nonexistent-id")
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert err["type"] == "not_found"

    def test_missing_file_returns_404(self, app_and_client):
        app, client, state = app_and_client
        state.image_persistence.get_generation = AsyncMock(
            return_value={"file_path": "/nonexistent/path/image.png"},
        )

        resp = client.get("/v1/images/some-id")
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert "not found on disk" in err["message"]

    def test_persistence_unavailable_returns_503(self, app_and_client):
        app, client, state = app_and_client
        state.image_persistence = None

        resp = client.get("/v1/images/some-id")
        assert resp.status_code == 503
        err = resp.json()["error"]
        assert err["type"] == "server_error"
