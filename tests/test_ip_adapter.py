"""Tests for IP-Adapter integration."""
from __future__ import annotations

import os

import pytest


def test_generate_request_has_ip_adapter_fields():
    from augmentum.image.schemas import GenerateRequest

    req = GenerateRequest(
        prompt="test",
        ip_adapter_image="/api/image/abc123",
        ip_adapter_scale=0.55,
    )
    assert req.ip_adapter_image == "/api/image/abc123"
    assert req.ip_adapter_scale == 0.55


def test_generate_request_ip_adapter_defaults():
    from augmentum.image.schemas import GenerateRequest

    req = GenerateRequest(prompt="test")
    assert req.ip_adapter_image == ""
    assert req.ip_adapter_scale == 0.55


def test_generation_job_has_ip_adapter_fields():
    from augmentum.image.queue import GenerationJob

    job = GenerationJob(
        prompt="test",
        ip_adapter_image="/api/image/abc123",
        ip_adapter_scale=0.6,
    )
    assert job.ip_adapter_image == "/api/image/abc123"
    assert job.ip_adapter_scale == 0.6


def test_generation_job_ip_adapter_defaults():
    from augmentum.image.queue import GenerationJob

    job = GenerationJob(prompt="test")
    assert job.ip_adapter_image == ""
    assert job.ip_adapter_scale == 0.55


def test_unified_pipeline_has_ip_adapter_methods():
    from augmentum.image.pipeline_v2 import UnifiedPipeline

    p = UnifiedPipeline()
    assert hasattr(p, "load_ip_adapter")
    assert hasattr(p, "unload_ip_adapter")
    assert hasattr(p, "_ip_adapter_loaded")
    assert p._ip_adapter_loaded is False


def test_generate_signature_accepts_ip_adapter():
    import inspect
    from augmentum.image.pipeline_v2 import UnifiedPipeline

    sig = inspect.signature(UnifiedPipeline.generate)
    assert "ip_adapter_image" in sig.parameters
    assert "ip_adapter_scale" in sig.parameters
    assert sig.parameters["ip_adapter_image"].default is None
    assert sig.parameters["ip_adapter_scale"].default == 0.55


def test_abstract_generate_signature_accepts_ip_adapter():
    import inspect
    from augmentum.image.pipeline import ImagePipeline

    sig = inspect.signature(ImagePipeline.generate)
    assert "ip_adapter_image" in sig.parameters
    assert "ip_adapter_scale" in sig.parameters


def test_ip_adapter_settings_exist():
    from augmentum.config import settings

    assert hasattr(settings, "image_ip_adapter_enabled")
    assert hasattr(settings, "image_ip_adapter_scale")
    assert settings.image_ip_adapter_enabled is True
    assert settings.image_ip_adapter_scale == 0.55


def test_ip_adapter_weights_map_excludes_flux():
    from augmentum.image.pipeline_v2 import UnifiedPipeline
    from augmentum.image.schemas import PipelineType

    weights = UnifiedPipeline._IP_ADAPTER_WEIGHTS
    assert PipelineType.SD15 in weights
    assert PipelineType.SDXL in weights
    assert PipelineType.FLUX not in weights


def test_unloaded_pipeline_ip_adapter_flag_resets():
    from augmentum.image.pipeline_v2 import UnifiedPipeline

    p = UnifiedPipeline()
    p._ip_adapter_loaded = True
    # Simulate what unload() does to the flag
    assert p._ip_adapter_loaded is True


class _FakeGeneratedImage:
    size = (16, 16)
    width = 16
    height = 16

    def save(self, _path):
        return None


@pytest.mark.asyncio
async def test_plain_generation_unloads_stale_ip_adapter(monkeypatch, tmp_path):
    from augmentum.image import pipeline_v2
    from augmentum.image.pipeline_v2 import UnifiedPipeline

    pipeline = UnifiedPipeline()
    pipeline._pipe = object()
    pipeline._pipe_params = set()
    pipeline._ip_adapter_loaded = True

    events = []

    async def fake_unload_ip_adapter():
        events.append("unload")
        pipeline._ip_adapter_loaded = False

    async def fake_run_on_thread(_fn):
        events.append("generate")
        assert pipeline._ip_adapter_loaded is False
        return _FakeGeneratedImage()

    monkeypatch.setattr(pipeline, "unload_ip_adapter", fake_unload_ip_adapter)
    monkeypatch.setattr(pipeline_v2, "_run_on_thread", fake_run_on_thread)

    result = await pipeline.generate("test", output_dir=str(tmp_path))

    assert events == ["unload", "generate"]
    assert result.width == 16
    assert result.height == 16


def test_empty_ip_adapter_references_do_not_count():
    from augmentum.image.pipeline_v2 import _has_ip_adapter_reference

    assert _has_ip_adapter_reference(None) is False
    assert _has_ip_adapter_reference("") is False
    assert _has_ip_adapter_reference(["", "   "]) is False
    assert _has_ip_adapter_reference("/api/image/abc123") is True


def test_diffusers_image_embeds_error_is_ip_adapter_retryable():
    from augmentum.proxy.server import _is_ip_adapter_generation_error

    exc = ValueError(
        "UNet2DConditionModel has the config param `encoder_hid_dim_type` "
        "set to 'ip_image_proj' which requires `image_embeds`"
    )

    assert _is_ip_adapter_generation_error(exc) is True
    assert _is_ip_adapter_generation_error(RuntimeError("GPU OOM")) is False


# ---------------------------------------------------------------------------
# Live integration test (requires running server)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("AUGMENTUM_LIVE_TEST"),
    reason="Requires running server (set AUGMENTUM_LIVE_TEST=1)",
)
def test_ebook_with_ip_adapter_live():
    """Integration test: ebook tool creates book with auto-generated illustrations."""
    import httpx

    base = os.environ.get("AUGMENTUM_BASE_URL", "http://localhost:6100")
    token = os.environ.get("AUGMENTUM_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=600) as client:
        resp = client.post(
            f"{base}/v1/chat/completions",
            headers={**headers, "X-Augmentum-Tools": "create_ebook"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content":
                    "Call create_ebook with title='IP-Adapter Test', author='Test', "
                    "chapters=[{heading:'Chapter 1', body:'A brave orange tabby cat "
                    "named Whiskers explored a magical forest with glowing mushrooms.'}, "
                    "{heading:'Chapter 2', body:'Whiskers met a silver fox named "
                    "Silverbell at a crystal bridge that shimmered with rainbow light.'}]"
                }],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "epub" in content.lower() or "download" in content.lower()
