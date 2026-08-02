"""Smoke tests for the vision provider abstraction.

Validates the public surface without bringing up an actual
llama-server subprocess. SmolVLMSibling.start() with empty paths or
missing files should fail gracefully (return False, log warning) so
the captioner pipeline can run with vision_provider_enabled=False
without crashing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_module_imports():
    """The vision package's public surface must import without side
    effects. Used as the cheapest possible regression check on
    refactors of the abstraction layer."""
    from augmentum.vision import (
        PrimaryVisionProvider,
        SmolVLMProvider,
        SmolVLMSibling,
        VisionProvider,
        VisionRouter,
        Workload,
    )
    assert VisionProvider is not None
    assert SmolVLMSibling is not None
    assert SmolVLMProvider is not None
    assert PrimaryVisionProvider is not None
    assert VisionRouter is not None
    assert Workload is not None


def test_vision_provider_is_abstract():
    """Direct instantiation should fail — VisionProvider is ABC."""
    from augmentum.vision.provider import VisionProvider
    with pytest.raises(TypeError):
        VisionProvider()  # noqa: F841 — instantiation must fail


def test_smolvlm_config_dataclass():
    """SmolVLMConfig: confirm field defaults match the launch
    description (port 8092, CPU-only, captions-friendly ctx)."""
    from augmentum.vision.provider import SmolVLMConfig
    cfg = SmolVLMConfig()
    assert cfg.backend_port == 8092
    assert cfg.gpu_layers == 0  # CPU-only default per the substrate philosophy
    assert cfg.ctx_size == 8192  # captions don't need much
    assert cfg.base_model_path == ""
    assert cfg.mmproj_path == ""


@pytest.mark.asyncio
async def test_smolvlm_sibling_no_model_path():
    """Empty base_model_path → start() returns False and doesn't crash."""
    from augmentum.vision.provider import SmolVLMConfig, SmolVLMSibling
    sib = SmolVLMSibling(SmolVLMConfig(base_model_path=""))
    assert await sib.start() is False
    assert sib.base_url == ""
    assert await sib.is_ready() is False


@pytest.mark.asyncio
async def test_smolvlm_sibling_missing_model_file():
    """Configured but non-existent file → start() fails gracefully."""
    from augmentum.vision.provider import SmolVLMConfig, SmolVLMSibling
    sib = SmolVLMSibling(SmolVLMConfig(
        base_model_path="/does/not/exist.gguf",
    ))
    assert await sib.start() is False
    assert await sib.is_ready() is False


@pytest.mark.asyncio
async def test_smolvlm_sibling_stop_idempotent():
    """stop() on a never-started sibling should be a no-op, not crash."""
    from augmentum.vision.provider import SmolVLMConfig, SmolVLMSibling
    sib = SmolVLMSibling(SmolVLMConfig())
    # Should not raise:
    await sib.stop()
    await sib.stop()


@pytest.mark.asyncio
async def test_primary_vision_provider_no_app_state():
    """PrimaryVisionProvider with None app_state → not available."""
    from augmentum.vision.provider import PrimaryVisionProvider
    p = PrimaryVisionProvider(app_state=None)
    assert await p.is_available() is False


@pytest.mark.asyncio
async def test_primary_vision_provider_no_llama_manager():
    """app_state without llama_manager → not available."""
    from augmentum.vision.provider import PrimaryVisionProvider
    from types import SimpleNamespace
    p = PrimaryVisionProvider(app_state=SimpleNamespace())
    assert await p.is_available() is False


@pytest.mark.asyncio
async def test_primary_vision_provider_text_only_model():
    """Loaded primary without mmproj → not available (text-only)."""
    from augmentum.vision.provider import PrimaryVisionProvider
    from types import SimpleNamespace
    mgr = SimpleNamespace(
        state=SimpleNamespace(name="READY"),
        current_mmproj_path="",  # text-only model
    )
    app_state = SimpleNamespace(llama_manager=mgr)
    p = PrimaryVisionProvider(app_state=app_state)
    assert await p.is_available() is False


@pytest.mark.asyncio
async def test_primary_vision_provider_vl_capable():
    """Loaded primary with paired mmproj → available."""
    from augmentum.vision.provider import PrimaryVisionProvider
    from types import SimpleNamespace
    mgr = SimpleNamespace(
        state=SimpleNamespace(name="READY"),
        current_mmproj_path="/models/some/mmproj.gguf",
    )
    app_state = SimpleNamespace(llama_manager=mgr)
    p = PrimaryVisionProvider(app_state=app_state)
    assert await p.is_available() is True


@pytest.mark.asyncio
async def test_smolvlm_provider_returns_empty_when_sibling_not_ready():
    """The provider's caption() returns empty (not raises) when the
    sibling isn't up. This protects the captioner pipeline from
    crashing when vision_provider_enabled is False."""
    from augmentum.vision.provider import (
        SmolVLMConfig,
        SmolVLMProvider,
        SmolVLMSibling,
    )
    sib = SmolVLMSibling(SmolVLMConfig())  # never started
    http = MagicMock()
    prov = SmolVLMProvider(sib, http)
    text = await prov.caption(b"\x89PNG\r\n\x1a\n")
    assert text == ""


def test_ensure_stb_decodable_passes_png_through():
    """PNG bytes should pass through unchanged — no Pillow round-trip
    cost on the common case."""
    from augmentum.vision.provider import _ensure_stb_decodable
    png = b"\x89PNG\r\n\x1a\n" + b"rest-of-png-bytes"
    assert _ensure_stb_decodable(png) is png


def test_ensure_stb_decodable_passes_jpeg_through():
    from augmentum.vision.provider import _ensure_stb_decodable
    jpeg = b"\xff\xd8\xff\xe0" + b"rest-of-jpeg"
    assert _ensure_stb_decodable(jpeg) is jpeg


def test_ensure_stb_decodable_transcodes_webp_to_png():
    """WebP isn't decodable by stb_image (the library mtmd_helper
    uses). Without this transcode, llama-server returns 400. Regression
    pinned 2026-06-08 after 9 hourly caption failures on a single
    WebP in the file_index."""
    import io
    from PIL import Image
    from augmentum.vision.provider import _ensure_stb_decodable

    # Build a tiny real WebP in memory.
    src = Image.new("RGB", (8, 8), (255, 0, 0))
    buf = io.BytesIO()
    src.save(buf, format="WEBP")
    webp_bytes = buf.getvalue()
    assert webp_bytes[:4] == b"RIFF" and webp_bytes[8:12] == b"WEBP"

    out = _ensure_stb_decodable(webp_bytes)
    assert out is not None
    assert out[:8] == b"\x89PNG\r\n\x1a\n"


def test_ensure_stb_decodable_returns_none_on_garbage():
    """Unrecoverable bytes return None so the caller can log + skip
    instead of sending garbage that produces an obscure 400."""
    from augmentum.vision.provider import _ensure_stb_decodable
    assert _ensure_stb_decodable(b"this is not an image at all") is None


# ── Gemma (classifier) vision payload: params + instruct + multi-frame ──

_PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _CapturingHttp:
    """Captures the JSON payload posted to /v1/chat/completions."""

    def __init__(self):
        self.payload = None

    async def post(self, url, json=None, timeout=None):
        self.payload = json

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"choices": [{"message": {"content": "a tidy desk"}}]}

        return _Resp()


@pytest.mark.asyncio
async def test_classifier_vision_payload_uses_gemma_params_and_instruct():
    """Gemma path posts its recommended sampling + thinking OFF (instruct,
    low-latency captioning role)."""
    from augmentum.vision.provider import (
        ClassifierVisionProvider,
        _CAPTION_SAMPLING,
    )
    http = _CapturingHttp()
    prov = ClassifierVisionProvider("http://classifier:9099/v1", http)
    out = await prov.caption(_PNG_1PX, prompt="what is this?")
    assert out == "a tidy desk"
    p = http.payload
    # Low-variance caption sampling (shared with SmolVLM) — deterministic
    # captions that don't confabulate into downstream memory (bake-off 2026-06-18).
    assert p["temperature"] == _CAPTION_SAMPLING["temperature"]
    assert p["top_p"] == _CAPTION_SAMPLING["top_p"]
    assert p["top_k"] == _CAPTION_SAMPLING["top_k"]
    # Pinned seed is forwarded so identical frames yield identical captions —
    # the dedicated, isolated live-caption profile (not the routing sampler).
    assert p["seed"] == _CAPTION_SAMPLING["seed"]
    assert p["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_classifier_vision_multiframe_sends_one_sequence():
    """Extra frames ride in the SAME message (one clip, not N stills)."""
    from augmentum.vision.provider import ClassifierVisionProvider
    http = _CapturingHttp()
    prov = ClassifierVisionProvider("http://classifier:9099/v1", http)
    await prov.caption(_PNG_1PX, prompt="what's happening?", frames=[_PNG_1PX, _PNG_1PX])
    content = http.payload["messages"][0]["content"]
    image_parts = [c for c in content if c.get("type") == "image_url"]
    text_parts = [c for c in content if c.get("type") == "text"]
    assert len(image_parts) == 3   # frame 0 + 2 extra
    assert len(text_parts) == 1


@pytest.mark.asyncio
async def test_smolvlm_payload_omits_thinking_kwarg():
    """SmolVLM's template doesn't branch on thinking — we don't ship the
    kwarg there (only Gemma opts in)."""
    from augmentum.vision.provider import _caption_via_openai_endpoint
    http = _CapturingHttp()
    out = await _caption_via_openai_endpoint(
        http, "http://sibling:8092", _PNG_1PX,
        prompt="hi", max_tokens=64, timeout_s=5.0, model="smolvlm",
    )
    assert out == "a tidy desk"
    assert "chat_template_kwargs" not in http.payload
    assert http.payload["temperature"] == 0.2   # steady default


def test_caption_prompt_structured_and_grounded():
    """The unified captioner prompt is structured (SEES/MAIN), forbids
    invention, folds in the user's question, and asserts reality only on
    the live-camera path (bake-off 2026-06-18)."""
    from augmentum.models.base import _caption_prompt_for
    # Structured + anti-confabulation on every variant.
    for lc in (False, True):
        p = _caption_prompt_for("what is this plant?", multi=False, live_camera=lc)
        assert "SEES:" in p and "MAIN:" in p
        assert "do NOT invent" in p
        assert "word-for-word" in p
        assert "what is this plant?" in p   # question folded in
    # Live-camera adds the "this is REAL, not fiction" grounding; the plain
    # image path does not claim a live camera.
    live = _caption_prompt_for("what am I holding?", multi=True, live_camera=True)
    assert "REAL" in live and "live camera" in live
    plain = _caption_prompt_for("", multi=False, live_camera=False)
    assert "live camera" not in plain
    # Still structured even with no user question.
    assert "SEES:" in plain and "MAIN:" in plain
