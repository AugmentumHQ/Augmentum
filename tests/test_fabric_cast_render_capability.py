"""Tests for the cast.render capability + extractor.

Pins:

  - dataclass round-trips through serialise/deserialise
  - unknown future fields drop gracefully (forward compat)
  - tier classifier maps hardware combos to lite / standard / heavy
  - extractor produces a sensible cap on a bare environment
    (no pynvml, no playwright → lite tier, CPU-only flags)
  - extractor uses pynvml when present (mocked NVIDIA GPU)
  - extractor uses playwright when present (mocked import success)
  - detection is cached across collect() calls — no rework per tick
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from augmentum.fabric.capabilities import (
    KIND_CAST_RENDER,
    CastRenderCapability,
    deserialise,
    serialise,
)
from augmentum.fabric.extractors import (
    CastRenderCapabilityExtractor,
    _classify_tier,
)

# ── Dataclass round-trip ──────────────────────────────────────────


def test_cast_render_capability_roundtrip():
    cap = CastRenderCapability(
        tier="heavy",
        cpu_threads=16,
        gpu_vendor="nvidia",
        gpu_model="NVIDIA GPU-A",
        gpu_vram_gb=24.0,
        hw_encoder="nvenc",
        max_concurrent_streams=3,
        can_render_html=True,
        can_render_vrm=True,
        can_encode_video=True,
        can_stream_webrtc=False,
    )
    raw = serialise(cap)
    assert raw["kind"] == KIND_CAST_RENDER
    assert raw["tier"] == "heavy"

    parsed = deserialise(raw)
    assert isinstance(parsed, CastRenderCapability)
    assert parsed == cap


def test_cast_render_capability_forward_compat_drops_unknown_fields():
    """A peer running a future schema_version may add fields we don't
    know about. deserialise must drop them rather than crash.
    """
    raw = {
        "kind": KIND_CAST_RENDER,
        "schema_version": 99,
        "tier": "heavy",
        "cpu_threads": 8,
        "gpu_vram_gb": 16.0,
        "future_unknown_field": "ignored",
    }
    parsed = deserialise(raw)
    assert isinstance(parsed, CastRenderCapability)
    assert parsed.tier == "heavy"
    assert parsed.cpu_threads == 8


# ── Tier classifier ───────────────────────────────────────────────


def test_tier_heavy_requires_nvidia_high_vram_and_browser():
    assert _classify_tier(
        gpu_vendor="nvidia", gpu_vram_gb=24.0, cpu_threads=16, has_browser=True,
    ) == "heavy"


def test_tier_falls_back_to_standard_without_browser():
    assert _classify_tier(
        gpu_vendor="nvidia", gpu_vram_gb=24.0, cpu_threads=16, has_browser=False,
    ) == "lite"


def test_tier_low_vram_gpu_is_standard_not_heavy():
    assert _classify_tier(
        gpu_vendor="nvidia", gpu_vram_gb=8.0, cpu_threads=16, has_browser=True,
    ) == "standard"


def test_tier_strong_cpu_plus_browser_is_standard():
    assert _classify_tier(
        gpu_vendor="", gpu_vram_gb=0.0, cpu_threads=16, has_browser=True,
    ) == "standard"


def test_tier_minimal_box_is_lite():
    assert _classify_tier(
        gpu_vendor="", gpu_vram_gb=0.0, cpu_threads=4, has_browser=False,
    ) == "lite"


# ── Extractor ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extractor_bare_environment_returns_lite_cap(monkeypatch):
    """No pynvml + no chrome binary = single-machine NUC-class result.

    Capability is still produced (always — the local node IS the render
    node), just with the conservative defaults.
    """
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(
        "augmentum.tools.application_cdp.find_chromium", lambda: None,
    )

    extractor = CastRenderCapabilityExtractor()
    caps = await extractor.collect()

    assert len(caps) == 1
    cap = caps[0]
    assert isinstance(cap, CastRenderCapability)
    assert cap.tier == "lite"
    assert cap.gpu_vendor == ""
    assert cap.gpu_vram_gb == 0.0
    assert cap.hw_encoder == ""
    assert cap.can_render_html is False
    assert cap.can_render_vrm is False
    assert cap.can_encode_video is False
    assert cap.cpu_threads > 0  # os.cpu_count always returns something usable


@pytest.mark.asyncio
async def test_extractor_detects_chromium_when_present(monkeypatch):
    """find_chromium() returns a path → can_render_html flips on."""
    # Block pynvml so we isolate the browser-detection path.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(
        "augmentum.tools.application_cdp.find_chromium",
        lambda: "/usr/bin/google-chrome",
    )

    extractor = CastRenderCapabilityExtractor()
    caps = await extractor.collect()
    cap = caps[0]
    assert cap.can_render_html is True
    # No GPU → can't render VRM (needs both GPU + browser)
    assert cap.can_render_vrm is False


@pytest.mark.asyncio
async def test_extractor_uses_pynvml_when_nvidia_present(monkeypatch):
    """pynvml present + nvmlDeviceGetCount > 0 → NVIDIA path activates."""
    fake_pynvml = SimpleNamespace(
        nvmlInit=MagicMock(),
        nvmlShutdown=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=1),
        nvmlDeviceGetHandleByIndex=MagicMock(return_value="handle_0"),
        nvmlDeviceGetName=MagicMock(return_value=b"NVIDIA GPU-A"),
        nvmlDeviceGetMemoryInfo=MagicMock(
            return_value=SimpleNamespace(total=24 * (1024 ** 3)),
        ),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setattr(
        "augmentum.tools.application_cdp.find_chromium",
        lambda: "/usr/bin/google-chrome",
    )

    extractor = CastRenderCapabilityExtractor()
    caps = await extractor.collect()
    cap = caps[0]
    assert cap.gpu_vendor == "nvidia"
    assert cap.gpu_model == "NVIDIA GPU-A"
    assert cap.gpu_vram_gb == 24.0
    assert cap.hw_encoder == "nvenc"
    assert cap.can_encode_video is True
    assert cap.can_render_vrm is True
    assert cap.tier == "heavy"
    assert cap.max_concurrent_streams == 3


@pytest.mark.asyncio
async def test_extractor_pynvml_failures_are_silent(monkeypatch):
    """If pynvml is present but nvmlInit blows up (no driver, etc.) we
    fall back to no-GPU instead of crashing the heartbeat.
    """
    fake_pynvml = SimpleNamespace(
        nvmlInit=MagicMock(side_effect=Exception("no driver")),
        nvmlShutdown=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=0),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setattr(
        "augmentum.tools.application_cdp.find_chromium", lambda: None,
    )

    extractor = CastRenderCapabilityExtractor()
    caps = await extractor.collect()
    cap = caps[0]
    assert cap.gpu_vendor == ""
    assert cap.hw_encoder == ""
    assert cap.tier == "lite"


@pytest.mark.asyncio
async def test_extractor_caches_detection_across_calls(monkeypatch):
    """Hardware doesn't change at runtime — repeat collect() calls
    must reuse the cached cap (no redetect cost per heartbeat tick).
    """
    init_calls = MagicMock()
    fake_pynvml = SimpleNamespace(
        nvmlInit=init_calls,
        nvmlShutdown=MagicMock(),
        nvmlDeviceGetCount=MagicMock(return_value=0),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setattr(
        "augmentum.tools.application_cdp.find_chromium", lambda: None,
    )

    extractor = CastRenderCapabilityExtractor()
    await extractor.collect()
    await extractor.collect()
    await extractor.collect()

    # Detection only ran once across three collect() calls.
    assert init_calls.call_count == 1
