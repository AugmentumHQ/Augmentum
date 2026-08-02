"""Vision router routing-matrix tests.

The router's workload-hint logic is load-bearing — it's what keeps
background pipelines (file_index captioner, future screen indexer)
from stealing KV slots from the user's active conversation. These
tests pin that contract: when the workload is BACKGROUND, route to
SmolVLM even if the primary is VL-capable.
"""

from __future__ import annotations

import pytest


class _FakeProvider:
    """Async-safe mock that records calls and returns canned captions."""

    def __init__(self, available: bool = True, caption_text: str = "fake caption"):
        self._available = available
        self._caption_text = caption_text
        self.caption_calls = 0
        self.last_frames = None

    async def is_available(self) -> bool:
        return self._available

    async def caption(
        self,
        image_bytes,
        *,
        prompt: str = "",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        frames=None,
    ) -> str:
        self.caption_calls += 1
        self.last_frames = frames
        return self._caption_text


# ── Capability detection ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_has_any_provider_with_none():
    from augmentum.vision.router import VisionRouter
    router = VisionRouter(primary=None, smolvlm=None)
    assert router.has_any_provider is False


@pytest.mark.asyncio
async def test_has_any_provider_primary_only():
    from augmentum.vision.router import VisionRouter
    router = VisionRouter(primary=_FakeProvider(), smolvlm=None)
    assert router.has_any_provider is True


@pytest.mark.asyncio
async def test_is_available_no_providers_ready():
    from augmentum.vision.router import VisionRouter
    router = VisionRouter(
        primary=_FakeProvider(available=False),
        smolvlm=_FakeProvider(available=False),
    )
    assert await router.is_available() is False


@pytest.mark.asyncio
async def test_is_available_smolvlm_only():
    from augmentum.vision.router import VisionRouter
    router = VisionRouter(
        primary=_FakeProvider(available=False),
        smolvlm=_FakeProvider(available=True),
    )
    assert await router.is_available() is True


# ── Routing matrix — Workload.INTERACTIVE ────────────────────────────


@pytest.mark.asyncio
async def test_interactive_prefers_primary():
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.INTERACTIVE)
    assert text == "primary"
    assert primary.caption_calls == 1
    assert smolvlm.caption_calls == 0


@pytest.mark.asyncio
async def test_interactive_falls_back_when_primary_unavailable():
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(available=False, caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.INTERACTIVE)
    assert text == "smolvlm"


# ── Routing matrix — Workload.BACKGROUND (load-bearing!) ─────────────


@pytest.mark.asyncio
async def test_background_routes_to_smolvlm_even_when_primary_available():
    """The load-bearing decision. Background workloads MUST route to
    SmolVLM even if the primary is VL-capable, to keep the primary's
    KV cache clean during active user conversation."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.BACKGROUND)
    assert text == "smolvlm"
    assert primary.caption_calls == 0
    assert smolvlm.caption_calls == 1


@pytest.mark.asyncio
async def test_background_falls_back_to_primary_when_smolvlm_unavailable():
    """If SmolVLM is down, background still works — falls back to
    primary rather than returning empty."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(available=False, caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.BACKGROUND)
    assert text == "primary"


# ── Routing matrix — Workload.QUALITY / AUTO ─────────────────────────


@pytest.mark.asyncio
async def test_quality_prefers_primary():
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.QUALITY)
    assert text == "primary"


@pytest.mark.asyncio
async def test_auto_prefers_primary():
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.AUTO)
    assert text == "primary"


# ── Fallback on primary failure ──────────────────────────────────────


@pytest.mark.asyncio
async def test_primary_empty_falls_back_to_smolvlm():
    """If primary returns empty (failure mode), router retries against
    SmolVLM rather than returning empty to the caller."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="")  # simulated failure
    smolvlm = _FakeProvider(caption_text="smolvlm")
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.INTERACTIVE)
    assert text == "smolvlm"
    assert primary.caption_calls == 1
    assert smolvlm.caption_calls == 1


@pytest.mark.asyncio
async def test_background_failure_no_fallback_to_primary():
    """Background SmolVLM failure does NOT fall back to primary — that
    would defeat the KV-cache protection."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="")  # simulated failure
    router = VisionRouter(primary=primary, smolvlm=smolvlm)
    text = await router.caption(b"", workload=Workload.BACKGROUND)
    assert text == ""
    assert primary.caption_calls == 0  # MUST NOT touch primary


# ── No-provider edge cases ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_caption_with_no_providers_returns_empty():
    from augmentum.vision.router import VisionRouter, Workload
    router = VisionRouter(primary=None, smolvlm=None)
    assert await router.caption(b"") == ""
    assert await router.caption(b"", workload=Workload.BACKGROUND) == ""


# ── Classifier vision provider (Gemma sidecar reused for vision) ──────


@pytest.mark.asyncio
async def test_has_any_provider_classifier_only():
    from augmentum.vision.router import VisionRouter
    router = VisionRouter(primary=None, smolvlm=None, classifier=_FakeProvider())
    assert router.has_any_provider is True


@pytest.mark.asyncio
async def test_is_available_classifier_only():
    from augmentum.vision.router import VisionRouter
    router = VisionRouter(
        primary=_FakeProvider(available=False), smolvlm=None,
        classifier=_FakeProvider(available=True),
    )
    assert await router.is_available() is True


@pytest.mark.asyncio
async def test_background_prefers_classifier_over_smolvlm():
    """The classifier (GPU-resident Gemma) is the preferred dedicated VL
    for background/frame work — ahead of the default-CPU SmolVLM sibling,
    and never touching the primary. This is the path a live frame loop rides."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    classifier = _FakeProvider(caption_text="classifier")
    router = VisionRouter(primary=primary, smolvlm=smolvlm, classifier=classifier)
    text = await router.caption(b"", workload=Workload.BACKGROUND)
    assert text == "classifier"
    assert classifier.caption_calls == 1
    assert smolvlm.caption_calls == 0
    assert primary.caption_calls == 0


@pytest.mark.asyncio
async def test_background_classifier_failure_falls_back_to_smolvlm_not_primary():
    """A text-only / down classifier returns empty → fall back to SmolVLM,
    but still never perturb the primary on a background job."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    classifier = _FakeProvider(caption_text="")  # serving text-only / down
    router = VisionRouter(primary=primary, smolvlm=smolvlm, classifier=classifier)
    text = await router.caption(b"", workload=Workload.BACKGROUND)
    assert text == "smolvlm"
    assert primary.caption_calls == 0


@pytest.mark.asyncio
async def test_interactive_classifier_beats_smolvlm_when_primary_text_only():
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(available=False, caption_text="primary")
    smolvlm = _FakeProvider(caption_text="smolvlm")
    classifier = _FakeProvider(caption_text="classifier")
    router = VisionRouter(primary=primary, smolvlm=smolvlm, classifier=classifier)
    text = await router.caption(b"", workload=Workload.INTERACTIVE)
    assert text == "classifier"
    assert smolvlm.caption_calls == 0


@pytest.mark.asyncio
async def test_interactive_vl_primary_still_beats_classifier():
    """A capable primary VL is still preferred for interactive/quality —
    the classifier is the sibling-class fallback, not a primary replacement."""
    from augmentum.vision.router import VisionRouter, Workload
    primary = _FakeProvider(caption_text="primary")
    classifier = _FakeProvider(caption_text="classifier")
    router = VisionRouter(primary=primary, smolvlm=None, classifier=classifier)
    text = await router.caption(b"", workload=Workload.INTERACTIVE)
    assert text == "primary"
    assert classifier.caption_calls == 0


@pytest.mark.asyncio
async def test_classifier_provider_availability_gates_on_base_url():
    from augmentum.vision.provider import ClassifierVisionProvider
    assert await ClassifierVisionProvider("http://classifier:8091/v1", None).is_available() is True
    assert await ClassifierVisionProvider("", None).is_available() is False
