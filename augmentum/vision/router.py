"""Capability + workload-aware vision dispatcher.

The :class:`VisionRouter` is the public entry point for vision work
across Augmentum. Callers don't pick a provider; they declare what
their workload is, and the router decides.

Two routing axes:

1. **Capability**: which providers are actually available right now?
   The primary model may be text-only. The SmolVLM sibling may be
   off. Both may be available.

2. **Workload hint**: is this for an interactive user moment (chat
   image upload) or a background pipeline (screen index, file_index
   backfill, security-cam first-pass)?

Routing matrix::

    Workload      | Primary VL available | Primary text-only
    ──────────────┼──────────────────────┼──────────────────
    interactive   | Primary              | SmolVLM
    background    | SmolVLM              | SmolVLM
    quality       | Primary if available | SmolVLM
    auto          | Primary if available | SmolVLM

The ``background`` row is the design's load-bearing decision: even
when the primary IS vision-capable, background pipelines route to
SmolVLM. This is what keeps the primary's KV cache clean during
active conversation. Other products don't draw this distinction.

A failed primary call automatically retries against SmolVLM; a
failed SmolVLM call returns empty (the caller decides whether to
skip enrichment).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.vision.provider import (
        ClassifierVisionProvider,
        PrimaryVisionProvider,
        SmolVLMProvider,
        VisionProvider,
    )

log = get_logger(__name__)


class Workload(str, Enum):
    """Workload hints passed to :class:`VisionRouter`.

    ``interactive`` — user is watching, latency matters, quality
    matters more. Examples: chat image upload, drag-drop into surface.

    ``background`` — pipeline work, no user waiting, must not perturb
    primary. Examples: file_index backfill, screen indexer, security
    cam first-pass.

    ``quality`` — caller explicitly wants the best available model
    even at higher latency cost. Examples: artifact-studio
    accessibility alt-text, comic-cover OCR for catalog.

    ``auto`` — let the router pick the default (currently behaves like
    ``interactive``). Use when the caller has no opinion.
    """

    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    QUALITY = "quality"
    AUTO = "auto"


class VisionRouter:
    """Pick a provider based on capability + workload hint.

    Construction is cheap — the providers are passed in already
    instantiated. Use :func:`build_default_router` to construct one
    wired against the FastAPI ``app.state``.
    """

    def __init__(
        self,
        primary: PrimaryVisionProvider | None,
        smolvlm: SmolVLMProvider | None,
        classifier: ClassifierVisionProvider | None = None,
    ) -> None:
        self._primary = primary
        self._smolvlm = smolvlm
        # The classifier sidecar acting as a vision provider when it's a
        # multimodal model (Gemma 4). A sibling-class "dedicated small VL"
        # like SmolVLM, but GPU-resident — so it's PREFERRED over SmolVLM
        # for background/frame work that must not perturb the primary.
        self._classifier = classifier

    @property
    def has_any_provider(self) -> bool:
        """True iff at least one provider is configured. Routes will
        return empty captions when False — caller should check before
        spending downstream work."""
        return (
            self._primary is not None
            or self._smolvlm is not None
            or self._classifier is not None
        )

    @property
    def primary_provider(self) -> PrimaryVisionProvider | None:
        """Direct accessor for the primary provider (status route)."""
        return self._primary

    @property
    def smolvlm_provider(self) -> SmolVLMProvider | None:
        """Direct accessor for the SmolVLM provider (status route)."""
        return self._smolvlm

    @property
    def classifier_provider(self) -> ClassifierVisionProvider | None:
        """Direct accessor for the classifier-vision provider (status route)."""
        return self._classifier

    def set_smolvlm(self, smolvlm: SmolVLMProvider | None) -> None:
        """Attach or detach the SmolVLM provider at runtime.

        Used by ``POST /api/vision/restart`` to make the master
        ``vision_provider_enabled`` toggle reactive — flipping it on
        creates the sibling and wires the provider in; flipping it off
        clears it so :meth:`is_available` reports honestly.
        """
        self._smolvlm = smolvlm

    async def is_available(self) -> bool:
        """True iff at least one provider would accept a caption call
        right now. Stricter than :attr:`has_any_provider` — checks
        live readiness."""
        if self._primary is not None and await self._primary.is_available():
            return True
        if self._classifier is not None and await self._classifier.is_available():
            return True
        if self._smolvlm is not None and await self._smolvlm.is_available():
            return True
        return False

    async def caption(
        self,
        image_bytes: bytes,
        *,
        prompt: str = "Describe this image in one short sentence.",
        max_tokens: int = 128,
        timeout_s: float = 30.0,
        workload: Workload = Workload.AUTO,
        frames: list[bytes] | None = None,
    ) -> str:
        """Caption an image. Returns empty string on full failure.

        Routes per the matrix in this module's docstring. On primary
        failure with workload != ``background``, automatically retries
        against SmolVLM rather than returning empty. ``frames`` are extra
        live-camera frames understood as one clip by video-capable
        providers (the classifier/Gemma); single-image providers ignore
        them.
        """
        provider = await self._select(workload)
        if provider is None:
            log.info("vision_route_no_provider", workload=workload.value)
            return ""

        text = await provider.caption(
            image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            frames=frames,
        )
        if text:
            return text

        # Automatic fallback: the picked provider returned nothing (down,
        # timed out, or — for the classifier — launched text-only). Try the
        # remaining available providers in preference order, never re-trying
        # the one that just failed, and never pulling the primary into a
        # background job (that's the keep-KV-clean invariant).
        for fb in (self._classifier, self._smolvlm, self._primary):
            if fb is None or fb is provider:
                continue
            if fb is self._primary and workload == Workload.BACKGROUND:
                continue
            if not await fb.is_available():
                continue
            log.info("vision_route_fallback", to=type(fb).__name__)
            text = await fb.caption(
                image_bytes,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                frames=frames,
            )
            if text:
                return text
        return text or ""

    async def _select(self, workload: Workload) -> VisionProvider | None:
        """Apply the routing matrix."""
        primary_ok = (
            self._primary is not None and await self._primary.is_available()
        )
        classifier_ok = (
            self._classifier is not None and await self._classifier.is_available()
        )
        smolvlm_ok = (
            self._smolvlm is not None and await self._smolvlm.is_available()
        )

        # Background prefers a dedicated small VL to keep the primary KV
        # cache clean. The classifier (GPU-resident Gemma) wins over the
        # default-CPU SmolVLM sibling when both are up — this is the path
        # a live frame loop rides.
        if workload == Workload.BACKGROUND:
            if classifier_ok:
                return self._classifier
            if smolvlm_ok:
                return self._smolvlm
            if primary_ok:
                return self._primary
            return None

        # Interactive / quality / auto: best-quality primary first when
        # capable, then the dedicated small VLs (classifier before SmolVLM).
        if primary_ok:
            return self._primary
        if classifier_ok:
            return self._classifier
        if smolvlm_ok:
            return self._smolvlm
        return None


__all__ = ["VisionRouter", "Workload"]
