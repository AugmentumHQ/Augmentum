"""Surface-agnostic reshape contract — the muscle that works on every hat.

Augmentum already DELIVERS experiences across many surfaces (VR world, mobile
screen, desktop UI, media stream, knowledge view, document, voice turn). The
reshape pattern is identical across all of them:

    intent (asked, any modality) → candidate change → VERIFY → APPLY reversibly
        → (keep / await human pick / revert)  → archive the lesson

What changes per surface is ONLY the actuator. So a surface is just four muscles
behind a uniform contract — ``apply`` / ``revert`` / a per-surface ``verifier``
(the oracle that says "did it work *here*") / ``capture`` (the snapshot that feeds
the multi-model visual pick). This is the action-registry "one verb, every
surface" insight lifted one level: **one reshape, every surface.**

Adapters are dataclasses-of-callables (mirroring ``verifier.Verifier``), so every
external effect is injected and the layer is pure/testable. The existing code path
(candidate→verify_change→promote→rollback) is conceptually the adapter for the
*code* surface; this module is the general contract a non-code surface (config,
media, VR…) implements to gain the same powers. Grow muscles one adapter at a
time — the core never changes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.selfedit.verifier import Verifier
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Change classes (cross-surface). ATOMIC = settings-feel (instant, two-way door);
# BUILD = project-feel (multi-step, coarser revert). Mirrors the grows-with-user split.
CLASS_ADAPTATION = "adaptation"   # atomic, per-user config/state — the cheapest oracle
CLASS_BUILD = "build"             # heavier, produced by the edit/build loop


@dataclass
class ReshapeChange:
    """A concrete proposed change on a surface. ``payload`` is opaque — only the
    target surface's adapter understands it."""

    surface: str
    change_class: str
    payload: dict = field(default_factory=dict)
    intent: str = ""                 # what the user asked (for the oracle + the record)
    actor: str = ""                  # user_id — adapters MUST scope writes by this

    def to_dict(self) -> dict:
        return {"surface": self.surface, "change_class": self.change_class,
                "intent": self.intent[:500], "actor": self.actor,
                "payload_keys": sorted(self.payload.keys())}


@dataclass
class ReshapeOutcome:
    """Result of an adapter's ``apply`` — applied + an opaque token to undo it."""

    applied: bool
    revert_token: str = ""
    detail: str = ""


@dataclass
class CaptureArtifact:
    """A snapshot of the change for the visual-pick / the archive."""

    kind: str                        # screenshot | state | diff | text
    ref: str = ""                    # path / handle
    summary: str = ""


# The four muscles (all injected per adapter).
ApplyFn = Callable[[ReshapeChange], Awaitable[ReshapeOutcome]]
RevertFn = Callable[[str], Awaitable[bool]]
VerifierFactory = Callable[[ReshapeChange], Verifier]   # the per-surface oracle for THIS change
CaptureFn = Callable[[ReshapeChange], Awaitable[CaptureArtifact]]


@dataclass
class SurfaceAdapter:
    name: str
    change_classes: tuple[str, ...]
    apply: ApplyFn
    revert: RevertFn
    make_verifier: VerifierFactory
    capture: CaptureFn
    note: str = ""                   # human note on this surface's oracle/maturity

    def handles(self, change_class: str) -> bool:
        return change_class in self.change_classes


# --- registry (mirrors verifier.py / health.py) ---------------------------

_SURFACES: dict[str, SurfaceAdapter] = {}


def register_surface(adapter: SurfaceAdapter) -> None:
    _SURFACES[adapter.name] = adapter
    log.info("reshape_surface_registered", surface=adapter.name,
             classes=list(adapter.change_classes))


def get_surface(name: str) -> SurfaceAdapter | None:
    return _SURFACES.get(name)


def registered_surfaces() -> dict[str, SurfaceAdapter]:
    return dict(_SURFACES)


def clear_surfaces() -> None:  # for tests
    _SURFACES.clear()
