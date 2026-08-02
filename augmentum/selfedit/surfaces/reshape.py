"""The surface-agnostic reshape orchestration.

One function works for every surface: route the change to its adapter, apply it
reversibly, verify it through the adapter's per-surface oracle (reusing the
existing honest-tier ``verifier.verify``), then KEEP / await human pick / REVERT
per the corrected promotion predicate — **oracle-tier, not surface**:

  - required failure          → auto-revert (a broken change never lingers)
  - VERIFIED (mechanical)     → keep, auto-promotable
  - PROBABLE / HUMAN_REQUIRED → keep applied but flag needs_human (the "see it,
                                keep it" model — the user's pick is the verdict)

Nothing surface-specific lives here; that's all in the adapter. Adding a surface
(media, VR, …) needs zero changes to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.selfedit.surfaces.base import (
    CaptureArtifact,
    ReshapeChange,
    SurfaceAdapter,
    get_surface,
)
from augmentum.selfedit.verifier import (
    TIER_FAILED,
    TIER_HUMAN_REQUIRED,
    TIER_PROBABLE,
    Verdict,
    Verifier,
    verify,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReshapeResult:
    surface: str
    applied: bool
    kept: bool                       # still applied when reshape() returned
    auto_promotable: bool
    needs_human: bool                # applied + intent unconfirmed → awaiting the user's pick
    revert_token: str = ""
    verdict: Verdict | None = None
    capture: CaptureArtifact | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "surface": self.surface, "applied": self.applied, "kept": self.kept,
            "auto_promotable": self.auto_promotable, "needs_human": self.needs_human,
            "revert_token": self.revert_token,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "capture": vars(self.capture) if self.capture else None,
            "detail": self.detail,
        }


async def reshape(change: ReshapeChange, *,
                  adapter: SurfaceAdapter | None = None,
                  extra_verifiers: dict[str, Verifier] | None = None,
                  judgment_confidence_floor: float = 0.7) -> ReshapeResult:
    """Apply ``change`` to its surface, verify, and keep/hold/revert by oracle-tier."""
    adapter = adapter or get_surface(change.surface)
    if adapter is None:
        return ReshapeResult(change.surface, applied=False, kept=False,
                             auto_promotable=False, needs_human=False,
                             detail=f"no adapter registered for surface '{change.surface}'")
    if not adapter.handles(change.change_class):
        return ReshapeResult(change.surface, applied=False, kept=False,
                             auto_promotable=False, needs_human=False,
                             detail=f"surface '{adapter.name}' does not handle "
                                    f"change_class '{change.change_class}'")

    # Apply to a reversible state.
    outcome = await adapter.apply(change)
    if not outcome.applied:
        return ReshapeResult(change.surface, applied=False, kept=False,
                             auto_promotable=False, needs_human=False,
                             detail=f"apply failed: {outcome.detail}")

    capture = await adapter.capture(change)

    # Verify through this surface's oracle (+ any caller extras), honest-tier router.
    pool: dict[str, Verifier] = {}
    surf_verifier = adapter.make_verifier(change)
    pool[surf_verifier.name] = surf_verifier
    if extra_verifiers:
        pool.update(extra_verifiers)
    verdict = await verify({"change": change}, intent_class=change.change_class,
                           verifiers=pool, judgment_confidence_floor=judgment_confidence_floor)

    # Decide keep / hold / revert by oracle-tier (NOT by surface).
    if verdict.tier == TIER_FAILED:
        reverted = await adapter.revert(outcome.revert_token)
        log.info("reshape_reverted_on_failure", surface=adapter.name,
                 reverted=reverted, summary=verdict.summary)
        return ReshapeResult(adapter.name, applied=True, kept=False, auto_promotable=False,
                             needs_human=False, revert_token=outcome.revert_token,
                             verdict=verdict, capture=capture,
                             detail=f"reverted (verify failed); revert_ok={reverted}")

    needs_human = verdict.tier in (TIER_PROBABLE, TIER_HUMAN_REQUIRED)
    result = ReshapeResult(
        adapter.name, applied=True, kept=True, auto_promotable=verdict.auto_promotable,
        needs_human=needs_human, revert_token=outcome.revert_token, verdict=verdict,
        capture=capture,
        detail="kept (mechanically verified)" if verdict.auto_promotable
        else "applied; awaiting human pick" if needs_human else "kept")
    log.info("reshape_done", **{k: result.to_dict()[k] for k in
                                ("surface", "kept", "auto_promotable", "needs_human")})
    return result
