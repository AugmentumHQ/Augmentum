"""The surface-reshape ENGINE — natural-language ask → verified, reversible,
recorded change on the right surface.

This is the spine that makes "ask Augmentum to change something and it just
happens, safely" real, surface-agnostically. It's the surface-layer analogue of
the code-layer ``orchestrator.run_self_edit``: it ties together

    request (any modality) → CLASSIFY to a surface change → reshape() → RECORD

The classifier (NL → ``ReshapeChange``) and the recorder are INJECTED — the real
classifier is model-backed and sees the closed list of registered surfaces; the
real recorder writes to the never-pruned archive (``selfedit.store``). So the
engine itself is pure and testable, calls no model and no DB, and adding a surface
needs zero changes here (it just appears in the surface list handed to the
classifier).

One verb, every surface, every modality: a single ``run_reshape_request`` is what
chat / voice / the companion / the manual input all call.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from augmentum.selfedit.surfaces.base import ReshapeChange, registered_surfaces
from augmentum.selfedit.surfaces.reshape import ReshapeResult, reshape
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Engine-level outcome statuses (map to the archive at the recorder).
STATUS_PROMOTED = "promoted"            # kept + mechanically verified (auto)
STATUS_APPLIED_PENDING = "applied_pending"  # kept + needs the user's pick (see-it-keep-it)
STATUS_REVERTED = "reverted"           # verify failed → auto-undone
STATUS_FAILED = "failed"               # could not apply
STATUS_UNMAPPED = "unmapped"           # couldn't turn the ask into a change


@dataclass
class ReshapeRequest:
    ask: str                           # the user's natural-language request
    actor: str                         # user_id — scopes every write
    surface_hint: str = ""             # optional ("VR" / "settings"); classifier may ignore


# Injected: NL request + the live surface list → a concrete change (or None if unmappable).
Classifier = Callable[["ReshapeRequest", list[str]], Awaitable[ReshapeChange | None]]
# Injected archive hooks (the real ones write to selfedit.store, the never-pruned lineage).
OnStart = Callable[[str, "ReshapeRequest", ReshapeChange], Awaitable[None]]
OnFinish = Callable[[str, str, ReshapeResult, str], Awaitable[None]]  # (id, actor, result, status)


@dataclass
class EngineResult:
    attempt_id: str
    mapped: bool
    status: str
    change: ReshapeChange | None = None
    reshape: ReshapeResult | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "attempt_id": self.attempt_id, "mapped": self.mapped, "status": self.status,
            "change": self.change.to_dict() if self.change else None,
            "reshape": self.reshape.to_dict() if self.reshape else None,
            "detail": self.detail,
        }


def _attempt_id(request: ReshapeRequest) -> str:
    """Deterministic id (no RNG/clock) so a run is reproducible/resumable."""
    h = hashlib.sha1(f"{request.actor}|{request.ask}|{request.surface_hint}".encode()).hexdigest()
    return f"rsh-{h[:12]}"


def _status_for(result: ReshapeResult) -> str:
    if not result.applied:
        return STATUS_FAILED
    if not result.kept:
        return STATUS_REVERTED
    if result.auto_promotable:
        return STATUS_PROMOTED
    return STATUS_APPLIED_PENDING


async def run_reshape_request(request: ReshapeRequest, *, classify: Classifier,
                              on_start: OnStart | None = None,
                              on_finish: OnFinish | None = None,
                              attempt_id: str | None = None,
                              **reshape_kwargs) -> EngineResult:
    """Classify the ask to a surface change, reshape it, record the attempt.
    ``**reshape_kwargs`` pass through to ``reshape`` (e.g. ``extra_verifiers``)."""
    surfaces = list(registered_surfaces().keys())
    change = await classify(request, surfaces)
    aid = attempt_id or _attempt_id(request)

    if change is None:
        log.info("reshape_request_unmapped", actor=request.actor, surfaces=surfaces)
        return EngineResult(aid, mapped=False, status=STATUS_UNMAPPED,
                            detail="could not map the request to a known surface change")

    if not change.actor:                # carry the requester through if the classifier omitted it
        change.actor = request.actor

    if on_start:
        await on_start(aid, request, change)

    result = await reshape(change, **reshape_kwargs)
    status = _status_for(result)

    if on_finish:
        await on_finish(aid, change.actor, result, status)

    log.info("reshape_request_done", attempt_id=aid, surface=change.surface, status=status)
    return EngineResult(aid, mapped=True, status=status, change=change,
                        reshape=result, detail=result.detail)
