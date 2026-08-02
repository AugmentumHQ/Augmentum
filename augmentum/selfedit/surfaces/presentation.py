"""Unified interaction contract — what the user SEES/HEARS after a reshape,
identical across every surface.

Seamlessness needs ONE presentation, rendered per surface, not bespoke UX in chat
vs voice vs the Workshop. Given an ``EngineResult``, ``present()`` returns the
headline, an honest "what happened" detail (never a black box), a short spoken
form, and the right ACTIONS — so:

  - reversible auto-applied change (config/Adaptation, VERIFIED) → "Done: X" + an
    **Undo** (post-hoc, no blocking dialog — optimistic + reversible);
  - applied-but-unconfirmed (taste, needs_human) → "X — keep it?" + Keep/Undo
    (the see-it/keep-it loop);
  - heavier proposal awaiting approval → Approve / Not now / Never (reuses the
    chat offer-chip vocabulary);
  - reverted/failed → honest "tried X, didn't hold" + Retry;
  - unmapped → honest "couldn't turn that into a change" + Open Workshop.

Pure + testable. The chat renderer shows headline+detail+chips; voice speaks
``speech`` and listens for keep/undo; the Workshop shows a row with the same
actions. One contract → consistent, low-friction interaction everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.selfedit.surfaces.engine import (
    STATUS_APPLIED_PENDING,
    STATUS_FAILED,
    STATUS_PROMOTED,
    STATUS_REVERTED,
    EngineResult,
)

# Presentation status (what the UI styles on).
PRES_APPLIED = "applied"          # done + reversible (Undo)
PRES_CONFIRM = "needs_confirm"    # applied, awaiting the user's pick (Keep/Undo)
PRES_PROPOSED = "proposed"        # heavier change awaiting approval (Approve/Not now/Never)
PRES_REVERTED = "reverted"
PRES_FAILED = "failed"
PRES_UNMAPPED = "unmapped"

# Action ids (stable; the offer-chip / voice / workshop map these to controls).
ACT_UNDO = "undo"
ACT_KEEP = "keep"
ACT_APPROVE = "approve"
ACT_DISMISS = "dismiss"          # "not now"
ACT_NEVER = "never"
ACT_RETRY = "retry"
ACT_OPEN_WORKSHOP = "open_workshop"


@dataclass
class ReshapeAction:
    id: str
    label: str
    attempt_id: str = ""
    revert_token: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "attempt_id": self.attempt_id,
                "revert_token": self.revert_token}


@dataclass
class ReshapePresentation:
    status: str
    headline: str
    detail: str = ""
    speech: str = ""
    actions: list[ReshapeAction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"status": self.status, "headline": self.headline, "detail": self.detail,
                "speech": self.speech, "actions": [a.to_dict() for a in self.actions]}


def _what(result: EngineResult) -> str:
    """The human 'what' — the classifier's surfaced interpretation, else the ask."""
    if result.change and result.change.intent:
        return result.change.intent
    return "that change"


def _detail(result: EngineResult) -> str:
    rs = result.reshape
    if rs and rs.capture and rs.capture.summary:
        return rs.capture.summary
    return result.detail or ""


def present(result: EngineResult) -> ReshapePresentation:
    """Turn an engine result into the one presentation every surface renders."""
    what = _what(result)
    aid = result.attempt_id
    token = result.reshape.revert_token if result.reshape else ""
    detail = _detail(result)

    if result.status == STATUS_PROMOTED:
        return ReshapePresentation(
            PRES_APPLIED, headline=f"Done — {what}.", detail=detail,
            speech=f"Done. {what}. Say undo to revert.",
            actions=[ReshapeAction(ACT_UNDO, "Undo", aid, token)])

    if result.status == STATUS_APPLIED_PENDING:
        return ReshapePresentation(
            PRES_CONFIRM, headline=f"{what} — keep it?", detail=detail,
            speech=f"{what}. Keep it, or say undo.",
            actions=[ReshapeAction(ACT_KEEP, "Keep", aid, token),
                     ReshapeAction(ACT_UNDO, "Undo", aid, token)])

    if result.status == STATUS_REVERTED:
        return ReshapePresentation(
            PRES_REVERTED, headline=f"Couldn't apply {what} — reverted.", detail=detail,
            speech=f"I tried {what}, but it didn't hold, so I reverted it.",
            actions=[ReshapeAction(ACT_RETRY, "Try again", aid)])

    if result.status == STATUS_FAILED:
        return ReshapePresentation(
            PRES_FAILED, headline="That change didn't go through.", detail=detail,
            speech="That change didn't go through.",
            actions=[ReshapeAction(ACT_RETRY, "Try again", aid),
                     ReshapeAction(ACT_OPEN_WORKSHOP, "Open Workshop")])

    # unmapped (or anything unexpected) — honest, never a silent no-op.
    return ReshapePresentation(
        PRES_UNMAPPED, headline="I couldn't turn that into a change I can make.",
        detail=detail,
        speech="I'm not sure how to change that — want to open the Workshop?",
        actions=[ReshapeAction(ACT_OPEN_WORKSHOP, "Open Workshop")])


def proposed_presentation(result: EngineResult) -> ReshapePresentation:
    """For the strict approval/test posture: a change held for explicit approval
    BEFORE it applies (Approve / Not now / Never) — the offer-chip vocabulary. Use
    when a surface chooses to preview-then-apply rather than apply-then-undo."""
    what = _what(result)
    aid = result.attempt_id
    return ReshapePresentation(
        PRES_PROPOSED, headline=f"Want me to {what}?", detail=_detail(result),
        speech=f"Want me to {what}? Say yes, not now, or never.",
        actions=[ReshapeAction(ACT_APPROVE, "Approve", aid),
                 ReshapeAction(ACT_DISMISS, "Not now", aid),
                 ReshapeAction(ACT_NEVER, "Never", aid)])
