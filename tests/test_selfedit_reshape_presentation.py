"""Tests for the unified reshape presentation contract (one UX, every surface).

Pure mapping. Load-bearing:
  - auto-applied (VERIFIED) → "Done" + Undo (post-hoc, reversible);
  - applied-pending → Keep/Undo (see-it/keep-it);
  - reverted/failed → honest message + Retry (never silent);
  - unmapped → honest "couldn't" + Open Workshop (no black-box no-op);
  - actions carry the attempt_id + revert_token; speech is populated for voice.
"""

from __future__ import annotations

from augmentum.selfedit.surfaces import (
    CaptureArtifact,
    ReshapeChange,
    present,
    proposed_presentation,
)
from augmentum.selfedit.surfaces.engine import (
    STATUS_APPLIED_PENDING,
    STATUS_FAILED,
    STATUS_PROMOTED,
    STATUS_REVERTED,
    STATUS_UNMAPPED,
    EngineResult,
)
from augmentum.selfedit.surfaces.presentation import (
    ACT_KEEP,
    ACT_OPEN_WORKSHOP,
    ACT_UNDO,
    PRES_APPLIED,
    PRES_CONFIRM,
    PRES_FAILED,
    PRES_PROPOSED,
    PRES_REVERTED,
    PRES_UNMAPPED,
)
from augmentum.selfedit.surfaces.reshape import ReshapeResult


def _result(status, *, intent="set theme=dark", token="tok", summary="theme = 'dark'"):
    change = ReshapeChange(surface="config", change_class="adaptation",
                           payload={"key": "theme", "value": "dark"}, intent=intent, actor="u1")
    rs = ReshapeResult(surface="config", applied=True, kept=status != STATUS_REVERTED,
                       auto_promotable=status == STATUS_PROMOTED,
                       needs_human=status == STATUS_APPLIED_PENDING, revert_token=token,
                       capture=CaptureArtifact(kind="state", ref="theme", summary=summary))
    return EngineResult(attempt_id="att1", mapped=status != STATUS_UNMAPPED,
                        status=status, change=change, reshape=rs)


def _act_ids(p):
    return [a.id for a in p.actions]


def test_promoted_is_done_with_undo():
    p = present(_result(STATUS_PROMOTED))
    assert p.status == PRES_APPLIED
    assert "set theme=dark" in p.headline and "theme = 'dark'" in p.detail
    assert _act_ids(p) == [ACT_UNDO]
    undo = p.actions[0]
    assert undo.attempt_id == "att1" and undo.revert_token == "tok"
    assert "undo" in p.speech.lower()                     # voice gets a spoken undo cue


def test_applied_pending_is_keep_or_undo():
    p = present(_result(STATUS_APPLIED_PENDING))
    assert p.status == PRES_CONFIRM
    assert _act_ids(p) == [ACT_KEEP, ACT_UNDO]
    assert "keep" in p.headline.lower()


def test_reverted_and_failed_are_honest_with_retry():
    rev = present(_result(STATUS_REVERTED))
    assert rev.status == PRES_REVERTED and "retry" in _act_ids(rev)
    fail = present(_result(STATUS_FAILED))
    assert fail.status == PRES_FAILED and ACT_OPEN_WORKSHOP in _act_ids(fail)


def test_unmapped_is_honest_not_silent():
    p = present(_result(STATUS_UNMAPPED))
    assert p.status == PRES_UNMAPPED
    assert "couldn't" in p.headline.lower()               # honest, never a silent no-op
    assert _act_ids(p) == [ACT_OPEN_WORKSHOP]


def test_proposed_presentation_uses_offer_vocabulary():
    p = proposed_presentation(_result(STATUS_APPLIED_PENDING))
    assert p.status == PRES_PROPOSED
    assert _act_ids(p) == ["approve", "dismiss", "never"]  # the offer-chip path
    assert "want me to" in p.headline.lower()
