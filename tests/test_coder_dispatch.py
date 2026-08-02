"""Tests for the orchestrator dispatch contract (P4).

Covers ``CoderDispatch.for_orchestrator_dispatch`` (the full-contract
constructor) and its rendering, plus the CoderHandler seam that stores a
dispatch and precomputes its fork-system block.
"""
from __future__ import annotations

from augmentum.coder.dispatch import (
    CoderDispatch,
    render_dispatch_system,
)
from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
from augmentum.promises.models import Promise, Verification, VerificationKind


def _promise(desc: str) -> Promise:
    return Promise(description=desc, verify=Verification(kind=VerificationKind.ALWAYS))


# ── for_orchestrator_dispatch ───────────────────────────────────────────────

def test_orchestrator_dispatch_populates_full_contract():
    d = CoderDispatch.for_orchestrator_dispatch(
        workspace_id="ws_1", user_id="u1", task="build the parser",
        success_criteria=(_promise("parser handles nested lists"),),
        constraints=("no new deps",),
        context_brief={"user_mood": "focused"},
        cost_tier="thorough", permission_mode="confirm_mutations",
    )
    assert d.workspace_id == "ws_1"
    assert d.task == "build the parser"
    assert len(d.success_criteria) == 1
    assert d.constraints == ("no new deps",)
    assert d.cost_tier == "thorough"
    assert d.permission_mode == "confirm_mutations"
    assert d.context_brief == {"user_mood": "focused"}


def test_orchestrator_dispatch_defaults_match_balanced_auto():
    d = CoderDispatch.for_orchestrator_dispatch(
        workspace_id="ws_1", user_id="u1", task="t",
    )
    assert d.success_criteria == ()
    assert d.cost_tier == "balanced"
    assert d.permission_mode == "auto"
    assert d.journal_scope == "workspace"


def test_direct_user_turn_stays_minimal():
    d = CoderDispatch.for_direct_user_turn(workspace_id="ws", user_id="u", task="t")
    assert d.success_criteria == ()
    assert d.context_brief is None


# ── render_dispatch_system ──────────────────────────────────────────────────

def test_render_includes_task_criteria_and_mode():
    d = CoderDispatch.for_orchestrator_dispatch(
        workspace_id="ws_1", user_id="u1", task="build the parser",
        success_criteria=(_promise("parser handles nested lists"),),
        permission_mode="confirm_mutations",
    )
    rendered = render_dispatch_system(d, fork_prompt=DISPATCH_FORK_SYSTEM)
    assert "build the parser" in rendered
    assert "parser handles nested lists" in rendered
    assert "confirm_mutations" in rendered


def test_render_empty_criteria_notes_tqg_fallback():
    d = CoderDispatch.for_orchestrator_dispatch(
        workspace_id="ws_1", user_id="u1", task="t",
    )
    rendered = render_dispatch_system(d, fork_prompt=DISPATCH_FORK_SYSTEM)
    assert "Termination Quality Gate" in rendered  # empty-criteria fallback text


# ── CoderHandler seam (P4.2) ────────────────────────────────────────────────

def test_handler_stores_dispatch_and_precomputes_block():
    from augmentum.modes.coder.handler import CoderHandler
    d = CoderDispatch.for_orchestrator_dispatch(
        workspace_id="ws_1", user_id="u1", task="build the parser",
        success_criteria=(_promise("parser handles nested lists"),),
    )
    h = CoderHandler(object(), user_id="u1", session_id="ws_1", dispatch=d)
    assert h._dispatch is d
    assert "build the parser" in h._dispatch_system_block
    assert "parser handles nested lists" in h._dispatch_system_block


def test_handler_without_dispatch_has_empty_block():
    from augmentum.modes.coder.handler import CoderHandler
    h = CoderHandler(object(), user_id="u1", session_id="ws_1")
    assert h._dispatch is None
    assert h._dispatch_system_block == ""
