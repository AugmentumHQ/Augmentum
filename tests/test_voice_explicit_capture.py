"""Guards the invariant: explicitly-captured input is never silently ignored.

The load-bearing rule (Matt 2026-07-26): typing into the stage manager and
hitting Send is UNAMBIGUOUS intent — the companion must never be able to drop
it. This regressed once already (the promotion covered ``idle`` but not
``drop``, so a coherent paragraph the router tagged ``goal=drop``/
``addressed=False`` was silently discarded). These tests pin the pure decision
helpers extracted from ``_maybe_dispatch_intent`` so a third regression trips
CI, not a user.
"""
from __future__ import annotations

import pytest

from augmentum.proxy.voice_routes import (
    _explicit_addressed_effective,
    _promote_explicit_goal,
)

_ALL_GOALS = ("act", "converse", "clarify", "idle", "drop")


# ── The core guarantee: stage-send is never ignored ─────────────────────────

@pytest.mark.parametrize("goal", _ALL_GOALS)
def test_stage_send_is_never_ignored_for_any_goal(goal):
    """No router verdict — not even drop with addressed=False and
    coherent=False — can make a typed+Sent turn resolve to 'ignore'."""
    eff_goal = _promote_explicit_goal(explicit_capture=True, goal=goal)
    engaged = _explicit_addressed_effective(
        from_stage_send=True, explicit_capture=True,
        coherent=False, addressed=False, confidence=0.0,
        goal=eff_goal, effective_threshold=0.99, in_followup=False,
    )
    assert engaged is True, f"stage-send was ignored on goal={goal}"


# ── Goal promotion ──────────────────────────────────────────────────────────

def test_explicit_capture_promotes_idle_and_drop_to_converse():
    assert _promote_explicit_goal(explicit_capture=True, goal="idle") == "converse"
    assert _promote_explicit_goal(explicit_capture=True, goal="drop") == "converse"


@pytest.mark.parametrize("goal", ("act", "converse", "clarify"))
def test_explicit_capture_leaves_actionable_goals_untouched(goal):
    assert _promote_explicit_goal(explicit_capture=True, goal=goal) == goal


@pytest.mark.parametrize("goal", _ALL_GOALS)
def test_non_explicit_never_promotes(goal):
    # Ambient input keeps the router's verdict verbatim.
    assert _promote_explicit_goal(explicit_capture=False, goal=goal) == goal


# ── Explicit PTT (not stage-send) keeps the coherence veto ──────────────────

def test_ptt_explicit_engages_on_coherent_converse():
    assert _explicit_addressed_effective(
        from_stage_send=False, explicit_capture=True,
        coherent=True, addressed=False, confidence=0.0,
        goal="converse", effective_threshold=0.7, in_followup=False,
    ) is True


def test_ptt_explicit_still_drops_incoherent_garbage():
    # A cough on an open mic shouldn't force a reply — coherence veto stands
    # for spoken explicit capture (this is why stage-send is a separate branch).
    assert _explicit_addressed_effective(
        from_stage_send=False, explicit_capture=True,
        coherent=False, addressed=True, confidence=1.0,
        goal="converse", effective_threshold=0.7, in_followup=False,
    ) is False


# ── Ambient uses the confidence / followup gate ─────────────────────────────

def test_ambient_below_threshold_not_engaged():
    assert _explicit_addressed_effective(
        from_stage_send=False, explicit_capture=False,
        coherent=True, addressed=True, confidence=0.5,
        goal="converse", effective_threshold=0.7, in_followup=False,
    ) is False


def test_ambient_in_followup_window_engaged():
    assert _explicit_addressed_effective(
        from_stage_send=False, explicit_capture=False,
        coherent=True, addressed=False, confidence=0.0,
        goal="converse", effective_threshold=0.7, in_followup=True,
    ) is True


def test_ambient_idle_goal_not_engaged_even_if_addressed():
    # idle is NOT promoted for ambient, so a bare "thanks" overheard stays quiet.
    assert _explicit_addressed_effective(
        from_stage_send=False, explicit_capture=False,
        coherent=True, addressed=True, confidence=0.99,
        goal="idle", effective_threshold=0.7, in_followup=False,
    ) is False


# ── The engage decision is binary: only idle/drop stay silent ───────────────

def test_unknown_goal_engages_never_silently_dropped():
    # The predicate is "not in {idle, drop}", not an act/converse/clarify
    # allow-list — so a new or unexpected goal ENGAGES rather than vanishing.
    # This is the never-ignore invariant applied to the taxonomy itself.
    for surface in (
        dict(from_stage_send=False, explicit_capture=True, coherent=True,
             addressed=False, confidence=0.0, in_followup=False),
        dict(from_stage_send=False, explicit_capture=False, coherent=True,
             addressed=True, confidence=0.99, in_followup=False),
    ):
        assert _explicit_addressed_effective(
            goal="handoff", effective_threshold=0.7, **surface,
        ) is True, "an unknown goal was silently dropped"
