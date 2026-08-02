"""Sovereign Perception Pipeline — the L3 judgment gate + interruption budget.

This is the proof that the pipeline is aware, not an echo machine: the SAME
insight becomes silence, a glanceable digest line, or a spoken interruption purely
by the gate's decision. Load-bearing:
  - pull-first default (worth keeping ≠ worth interrupting);
  - regret gates the COSTLY channel (a dismissive user raises the interrupt bar);
  - in-conversation is the cheapest channel (no budget, lower bar);
  - the interruption budget makes over-nagging structurally impossible;
  - consequential actions route to gated consent, never auto-fire;
  - expired insights die quietly.
"""

from __future__ import annotations

from augmentum.companion_runtime.perception import (
    ACT_WITH_CONSENT,
    FILE_FOR_PULL,
    SILENT,
    SPEAK,
    Insight,
    InterruptionBudgetStore,
    JudgmentConfig,
    decide_delivery,
)
from augmentum.companion_runtime.perception.budget import (
    can_spend,
    recent_in_window,
    remaining,
)


def _insight(**kw) -> Insight:
    base = {
        "kind": "logistics.flight_change",
        "summary": "your 6pm flight slipped to 8",
        "value": 0.9,
        "confidence": 0.9,
        "time_critical": True,
    }
    base.update(kw)
    return Insight(**base)


# --- Insight shape ---------------------------------------------------------

def test_shape_defaults_to_kind_head():
    assert _insight(kind="social.repeated_contact").shape == "social"
    assert Insight(kind="standalone", summary="x").shape == "standalone"


def test_base_score_is_value_times_confidence_clamped():
    assert abs(_insight(value=0.5, confidence=0.4).base_score - 0.2) < 1e-9
    assert _insight(value=2.0, confidence=2.0).base_score == 1.0   # clamped


# --- pull-first default ----------------------------------------------------

def test_high_value_not_time_critical_goes_to_pull_not_speak():
    # worth keeping, but no reason to interrupt → the digest, not a notification
    d = decide_delivery(
        _insight(time_critical=False), regret_multiplier=1.0,
        budget_remaining=5, in_conversation=False,
    )
    assert d.channel == FILE_FOR_PULL and not d.spent_budget


def test_weak_insight_is_silent_recall_only():
    d = decide_delivery(
        _insight(value=0.3, confidence=0.3, time_critical=False),
        budget_remaining=5,
    )
    assert d.channel == SILENT   # base 0.09 < pull_floor


# --- the interrupt path (and what gates it) --------------------------------

def test_time_critical_high_value_with_budget_speaks_and_spends():
    d = decide_delivery(
        _insight(), regret_multiplier=1.0, budget_remaining=2, in_conversation=False,
    )
    assert d.channel == SPEAK and d.spent_budget is True


def test_interrupt_downgrades_to_pull_when_budget_exhausted():
    # SAME strong, time-critical insight — but the budget is spent → pull, no nag.
    d = decide_delivery(_insight(), regret_multiplier=1.0, budget_remaining=0)
    assert d.channel == FILE_FOR_PULL and not d.spent_budget
    assert "budget" in d.reason


def test_dismissive_user_raises_the_interrupt_bar():
    # regret 0.5 (they dismiss her) halves effective score → 0.81*0.5=0.405 < push_bar
    d = decide_delivery(_insight(), regret_multiplier=0.5, budget_remaining=5)
    assert d.channel == FILE_FOR_PULL   # the costly channel is gated by regret
    # an engaged user (regret>1) clears it
    d2 = decide_delivery(_insight(), regret_multiplier=1.0, budget_remaining=5)
    assert d2.channel == SPEAK


def test_time_critical_but_below_push_bar_falls_to_pull():
    d = decide_delivery(
        _insight(value=0.6, confidence=0.6), regret_multiplier=1.0, budget_remaining=5,
    )  # base 0.36 < push_bar 0.65
    assert d.channel == FILE_FOR_PULL and "push_bar" in d.reason


# --- in-conversation is the cheapest channel -------------------------------

def test_in_conversation_speaks_without_spending_budget():
    # not time-critical, zero budget — but she's already talking, so mention it free
    d = decide_delivery(
        _insight(time_critical=False, value=0.7, confidence=0.8),
        regret_multiplier=1.0, budget_remaining=0, in_conversation=True,
    )
    assert d.channel == SPEAK and d.spent_budget is False


def test_in_conversation_still_silent_when_too_weak():
    d = decide_delivery(
        _insight(time_critical=False, value=0.4, confidence=0.4),
        in_conversation=True, budget_remaining=0,
    )  # base 0.16 < pull_floor → nothing, even in conversation
    assert d.channel == SILENT


# --- consequential actions + decay -----------------------------------------

def test_consequential_action_routes_to_gated_consent():
    d = decide_delivery(
        _insight(suggested_action="message.send", stakes="disruptive"),
        budget_remaining=5,
    )
    assert d.channel == ACT_WITH_CONSENT


def test_trivial_action_does_not_force_consent():
    d = decide_delivery(
        _insight(suggested_action="navigate.open_surface", stakes="trivial_reversible"),
        budget_remaining=5,
    )
    assert d.channel != ACT_WITH_CONSENT   # trivial actions flow through normal gate


def test_expired_insight_is_silent():
    d = decide_delivery(
        _insight(expires_at=100.0), now=200.0, budget_remaining=5,
    )
    assert d.channel == SILENT and "expired" in d.reason


# --- config overrides ------------------------------------------------------

def test_config_thresholds_shift_the_boundary():
    weak = _insight(value=0.5, confidence=0.5)   # base 0.25
    # default pull_floor 0.30 → SILENT
    assert decide_delivery(weak, budget_remaining=5).channel == SILENT
    # lower the floor → it now reaches the digest
    loose = JudgmentConfig(pull_floor=0.20)
    assert decide_delivery(weak, budget_remaining=5, config=loose).channel == FILE_FOR_PULL


# --- the interruption budget (pure + store) --------------------------------

def test_recent_in_window_counts_only_inside_window():
    now = 1000.0
    ts = [now - 10, now - 100, now - (25 * 3600)]   # last one is >24h old
    assert recent_in_window(ts, now) == 2


def test_remaining_and_can_spend_pure():
    now = 1000.0
    ts = [now - 10, now - 20]
    assert remaining(ts, now, cap=3) == 1 and can_spend(ts, now, cap=3)
    assert remaining(ts, now, cap=2) == 0 and not can_spend(ts, now, cap=2)


def test_budget_store_spends_then_exhausts():
    store = InterruptionBudgetStore(cap=2)
    now = 1000.0
    assert store.remaining("u", now) == 2
    assert store.spend("u", now) is True
    assert store.spend("u", now + 1) is True
    assert store.remaining("u", now + 2) == 0
    # over-spend is structurally impossible even if a caller forgets to check
    assert store.spend("u", now + 3) is False


def test_budget_store_window_frees_up_over_time():
    store = InterruptionBudgetStore(cap=1)
    now = 1000.0
    assert store.spend("u", now) is True
    assert store.can_spend("u", now + 60) is False          # still in-window
    assert store.can_spend("u", now + 25 * 3600) is True    # window rolled past


def test_budget_store_is_per_user():
    store = InterruptionBudgetStore(cap=1)
    now = 1000.0
    store.spend("alice", now)
    assert not store.can_spend("alice", now)
    assert store.can_spend("bob", now)   # bob's budget is his own
