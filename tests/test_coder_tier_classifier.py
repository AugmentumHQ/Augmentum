"""Tests for ``classify_tier`` — Phase 1.1 of the coder foundation.

Covers each tier's positive trigger, the disqualifying signals that
keep REFLEX terse + single-action, and the COMPOSED default.
"""
from __future__ import annotations

import pytest

from augmentum.modes.coder.intent import (
    TIER_LIMITS,
    Tier,
    TierClassification,
    classify_tier,
)

# ---------------------------------------------------------------------------
# REFLEX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "Add the missing json import to main.py",
    "Fix the typo in the README",
    "Rename foo to bar in utils.py",
    "Remove the unused variable in handler.py",
    "Import requests at the top of fetch.py",
    "Update the version string to 1.2.3",
    "Change the timeout from 5 to 10",
])
def test_reflex_for_terse_single_actions(msg: str) -> None:
    c = classify_tier(latest_text=msg)
    assert c.tier == Tier.REFLEX, f"{msg!r} → {c}"
    assert c.reason == "terse_single_action"


@pytest.mark.parametrize("msg", [
    # Multi-step: "and then"
    "Add the json import and then update main.py to use it",
    # Multi-step: "after that"
    "Fix the typo, after that run the tests",
    # Scope-broadening: "across"
    "Rename foo to bar across the codebase",
    # Scope-broadening: "everywhere"
    "Update the import statement everywhere",
    # Scope-broadening: "every file"
    "Change the timeout in every file",
    # Refactor language disqualifies even with reflex verb
    "Update the function and refactor the surrounding logic",
])
def test_reflex_disqualified_by_scope_signals(msg: str) -> None:
    c = classify_tier(latest_text=msg)
    assert c.tier != Tier.REFLEX, f"{msg!r} unexpectedly REFLEX: {c}"


def test_reflex_disqualified_by_length() -> None:
    long_msg = (
        "Add the missing json import to main.py because the function "
        "I introduced last week relies on it for serialization output "
        "and the tests are failing because of that"
    )
    assert len(long_msg) > 140
    c = classify_tier(latest_text=long_msg)
    assert c.tier != Tier.REFLEX


# ---------------------------------------------------------------------------
# SURGICAL — single action verb but not terse enough for REFLEX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    # Has reflex verb but exceeds 140 chars
    "Add a max_retries parameter to fetch_data, defaulting to 3, with "
    "exponential backoff between attempts and a clear error message.",
    # Implement verb without compose-broadening signals
    "Implement a cache layer for the database adapter with TTL support",
    # Debug verb
    "Debug why the integration test is failing intermittently",
    "Investigate the memory leak in the worker pool",
])
def test_surgical_for_single_action_non_terse(msg: str) -> None:
    c = classify_tier(latest_text=msg)
    assert c.tier == Tier.SURGICAL, f"{msg!r} → {c}"


# ---------------------------------------------------------------------------
# COMPOSED — multi-file / cross-cutting language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "Refactor the auth handler into smaller pieces",
    "Extract the duplicate date-parsing logic into a shared helper",
    "Consolidate the error-handling across the codebase",
    "Restructure the modules so config lives in its own package",
    "Deduplicate the validation logic across the routes",
    "Split the giant handler.py into focused mixins",
])
def test_composed_for_cross_cutting_language(msg: str) -> None:
    c = classify_tier(latest_text=msg)
    assert c.tier == Tier.COMPOSED, f"{msg!r} → {c}"


def test_composed_default_when_no_signals() -> None:
    c = classify_tier(latest_text="show me")
    assert c.tier == Tier.COMPOSED
    assert c.reason == "default"


def test_composed_default_when_message_empty() -> None:
    c = classify_tier(latest_text="")
    assert c.tier == Tier.COMPOSED
    assert c.reason == "default"


# ---------------------------------------------------------------------------
# PROJECT — from-scratch / migration / build-a-thing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("msg", [
    "Build a CLI calculator from scratch",
    "Create an app that tracks expenses",
    "Scaffold a new service with FastAPI",
    "Set up a web app for managing tasks",
    "Migrate from requests to httpx",
    "Port from Python 2 to Python 3",
    "Rewrite the auth layer in async",
])
def test_project_for_creation_and_migration(msg: str) -> None:
    c = classify_tier(latest_text=msg)
    assert c.tier == Tier.PROJECT, f"{msg!r} → {c}"


def test_project_when_empty_workspace_and_creation_verb() -> None:
    # Even without explicit "from scratch" phrasing, an empty workspace
    # with a creation verb is PROJECT.
    c = classify_tier(latest_text="Make a small calculator", workspace_file_count=0)
    assert c.tier == Tier.PROJECT
    assert c.reason == "empty_workspace_with_creation_verb"


def test_non_empty_workspace_with_creation_verb_falls_through() -> None:
    # Workspace has files → don't auto-promote to PROJECT just because
    # the verb is "make".
    c = classify_tier(latest_text="Make the timeout configurable", workspace_file_count=20)
    assert c.tier != Tier.PROJECT


# ---------------------------------------------------------------------------
# Goal-text continuation behavior (mirrors classify_turn_intent contract)
# ---------------------------------------------------------------------------


def test_continuation_uses_goal_text_for_classification() -> None:
    # User said "continue" but goal_text carries the actual intent
    c = classify_tier(latest_text="continue", goal_text="Refactor the auth handler")
    assert c.tier == Tier.COMPOSED


def test_latest_text_wins_over_goal_text_when_present() -> None:
    # If both signal but conflict, goal_text is the subject. This tests
    # that goal_text takes precedence as documented.
    c = classify_tier(
        latest_text="add a comment",
        goal_text="Refactor the entire auth subsystem across the codebase",
    )
    # goal_text is checked first (subject = goal_text or latest_text)
    assert c.tier == Tier.COMPOSED


# ---------------------------------------------------------------------------
# TIER_LIMITS sanity
# ---------------------------------------------------------------------------


def test_tier_limits_covers_every_tier() -> None:
    for tier in Tier:
        assert tier in TIER_LIMITS, f"missing limit for {tier!r}"


def test_tier_limits_are_monotonic_in_iterations() -> None:
    """Each tier should allow at least as many iterations as the one below."""
    order = [Tier.REFLEX, Tier.SURGICAL, Tier.COMPOSED, Tier.PROJECT]
    iters = [TIER_LIMITS[t].max_iterations for t in order]
    assert iters == sorted(iters), f"non-monotonic iterations: {iters}"


def test_tier_limits_are_monotonic_in_tokens() -> None:
    order = [Tier.REFLEX, Tier.SURGICAL, Tier.COMPOSED, Tier.PROJECT]
    toks = [TIER_LIMITS[t].max_tokens for t in order]
    assert toks == sorted(toks), f"non-monotonic tokens: {toks}"


def test_reflex_limits_are_strict() -> None:
    """REFLEX is supposed to be cheap and fast — guard against drift."""
    limit = TIER_LIMITS[Tier.REFLEX]
    assert limit.max_iterations <= 3
    assert limit.max_tokens <= 3000


# ---------------------------------------------------------------------------
# Returned classification shape
# ---------------------------------------------------------------------------


def test_classification_carries_signal_trail() -> None:
    c = classify_tier(latest_text="Add the json import")
    assert isinstance(c, TierClassification)
    assert c.signals, "signals should not be empty for a matched classification"


def test_classification_default_has_empty_signals() -> None:
    c = classify_tier(latest_text="show me")
    assert c.tier == Tier.COMPOSED
    assert c.signals == ()
