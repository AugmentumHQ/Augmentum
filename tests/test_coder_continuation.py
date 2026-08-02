"""Tests for the continuation-request detector + goal-split helper.

Motivation: observed 2026-04-22 in a RetroArch setup transcript — the
user says "continue please" or "monitor the download", and the agent
re-runs its plan phase, re-derives intent from scratch, and parrots
the original "what is this?" project summary. The continuation
detector short-circuits the plan phase and routes the turn straight
to act; the goal-split helper ensures the sticky reminder carries
BOTH the substantive goal AND the latest input so the model doesn't
treat "continue" as a new objective.

Coverage:

* **Positive continuations** — pure ("continue", "keep going"),
  polite ("continue please"), status queries ("what's the status"),
  monitor/check requests ("monitor the download").
* **Negative** — long instructions that happen to start with
  continuation words ("continue to add tests") MUST NOT match. A
  regression here would route real work requests to the act phase
  with no fresh plan.
* **Goal-split walk-back** — latest=continuation, prior=substantive
  returns both correctly. All-continuations session returns the
  latest for both. Empty message list returns empty tuple. System-
  reminder self-injections are skipped during the walk.

Run: python -m pytest tests/test_coder_continuation.py -v
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from augmentum.modes.coder.handler import (
    _extract_goal_split,
    _is_continuation_request,
)

# ---------------------------------------------------------------------------
# Continuation detector — positive matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "continue",
    "continue.",
    "continue please",
    "please continue",
    "Continue Please",       # case-insensitive
    "  continue  please  ",  # whitespace-tolerant
    "keep going",
    "go ahead",
    "go on",
    "proceed",
    "carry on",
    "resume",
    "resume please",
    # Monitor / status forms
    "monitor the download",
    "monitor download",
    "watch the build",
    "check on the install",
    "please monitor the download",
    # Status queries
    "what's the status",
    "what's the progress",
    "how's it going",
    "how is it going",
    "any update",
    "any updates",
    "any updates?",
    "update",
    "status",
    "status?",
    "is it done",
    "is it done?",
    "done?",
    # Polite padding
    "continue thanks",
    "continue thank you",
    "proceed please",
    # Standalone politeness tokens count as continuation
    "please",
    "thanks",
])
def test_is_continuation_request_positive(text):
    assert _is_continuation_request(text) is True, f"should match: {text!r}"


# ---------------------------------------------------------------------------
# Continuation detector — NEGATIVE (must not false-positive on real work)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    # Real instructions that START with a continuation keyword but carry
    # substantive content. Anchored regex must NOT treat these as
    # continuations — doing so would route a real work request to the
    # act phase with stale state.
    "continue to add more tests",
    "continue refactoring the auth module",
    "keep going with the cleanup but also fix the imports",
    "proceed with the migration to sqlite",
    # Questions about the project (not status of a task)
    "what is this?",
    "what does this do?",
    # Creation verbs
    "build the thing",
    "run the tests",
    "write a function for X",
    # Empty / whitespace
    "",
    "   ",
    "\n\n",
    # Long narrative
    "I'd like you to take a look at the auth middleware and refactor it",
    # "monitor" as a verb with extra content beyond the target
    "monitor the download and also check the disk usage on the side",
])
def test_is_continuation_request_negative(text):
    assert _is_continuation_request(text) is False, f"should not match: {text!r}"


# ---------------------------------------------------------------------------
# Goal-split walk-back
# ---------------------------------------------------------------------------


@dataclass
class _Msg:
    """Minimal stand-in for ``Message`` — the helper only reads role +
    content via getattr. Avoids pulling the real model module into
    these unit tests."""
    role: str
    content: str


def test_extract_goal_split_empty_messages():
    latest, goal = _extract_goal_split([])
    assert latest == ""
    assert goal == ""


def test_extract_goal_split_only_system():
    """System-only message list shouldn't fabricate a goal."""
    latest, goal = _extract_goal_split([_Msg("system", "you are helpful")])
    assert latest == ""
    assert goal == ""


def test_extract_goal_split_latest_is_substantive():
    msgs = [
        _Msg("system", "sys"),
        _Msg("user", "hey"),
        _Msg("assistant", "hi"),
        _Msg("user", "what's in this repo?"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "what's in this repo?"
    assert goal == "what's in this repo?"


def test_extract_goal_split_walks_back_past_continuation():
    """Core case: latest is 'continue please', prior substantive
    message is the real goal. Split must surface both."""
    msgs = [
        _Msg("system", "sys"),
        _Msg("user", "set up retroarch and serve it on port 8080"),
        _Msg("assistant", "running..."),
        _Msg("user", "continue please"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "continue please"
    assert goal == "set up retroarch and serve it on port 8080"


def test_extract_goal_split_walks_past_multiple_continuations():
    """Two continuations in a row — walk all the way back to the
    substantive message."""
    msgs = [
        _Msg("user", "write unit tests for auth.py"),
        _Msg("assistant", "..."),
        _Msg("user", "continue"),
        _Msg("assistant", "..."),
        _Msg("user", "keep going"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "keep going"
    assert goal == "write unit tests for auth.py"


def test_extract_goal_split_all_continuations_falls_back_to_latest():
    """Session opens with continuation phrases and never gets a
    substantive user message — return the latest as goal too. Rare
    but possible (user types 'continue' into a session where a task
    was set programmatically from a prior browser tab, etc.)."""
    msgs = [
        _Msg("user", "continue"),
        _Msg("assistant", "..."),
        _Msg("user", "keep going"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "keep going"
    assert goal == "keep going"


def test_extract_goal_split_skips_sticky_reminder_messages():
    """Sticky reminders are injected as user messages starting with
    '<system-reminder>'. They echo the goal and would short-circuit
    the walk-back (treating the reminder as the 'prior substantive'
    message). Must be filtered so the walk reaches the true prior
    user turn."""
    msgs = [
        _Msg("user", "debug the flaky test in test_api.py"),
        _Msg("user", "<system-reminder>\nGoal: debug the flaky test"),
        _Msg("assistant", "..."),
        _Msg("user", "continue"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "continue"
    assert goal == "debug the flaky test in test_api.py"


def test_extract_goal_split_strips_whitespace():
    msgs = [
        _Msg("user", "  build the project   "),
        _Msg("user", "  continue please  "),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "continue please"
    assert goal == "build the project"


def test_extract_goal_split_skips_empty_user_messages():
    """Empty / whitespace-only user messages are treated as non-
    existent — the helper walks past them to find real content."""
    msgs = [
        _Msg("user", "set up nginx"),
        _Msg("user", "   "),
        _Msg("user", ""),
        _Msg("user", "continue"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "continue"
    assert goal == "set up nginx"
