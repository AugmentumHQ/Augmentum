"""Tests for coder goal-cleaning — _clean_user_text + its seams.

The coder UI wraps user input as ``[Terminal context]\\n<buffer>\\n// <intent>``.
Left raw, that envelope pollutes the turn archive's ``user_goal`` (→ recall
embeddings + display), the ``<prior_turns>`` ring, and sticky reminders — a live
audit (2026-06-25) found ~50% of archived goals were terminal-context noise.
``_clean_user_text`` de-wraps at every seam where a raw message becomes a goal.
"""
from __future__ import annotations

from augmentum.modes.coder.handler import _clean_user_text, _extract_goal_split


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


# ---------------------------------------------------------------------------
# _clean_user_text
# ---------------------------------------------------------------------------

def test_clean_extracts_intent_line():
    raw = "[Terminal context]\n  3 | import time\n  ✓ ran tests\n// fix the auth 401 bug"
    assert _clean_user_text(raw) == "fix the auth 401 bug"


def test_clean_last_intent_line_wins():
    raw = "// first\nsome buffer\n// the real ask"
    assert _clean_user_text(raw) == "the real ask"


def test_clean_strips_terminal_preamble_no_intent():
    raw = "[Terminal context]\n   ✓ Applied 1 edits to '/workspace/app.py'\n\nadd a healthcheck endpoint"
    # No // intent → preamble stripped, first remaining line returned.
    assert _clean_user_text(raw) == "add a healthcheck endpoint"


def test_clean_intent_beats_preamble():
    raw = "[Terminal context]\nblah blah\n// actual intent here"
    assert _clean_user_text(raw) == "actual intent here"


def test_clean_plain_text_passthrough_full():
    raw = "refactor the parser\nand add tests"
    assert _clean_user_text(raw, single_line=False) == "refactor the parser\nand add tests"


def test_clean_plain_text_single_line():
    raw = "refactor the parser\nand add tests"
    assert _clean_user_text(raw, single_line=True) == "refactor the parser"


def test_clean_empty():
    assert _clean_user_text("") == ""
    assert _clean_user_text(None) == ""  # type: ignore[arg-type]


def test_clean_is_idempotent_on_clean_input():
    once = _clean_user_text("[Terminal context]\nbuf\n// do the thing", single_line=False)
    assert _clean_user_text(once, single_line=False) == once == "do the thing"


def test_clean_real_live_junk_sample():
    # Verbatim shape from the live archive audit (T44).
    raw = (
        "[Terminal context]\n"
        "   ✓ Applied 1 edits to '/workspace/erome-index/templates/search.html' "
        "atomically (exact: 1)."
    )
    out = _clean_user_text(raw, single_line=False)
    assert not out.startswith("[Terminal context]")


# ---------------------------------------------------------------------------
# _extract_goal_split — the wrapped goal is cleaned end-to-end
# ---------------------------------------------------------------------------

def test_extract_goal_split_cleans_terminal_wrapper():
    msgs = [_Msg("user", "[Terminal context]\n  buffer noise\n// set up retroarch on port 8080")]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "set up retroarch on port 8080"
    assert goal == "set up retroarch on port 8080"


def test_extract_goal_split_plain_unchanged():
    # Regression guard: plain messages (the existing continuation tests) are
    # untouched by cleaning.
    msgs = [_Msg("user", "what's in this repo?")]
    latest, goal = _extract_goal_split(msgs)
    assert latest == goal == "what's in this repo?"


def test_extract_goal_split_walks_back_past_wrapped_continuation():
    # A wrapped "continue" should be recognized as a continuation (cleaning runs
    # BEFORE the continuation check), walking back to the wrapped substantive goal.
    msgs = [
        _Msg("user", "[Terminal context]\nx\n// add timezone handling to the API"),
        _Msg("user", "[Terminal context]\ny\n// continue"),
    ]
    latest, goal = _extract_goal_split(msgs)
    assert latest == "continue"
    assert goal == "add timezone handling to the API"
