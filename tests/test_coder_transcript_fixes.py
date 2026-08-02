"""Regression tests for issues captured in the 2026-04-21 Pong-game transcript.

Two distinct leaks were surfaced in the real-world session:

  1. Plan-phase preamble filter missed one-line prose. The regex
     ``(?:^|\\n)\\s*(Plan|Question)\\s*:`` required a newline before the
     marker; weak models concatenated "The user is asking... Plan: do
     X" onto one logical line, so the preamble streamed to chat
     verbatim.

  2. ``shell_exec`` output was "(command produced no output)" on exit
     0 with empty stdout. The model interpreted the ambiguous phrasing
     as "something might be wrong" and re-ran the same compile checks
     up to 8 times in a row. Explicit "exit 0, command succeeded" is
     unambiguous.
"""
from __future__ import annotations

from augmentum.modes.coder.phase_plan import _PLAN_MARKER_RE

# ---------------------------------------------------------------------------
# Plan-phase marker regex
# ---------------------------------------------------------------------------


def test_one_line_preamble_gets_cut():
    """The actual 2026-04-21 failure. Preamble + marker on one line."""
    s = (
        "The user is asking what can we do here — this is an "
        "INFORMATIONAL request. I should read the project files. "
        "Plan: inspect workspace contents and summarise\n"
        "1. Run dir_tree on /workspace"
    )
    m = _PLAN_MARKER_RE.search(s)
    assert m is not None, "Expected to match 'Plan:' anywhere in the string"
    visible = s[m.start():]
    assert visible.startswith("Plan:")
    assert "Run dir_tree" in visible
    # Crucially — the pre-marker monologue should not be in the
    # portion we'd stream to chat.
    assert "The user is asking" not in visible
    assert "INFORMATIONAL" not in visible


def test_newline_preamble_still_matches():
    """The 'proper' grammar — plan on its own line — still matches."""
    s = (
        "<think>model internal</think>\n\n"
        "Plan: describe project contents\n"
        "1. dir_tree on /workspace\n"
        "2. Summarise"
    )
    m = _PLAN_MARKER_RE.search(s)
    assert m is not None
    visible = s[m.start():]
    assert visible.startswith("Plan: describe")


def test_question_marker_matches():
    """VAGUE branch of PLAN_SYSTEM emits 'Question:' instead."""
    s = "Pre-chatter. Question: what kind of file would you like?"
    m = _PLAN_MARKER_RE.search(s)
    assert m is not None
    assert s[m.start():].startswith("Question:")


def test_lowercase_plan_in_prose_does_not_match():
    """Case-sensitive — 'my plan:' in casual prose should NOT be
    treated as the section marker, or we'd falsely truncate a user's
    answer that happens to mention planning.
    """
    s = "My plan: eat lunch. Then code."
    m = _PLAN_MARKER_RE.search(s)
    # ``\bplan:\b`` is case-sensitive in our regex → no match on
    # lowercase.
    assert m is None


def test_supplant_does_not_match():
    """Word-boundary — words containing 'plan' as substring do NOT
    trigger. 'Supplant:' is not the grammar marker.
    """
    s = "This would supplant: the existing approach. Plan: actually start."
    m = _PLAN_MARKER_RE.search(s)
    # Must match the ACTUAL 'Plan:' further on, not 'supplant:'
    assert m is not None
    assert s[m.start():].startswith("Plan:")


def test_empty_string_no_match():
    assert _PLAN_MARKER_RE.search("") is None


def test_marker_at_start():
    s = "Plan: do X\n1. step"
    m = _PLAN_MARKER_RE.search(s)
    assert m is not None
    assert m.start() == 0


# ---------------------------------------------------------------------------
# shell_exec silent-success messaging
# ---------------------------------------------------------------------------


def test_shell_exec_silent_success_message_is_unambiguous():
    """Regression guard: the string returned for a successful command
    with no stdout must explicitly indicate success + exit code, not
    "(command produced no output)" which weak models misread as
    potentially-failed and retry.
    """
    import augmentum.coder.tools as tools_src
    src = open(tools_src.__file__, encoding="utf-8").read()
    # The old ambiguous phrasing should be GONE
    assert "command produced no output" not in src, (
        "'command produced no output' triggered 8+ retries in the "
        "2026-04-21 Pong transcript; replace with an explicit "
        "success signal like 'exit 0, command succeeded with no stdout'."
    )
    # The new phrasing must clearly signal success
    assert "exit 0" in src
