"""Guards the salvage-path invariant: a coherent transcript is never silently
dropped on a *reachable-backend* router failure.

``_regex_fallback`` is only reached on timeout / parse / error failures — all
three mean the backend was reachable and STT produced a coherent transcript;
only the router call hiccuped. ``error_fallback`` used to be the odd one out: on
a ``no_signal`` transcript it DROPPED (silent), while ``timeout``/``parse``
leaned addressed. That was the last silent-on-coherent path in the router; these
tests pin all three to one consistent, engage-not-drop posture (Matt 2026-07-27,
the silent-subtractive-input-gates invariant). Fixtures' signals are verified
against is_addressed() in the preconditions so the test can't silently rot.
"""
from __future__ import annotations

import pytest

from augmentum.architect.address import is_addressed
from augmentum.architect.voice_router import _regex_fallback

_SALVAGEABLE = ("timeout_fallback", "parse_fallback", "error_fallback")

# Coherent one-on-one statement that hits no canonical structure → no_signal.
_NO_SIGNAL = "the coder run finished a moment ago"
# Start-anchored narration about another person → high-precision ambient.
_THIRD_PERSON = "he told her to leave the room"
# Low-precision declarative musing → self_talk.
_SELF_TALK = "i think the build is done"


def test_fixture_signals_are_what_we_think():
    # Precondition: if these drift, the assertions below are meaningless — fail
    # loudly instead of silently testing nothing.
    assert is_addressed(_NO_SIGNAL).signal == "no_signal"
    assert is_addressed(_THIRD_PERSON).signal == "third_person"
    assert is_addressed(_SELF_TALK).signal == "self_talk"


def test_error_fallback_engages_coherent_no_signal_not_drop():
    # THE regression: error_fallback on a coherent no_signal transcript.
    d = _regex_fallback(_NO_SIGNAL, "m", 100, "error_fallback")
    assert d.addressed is True, "error_fallback silently dropped a coherent turn"
    assert d.goal == "converse"


@pytest.mark.parametrize("text", [_NO_SIGNAL, "play some jazz", "what time is it in tokyo", _SELF_TALK])
def test_all_salvageable_failures_are_consistent(text):
    # timeout / parse / error are ONE class — same verdict for the same text.
    verdicts = {
        pf: (_regex_fallback(text, "m", 100, pf).addressed,
             _regex_fallback(text, "m", 100, pf).goal)
        for pf in _SALVAGEABLE
    }
    assert len(set(verdicts.values())) == 1, f"inconsistent across failures: {verdicts}"


def test_strong_ambient_third_person_still_drops_on_error_fallback():
    # Don't over-engage: high-precision narration still drops on every path.
    for pf in _SALVAGEABLE:
        d = _regex_fallback(_THIRD_PERSON, "m", 100, pf)
        assert d.addressed is False and d.goal == "drop", f"{pf} over-engaged narration"


def test_self_talk_released_on_error_fallback():
    # Low-precision self_talk is released (engaged) on all salvageable failures
    # — the regex can't tell musing from a real continuation, so on a backend-
    # reachable failure it must not eat the turn.
    d = _regex_fallback(_SELF_TALK, "m", 100, "error_fallback")
    assert d.addressed is True
