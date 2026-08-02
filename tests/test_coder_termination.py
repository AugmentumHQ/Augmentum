"""Termination Quality Gate tests — Phase 3.6 of the coder foundation.

Pure unit tests for the three primitives in
``augmentum/coder/termination.py`` plus the composed
``evaluate_termination`` decision tree. No Augmentum stack, no async,
no DB — the gate is pure logic and the tests stay that way.

Coverage:
* ``classify_user_demand`` — INSISTENT / PASSIVE / UNKNOWN dispatch
  across phrasing variants.
* ``classify_prose`` — EMPTY / BAILOUT / SUBSTANTIVE thresholds with
  the headline failure case pinned.
* ``intent_is_action`` — action / non-action / UNKNOWN classification.
* ``evaluate_termination`` — every branch of the decision tree, plus
  the headline regression (``"I read the file but the middle was
  elided."`` under an INSISTENT user demand must NOT accept stop).
* ``TerminationVerdict.explain`` — the human-readable explanation
  reflects the verdict.
"""
from __future__ import annotations

import pytest

from augmentum.coder.termination import (
    NUDGE_BAILOUT,
    NUDGE_INSISTENCE,
    NUDGE_NO_PROGRESS,
    REASON_ALREADY_NUDGED,
    REASON_NUDGE_BAILOUT,
    REASON_NUDGE_EMPTY,
    REASON_NUDGE_INSISTENT,
    REASON_RECENT_PROGRESS,
    REASON_SUBSTANTIVE_ACTIVE,
    REASON_SUBSTANTIVE_NON_ACTION,
    REASON_SUBSTANTIVE_PASSIVE,
    ProseKind,
    TerminationContext,
    TerminationVerdict,
    UserDemand,
    classify_prose,
    classify_user_demand,
    evaluate_termination,
    intent_is_action,
)
from augmentum.modes.coder.intent import TurnIntentKind

# ---------------------------------------------------------------------------
# Primitive 1: classify_user_demand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Continue until absolutely finished",
        "Continue until truly done",
        "Continue until the task is complete",
        "keep going until done",
        "keep going until you're finished",
        "don't stop",
        "Don't Stop now",
        "go all the way through",
        "all the way",
        "Fully implement the feature",
        "completely finish this",
        "absolutely complete the work",
        "Finish the whole thing",
        "Complete the entire task",
    ],
)
def test_classify_user_demand_insistent_variants(text):
    """Insistence is the strongest signal. Family-pattern catches every
    phrasing of the same intent without enumerating exact phrases."""
    assert classify_user_demand(text) == UserDemand.INSISTENT


@pytest.mark.parametrize(
    "text",
    [
        "Explain how this works",
        "Tell me about the auth flow",
        "walk me through the code",
        "summarize the changes",
        "describe the architecture",
        "What does this function do?",
        "Where is the config loaded?",
        "Why does the test fail?",
        "How does X interact with Y?",
    ],
)
def test_classify_user_demand_passive_variants(text):
    """Passive markers — analysis-style asks where prose is the
    expected end state. Should classify regardless of casing."""
    assert classify_user_demand(text) == UserDemand.PASSIVE


@pytest.mark.parametrize(
    "text",
    [
        "Fix the bug in cart.py",
        "Add error handling",
        "Refactor the auth handler",
        "now do it",
        "make it async",
        "",
        "   ",
    ],
)
def test_classify_user_demand_unknown_default(text):
    """No insistence / passive markers → UNKNOWN. The composed gate
    falls back to the intent_kind signal, not the demand."""
    assert classify_user_demand(text) == UserDemand.UNKNOWN


def test_classify_user_demand_insistent_beats_passive():
    """Order matters: insistence detection runs before passive. A
    message containing both signals should classify as INSISTENT —
    the user explicitly demanded completion, that supersedes the
    'analysis' read on the rest of the message."""
    text = "Tell me about the issue and don't stop until you've fixed it"
    assert classify_user_demand(text) == UserDemand.INSISTENT


# ---------------------------------------------------------------------------
# Primitive 2: classify_prose
# ---------------------------------------------------------------------------


def test_classify_prose_empty():
    """Whitespace-only or under 20 chars = EMPTY."""
    assert classify_prose("") == ProseKind.EMPTY
    assert classify_prose("   \n\n") == ProseKind.EMPTY
    assert classify_prose("Done.") == ProseKind.EMPTY  # 5 chars
    assert classify_prose("OK, wrapping up") == ProseKind.EMPTY  # 15 chars


def test_classify_prose_headline_bailout_case():
    """The exact failure case from the user-reported example. Must
    classify as BAILOUT — pre-3.6 this passed as substantive at 47
    chars >= 40, which is the bug we're fixing.

    DO NOT loosen the prose classifier in a way that lets this test
    re-pass as substantive. If you need to, raise the length floor
    AND keep the sentence-count gate. Single-sentence short prose
    is the bailout pattern, period."""
    text = "I read the file but the middle was elided."
    # 42 chars, one sentence — pre-3.6 the 40-char floor accepted this
    # as substantive. Post-3.6 it classifies as BAILOUT because the
    # length-AND-sentence-count gate requires 2+ sentences in this band.
    assert len(text) > 40
    assert classify_prose(text) == ProseKind.BAILOUT


@pytest.mark.parametrize(
    "text",
    [
        "I would need more information to continue.",
        "Let me know if you want me to proceed further.",
        "I'll add the helper next.",
        "I can implement this if you'd like.",
        "Should I continue with the next step?",
    ],
)
def test_classify_prose_other_bailouts(text):
    """The bailout pattern: short (under 200 chars), single-sentence,
    excuse-shaped. Caught structurally, not by phrase-matching."""
    assert classify_prose(text) == ProseKind.BAILOUT


def test_classify_prose_substantive_short_multi_sentence():
    """A compact completion summary like
    'Done. Added helper, ran tests, all pass.' is 41 chars and 4
    sentences — short but multi-clause. Must classify as SUBSTANTIVE
    so the gate accepts the legitimate stop. This is the
    counter-test to the headline bailout case."""
    text = "Done. Added helper, ran tests, all pass."
    assert classify_prose(text) == ProseKind.SUBSTANTIVE


def test_classify_prose_substantive_long_single_sentence():
    """A single long-paragraph sentence (>= 200 chars) is substantive
    even without sentence-terminator count — length floor alone
    justifies it."""
    text = "x" * 220
    assert classify_prose(text) == ProseKind.SUBSTANTIVE


def test_classify_prose_substantive_long_multi_sentence():
    """The genuine answer pattern: multi-sentence, real content,
    explicit references."""
    text = (
        "I traced the issue to a null pointer in handler_init(). "
        "Cannot fix from here — the file lives in a system library "
        "outside the workspace. You will need to patch /usr/lib/foo.so "
        "per issue 123, or override LD_PRELOAD with a wrapper."
    )
    assert classify_prose(text) == ProseKind.SUBSTANTIVE


# ---------------------------------------------------------------------------
# Primitive 3: intent_is_action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,expected",
    [
        (TurnIntentKind.IMPLEMENT, True),
        (TurnIntentKind.DEBUG, True),
        (TurnIntentKind.OPERATE, True),
        (TurnIntentKind.UNKNOWN, True),  # bias toward action on ambiguity
        (TurnIntentKind.INSPECT, False),
        (TurnIntentKind.REVIEW, False),
        (TurnIntentKind.RESEARCH, False),
    ],
)
def test_intent_is_action(kind, expected):
    """Action intents = IMPLEMENT/DEBUG/OPERATE plus UNKNOWN (defensive
    bias). Read-only intents (INSPECT/REVIEW/RESEARCH) legitimately
    produce prose-only outcomes. UNKNOWN treated as action because the
    cost of a false-passive is silent task abandonment — the bug we're
    fixing."""
    assert intent_is_action(kind) is expected


# ---------------------------------------------------------------------------
# evaluate_termination — decision tree
# ---------------------------------------------------------------------------


def _ctx(
    *,
    user_text="Fix the bug",
    intent_kind=TurnIntentKind.DEBUG,
    clean_prose="",
    total_writes=0,
    had_recent_progress=False,
    continuation_nudged=False,
):
    """Build a TerminationContext with sensible defaults. Tests can
    override individual fields to exercise specific branches."""
    return TerminationContext(
        user_text=user_text,
        intent_kind=intent_kind,
        clean_prose=clean_prose,
        total_writes=total_writes,
        had_recent_progress=had_recent_progress,
        continuation_nudged=continuation_nudged,
    )


def test_evaluate_already_nudged_accepts():
    """Rule 1: bound the nudge depth at 1. Once we've nudged, we
    accept the next stop regardless of prose quality. Without this
    the loop could nudge forever on a model that won't recover."""
    v = evaluate_termination(_ctx(continuation_nudged=True, clean_prose=""))
    assert v.accept_stop is True
    assert v.reason == REASON_ALREADY_NUDGED


def test_evaluate_recent_progress_accepts():
    """Rule 2: writes happened recently; prose is a wrap-up. The
    quality of the prose doesn't matter — work was done."""
    v = evaluate_termination(_ctx(had_recent_progress=True, clean_prose="Done."))
    assert v.accept_stop is True
    assert v.reason == REASON_RECENT_PROGRESS


def test_evaluate_already_nudged_takes_precedence_over_progress():
    """Both rules 1 and 2 yield accept; rule 1 fires first because it
    bounds the nudge depth contract, regardless of progress. Pin the
    order so a future refactor doesn't accidentally swap them."""
    v = evaluate_termination(_ctx(
        had_recent_progress=True, continuation_nudged=True,
    ))
    assert v.accept_stop is True
    assert v.reason == REASON_ALREADY_NUDGED


def test_evaluate_insistent_zero_writes_nudges_regardless_of_prose():
    """Rule 3: the headline gate behaviour. INSISTENT user demand +
    zero writes = mandatory nudge, regardless of how substantive the
    prose looks. The user asked for completion; a prose summary is
    not completion."""
    long_substantive = (
        "I have analyzed the codebase thoroughly. The issue spans "
        "several modules and would require coordinated edits. I "
        "would recommend you start with the auth module first, then "
        "the routing layer, then finally the storage backend."
    )
    v = evaluate_termination(_ctx(
        user_text="Please continue until absolutely finished",
        intent_kind=TurnIntentKind.IMPLEMENT,
        clean_prose=long_substantive,
        total_writes=0,
    ))
    assert v.accept_stop is False
    assert v.reason == REASON_NUDGE_INSISTENT
    assert v.nudge_kind == NUDGE_INSISTENCE


def test_evaluate_headline_failure_case():
    """The exact failure pattern reported by the user: short
    bailout-shaped prose under a 'continue until absolutely finished'
    instruction. Must NOT accept stop. This is the regression-pin
    for Phase 3.6."""
    v = evaluate_termination(_ctx(
        user_text="Yes please, continue until absolutely finished",
        intent_kind=TurnIntentKind.DEBUG,
        clean_prose="I read the file but the middle was elided.",
        total_writes=0,
    ))
    assert v.accept_stop is False
    assert v.reason == REASON_NUDGE_INSISTENT


def test_evaluate_insistent_with_writes_still_under_recent_progress():
    """If insistent BUT writes happened, rule 2 (recent_progress)
    fires before rule 3 — accept the stop. The user asked for
    completion AND we delivered some, so prose-as-wrap-up is fine."""
    v = evaluate_termination(_ctx(
        user_text="continue until done",
        had_recent_progress=True, total_writes=2,
        clean_prose="Done — wrote the helper and the tests.",
    ))
    assert v.accept_stop is True
    assert v.reason == REASON_RECENT_PROGRESS


def test_evaluate_action_intent_zero_writes_empty_prose_nudges():
    """Rule 4 sub-case: action intent, no writes, empty prose ⇒
    no-progress nudge."""
    v = evaluate_termination(_ctx(
        intent_kind=TurnIntentKind.IMPLEMENT,
        clean_prose="", total_writes=0,
    ))
    assert v.accept_stop is False
    assert v.reason == REASON_NUDGE_EMPTY
    assert v.nudge_kind == NUDGE_NO_PROGRESS


def test_evaluate_action_intent_zero_writes_bailout_nudges():
    """Rule 4 sub-case: action intent, no writes, bailout-shaped
    prose ⇒ bailout nudge. Different framing than no-progress because
    the model DID say something — just something evasive."""
    v = evaluate_termination(_ctx(
        intent_kind=TurnIntentKind.IMPLEMENT,
        clean_prose="I would need more context to continue.",
        total_writes=0,
    ))
    assert v.accept_stop is False
    assert v.reason == REASON_NUDGE_BAILOUT
    assert v.nudge_kind == NUDGE_BAILOUT


def test_evaluate_action_intent_zero_writes_substantive_accepts():
    """Rule 4 happy path: action intent + zero writes is OK if the
    model produced a substantive answer (e.g., genuine 'I'm
    blocked because' explanation with specifics)."""
    v = evaluate_termination(_ctx(
        intent_kind=TurnIntentKind.IMPLEMENT,
        clean_prose=(
            "I traced the bug to a system library outside the workspace. "
            "Specifically, /usr/lib/foo.so calls handler_init() with a "
            "null pointer; we cannot patch that file from here. To fix: "
            "either rebuild libfoo from source, or use LD_PRELOAD with "
            "a wrapper. I've documented the trace in /workspace/issue.md."
        ),
        total_writes=0,
    ))
    assert v.accept_stop is True
    assert v.reason == REASON_SUBSTANTIVE_ACTIVE


def test_evaluate_passive_intent_substantive_accepts():
    """Rule 5: passive request + substantive prose = legitimate
    analysis-only outcome with no writes."""
    v = evaluate_termination(_ctx(
        user_text="Explain how the auth flow works",
        intent_kind=TurnIntentKind.INSPECT,
        clean_prose=(
            "The auth flow has three phases. First, the client sends "
            "a token to /auth/verify. Second, the middleware checks "
            "the signature against the rotating key. Third, the user "
            "scope is loaded from the database."
        ),
        total_writes=0,
    ))
    assert v.accept_stop is True
    assert v.reason == REASON_SUBSTANTIVE_PASSIVE


def test_evaluate_inspect_intent_no_passive_marker_substantive_accepts():
    """Rule 5: non-action intent with substantive prose + no passive
    marker on the user message still accepts. The intent kind alone
    is enough to permit prose-only outcomes."""
    v = evaluate_termination(_ctx(
        user_text="Look at the deps",  # no passive marker
        intent_kind=TurnIntentKind.INSPECT,
        clean_prose=(
            "The dependencies are minimal. Three runtime: requests, "
            "pydantic, structlog. Two dev: pytest, ruff. Pinned via "
            "pyproject.toml; no lockfile."
        ),
        total_writes=0,
    ))
    assert v.accept_stop is True
    assert v.reason == REASON_SUBSTANTIVE_NON_ACTION


def test_evaluate_inspect_intent_bailout_prose_still_nudges():
    """Even under non-action intent, a bailout-shaped one-liner is
    not a real answer. The rule isn't 'inspect = always accept' —
    it's 'inspect lets you skip writes IF you produce a real prose
    answer'."""
    v = evaluate_termination(_ctx(
        user_text="What does this do?",
        intent_kind=TurnIntentKind.INSPECT,
        clean_prose="I don't know.",
        total_writes=0,
    ))
    assert v.accept_stop is False
    assert v.reason == REASON_NUDGE_EMPTY  # under 20 chars


def test_evaluate_inspect_intent_short_bailout_nudges_with_bailout_kind():
    """Distinguish empty (nothing said) from bailout (something said
    but evasive). Both nudge but with different framings."""
    v = evaluate_termination(_ctx(
        user_text="What does this do?",
        intent_kind=TurnIntentKind.INSPECT,
        clean_prose="I would need to look at additional files.",
        total_writes=0,
    ))
    assert v.accept_stop is False
    assert v.reason == REASON_NUDGE_BAILOUT
    assert v.nudge_kind == NUDGE_BAILOUT


# ---------------------------------------------------------------------------
# TerminationVerdict.explain
# ---------------------------------------------------------------------------


def test_explain_known_accept_reason():
    """Known reasons get a curated human explanation; the curated
    text starts with 'Accepted' for accepted stops."""
    v = TerminationVerdict(
        accept_stop=True, reason=REASON_RECENT_PROGRESS,
    )
    msg = v.explain()
    assert msg.startswith("Accepted stop"), msg
    assert "wrap-up" in msg or "writes" in msg


def test_explain_known_nudge_reason():
    """Known nudge reasons get a curated human explanation starting
    with 'Nudged'."""
    v = TerminationVerdict(
        accept_stop=False,
        reason=REASON_NUDGE_INSISTENT,
        nudge_kind=NUDGE_INSISTENCE,
    )
    msg = v.explain()
    assert msg.startswith("Nudged"), msg
    assert "completion" in msg or "demanded" in msg


def test_explain_unknown_reason_falls_back_gracefully():
    """A future code path that mints a new reason tag without adding
    it to the explanation table must still produce a non-empty
    message — the trace shouldn't break on unknown tags."""
    v_accept = TerminationVerdict(accept_stop=True, reason="some_new_tag")
    assert "some_new_tag" in v_accept.explain()
    assert "Accepted" in v_accept.explain()

    v_nudge = TerminationVerdict(
        accept_stop=False, reason="another_new_tag", nudge_kind="x",
    )
    assert "another_new_tag" in v_nudge.explain()


# ---------------------------------------------------------------------------
# Defensive: empty-input handling
# ---------------------------------------------------------------------------


def test_classify_user_demand_empty_string():
    assert classify_user_demand("") == UserDemand.UNKNOWN


def test_classify_user_demand_whitespace_only():
    assert classify_user_demand("   \n\n") == UserDemand.UNKNOWN


def test_classify_prose_none_safely_handled():
    """Defensive: callers shouldn't pass None, but if a future bug
    routes None into the gate, we should classify as EMPTY rather
    than crash."""
    assert classify_prose(None) == ProseKind.EMPTY  # type: ignore[arg-type]


def test_evaluate_termination_returns_immutable_verdict():
    """TerminationVerdict is frozen — slot the dataclass invariant in
    so a future mutability change is caught here."""
    v = evaluate_termination(_ctx())
    with pytest.raises((AttributeError, Exception)):
        v.accept_stop = not v.accept_stop  # type: ignore[misc]
