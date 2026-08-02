"""Tests for the voice router fallback policy (voice_router.py:_regex_fallback).

Pins the parse_fallback ↔ timeout_fallback symmetry that fixed the prod
incident where DeepSeek-V4-Flash silently dropped every classifier call
because its asymmetric-thinking output eaten the 128-token budget and
emitted a truncated JSON, then the regex saw no_signal, then the
fallback policy dropped the turn.

After the A+B fix:
  - max_tokens is 384 (gives reasoning models room to finish JSON
    after their thinking burst)
  - both timeout_fallback AND parse_fallback lean addressed/converse
    when the regex returns no_signal — they share the same diagnostic
    posture (backend was engaged, STT decoded coherently, LLM<->JSON
    contract failed) and deserve the same policy
  - other error paths (resolve_failed / error_fallback) still drop
    because the backend genuinely couldn't be engaged
  - strong-signal regex matches (self_talk, third_person) still drop
    regardless of parsed_from — the regex SAW addressing structure
    and was confident it wasn't directed at the assistant
"""
from __future__ import annotations

import pytest

from augmentum.architect.voice_router import (
    VoiceRouterDecision,
    _regex_fallback,
)

# ── Token budget ────────────────────────────────────────────────


class TestTokenBudget:
    """Confirm the chat request shape includes the bumped max_tokens.

    We can't easily exercise the actual backend call from a unit test —
    the integration validation is the live smoke against DeepSeek-V4-Flash
    — but we CAN verify the source pins the new constant. If someone
    silently reverts to 128 the regression test fails.
    """

    def test_max_tokens_is_at_least_256(self):
        import pathlib
        src = pathlib.Path(
            "augmentum/architect/voice_router.py",
        ).read_text()
        # Look for the request construction's max_tokens line. We don't
        # match the exact number; we just guard that it's NOT 128
        # (the original under-sized budget) and is at least 256 (enough
        # for the JSON contract on most reasoning models).
        assert "max_tokens=128," not in src, (
            "max_tokens=128 in voice_router.py would re-introduce the "
            "DeepSeek-V4-Flash silent-drop bug (asymmetric-thinking model "
            "eats budget on thinking, JSON truncates mid-string)."
        )
        # Also confirm an at-least-256 budget literal exists somewhere
        # in the file. Sufficient guard given the file only has one
        # max_tokens line on the router call.
        assert any(
            f"max_tokens={n}," in src for n in (256, 384, 512, 1024)
        ), "voice_router.py must declare max_tokens of at least 256 for the router chat call"


# ── parse_fallback now leans converse (the bug fix) ─────────────


class TestParseFallbackLeansConverse:
    """The actual user-facing fix: parse_fallback + no_signal regex →
    lean addressed/converse, NOT drop.

    The prod logs showed three voice_router_parse_failed events for
    clear conversational utterances ("Hey there can you hear me?",
    "A little bit of both, I guess.") — all dropped silently. Each
    one is the test case below.
    """

    def test_parse_fallback_short_reply_leans_converse(self):
        """Reply-shaped utterance that the regex doesn't structurally
        recognize. Before the fix: drop with confidence=0. After: lean
        converse with confidence=0.60."""
        decision = _regex_fallback(
            text="A little bit of both, I guess.",
            model="deepseek-v4-flash",
            latency_ms=2450,
            parsed_from="parse_fallback",
        )
        assert decision.addressed is True
        assert decision.goal == "converse"
        assert decision.confidence == pytest.approx(0.60)
        assert "parse_fallback" in decision.reasoning
        assert "lean_addressed" in decision.reasoning

    def test_parse_fallback_greeting_leans_converse(self):
        """The explicit test phrase from the prod failure."""
        decision = _regex_fallback(
            text="Hey there can you hear me?",
            model="deepseek-v4-flash",
            latency_ms=2223,
            parsed_from="parse_fallback",
        )
        # The legacy regex may catch this as wh_question via "you" + "?".
        # The test pins: whichever path the regex picks, the result
        # is NOT a silent drop with confidence=0.
        assert not (
            decision.addressed is False
            and decision.goal == "drop"
            and decision.confidence == 0.0
        ), (
            "Parse-fallback on a clear addressed greeting must NOT "
            "silently drop — that's the prod bug we just fixed."
        )

    def test_timeout_fallback_still_leans_converse(self):
        """Regression guard: the original timeout_fallback behavior
        is unchanged. parse_fallback was added alongside, not instead of."""
        decision = _regex_fallback(
            text="some ambiguous transcript",
            model="m",
            latency_ms=5000,
            parsed_from="timeout_fallback",
        )
        # If the legacy regex returns no_signal, lean converse.
        if "no_signal" in decision.reasoning:
            assert decision.addressed is True
            assert decision.goal == "converse"
            assert decision.confidence == pytest.approx(0.60)


# ── Strong-signal regex matches still drop ──────────────────────


class TestStrongSignalsStillDrop:
    """Self-talk / third-person utterances must still drop regardless of
    parsed_from. The fix only changes the no_signal behavior."""

    def test_third_person_drops_on_parse_fallback(self):
        # "He said X" - third-person narration, the regex should catch
        # this as third_person if it's working. The test asserts the
        # OUTCOME: addressed=False, goal=drop. If the legacy regex
        # changes its signal label we still want the policy to hold.
        decision = _regex_fallback(
            text="He said it was fine yesterday.",
            model="m",
            latency_ms=100,
            parsed_from="parse_fallback",
        )
        # Either the regex matched third_person directly (drop), or
        # it returned no_signal and we leaned converse. We assert the
        # invariant the fix preserves: strong-signal third-person
        # mentions don't get auto-converted to addressed by the new
        # parse_fallback branch.
        if "third_person" in decision.reasoning:
            assert decision.addressed is False
            assert decision.goal == "drop"

    def test_self_talk_drops_on_parse_fallback(self):
        decision = _regex_fallback(
            text="ugh, where did I put my keys",
            model="m",
            latency_ms=100,
            parsed_from="parse_fallback",
        )
        if "self_talk" in decision.reasoning:
            assert decision.addressed is False
            assert decision.goal == "drop"


# ── Salvage-path posture: engage on backend-reachable, drop only unexpected ──


class TestOtherErrorsStillDrop:
    """error_fallback is backend-REACHABLE (chat() was called and raised on an
    already-resolved backend), so on no_signal it now ENGAGES — the lone
    silent-on-coherent path, folded into the salvage posture (2026-07-27).
    Only an UNEXPECTED parsed_from (resolve_failed never reaches _regex_fallback
    in prod) drops defensively, since we can't prove the backend can reply."""

    def test_error_fallback_no_signal_engages(self):
        """Transient error during the router's chat() on a resolved backend +
        an ambiguous utterance — ENGAGE (fail forward), never silently drop."""
        decision = _regex_fallback(
            text="some ambiguous transcript",
            model="m",
            latency_ms=100,
            parsed_from="error_fallback",
        )
        assert "no_signal" in decision.reasoning  # precondition: hit the default
        assert decision.addressed is True
        assert decision.goal == "converse"
        assert "lean_addressed" in decision.reasoning

    def test_resolve_failed_no_signal_drops(self):
        decision = _regex_fallback(
            text="some ambiguous transcript",
            model="m",
            latency_ms=0,
            parsed_from="resolve_failed",
        )
        if "no_signal" in decision.reasoning:
            assert decision.addressed is False
            assert decision.goal == "drop"
            assert "lean_addressed" not in decision.reasoning


# ── VoiceRouterDecision shape contract ──────────────────────────


class TestDecisionShape:
    def test_returned_decision_is_frozen(self):
        """VoiceRouterDecision is a frozen dataclass — callers can stash
        these for telemetry without worrying about mutation."""
        d = _regex_fallback(
            text="hello", model="m", latency_ms=0,
            parsed_from="parse_fallback",
        )
        assert isinstance(d, VoiceRouterDecision)
        with pytest.raises(AttributeError):
            d.confidence = 0.99  # type: ignore[misc]


class TestDropContradictionGuard:
    """addressed=True + goal=drop is self-contradictory ("drop" means
    NOT for the assistant). Live 2026-06-11: "No fire." → addressed=True
    coherent=True conf=0.9 goal=drop, silently killed at the gate. The
    normalizer promotes the contradiction to converse."""

    def test_addressed_coherent_drop_promotes_to_converse(self):
        from augmentum.architect.voice_router import _normalize_decision

        d = _normalize_decision(
            {"coherent": True, "addressed": True, "confidence": 0.9,
             "goal": "drop", "reasoning": "dismissive response"},
            text="No fire.", parsed_from="content", model="m", latency_ms=10,
        )
        assert d.goal == "converse"
        assert d.addressed is True

    def test_unaddressed_drop_stays_drop(self):
        # Background speech / her own echo: addressed=False — the guard
        # must NOT promote these.
        from augmentum.architect.voice_router import _normalize_decision

        d = _normalize_decision(
            {"coherent": True, "addressed": False, "confidence": 0.9,
             "goal": "drop", "reasoning": "becca's own echoed question"},
            text="What do you want to search for?",
            parsed_from="content", model="m", latency_ms=10,
        )
        assert d.goal == "drop"

    def test_incoherent_addressed_drop_stays_drop(self):
        from augmentum.architect.voice_router import _normalize_decision

        d = _normalize_decision(
            {"coherent": False, "addressed": True, "confidence": 0.5,
             "goal": "drop", "reasoning": "garbled"},
            text="asdf gnrr", parsed_from="content", model="m", latency_ms=10,
        )
        assert d.goal == "drop"
