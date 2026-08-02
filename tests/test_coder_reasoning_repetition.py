"""Unit tests for the reasoning-stream repetition meter.

Phase 1a instrument (augmentum/coder/reasoning_repetition.py): measures
reasoning-token volume + repetition signals per turn so we can baseline
how often models loop INSIDE their reasoning stream (invisible to every
tool-call-level breaker) before building the interrupt-and-salvage detector.
"""
from __future__ import annotations

from augmentum.coder.reasoning_repetition import ReasoningRepetitionMeter


def test_empty_reasoning_returns_zero():
    m = ReasoningRepetitionMeter()
    assert m.summary() == {"reasoning_tokens": 0}


def test_healthy_reasoning_not_flagged():
    m = ReasoningRepetitionMeter()
    m.feed(
        "First I will read the config module to find the port setting. "
        "Then I will check how the server binds it and write the fix. "
        "Finally I will run the test to confirm the change works."
    )
    s = m.summary()
    assert s["reasoning_tokens"] > 20
    assert s["looped_suspected"] is False
    assert s["short_ngram_top_repeat"] < 5


def test_tight_token_cycle_flagged():
    m = ReasoningRepetitionMeter()
    # A degenerate short cycle repeated well past the flag threshold.
    m.feed("let me check the file again " * 12)
    s = m.summary()
    assert s["looped_suspected"] is True
    assert s["short_ngram_top_repeat"] >= 5


def test_semantic_span_loop_flagged():
    m = ReasoningRepetitionMeter()
    # Same reasoning step restated verbatim — long-window (sentence) loop,
    # with enough filler between that no tight token n-gram dominates.
    step = "I should verify the dev server is running before I screenshot it."
    fillers = [
        "The port might be eight thousand or three thousand somewhere.",
        "Perhaps chromium never launched inside this workspace at all.",
        "It could also be a playwright install that silently failed once.",
    ]
    parts = []
    for f in fillers * 2:
        parts.append(step)
        parts.append(f)
    m.feed(" ".join(parts))
    s = m.summary()
    assert s["long_span_top_repeat"] >= 3
    assert s["looped_suspected"] is True


def test_feed_is_bounded():
    m = ReasoningRepetitionMeter(max_chars=100)
    m.feed("x" * 60)
    m.feed("y" * 60)  # overruns the cap
    s = m.summary()
    assert s["reasoning_truncated"] is True
    assert s["reasoning_chars"] == 100


def test_streaming_deltas_accumulate():
    # Fed as many small deltas (as the ledger does per coalesced chunk),
    # the meter sees the same trace as one blob would.
    whole = "check the file again " * 10
    m_stream = ReasoningRepetitionMeter()
    for tok in whole.split(" "):
        m_stream.feed(tok + " ")
    m_blob = ReasoningRepetitionMeter()
    m_blob.feed(whole)
    assert (
        m_stream.summary()["short_ngram_top_repeat"]
        == m_blob.summary()["short_ngram_top_repeat"]
    )
