"""Reasoning-stream repetition meter — the coder loop's reasoning-token instrument.

Phase 1a of the adaptive-supervision design
(``docs/superpowers/specs/2026-08-01-coder-adaptive-supervision-design.md``).

Every existing coder guard (``augmentum/loops/breakers.py``) watches
tool-call / iteration granularity. A model that burns wall-clock *inside its
reasoning stream* — restating the same step for a minute, or degenerating into
a tight token cycle — emits ZERO tool calls and is invisible to all of them.
Matt observed exactly this in Qwen3.6 and Ornith runs.

Before building the interrupt-and-salvage detector (Phase 1b) we need to
*measure* how often it happens and how bad it gets. This meter is that
instrument: it ingests the coalesced reasoning text of a whole turn and, at
``summary()``, reports the reasoning-token volume plus two repetition signals —

  * **short window** (token n-grams, ``short_n``): catches tight degenerate
    cycles (``the the the``, a repeated short clause).
  * **long window** (sentence/line spans): catches the *semantic* loop — the
    same reasoning step restated verbatim, recurring every so often.

It is a pure, no-I/O accumulator (mirrors ``turn_progress.py`` discipline).
``looped_suspected`` is a generous boolean for telemetry triage only; the real
thresholds get tuned in Phase 1b once the instrument shows the baseline. The
n-gram machinery here is deliberately the same shape the 1b detector will run
live against ``ThinkingStreamBuffer`` — instrument first, promote to detector
second.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Split reasoning into sentence/line spans for the long-window pass. Newlines
# and sentence-final punctuation both terminate a span; empty spans dropped.
_SPAN_SPLIT = re.compile(r"[.!?\n]+")


@dataclass
class ReasoningRepetitionMeter:
    """Per-turn reasoning-repetition accumulator.

    ``feed()`` is called with each coalesced reasoning delta (the ``thinking``
    text of a ``reasoning_delta`` chunk). ``summary()`` folds the whole turn's
    reasoning once at finish — cheap enough to run per-run, never per-delta.
    """

    # Bound memory against a pathological looping run (luna hit 90 compactions).
    max_chars: int = 2_000_000
    short_n: int = 8   # token n-gram width for tight loops
    # Long window counts recurrence of any single sentence/line span (n=1):
    # a semantic loop is the SAME reasoning step restated verbatim, recurring
    # non-consecutively — not three identical spans in a row.
    long_n: int = 1
    # Generous triage thresholds for the ``looped_suspected`` flag only.
    short_repeat_flag: int = 5
    long_repeat_flag: int = 3

    _parts: list[str] = field(default_factory=list)
    _chars: int = 0
    _truncated: bool = False

    def feed(self, text: str) -> None:
        if not text or self._truncated:
            return
        room = self.max_chars - self._chars
        if room <= 0:
            self._truncated = True
            return
        if len(text) > room:
            text = text[:room]
            self._truncated = True
        self._parts.append(text)
        self._chars += len(text)

    def _top_repeat(self, units: list[str], n: int) -> tuple[int, float]:
        """Return (max repeat count of any n-gram, repeated-fraction) over
        ``units`` sliced into contiguous n-grams. A count of ``k`` means the
        same n-gram appeared ``k`` times; ``1`` means everything was distinct."""
        total = len(units) - n + 1
        if total <= 0:
            return 0, 0.0
        counts: dict[int, int] = {}
        for i in range(total):
            # Hash the n-gram rather than storing the tuple — keeps memory flat
            # on a multi-MB trace while still counting exact repeats.
            h = hash(tuple(units[i : i + n]))
            counts[h] = counts.get(h, 0) + 1
        top = max(counts.values())
        repeated = sum(c for c in counts.values() if c > 1)
        return top, repeated / total

    def summary(self) -> dict:
        """Fold the turn's reasoning into a telemetry dict. Returns
        ``{"reasoning_tokens": 0}`` when the model emitted no reasoning."""
        text = "".join(self._parts)
        tokens = text.split()
        if not tokens:
            return {"reasoning_tokens": 0}

        short_top, short_ratio = self._top_repeat(tokens, self.short_n)

        spans = [s.strip().lower() for s in _SPAN_SPLIT.split(text)]
        spans = [s for s in spans if s]
        long_top, _ = self._top_repeat(spans, self.long_n)

        looped = short_top >= self.short_repeat_flag or long_top >= self.long_repeat_flag

        return {
            "reasoning_tokens": len(tokens),
            "reasoning_chars": self._chars,
            "reasoning_truncated": self._truncated,
            "short_ngram_top_repeat": short_top,
            "short_ngram_repeat_ratio": round(short_ratio, 4),
            "long_span_top_repeat": long_top,
            "looped_suspected": looped,
        }


__all__ = ["ReasoningRepetitionMeter"]
