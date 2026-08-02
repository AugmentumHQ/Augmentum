"""Calibrated voice — express a memory's EARNED tier as epistemic register.

Earned Understanding (P1, the Calibrated Voice): confidence is spoken as
*tone*, not asserted as fact. A CORE belief is bedrock — she states it
plainly. ACTIVE is settled — she speaks it normally. PROVISIONAL is an
unproven impression — she hedges and offers to confirm rather than asserting
it. The recall surfaces prefix each fact with a short register cue so the
model calibrates HOW confidently it speaks each one, and never states an
unearned impression as a confident fact.

This is the inverse of the failure mode in feedback_memory_echo_chamber: not
"recite everything," but "speak each thing with exactly the confidence it has
earned." See docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
"""

from __future__ import annotations

from typing import Any

# Register cue per tier — a terse private hint the model reads as a confidence
# signal, NOT text to recite. The composer header explains the brackets.
_REGISTER: dict[str, str] = {
    "core": "certain",
    "active": "fairly sure",
    "provisional": "unconfirmed — hedge, ask to confirm",
    "archive": "old, may have faded",
}

# Honest-gap note for the memory_recall TOOL path (the model explicitly reached
# for memory and found nothing). Gated to the tool — never the ambient lane —
# so "memory checked, came up empty" is distinguishable from a confident guess.
HONEST_EMPTY_NOTE = (
    "No stored memories match this. Tell the user you don't have anything "
    "saved about it rather than guessing or inventing an answer."
)


def _tier_of(row: Any) -> str:
    """Best-effort tier string from a Memory object or a dict row."""
    t = row.get("tier", "") if isinstance(row, dict) else getattr(row, "tier", "")
    return (t if isinstance(t, str) else getattr(t, "value", "")) or ""


def _content_of(row: Any) -> str:
    if isinstance(row, dict):
        text = row.get("content", "") or ""
    else:
        text = getattr(row, "content", None) or ""
    return (text or "").strip().replace("\n", " ")


def register_label(tier: Any) -> str:
    """Map a tier (str or MemoryTier) to its spoken-confidence cue.

    Unknown/missing tiers default to ``active`` ("fairly sure") — never to
    the most-confident register, honoring the round-DOWN-on-ambiguity rule
    (over-hedging is recoverable; over-claiming breaks trust)."""
    t = tier if isinstance(tier, str) else getattr(tier, "value", "")
    return _REGISTER.get((t or "").lower(), _REGISTER["active"])


def calibrated_bullets(rows: Any, *, limit: int = 6, max_chars: int = 240) -> str:
    """Recalled memories → bullets, each prefixed with its register cue.

    ``- [certain] User's name is Matt`` vs ``- [unconfirmed — hedge, ask to
    confirm] User mentioned liking jazz``. The model speaks each at the
    indicated confidence; the bracket is a cue, not text to read aloud.
    """
    lines: list[str] = []
    for row in rows or []:
        text = _content_of(row)
        if not text:
            continue
        lines.append(f"- [{register_label(_tier_of(row))}] {text[:max_chars]}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)
