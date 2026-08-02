"""Vocal acknowledgements — latency masking with variety.

A short, commitment-free line spoken the moment a turn starts
computing reads as "she heard me"; the actual reply then lands on a
listener who never experienced the gap. Industry-standard perceived-
latency trick, persona-respecting here because the line is synthesized
through the session's own voice/provider pipeline (never a canned
foreign voice).

Variety contract (Matt, 2026-06-13): long sessions must not feel
robotic — a shuffle-bag guarantees no line repeats until the whole
pool has been heard, AND the bag seeds a few SILENT slots so some
turns get no ack at all (silence is itself a natural variant). The
bag reshuffles on exhaustion, rejecting an immediate back-to-back
repeat across the boundary.

Lines are deliberately neutral: no promises ("sure!" before a turn
that might fail), no names/pronouns (OSS deployments configure their
own companions — persona-agnostic strings law).
"""

from __future__ import annotations

import random

# Spoken pool. Short enough that synthesis is near-instant and the
# real first sentence never queues behind a paragraph.
_LINES: tuple[str, ...] = (
    "Mm-hm.",
    "Okay.",
    "On it.",
    "One sec.",
    "Let me look.",
    "Hmm.",
    "Right.",
    "Checking.",
    "Give me a sec.",
    "Mm.",
    "Alright.",
    "Let me see.",
)

# Silent slots mixed into every bag — some turns simply don't get an
# ack, which keeps the habit from becoming a tic.
_SILENT_SLOTS = 4

# session_key -> remaining bag (popped from the end)
_bags: dict[str, list[str]] = {}
# session_key -> last spoken line (boundary repeat guard)
_last: dict[str, str] = {}

_MAX_SESSIONS = 256


def _new_bag(session_key: str) -> list[str]:
    bag = list(_LINES) + [""] * _SILENT_SLOTS
    random.shuffle(bag)
    # Never let the first draw of the new bag repeat the last spoken
    # line of the old one.
    last = _last.get(session_key)
    if last and bag[-1] == last:
        for i, line in enumerate(bag):
            if line != last:
                bag[-1], bag[i] = bag[i], bag[-1]
                break
    return bag


def next_ack(session_key: str) -> str:
    """Next ack line for this session — '' means stay silent this turn."""
    if len(_bags) > _MAX_SESSIONS:
        _bags.clear()
        _last.clear()
    bag = _bags.get(session_key)
    if not bag:
        bag = _new_bag(session_key)
        _bags[session_key] = bag
    line = bag.pop()
    if line:
        _last[session_key] = line
    return line


def reset(session_key: str) -> None:
    """Test hook / session teardown."""
    _bags.pop(session_key, None)
    _last.pop(session_key, None)
