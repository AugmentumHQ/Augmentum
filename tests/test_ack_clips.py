"""Ack-clip variety contract (latency MVP, 2026-06-13).

The latency mask must not become a tic in a long session: the shuffle-
bag guarantees no line repeats until the whole pool is heard, seeds
silent slots so some turns get nothing, and rejects an immediate
back-to-back repeat across a bag boundary.
"""

from __future__ import annotations

import random

from augmentum.voice import ack_clips


def setup_function():
    ack_clips.reset("s")
    random.seed(1234)


def test_no_repeat_within_a_bag():
    # One full bag = len(_LINES) spoken lines + _SILENT_SLOTS blanks.
    bag_size = len(ack_clips._LINES) + ack_clips._SILENT_SLOTS
    draws = [ack_clips.next_ack("s") for _ in range(bag_size)]
    spoken = [d for d in draws if d]
    # Every spoken line in a bag is distinct (coverage before repeat).
    assert len(spoken) == len(set(spoken))
    # The full line pool was covered exactly once.
    assert set(spoken) == set(ack_clips._LINES)


def test_silent_slots_present():
    bag_size = len(ack_clips._LINES) + ack_clips._SILENT_SLOTS
    draws = [ack_clips.next_ack("s") for _ in range(bag_size)]
    assert draws.count("") == ack_clips._SILENT_SLOTS


def test_no_back_to_back_repeat_across_bag_boundary():
    # Draw two full bags; no spoken line may immediately follow itself,
    # including across the reshuffle seam.
    bag_size = len(ack_clips._LINES) + ack_clips._SILENT_SLOTS
    draws = [ack_clips.next_ack("s") for _ in range(bag_size * 2)]
    spoken_seq = [d for d in draws if d]
    for a, b in zip(spoken_seq, spoken_seq[1:], strict=False):
        assert a != b, f"back-to-back repeat: {a!r}"


def test_sessions_are_independent():
    ack_clips.reset("a")
    ack_clips.reset("b")
    a = ack_clips.next_ack("a")
    b = ack_clips.next_ack("b")
    # Independent bags — one session draining doesn't affect another.
    assert isinstance(a, str) and isinstance(b, str)
    # 'a' has popped one; 'b' is untouched and still has a full bag.
    assert "b" in ack_clips._bags
    assert len(ack_clips._bags["b"]) == (
        len(ack_clips._LINES) + ack_clips._SILENT_SLOTS - 1
    )


def test_reset_clears_state():
    ack_clips.next_ack("s")
    ack_clips.reset("s")
    assert "s" not in ack_clips._bags
    assert "s" not in ack_clips._last


def test_all_lines_are_persona_neutral():
    # OSS deployments configure their own companions — no names/pronouns
    # in shipped strings, and no commitments that a failed turn betrays.
    banned = ("becca", " he ", " she ", " her ", " his ", "sure!", "yes!")
    for line in ack_clips._LINES:
        low = f" {line.lower()} "
        for b in banned:
            assert b not in low, f"{line!r} contains {b!r}"
