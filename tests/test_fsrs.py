"""FSRS-5 scheduler tests — deterministic, no external reference needed.

Pins the hand-computable anchors (first-review intervals/difficulty from
the published default weights) plus the structural invariants the
algorithm must satisfy.
"""

from __future__ import annotations

import pytest

from augmentum.learning import fsrs
from augmentum.learning.fsrs import AGAIN, EASY, GOOD, HARD, CardState


def test_factor_decay_relationship():
    # FACTOR is defined so the scheduled interval equals the stability
    # when request_retention == 0.9.
    assert abs(fsrs.FACTOR - 19 / 81) < 1e-9
    assert fsrs._interval_from_stability(10.0, 0.9) == 10
    assert fsrs._interval_from_stability(3.173, 0.9) == 3


def test_first_review_intervals_match_default_weights():
    new = CardState()
    w = fsrs.DEFAULT_WEIGHTS
    assert fsrs.schedule(new, HARD).interval_days == round(w[1])   # ~1
    assert fsrs.schedule(new, GOOD).interval_days == round(w[2])   # ~3
    assert fsrs.schedule(new, EASY).interval_days == round(w[3])   # ~16
    again = fsrs.schedule(new, AGAIN)
    # S0(Again) = w[0] ≈ 0.4 rounds to 0 → clamped to the floor.
    assert again.interval_days == fsrs.MIN_INTERVAL_DAYS
    assert again.reps == 1
    # Failing a brand-new card is not a lapse.
    assert again.lapses == 0


def test_first_review_difficulty():
    new = CardState()
    good = fsrs.schedule(new, GOOD)
    # D0(3) = w4 - e^(w5*2) + 1 ≈ 5.28
    assert good.difficulty == pytest.approx(5.28, abs=0.05)
    assert 1.0 <= good.difficulty <= 10.0
    # Again seeds a harder card, Easy an easier one.
    assert fsrs.schedule(new, AGAIN).difficulty > good.difficulty
    assert fsrs.schedule(new, EASY).difficulty < good.difficulty


def test_recall_grows_stability():
    first = fsrs.schedule(CardState(), GOOD)
    card = CardState(difficulty=first.difficulty, stability=first.stability, reps=first.reps)
    second = fsrs.schedule(card, GOOD, elapsed_days=first.interval_days)
    assert second.stability > card.stability
    assert second.interval_days >= first.interval_days
    assert second.reps == 2
    assert second.lapses == 0


def test_lapse_resets_stability_and_counts():
    card = CardState(difficulty=5.0, stability=120.0, reps=8, lapses=0)
    res = fsrs.schedule(card, AGAIN, elapsed_days=120.0)
    assert res.stability < card.stability
    assert res.lapses == 1
    assert res.reps == 9
    assert res.interval_days >= fsrs.MIN_INTERVAL_DAYS


def test_grade_ordering_invariant():
    card = CardState(difficulty=5.0, stability=10.0, reps=3, lapses=0)
    ivals = fsrs.preview_intervals(card, elapsed_days=10.0)
    assert ivals[AGAIN] <= ivals[HARD] <= ivals[GOOD] <= ivals[EASY]
    # And on a brand-new card.
    new_ivals = fsrs.preview_intervals(CardState())
    assert new_ivals[AGAIN] <= new_ivals[HARD] <= new_ivals[GOOD] <= new_ivals[EASY]


def test_difficulty_stays_clamped_under_repetition():
    card = CardState(difficulty=9.9, stability=5.0, reps=4, lapses=2)
    for _ in range(15):
        res = fsrs.schedule(card, AGAIN, elapsed_days=1.0)
        assert 1.0 <= res.difficulty <= 10.0
        card = CardState(res.difficulty, res.stability, res.reps, res.lapses)
    card = CardState(difficulty=1.1, stability=50.0, reps=5, lapses=0)
    for _ in range(15):
        res = fsrs.schedule(card, EASY, elapsed_days=30.0)
        assert 1.0 <= res.difficulty <= 10.0
        card = CardState(res.difficulty, res.stability, res.reps, res.lapses)


def test_invalid_grade_rejected():
    with pytest.raises(ValueError):
        fsrs.schedule(CardState(), 0)
    with pytest.raises(ValueError):
        fsrs.schedule(CardState(), 5)


def test_mastery_buckets():
    assert fsrs.mastery_for(0, 0.0) == "new"
    assert fsrs.mastery_for(0, 999.0) == "new"          # reps wins
    assert fsrs.mastery_for(1, 5.0) == "learning"
    assert fsrs.mastery_for(3, 20.9) == "reviewing"
    assert fsrs.mastery_for(3, 21.0) == "mature"
    assert fsrs.mastery_for(10, 400.0) == "mature"
    assert fsrs.mastery_for(10, 400.0, lapses=3) == "leech"


def test_determinism():
    assert fsrs.schedule(CardState(), GOOD) == fsrs.schedule(CardState(), GOOD)
