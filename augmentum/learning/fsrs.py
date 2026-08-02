"""FSRS-5 spaced-repetition scheduler — deterministic, no dependencies.

Hand-rolled port of the canonical Free Spaced Repetition Scheduler (the
"FSRS-5" parameterisation: 19 weights, ``DECAY = -0.5``). Used by the
language-learning SRS loop to schedule vocabulary cards. Pure functions
over a small immutable state — no I/O, no globals — so it's trivially
testable against the reference ``fsrs-rs`` / ``py-fsrs`` output.

We use only the *long-term* scheduler (no Anki-style sub-day learning
steps): a relapse simply resets stability and schedules ~1 day out.
Same-day / short-term review handling is intentionally omitted — vocab
review in Augmentum is a once-or-twice-a-day flow where the long-term
path is an excellent approximation. Per-user weight optimisation is a
future phase; everyone gets the published default parameters for now.

Reference: https://github.com/open-spaced-repetition/fsrs4anki/wiki
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# FSRS-5 default weights (the published "default parameters" from the
# open-spaced-repetition project).
DEFAULT_WEIGHTS: tuple[float, ...] = (
    0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604,
    0.0046, 1.54575, 0.1192, 1.01925, 1.9395, 0.11, 0.29605,
    2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
)

# Forgetting curve: R(t, S) = (1 + FACTOR * t / S) ** DECAY.
# FACTOR is fixed so that, with the default request retention of 0.9,
# the scheduled interval comes out numerically equal to the stability.
DECAY: float = -0.5
FACTOR: float = 0.9 ** (1.0 / DECAY) - 1.0  # = 19/81 ≈ 0.234568

# Target recall probability at the scheduled review time.
DEFAULT_REQUEST_RETENTION: float = 0.9

# Interval bounds, in whole days.
MIN_INTERVAL_DAYS: int = 1
MAX_INTERVAL_DAYS: int = 365 * 10

# Difficulty is clamped to [1, 10].
_D_MIN, _D_MAX = 1.0, 10.0

# Stability never drops below this (avoids div-by-zero / pathological maths).
_S_FLOOR = 0.1

# Grades.
AGAIN, HARD, GOOD, EASY = 1, 2, 3, 4
GRADES: tuple[int, ...] = (AGAIN, HARD, GOOD, EASY)


@dataclass(frozen=True)
class CardState:
    """The persisted FSRS state of a single card.

    A *new* card (never reviewed) has ``reps == 0``; its
    stability/difficulty are ignored on the first review — the initial
    formulas seed them from the grade.
    """

    difficulty: float = 5.0
    stability: float = 0.0
    reps: int = 0
    lapses: int = 0


@dataclass(frozen=True)
class ScheduleResult:
    difficulty: float
    stability: float
    interval_days: int
    reps: int
    lapses: int


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _retrievability(elapsed_days: float, stability: float) -> float:
    if stability <= 0:
        return 0.0
    return (1.0 + FACTOR * max(0.0, elapsed_days) / stability) ** DECAY


def _init_difficulty(w: tuple[float, ...], grade: int) -> float:
    # D0(g) = w4 - e^(w5 * (g - 1)) + 1
    return _clamp(w[4] - math.exp(w[5] * (grade - 1)) + 1.0, _D_MIN, _D_MAX)


def _init_stability(w: tuple[float, ...], grade: int) -> float:
    # S0(g) = w[g - 1]
    return max(_S_FLOOR, w[grade - 1])


def _next_difficulty(w: tuple[float, ...], d: float, grade: int) -> float:
    delta = -w[6] * (grade - GOOD)            # 0 at Good
    d_prime = d + delta * (10.0 - d) / 9.0    # linear damping
    # Mean-reversion toward the Easy-init difficulty.
    d_next = w[7] * _init_difficulty(w, EASY) + (1.0 - w[7]) * d_prime
    return _clamp(d_next, _D_MIN, _D_MAX)


def _stability_on_recall(
    w: tuple[float, ...], d: float, s: float, r: float, grade: int
) -> float:
    hard_penalty = w[15] if grade == HARD else 1.0
    easy_bonus = w[16] if grade == EASY else 1.0
    growth = (
        math.exp(w[8])
        * (11.0 - d)
        * (s ** -w[9])
        * (math.exp(w[10] * (1.0 - r)) - 1.0)
        * hard_penalty
        * easy_bonus
    )
    return max(_S_FLOOR, s * (1.0 + growth))


def _stability_on_lapse(
    w: tuple[float, ...], d: float, s: float, r: float
) -> float:
    s_lapse = (
        w[11]
        * (d ** -w[12])
        * (((s + 1.0) ** w[13]) - 1.0)
        * math.exp(w[14] * (1.0 - r))
    )
    # Post-lapse stability never exceeds the pre-lapse value.
    return max(_S_FLOOR, min(s_lapse, s))


def _interval_from_stability(stability: float, request_retention: float) -> int:
    # Solve  request_retention = (1 + FACTOR * I / S) ** DECAY  for I:
    #   I = S / FACTOR * (request_retention ** (1 / DECAY) - 1)
    raw = stability / FACTOR * (request_retention ** (1.0 / DECAY) - 1.0)
    days = int(round(raw))
    return max(MIN_INTERVAL_DAYS, min(MAX_INTERVAL_DAYS, days))


def schedule(
    card: CardState,
    grade: int,
    *,
    elapsed_days: float = 0.0,
    request_retention: float = DEFAULT_REQUEST_RETENTION,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> ScheduleResult:
    """Advance a card by one review.

    ``elapsed_days`` is the real time since the previous review; it's
    ignored for a card's first review (``card.reps == 0``). Returns the
    new difficulty/stability, the next interval in whole days, and the
    updated rep/lapse counters.
    """
    if grade not in GRADES:
        raise ValueError(f"grade must be 1..4, got {grade!r}")
    w = weights

    if card.reps == 0:
        # First-ever review — seed from the initial formulas. Failing a
        # brand-new card is not a "lapse" (you never knew it).
        d = _init_difficulty(w, grade)
        s = _init_stability(w, grade)
        lapses = card.lapses
    else:
        r = _retrievability(elapsed_days, card.stability)
        d = _next_difficulty(w, card.difficulty, grade)
        if grade == AGAIN:
            s = _stability_on_lapse(w, d, card.stability, r)
            lapses = card.lapses + 1
        else:
            s = _stability_on_recall(w, d, card.stability, r, grade)
            lapses = card.lapses

    return ScheduleResult(
        difficulty=d,
        stability=s,
        interval_days=_interval_from_stability(s, request_retention),
        reps=card.reps + 1,
        lapses=lapses,
    )


def preview_intervals(
    card: CardState,
    *,
    elapsed_days: float = 0.0,
    request_retention: float = DEFAULT_REQUEST_RETENTION,
    weights: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> dict[int, int]:
    """Interval (whole days) the card would get for each of the four
    grades. Used to label the Again/Hard/Good/Easy buttons in the review
    UI before the user commits."""
    return {
        g: schedule(
            card,
            g,
            elapsed_days=elapsed_days,
            request_retention=request_retention,
            weights=weights,
        ).interval_days
        for g in GRADES
    }


# Mastery buckets ? coarse, derived from stability, review count, and
# lapse count. Used for UI badges, game readiness, and weak-word routing.
_MATURE_STABILITY_DAYS: float = 21.0
_REVIEWING_REPS: int = 3
_LEECH_LAPSES: int = 3


def mastery_for(reps: int, stability: float, lapses: int = 0) -> str:
    if reps <= 0:
        return "new"
    if lapses >= _LEECH_LAPSES:
        return "leech"
    if stability >= _MATURE_STABILITY_DAYS:
        return "mature"
    if reps >= _REVIEWING_REPS:
        return "reviewing"
    return "learning"
