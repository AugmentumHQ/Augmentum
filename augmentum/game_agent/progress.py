"""Externally-defined progress score for one session.

Why this module exists
----------------------
The agent could already *pursue* goals, but nothing could *grade* a run.
The goal stack's ``metric`` is authored by the planner itself, so it
answers "did the model hit the target the model set?" — useful in-session
for the stall watchdog, worthless for deciding whether a prompt or rule
change made the agent better. An agent that grades its own homework
cannot be iterated on.

So the headline number here is deliberately **planner-independent**: it
is computed only from things the model cannot write — pixels it could not
fake (visual buckets), dialogue the game printed, and whether the buttons
it pressed actually changed the screen. Planner-authored signals (metric
goals) are carried alongside as :attr:`ProgressScore.goals_completed` and
excluded from :attr:`score`, so a run can never be made to look better by
the model declaring victory.

Beyond loading
--------------
"Reached a title screen" is not progress. Loading and attract-mode
animations churn pixels endlessly, so raw visual novelty alone would
score a game that never got past its boot sequence. :attr:`reached_play`
gates the score on having observed at least one *productive* screen — a
screen label outside {title, loading, unknown}. Until that flips, the
score is zero no matter how much the pixels moved.

Interpretation
--------------
:attr:`score` is **ordinal**, not physical. It exists to rank two runs of
the same game under different code, not to mean anything on its own. The
weights are a judgement call, documented in :data:`_WEIGHTS`; compare
runs of similar duration, or use :attr:`score_per_min`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Screen labels that do NOT count as being in the game. Kept lowercase to
# match the narrator's normalized ``screen`` probe (see
# Orchestrator._SCENE_LABELS, lowercased on the way into the blackboard).
NON_PRODUCTIVE_SCREENS = frozenset({"title", "loading", "unknown", ""})

# effect_score below this means the press visibly did nothing. Same
# constant the universal dead-nav reflex rule keys off, kept in sync by
# reference rather than re-derived.
DEAD_INPUT_THRESHOLD = 48

# Pixel-only fallback for reaching play, used when NOTHING labels the
# screen — no RAM probe, no scene narrator (no VLM wired, or it is the
# no-GPU tier). Without this the gate depends on a signal that does not
# exist in the hardest case the scorer is FOR: a genuinely unfamiliar
# game would score 0 forever no matter how much real progress happened.
#
# Both conditions are required because either alone lies:
# - Distinct visual scenes alone: a loading spinner or attract-mode demo
#   churns pixels endlessly without the player being anywhere.
# - Effective inputs alone: one button that works on the title screen
#   (the one that starts the game) would flip the gate immediately.
# Together they say "our presses are moving us through distinct places",
# which is what being in a game means.
#
# This fallback applies ONLY when nothing labelled the screen at all. If
# labels exist and every one of them is title/loading/unknown, that is
# positive evidence we never got in, and it outranks a pixel guess — an
# animated loading screen can produce plenty of distinct frames while
# presses happen to coincide with the animation.
#
# Thresholds are calibrated against the TinyQuest eval case, where blind
# exploration measured a 0.21 effective-input ratio (it bumps walls
# constantly, which is honest play) against 0.00 for an agent mashing a
# dead button on the title screen. The gap between playing and not
# playing is wide; 0.15 sits inside it without punishing exploration.
# The 8-scene floor is set above what a short cycling animation
# produces, since a spinner repeats the same frames and its distinct
# count stops growing.
MIN_VISUAL_SCENES_FOR_PLAY = 8
MIN_EFFECTIVE_INPUTS_FOR_PLAY = 3
MIN_EFFECTIVE_RATIO_FOR_PLAY = 0.15

# Ordinal weights per novelty dimension. Rationale, not physics:
# - dialog is the strongest probe-free evidence of *story* advancement:
#   a new printed line means the game acknowledged an interaction.
# - screen (productive labels only) is coarse but meaningful — entering a
#   battle or a menu is a state transition.
# - visual is the broadest and noisiest signal, so it is weighted lowest;
#   it is what keeps text-light games scoreable at all.
# - tile is RAM-backed and exact where available, but absent on most
#   games, so it must never dominate.
_WEIGHTS: dict[str, float] = {
    "dialog": 3.0,
    "screen": 2.0,
    "tile": 1.0,
    "visual": 0.5,
}


@dataclass
class ProgressScore:
    """Machine-checked verdict for one session.

    Every field is derived from the live log / blackboard. Nothing here
    is writable by the planner except :attr:`goals_completed`, which is
    excluded from :attr:`score` for exactly that reason.
    """

    # Did the agent ever get past boot/title/loading into actual play?
    reached_play: bool = False
    # True when reached_play was inferred from pixels + input effect
    # rather than a labelled screen — i.e. nothing told us what we were
    # looking at. Reported so a reader can tell a strong verdict from an
    # inferred one.
    reached_play_inferred: bool = False
    # Distinct-key counts per novelty dimension.
    novelty: dict[str, int] = field(default_factory=dict)
    # Productive screen labels observed (small closed vocabulary).
    productive_screens: list[str] = field(default_factory=list)
    # Input effectiveness: acked presses that visibly changed the screen.
    inputs_acked: int = 0
    inputs_effective: int = 0
    # Planner-authored. Reported, never scored.
    goals_completed: int = 0
    duration_ms: int = 0
    # Ordinal headline number. 0 until reached_play.
    score: float = 0.0
    verdict: str = "no_progress"

    @property
    def effective_input_ratio(self) -> float:
        """Fraction of acked presses that did something. 0.0 when none."""

        if self.inputs_acked <= 0:
            return 0.0
        return self.inputs_effective / self.inputs_acked

    @property
    def score_per_min(self) -> float:
        """Score normalized by wall-clock, for unequal-length runs."""

        minutes = self.duration_ms / 60000.0
        if minutes <= 0:
            return 0.0
        return self.score / minutes

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProgressScore:
        """Rebuild from :meth:`to_dict` output (e.g. a session_end trailer).

        Unknown keys are ignored so an older log stays readable after
        the scorecard gains fields — an eval harness must not choke on
        the baseline it is comparing against.
        """

        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effective_input_ratio"] = round(self.effective_input_ratio, 4)
        d["score_per_min"] = round(self.score_per_min, 3)
        d["score"] = round(self.score, 3)
        return d


def _verdict(*, reached_play: bool, score: float, inputs_acked: int) -> str:
    """Coarse human-readable bucket. Ordering matters: check worst first."""

    if inputs_acked <= 0:
        return "no_inputs"
    if not reached_play:
        return "loaded_only"
    if score <= 0:
        return "no_progress"
    return "progressing"


def score_session(
    *,
    novelty: dict[str, int],
    screens_seen: list[Any],
    inputs_acked: int,
    inputs_effective: int,
    duration_ms: int,
    goals_completed: int = 0,
) -> ProgressScore:
    """Grade one session from log-derived primitives.

    Use when:
    - Closing a session, or re-scoring a finished log offline. Takes
      plain values (not a :class:`WorldState`) so an eval harness can
      score a replayed NDJSON log without booting an orchestrator.

    Expects:
    - ``novelty`` is :meth:`WorldState.novelty_snapshot` output.
    - ``screens_seen`` is :meth:`WorldState.novelty_keys` for ``screen``.

    Returns:
    - A :class:`ProgressScore`. Never raises on odd input — a bad run
      must still produce a comparable (low) number rather than an error.
    """

    # ``None`` is filtered BEFORE stringifying: str(None) is "none", which
    # would sail past the non-productive filter and falsely flip
    # reached_play on a game whose screen probe is absent — the exact
    # false-progress reading this gate exists to stop.
    productive = sorted(
        str(s).strip().lower()
        for s in screens_seen
        if s is not None and str(s).strip().lower() not in NON_PRODUCTIVE_SCREENS
    )
    # Was ANYTHING labelling the screen (RAM probe or scene narrator)?
    # If so, its verdict stands even when it says "still on the title" —
    # positive evidence beats inference. The pixel path fills a vacuum,
    # it does not overrule a witness.
    has_labels = any(
        s is not None and str(s).strip() for s in screens_seen
    )
    n_effective = max(0, min(inputs_effective, max(0, inputs_acked)))
    ratio = (n_effective / inputs_acked) if inputs_acked > 0 else 0.0
    reached_play_by_pixels = (
        not has_labels
        and int(novelty.get("visual", 0) or 0) >= MIN_VISUAL_SCENES_FOR_PLAY
        and n_effective >= MIN_EFFECTIVE_INPUTS_FOR_PLAY
        and ratio >= MIN_EFFECTIVE_RATIO_FOR_PLAY
    )
    # A labelled productive screen is the strong signal; the pixel-only
    # path is the fallback for games where nothing labels the screen.
    reached_play = bool(productive) or reached_play_by_pixels

    score = 0.0
    if reached_play:
        for dim, weight in _WEIGHTS.items():
            count = int(novelty.get(dim, 0) or 0)
            if dim == "screen":
                # Only productive screens earn credit — a title/loading
                # flip-flop must not read as exploration.
                count = len(productive)
            score += weight * max(0, count)

    return ProgressScore(
        reached_play=reached_play,
        reached_play_inferred=bool(not productive and reached_play_by_pixels),
        novelty=dict(novelty),
        productive_screens=productive,
        inputs_acked=max(0, inputs_acked),
        inputs_effective=max(0, min(inputs_effective, inputs_acked)),
        goals_completed=max(0, goals_completed),
        duration_ms=max(0, duration_ms),
        score=score,
        verdict=_verdict(
            reached_play=reached_play, score=score, inputs_acked=inputs_acked
        ),
    )


def score_from_world(
    world: Any, *, inputs_acked: int, inputs_effective: int, duration_ms: int
) -> ProgressScore:
    """Convenience wrapper over a live :class:`~augmentum.game_agent.world.WorldState`."""

    return score_session(
        novelty=world.novelty_snapshot(),
        screens_seen=world.novelty_keys("screen"),
        inputs_acked=inputs_acked,
        inputs_effective=inputs_effective,
        duration_ms=duration_ms,
        goals_completed=world.goals_completed(),
    )


__all__ = [
    "DEAD_INPUT_THRESHOLD",
    "NON_PRODUCTIVE_SCREENS",
    "ProgressScore",
    "score_from_world",
    "score_session",
]
