"""Translate a playtest scorecard into actionable defects for the dev agent.

This is the connective tissue that makes the game foundry a *closed loop*
rather than a one-shot generator: the game_agent plays a generated build,
:mod:`augmentum.game_agent.progress` scores it with signals the generator
**cannot fake** (pixels that moved, presses that changed the screen, screens
actually reached), and this module turns that scorecard into a structured
brief the developing agent can act on for the next generation pass.

Why derive defects from the *unfakeable* score
----------------------------------------------
The whole point of ``progress.py`` is that its numbers are externally
defined — the planner can't inflate them. So the defects we derive here
inherit that honesty: "the game never reached a playable screen" is a fact
about pixels, not a model's opinion. That is what lets pass N+1 be judged
against pass N and the score legitimately climb.

The mapping is intentionally conservative: each rule keys off one clear
signal, orders worst-first (a boot failure makes input/goal analysis moot),
and every :class:`Defect` carries the raw numbers it was derived from so a
human (or the dev agent) can audit the call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Aggregate input-effectiveness floor. Distinct from progress.py's
# DEAD_INPUT_THRESHOLD (a per-press frame-diff score): this is the fraction
# of ALL acked presses that visibly did something. Below this, controls are
# broadly unresponsive — either the input mapping is wrong or the game isn't
# consuming key events. Set low so we only flag genuinely dead control, not
# a game with some no-op buttons (menus, held-state keys).
DEAD_INPUT_RATIO = 0.15
# Below this many acked presses we don't judge input effectiveness at all —
# too small a sample to distinguish "dead controls" from "barely tried".
MIN_ACKED_FOR_INPUT_VERDICT = 6

DefectKind = Literal["boot", "dead_input", "softlock", "difficulty_wall", "looks_wrong"]
DefectSeverity = Literal["blocker", "major", "minor"]

# Stable ordering for rendered briefs — worst first. A boot defect makes the
# rest moot, so it must lead; cosmetic notes trail.
_KIND_ORDER: dict[str, int] = {
    "boot": 0,
    "dead_input": 1,
    "softlock": 2,
    "difficulty_wall": 3,
    "looks_wrong": 4,
}


@dataclass(frozen=True)
class Defect:
    """One actionable problem found by playtesting, with its evidence."""

    kind: DefectKind
    severity: DefectSeverity
    detail: str  # human-readable, handed verbatim to the dev agent
    signal: dict[str, Any] = field(default_factory=dict)  # raw numbers it came from


def defects_from_progress(
    progress: dict[str, Any] | None,
    *,
    duration_ms: int | None = None,
    vision_notes: list[str] | None = None,
) -> list[Defect]:
    """Derive the defect list from a ``ProgressScore.to_dict()`` payload.

    ``progress`` is the dict carried in ``SessionEndPayload.progress`` (may be
    ``None`` if scoring itself failed — treated as a blocker). ``vision_notes``
    are optional free-text observations from the render/vision-verify stage,
    each surfaced as a ``looks_wrong`` minor. The returned list is ordered
    worst-first and deduped by kind.
    """
    defects: list[Defect] = []

    if not progress:
        # Scoring failed outright — the session produced no gradable signal.
        # That is itself a blocker: the dev agent should assume the build did
        # not run rather than that it ran perfectly.
        defects.append(Defect(
            kind="boot",
            severity="blocker",
            detail=(
                "No scorecard was produced — the playtest session yielded no "
                "gradable signal. Treat the build as non-running: verify it "
                "launches, renders a canvas, and emits progress postMessages."
            ),
            signal={"progress": None},
        ))
        return defects

    reached_play = bool(progress.get("reached_play", False))
    inputs_acked = int(progress.get("inputs_acked", 0) or 0)
    inputs_effective = int(progress.get("inputs_effective", 0) or 0)
    goals_completed = int(progress.get("goals_completed", 0) or 0)
    score = float(progress.get("score", 0.0) or 0.0)
    ratio = float(progress.get("effective_input_ratio", 0.0) or 0.0)

    # 1. Boot — never got past title/loading into play. Everything else is
    #    moot until this clears, so it leads and short-circuits input/goal
    #    analysis (those signals are meaningless pre-play).
    if not reached_play:
        defects.append(Defect(
            kind="boot",
            severity="blocker",
            detail=(
                "The game never reached a playable screen (stuck on "
                "title/loading/unknown). Ensure the build boots directly into "
                "play or past its menus quickly, and that it emits a "
                "postMessage({type:'screen', label:'play'}) once interactive."
            ),
            signal={"reached_play": False, "inputs_acked": inputs_acked},
        ))
        # Still surface any vision notes — a broken boot can coexist with a
        # visibly-wrong asset, and the dev agent benefits from both.
        _append_vision(defects, vision_notes)
        return _ordered(defects)

    # 2. Dead input — reached play but presses do nothing. Wrong key mapping
    #    or the game isn't consuming synthetic KeyboardEvents.
    if inputs_acked >= MIN_ACKED_FOR_INPUT_VERDICT and ratio < DEAD_INPUT_RATIO:
        defects.append(Defect(
            kind="dead_input",
            severity="major",
            detail=(
                f"Controls are largely unresponsive — only {inputs_effective} of "
                f"{inputs_acked} presses ({ratio:.0%}) changed the screen. Check "
                "that AUGMENTUM_GAME.semantic_to_key maps to the KeyboardEvent "
                "codes the game actually listens for, and that the game reads "
                "key events on the document/window (not a focused element)."
            ),
            signal={"inputs_acked": inputs_acked,
                    "inputs_effective": inputs_effective,
                    "effective_input_ratio": ratio},
        ))

    # 3. Softlock — in play, inputs do land, but no scored progress at all.
    #    The player can act but cannot advance: dead-end state, missing win
    #    condition, or unreachable objective.
    if reached_play and score <= 0.0 and inputs_effective > 0:
        defects.append(Defect(
            kind="softlock",
            severity="major",
            detail=(
                "Inputs affect the screen but the run made zero scored progress "
                "— likely a softlock or dead-end: no reachable next state, no "
                "win condition, or the objective can't actually be pursued from "
                "the starting state. Verify the game loop advances and emits "
                "progress postMessages as the player acts."
            ),
            signal={"score": score, "inputs_effective": inputs_effective},
        ))

    # 4. Difficulty wall — playable and progressing, but no objective met.
    #    Softest signal; minor because the build IS functional.
    if reached_play and goals_completed == 0 and score > 0.0:
        defects.append(Defect(
            kind="difficulty_wall",
            severity="minor",
            detail=(
                "The game is playable and the agent made progress but completed "
                "no objectives. Consider whether the first objective is reachable "
                "within a short session — lower the initial difficulty or bring "
                "the first goal closer to the start."
            ),
            signal={"goals_completed": 0, "score": score},
        ))

    _append_vision(defects, vision_notes)
    return _ordered(defects)


def _append_vision(defects: list[Defect], vision_notes: list[str] | None) -> None:
    """Fold render/vision-verify observations in as cosmetic minors."""
    for note in (vision_notes or []):
        note = (note or "").strip()
        if note:
            defects.append(Defect(
                kind="looks_wrong",
                severity="minor",
                detail=note,
                signal={"source": "vision_verify"},
            ))


def _ordered(defects: list[Defect]) -> list[Defect]:
    """Worst-first, deduped by (kind, detail). Stable within a kind."""
    seen: set[tuple[str, str]] = set()
    out: list[Defect] = []
    for d in sorted(defects, key=lambda d: _KIND_ORDER.get(d.kind, 99)):
        key = (d.kind, d.detail)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def relay_brief(defects: list[Defect], score: dict[str, Any] | None) -> str:
    """Render an ordered, dev-agent-ready feedback block.

    Returned string is meant to be injected into the next generation pass's
    prompt. When there are no defects, it says so plainly (and reports the
    score) so the dev agent doesn't invent problems — a clean playtest is a
    valid, informative outcome.
    """
    lines: list[str] = ["## Playtest feedback (from autonomous play)"]

    if score:
        s = score.get("score")
        spm = score.get("score_per_min")
        reached = score.get("reached_play")
        lines.append(
            f"Scorecard: score={s} score_per_min={spm} reached_play={reached} "
            f"inputs {score.get('inputs_effective')}/{score.get('inputs_acked')} effective."
        )

    if not defects:
        lines.append(
            "No blocking defects found — the build booted, controls responded, "
            "and the agent made scored progress. Focus the next pass on depth/"
            "polish rather than fixing breakage."
        )
        return "\n".join(lines)

    lines.append("")
    lines.append("Fix these, worst first:")
    for i, d in enumerate(defects, 1):
        lines.append(f"{i}. [{d.severity.upper()}] {d.kind}: {d.detail}")
    return "\n".join(lines)
