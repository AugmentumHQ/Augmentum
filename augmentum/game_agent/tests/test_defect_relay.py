"""Unit tests for the playtest defect relay + playtest framing.

These derive defects from synthetic ``ProgressScore.to_dict()`` payloads —
no Docker, no game, no LLM — so they run in the plain pytest lane and pin
the mapping that turns unfakeable play signals into dev-agent feedback.
"""
from __future__ import annotations

from augmentum.game_agent.defect_relay import (
    DEAD_INPUT_RATIO,
    Defect,
    defects_from_progress,
    relay_brief,
)
from augmentum.game_agent.playtest import build_playtest_objective
from augmentum.game_agent.progress import ProgressScore


def _score(**overrides) -> dict:
    """A ProgressScore.to_dict() with sensible defaults, overridable."""
    ps = ProgressScore(
        reached_play=True,
        inputs_acked=20,
        inputs_effective=15,
        goals_completed=1,
        duration_ms=60000,
        score=5.0,
    )
    d = ps.to_dict()
    d.update(overrides)
    return d


def _kinds(defects: list[Defect]) -> list[str]:
    return [d.kind for d in defects]


def test_none_progress_is_boot_blocker():
    defects = defects_from_progress(None)
    assert _kinds(defects) == ["boot"]
    assert defects[0].severity == "blocker"


def test_not_reached_play_is_boot_and_short_circuits():
    d = _score(reached_play=False, inputs_acked=0, inputs_effective=0, score=0.0)
    defects = defects_from_progress(d)
    # Boot leads and input/goal analysis is suppressed (moot pre-play).
    assert _kinds(defects) == ["boot"]


def test_dead_input_flagged_when_ratio_below_floor():
    # 1/20 effective = 0.05 ratio, well under the floor.
    d = _score(inputs_acked=20, inputs_effective=1,
               effective_input_ratio=0.05, score=2.0)
    defects = defects_from_progress(d)
    assert "dead_input" in _kinds(defects)
    di = next(x for x in defects if x.kind == "dead_input")
    assert di.severity == "major"
    assert di.signal["inputs_acked"] == 20


def test_dead_input_not_flagged_on_tiny_sample():
    # Only 3 presses — too small a sample to call controls dead.
    d = _score(inputs_acked=3, inputs_effective=0,
               effective_input_ratio=0.0, score=1.0)
    defects = defects_from_progress(d)
    assert "dead_input" not in _kinds(defects)


def test_softlock_when_play_reached_inputs_land_but_no_score():
    d = _score(score=0.0, inputs_acked=20, inputs_effective=12,
               effective_input_ratio=0.6, goals_completed=0)
    defects = defects_from_progress(d)
    assert "softlock" in _kinds(defects)


def test_difficulty_wall_when_progress_but_no_goals():
    d = _score(score=4.0, goals_completed=0, inputs_effective=10,
               effective_input_ratio=0.5)
    defects = defects_from_progress(d)
    assert "difficulty_wall" in _kinds(defects)
    assert next(x for x in defects if x.kind == "difficulty_wall").severity == "minor"


def test_clean_run_has_no_defects():
    d = _score()  # reached play, responsive, scored, a goal met
    defects = defects_from_progress(d)
    assert defects == []


def test_vision_notes_become_looks_wrong_minors():
    d = _score()
    defects = defects_from_progress(d, vision_notes=["The crate looks flat and untextured."])
    assert _kinds(defects) == ["looks_wrong"]
    assert defects[0].severity == "minor"


def test_defects_ordered_worst_first():
    # Construct a payload that trips dead_input + softlock + a vision note.
    d = _score(score=0.0, inputs_acked=20, inputs_effective=1,
               effective_input_ratio=0.05, goals_completed=0)
    defects = defects_from_progress(d, vision_notes=["colors clash"])
    order = _kinds(defects)
    # dead_input (1) before softlock (2) before looks_wrong (4).
    assert order.index("dead_input") < order.index("softlock") < order.index("looks_wrong")


def test_relay_brief_clean_run_is_explicit():
    brief = relay_brief([], _score())
    assert "No blocking defects" in brief
    assert "score=" in brief


def test_relay_brief_lists_defects_numbered():
    d = _score(reached_play=False)
    defects = defects_from_progress(d)
    brief = relay_brief(defects, d)
    assert "1. [BLOCKER] boot" in brief


def test_dead_input_ratio_boundary():
    # Exactly at the floor is NOT flagged (strict less-than).
    d = _score(inputs_acked=20, inputs_effective=int(20 * DEAD_INPUT_RATIO),
               effective_input_ratio=DEAD_INPUT_RATIO, score=1.0)
    assert "dead_input" not in _kinds(defects_from_progress(d))


def test_build_playtest_objective_wraps_goal():
    out = build_playtest_objective("collect all 3 coins")
    assert "collect all 3 coins." in out
    assert "graded automatically" in out
    # Empty falls back to a generic target.
    assert "win or clear-progress" in build_playtest_objective("")
