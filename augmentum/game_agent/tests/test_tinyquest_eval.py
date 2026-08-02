"""The loop, closed on a real game: run → score → diff, unattended.

TinyQuest is a genuine game with a title gate, walls, four visually
distinct rooms, and an NPC — and crucially, NO RAM probes and no VLM in
these runs. Everything the score sees, it reads off the pixels.

The headline test drives two different agent policies through it and
asserts the scorecard tells them apart: one that never presses start
must score zero, and one that plays must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from augmentum.game_agent.eval_games.demo import (
    explorer_policy,
    masher_policy,
    run_variant,
)
from augmentum.game_agent.eval_games.tinyquest import (
    COLS,
    ROWS,
    TinyQuest,
)
from augmentum.game_agent.evalsuite import diff_suites
from augmentum.game_agent.progress import score_session

# Long enough for the explorer to cross rooms, short enough for CI.
_SESSION_S = 5.0


# ── the game's own rules ──────────────────────────────────────────────


def test_title_gate_only_opens_on_confirm() -> None:
    """@example: the game does not start until the right button is pressed.

    This is what makes TinyQuest a fair test of "beyond loading": an
    agent that mashes the wrong button genuinely never plays.
    """

    g = TinyQuest()
    for dead in ("cancel", "nav_up", "nav_left"):
        g.press(dead)
        assert g.started is False
    g.press("confirm")
    assert g.started is True


def test_walls_block_movement() -> None:
    """@example: bumping a wall does not move the player."""

    g = TinyQuest()
    g.press("confirm")
    g.px, g.py = 1, 1
    g.press("nav_up")  # row 0 is border wall
    assert (g.px, g.py) == (1, 1)
    g.press("nav_right")
    assert (g.px, g.py) == (2, 1)


def test_doors_move_between_visually_distinct_rooms() -> None:
    """@example: walking through a door changes room and repaints the screen."""

    g = TinyQuest()
    g.press("confirm")
    before = g.render()
    g.px, g.py = COLS - 2, ROWS // 2
    g.press("nav_right")  # onto the east door

    assert g.room == 1
    assert g.rooms_visited == {0, 1}
    assert g.render() != before


def test_render_is_deterministic() -> None:
    """@example: identical state renders identical pixels.

    Two suite runs are only comparable if the game is not a source of
    variance.
    """

    a, b = TinyQuest(), TinyQuest()
    for s in ("confirm", "nav_right", "nav_right", "nav_down"):
        a.press(s)
        b.press(s)
    assert a.render() == b.render()


# ── the probe-free gate ───────────────────────────────────────────────


def test_pixels_alone_can_establish_reached_play() -> None:
    """@example: an unlabelled game can still prove it got past loading.

    ROOT CAUSE:
      reached_play originally required a labelled screen. On a game with
      no RAM probe and no VLM — the hardest and most important case —
      nothing ever labels anything, so the score was pinned at zero no
      matter how much real progress happened.
    """

    s = score_session(
        novelty={"visual": 12},
        screens_seen=[],  # nothing labelled anything
        inputs_acked=40,
        inputs_effective=9,
        duration_ms=6_000,
    )

    assert s.reached_play is True
    assert s.reached_play_inferred is True
    assert s.score > 0


def test_screen_labels_outrank_pixel_inference() -> None:
    """@example: a witness saying 'still loading' beats a pixel guess.

    ROOT CAUSE:
      An animated loading screen produces plenty of distinct frames, and
      presses can coincide with the animation. If the pixel fallback
      could overrule a label, boot churn would score as play.
    """

    s = score_session(
        novelty={"visual": 99},
        screens_seen=["title", "loading"],  # a labeller exists, and it says no
        inputs_acked=40,
        inputs_effective=30,
        duration_ms=6_000,
    )

    assert s.reached_play is False
    assert s.score == 0.0
    assert s.verdict == "loaded_only"


# ── the closed loop, on the real game ─────────────────────────────────


@pytest.mark.asyncio
async def test_real_game_run_score_diff_distinguishes_playing_from_not(
    tmp_path: Path,
) -> None:
    """@example: THE goal condition — a real game, run, scored, and diffed.

    Two agent policies play a real game unattended. The scorecard must
    rank the one that actually played above the one that only sat on the
    title screen, and the diff must report that as an improvement. No
    human watches any part of this.

    The model is stubbed (deterministic policies) so this runs offline;
    the game, frames, input-effect measurement, novelty, scoring and
    diff are all real.
    """

    masher = await run_variant(
        "masher", masher_policy(), tmp_path, session_s=_SESSION_S
    )
    explorer = await run_variant(
        "explorer", explorer_policy(), tmp_path, session_s=_SESSION_S
    )

    m, e = masher.cases[0], explorer.cases[0]

    # Both sessions completed on their own.
    assert masher.n_failed == 0 and explorer.n_failed == 0

    # The agent that never pressed start is refused any credit.
    assert m.reached_play is False
    assert m.score == 0.0
    assert m.verdict == "loaded_only"

    # The agent that played is credited — from pixels alone.
    assert e.reached_play is True
    assert e.progress["reached_play_inferred"] is True
    assert e.score > 0
    assert e.verdict == "progressing"
    assert e.progress["novelty"]["visual"] > m.progress["novelty"]["visual"]

    # And the comparison a developer would actually run says "better".
    diff = diff_suites(masher, explorer)
    assert diff["verdict"] == "better"
    assert diff["shared_total_delta"] > 0
    assert diff["newly_failing"] == []


@pytest.mark.asyncio
async def test_real_game_run_is_reproducible(tmp_path: Path) -> None:
    """@example: the same policy twice yields the same verdict.

    A diff is only trustworthy if re-running an UNCHANGED agent reports
    no change. Scores are wall-clock sensitive (more turns fit in a
    faster run), so this pins the verdict and the play gate rather than
    the exact float.
    """

    a = await run_variant("a", explorer_policy(), tmp_path, session_s=_SESSION_S)
    b = await run_variant("b", explorer_policy(), tmp_path, session_s=_SESSION_S)

    assert a.cases[0].verdict == b.cases[0].verdict == "progressing"
    assert a.cases[0].reached_play == b.cases[0].reached_play is True
