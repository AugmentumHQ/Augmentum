"""Progress-scoring spine: visual bucketing + the session scorecard.

These guard the property that makes the agent iterable: a run must
produce a number that the *planner cannot influence*, so two runs of the
same game under different code are comparable.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from augmentum.game_agent.orchestrator import Orchestrator
from augmentum.game_agent.perception import _fingerprint_bytes, visual_bucket
from augmentum.game_agent.progress import (
    DEAD_INPUT_THRESHOLD,
    score_from_world,
    score_session,
)
from augmentum.game_agent.surfaces.mock import MockAdapter
from augmentum.game_agent.world import WorldState


def _frame(bg: tuple[int, int, int], *, sprite_x: int | None = None) -> bytes:
    """A GBA-sized PNG: flat background plus an optional small sprite."""

    im = Image.new("RGB", (240, 160), bg)
    if sprite_x is not None:
        ImageDraw.Draw(im).rectangle(
            [sprite_x, 80, sprite_x + 8, 88], fill=(255, 255, 255)
        )
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _bucket_of(png: bytes) -> str | None:
    return visual_bucket(_fingerprint_bytes(png))


# ── visual bucketing ──────────────────────────────────────────────────


def test_visual_bucket_is_stable_across_sprite_motion() -> None:
    """@example: a sprite moving within a scene is NOT a new place.

    ROOT CAUSE:
      Visual novelty is the only unbounded probe-free progress signal, so
      if within-scene animation minted new buckets the score would climb
      while the agent stood still — exactly the false-progress reading
      the metric exists to prevent.
    """

    a = _bucket_of(_frame((40, 90, 40), sprite_x=40))
    b = _bucket_of(_frame((40, 90, 40), sprite_x=60))

    assert a is not None
    assert a == b


def test_visual_bucket_separates_distinct_scenes() -> None:
    """@example: a different screen yields a different bucket."""

    assert _bucket_of(_frame((40, 90, 40))) != _bucket_of(_frame((200, 30, 30)))


def test_visual_bucket_rejects_malformed_input() -> None:
    """@example: bad input yields no bucket rather than a wrong one."""

    assert visual_bucket(None) is None
    assert visual_bucket(b"") is None
    assert visual_bucket(b"\x00" * 17) is None


def test_visual_bucket_is_deterministic() -> None:
    """@example: the same frame always buckets identically.

    Cross-run comparison is meaningless if bucketing drifts.
    """

    png = _frame((10, 120, 200), sprite_x=100)
    assert _bucket_of(png) == _bucket_of(png)


# ── scoring: the "beyond loading" gate ────────────────────────────────


def test_loading_only_session_scores_zero() -> None:
    """@example: churning pixels on a title/loading screen is not progress.

    ROOT CAUSE:
      Attract-mode and loading animations generate unbounded visual
      novelty. Without the productive-screen gate, a game that never
      booted past its title would outscore one that reached play.
    """

    score = score_session(
        novelty={"visual": 40, "screen": 3},
        screens_seen=["title", "loading", "unknown"],
        inputs_acked=12,
        inputs_effective=6,
        duration_ms=60_000,
    )

    assert score.reached_play is False
    assert score.score == 0.0
    assert score.verdict == "loaded_only"
    # The raw evidence is still reported — gated, not discarded.
    assert score.novelty["visual"] == 40


def test_reaching_play_unlocks_scoring() -> None:
    """@example: one productive screen flips the gate and earns credit."""

    score = score_session(
        novelty={"visual": 40, "screen": 4},
        screens_seen=["title", "loading", "overworld"],
        inputs_acked=12,
        inputs_effective=6,
        duration_ms=60_000,
    )

    assert score.reached_play is True
    assert score.score > 0
    assert score.verdict == "progressing"
    assert score.productive_screens == ["overworld"]


def test_screen_credit_counts_only_productive_screens() -> None:
    """@example: title/loading flip-flop earns no screen credit.

    Both sessions report 4 distinct screen labels; only the productive
    ones may be scored, so the one that actually entered the game must
    score strictly higher.
    """

    boot_churn = score_session(
        novelty={"screen": 4},
        screens_seen=["title", "loading", "unknown", "overworld"],
        inputs_acked=5,
        inputs_effective=5,
        duration_ms=10_000,
    )
    real_play = score_session(
        novelty={"screen": 4},
        screens_seen=["overworld", "battle", "menu", "dialog"],
        inputs_acked=5,
        inputs_effective=5,
        duration_ms=10_000,
    )

    assert boot_churn.score < real_play.score


def test_no_inputs_is_its_own_verdict() -> None:
    """@example: a session that never pressed anything is distinguishable.

    'Agent never acted' and 'agent acted and got nowhere' are different
    failures and must not collapse into one bucket.
    """

    score = score_session(
        novelty={},
        screens_seen=[],
        inputs_acked=0,
        inputs_effective=0,
        duration_ms=5_000,
    )
    assert score.verdict == "no_inputs"
    assert score.effective_input_ratio == 0.0


# ── scoring: planner independence ─────────────────────────────────────


def test_planner_authored_goals_do_not_move_the_score() -> None:
    """@example: the model cannot inflate its own grade.

    ROOT CAUSE:
      Goal metrics are authored by the planner, so a model that declared
      victory would otherwise score well without playing. goals_completed
      is reported for context but excluded from `score`.
    """

    kwargs = dict(
        novelty={"visual": 10},
        screens_seen=["overworld"],
        inputs_acked=4,
        inputs_effective=2,
        duration_ms=30_000,
    )
    honest = score_session(**kwargs, goals_completed=0)
    boastful = score_session(**kwargs, goals_completed=99)

    assert honest.score == boastful.score
    assert boastful.goals_completed == 99


def test_effective_ratio_is_clamped_to_acked() -> None:
    """@example: nonsense counts cannot produce a >1 effectiveness ratio."""

    score = score_session(
        novelty={},
        screens_seen=["overworld"],
        inputs_acked=3,
        inputs_effective=999,
        duration_ms=1_000,
    )
    assert score.effective_input_ratio == 1.0


def test_score_per_min_normalizes_unequal_runs() -> None:
    """@example: a run that progresses faster rates higher per minute."""

    fast = score_session(
        novelty={"visual": 20}, screens_seen=["overworld"],
        inputs_acked=1, inputs_effective=1, duration_ms=60_000,
    )
    slow = score_session(
        novelty={"visual": 20}, screens_seen=["overworld"],
        inputs_acked=1, inputs_effective=1, duration_ms=600_000,
    )
    assert fast.score == slow.score
    assert fast.score_per_min > slow.score_per_min


def test_scoring_never_raises_on_degenerate_input() -> None:
    """@example: grading a broken run yields a low number, not an exception."""

    score = score_session(
        novelty={"visual": -5},
        screens_seen=[None, ""],
        inputs_acked=-1,
        inputs_effective=-1,
        duration_ms=-1,
    )
    assert score.score == 0.0
    assert score.duration_ms == 0


# ── scoring off a live WorldState ─────────────────────────────────────


def test_score_from_world_reads_the_novelty_tracker() -> None:
    """@example: the blackboard's novelty counts feed the scorecard."""

    world = WorldState()
    world.note("visual", "aaa", t_ms=0)
    world.note("visual", "bbb", t_ms=10)
    world.note("visual", "aaa", t_ms=20)  # revisit — not novel
    world.note("screen", "overworld", t_ms=30)
    world.note("dialog", "PROF: hello there", t_ms=40)

    score = score_from_world(
        world, inputs_acked=10, inputs_effective=7, duration_ms=60_000
    )

    assert score.novelty["visual"] == 2
    assert score.reached_play is True
    assert score.effective_input_ratio == 0.7
    assert score.score > 0


def test_world_novelty_keys_reflect_distinct_screens() -> None:
    """@example: novelty_keys exposes the small screen vocabulary."""

    world = WorldState()
    world.note("screen", "title", t_ms=0)
    world.note("screen", "overworld", t_ms=10)
    assert sorted(world.novelty_keys("screen")) == ["overworld", "title"]
    assert world.distinct_count("screen") == 2


# ── the shared dead-input threshold ───────────────────────────────────


def test_reflex_and_scorer_agree_on_dead_input() -> None:
    """@example: one calibrated threshold, not three copies.

    ROOT CAUSE:
      The value lived as a literal in the orchestrator (twice) and as a
      private constant in the universal rule pack. If the scorer graded
      'that press worked' by a different number than the reflex layer
      acted on, the score would describe a run that never happened.
    """

    from augmentum.game_agent.rule_packs.universal import _DEAD_EFFECT_THRESHOLD

    assert _DEAD_EFFECT_THRESHOLD is DEAD_INPUT_THRESHOLD


# ── end to end ────────────────────────────────────────────────────────


async def _stub_llm(_prompt: str, _frames: list[bytes]) -> str:
    return json.dumps(
        {
            "observations": ["acting"],
            "state_update": "turn taken",
            "actions": [{"semantic": "advance", "duration_ms": 100}],
            "confidence": 0.5,
            "next_check_in_ms": 200,
        }
    )


@pytest.mark.asyncio
async def test_session_end_carries_a_scorecard(tmp_path: Path) -> None:
    """@example: every finished session is graded in its trailer.

    This is the property the whole eval loop rests on — if a run does not
    self-report a comparable number, 'did this change help?' cannot be
    answered without a human watching.
    """

    log_path = tmp_path / "session.ndjson"
    orch = Orchestrator(
        log_path=str(log_path),
        surface_kind="mock",
        adapter=MockAdapter(script=[]),
        llm=_stub_llm,
        objective="be gradeable",
    )

    async def stopper() -> None:
        await asyncio.sleep(0.3)
        orch.stop("completed")

    stop_task = asyncio.create_task(stopper())
    end = await orch.run()
    await stop_task

    assert end.progress is not None
    assert "score" in end.progress
    assert "verdict" in end.progress
    assert end.progress["reached_play"] is False  # mock never reports a screen

    # And it survives the round-trip to disk, where an eval harness reads it.
    lines = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if line.strip()
    ]
    trailer = lines[-1]
    assert trailer["kind"] == "session_end"
    assert trailer["payload"]["progress"]["verdict"] == end.progress["verdict"]
