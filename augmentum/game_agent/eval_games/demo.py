"""One full run → score → diff cycle on a real game, unattended.

Run it::

    python -m augmentum.game_agent.eval_games.demo

It plays TinyQuest twice with two different agent policies, scores each
session from the pixels alone, and prints the diff between the two suite
runs — the same artifact a real before/after comparison produces.

What is real here and what is not
---------------------------------
Real: the game and its rules, the rendered frames, the input effect
scores (measured by diffing frames), the perception pipeline, the
novelty tracking, the scorecard, and the diff.

Stubbed: the *model*. The two policies below stand in for the slow-path
LLM so this runs offline and deterministically. That makes it an honest
test of the measurement spine — does the score distinguish an agent that
plays from one that does not? — and not a test of model intelligence.
Point ``llm=`` at a real backend to make it the latter.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from augmentum.game_agent.eval_games.tinyquest import TinyQuestAdapter
from augmentum.game_agent.evalsuite import (
    SuiteCase,
    SuiteResult,
    diff_suites,
    format_diff,
    make_orchestrator_runner,
    run_suite,
)

# A route that actually plays: start the game, cross east into room 1,
# talk to the NPC, then work south and west through two more rooms.
# Padded with repeats because bumping a wall is a legal (and useful)
# move for an agent that cannot see a map.
_ROUTE: list[str] = (
    ["confirm"]
    + ["nav_right"] * 8
    + ["nav_right"] * 8
    + ["confirm", "confirm"]
    + ["nav_down"] * 5
    + ["nav_down"] * 6
    + ["nav_left"] * 8
    + ["nav_up"] * 6
)


def _plan(actions: list[str], *, next_ms: int = 50) -> str:
    return json.dumps(
        {
            "observations": ["following the route"],
            "state_update": "",
            "actions": [{"semantic": s, "duration_ms": 20} for s in actions],
            "confidence": 0.6,
            "next_check_in_ms": next_ms,
        }
    )


def explorer_policy(batch: int = 4) -> Callable[..., Any]:
    """An agent that presses start and then explores the world."""

    state = {"i": 0}

    async def _llm(_prompt: str, _frames: list[bytes]) -> str:
        i = state["i"]
        state["i"] = i + batch
        chunk = _ROUTE[i : i + batch]
        if not chunk:
            # Route exhausted — keep nudging rather than going silent, so
            # the session ends because time ran out, not because the
            # agent stopped playing.
            chunk = ["nav_up", "nav_right"]
        return _plan(chunk)

    return _llm


def masher_policy() -> Callable[..., Any]:
    """An agent that never presses start.

    It hammers a button that does nothing on the title screen. This is
    the failure the score must refuse to reward: the game loaded, inputs
    were sent, and nothing was ever played.
    """

    async def _llm(_prompt: str, _frames: list[bytes]) -> str:
        return _plan(["cancel"] * 4)

    return _llm


CASES = [SuiteCase(name="tinyquest", game="tinyquest", objective="explore the world", timeout_s=60)]


async def run_variant(
    label: str, policy: Callable[..., Any], out_dir: Path, *, session_s: float
) -> SuiteResult:
    runner = make_orchestrator_runner(
        log_dir=out_dir / label,
        adapter_factory=lambda _c: TinyQuestAdapter(),
        llm=policy,
        surface_kind="mock",
        session_s=session_s,
    )
    return await run_suite(CASES, runner, label=label)


async def main(out_dir: Path | str = "eval-out", session_s: float = 6.0) -> int:
    out = Path(out_dir)
    before = await run_variant("masher", masher_policy(), out, session_s=session_s)
    after = await run_variant("explorer", explorer_policy(), out, session_s=session_s)
    before.save(out / "masher.json")
    after.save(out / "explorer.json")

    for suite in (before, after):
        c = suite.cases[0]
        print(
            f"[{suite.label:8}] verdict={c.verdict:12} score={c.score:7.2f} "
            f"reached_play={c.reached_play} "
            f"visual={c.progress.get('novelty', {}).get('visual', 0)} "
            f"eff={c.progress.get('effective_input_ratio')}"
        )
    print()
    print(format_diff(diff_suites(before, after)))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(asyncio.run(main()))
