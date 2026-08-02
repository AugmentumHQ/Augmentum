"""Multi-run eval harness: the run → score → diff → repeat loop.

These guard the properties that make an unattended suite trustworthy: a
wedged case cannot take the suite down, a broken case cannot vanish from
the result, and a diff cannot report "better" while something broke.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from augmentum.game_agent.evalsuite import (
    CaseResult,
    SuiteCase,
    SuiteResult,
    diff_suites,
    format_diff,
    run_case,
    run_suite,
)
from augmentum.game_agent.progress import score_session


def _score(value: float, *, reached: bool = True):
    """A ProgressScore with a chosen headline score."""

    s = score_session(
        novelty={},
        screens_seen=["overworld"] if reached else ["title"],
        inputs_acked=1,
        inputs_effective=1,
        duration_ms=1000,
    )
    s.score = value
    return s


def _case(name: str, **kw) -> SuiteCase:
    kw.setdefault("objective", "make progress")
    kw.setdefault("game", "testgame")
    return SuiteCase(name=name, **kw)


# ── per-case isolation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wedged_case_times_out_instead_of_hanging() -> None:
    """@example: a case that never returns becomes a timeout row.

    ROOT CAUSE:
      'Unattended' is only true if a hang is a RESULT, not an operator
      waiting forever. A hanging module is exactly what blocked
      automation before this harness existed.
    """

    async def never_returns(_case: SuiteCase):
        await asyncio.sleep(3600)

    result = await run_case(_case("wedged", timeout_s=0.05), never_returns)

    assert result.status == "timeout"
    assert result.score == 0.0
    assert "0.05" in result.error


@pytest.mark.asyncio
async def test_crashing_case_is_recorded_not_dropped() -> None:
    """@example: a case that raises still produces a row.

    ROOT CAUSE:
      Dropping failed cases makes a change that breaks half the suite
      look like an improvement on the surviving half.
    """

    async def explodes(_case: SuiteCase):
        raise RuntimeError("adapter died")

    result = await run_case(_case("boom"), explodes)

    assert result.status == "error"
    assert "adapter died" in result.error
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_one_bad_case_does_not_kill_the_suite() -> None:
    """@example: the suite survives its worst member and scores the rest."""

    async def runner(case: SuiteCase):
        if case.name == "bad":
            raise RuntimeError("nope")
        if case.name == "slow":
            await asyncio.sleep(3600)
        return _score(5.0)

    suite = await run_suite(
        [
            _case("good"),
            _case("bad"),
            _case("slow", timeout_s=0.05),
            _case("good2"),
        ],
        runner,
        label="run-a",
    )

    assert len(suite.cases) == 4
    assert suite.n_failed == 2
    assert suite.total_score == 10.0
    assert {c.name for c in suite.cases if c.status == "ok"} == {"good", "good2"}


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed() -> None:
    """@example: CancelledError propagates so Ctrl-C still works.

    A blanket `except Exception` that also ate cancellation would make
    the harness unkillable.
    """

    async def cancels(_case: SuiteCase):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_case(_case("cancelled"), cancels)


# ── aggregation + persistence ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_suite_result_round_trips_through_disk(tmp_path: Path) -> None:
    """@example: a suite result is a durable, comparable artifact.

    The whole point is comparing today's run to one from before a
    change, so it must survive the process.
    """

    async def runner(_case: SuiteCase):
        return _score(3.0)

    suite = await run_suite([_case("a"), _case("b")], runner, label="baseline")
    path = suite.save(tmp_path / "baseline.json")

    loaded = SuiteResult.load(path)
    assert loaded.label == "baseline"
    assert loaded.total_score == 6.0
    assert loaded.by_name()["a"].verdict == suite.by_name()["a"].verdict

    # And it is plain JSON a human or CI step can read.
    raw = json.loads(path.read_text())
    assert raw["n_cases"] == 2
    assert raw["n_reached_play"] == 2


# ── the diff ──────────────────────────────────────────────────────────


def _suite(label: str, scores: dict[str, float], **status) -> SuiteResult:
    return SuiteResult(
        label=label,
        cases=[
            CaseResult(
                name=n, score=v, status=status.get(n, "ok"), reached_play=True
            )
            for n, v in scores.items()
        ],
    )


def test_diff_reports_better_when_everything_improved() -> None:
    """@example: a clean win reads as 'better'."""

    d = diff_suites(_suite("b", {"a": 1.0}), _suite("a", {"a": 4.0}))
    assert d["verdict"] == "better"
    assert d["shared_total_delta"] == 3.0
    assert d["improved"][0]["name"] == "a"


def test_diff_flags_mixed_results_rather_than_hiding_them() -> None:
    """@example: a flat total with one game up and one down is 'mixed'.

    ROOT CAUSE:
      Judging on the total alone lets a change that doubles one game and
      kills another pass as 'no change'.
    """

    d = diff_suites(
        _suite("b", {"x": 4.0, "y": 4.0}),
        _suite("a", {"x": 8.0, "y": 0.0}),
    )

    assert d["shared_total_delta"] == 0.0
    assert d["verdict"] == "mixed"
    assert [r["name"] for r in d["regressed"]] == ["y"]


def test_diff_calls_out_newly_broken_cases_over_score() -> None:
    """@example: a case that started timing out is 'broke_cases', not 'worse'.

    A break and a mild regression demand different responses, so they
    must not collapse into one verdict.
    """

    d = diff_suites(
        _suite("b", {"x": 4.0}),
        _suite("a", {"x": 0.0}, x="timeout"),
    )

    assert d["verdict"] == "broke_cases"
    assert d["newly_failing"] == ["x"]


def test_diff_ignores_cases_present_in_only_one_run() -> None:
    """@example: adding a case is not an improvement.

    ROOT CAUSE:
      Summing across a changed case list reads a changed SUITE as a
      changed AGENT.
    """

    d = diff_suites(
        _suite("b", {"shared": 2.0, "gone": 5.0}),
        _suite("a", {"shared": 2.0, "new": 9.0}),
    )

    assert d["verdict"] == "unchanged"
    assert d["shared_total_delta"] == 0.0
    assert d["added"] == ["new"]
    assert d["removed"] == ["gone"]


@pytest.mark.asyncio
async def test_real_sessions_run_score_and_diff_unattended(tmp_path: Path) -> None:
    """@example: the full loop — run real sessions, score, persist, diff.

    This is the goal condition in one test: two suite runs of real
    orchestrator sessions complete with no human involvement, each
    produces a comparable scorecard, and the diff answers whether
    anything moved between them.
    """

    from augmentum.game_agent.evalsuite import make_orchestrator_runner
    from augmentum.game_agent.surfaces.mock import MockAdapter

    async def stub_llm(_prompt: str, _frames: list[bytes]) -> str:
        return json.dumps(
            {
                "observations": ["acting"],
                "state_update": "",
                "actions": [{"semantic": "advance", "duration_ms": 10}],
                "confidence": 0.5,
                "next_check_in_ms": 50,
            }
        )

    cases = [_case("alpha", timeout_s=20), _case("beta", timeout_s=20)]

    async def _run(label: str) -> SuiteResult:
        runner = make_orchestrator_runner(
            log_dir=tmp_path / label,
            adapter_factory=lambda _c: MockAdapter(script=[]),
            llm=stub_llm,
            session_s=0.2,
        )
        return await run_suite(cases, runner, label=label)

    before = await _run("before")
    after = await _run("after")

    # Every case completed on its own and carries a real scorecard.
    assert before.n_failed == 0
    assert len(before.cases) == 2
    for c in before.cases:
        assert c.status == "ok"
        assert c.progress["verdict"] in {
            "no_inputs", "no_progress", "loaded_only", "progressing",
        }

    # The artifact survives the process, which is what makes re-running
    # after a change comparable at all.
    path = before.save(tmp_path / "before.json")
    reloaded = SuiteResult.load(path)

    diff = diff_suites(reloaded, after)
    assert diff["verdict"] in {
        "unchanged", "better", "worse", "mixed", "broke_cases",
    }
    assert diff["added"] == [] and diff["removed"] == []
    assert format_diff(diff)


def test_format_diff_is_readable() -> None:
    """@example: the diff renders to something a human can scan."""

    text = format_diff(
        diff_suites(_suite("b", {"a": 1.0}), _suite("a", {"a": 2.0}))
    )
    assert "verdict: better" in text
    assert "+ a" in text
