"""Multi-run evaluation harness: run → score → diff → repeat.

The scorer in :mod:`augmentum.game_agent.progress` grades ONE session.
This module closes the loop that makes the agent iterable: run a fixed
suite of cases unattended, aggregate the scorecards, and diff two suite
results so "did this prompt/rule/model change help?" is answerable
without a human watching any of it.

Three disciplines, each learned from a way this can silently lie:

* **Every case is time-boxed.** A wedged session must cost one case, not
  the suite. Unattended means a hang is a *result* ("timeout"), never an
  operator waiting forever — the failure mode that motivated this module
  was a test module that hung indefinitely and blocked all automation.
* **A failed case still produces a row.** Crashes and timeouts land in
  the result as scored-zero entries with their reason attached. Dropping
  them would make a change that breaks half the suite look like an
  improvement on the surviving half.
* **Diffs are computed per case, never on the total alone.** A suite
  total can stay flat while one game doubles and another dies; the
  regressions list exists so that cannot pass as "no change".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from augmentum.game_agent.progress import ProgressScore

log = structlog.get_logger(__name__)

# Default per-case ceiling. Generous enough for a real session to make
# visible progress, short enough that a full suite of wedged cases still
# terminates within a coffee break.
DEFAULT_CASE_TIMEOUT_S = 300.0

# A session runner: given the case, drive it to completion and return the
# session's ProgressScore. Injected so this module never has to know
# whether the case is a mock, an emulator bridge, or a live game.
CaseRunner = Callable[["SuiteCase"], Awaitable[ProgressScore]]


@dataclass
class SuiteCase:
    """One reproducible unit of evaluation.

    ``seed`` is carried (not consumed here) so a runner can pin whatever
    stochastic surface it owns — sampling, spawn position, ROM save
    state. Two suite runs are only comparable if their cases match on
    (name, game, objective, seed).
    """

    name: str
    objective: str
    game: str = ""
    seed: int = 0
    timeout_s: float = DEFAULT_CASE_TIMEOUT_S
    meta: dict[str, Any] = field(default_factory=dict)

    def identity(self) -> tuple[str, str, str, int]:
        """What must match for two runs of this case to be comparable."""

        return (self.name, self.game, self.objective, self.seed)


@dataclass
class CaseResult:
    """One case's outcome. Always produced, even on failure."""

    name: str
    game: str = ""
    objective: str = ""
    seed: int = 0
    # "ok" | "timeout" | "error"
    status: str = "ok"
    score: float = 0.0
    verdict: str = "no_progress"
    reached_play: bool = False
    wall_ms: int = 0
    error: str = ""
    progress: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    """Aggregate over one suite run. The comparable artifact."""

    label: str = ""
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        return sum(c.score for c in self.cases)

    @property
    def n_reached_play(self) -> int:
        return sum(1 for c in self.cases if c.reached_play)

    @property
    def n_failed(self) -> int:
        """Cases that did not complete cleanly (timeout or crash)."""

        return sum(1 for c in self.cases if c.status != "ok")

    def by_name(self) -> dict[str, CaseResult]:
        return {c.name: c for c in self.cases}

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total_score": round(self.total_score, 3),
            "n_cases": len(self.cases),
            "n_reached_play": self.n_reached_play,
            "n_failed": self.n_failed,
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> SuiteResult:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            label=str(raw.get("label") or ""),
            cases=[
                CaseResult(
                    **{
                        k: v
                        for k, v in c.items()
                        if k in CaseResult.__dataclass_fields__
                    }
                )
                for c in raw.get("cases", [])
            ],
        )


async def run_case(case: SuiteCase, runner: CaseRunner) -> CaseResult:
    """Run one case to completion, a timeout, or an error — never a hang.

    Returns:
    - A :class:`CaseResult` in every path. This function does not raise
      on case failure; a suite must survive its worst member.
    """

    started = time.monotonic()
    base = {
        "name": case.name,
        "game": case.game,
        "objective": case.objective,
        "seed": case.seed,
    }
    try:
        score = await asyncio.wait_for(runner(case), timeout=case.timeout_s)
    except TimeoutError:  # asyncio.TimeoutError is an alias on 3.11+
        log.warning(
            "game_agent.eval_case_timeout", case=case.name, timeout_s=case.timeout_s
        )
        return CaseResult(
            **base,
            status="timeout",
            wall_ms=int((time.monotonic() - started) * 1000),
            error=f"exceeded {case.timeout_s}s",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("game_agent.eval_case_error", case=case.name, error=str(exc))
        return CaseResult(
            **base,
            status="error",
            wall_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )

    return CaseResult(
        **base,
        status="ok",
        score=score.score,
        verdict=score.verdict,
        reached_play=score.reached_play,
        wall_ms=int((time.monotonic() - started) * 1000),
        progress=score.to_dict(),
    )


async def run_suite(
    cases: list[SuiteCase], runner: CaseRunner, *, label: str = ""
) -> SuiteResult:
    """Run every case sequentially and aggregate.

    Sequential on purpose: these sessions contend for a GPU and a model
    slot, so running them concurrently would make wall-clock timings
    (and any timeout calibrated against them) meaningless.
    """

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        log.info("game_agent.eval_case_start", case=case.name, i=i, n=len(cases))
        results.append(await run_case(case, runner))
    suite = SuiteResult(label=label, cases=results)
    log.info(
        "game_agent.eval_suite_done",
        label=label,
        total_score=suite.total_score,
        n_failed=suite.n_failed,
    )
    return suite


def make_orchestrator_runner(
    *,
    log_dir: Path | str,
    adapter_factory: Callable[[SuiteCase], Any],
    llm: Any,
    surface_kind: str = "mock",
    session_s: float = 60.0,
    orchestrator_kwargs: dict[str, Any] | None = None,
) -> CaseRunner:
    """Build a :data:`CaseRunner` that drives a real Orchestrator.

    Use when:
    - Running an actual suite. ``adapter_factory`` returns a fresh
      adapter per case (adapters are stateful and single-session).

    Expects:
    - ``session_s`` is the in-session play budget. Keep it comfortably
      below ``SuiteCase.timeout_s``, which is the outer safety net for a
      session that fails to stop when asked.

    Returns:
    - An awaitable runner yielding the session's :class:`ProgressScore`,
      read back from the session_end trailer so the harness grades a run
      by exactly what landed in the log.
    """

    # Imported here: the orchestrator pulls in the whole agent stack, and
    # the diff/report half of this module must stay importable without it.
    from augmentum.game_agent.orchestrator import Orchestrator

    root = Path(log_dir)

    async def _run(case: SuiteCase) -> ProgressScore:
        root.mkdir(parents=True, exist_ok=True)
        orch = Orchestrator(
            log_path=str(root / f"{case.name}.ndjson"),
            surface_kind=surface_kind,
            adapter=adapter_factory(case),
            llm=llm,
            objective=case.objective,
            **(orchestrator_kwargs or {}),
        )

        async def _stopper() -> None:
            await asyncio.sleep(session_s)
            orch.stop("completed")

        stopper = asyncio.create_task(_stopper())
        try:
            end = await orch.run()
        finally:
            stopper.cancel()
            # Reaping our own cancelled timer — not an error path.
            with contextlib.suppress(asyncio.CancelledError):
                await stopper

        if not end.progress:
            # Scoring failed inside the orchestrator (it is fail-open).
            # Surface that as an explicit zero rather than inventing one.
            raise RuntimeError("session produced no progress scorecard")
        return ProgressScore.from_dict(end.progress)

    return _run


@dataclass
class CaseDelta:
    name: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


def diff_suites(
    before: SuiteResult, after: SuiteResult, *, epsilon: float = 1e-9
) -> dict[str, Any]:
    """Compare two suite runs and say whether the change helped.

    Use when:
    - Deciding to keep or revert a prompt/rule/model change.

    Returns:
    - A dict with per-case improvements/regressions, the totals, and
      ``added``/``removed`` case names. Cases present in only one run are
      never counted as a delta — that would read a changed suite as a
      changed agent.
    """

    b, a = before.by_name(), after.by_name()
    shared = sorted(set(b) & set(a))

    improved: list[CaseDelta] = []
    regressed: list[CaseDelta] = []
    for name in shared:
        d = CaseDelta(name=name, before=b[name].score, after=a[name].score)
        if d.delta > epsilon:
            improved.append(d)
        elif d.delta < -epsilon:
            regressed.append(d)

    shared_before = sum(b[n].score for n in shared)
    shared_after = sum(a[n].score for n in shared)
    # Failures are surfaced separately from score: a case that newly
    # times out often scores 0, which reads as a mild regression when it
    # is actually a break.
    newly_failing = sorted(
        n for n in shared if b[n].status == "ok" and a[n].status != "ok"
    )

    return {
        "improved": [
            {"name": d.name, "before": d.before, "after": d.after, "delta": d.delta}
            for d in sorted(improved, key=lambda x: -x.delta)
        ],
        "regressed": [
            {"name": d.name, "before": d.before, "after": d.after, "delta": d.delta}
            for d in sorted(regressed, key=lambda x: x.delta)
        ],
        "newly_failing": newly_failing,
        "shared_total_before": round(shared_before, 3),
        "shared_total_after": round(shared_after, 3),
        "shared_total_delta": round(shared_after - shared_before, 3),
        "added": sorted(set(a) - set(b)),
        "removed": sorted(set(b) - set(a)),
        # Honest headline: a change is only "better" if nothing broke.
        "verdict": _diff_verdict(
            any_new_failures=bool(newly_failing),
            any_improved=bool(improved),
            any_regressed=bool(regressed),
        ),
    }


def _diff_verdict(
    *, any_new_failures: bool, any_improved: bool, any_regressed: bool
) -> str:
    """Classify a diff by WHICH cases moved, not by the net total.

    Deliberately ignores ``total_delta``: a change that doubles one game
    and kills another nets to zero, and calling that "unchanged" (or
    letting the sign of the total call it "worse") hides the fact that
    two different things happened and need two different responses.
    """

    if any_new_failures:
        return "broke_cases"
    if any_improved and any_regressed:
        return "mixed"
    if any_regressed:
        return "worse"
    if any_improved:
        return "better"
    return "unchanged"


def format_diff(diff: dict[str, Any]) -> str:
    """Human-readable one-screen summary of :func:`diff_suites`."""

    lines = [
        f"verdict: {diff['verdict']}  "
        f"total {diff['shared_total_before']} -> {diff['shared_total_after']} "
        f"({diff['shared_total_delta']:+})"
    ]
    for key, arrow in (("improved", "+"), ("regressed", "-")):
        for row in diff[key]:
            lines.append(
                f"  {arrow} {row['name']}: {row['before']} -> {row['after']} "
                f"({row['delta']:+.3f})"
            )
    if diff["newly_failing"]:
        lines.append(f"  ! newly failing: {', '.join(diff['newly_failing'])}")
    for key in ("added", "removed"):
        if diff[key]:
            lines.append(f"  ({key}: {', '.join(diff[key])})")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CASE_TIMEOUT_S",
    "CaseResult",
    "CaseRunner",
    "SuiteCase",
    "SuiteResult",
    "diff_suites",
    "format_diff",
    "make_orchestrator_runner",
    "run_case",
    "run_suite",
]
