"""Bug Finder eval runner.

CLI driver around the scoring harness in `eval_harness.py`. Two modes:

    # Dry-run: load fixtures, show what would be measured, exit.
    python -m augmentum.bug_finder.eval_runner

    # Score saved run reports against fixtures (produces the aggregate
    # EvalReport with precision / recall / FP-bait survival).
    python -m augmentum.bug_finder.eval_runner --reports-dir ./bf_runs/

Reports-dir layout: one JSON file per fixture, named `<fixture_id>.json`,
containing a serialized `BugFinderRunReport` (the same shape the orchestrator
writes into `bug_finder_runs.report_json`). The simplest way to produce one
is to run a bug-finder pipeline against each fixture via the UI/API and
copy the run report out of the database:

    sqlite3 data/augmentum.db \\
      "SELECT report_json FROM bug_finder_runs WHERE workspace_id = '<id>'" \\
      > <fixture_id>.json

A future revision can wire up a `--live` mode that drives the orchestrator
in-process against fixture workspaces — see the design note at the end of
the file. Keeping that out of the runner today preserves a clean
separation: the harness only depends on `Finding` dataclass shape, not on
the rest of Augmentum's runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from augmentum.bug_finder.eval_harness import (
    EvalReport,
    Fixture,
    FixtureScore,
    load_fixture_set,
    score_fixture,
)
from augmentum.bug_finder.findings import Finding


def _findings_from_report(report: dict[str, Any]) -> list[Finding]:
    """Rehydrate Finding dataclasses from a serialized run-report JSON.

    Accepts either the top-level orchestrator JSON or the row dict the
    routes return (which nests the report under a `report` key)."""
    body = report.get("report") if "report" in report else report
    raw_findings = body.get("findings") if isinstance(body, dict) else None
    if not isinstance(raw_findings, list):
        return []
    out: list[Finding] = []
    for row in raw_findings:
        if not isinstance(row, dict):
            continue
        try:
            evidence = tuple(row.get("evidence_paths") or ())
            out.append(Finding(
                id=str(row.get("id") or ""),
                file=str(row.get("file") or ""),
                function=str(row.get("function") or "<module>"),
                claim=str(row.get("claim") or ""),
                claim_signature=str(row.get("claim_signature") or "other"),
                severity=str(row.get("severity") or "medium"),
                evidence_paths=evidence,
                suggested_repro=str(row.get("suggested_repro") or ""),
                status=str(row.get("status") or "speculative"),
                runs_to_confirm=int(row.get("runs_to_confirm") or 0),
                total_runs=int(row.get("total_runs") or 0),
                repro_path=str(row.get("repro_path") or ""),
                repro_command=str(row.get("repro_command") or ""),
                repro_output=str(row.get("repro_output") or ""),
                invariant=str(row.get("invariant") or ""),
                patch=str(row.get("patch") or ""),
                fix_attempts=int(row.get("fix_attempts") or 0),
                notes=list(row.get("notes") or []),
            ))
        except (TypeError, ValueError):
            continue
    return out


def _cost_from_report(report: dict[str, Any]) -> tuple[int, int, int]:
    body = report.get("report") if "report" in report else report
    ledger = body.get("cost_ledger") if isinstance(body, dict) else None
    if not isinstance(ledger, list):
        return (0, 0, 0)
    tokens_in = sum(int(e.get("tokens_in", 0) or 0) for e in ledger if isinstance(e, dict))
    tokens_out = sum(int(e.get("tokens_out", 0) or 0) for e in ledger if isinstance(e, dict))
    wallclock = sum(int(e.get("wallclock_ms", 0) or 0) for e in ledger if isinstance(e, dict))
    return (tokens_in, tokens_out, wallclock)


def score_from_saved_run(
    fixture: Fixture,
    report_path: Path,
) -> FixtureScore | None:
    """Score one fixture against a saved orchestrator report.

    Returns None when the report file is missing — the runner treats
    that as "fixture not yet measured" rather than failing the whole run.
    """
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    findings = _findings_from_report(report)
    cost_in, cost_out, cost_ms = _cost_from_report(report)
    return score_fixture(
        fixture, findings,
        cost_tokens_in=cost_in,
        cost_tokens_out=cost_out,
        cost_wallclock_ms=cost_ms,
    )


def _format_score_line(score: FixtureScore) -> str:
    mark = "PASS" if score.passed else "FAIL"
    if score.kind == "true_positive":
        detail = (
            f"matched_strong={score.matched_strong}/{score.expected_count}"
            f" weak={score.matched_weak} extra_confirmed={score.extra_confirmed_findings}"
        )
    else:
        detail = f"extra_confirmed={score.extra_confirmed_findings}"
    cost = ""
    if score.cost_tokens_in or score.cost_wallclock_ms:
        cost = (
            f"  [{score.cost_tokens_in + score.cost_tokens_out} tok,"
            f" {score.cost_wallclock_ms // 1000}s]"
        )
    return f"  [{mark}] {score.fixture_id} ({score.kind}): {detail}{cost}"


def format_report(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Bug Finder eval report")
    lines.append("=" * 72)
    for s in report.fixtures:
        lines.append(_format_score_line(s))
    lines.append("-" * 72)
    lines.append(f"Precision        : {report.precision * 100:6.2f}%")
    lines.append(f"Recall           : {report.recall * 100:6.2f}%")
    lines.append(f"FP-bait survival : {report.fp_bait_survival * 100:6.2f}%")
    lines.append(f"PoC build rate   : {report.poc_build_rate * 100:6.2f}%")
    lines.append(f"Passed fixtures  : {report.passed_count} / {len(report.fixtures)}")
    lines.append(f"AGGREGATE SCORE  : {report.aggregate_score:6.1f} / 100")
    lines.append("=" * 72)
    return "\n".join(lines)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="augmentum.bug_finder.eval_runner",
        description="Score bug-finder runs against the canonical fixture set.",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests" / "bug_finder_fixtures",
        help="Directory containing fixture subdirectories.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help=(
            "Directory of saved BugFinderRunReport JSON files, one per "
            "fixture (named <fixture_id>.json). Without this flag, dry-run "
            "mode reports what would be measured and exits."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Output format. JSON is machine-readable; human is the default.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the report to a file instead of stdout.",
    )
    args = parser.parse_args(argv)

    fixtures = load_fixture_set(args.fixtures_root)

    if args.reports_dir is None:
        # Dry run.
        print(f"[dry-run] Loaded {len(fixtures)} fixtures from {args.fixtures_root}")
        for f in fixtures:
            print(
                f"  {f.fixture_id:40s}  kind={f.kind:14s}  "
                f"expected={len(f.expected_findings)}"
            )
        print(
            f"\nNothing to score — pass --reports-dir <path> with one "
            f"<fixture_id>.json per fixture.",
        )
        return 0

    scores: list[FixtureScore] = []
    missing: list[str] = []
    for fixture in fixtures:
        report_path = args.reports_dir / f"{fixture.fixture_id}.json"
        score = score_from_saved_run(fixture, report_path)
        if score is None:
            missing.append(fixture.fixture_id)
            # Score the missing run as a "did nothing" pass — preserves
            # the aggregate's denominators.
            score = score_fixture(fixture, [])
        scores.append(score)

    report = EvalReport(fixtures=scores)
    if args.format == "json":
        out = json.dumps(report.to_dict(), indent=2)
    else:
        out = format_report(report)
        if missing:
            out += "\n\n" + (
                f"Note: {len(missing)} fixture(s) had no report file:\n"
                + "\n".join(f"  - {fid}" for fid in missing)
            )

    if args.out:
        args.out.write_text(out + "\n", encoding="utf-8")
        print(f"Wrote report to {args.out}", file=sys.stderr)
    else:
        print(out)
    return 0 if report.aggregate_score >= 75.0 and not missing else 1


# ---------------------------------------------------------------------------
# Design note — in-process live mode (deferred to its own session)
# ---------------------------------------------------------------------------
#
# To run the orchestrator in-process against each fixture, you need:
#
#   1. A ContainerManager (Docker-required). Coder uses
#      app.state.container_manager; the runner can construct one with
#      `aiodocker.Docker()` directly.
#   2. A ProviderRegistry to resolve model names → backends. Build it the
#      same way `create_app()` does: via `augmentum.providers.registry`.
#   3. A temporary workspace for each fixture — copy the fixture's source
#      files into a fresh `project_checkouts` directory, register it via
#      `coder.containers.ContainerManager.start_workspace()`, and stop it
#      after the run.
#
# The cleanest wire-up is probably to expose this as an in-app HTTP route
# (e.g. `POST /api/bug-finder/eval/{fixture_id}`) so it inherits the full
# app context, then have the CLI just poll that endpoint per fixture.
# That keeps the runner's surface area small and avoids duplicating
# coder's container glue.


if __name__ == "__main__":
    raise SystemExit(run_cli())
