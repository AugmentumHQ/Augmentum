"""Subagent worker-loop scorecard bench.

Phase 0 of the subagent-professionalization program
(``docs/superpowers/specs/2026-06-19-subagent-professionalization.md``).
Produces the one-command scorecard that every later phase is graded
against — and captures the committed baseline (step 0.4).

Two sources, both offline (no model required) so a baseline can be taken
from real usage today:

  * ``--db PATH``       read persisted ``coder_subagent_runs`` rows (the
                        live history). Defaults to ``{data_dir}/augmentum.db``.
  * ``--fixture PATH``  read a JSON list of run-dicts (the eval-set shape;
                        what the future ``--run-live`` path will emit).

Verification accuracy needs ground truth, so pass ``--labels PATH`` — a
JSON map of ``{subagent_id: true|false}`` (did the run ACTUALLY meet its
criteria). Without labels the scorecard reports accuracy as ``null``; the
budget + tool-efficiency + stop-reason rollups need no labels.

Usage
-----
    # Baseline from live history:
    python scripts/subagent_bench.py --out docs/superpowers/specs/data/subagent-baseline-2026-06-19.json

    # From a fixture, with ground-truth labels:
    python scripts/subagent_bench.py --fixture runs.json --labels labels.json

The ``--run-live`` eval-set execution (spec step 0.1) is a follow-on: it
runs the fixture tasks against a model and emits the run-dicts this bench
consumes. The grading path here is the part that must be correct first.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Reuse the canonical column list + row parser — do NOT re-derive the
# schema here (it would drift from persistence.py).
from augmentum.agents.metrics import aggregate
from augmentum.agents.persistence import _COLUMNS, _row_to_dict


def _default_db() -> str | None:
    try:
        from augmentum.config import settings  # lazy: avoid import cost when unused

        return f"{settings.data_dir}/augmentum.db"
    except Exception:
        return None


def _load_from_db(db_path: str, *, limit: int, role: str | None) -> list[dict[str, Any]]:
    if not Path(db_path).exists():
        raise SystemExit(f"db not found: {db_path}")
    # Read-only connection so a running server isn't disturbed.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = ", ".join(_COLUMNS)
        sql = f"SELECT {cols} FROM coder_subagent_runs"
        params: list[Any] = []
        if role:
            sql += " WHERE role = ?"
            params.append(role)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(max(1, min(limit, 100_000)))
        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                raise SystemExit(
                    f"{db_path}: no coder_subagent_runs table "
                    "(unmigrated or empty db -- point at the populated "
                    "augmentum.db, e.g. inside the container's data volume)"
                ) from exc
            raise
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def _load_from_fixture(fixture_path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("fixture must be a JSON list of run-dicts")
    return data


def _build_graded(
    runs: list[dict[str, Any]], labels: dict[str, bool]
) -> list[tuple[dict[str, Any], bool]]:
    graded: list[tuple[dict[str, Any], bool]] = []
    for r in runs:
        sid = str(r.get("subagent_id", ""))
        if sid in labels:
            graded.append((r, bool(labels[sid])))
    return graded


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _print_scorecard(card: Any, *, n_labeled: int) -> None:
    d = card.to_dict()
    print("\n=== Subagent worker-loop scorecard ===")
    print(f"runs:                 {d['n_runs']}")
    print(f"stop reasons:         {d['stop_reasons']}")
    print(f"verification counts:  {d['verification_counts']}")
    print(f"mean iterations:      {_fmt(d['mean_iterations'])}")
    print(f"mean tokens in/out:   {_fmt(d['mean_tokens_in'])} / {_fmt(d['mean_tokens_out'])}")
    print(f"mean wallclock (ms):  {_fmt(d['mean_wallclock_ms'])}")
    print(f"mean tool-efficiency: {_fmt(d['mean_tool_efficiency'])}")
    v = d["verification"]
    if n_labeled:
        print(f"verification accuracy:    {_fmt(v['accuracy'])}  (n={v['n_labeled']})")
        print(f"  false-positive rate:    {_fmt(v['false_positive_rate'])}  "
              f"(judge passed an actually-failing run -- Phase 1 target)")
        print(f"  false-negative rate:    {_fmt(v['false_negative_rate'])}")
    else:
        print("verification accuracy:    - (no --labels; provide ground truth to grade)")
    print(f"reward-hacking delta-gap: {_fmt(d['reward_hacking_gap'])}  "
          f"(wired; populated in Phase 2)")
    if d["per_role"]:
        print("\nper-role:")
        for role, rr in sorted(d["per_role"].items()):
            print(f"  {role:16s} n={rr['n_runs']:<4d} "
                  f"eff={_fmt(rr['mean_tool_efficiency'])} "
                  f"iters={_fmt(rr['mean_iterations'])} "
                  f"stops={rr['stop_reasons']}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Subagent worker-loop scorecard bench")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--db", default=None, help="path to augmentum.db (default: {data_dir}/augmentum.db)")
    src.add_argument("--fixture", default=None, help="JSON list of run-dicts")
    ap.add_argument("--labels", default=None, help="JSON {subagent_id: bool} ground truth")
    ap.add_argument("--role", default=None, help="filter to one role")
    ap.add_argument("--limit", type=int, default=500, help="max runs (db source)")
    ap.add_argument("--out", default=None, help="write the scorecard JSON here (the baseline artifact)")
    args = ap.parse_args(argv)

    if args.fixture:
        runs = _load_from_fixture(args.fixture)
    else:
        db_path = args.db or _default_db()
        if not db_path:
            ap.error("no --db/--fixture and could not resolve a default db path")
        runs = _load_from_db(db_path, limit=args.limit, role=args.role)

    if not runs:
        print("no subagent runs found -- nothing to score "
              "(run some dispatches, or pass --fixture).", file=sys.stderr)

    labels: dict[str, bool] = {}
    if args.labels:
        labels = {str(k): bool(v) for k, v in
                  json.loads(Path(args.labels).read_text(encoding="utf-8")).items()}
    graded = _build_graded(runs, labels) if labels else None

    card = aggregate(runs, graded=graded)
    _print_scorecard(card, n_labeled=len(graded or []))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
        print(f"wrote scorecard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
