"""Cross-codebase baseline benchmark for the bug-finder agnostic substrate.

Clones a curated set of mature Python projects (varied size + style),
runs ``run_agnostic_stage`` against each, and emits one JSONL row per
project to ``.augmentum-bench/baseline_<timestamp>.jsonl``.

Purpose: establish a baseline of how the substrate behaves on real
codebases (raw count, severity mix, scanner mix, top patterns,
wallclock) so we can iterate on detection methods and watch the
numbers move over time.

Run:
    python scripts/baseline_benchmark.py
    python scripts/baseline_benchmark.py --only fastapi,flask
    python scripts/baseline_benchmark.py --keep-clones    # for follow-up inspection
    python scripts/baseline_benchmark.py --skip-large     # skip django/airflow/scrapy

Output:
    .augmentum-bench/
        baseline_2026-06-02T231503.jsonl   # one row per project
        baseline_2026-06-02T231503.md      # comparison report

History is append-only: rerun the bench at any time to see how
detection numbers move (regressions, new rules, suppression growth).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Make `augmentum` importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# stdout/stderr UTF-8 — Windows console crashes on ✓ otherwise.
try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Targets — mature Python codebases, varied size + domain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    slug: str
    git_url: str
    branch: str = ""           # empty → default branch
    is_large: bool = False     # skipped under --skip-large
    notes: str = ""


_TARGETS: tuple[Target, ...] = (
    Target(
        slug="httpx",
        git_url="https://github.com/encode/httpx",
        notes="modern async HTTP client; medium-small (~25kloc), clean codebase",
    ),
    Target(
        slug="flask",
        git_url="https://github.com/pallets/flask",
        notes="WSGI web framework; mature, conservative style",
    ),
    Target(
        slug="requests",
        git_url="https://github.com/psf/requests",
        notes="legacy HTTP client; idiomatic Python 2/3 era patterns",
    ),
    Target(
        slug="fastapi",
        git_url="https://github.com/fastapi/fastapi",
        notes="modern ASGI framework; type-heavy",
    ),
    Target(
        slug="pydantic",
        git_url="https://github.com/pydantic/pydantic",
        notes="validation + serialization; heavy use of metaclasses + Rust core",
    ),
    Target(
        slug="celery",
        git_url="https://github.com/celery/celery",
        is_large=True,
        notes="distributed task queue; sprawling, legacy patterns",
    ),
    Target(
        slug="scrapy",
        git_url="https://github.com/scrapy/scrapy",
        is_large=True,
        notes="async crawler; subprocess + parsing-heavy",
    ),
    Target(
        slug="django",
        git_url="https://github.com/django/django",
        is_large=True,
        notes="reference-class web framework; ~250kloc, ORM-heavy",
    ),
)


# ---------------------------------------------------------------------------
# Output rows
# ---------------------------------------------------------------------------


@dataclass
class BenchRow:
    """One bench result for one target."""

    slug: str
    git_url: str
    ok: bool
    error: str = ""

    # Inputs
    clone_seconds: float = 0.0
    file_count: int = 0
    py_count: int = 0
    branch: str = ""
    commit: str = ""

    # Findings
    total_raw: int = 0
    seeded_into_pipeline: int = 0
    suppressed: int = 0
    scanner_counts: dict[str, int] = field(default_factory=dict)
    severity_counts: dict[str, int] = field(default_factory=dict)
    top_patterns: list[tuple[str, int]] = field(default_factory=list)
    most_flagged_files: list[tuple[str, int]] = field(default_factory=list)

    # Timing
    scan_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shallow_clone(url: str, dest: Path, branch: str = "") -> tuple[bool, str]:
    cmd = ["git", "clone", "--depth", "1", "--quiet"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return False, "clone timeout"
    except subprocess.CalledProcessError as exc:
        return False, f"clone failed: {exc.stderr.decode(errors='ignore')[:160]}"
    return True, ""


def _head_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), check=True, capture_output=True, timeout=10,
        )
        return out.stdout.decode().strip()[:12]
    except Exception:    # noqa: BLE001
        return ""


def _count_files(root: Path) -> tuple[int, int]:
    """Return (all_files, py_files). Skips .git/."""
    total = 0
    py = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = p.parts
        if any(seg == ".git" for seg in parts):
            continue
        total += 1
        if p.suffix == ".py":
            py += 1
    return total, py


def _bench_one(target: Target, *, keep: bool = False) -> BenchRow:
    """Clone + agnostic-scan one target."""
    from augmentum.bug_finder.agnostic_stage import run_agnostic_stage

    row = BenchRow(slug=target.slug, git_url=target.git_url, ok=False)
    cell = tempfile.mkdtemp(prefix=f"bench_{target.slug}_")
    clone_dir = Path(cell) / "repo"

    print(f"\n== {target.slug}")
    print(f"   {target.git_url}")
    if target.notes:
        print(f"   {target.notes}")

    try:
        # 1. Clone
        t0 = time.monotonic()
        ok, err = _shallow_clone(target.git_url, clone_dir, target.branch)
        row.clone_seconds = round(time.monotonic() - t0, 2)
        if not ok:
            row.error = err
            print(f"   FAIL: {err}")
            return row

        row.branch = target.branch or "default"
        row.commit = _head_commit(clone_dir)
        row.file_count, row.py_count = _count_files(clone_dir)
        print(f"   cloned in {row.clone_seconds}s "
              f"({row.file_count} files, {row.py_count} py) @ {row.commit}")

        # 2. Agnostic scan — Bandit + Ruff + custom checks (none on a
        # fresh clone) + workspace suppressions (none) + pattern memory
        # (write fresh).
        t0 = time.monotonic()
        result = run_agnostic_stage(
            clone_dir,
            record_patterns=True,   # build pattern memory for this codebase
        )
        row.scan_seconds = round(time.monotonic() - t0, 2)

        # 3. Pack metrics
        row.total_raw = result.total_raw
        row.seeded_into_pipeline = len(result.seeded_findings)
        row.suppressed = result.suppressed_count
        row.scanner_counts = dict(result.scanner_counts)

        sev_ct: Counter[str] = Counter()
        file_ct: Counter[str] = Counter()
        for f in result.seeded_findings:
            sev_ct[f.severity] += 1
            file_ct[f.file] += 1
        row.severity_counts = dict(sev_ct)
        row.most_flagged_files = file_ct.most_common(5)
        row.top_patterns = sorted(
            result.pattern_signatures.items(),
            key=lambda kv: -kv[1],
        )[:10]

        row.ok = True
        print(f"   raw={row.total_raw} "
              f"(scanners={', '.join(f'{k}={v}' for k, v in row.scanner_counts.items())})")
        sev_str = ", ".join(
            f"{k}={v}" for k, v in sorted(row.severity_counts.items())
        )
        print(f"   seeded={row.seeded_into_pipeline} ({sev_str})")
        print(f"   wallclock={row.scan_seconds}s")
        if row.top_patterns:
            print(f"   top patterns:")
            for sig, count in row.top_patterns[:5]:
                print(f"     {count:>5d}  {sig}")

    except Exception as exc:   # noqa: BLE001 — bench must not crash on one project
        row.error = f"{type(exc).__name__}: {exc}"
        print(f"   ERROR: {row.error}")
    finally:
        row.total_seconds = round(row.clone_seconds + row.scan_seconds, 2)
        if not keep:
            try:
                shutil.rmtree(cell, ignore_errors=True)
            except OSError:
                pass
        else:
            print(f"   clone kept at {clone_dir}")

    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _write_jsonl(out: Path, rows: list[BenchRow]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        for r in rows:
            fp.write(r.to_jsonl() + "\n")


def _format_md_report(rows: list[BenchRow], started_at: str) -> str:
    lines = [
        f"# Bug-finder agnostic-substrate baseline — {started_at}",
        "",
        "Generated by `scripts/baseline_benchmark.py`. One row per project, ",
        "clones at shallow depth then runs `run_agnostic_stage` (Bandit + Ruff + ",
        "any custom checks + workspace suppressions).",
        "",
        "## Summary",
        "",
        "| Project | files | py | raw | seeded | high | med | low | sec | sup | scan(s) |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        if not r.ok:
            lines.append(
                f"| {r.slug} | – | – | – | – | – | – | – | – | – | "
                f"_FAIL: {r.error[:50]}_ |"
            )
            continue
        sev = r.severity_counts
        ruff = r.scanner_counts.get("ruff", 0)
        bandit = r.scanner_counts.get("bandit", 0)
        lines.append(
            f"| {r.slug} | {r.file_count} | {r.py_count} | "
            f"{r.total_raw} | {r.seeded_into_pipeline} | "
            f"{sev.get('high', 0)} | {sev.get('medium', 0)} | "
            f"{sev.get('low', 0)} | bandit={bandit}/ruff={ruff} | "
            f"{r.suppressed} | {r.scan_seconds} |"
        )
    lines += ["", "## Per-project top patterns", ""]
    for r in rows:
        if not r.ok or not r.top_patterns:
            continue
        lines.append(f"### {r.slug} @ {r.commit}")
        lines.append("")
        for sig, cnt in r.top_patterns:
            lines.append(f"- `{sig}` × **{cnt}**")
        lines.append("")
        if r.most_flagged_files:
            lines.append(f"_Most-flagged files (medium+):_")
            for path, cnt in r.most_flagged_files:
                lines.append(f"  - `{path}` × {cnt}")
            lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-codebase baseline for the bug-finder substrate"
    )
    parser.add_argument(
        "--only",
        help="comma-separated slugs to run (default: all)",
        default="",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="skip django/celery/scrapy (saves ~10min on slow disks)",
    )
    parser.add_argument(
        "--keep-clones",
        action="store_true",
        help="don't delete the clones (useful for follow-up inspection)",
    )
    parser.add_argument(
        "--out-dir",
        default=".augmentum-bench",
        help="output dir for the JSONL + markdown report",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    targets = [
        t for t in _TARGETS
        if (not only or t.slug in only)
        and (not args.skip_large or not t.is_large)
    ]
    if not targets:
        print("no targets selected", file=sys.stderr)
        return 2

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    print(f"baseline run starting at {started_at}Z")
    print(f"targets: {', '.join(t.slug for t in targets)}")
    print()

    rows: list[BenchRow] = []
    run_t0 = time.monotonic()
    for target in targets:
        rows.append(_bench_one(target, keep=args.keep_clones))
    total_elapsed = time.monotonic() - run_t0

    # Persist
    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / f"baseline_{started_at}.jsonl"
    md_path = out_dir / f"baseline_{started_at}.md"
    _write_jsonl(jsonl_path, rows)
    md_path.write_text(_format_md_report(rows, started_at), encoding="utf-8")

    # Console summary
    print(f"\n=== bench complete in {total_elapsed:.1f}s ===")
    ok_count = sum(1 for r in rows if r.ok)
    print(f"projects scanned: {ok_count}/{len(rows)}")
    print(f"jsonl: {jsonl_path}")
    print(f"md:    {md_path}")
    print()
    print("totals:")
    total_raw = sum(r.total_raw for r in rows if r.ok)
    total_seeded = sum(r.seeded_into_pipeline for r in rows if r.ok)
    print(f"  raw findings: {total_raw}")
    print(f"  seeded (medium+): {total_seeded}")
    return 0 if ok_count == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
