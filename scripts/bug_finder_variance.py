"""Variance benchmark for the bug_finder pipeline.

Runs the SAME audit configuration N times against the SAME target,
optionally wiping the workspace substrate between runs to isolate
LLM-level variance from between-run pattern-memory feedback.

The goal is a single number: how often does the bug_finder give a
calling subagent the same answer for the same question?

Usage
-----

    python scripts/bug_finder_variance.py \\
        --workspace-id <WS_UUID> \\
        --runs 5 \\
        --model claude-sonnet-4-6 \\
        --verifier-model gpt-5.4 \\
        --enable-pen-test-leg \\
        --pen-test-boot-command "python -m tests.fixtures.vuln_app.app --port 8765" \\
        --pen-test-boot-port 8765 \\
        --pen-test-healthcheck-path "/" \\
        --wipe-substrate

When ``--wipe-substrate`` is passed, the harness clears the workspace
container's ``.augmentum/`` between each run via ``docker exec``. That
isolates the temperature-zero collapse: any remaining variance is
LLM-side (token-pacing, network), not substrate accumulation. Without
the flag, the measurement is "what would a subagent caller actually
see" — pattern memory + knowledge map accumulate as designed.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HTTP — copied/adapted from bug_finder_test_bench.py to keep this script
# standalone. The bench harness handles single-run + scoring; this one
# handles N-run aggregation.
# ---------------------------------------------------------------------------


_SSL_CONTEXT = ssl.create_default_context()
_SSL_CONTEXT.check_hostname = False
_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _request(
    method: str, url: str, *,
    headers: dict[str, str] | None = None,
    body: dict | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict | None]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(
            req, timeout=timeout, context=_SSL_CONTEXT,
        ) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        return status, (
            json.loads(raw.decode("utf-8") or "null") if raw else None
        )
    except ValueError:
        return status, None


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    run_index: int
    run_id: str
    stop_reason: str
    findings: list[dict]
    cost_ledger: list[dict]
    notes: list[str]
    wallclock_s: float

    @property
    def total_tokens(self) -> int:
        return sum(
            int(e.get("tokens_in") or 0) + int(e.get("tokens_out") or 0)
            for e in self.cost_ledger
        )

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def confirmed_count(self) -> int:
        return sum(
            1 for f in self.findings
            if str(f.get("status") or "").lower() == "confirmed"
        )

    def claim_signatures(self) -> tuple[str, ...]:
        return tuple(
            str(f.get("claim_signature") or "?")
            for f in self.findings
        )


def _kick_off_run(args, run_index: int) -> str:
    payload: dict[str, Any] = {
        "workspace_id": args.workspace_id,
        "primary_model": args.model,
        "verifier_model": args.verifier_model,
        "focus_paths": [],
        "max_chunks": args.max_chunks,
        "detector_runs_per_chunk": args.detector_runs_per_chunk,
        "enable_fuzz_leg": False,
        "enable_comprehension": True,
        "enable_pen_test_leg": args.enable_pen_test_leg,
    }
    if args.enable_pen_test_leg:
        payload["pen_test_boot_command"] = args.pen_test_boot_command
        payload["pen_test_boot_port"] = args.pen_test_boot_port
        payload["pen_test_healthcheck_path"] = args.pen_test_healthcheck_path
    status, body = _request(
        "POST", args.endpoint.rstrip("/") + "/api/bug-finder/runs",
        headers={"Authorization": f"Bearer {args.api_key}"},
        body=payload, timeout=30.0,
    )
    if status != 200 or not body or not body.get("run_id"):
        raise SystemExit(
            f"FATAL: kick-off failed for run #{run_index} "
            f"(status={status}, body={body})",
        )
    return str(body["run_id"])


def _poll_to_terminal(args, run_id: str, run_index: int) -> dict:
    started = time.monotonic()
    last_print = 0.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > args.timeout_seconds:
            raise SystemExit(
                f"FATAL: run #{run_index} ({run_id}) did not terminate "
                f"within {args.timeout_seconds}s",
            )
        try:
            status, body = _request(
                "GET",
                args.endpoint.rstrip("/")
                + f"/api/bug-finder/runs/{run_id}",
                headers={"Authorization": f"Bearer {args.api_key}"},
                timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [poll] transient error: {exc}; retrying...")
            time.sleep(args.poll_seconds)
            continue
        if status != 200 or not body:
            time.sleep(args.poll_seconds)
            continue
        sr = body.get("stop_reason") or "in-progress"
        report = body.get("report") or {}
        if elapsed - last_print >= 10:
            print(
                f"  [#{run_index} {int(elapsed):3d}s] "
                f"status={sr} "
                f"ledger={len(report.get('cost_ledger', []))} "
                f"findings={len(report.get('findings', []))}",
            )
            last_print = elapsed
        if sr in {"complete", "wallclock", "error", "cancelled"}:
            return body
        time.sleep(args.poll_seconds)


# ---------------------------------------------------------------------------
# Substrate wipe (workspace-side)
# ---------------------------------------------------------------------------


def _wipe_workspace_substrate(workspace_container: str) -> None:
    """Reset ``/workspace/.augmentum/`` inside the workspace container.

    Removes patterns + receipts + repros so the next run starts with
    a fresh on-disk substrate. The knowledge map is server-side and
    gets wiped separately via the API; see ``_forget_knowledge_map``.
    """
    cmd = [
        "docker", "exec", workspace_container,
        "bash", "-c",
        "rm -rf //workspace/.augmentum 2>/dev/null; "
        "mkdir -p //workspace/.augmentum && "
        "echo wiped",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        # Permission policies sometimes block the rm -rf pattern.
        # Continue — the knowledge-map wipe is the load-bearing reset.
        print(
            f"  [wipe] WARN: {result.stderr.strip() or 'unknown error'} "
            f"(continuing without on-disk wipe)",
        )
    else:
        print(f"  [wipe] {result.stdout.strip()}")


def _forget_knowledge_map(args, workspace_id: str) -> None:
    """Delete the server-side cached comprehender knowledge map for
    this workspace. Forces the next run to re-comprehend from scratch
    — important for variance benchmarks because the comprehender's
    first-contact brief biases every subsequent run's planner."""
    try:
        status, body = _request(
            "DELETE",
            args.endpoint.rstrip("/")
            + f"/api/bug-finder/knowledge/{workspace_id}",
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=15.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [forget-knowledge] WARN: {exc}")
        return
    if status == 200 and body is not None:
        print(
            f"  [forget-knowledge] forgotten="
            f"{body.get('forgotten')}",
        )
    else:
        print(
            f"  [forget-knowledge] WARN: status={status}, body={body}",
        )


def _restage_vuln_app(workspace_container: str, repo_root: Path) -> None:
    """Copy the vuln_app fixture into the workspace.

    Mirrors what was done by hand during the manual real-audit
    sequence. Idempotent — overwrites existing files.

    Note: vuln_app's docstring discloses ``Every block labeled ``# VULN-N:``
    is intentionally exploitable``. At ``temperature=0`` the planner
    deterministically reads this brief and decides to skip the
    workspace ("this is a test fixture, the vulns are intentional").
    Use ``--stage-augmentum-auth`` for a real-target variance bench."""
    fixture = repo_root / "tests" / "fixtures" / "vuln_app"
    if not fixture.is_dir():
        print(f"  [stage] WARN: vuln_app fixture not found at {fixture}")
        return
    subprocess.run(
        ["docker", "exec", workspace_container,
         "bash", "-c", "mkdir -p //workspace/vuln_app"],
        capture_output=True, timeout=15,
    )
    for fname in ("__init__.py", "app.py"):
        subprocess.run(
            ["docker", "cp", str(fixture / fname),
             f"{workspace_container}:/workspace/vuln_app/{fname}"],
            capture_output=True, timeout=15,
        )


_AUGMENTUM_AUTH_FILES = (
    ("augmentum/auth/__init__.py",     "/workspace/augmentum/auth/__init__.py"),
    ("augmentum/auth/guards.py",       "/workspace/augmentum/auth/guards.py"),
    ("augmentum/auth/middleware.py",   "/workspace/augmentum/auth/middleware.py"),
    ("augmentum/auth/models.py",       "/workspace/augmentum/auth/models.py"),
    ("augmentum/auth/passwords.py",    "/workspace/augmentum/auth/passwords.py"),
    ("augmentum/auth/scoping.py",      "/workspace/augmentum/auth/scoping.py"),
    ("augmentum/auth/session_manager.py", "/workspace/augmentum/auth/session_manager.py"),
    ("augmentum/auth/api_keys.py",     "/workspace/augmentum/auth/api_keys.py"),
    ("augmentum/proxy/auth_routes.py", "/workspace/augmentum/proxy/auth_routes.py"),
)


def _restage_augmentum_auth(
    workspace_container: str, repo_root: Path,
) -> None:
    """Stage the real ``augmentum/auth/*`` + ``auth_routes.py`` slice
    into the workspace. Real production code, no test-fixture docstring
    to short-circuit the planner. Use when measuring subagent variance
    on a representative real target.

    Idempotent — overwrites existing files."""
    subprocess.run(
        ["docker", "exec", workspace_container,
         "bash", "-c",
         "mkdir -p //workspace/augmentum/auth //workspace/augmentum/proxy"],
        capture_output=True, timeout=15,
    )
    missing: list[str] = []
    for src_rel, dst in _AUGMENTUM_AUTH_FILES:
        src = repo_root / src_rel
        if not src.is_file():
            missing.append(src_rel)
            continue
        subprocess.run(
            ["docker", "cp", str(src),
             f"{workspace_container}:{dst}"],
            capture_output=True, timeout=15,
        )
    if missing:
        print(f"  [stage-auth] WARN: missing {len(missing)} files: {missing[:3]}")
    else:
        print(
            f"  [stage-auth] staged {len(_AUGMENTUM_AUTH_FILES)} files "
            f"(augmentum/auth/* + auth_routes.py)",
        )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def _summarise(results: list[RunResult]) -> dict[str, Any]:
    finding_counts = [r.finding_count for r in results]
    confirmed_counts = [r.confirmed_count for r in results]
    token_counts = [r.total_tokens for r in results]
    wallclocks = [r.wallclock_s for r in results]

    # Per-signature hit rate: of the N runs, how many hit each signature?
    sig_hits: dict[str, int] = Counter()
    for r in results:
        for sig in set(r.claim_signatures()):
            sig_hits[sig] += 1

    # Per-(file, signature) hit rate: more precise, since the same
    # finding emerging in 3 of 5 runs IS the variance number we want.
    finding_keys: list[tuple[str, str]] = []
    fingerprint_hits: dict[tuple[str, str], int] = Counter()
    for r in results:
        seen_in_run: set[tuple[str, str]] = set()
        for f in r.findings:
            key = (
                str(f.get("file") or "?"),
                str(f.get("claim_signature") or "?"),
            )
            if key in seen_in_run:
                continue
            seen_in_run.add(key)
            fingerprint_hits[key] += 1
            finding_keys.append(key)

    def _stddev_or_zero(xs: list[float]) -> float:
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    return {
        "n_runs": len(results),
        "stop_reasons": Counter(r.stop_reason for r in results),
        "finding_count": {
            "min": min(finding_counts),
            "max": max(finding_counts),
            "mean": statistics.mean(finding_counts),
            "stddev": _stddev_or_zero([float(x) for x in finding_counts]),
            "per_run": finding_counts,
        },
        "confirmed_count": {
            "min": min(confirmed_counts),
            "max": max(confirmed_counts),
            "mean": statistics.mean(confirmed_counts),
            "stddev": _stddev_or_zero([float(x) for x in confirmed_counts]),
            "per_run": confirmed_counts,
        },
        "tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": statistics.mean(token_counts),
            "stddev": _stddev_or_zero([float(x) for x in token_counts]),
            "per_run": token_counts,
        },
        "wallclock_s": {
            "min": min(wallclocks),
            "max": max(wallclocks),
            "mean": statistics.mean(wallclocks),
            "stddev": _stddev_or_zero(wallclocks),
            "per_run": wallclocks,
        },
        "signature_hit_rate": {
            sig: f"{hits}/{len(results)}"
            for sig, hits in sig_hits.most_common()
        },
        "finding_fingerprint_hit_rate": {
            f"{file} :: {sig}": f"{hits}/{len(results)}"
            for (file, sig), hits in fingerprint_hits.most_common()
        },
    }


def _format_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  BUG FINDER VARIANCE BENCHMARK")
    lines.append("=" * 72)
    lines.append(f"  Runs                {summary['n_runs']}")
    lines.append(f"  Stop reasons        {dict(summary['stop_reasons'])}")
    lines.append("")

    def _stat_line(label: str, s: dict[str, Any]) -> str:
        return (
            f"  {label:20s}  min={s['min']:>8}  max={s['max']:>8}  "
            f"mean={s['mean']:>8.1f}  stddev={s['stddev']:>6.2f}  "
            f"per_run={s['per_run']}"
        )

    lines.append(_stat_line("Findings",        summary["finding_count"]))
    lines.append(_stat_line("Confirmed",       summary["confirmed_count"]))
    lines.append(_stat_line("Tokens",          summary["tokens"]))
    lines.append(_stat_line("Wallclock (s)",   summary["wallclock_s"]))
    lines.append("")
    lines.append("  Per-finding fingerprint hit rate (file :: signature)")
    for fp, rate in summary["finding_fingerprint_hit_rate"].items():
        lines.append(f"    {rate}   {fp}")
    lines.append("")
    lines.append("  Per-signature hit rate (any file)")
    for sig, rate in summary["signature_hit_rate"].items():
        lines.append(f"    {rate}   {sig}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-id", required=True)
    p.add_argument("--workspace-container", default="",
                   help="Docker container name for the workspace. "
                        "Auto-derived as augmentum-ws-<id-prefix> if empty.")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--model", required=True)
    p.add_argument("--verifier-model", default="")
    p.add_argument("--max-chunks", type=int, default=4)
    p.add_argument("--detector-runs-per-chunk", type=int, default=2)
    p.add_argument("--enable-pen-test-leg", action="store_true")
    p.add_argument("--pen-test-boot-command", default="")
    p.add_argument("--pen-test-boot-port", type=int, default=0)
    p.add_argument("--pen-test-healthcheck-path", default="/")
    p.add_argument("--wipe-substrate", action="store_true",
                   help="Wipe workspace .augmentum/ between runs so "
                        "pattern memory + knowledge map don't bias "
                        "subsequent runs. Isolates LLM-level variance.")
    p.add_argument("--forget-knowledge-map", action="store_true",
                   help="DELETE the server-side comprehender knowledge "
                        "map for this workspace before each run. The "
                        "knowledge map biases planner chunk selection "
                        "deterministically at temperature=0 — must be "
                        "cleared for an honest variance measurement.")
    p.add_argument("--restage-vuln-app", action="store_true",
                   help="Re-copy tests/fixtures/vuln_app into the "
                        "workspace before each run. Self-discloses as "
                        "intentional fixture — planner skips at temp=0. "
                        "Prefer --stage-augmentum-auth for real targets.")
    p.add_argument("--stage-augmentum-auth", action="store_true",
                   help="Stage augmentum/auth/* + auth_routes.py into "
                        "the workspace before each run. Real production "
                        "code with no test-fixture short-circuit; good "
                        "for real subagent-variance measurement.")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--poll-seconds", type=int, default=25)
    p.add_argument("--timeout-seconds", type=int, default=1500)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    workspace_container = args.workspace_container or (
        f"augmentum-ws-{args.workspace_id.split('-')[0]}"
    )

    results: list[RunResult] = []
    for i in range(1, args.runs + 1):
        print(f"\n== Run #{i}/{args.runs} ==")
        if args.wipe_substrate:
            _wipe_workspace_substrate(workspace_container)
        if args.forget_knowledge_map:
            _forget_knowledge_map(args, args.workspace_id)
        if args.restage_vuln_app:
            _restage_vuln_app(workspace_container, repo_root)
        if args.stage_augmentum_auth:
            _restage_augmentum_auth(workspace_container, repo_root)
        started = time.monotonic()
        run_id = _kick_off_run(args, i)
        print(f"  kicked off run_id={run_id}")
        body = _poll_to_terminal(args, run_id, i)
        elapsed = time.monotonic() - started
        report = body.get("report") or {}
        results.append(RunResult(
            run_index=i,
            run_id=run_id,
            stop_reason=str(body.get("stop_reason") or "?"),
            findings=list(report.get("findings", [])),
            cost_ledger=list(report.get("cost_ledger", [])),
            notes=list(report.get("notes", [])),
            wallclock_s=round(elapsed, 1),
        ))
        latest = results[-1]
        print(
            f"  done: {latest.finding_count} findings "
            f"({latest.confirmed_count} confirmed), "
            f"{latest.total_tokens:,} tokens, {latest.wallclock_s:.1f}s",
        )

    summary = _summarise(results)
    print()
    print(_format_summary(summary))

    out_dir = repo_root / ".augmentum-bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    out = out_dir / f"variance_{ts}.json"
    payload = {
        "ts": ts,
        "config": {
            "workspace_id": args.workspace_id,
            "runs": args.runs,
            "model": args.model,
            "verifier_model": args.verifier_model,
            "max_chunks": args.max_chunks,
            "detector_runs_per_chunk": args.detector_runs_per_chunk,
            "enable_pen_test_leg": args.enable_pen_test_leg,
            "wipe_substrate": args.wipe_substrate,
            "restage_vuln_app": args.restage_vuln_app,
        },
        "summary": {
            **summary,
            "stop_reasons": dict(summary["stop_reasons"]),
        },
        "runs": [
            {
                "run_index": r.run_index,
                "run_id": r.run_id,
                "stop_reason": r.stop_reason,
                "finding_count": r.finding_count,
                "confirmed_count": r.confirmed_count,
                "total_tokens": r.total_tokens,
                "wallclock_s": r.wallclock_s,
                "findings": r.findings,
                "notes": r.notes,
            }
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nvariance report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
