"""Bug-finder automated test bench.

Drives a full bug-finder run against a workspace via the live API,
polls to terminal state, and emits a structured scorecard. Designed
for two workflows:

  * **Interactive smoke** — run by hand from a dev shell to validate
    that an end-to-end run completes and surfaces sane output. Useful
    for iterating on the lead agent's decisions.
  * **Repeatable regression** — when given expected findings via a
    JSON fixture, compares actual findings against expected and scores
    precision / recall / FP-bait. Mirrors the eval_harness shape.

Usage
-----

Minimal — explore-mode run, no expectations:

    python scripts/bug_finder_test_bench.py \\
        --workspace-id WS_ID \\
        --model gpt-5.5@oai

Named-bug run (exercises the lead loop):

    python scripts/bug_finder_test_bench.py \\
        --workspace-id WS_ID \\
        --model deepseek-v4-pro \\
        --goal-mode named-bug \\
        --goal-desc "possible auth bypass when bot accounts are deleted" \\
        --goal-repro "DELETE /api/users/{id} with a bot user; check auth_sessions"

With fixture (regression):

    python scripts/bug_finder_test_bench.py \\
        --workspace-id WS_ID \\
        --model claude-opus-4-7 \\
        --fixture tests/bug_finder_bench_fixtures/augmentum_baseline.json

Env vars (or auto-detected from ~/.claude/review-config.json):

    AUGMENTUM_ENDPOINT   default https://localhost:6443
    AUGMENTUM_API_KEY    bearer token (sk-aug-*)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ssl


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


REVIEW_CONFIG_PATH = (
    Path.home() / ".claude" / "review-config.json"
)


def _load_config_from_disk() -> tuple[str, str]:
    """Reuse the review-hook config (same endpoint + key)."""
    if not REVIEW_CONFIG_PATH.exists():
        return "", ""
    try:
        data = json.loads(REVIEW_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    return (data.get("endpoint") or "").rstrip("/"), data.get("apiKey") or ""


def _resolve_endpoint_and_key(args) -> tuple[str, str]:
    endpoint = (
        args.endpoint
        or os.environ.get("AUGMENTUM_ENDPOINT", "")
        or ""
    )
    api_key = (
        args.api_key
        or os.environ.get("AUGMENTUM_API_KEY", "")
        or ""
    )
    if not endpoint or not api_key:
        disk_endpoint, disk_key = _load_config_from_disk()
        endpoint = endpoint or disk_endpoint
        api_key = api_key or disk_key
    if not endpoint:
        endpoint = "https://localhost:6443"
    if not api_key:
        print("FATAL: no API key. Pass --api-key, set AUGMENTUM_API_KEY, "
              "or have ~/.claude/review-config.json present.",
              file=sys.stderr)
        sys.exit(2)
    return endpoint.rstrip("/"), api_key


# ---------------------------------------------------------------------------
# HTTP — minimal, no third-party deps
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
    """Send a JSON request. Returns ``(status, parsed_body_or_None)``."""
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
        return status, (json.loads(raw.decode("utf-8") or "null")
                        if raw else None)
    except ValueError:
        return status, None


def _api_path(endpoint: str, path: str) -> str:
    return endpoint + (path if path.startswith("/") else "/" + path)


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


@dataclass
class BenchRequest:
    workspace_id: str
    model: str
    verifier_model: str = ""
    focus_paths: tuple[str, ...] = ()
    max_chunks: int = 8
    detector_runs_per_chunk: int = 2
    enable_fuzz_leg: bool = False
    enable_comprehension: bool = True
    enable_pen_test_leg: bool = False
    pen_test_boot_command: str = ""
    pen_test_boot_port: int = 0
    pen_test_healthcheck_path: str = "/"
    detector_model: str = ""
    detector_temperature: float | None = None
    detector_enable_thinking: bool | None = None
    detector_preserve_thinking: bool | None = None
    run_mode: str = ""            # empty = server default ("planner")
    goal_mode: str = "explore"     # "explore" or "named-bug"
    goal_desc: str = ""
    goal_repro: str = ""
    force_below_minimum: bool = False
    poll_seconds: int = 30
    timeout_seconds: int = 1800


@dataclass
class BenchOutcome:
    run_id: str
    stop_reason: str
    stop_detail: str
    duration_seconds: float
    findings: list[dict]
    cost_ledger: list[dict]
    notes: list[str]

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == "complete"

    @property
    def total_tokens_in(self) -> int:
        return sum(int(e.get("tokens_in") or 0) for e in self.cost_ledger)

    @property
    def total_tokens_out(self) -> int:
        return sum(int(e.get("tokens_out") or 0) for e in self.cost_ledger)

    @property
    def total_wallclock_seconds(self) -> float:
        return sum(int(e.get("wallclock_ms") or 0) for e in self.cost_ledger) / 1000.0


def _kick_off_run(
    endpoint: str, api_key: str, req: BenchRequest,
) -> str:
    payload: dict[str, Any] = {
        "workspace_id": req.workspace_id,
        "primary_model": req.model,
        "verifier_model": req.verifier_model,
        "focus_paths": list(req.focus_paths),
        "max_chunks": req.max_chunks,
        "detector_runs_per_chunk": req.detector_runs_per_chunk,
        "enable_fuzz_leg": req.enable_fuzz_leg,
        "enable_comprehension": req.enable_comprehension,
        "enable_pen_test_leg": req.enable_pen_test_leg,
        "pen_test_boot_command": req.pen_test_boot_command,
        "pen_test_boot_port": req.pen_test_boot_port,
        "pen_test_healthcheck_path": req.pen_test_healthcheck_path,
    }
    if req.detector_model:
        payload["detector_model"] = req.detector_model
    if req.detector_temperature is not None:
        payload["detector_temperature"] = req.detector_temperature
    if req.detector_enable_thinking is not None:
        payload["detector_enable_thinking"] = req.detector_enable_thinking
    if req.detector_preserve_thinking is not None:
        payload["detector_preserve_thinking"] = req.detector_preserve_thinking
    if req.run_mode:
        payload["run_mode"] = req.run_mode
    if req.goal_desc:
        payload["user_goal"] = {
            "mode": req.goal_mode,
            "description": req.goal_desc,
            "repro_hint": req.goal_repro,
        }
    if req.force_below_minimum:
        payload["force_below_minimum"] = True

    status, body = _request(
        "POST", _api_path(endpoint, "/api/bug-finder/runs"),
        headers={"Authorization": f"Bearer {api_key}"},
        body=payload, timeout=30.0,
    )
    if status != 200 or not body or not body.get("run_id"):
        raise SystemExit(
            f"FATAL: kick-off failed (status={status}, body={body})"
        )
    return str(body["run_id"])


def _poll_run(
    endpoint: str, api_key: str, run_id: str,
    poll_seconds: int = 30, timeout_seconds: int = 1800,
) -> BenchOutcome:
    started = time.monotonic()
    last_print = 0.0
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise SystemExit(
                f"FATAL: run {run_id} did not terminate within "
                f"{timeout_seconds}s",
            )
        status, body = _request(
            "GET",
            _api_path(endpoint, f"/api/bug-finder/runs/{run_id}"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        if status != 200 or not body:
            print(f"  [poll] status={status} body=None; retrying...")
            time.sleep(poll_seconds)
            continue
        stop_reason = body.get("stop_reason") or "in-progress"
        report = body.get("report") or {}
        findings_count = len(report.get("findings", []))
        ledger_count = len(report.get("cost_ledger", []))
        if elapsed - last_print >= 5:
            print(f"  [{int(elapsed):3d}s] status={stop_reason} "
                  f"ledger={ledger_count} findings={findings_count}")
            last_print = elapsed
        if stop_reason in {"complete", "wallclock", "error", "cancelled"}:
            return BenchOutcome(
                run_id=run_id,
                stop_reason=stop_reason,
                stop_detail=body.get("stop_detail") or "",
                duration_seconds=elapsed,
                findings=list(report.get("findings", [])),
                cost_ledger=list(report.get("cost_ledger", [])),
                notes=list(report.get("notes", [])),
            )
        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Fixture-based scoring (when provided)
# ---------------------------------------------------------------------------


@dataclass
class ExpectedFinding:
    """One expected finding to look for in the bench output.

    Match policy: a finding matches when its ``file`` contains the
    fixture's ``file`` substring (so workspace path prefixes don't
    foil matches) AND its ``claim_signature`` matches OR the fixture
    didn't pin one.
    """

    file: str
    claim_signature: str = ""
    function: str = ""
    severity_at_least: str = ""    # any | low | medium | high | critical
    note: str = ""


@dataclass
class FixtureScore:
    expected_total: int
    matched: int
    unmatched: list[ExpectedFinding] = field(default_factory=list)
    extra_findings: list[dict] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return (self.matched / self.expected_total) if self.expected_total else 0.0


_SEV_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _sev_ok(finding_sev: str, floor: str) -> bool:
    if not floor:
        return True
    return _SEV_RANK.get(finding_sev, 0) >= _SEV_RANK.get(floor, 0)


def _load_fixture(path: Path) -> list[ExpectedFinding]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_raw = raw.get("expected_findings") or raw if isinstance(raw, list) else []
    out: list[ExpectedFinding] = []
    for e in expected_raw:
        if not isinstance(e, dict) or not e.get("file"):
            continue
        out.append(ExpectedFinding(
            file=str(e["file"]),
            claim_signature=str(e.get("claim_signature") or ""),
            function=str(e.get("function") or ""),
            severity_at_least=str(e.get("severity_at_least") or ""),
            note=str(e.get("note") or ""),
        ))
    return out


def _score_against_fixture(
    findings: list[dict], expected: list[ExpectedFinding],
) -> FixtureScore:
    """Match each expected against the actual findings. Greedy first-match."""
    if not expected:
        return FixtureScore(expected_total=0, matched=0)
    consumed: set[int] = set()
    matched = 0
    unmatched: list[ExpectedFinding] = []
    for e in expected:
        hit = False
        for i, f in enumerate(findings):
            if i in consumed:
                continue
            if e.file not in str(f.get("file") or ""):
                continue
            if e.claim_signature and e.claim_signature != (f.get("claim_signature") or ""):
                continue
            if e.function and e.function != (f.get("function") or ""):
                continue
            if not _sev_ok(str(f.get("severity") or ""), e.severity_at_least):
                continue
            consumed.add(i)
            matched += 1
            hit = True
            break
        if not hit:
            unmatched.append(e)
    extras = [
        f for i, f in enumerate(findings) if i not in consumed
    ]
    return FixtureScore(
        expected_total=len(expected),
        matched=matched, unmatched=unmatched,
        extra_findings=extras,
    )


# ---------------------------------------------------------------------------
# Scorecard rendering
# ---------------------------------------------------------------------------


def _print_scorecard(
    req: BenchRequest, outcome: BenchOutcome,
    score: FixtureScore | None,
) -> None:
    print("\n" + "=" * 70)
    print(f"  BUG FINDER TEST BENCH — {outcome.run_id}")
    print("=" * 70)
    print(f"\n  Workspace        {req.workspace_id}")
    print(f"  Model            {req.model}")
    print(f"  Goal             {req.goal_mode}"
          + (f" — \"{req.goal_desc[:60]}\"" if req.goal_desc else ""))
    print(f"  Comprehension    {'on' if req.enable_comprehension else 'off'}")
    print(f"  Fuzz leg         {'on' if req.enable_fuzz_leg else 'off'}")
    print(f"  Pen-test leg     {'on' if req.enable_pen_test_leg else 'off'}"
          + (f" (boot={req.pen_test_boot_command!r}"
             f" port={req.pen_test_boot_port}"
             f" health={req.pen_test_healthcheck_path!r})"
             if req.enable_pen_test_leg else ""))

    print(f"\n  Stop reason      {outcome.stop_reason}"
          + (f" — {outcome.stop_detail}" if outcome.stop_detail else ""))
    print(f"  Duration         {outcome.duration_seconds:.1f}s "
          f"(LLM wallclock: {outcome.total_wallclock_seconds:.1f}s)")
    print(f"  Tokens           in={outcome.total_tokens_in:,}  "
          f"out={outcome.total_tokens_out:,}")
    print(f"  Findings         {len(outcome.findings)} total")

    if outcome.notes:
        print(f"\n  Notes")
        for n in outcome.notes:
            print(f"    · {n}")

    if outcome.cost_ledger:
        print(f"\n  Stage breakdown")
        # Group ledger entries by stage
        by_stage: dict[str, list[dict]] = {}
        for e in outcome.cost_ledger:
            by_stage.setdefault(str(e.get("stage") or "?"), []).append(e)
        for stage, entries in by_stage.items():
            iters = sum(int(e.get("iterations") or 0) for e in entries)
            tin = sum(int(e.get("tokens_in") or 0) for e in entries)
            tout = sum(int(e.get("tokens_out") or 0) for e in entries)
            ms = sum(int(e.get("wallclock_ms") or 0) for e in entries)
            print(f"    {stage:20s} ×{len(entries):2d} iters={iters:3d} "
                  f"tok_in={tin:8,} tok_out={tout:7,} "
                  f"wall={ms/1000:7.1f}s")

    if outcome.findings:
        print(f"\n  Top findings (severity-sorted, up to 10)")
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_findings = sorted(
            outcome.findings,
            key=lambda f: (
                sev_order.get(str(f.get("severity") or "info"), 4),
                -int(f.get("runs_to_confirm") or 0),
            ),
        )
        for f in sorted_findings[:10]:
            sev = str(f.get("severity") or "?").upper()
            status = str(f.get("status") or "?").upper()
            sig = str(f.get("claim_signature") or "?")
            claim = str(f.get("claim") or "").replace("\n", " ")[:120]
            file = str(f.get("file") or "?")
            print(f"    [{sev:<8s}/{status:<13s}] {sig:<18s}  {file}")
            print(f"      {claim}")

    if score is not None:
        print("\n  Fixture score")
        print(f"    Expected     {score.expected_total}")
        print(f"    Matched      {score.matched}")
        print(f"    Recall       {score.recall*100:.0f}%")
        print(f"    Extras       {len(score.extra_findings)}")
        if score.unmatched:
            print(f"\n    Unmatched expectations:")
            for e in score.unmatched[:5]:
                pin = ""
                if e.claim_signature:
                    pin += f" sig={e.claim_signature}"
                if e.function:
                    pin += f" fn={e.function}"
                if e.severity_at_least:
                    pin += f" sev>={e.severity_at_least}"
                print(f"      ✗ {e.file}{pin}"
                      + (f" — {e.note}" if e.note else ""))

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-id", required=True,
                   help="Coder workspace UUID to audit")
    p.add_argument("--model", required=True,
                   help="Primary model id (planner/detector/fixer/lead)")
    p.add_argument("--verifier-model", default="",
                   help="Optional separate verifier model")
    p.add_argument("--focus-paths", default="",
                   help="Comma-separated list of focus paths")
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--detector-runs-per-chunk", type=int, default=2)
    p.add_argument("--enable-fuzz-leg", action="store_true")
    p.add_argument("--disable-comprehension", action="store_true",
                   help="Skip the comprehension stage (faster, but loses "
                        "compounding benefit)")
    p.add_argument("--enable-pen-test-leg", action="store_true",
                   help="Run the dynamic pen-test leg on confirmed findings. "
                        "Boots the workspace's app and runs active HTTP "
                        "probes; off by default.")
    p.add_argument("--pen-test-boot-command", default="",
                   help="Hint for the pen_tester's boot_under_test tool "
                        "(e.g. 'python -m augmentum.proxy.server'). Empty "
                        "= let the subagent discover it.")
    p.add_argument("--pen-test-boot-port", type=int, default=0,
                   help="Port hint for the under-test app (0 = subagent picks)")
    p.add_argument("--pen-test-healthcheck-path", default="/",
                   help="Healthcheck path for boot verification (default: /)")
    p.add_argument("--detector-model", default="",
                   help="Override the detector role's model id (e.g. "
                        "'Qwen3.6-35B-A3B-IQ4_XS'). Empty = use --model. "
                        "Useful for testing local-model detectors against a "
                        "cloud verifier.")
    p.add_argument("--detector-temperature", type=float, default=None,
                   help="Detector sampling temperature (0.0..2.0). Defaults "
                        "to the config field (0.0 — the determinism floor). "
                        "Raise to >0 only for variance / thinking-mode "
                        "experiments where the recall lift is worth the FPs.")
    p.add_argument("--detector-thinking", dest="detector_enable_thinking",
                   action="store_const", const=True, default=None,
                   help="Enable chain-of-thought reasoning at the detector "
                        "(forwarded as ``chat_template_kwargs.enable_thinking="
                        "true``). Qwen 3.x / GLM-4.x / EXAONE 4.x / "
                        "Nemotron 3 Nano consume this kwarg.")
    p.add_argument("--detector-no-thinking", dest="detector_enable_thinking",
                   action="store_const", const=False,
                   help="Explicitly disable thinking at the detector. "
                        "Overrides the model's template default.")
    p.add_argument("--detector-preserve-thinking",
                   dest="detector_preserve_thinking",
                   action="store_const", const=True, default=None,
                   help="Keep ``<think>`` traces across multi-turn detector "
                        "history (Qwen 3.6 ``preserve_thinking`` template "
                        "kwarg). Other templates ignore this.")
    p.add_argument("--mode", dest="run_mode", default="",
                   choices=("", "planner", "static_chunk"),
                   help="Chunk-selection mode. Default ('' = server default "
                        "= planner) uses an LLM to curate chunks. "
                        "'static_chunk' walks the workspace via AST and "
                        "emits one chunk per function — Mythos-style "
                        "exhaustive sweep, bypasses the planner's token "
                        "budget cliff at broad scope.")
    p.add_argument("--goal-mode", default="explore",
                   choices=("explore", "named-bug"))
    p.add_argument("--goal-desc", default="")
    p.add_argument("--goal-repro", default="")
    p.add_argument("--fixture", default="",
                   help="Path to a JSON fixture with expected_findings "
                        "for regression scoring")
    p.add_argument("--force-below-minimum", action="store_true")
    p.add_argument("--endpoint", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    args = p.parse_args()

    endpoint, api_key = _resolve_endpoint_and_key(args)

    focus_paths = tuple(
        p.strip() for p in (args.focus_paths or "").split(",")
        if p.strip()
    )
    req = BenchRequest(
        workspace_id=args.workspace_id,
        model=args.model,
        verifier_model=args.verifier_model,
        focus_paths=focus_paths,
        max_chunks=args.max_chunks,
        detector_runs_per_chunk=args.detector_runs_per_chunk,
        enable_fuzz_leg=args.enable_fuzz_leg,
        enable_comprehension=not args.disable_comprehension,
        enable_pen_test_leg=args.enable_pen_test_leg,
        pen_test_boot_command=args.pen_test_boot_command,
        pen_test_boot_port=args.pen_test_boot_port,
        pen_test_healthcheck_path=args.pen_test_healthcheck_path,
        detector_model=args.detector_model,
        detector_temperature=args.detector_temperature,
        detector_enable_thinking=args.detector_enable_thinking,
        detector_preserve_thinking=args.detector_preserve_thinking,
        run_mode=args.run_mode,
        goal_mode=args.goal_mode,
        goal_desc=args.goal_desc,
        goal_repro=args.goal_repro,
        force_below_minimum=args.force_below_minimum,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    expected: list[ExpectedFinding] = []
    if args.fixture:
        fixture_path = Path(args.fixture)
        if not fixture_path.is_file():
            print(f"FATAL: fixture not found: {fixture_path}", file=sys.stderr)
            return 2
        expected = _load_fixture(fixture_path)

    print(f"== Kicking off run against {endpoint}")
    print(f"   workspace={req.workspace_id} model={req.model}")
    print(f"   goal_mode={req.goal_mode}"
          + (f" — \"{req.goal_desc[:80]}\"" if req.goal_desc else ""))
    run_id = _kick_off_run(endpoint, api_key, req)
    print(f"   run_id={run_id}")

    print(f"\n== Polling (every {req.poll_seconds}s, "
          f"timeout {req.timeout_seconds}s)")
    outcome = _poll_run(
        endpoint, api_key, run_id,
        poll_seconds=req.poll_seconds,
        timeout_seconds=req.timeout_seconds,
    )

    score = _score_against_fixture(outcome.findings, expected) if expected else None
    _print_scorecard(req, outcome, score)

    # Exit code policy: 0 on clean complete + (if fixture) full recall.
    if not outcome.succeeded:
        return 1
    if score is not None and score.recall < 1.0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
