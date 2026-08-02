"""Self-edit contract gate — a DIFFERENTIAL no-regression Verifier.

The strongest no-regression signal a self-edit can get, and the one the API-
testing literature says is the only real false-positive killer (Godefroid et
al., ISSTA 2020: the sole way to eliminate the backwards-compat / mocked-service
false alarm is a *dynamic differential run*, not a static diff).

Mechanism: probe every GET route on the base_ref, record its break-set; probe
the same routes on the candidate worktree; FAIL only if the edit INTRODUCED a
route break (crash / authz-flip) absent from the base. The in-process mock
noise is identical on both sides, so the differential cancels it — exactly the
noise our flat per-run baseline can't distinguish, gone for free.

Follows ``bootsmoke.py``'s shape: runs ``python -m augmentum.contracts.probe``
as a subprocess against a target dir (so it probes THAT worktree's code, not the
already-imported running server), SKIPs (never FAILs) on infra trouble, and
carries ``confirms_intent=False`` — a green gate proves the edit didn't break
routes, not that it did what was asked.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import tempfile
from pathlib import Path

from augmentum.selfedit.verifier import (
    FAIL,
    ORACLE_MECHANICAL,
    PASS,
    SKIP,
    Verifier,
    VerifierResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_NAME = "contract_regression"


async def _run_probe(target_dir: str, extra_args: list[str], *, timeout: float = 420.0) -> bool:
    """Run the contract probe as a subprocess in ``target_dir`` (so it imports
    that worktree's code). Secret-scrubbed env like boot-smoke — this exercises
    candidate code we're about to judge, so it must not carry the app's keys.
    Returns True on a clean exit path (probe ran), False on infra failure."""
    from augmentum.selfedit.sandbox import scrubbed_env

    argv = [sys.executable, "-m", "augmentum.contracts.probe", "--quiet", *extra_args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=target_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=scrubbed_env(),
        )
    except Exception as exc:  # noqa: BLE001 — couldn't launch → infra, not a regression
        log.info("contract_probe_launch_failed", target=target_dir, error=repr(exc))
        return False
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        log.info("contract_probe_timed_out", target=target_dir, timeout=timeout)
        return False
    # Exit code 1 (new breaks) is a valid outcome, not an infra failure; only a
    # crash-with-no-output (2 = harness couldn't run) is treated as unavailable.
    if proc.returncode == 2:
        tail = (out or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        log.info("contract_probe_errored", target=target_dir, tail=tail)
        return False
    return True


def _write_failure_report(cand_dir: str, fresh: list[dict]) -> str:
    """Write a rich, tool-navigable diagnostic artifact into the candidate
    worktree so the self-heal agent can Read the FULL tracebacks + loci + source
    — the way coder mode surfaces command output and the bug-finder writes repros
    — instead of working from a truncated prompt line. ``.augmentum/*`` is
    gitignored, so it never pollutes the commit. Returns the worktree-relative
    path (or "" on failure — the report is a bonus, never gates)."""
    rel = ".augmentum/contract_failures.md"
    try:
        path = Path(cand_dir) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Contract regression — your edit broke {len(fresh)} route(s)",
            "",
            "These GET routes responded normally on the base revision but FAIL after "
            "your edit — your change is the cause. For EACH one: open the file at its "
            "location, use the traceback below to find why it now errors, and fix it so "
            "the route responds non-5xx again. Edit only the file(s) you already "
            "changed; do not start over or add unrelated work. When you stop, the exact "
            "same route probe re-runs automatically to check your fix.",
            "",
        ]
        for i, f in enumerate(fresh, 1):
            lines += [
                f"## {i}. {f.get('route')}  [{f.get('mode')}]",
                "",
                f"- handler: `{f.get('handler') or '?'}`",
                f"- location: `{f.get('locus') or '?'}`",
            ]
            if f.get("source"):
                lines.append(f"- offending line: `{f.get('source')}`")
            exc = f.get("exception") or (f"HTTP {f.get('status')}" if f.get("status") else "")
            if exc:
                lines.append(f"- error: `{exc}`")
            if f.get("note"):
                lines.append(f"- note: {f.get('note')}")
            tb = (f.get("traceback") or "").strip()
            if tb:
                lines += ["", "Traceback:", "", "```", tb, "```"]
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return rel
    except Exception as exc:  # noqa: BLE001 — bonus artifact, never fail the gate on it
        log.info("contract_report_write_failed", cand=cand_dir, error=repr(exc))
        return ""


def contract_regression_verifier(
    *, base_dir: str, required: bool = True, cost: int = 8,
) -> Verifier:
    """Build the differential contract Verifier for a self-edit.

    ``base_dir`` is the pre-edit worktree (the base_ref checkout / repo_dir).
    Its break-set is probed once and cached, then every candidate is diffed
    against it. Gates on NEW crashes + authz-flips; a new *hang* is recorded in
    the baseline set but not gated (timing-variant, not a reliable signal).
    """
    _baseline_cache: dict[str, str | None] = {}

    async def _baseline_for(base: str) -> str | None:
        if base in _baseline_cache:
            return _baseline_cache[base]
        bpath = Path(tempfile.gettempdir()) / f"contract_base_{abs(hash(base)) & 0xffffffff:x}.json"
        ok = await _run_probe(base, [f"--update-baseline={bpath}"])
        result = str(bpath) if (ok and bpath.exists()) else None
        _baseline_cache[base] = result
        return result

    async def _run(ctx: dict) -> VerifierResult:
        cand = ctx.get("candidate_dir") or "."
        try:
            baseline = await _baseline_for(base_dir)
            if baseline is None:
                return VerifierResult(
                    _NAME, ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                    required=required, detail="base_ref probe unavailable — no differential",
                )
            out = Path(tempfile.gettempdir()) / "contract_candidate.json"
            with contextlib.suppress(OSError):
                out.unlink()
            ok = await _run_probe(cand, [f"--out={out}", f"--baseline={baseline}"])
            if not ok or not out.exists():
                return VerifierResult(
                    _NAME, ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                    required=required, detail="candidate probe did not produce a result",
                )
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — measurement failed → skip, don't false-fail
            return VerifierResult(
                _NAME, ORACLE_MECHANICAL, SKIP, confirms_intent=False,
                required=required, detail=f"contract gate unavailable: {exc!r}",
            )

        new_reg = int(data.get("new_regression", 0))
        new_block = int(data.get("new_hard_block", 0))
        probed = int(data.get("get_probed", 0))
        if new_reg or new_block:
            fresh = [
                f for f in data.get("findings", [])
                if f.get("new") and f.get("severity") in ("regression", "hard_block")
            ]
            # Frame each break as a coder-mode repair pointer: which route, the
            # handler + exact file:line, the offending source line, and what blew
            # up — enough for the self-heal agent to navigate straight there,
            # diagnose with its tools, and make it answer again. (This whole
            # string is what _repair_context hands the model; keep it dense.)
            def _one(f: dict) -> str:
                bits = [f"{f['route']} [{f['mode']}]"]
                if f.get("handler"):
                    bits.append(f"handler {f['handler']}")
                if f.get("locus"):
                    bits.append(f"at {f['locus']}")
                if f.get("source"):
                    bits.append(f"`{str(f['source'])[:70]}`")
                exc = f.get("exception") or (f"status {f['status']}" if f.get("status") else "")
                if exc:
                    bits.append(f"-> {str(exc)[:120]}")
                return " ".join(bits)

            # Rich artifact in the worktree (full tracebacks) + a dense inline
            # pointer, so the self-heal agent has everything coder mode's failure
            # loop would give it.
            report = _write_failure_report(cand, fresh)
            pointer = (
                f" Full diagnostics (tracebacks + loci) for all {len(fresh)} are in "
                f"`{report}` - READ THAT FILE FIRST, then fix each."
                if report else ""
            )
            shown = fresh[:3]
            more = f" (+{len(fresh) - len(shown)} more, all in the report)" \
                if len(fresh) > len(shown) else ""
            summary = " || ".join(_one(f) for f in shown)
            return VerifierResult(
                _NAME, ORACLE_MECHANICAL, FAIL, confirms_intent=False, score=0.0,
                required=required,
                detail=(
                    f"your edit introduced {new_reg + new_block} NEW route break(s) - "
                    f"go to each locus, diagnose the cause, and fix it so the route "
                    f"answers (non-5xx) again.{pointer} Breaks: {summary}{more}"
                ),
            )
        return VerifierResult(
            _NAME, ORACLE_MECHANICAL, PASS, confirms_intent=False, score=1.0,
            required=required, detail=f"no new route breaks ({probed} GET probed vs base_ref)",
        )

    return Verifier(
        _NAME, ORACLE_MECHANICAL, _run, ("*",), confirms_intent=False,
        cost=cost, required=required,
    )
