"""Evidence-grounded self-edit — hand the agent the scanner's SPECIFIC findings,
then mechanically confirm it resolved one.

The diagnosis from the first live runs: a generic objective ("remove a dead CSS
rule") makes every agent — even a frontier model — burn its whole budget
RE-DISCOVERING what the scanner already computed (which class, where, and which
flagged ones are actually false positives). The fix, grounded in our research
(self-improvement works only where outcomes are *verifiable*): carry the
evidence.

Two halves, both built here:

* **Evidence → objective.** ``enrich_target`` pulls the scanner's concrete
  findings for a debt target (the specific class names + files, with the
  acknowledged-live set already excluded) and composes an objective that hands the
  agent exactly what to verify and fix — so it acts on evidence, not a hunt.

* **Evidence → a CONFIRM oracle.** ``findings_confirm_verifier`` re-runs the same
  scanner on the candidate and diffs the FINDING SET against the baseline:
  confirmed iff it's a strict subset (≥1 resolved, 0 added). That's a *mechanical*
  oracle that ``confirms_intent`` → the verdict can reach ``verified`` /
  auto-promotable, not just ``human_required``. The agent's own ``search`` filters
  the scanner's false positives; the diff makes the resolution provable.

Scanners locate the repo by walking up from THEIR OWN file (``_common.find_root``
is ``__file__``-based), so to scan an arbitrary tree we import that tree's own
copy of the scripts in a subprocess (every full worktree checkout has them). This
module is therefore dual-mode: the async API spawns the subprocess; ``--probe``
IS the subprocess. Module-level imports stay stdlib-only so the probe runs as a
plain script (no augmentum package import); scanner + verifier imports are lazy.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field

_SCRIPTS_REL = os.path.join(".claude", "skills", "augmentum-dev", "scripts")

# Metrics whose scanner exposes per-finding detail we can ground on + confirm.
# (mixed_errors / ws_gaps remain count-only → omitted; they're GATED in loop.py so
# an ungrounded target is never handed to the local agent as a blind hunt — the
# observed 40-step / zero-edit failure mode.)
EVIDENCE_METRICS: tuple[str, ...] = (
    "code_quality.dead_css",
    "code_quality.console_log",
    "code_quality.silent_catches",
    "code_quality.tech_debt",
    "dead_code.orphaned_endpoints",
    "dead_code.ghost_calls",
    "async_blocking.errors",
    "async_blocking.warnings",
    "runtime.errors",
)


@dataclass
class Finding:
    """One concrete scanner finding, normalized across scanners."""
    key: str            # stable identity for set-diffing (symbol@file[:line])
    metric: str
    symbol: str         # the thing itself (class name, route, dotted call, marker)
    file: str = ""
    line: int = 0
    detail: str = ""
    fix: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "metric": self.metric, "symbol": self.symbol,
                "file": self.file, "line": self.line, "detail": self.detail,
                "fix": self.fix}

    @staticmethod
    def from_dict(d: dict) -> Finding:
        return Finding(key=d.get("key", ""), metric=d.get("metric", ""),
                       symbol=d.get("symbol", ""), file=d.get("file", ""),
                       line=int(d.get("line", 0) or 0), detail=d.get("detail", ""),
                       fix=d.get("fix", ""))


# ===========================================================================
# Probe side — runs as a subprocess, imports the TARGET TREE's scanners.
# ===========================================================================

def _load_ack(scripts_dir: str, filename: str, key: str) -> set:
    try:
        with open(os.path.join(scripts_dir, filename), encoding="utf-8") as f:
            return set(json.load(f).get(key, []))
    except (OSError, ValueError):
        return set()


def _probe(scripts_dir: str, metric_key: str) -> list[dict]:
    """Import the tree's scanners (already on sys.path) and return findings for
    ``metric_key`` as plain dicts. Best-effort: any failure → []."""
    out: list[Finding] = []
    try:
        if metric_key == "code_quality.dead_css":
            import code_quality as q
            ack = _load_ack(scripts_dir, "quality_suppressions.json", "dead_css_acknowledged")
            _missing, dead = q.check_css_js_classes()
            for d in dead:
                cls, f = d.get("class", ""), d.get("file", "")
                if not cls or cls in ack:
                    continue
                out.append(Finding(key=f"dead_css:{cls}@{f}", metric=metric_key,
                                   symbol=cls, file=f,
                                   detail=f"CSS class '.{cls}' defined in {f} with no JS/HTML reference",
                                   fix=f"remove the '.{cls}' rule from {f} if truly unused"))
        elif metric_key == "code_quality.console_log":
            import code_quality as q
            for d in q.check_console_logs():
                f, ln, lvl = d.get("file", ""), int(d.get("line", 0) or 0), d.get("level", "log")
                out.append(Finding(key=f"console_log:{f}:{ln}", metric=metric_key,
                                   symbol=f"console.{lvl}", file=f, line=ln,
                                   detail=f"console.{lvl} left in {f}:{ln}",
                                   fix="remove the console statement (or gate it behind a debug flag)"))
        elif metric_key == "code_quality.silent_catches":
            import code_quality as q
            for d in q.check_silent_catches():
                f, ln = d.get("file", ""), int(d.get("line", 0) or 0)
                out.append(Finding(key=f"silent_catch:{f}:{ln}", metric=metric_key,
                                   symbol=f"catch@{f}:{ln}", file=f, line=ln,
                                   detail=d.get("description", "silent catch — error swallowed"),
                                   fix="log the error (console.warn / log.warning) instead of swallowing it"))
        elif metric_key == "code_quality.tech_debt":
            import code_quality as q
            for d in q.check_tech_debt():
                f, ln, mk = d.get("file", ""), int(d.get("line", 0) or 0), d.get("marker", "TODO")
                out.append(Finding(key=f"tech_debt:{f}:{ln}", metric=metric_key,
                                   symbol=f"{mk}@{f}:{ln}", file=f, line=ln,
                                   detail=f"{mk}: {d.get('text', '')}",
                                   fix="resolve the marked item, or remove the marker if obsolete"))
        elif metric_key == "dead_code.orphaned_endpoints":
            import dead_code as dc
            for d in dc.find_orphaned_endpoints():
                m, p = d.get("method", ""), d.get("path", "")
                f, ln = d.get("file", ""), int(d.get("line", 0) or 0)
                out.append(Finding(key=f"orphan:{m} {p}", metric=metric_key,
                                   symbol=f"{m} {p}", file=f, line=ln,
                                   detail=f"endpoint {m} {p} ({d.get('handler', '')}) has no frontend caller",
                                   fix="wire a caller, or remove the endpoint if truly unused (human decision)"))
        elif metric_key == "dead_code.ghost_calls":
            import dead_code as dc
            for d in dc.find_ghost_calls():
                m, u = d.get("method", ""), d.get("url", "")
                f, ln = d.get("file", ""), int(d.get("line", 0) or 0)
                out.append(Finding(key=f"ghost:{m} {u}@{f}:{ln}", metric=metric_key,
                                   symbol=f"{m} {u}", file=f, line=ln,
                                   detail=f"frontend calls {m} {u} but no backend route matches",
                                   fix="fix the URL/method, or add the missing route"))
        elif metric_key in ("async_blocking.errors", "async_blocking.warnings"):
            import async_blocking as ab
            want = "error" if metric_key.endswith("errors") else "warning"
            for fnd in ab.scan():
                sev = getattr(fnd, "severity", "")
                if sev != want:
                    continue
                p, ln = getattr(fnd, "path", ""), int(getattr(fnd, "line", 0) or 0)
                fn = getattr(fnd, "func", "")
                out.append(Finding(key=f"async_block:{fn}@{p}:{ln}", metric=metric_key,
                                   symbol=fn, file=p, line=ln,
                                   detail=f"blocking call {fn}() in async context at {p}:{ln}",
                                   fix=getattr(fnd, "fix", "offload to a thread / use the async API")))
        elif metric_key == "runtime.errors":
            # The runtime scanner emits findings as formatted "rel:line  desc" strings
            # (errors prefixed "x ", acknowledged ones marked "[suppressed]"). Parse
            # the ERROR lines into grounded findings so the agent gets the exact
            # location instead of hunting (the cap-with-zero-edits failure mode).
            import re as _re

            import runtime_checks as rc
            _loc = _re.compile(r"([\w./-]+\.(?:py|js)):(\d+)")
            for _name in dir(rc):
                if not _name.startswith("check_"):
                    continue
                try:  # signatures vary; [] suppressions = unfiltered, we drop [suppressed]
                    found, _n = getattr(rc, _name)([], False)
                except Exception:  # noqa: BLE001,PERF203 — skip a check that doesn't fit
                    continue
                for s in found or []:
                    if "[suppressed]" in s or not rc._finding_is_error(s):
                        continue
                    m = _loc.search(s)
                    if not m:
                        continue
                    f, ln = m.group(1), int(m.group(2))
                    clean = _re.sub(r"\x1b\[[0-9;]*m", "", s).strip().lstrip("x").strip()
                    clean = _loc.sub("", clean, count=1).strip()  # drop the repeated loc
                    out.append(Finding(key=f"runtime:{f}:{ln}", metric=metric_key,
                                       symbol=f"runtime@{f}:{ln}", file=f, line=ln,
                                       detail=clean[:200] or "runtime bug-pattern flagged as an error",
                                       fix="fix the flagged runtime bug-pattern at this location"))
    except Exception as exc:  # noqa: BLE001 — probe is best-effort; print nothing usable
        print(json.dumps({"_error": repr(exc)}), file=sys.stderr)
        return []
    return [f.to_dict() for f in out]


def _probe_main(argv: list[str]) -> int:
    # argv: [tree_dir, metric_key]
    if len(argv) < 2:
        print("[]")
        return 0
    tree_dir, metric_key = argv[0], argv[1]
    scripts_dir = os.path.join(tree_dir, _SCRIPTS_REL)
    sys.path.insert(0, scripts_dir)
    print(json.dumps(_probe(scripts_dir, metric_key)))
    return 0


# ===========================================================================
# Async API — runs server-side, spawns the probe against any tree.
# ===========================================================================

async def extract_findings(tree_dir: str, metric_key: str, *,
                           timeout: float = 180.0) -> list[Finding]:
    """Scan ``tree_dir`` for ``metric_key``'s concrete findings by running this
    module's probe against that tree's own scanner scripts. Returns [] on any
    failure (missing scripts, scanner error, timeout) — evidence is best-effort,
    the loop degrades to the generic objective + no-regression oracle."""
    if metric_key not in EVIDENCE_METRICS:
        return []
    if not os.path.isdir(os.path.join(tree_dir, _SCRIPTS_REL)):
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, os.path.abspath(__file__), "--probe", tree_dir, metric_key,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, _err = await asyncio.wait_for(proc.communicate(), timeout)
    except (OSError, TimeoutError):
        return []
    try:
        data = json.loads((out or b"").decode("utf-8", "replace").strip() or "[]")
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    return [Finding.from_dict(d) for d in data if isinstance(d, dict)]


def build_evidence_objective(base_objective: str, scanner: str, metric: str,
                             findings: list[Finding], *, max_items: int = 25) -> str:
    """Compose the agent brief: the base objective + the concrete flagged items +
    the verification contract (so the agent optimizes for the real oracle)."""
    shown = findings[:max_items]
    lines = []
    for f in shown:
        loc = f.file + (f":{f.line}" if f.line else "")
        lines.append(f"- {f.symbol}  ({loc}) — {f.detail}")
    more = f"\n  …and {len(findings) - len(shown)} more." if len(findings) > len(shown) else ""
    return (
        f"{base_objective}\n\n"
        f"EVIDENCE — the {scanner} scanner flagged these {len(findings)} specific "
        f"'{metric}' findings (known false positives already excluded). Pick ONE you "
        f"can CONFIRM is truly resolvable (use the search tool to verify, e.g. that a "
        f"class has no reference anywhere), then make the smallest correct fix to that "
        f"one — go straight to it, don't re-survey the tree:\n"
        + "\n".join(lines) + more +
        f"\n\nEDIT DISCIPLINE: fix it by editing the EXISTING file directly with the "
        f"edit_file tool (e.g. delete the dead rule in place). Do NOT create any new "
        f"files, helper scripts, or tooling — change only the file that contains the "
        f"finding.\n\n"
        f"HOW THIS IS CHECKED: after your edit the {scanner} scanner is re-run on "
        f"your changes. SUCCESS = the specific finding you fixed is GONE from its "
        f"results, with ZERO new '{metric}' findings introduced and no other "
        f"regression. So make a real, minimal, correct fix — not a workaround, and "
        f"don't touch unrelated findings."
    )


def findings_confirm_verifier(*, metric_key: str, baseline_keys: frozenset,
                              intent_classes: tuple[str, ...] = ("*",),
                              timeout: float = 180.0):
    """A MECHANICAL, intent-confirming verifier: re-scan the candidate for the same
    metric and diff against the baseline finding set. PASS (verified) iff ≥1
    baseline finding was resolved and NONE were added; FAIL (a real regression the
    coarse audit can miss — e.g. a 1-for-1 swap that leaves the count unchanged) if
    any new finding appeared; SKIP if nothing in this metric changed (let the
    no-regression oracle decide). Closes over ``baseline_keys`` like the
    preference verifier closes over its store."""
    from augmentum.selfedit.verifier import (
        FAIL,
        PASS,
        SKIP,
        VerifierResult,
        mechanical_verifier,
    )
    name = f"findings_confirm:{metric_key}"

    async def _run(ctx: dict) -> VerifierResult:
        cand = ctx.get("candidate_dir") or ""
        if not cand:
            return VerifierResult(name, "mechanical", SKIP, confirms_intent=True,
                                  required=False, detail="no candidate dir")
        cand_keys = {f.key for f in await extract_findings(cand, metric_key, timeout=timeout)}
        resolved = baseline_keys - cand_keys
        added = cand_keys - baseline_keys
        if added:
            return VerifierResult(name, "mechanical", FAIL, confirms_intent=True,
                                  score=0.0, required=True,
                                  detail=f"introduced {len(added)} new {metric_key} finding(s): "
                                         + ", ".join(sorted(added)[:5]))
        if not resolved:
            # The objective WAS to resolve a flagged finding. Resolving none means
            # the agent didn't do the job (e.g. wrote a helper script, edited the
            # wrong thing) — FAIL so the attempt is rejected and the ladder climbs,
            # rather than settling a useless change at human_required.
            return VerifierResult(name, "mechanical", FAIL, confirms_intent=True,
                                  score=0.0, required=True,
                                  detail=f"resolved no {metric_key} finding — the flagged "
                                         "item was not actually fixed")
        return VerifierResult(name, "mechanical", PASS, confirms_intent=True,
                              score=1.0, required=False,
                              detail=f"resolved {len(resolved)} {metric_key} finding(s) "
                                     f"({', '.join(sorted(resolved)[:5])}), 0 added")

    return mechanical_verifier(name, _run, confirms_intent=True,
                               intent_classes=intent_classes, cost=6, required=True)


@dataclass
class TargetEnrichment:
    objective: str = ""
    verifiers: list = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.findings)


async def enrich_target(evidence_tree: str, scanner: str, metric: str,
                        base_objective: str, *, max_items: int = 25) -> TargetEnrichment:
    """For a debt target with a detail-bearing scanner: scan the baseline tree for
    the concrete findings, and return an evidence-grounded objective + a mechanical
    confirm verifier keyed to that baseline set. Empty (ungrounded) when the metric
    has no extractor or the scan finds nothing — caller keeps the generic path."""
    metric_key = f"{scanner}.{metric}"
    if metric_key not in EVIDENCE_METRICS:
        return TargetEnrichment()
    findings = await extract_findings(evidence_tree, metric_key, timeout=180.0)
    if not findings:
        return TargetEnrichment()
    objective = build_evidence_objective(base_objective, scanner, metric, findings,
                                         max_items=max_items)
    verifier = findings_confirm_verifier(
        metric_key=metric_key, baseline_keys=frozenset(f.key for f in findings))
    return TargetEnrichment(objective=objective, verifiers=[verifier], findings=findings)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--probe":
        raise SystemExit(_probe_main(args[1:]))
    raise SystemExit("usage: evidence.py --probe <tree_dir> <scanner.metric>")
