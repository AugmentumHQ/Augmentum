"""The Application Health Signal — the single source of truth for "is Augmentum
healthy, and did this change make it better or worse?"

This is the keystone of self-improvement. Self-improvement only works where
outcomes are verifiable, so one authoritative, app-wide health assessment serves
THREE jobs at once:

  1. **Promotion gate** — a candidate may go live only if it's healthy AND not a
     regression vs the last-good baseline.
  2. **Rollback trigger** — after promotion, if live health regresses vs
     baseline, revert.
  3. **Fitness function** — the score the whole loop optimizes; the pillar's
     "mistake" is simply a negative delta.

Design:
* **Registry of probes.** Each subsystem contributes a probe (a dimension of
  health). Coverage is composable and grows over time — and a surface is only
  safely self-editable once a probe covers it, so *the registry's reach is the
  self-improvement reach.*
* **Built on the gate.** A gate Check adapts straight into a probe
  (``probe_from_check``), so the fitness gate is a subset of the health signal —
  one consistent definition of "working," app-wide.
* **Baseline + delta.** Health is judged as a DELTA against a persisted
  known-good baseline (better/worse), not just an absolute — which is exactly
  what promotion ("no regression") and rollback ("regressed → revert") need.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.selfedit import gate as _gate
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DimensionResult:
    """One measured dimension of app health."""
    name: str
    ok: bool                 # hard pass/fail (for required dimensions)
    score: float             # graded health 0.0..1.0 (1.0 = perfect)
    detail: str = ""
    weight: float = 1.0      # contribution to the aggregate score
    required: bool = True     # a required dimension that's not ok → report not ok
    measured: bool = True     # False = couldn't measure (skipped); excluded from aggregate

    def to_dict(self) -> dict:
        return {
            "name": self.name, "ok": self.ok, "score": round(self.score, 4),
            "detail": self.detail[:2000], "weight": self.weight,
            "required": self.required, "measured": self.measured,
        }

    @staticmethod
    def from_dict(d: dict) -> DimensionResult:
        return DimensionResult(
            name=d["name"], ok=bool(d.get("ok")), score=float(d.get("score", 0.0)),
            detail=d.get("detail", ""), weight=float(d.get("weight", 1.0)),
            required=bool(d.get("required", True)), measured=bool(d.get("measured", True)),
        )


@dataclass
class HealthReport:
    score: float                                   # weighted aggregate 0..1
    ok: bool                                        # all required+measured dims ok
    dimensions: list[DimensionResult] = field(default_factory=list)
    ref: str = ""                                   # what was assessed (sha | "live" | "candidate:<id>")
    at: float = 0.0                                 # epoch seconds (caller-stamped)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4), "ok": self.ok, "ref": self.ref, "at": self.at,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(d: dict) -> HealthReport:
        return HealthReport(
            score=float(d.get("score", 0.0)), ok=bool(d.get("ok")),
            ref=d.get("ref", ""), at=float(d.get("at", 0.0)),
            dimensions=[DimensionResult.from_dict(x) for x in d.get("dimensions", [])],
        )

    def dim(self, name: str) -> DimensionResult | None:
        return next((x for x in self.dimensions if x.name == name), None)


Probe = Callable[[], Awaitable[DimensionResult]]

# name -> probe. The app-wide registry; subsystems extend it to grow coverage.
_REGISTRY: dict[str, Probe] = {}


def register_probe(name: str, probe: Probe) -> None:
    """Register an app-health probe. Re-registering the same name replaces it."""
    _REGISTRY[name] = probe


def registered_probes() -> dict[str, Probe]:
    return dict(_REGISTRY)


def clear_registry() -> None:  # for tests
    _REGISTRY.clear()


def probe_from_check(check: _gate.Check, *, weight: float = 1.0) -> Probe:
    """Adapt a fitness-gate Check into a health probe. pass→1.0/ok, fail→0.0/not
    ok, skip→unmeasured (excluded from the aggregate, never a false regression)."""
    async def _probe() -> DimensionResult:
        status, detail = await check.run()
        if status == "skip":
            return DimensionResult(check.name, ok=True, score=1.0, detail=detail,
                                   weight=weight, required=check.required, measured=False)
        ok = status == "pass"
        return DimensionResult(check.name, ok=ok, score=1.0 if ok else 0.0,
                               detail=detail, weight=weight, required=check.required)
    return _probe


async def assess(probes: dict[str, Probe] | None = None, *, ref: str = "", at: float = 0.0) -> HealthReport:
    """Run probes (default: the registry) and aggregate into one HealthReport.

    Aggregate score = weighted mean over MEASURED dimensions. ``ok`` = every
    required, measured dimension is ok. ``at`` is caller-stamped (epoch seconds);
    pass it in so this stays deterministic/testable."""
    probes = registered_probes() if probes is None else probes
    dims: list[DimensionResult] = []
    for name, probe in probes.items():
        try:
            d = await probe()
        except Exception as exc:  # noqa: BLE001 — a crashing probe is a failed dimension
            d = DimensionResult(name, ok=False, score=0.0, detail=f"probe crashed: {exc!r}")
        d.name = d.name or name
        dims.append(d)
    measured = [d for d in dims if d.measured]
    wsum = sum(d.weight for d in measured) or 1.0
    score = sum(d.score * d.weight for d in measured) / wsum
    ok = all(d.ok for d in measured if d.required)
    return HealthReport(score=score, ok=ok, dimensions=dims, ref=ref, at=at)


@dataclass
class HealthDelta:
    ok: bool                                  # no REQUIRED dimension regressed
    score_delta: float                        # current.score - baseline.score
    regressions: list[str] = field(default_factory=list)   # dims that got worse
    improvements: list[str] = field(default_factory=list)  # dims that got better
    new_failures: list[str] = field(default_factory=list)  # required dims newly not-ok

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "score_delta": round(self.score_delta, 4),
            "regressions": self.regressions, "improvements": self.improvements,
            "new_failures": self.new_failures,
        }


def compare(current: HealthReport, baseline: HealthReport | None, *, eps: float = 1e-9) -> HealthDelta:
    """Better-or-worse vs the known-good baseline. With no baseline, the delta is
    OK iff the current report itself is ok (first run sets the bar).

    A *regression* is a dimension whose score dropped; a *new_failure* is a
    required dimension that was ok in baseline and isn't now. Promotion requires
    ``delta.ok``; rollback fires when live's delta is not ok."""
    if baseline is None:
        return HealthDelta(ok=current.ok, score_delta=0.0,
                           new_failures=[d.name for d in current.dimensions
                                         if d.required and d.measured and not d.ok])
    regressions, improvements, new_failures = [], [], []
    base_by = {d.name: d for d in baseline.dimensions}
    for cur in current.dimensions:
        if not cur.measured:
            continue
        base = base_by.get(cur.name)
        if base is None:
            continue  # brand-new dimension — neither regression nor improvement
        if cur.score < base.score - eps:
            regressions.append(cur.name)
        elif cur.score > base.score + eps:
            improvements.append(cur.name)
        if cur.required and base.ok and not cur.ok:
            new_failures.append(cur.name)
    # OK iff no required dimension regressed into failure.
    ok = not new_failures
    return HealthDelta(ok=ok, score_delta=current.score - baseline.score,
                       regressions=regressions, improvements=improvements,
                       new_failures=new_failures)


# ---------------------------------------------------------------------------
# Baseline persistence (the "last known-good" the delta is measured against)
# ---------------------------------------------------------------------------

def save_baseline(report: HealthReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.to_json())


def load_baseline(path: str) -> HealthReport | None:
    try:
        with open(path, encoding="utf-8") as f:
            return HealthReport.from_dict(json.load(f))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — a corrupt baseline must not wedge the loop
        log.warning("health_baseline_load_failed", path=path, error=repr(exc))
        return None


# ---------------------------------------------------------------------------
# The standard app-wide probe set (built on the fitness gate)
# ---------------------------------------------------------------------------

def default_probes(target_dir: str, *, test_paths: list[str] | None = None,
                   smoke_modules: list[str] | None = None) -> dict[str, Probe]:
    """The baseline app-health probes, derived from the standard gate checks.
    This is the seed; subsystems register richer probes over time to widen
    coverage (and thus the safe self-edit reach)."""
    probes: dict[str, Probe] = {}
    for check in _gate.default_app_gate(target_dir, test_paths=test_paths,
                                        smoke_modules=smoke_modules):
        probes[check.name] = probe_from_check(check)
    return probes


def now() -> float:
    """Wall-clock stamp helper (kept in one place so callers/tests can avoid it)."""
    return time.time()
