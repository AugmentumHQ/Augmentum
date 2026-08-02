"""Application Health Signal tests — the gate + rollback + fitness keystone."""

from __future__ import annotations

import os
import tempfile

from augmentum.selfedit import gate
from augmentum.selfedit import health as H


def _probe(name, *, ok=True, score=1.0, weight=1.0, required=True, measured=True):
    async def p():
        return H.DimensionResult(name, ok=ok, score=score, weight=weight,
                                 required=required, measured=measured)
    return p


async def test_assess_weighted_aggregate_and_ok():
    probes = {
        "a": _probe("a", ok=True, score=1.0, weight=3.0),
        "b": _probe("b", ok=True, score=0.0, weight=1.0),
    }
    r = await H.assess(probes, ref="test")
    # weighted mean: (1.0*3 + 0.0*1)/4 = 0.75
    assert abs(r.score - 0.75) < 1e-6
    assert r.ok is True  # both ok (score doesn't sink ok; ok flag does)
    assert r.ref == "test"


async def test_required_failure_sinks_ok_but_skip_excluded():
    probes = {
        "req": _probe("req", ok=False, score=0.0, required=True),
        "skipped": _probe("skipped", ok=False, score=0.0, measured=False),
    }
    r = await H.assess(probes)
    assert r.ok is False                       # required dim failed
    assert r.dim("skipped").measured is False  # excluded from aggregate
    assert abs(r.score - 0.0) < 1e-6           # only 'req' counted


async def test_crashing_probe_is_failed_dimension():
    async def boom():
        raise RuntimeError("nope")
    r = await H.assess({"x": boom})
    assert r.ok is False and r.dim("x").ok is False and "nope" in r.dim("x").detail


async def test_registry_register_and_assess():
    H.clear_registry()
    try:
        H.register_probe("r1", _probe("r1", score=1.0))
        r = await H.assess()  # uses registry
        assert r.dim("r1") is not None and r.ok is True
    finally:
        H.clear_registry()


async def test_probe_from_check_maps_status():
    def chk(name, status):
        async def run():
            return (status, f"{name}:{status}")
        return gate.Check(name, run)
    assert (await H.probe_from_check(chk("p", "pass"))()).ok is True
    fail = await H.probe_from_check(chk("f", "fail"))()
    assert fail.ok is False and fail.score == 0.0
    skip = await H.probe_from_check(chk("s", "skip"))()
    assert skip.measured is False  # skip → unmeasured, never a false regression


async def test_compare_detects_regression_and_improvement():
    base = await H.assess({
        "compile": _probe("compile", ok=True, score=1.0),
        "perf": _probe("perf", ok=True, score=0.5),
    })
    cur = await H.assess({
        "compile": _probe("compile", ok=False, score=0.0),  # regressed into failure
        "perf": _probe("perf", ok=True, score=0.9),          # improved
    })
    delta = H.compare(cur, base)
    assert "compile" in delta.regressions
    assert "compile" in delta.new_failures
    assert "perf" in delta.improvements
    assert delta.ok is False           # a required dim regressed → block promote / trigger rollback
    assert delta.score_delta < 0


async def test_compare_no_baseline_uses_current_ok():
    cur = await H.assess({"a": _probe("a", ok=True, score=1.0)})
    assert H.compare(cur, None).ok is True
    bad = await H.assess({"a": _probe("a", ok=False, score=0.0)})
    assert H.compare(bad, None).ok is False


async def test_baseline_roundtrip():
    r = await H.assess({"a": _probe("a", ok=True, score=0.8)}, ref="sha123", at=42.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "baseline.json")
        assert H.load_baseline(p) is None  # missing → None
        H.save_baseline(r, p)
        loaded = H.load_baseline(p)
        assert loaded is not None
        assert loaded.ref == "sha123" and abs(loaded.score - 0.8) < 1e-6
        assert loaded.dim("a").score == 0.8


async def test_default_probes_real_compile_dimension():
    # The seed app-wide set, built on the gate, runs for real against a tree.
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "augmentum")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "ok.py"), "w") as f:
            f.write("x = 1\n")
        r = await H.assess(H.default_probes(d), ref="candidate")
        assert r.dim("compile") is not None and r.dim("compile").ok is True
