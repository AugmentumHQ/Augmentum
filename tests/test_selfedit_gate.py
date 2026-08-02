"""Fitness-gate tests — the core decision logic + the standard app checks."""

from __future__ import annotations

import os
import tempfile

from augmentum.selfedit.gate import (
    Check,
    _is_infra_failure,
    _module_available,
    compile_check,
    default_app_gate,
    pytest_check,
    run_gate,
    smoke_import_check,
)


def _check(name, status, required=True):
    async def run():
        return (status, f"{name} says {status}")
    return Check(name, run, required=required)


async def test_required_fail_sinks_gate():
    v = await run_gate([_check("a", "pass"), _check("b", "fail")])
    assert v.passed is False
    assert "b" in v.summary


async def test_advisory_fail_does_not_sink():
    v = await run_gate([_check("a", "pass"), _check("opt", "fail", required=False)])
    assert v.passed is True


async def test_skip_does_not_sink():
    v = await run_gate([_check("a", "pass"), _check("missing", "skip")])
    assert v.passed is True
    assert any(c.status == "skip" for c in v.checks)


async def test_crashing_check_is_a_failure():
    async def boom():
        raise RuntimeError("kaboom")
    v = await run_gate([Check("boom", boom, required=True)])
    assert v.passed is False
    assert "kaboom" in v.checks[0].detail


async def test_verdict_serializable():
    v = await run_gate([_check("a", "pass")])
    d = v.to_dict()
    assert d["passed"] is True and d["checks"][0]["name"] == "a"
    assert isinstance(v.to_json(), str)


async def test_compile_check_passes_on_valid_tree():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "augmentum")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "mod.py"), "w") as f:
            f.write("x = 1\n")
        v = await run_gate([compile_check(pkg)])
        assert v.passed is True


async def test_default_app_gate_fails_on_syntax_error():
    with tempfile.TemporaryDirectory() as d:
        pkg = os.path.join(d, "augmentum")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "broken.py"), "w") as f:
            f.write("def oops(:\n    pass\n")  # syntax error
        v = await run_gate(default_app_gate(d))
        assert v.passed is False
        assert any(c.name == "compile" and c.status == "fail" for c in v.checks)


# ---------------------------------------------------------------------------
# infrastructure-aware oracles — the live-smoke-test finding (2026-07-02):
# an oracle that can't run its own framework must SKIP (inconclusive), never
# FAIL (false negative that rejects good code + poisons the archive).
# ---------------------------------------------------------------------------

def test_infra_failure_detection():
    assert _is_infra_failure(127, "anything") is True
    assert _is_infra_failure(1, "/usr/bin/python: No module named pytest") is True
    assert _is_infra_failure(1, "could not launch python") is True
    # a real test failure is NOT infra
    assert _is_infra_failure(1, "assert 1 == 2\nE  AssertionError") is False


def test_module_available_matches_reality():
    assert _module_available("os") is True
    assert _module_available("a_module_that_cannot_exist_xyz") is False


async def test_pytest_check_skips_when_pytest_absent(monkeypatch):
    # the exact live bug: pytest missing in the verify env → SKIP, not FAIL
    monkeypatch.setattr("augmentum.selfedit.gate._module_available",
                        lambda m: m != "pytest")
    v = await run_gate([pytest_check(["tests/test_x.py"], cwd=".")])
    assert v.passed is True   # a SKIP never sinks the gate
    assert v.checks[0].status == "skip"
    assert "pytest not available" in v.checks[0].detail


async def test_pytest_check_no_tests_is_skip():
    v = await run_gate([pytest_check([], cwd=".")])
    assert v.checks[0].status == "skip"


async def test_smoke_import_broken_module_is_a_real_fail():
    # a genuine ImportError is a REAL signal (module broken) → FAIL, not skip
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "brokenmod.py"), "w") as f:
            f.write("import a_package_that_does_not_exist_xyz\n")
        v = await run_gate([smoke_import_check("brokenmod", cwd=d)])
        assert v.passed is False and v.checks[0].status == "fail"
