"""Tests for the builder_eval judge — the execution-truth gate.

The judge is pure (no Docker / no backend): it grades a finished build's tool
trail against the Frontend App Builder Power's definition of done. These tests
feed synthetic trails so the scoring logic is pinned independently of any live
build. Run directly (the pytest conftest pulls the full app) via:

    .venv/Scripts/python.exe -c "import tests.test_builder_eval as t; t._run_all()"
"""
from __future__ import annotations

try:
    from scripts.builder_eval import (
        SCENARIOS,
        judge_build,
    )
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"scripts.builder_eval not importable in this build: {_import_exc}", allow_module_level=True)


def _steps(*tools: str) -> list[dict]:
    """Build an ordered trail from a tool sequence."""
    return [{"i": i, "iteration": i, "tool": t, "preview": ""} for i, t in enumerate(tools)]


# A trail that hits the entire floor for a calculator (3 drives, 3 asserts).
_FULL_CALC = _steps(
    "builder_design_system", "builder_reference",
    "file_write", "file_write",
    "service_start", "browser_open",
    "browser_type", "browser_evaluate",
    "browser_type", "browser_evaluate",
    "browser_click", "browser_evaluate",
    "finish_task",
)


def test_full_trail_passes():
    v = judge_build(steps=_FULL_CALC, status="completed", artifact_ok=True,
                    kind="calculator", min_drives=3, min_asserts=3)
    assert v["passed"] is True, v
    assert v["score"] == 1.0
    assert v["failed_checks"] == []
    assert all(v["hard"].values())
    assert v["depth"]["enough_drives"] and v["depth"]["enough_asserts"]
    assert v["soft"]["pulled_resources"] is True
    assert v["drives"] == 3 and v["asserts"] == 3


def test_no_server_fails_floor():
    trail = [s for s in _FULL_CALC if s["tool"] != "service_start"]
    v = judge_build(steps=trail, status="completed", artifact_ok=True, kind="calculator")
    assert v["passed"] is False
    assert v["hard"]["ran_server"] is False
    assert "ran_server" in v["failed_checks"]


def test_no_assertion_fails_floor():
    """The headline failure mode: it built + opened it but never verified."""
    trail = [s for s in _FULL_CALC if s["tool"] not in ("browser_evaluate", "browser_verify")]
    v = judge_build(steps=trail, status="completed", artifact_ok=True, kind="calculator")
    assert v["passed"] is False
    assert v["hard"]["asserted_behavior"] is False
    assert "asserted_behavior" in v["failed_checks"]


def test_no_artifact_fails_floor():
    v = judge_build(steps=_FULL_CALC, status="completed", artifact_ok=False, kind="calculator")
    assert v["passed"] is False
    assert v["hard"]["published_artifact"] is False


def test_unfinished_status_fails_floor():
    v = judge_build(steps=_FULL_CALC, status="error", artifact_ok=True, kind="calculator")
    assert v["passed"] is False
    assert v["hard"]["finished_clean"] is False


def test_browser_verify_counts_as_assertion():
    trail = _steps("file_write", "service_start", "browser_open",
                   "browser_click", "browser_verify", "finish_task")
    v = judge_build(steps=trail, status="completed", artifact_ok=True,
                    kind="dashboard", min_drives=1, min_asserts=1)
    assert v["hard"]["asserted_behavior"] is True
    assert v["passed"] is True


def test_depth_thresholds_independent_of_hard_floor():
    """A single drive+assert clears the binary floor but flags shallow depth
    when the kind asks for more — depth is a quality signal, not a hard gate."""
    trail = _steps("file_write", "service_start", "browser_open",
                   "browser_type", "browser_evaluate", "finish_task")
    v = judge_build(steps=trail, status="completed", artifact_ok=True,
                    kind="calculator", min_drives=3, min_asserts=3)
    assert v["passed"] is True              # hard floor met
    assert v["depth"]["enough_drives"] is False   # but shallow for a calculator
    assert v["depth"]["enough_asserts"] is False


def test_guessed_instead_of_pulling_resources():
    trail = _steps("file_write", "service_start", "browser_open",
                   "browser_click", "browser_evaluate", "finish_task")
    v = judge_build(steps=trail, status="completed", artifact_ok=True, kind="form")
    assert v["soft"]["pulled_resources"] is False   # never called builder_*
    assert v["passed"] is True                       # still passes the hard floor


def test_empty_trail_fails_everything():
    v = judge_build(steps=[], status="error", artifact_ok=False, kind="game")
    assert v["passed"] is False
    assert v["score"] == 0.0
    assert set(v["failed_checks"]) == set(v["hard"].keys())


def test_code_edit_counts_as_writing():
    trail = _steps("code_edit", "service_start", "browser_open",
                   "browser_type", "browser_evaluate", "finish_task")
    v = judge_build(steps=trail, status="completed", artifact_ok=True, kind="form")
    assert v["hard"]["wrote_code"] is True


def test_scenarios_cover_each_kind():
    kinds = {s.kind for s in SCENARIOS}
    # base widgets + the comprehensive "app" tier
    assert {"calculator", "form", "dashboard", "game"} <= kinds
    assert "app" in kinds  # comprehensive stateful apps


def _run_all() -> None:
    """Direct runner (pytest's conftest imports the full app, which hangs in
    some envs). Invoke each test_* and report."""
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
