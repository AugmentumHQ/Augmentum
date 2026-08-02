"""Tests for the custom-check runner.

Writes synthetic checks into a tmp workspace and runs them via the
discovery + execution flow. Covers happy path, broken checks
(import errors, missing run, returning non-list), and the underscore-
prefix disable convention.
"""

from __future__ import annotations

from pathlib import Path

from augmentum.bug_finder.custom_check_runner import (
    collect_all_findings,
    list_custom_checks,
    run_all_custom_checks,
)
from augmentum.bug_finder.workspace_substrate import (
    custom_checks_dir,
    ensure_substrate,
)


def _mk_check(workspace: Path, name: str, source: str) -> Path:
    ensure_substrate(workspace)
    target = custom_checks_dir(workspace) / f"{name}.py"
    target.write_text(source, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_custom_checks_empty_when_no_substrate(tmp_path: Path) -> None:
    assert list_custom_checks(tmp_path) == []


def test_list_custom_checks_returns_py_files(tmp_path: Path) -> None:
    _mk_check(tmp_path, "user_id_scoping", "def run(root):\n    return []\n")
    _mk_check(tmp_path, "auth_middleware", "def run(root):\n    return []\n")
    checks = list_custom_checks(tmp_path)
    assert {p.stem for p in checks} == {"user_id_scoping", "auth_middleware"}


def test_list_custom_checks_skips_underscore_prefixed(tmp_path: Path) -> None:
    """Files starting with ``_`` are disabled by convention."""
    _mk_check(tmp_path, "active", "def run(root):\n    return []\n")
    _mk_check(tmp_path, "_disabled", "def run(root):\n    return []\n")
    checks = list_custom_checks(tmp_path)
    assert [p.stem for p in checks] == ["active"]


# ---------------------------------------------------------------------------
# Execution — happy paths
# ---------------------------------------------------------------------------


def test_run_returns_findings_in_normalized_shape(tmp_path: Path) -> None:
    source = '''
from pathlib import Path
def run(root: Path) -> list[dict]:
    return [{
        "severity": "high",
        "category": "test_finding",
        "file": "src/x.py",
        "line": 42,
        "message": "synthetic finding for test",
        "fix": "do the right thing",
    }]
'''
    _mk_check(tmp_path, "synthetic", source)
    results = run_all_custom_checks(tmp_path)
    assert len(results) == 1
    assert results[0].succeeded
    assert len(results[0].findings) == 1
    f = results[0].findings[0]
    assert f.scanner == "custom:synthetic"
    assert f.severity == "high"
    assert f.category == "test_finding"
    assert f.file == "src/x.py"
    assert f.line == 42


def test_collect_all_findings_flattens(tmp_path: Path) -> None:
    _mk_check(tmp_path, "check_a", '''
def run(root):
    return [
        {"severity":"high","category":"a","file":"a.py","line":1,"message":""},
        {"severity":"low","category":"a","file":"a.py","line":2,"message":""},
    ]
''')
    _mk_check(tmp_path, "check_b", '''
def run(root):
    return [{"severity":"medium","category":"b","file":"b.py","line":1,"message":""}]
''')
    findings = collect_all_findings(tmp_path)
    assert len(findings) == 3


def test_severity_normalizes_unknown_to_medium(tmp_path: Path) -> None:
    _mk_check(tmp_path, "x", '''
def run(root):
    return [{"severity":"weirdness","category":"x","file":"x.py","line":1,"message":""}]
''')
    findings = collect_all_findings(tmp_path)
    assert findings[0].severity == "medium"


def test_missing_category_defaults_to_check_name(tmp_path: Path) -> None:
    _mk_check(tmp_path, "my_check", '''
def run(root):
    return [{"file":"x.py","line":1,"message":""}]
''')
    findings = collect_all_findings(tmp_path)
    assert findings[0].category == "my_check"


# ---------------------------------------------------------------------------
# Execution — failure modes (isolated)
# ---------------------------------------------------------------------------


def test_check_with_syntax_error_is_isolated(tmp_path: Path) -> None:
    """One broken check shouldn't break the others. A syntax error is now
    caught by the load-time safety gate (which ast.parses the source)
    before exec, so it's rejected rather than import-erroring."""
    _mk_check(tmp_path, "broken", "def run(  # missing colon\n    pass\n")
    _mk_check(tmp_path, "ok", '''
def run(root):
    return [{"file":"x.py","line":1,"severity":"low","message":""}]
''')
    results = run_all_custom_checks(tmp_path)
    by_name = {r.check_name: r for r in results}
    assert not by_name["broken"].succeeded
    assert "syntax error" in by_name["broken"].error.lower()
    assert by_name["ok"].succeeded
    assert len(by_name["ok"].findings) == 1


# ---------------------------------------------------------------------------
# Security — load-time safety gate (defense against planted/untrusted checks)
# ---------------------------------------------------------------------------


def test_check_importing_subprocess_is_rejected_without_executing(
    tmp_path: Path,
) -> None:
    """A check that imports outside the read-only stdlib allowlist must be
    rejected at LOAD time and never reach exec_module. This is the real
    threat: a .py planted in a cloned repo's custom_checks dir (bypassing
    the write-time gate) would otherwise exec arbitrary code on the host."""
    canary = tmp_path / "canary.txt"
    source = f'''
import subprocess
from pathlib import Path
Path(r"{canary}").write_text("pwned")
def run(root):
    return []
'''
    _mk_check(tmp_path, "evil_subprocess", source)
    results = run_all_custom_checks(tmp_path)
    assert len(results) == 1
    assert not results[0].succeeded
    assert "safety gate" in results[0].error.lower()
    assert "subprocess" in results[0].error.lower()
    # The module-level side effect must NOT have run.
    assert not canary.exists()


def test_check_using_exec_is_rejected(tmp_path: Path) -> None:
    source = '''
def run(root):
    exec("import os; os.system('echo pwned')")
    return []
'''
    _mk_check(tmp_path, "evil_exec", source)
    results = run_all_custom_checks(tmp_path)
    assert not results[0].succeeded
    assert "safety gate" in results[0].error.lower()


def test_check_with_dunder_escape_is_rejected(tmp_path: Path) -> None:
    """The classic ().__class__.__bases__ sandbox-escape chain."""
    source = '''
def run(root):
    cls = ().__class__.__bases__
    return []
'''
    _mk_check(tmp_path, "evil_dunder", source)
    results = run_all_custom_checks(tmp_path)
    assert not results[0].succeeded
    assert "safety gate" in results[0].error.lower()


def test_valid_ast_check_still_runs_after_gate(tmp_path: Path) -> None:
    """Regression: a legitimate AST/grep check using the allowed stdlib
    set passes the gate and executes normally."""
    source = '''
import ast
import re
from pathlib import Path
def run(root: Path) -> list[dict]:
    return [{"severity":"low","category":"ok","file":"a.py","line":1,"message":"ran"}]
'''
    _mk_check(tmp_path, "legit", source)
    results = run_all_custom_checks(tmp_path)
    assert results[0].succeeded
    assert len(results[0].findings) == 1
    assert results[0].findings[0].message == "ran"


def test_check_missing_run_function_is_reported(tmp_path: Path) -> None:
    _mk_check(tmp_path, "norun", "def main():\n    pass\n")
    results = run_all_custom_checks(tmp_path)
    assert not results[0].succeeded
    assert "run" in results[0].error.lower()


def test_check_returning_non_list_is_reported(tmp_path: Path) -> None:
    _mk_check(tmp_path, "wrong", '''
def run(root):
    return "not a list"
''')
    results = run_all_custom_checks(tmp_path)
    assert not results[0].succeeded
    assert "list" in results[0].error.lower()


def test_check_raising_at_runtime_is_reported(tmp_path: Path) -> None:
    _mk_check(tmp_path, "raises", '''
def run(root):
    raise RuntimeError("synthetic failure")
''')
    results = run_all_custom_checks(tmp_path)
    assert not results[0].succeeded
    assert "RuntimeError" in results[0].error
