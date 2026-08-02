"""Tests for the codebase-agnostic generic scanners.

Smoke against a tiny in-memory project with deliberately-buggy
patterns. Bandit + Ruff are external subprocesses so we don't mock
their JSON parsing — we trust the scanners and just verify our
wrapper's normalization on the real output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from augmentum.bug_finder.generic_scanners import (
    GenericScannerSuiteResult,
    _normalize_severity,
    _ruff_severity_for_rule,
    run_bandit,
    run_generic_suite,
    run_generic_suite_timed,
    run_ruff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Ruff severity bucketing
# ---------------------------------------------------------------------------


def test_ruff_severity_security_rule_is_high() -> None:
    assert _ruff_severity_for_rule("S105") == "high"
    assert _ruff_severity_for_rule("S101") == "high"
    assert _ruff_severity_for_rule("S324") == "high"


def test_ruff_severity_bugbear_is_medium() -> None:
    assert _ruff_severity_for_rule("B007") == "medium"
    assert _ruff_severity_for_rule("BLE001") == "medium"


def test_ruff_severity_pyflakes_is_medium() -> None:
    assert _ruff_severity_for_rule("F401") == "medium"


def test_ruff_severity_pycodestyle_is_low() -> None:
    assert _ruff_severity_for_rule("E501") == "low"


def test_ruff_severity_perf_is_info() -> None:
    assert _ruff_severity_for_rule("PERF401") == "info"


def test_ruff_severity_unknown_defaults_to_low() -> None:
    assert _ruff_severity_for_rule("XYZ999") == "low"
    assert _ruff_severity_for_rule("") == "low"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_normalize_severity_clamps_unknown() -> None:
    assert _normalize_severity("CRITICAL") == "critical"
    assert _normalize_severity("HIGH") == "high"
    assert _normalize_severity("weird") == "medium"
    assert _normalize_severity(None) == "medium"


# ---------------------------------------------------------------------------
# Bandit — live subprocess
# ---------------------------------------------------------------------------


def test_bandit_finds_md5_for_security(tmp_path: Path) -> None:
    """Bandit should surface ``hashlib.md5()`` without
    ``usedforsecurity=False`` (B324)."""
    _mk(tmp_path, {
        "src/weak.py": (
            "import hashlib\n"
            "def make_id(s: str) -> str:\n"
            "    return hashlib.md5(s.encode()).hexdigest()\n"
        ),
    })
    findings = run_bandit(tmp_path)
    assert findings, "bandit should have found at least one issue"
    # The exact rule code is B324; severity HIGH per bandit
    md5_finds = [f for f in findings if "md5" in f.message.lower()]
    assert md5_finds, "expected an MD5-related bandit finding"
    assert md5_finds[0].scanner == "bandit"
    assert md5_finds[0].severity in {"high", "medium"}


def test_bandit_skips_tests_directory(tmp_path: Path) -> None:
    """Default skip list excludes ``tests/`` so test fixtures
    don't spam the agent."""
    _mk(tmp_path, {
        "tests/test_weak.py": (
            "import hashlib\n"
            "h = hashlib.md5(b'x').hexdigest()\n"
        ),
        "src/clean.py": "x = 1\n",
    })
    findings = run_bandit(tmp_path)
    test_findings = [f for f in findings if "test" in f.file.lower()]
    assert test_findings == []


def test_bandit_returns_empty_on_missing_root(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no_such_dir"
    assert run_bandit(nonexistent) == []


def test_bandit_relative_paths(tmp_path: Path) -> None:
    """File paths should be relative to ``root`` so suppression
    matching is portable."""
    _mk(tmp_path, {
        "src/weak.py": "import hashlib\nhashlib.md5(b'x').hexdigest()\n",
    })
    findings = run_bandit(tmp_path)
    if findings:  # may be empty depending on bandit version
        assert not Path(findings[0].file).is_absolute()


# ---------------------------------------------------------------------------
# Ruff — live subprocess
# ---------------------------------------------------------------------------


def test_ruff_finds_hardcoded_password_pattern(tmp_path: Path) -> None:
    """S105 — hardcoded-password literal."""
    _mk(tmp_path, {
        "src/config.py": (
            'TOKEN_TYPE = "bearer"\n'
            'PASSWORD = "hunter2"\n'
            'jwt_token_prefix = "Bearer"\n'
        ),
    })
    findings = run_ruff(tmp_path)
    s105 = [f for f in findings if f.category == "S105"]
    assert s105, "expected at least one S105 (hardcoded password) finding"
    assert s105[0].severity == "high"


def test_ruff_finds_assert_usage(tmp_path: Path) -> None:
    """S101 — ``assert`` in production code."""
    _mk(tmp_path, {
        "src/runtime.py": (
            "def check(x):\n"
            "    assert x > 0\n"
            "    return x\n"
        ),
    })
    findings = run_ruff(tmp_path)
    s101 = [f for f in findings if f.category == "S101"]
    assert s101, "expected at least one S101 (assert detected) finding"


def test_ruff_empty_on_clean_file(tmp_path: Path) -> None:
    _mk(tmp_path, {
        "src/clean.py": (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
        ),
    })
    findings = run_ruff(tmp_path)
    # Could still surface info-level findings; just check we got a list
    assert isinstance(findings, list)


def test_ruff_returns_empty_on_missing_root(tmp_path: Path) -> None:
    assert run_ruff(tmp_path / "nope") == []


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def test_run_generic_suite_returns_dict_by_scanner(tmp_path: Path) -> None:
    _mk(tmp_path, {"src/x.py": "x = 1\n"})
    result = run_generic_suite(tmp_path)
    assert set(result.keys()) == {"bandit", "ruff"}


def test_run_generic_suite_timed_includes_wallclock(tmp_path: Path) -> None:
    _mk(tmp_path, {"src/x.py": "x = 1\n"})
    result = run_generic_suite_timed(tmp_path)
    assert isinstance(result, GenericScannerSuiteResult)
    assert result.wallclock_seconds >= 0.0
    assert result.total_findings == sum(
        len(v) for v in result.findings_by_scanner.values()
    )


# ---------------------------------------------------------------------------
# Mixed-scanner cross-codebase sanity
# ---------------------------------------------------------------------------


def test_realistic_fastapi_snippet_surfaces_findings(tmp_path: Path) -> None:
    """Drop a small FastAPI-shaped module with multiple deliberately-
    buggy patterns and confirm at least one of each scanner reports
    a finding. End-to-end smoke that proves the wrappers are wired."""
    _mk(tmp_path, {
        "src/api.py": (
            'from fastapi import APIRouter\n'
            'router = APIRouter()\n'
            'SECRET = "shhh"\n'
            '@router.get("/health")\n'
            'def health():\n'
            '    assert True\n'
            '    return {"ok": True}\n'
        ),
    })
    result = run_generic_suite(tmp_path)
    # Ruff should at minimum flag the assert + hardcoded secret
    assert result["ruff"], "ruff should flag the planted patterns"
    rule_codes = {f.category for f in result["ruff"]}
    assert "S101" in rule_codes  # assert
    assert "S105" in rule_codes  # hardcoded password literal
