"""Symbolic patch re-verification gate tests.

Pure-logic tests: parsing Semgrep JSON output + extracting file paths
from unified diffs. The container-side integration is tier 3 (live)
and lives in `--run-live` flows once Semgrep is in the container
image; until then the orchestrator's graceful-skip path keeps the
gate from interfering.
"""

from __future__ import annotations

import json

from augmentum.bug_finder.symbolic_gate import (
    SymbolicFinding,
    SymbolicGateResult,
    _parse_semgrep_json,
    extract_patched_files,
)


# ---------------------------------------------------------------------------
# extract_patched_files
# ---------------------------------------------------------------------------


def test_extract_patched_files_simple() -> None:
    diff = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "index abc..def 100644\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+line2_fixed\n"
        " line3\n"
    )
    assert extract_patched_files(diff) == ["src/foo.py"]


def test_extract_patched_files_multiple() -> None:
    diff = (
        "diff --git a/a.py b/a.py\n"
        "@@ -1 +1 @@\n"
        "diff --git a/b.py b/b.py\n"
        "@@ -1 +1 @@\n"
        "diff --git a/sub/c.py b/sub/c.py\n"
        "@@ -1 +1 @@\n"
    )
    assert extract_patched_files(diff) == ["a.py", "b.py", "sub/c.py"]


def test_extract_patched_files_dedupes() -> None:
    """Same file referenced twice (e.g., rename then edit) → one entry."""
    diff = (
        "diff --git a/x.py b/x.py\n"
        "diff --git a/x.py b/x.py\n"
    )
    assert extract_patched_files(diff) == ["x.py"]


def test_extract_patched_files_handles_deletions() -> None:
    """A deletion shows `b/dev/null` — should be excluded; the `a/`
    side is irrelevant for post-patch scanning."""
    diff = (
        "diff --git a/old.py b/old.py\n"
        "deleted file mode 100644\n"
        "diff --git a/kept.py b/kept.py\n"
    )
    # Both still have b/<name> paths; only /dev/null would be excluded
    files = extract_patched_files(diff)
    assert "kept.py" in files
    assert "old.py" in files


def test_extract_patched_files_empty_diff() -> None:
    assert extract_patched_files("") == []
    assert extract_patched_files("no diff here\n") == []


# ---------------------------------------------------------------------------
# _parse_semgrep_json
# ---------------------------------------------------------------------------


def test_parse_semgrep_json_empty_results() -> None:
    """No findings — gate should pass."""
    out = json.dumps({"results": [], "errors": []})
    assert _parse_semgrep_json(out) == []


def test_parse_semgrep_json_normalizes_one_finding() -> None:
    out = json.dumps({
        "results": [
            {
                "check_id": "python.lang.security.audit.dangerous-pickle",
                "path": "src/cache.py",
                "start": {"line": 12, "col": 1},
                "end": {"line": 14, "col": 25},
                "extra": {
                    "severity": "ERROR",
                    "message": "pickle.loads on untrusted input",
                },
            },
        ],
    })
    findings = _parse_semgrep_json(out)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "python.lang.security.audit.dangerous-pickle"
    assert f.file == "src/cache.py"
    assert f.line_start == 12
    assert f.line_end == 14
    assert f.severity == "ERROR"
    assert "pickle.loads" in f.message


def test_parse_semgrep_json_tolerates_warning_prefix() -> None:
    """Semgrep sometimes prepends stderr warnings to stdout; the parser
    should locate the JSON document regardless."""
    out = (
        "Loading rules from semgrep registry...\n"
        "WARNING: 1 rule not found\n"
        + json.dumps({"results": [{
            "check_id": "py.bad", "path": "f.py",
            "start": {"line": 1}, "end": {"line": 1},
            "extra": {"severity": "WARNING", "message": "bad"},
        }]})
    )
    findings = _parse_semgrep_json(out)
    assert len(findings) == 1
    assert findings[0].rule_id == "py.bad"


def test_parse_semgrep_json_invalid_json_returns_empty() -> None:
    """Tolerate malformed output — graceful skip rather than crash."""
    assert _parse_semgrep_json("not json at all") == []
    assert _parse_semgrep_json("{not valid}") == []


def test_parse_semgrep_json_missing_results_array_returns_empty() -> None:
    assert _parse_semgrep_json(json.dumps({"errors": ["whatever"]})) == []


def test_parse_semgrep_json_handles_missing_fields() -> None:
    """A row without extra/severity/message shouldn't crash the parser."""
    out = json.dumps({
        "results": [
            {"check_id": "x", "path": "p.py", "start": {"line": 5}, "end": {"line": 5}},
        ],
    })
    findings = _parse_semgrep_json(out)
    assert len(findings) == 1
    assert findings[0].severity == "UNKNOWN"
    assert findings[0].message == ""


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


def test_result_summary_passed() -> None:
    r = SymbolicGateResult(passed=True, semgrep_version="1.x")
    assert "passed" in r.summary
    assert "no new findings" in r.summary


def test_result_summary_skipped() -> None:
    r = SymbolicGateResult(
        passed=True, skipped=True, skip_reason="not installed",
    )
    assert "skipped" in r.summary
    assert "not installed" in r.summary


def test_result_summary_rejected() -> None:
    r = SymbolicGateResult(passed=False, new_findings=[
        SymbolicFinding(
            rule_id="x", file="f.py", line_start=1, line_end=1,
            severity="ERROR", message="bad",
        ),
        SymbolicFinding(
            rule_id="y", file="f.py", line_start=2, line_end=2,
            severity="ERROR", message="bad2",
        ),
    ])
    assert "rejected" in r.summary
    assert "2 new finding" in r.summary
