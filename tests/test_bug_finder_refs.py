"""Tests for the structured-reference loaders."""

from __future__ import annotations

import json
from pathlib import Path

from augmentum.bug_finder import refs
from augmentum.bug_finder.refs import (
    FrontendCallRecord,
    RouteRecord,
    SuppressionRule,
)


def _mk(tmp_path: Path, name: str, payload: dict) -> Path:
    ref_dir = tmp_path / ".claude" / "skills" / "augmentum-dev" / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    refs.clear_cache()
    return tmp_path


# ---------------------------------------------------------------------------
# routes.json
# ---------------------------------------------------------------------------


def test_load_routes_returns_typed_records(tmp_path: Path) -> None:
    _mk(tmp_path, "routes.json", {
        "endpoints": [
            {"method": "GET", "path": "/api/health",
             "handler": "h", "file": "x.py", "line": 5},
        ],
    })
    rows = refs.load_routes(tmp_path)
    assert rows == [
        RouteRecord(method="GET", path="/api/health",
                    handler="h", file="x.py", line=5),
    ]


def test_load_routes_returns_empty_for_missing_file(tmp_path: Path) -> None:
    refs.clear_cache()
    assert refs.load_routes(tmp_path) == []


def test_load_routes_skips_malformed_rows(tmp_path: Path) -> None:
    _mk(tmp_path, "routes.json", {
        "endpoints": [
            {"method": "GET", "path": "/ok", "file": "x.py"},
            {"method": "", "path": "/bad", "file": "x.py"},     # dropped
            {"method": "GET", "path": "", "file": "x.py"},      # dropped
            {"method": "GET", "path": "/no-file"},               # dropped
            "string",                                             # dropped
        ],
    })
    rows = refs.load_routes(tmp_path)
    assert len(rows) == 1
    assert rows[0].path == "/ok"


def test_list_routes_filters_by_method(tmp_path: Path) -> None:
    _mk(tmp_path, "routes.json", {
        "endpoints": [
            {"method": "GET",  "path": "/api/a", "file": "x.py"},
            {"method": "POST", "path": "/api/b", "file": "x.py"},
            {"method": "GET",  "path": "/api/c", "file": "x.py"},
        ],
    })
    only_get = refs.list_routes(tmp_path, method="get")
    assert {r.path for r in only_get} == {"/api/a", "/api/c"}


def test_list_routes_filters_by_path_substring(tmp_path: Path) -> None:
    _mk(tmp_path, "routes.json", {
        "endpoints": [
            {"method": "GET", "path": "/api/auth/login", "file": "x.py"},
            {"method": "GET", "path": "/api/billing/charge", "file": "x.py"},
            {"method": "GET", "path": "/api/auth/logout", "file": "x.py"},
        ],
    })
    auth = refs.list_routes(tmp_path, path_substr="/auth/")
    assert len(auth) == 2
    assert all("/auth/" in r.path for r in auth)


def test_list_routes_respects_limit(tmp_path: Path) -> None:
    _mk(tmp_path, "routes.json", {
        "endpoints": [
            {"method": "GET", "path": f"/api/x{i}", "file": "x.py"}
            for i in range(20)
        ],
    })
    assert len(refs.list_routes(tmp_path, limit=5)) == 5


# ---------------------------------------------------------------------------
# frontend_api_calls.json
# ---------------------------------------------------------------------------


def test_load_frontend_calls_returns_typed_records(tmp_path: Path) -> None:
    _mk(tmp_path, "frontend_api_calls.json", {
        "calls": [
            {"url": "/api/items", "method": "GET", "file": "ui/foo.js"},
        ],
    })
    rows = refs.load_frontend_calls(tmp_path)
    assert rows == [
        FrontendCallRecord(url="/api/items", method="GET",
                           file="ui/foo.js"),
    ]


def test_find_callers_of_matches_substring(tmp_path: Path) -> None:
    _mk(tmp_path, "frontend_api_calls.json", {
        "calls": [
            {"url": "/api/auth/login",  "method": "POST", "file": "ui/auth.js"},
            {"url": "/api/auth/logout", "method": "POST", "file": "ui/auth.js"},
            {"url": "/api/billing",     "method": "GET",  "file": "ui/bill.js"},
        ],
    })
    auth_callers = refs.find_callers_of(tmp_path, path_substr="/auth/")
    assert len(auth_callers) == 2
    assert all(c.file == "ui/auth.js" for c in auth_callers)


# ---------------------------------------------------------------------------
# security_exceptions.json — suppressions
# ---------------------------------------------------------------------------


def test_load_suppressions_returns_typed_records(tmp_path: Path) -> None:
    _mk(tmp_path, "security_exceptions.json", {
        "exceptions": [
            {
                "id": "cors-wildcard",
                "pattern": "cors.*\\*",
                "files": ["augmentum/proxy/server.py"],
            },
        ],
    })
    rules = refs.load_suppressions(tmp_path)
    assert rules == [
        SuppressionRule(
            rule_id="cors-wildcard",
            pattern="cors.*\\*",
            files=("augmentum/proxy/server.py",),
        ),
    ]


def test_is_finding_suppressed_matches_when_file_and_pattern_both_match(
    tmp_path: Path,
) -> None:
    _mk(tmp_path, "security_exceptions.json", {
        "exceptions": [
            {"id": "rule-x", "pattern": "cors-wildcard",
             "files": ["server.py"]},
        ],
    })
    rid = refs.is_finding_suppressed(
        tmp_path,
        file="augmentum/proxy/server.py", pattern="cors-wildcard",
    )
    assert rid == "rule-x"


def test_is_finding_suppressed_returns_empty_when_no_match(
    tmp_path: Path,
) -> None:
    _mk(tmp_path, "security_exceptions.json", {
        "exceptions": [
            {"id": "rule-x", "pattern": "cors-wildcard",
             "files": ["server.py"]},
        ],
    })
    # File doesn't match
    assert refs.is_finding_suppressed(
        tmp_path, file="some/other.py", pattern="cors-wildcard",
    ) == ""
    # Pattern doesn't match
    assert refs.is_finding_suppressed(
        tmp_path, file="server.py", pattern="something-else",
    ) == ""


# ---------------------------------------------------------------------------
# has_augmentum_dev_refs
# ---------------------------------------------------------------------------


def test_has_refs_false_when_directory_missing(tmp_path: Path) -> None:
    refs.clear_cache()
    assert not refs.has_augmentum_dev_refs(tmp_path)


def test_has_refs_true_when_any_json_present(tmp_path: Path) -> None:
    _mk(tmp_path, "routes.json", {"endpoints": []})
    assert refs.has_augmentum_dev_refs(tmp_path)


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


def test_cache_avoids_re_reading_same_root(tmp_path: Path) -> None:
    """Two loads of the same file should hit the cache after the first."""
    _mk(tmp_path, "routes.json", {
        "endpoints": [{"method": "GET", "path": "/a", "file": "x.py"}],
    })
    a = refs.load_routes(tmp_path)
    # Mutate the file under cache; cached load should still return old
    (tmp_path / ".claude" / "skills" / "augmentum-dev" / "references"
        / "routes.json").write_text(json.dumps({
            "endpoints": [
                {"method": "GET", "path": "/b", "file": "x.py"},
            ],
        }), encoding="utf-8")
    b = refs.load_routes(tmp_path)
    assert a == b   # cache wins
    refs.clear_cache()
    c = refs.load_routes(tmp_path)
    assert c[0].path == "/b"   # new payload after clear
