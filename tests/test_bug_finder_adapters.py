"""Framework adapter tests.

Focus on the FastAPI adapter (Augmentum's own framework) — covers the
cached-routes-json path, the live AST scan path, and the test-command
detector. The NullAdapter is exercised via the registry's
``adapter_for_framework("unknown")`` fall-back.
"""

from __future__ import annotations

import json
from pathlib import Path

from augmentum.bug_finder.adapters import (
    AdapterRouteHint,
    FastAPIAdapter,
    NullAdapter,
    adapter_for_framework,
)


def _mk(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_maps_fastapi_to_adapter() -> None:
    adapter = adapter_for_framework("fastapi")
    assert isinstance(adapter, FastAPIAdapter)
    assert adapter.name == "fastapi"


def test_registry_falls_back_to_null_for_unknown() -> None:
    adapter = adapter_for_framework("not-a-framework")
    assert isinstance(adapter, NullAdapter)
    assert adapter.name == "null"


def test_registry_handles_empty_input_with_null() -> None:
    assert isinstance(adapter_for_framework(""), NullAdapter)
    assert isinstance(adapter_for_framework("   "), NullAdapter)


# ---------------------------------------------------------------------------
# FastAPIAdapter.list_routes — cached path
# ---------------------------------------------------------------------------


def test_fastapi_reads_cached_routes_when_present(tmp_path: Path) -> None:
    cache = {
        "endpoints": [
            {
                "method": "GET", "path": "/api/health",
                "handler": "health_check",
                "file": "src/api.py", "line": 12,
            },
            {
                "method": "POST", "path": "/api/users",
                "handler": "create_user",
                "file": "src/users.py", "line": 88,
            },
        ],
    }
    _mk(tmp_path, {
        ".claude/skills/augmentum-dev/references/routes.json":
            json.dumps(cache),
    })
    routes = FastAPIAdapter().list_routes(tmp_path)
    assert len(routes) == 2
    assert routes[0] == AdapterRouteHint(
        method="GET", path="/api/health",
        handler="health_check", file="src/api.py", line=12,
    )


def test_fastapi_cached_path_skips_malformed_rows(tmp_path: Path) -> None:
    cache = {
        "endpoints": [
            {"method": "GET", "path": "/ok", "file": "x.py"},
            {"method": "GET", "path": "", "file": "x.py"},        # dropped
            {"path": "/missing-method", "file": "x.py"},          # dropped
            "not a dict",                                          # dropped
        ],
    }
    _mk(tmp_path, {
        ".claude/skills/augmentum-dev/references/routes.json":
            json.dumps(cache),
    })
    routes = FastAPIAdapter().list_routes(tmp_path)
    assert len(routes) == 1
    assert routes[0].path == "/ok"


# ---------------------------------------------------------------------------
# FastAPIAdapter.list_routes — live AST scan
# ---------------------------------------------------------------------------


def test_fastapi_live_scan_finds_router_get(tmp_path: Path) -> None:
    src = (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "\n"
        "@router.get('/api/health')\n"
        "async def health_check():\n"
        "    return {'ok': True}\n"
        "\n"
        "@router.post('/api/items')\n"
        "def create_item(body: dict):\n"
        "    return body\n"
    )
    _mk(tmp_path, {"src/api_routes.py": src})
    routes = FastAPIAdapter().list_routes(tmp_path)
    paths = sorted(r.path for r in routes)
    assert "/api/health" in paths
    assert "/api/items" in paths
    methods = {r.path: r.method for r in routes}
    assert methods["/api/health"] == "GET"
    assert methods["/api/items"] == "POST"


def test_fastapi_live_scan_picks_up_app_prefix_too(tmp_path: Path) -> None:
    """``@app.X('...')`` (no router intermediate) should also resolve.
    Common in single-file FastAPI apps."""
    src = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "\n"
        "@app.get('/hello')\n"
        "def hello():\n"
        "    return 'hi'\n"
    )
    _mk(tmp_path, {"routes.py": src})
    routes = FastAPIAdapter().list_routes(tmp_path)
    assert any(r.path == "/hello" and r.method == "GET" for r in routes)


def test_fastapi_live_scan_skips_non_route_decorators(tmp_path: Path) -> None:
    """``@property`` and other non-route decorators must not pollute
    the route list. The verb is part of the match guard."""
    src = (
        "@property\n"
        "def foo(self): return 1\n"
        "\n"
        "@cache.memoize(ttl=60)\n"
        "def bar(): pass\n"
    )
    _mk(tmp_path, {"routes.py": src})
    routes = FastAPIAdapter().list_routes(tmp_path)
    assert routes == []


def test_fastapi_live_scan_handles_websocket_decorator(tmp_path: Path) -> None:
    src = (
        "@router.websocket('/ws')\n"
        "async def ws_handler(socket):\n"
        "    pass\n"
    )
    _mk(tmp_path, {"routes.py": src})
    routes = FastAPIAdapter().list_routes(tmp_path)
    assert len(routes) == 1
    assert routes[0].method == "WEBSOCKET"
    assert routes[0].path == "/ws"


def test_fastapi_live_scan_tolerates_syntax_error(tmp_path: Path) -> None:
    """A broken route file shouldn't black-hole the whole scan. The
    regex fallback should still surface what's syntactically obvious."""
    src = (
        "@router.get('/api/x')\n"
        "def broken(  # missing colon + paren\n"
        "    return 1\n"
    )
    _mk(tmp_path, {"routes.py": src})
    routes = FastAPIAdapter().list_routes(tmp_path)
    # Regex fallback recovers the route
    assert any(r.path == "/api/x" for r in routes)


def test_fastapi_returns_empty_on_no_routes_no_cache(tmp_path: Path) -> None:
    """Bare project with no route files at all — empty result, no
    exception. Caller gracefully degrades."""
    routes = FastAPIAdapter().list_routes(tmp_path)
    assert routes == []


# ---------------------------------------------------------------------------
# settings_files / test_command / identify_route_file
# ---------------------------------------------------------------------------


def test_fastapi_lists_known_settings_files(tmp_path: Path) -> None:
    _mk(tmp_path, {
        "config.py": "X = 1",
        "pyproject.toml": "[project]\nname = 'x'",
        ".env.example": "X=1",
    })
    hints = FastAPIAdapter().list_settings_files(tmp_path)
    files = {h.file for h in hints}
    assert "config.py" in files
    assert "pyproject.toml" in files
    assert ".env.example" in files


def test_fastapi_settings_skips_venv_and_node_modules(tmp_path: Path) -> None:
    _mk(tmp_path, {
        ".venv/lib/site-packages/some/config.py": "X = 1",
        "node_modules/dep/package.json": "{}",
        "src/config.py": "X = 1",
    })
    files = {h.file for h in FastAPIAdapter().list_settings_files(tmp_path)}
    assert "src/config.py" in files
    assert not any(".venv" in f for f in files)
    assert not any("node_modules" in f for f in files)


def test_identify_route_file_recognizes_convention() -> None:
    a = FastAPIAdapter()
    assert a.identify_route_file("src/auth_routes.py")
    assert a.identify_route_file("api/routes.py")
    assert a.identify_route_file("router.py")
    assert not a.identify_route_file("models.py")
    assert not a.identify_route_file("config.py")


def test_identify_test_command_pytest_detection(tmp_path: Path) -> None:
    _mk(tmp_path, {"pyproject.toml": "[project]\nname='x'"})
    assert FastAPIAdapter().identify_test_command(tmp_path) == "pytest"


def test_identify_test_command_empty_when_unrecognized(tmp_path: Path) -> None:
    assert FastAPIAdapter().identify_test_command(tmp_path) == ""


# ---------------------------------------------------------------------------
# NullAdapter
# ---------------------------------------------------------------------------


def test_null_adapter_returns_empty_everywhere(tmp_path: Path) -> None:
    a = NullAdapter()
    assert a.list_routes(tmp_path) == []
    assert a.list_jobs(tmp_path) == []
    assert a.list_settings_files(tmp_path) == []
    assert a.identify_route_file("anything.py") is False
    assert a.identify_test_command(tmp_path) == ""
