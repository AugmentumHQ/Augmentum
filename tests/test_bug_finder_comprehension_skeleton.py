"""Tests for the deterministic comprehension skeleton.

Pure logic + a fake ContainerManager that satisfies the ``run_command``
contract the skeleton builder relies on. We cover:

  * The pure helpers (route-decorator regex, candidate-file picker).
  * The render path — output is the prompt-injected text, so changes
    to the format land here loudly.
  * Mocked-CM happy path — verifies the skeleton populates languages,
    framework, subsystems, routes.
  * Mocked-CM degraded path — when individual commands return empty,
    a partial skeleton + discovery notes still come out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from augmentum.bug_finder.comprehension_skeleton import (
    CodebaseSkeleton,
    RouteHint,
    SubsystemHint,
    _ROUTE_DECORATOR_RE,
    _pick_candidate_pillar_files,
    build_skeleton,
)


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_route_decorator_regex_matches_common_shapes() -> None:
    """The regex needs to recognize the major frameworks' decorator shapes."""
    cases = [
        ("@app.get('/api/health')",                       "GET",       "/api/health"),
        ("@router.post('/api/items')",                    "POST",      "/api/items"),
        ("@api_router.delete('/x/{id}')",                 "DELETE",    "/x/{id}"),
        ("@app.websocket('/ws')",                         "WEBSOCKET", "/ws"),
        ('@router.put("/y/{id}")',                        "PUT",       "/y/{id}"),
        ("@app.route('/hello', methods=['POST'])",        "ROUTE",     "/hello"),
    ]
    for line, expected_method, expected_path in cases:
        m = _ROUTE_DECORATOR_RE.search(line)
        assert m is not None, f"failed on: {line}"
        assert m.group(1).upper() == expected_method
        assert m.group(2) == expected_path


def test_route_decorator_regex_rejects_non_routes() -> None:
    """Decorator-shaped lines that aren't routes shouldn't match."""
    assert _ROUTE_DECORATOR_RE.search("@pytest.fixture") is None
    assert _ROUTE_DECORATOR_RE.search("@property") is None
    assert _ROUTE_DECORATOR_RE.search("@dataclass(frozen=True)") is None


def test_pick_candidate_pillar_files_takes_top_subsystem_inits() -> None:
    """The candidate-files heuristic must pull the most-likely-pillar
    files: top-N subsystem __init__.py plus a few routes."""
    subsystems = (
        SubsystemHint(path="augmentum/auth",       file_count=42),
        SubsystemHint(path="augmentum/narrative",  file_count=88),
        SubsystemHint(path="augmentum/coder",      file_count=120),
    )
    routes = (
        RouteHint(method="GET",  path="/api/x",
                  file="augmentum/proxy/foo_routes.py", line=10),
        RouteHint(method="POST", path="/api/y",
                  file="augmentum/proxy/foo_routes.py", line=22),
        RouteHint(method="GET",  path="/api/z",
                  file="augmentum/proxy/bar_routes.py", line=11),
    )
    settings_files = ("config.py", "pyproject.toml")
    picked = _pick_candidate_pillar_files(
        subsystems, routes, settings_files, framework="fastapi",
    )
    # Each subsystem's __init__.py should be in the picks
    for s in subsystems:
        assert f"{s.path}/__init__.py" in picked
    # At least one route file
    assert any(p.endswith("_routes.py") for p in picked)
    # Config file
    assert "config.py" in picked


def test_pick_candidate_pillar_files_deduplicates() -> None:
    """The same path shouldn't show up twice even if heuristics overlap."""
    subsystems = (
        SubsystemHint(path="src", file_count=5),
    )
    routes = (
        RouteHint(method="GET", path="/a", file="src/__init__.py", line=1),
    )
    picked = _pick_candidate_pillar_files(
        subsystems, routes, settings_files=(), framework="",
    )
    # src/__init__.py from both the subsystem AND the route file path
    assert picked.count("src/__init__.py") == 1


# ---------------------------------------------------------------------------
# CodebaseSkeleton.render_for_prompt
# ---------------------------------------------------------------------------


def test_render_empty_skeleton_still_produces_structured_output() -> None:
    """A zero-content skeleton renders the header at minimum so the
    comprehender knows the builder ran."""
    skel = CodebaseSkeleton()
    out = skel.render_for_prompt()
    assert "Deterministic skeleton" in out
    assert "Workspace root" in out


def test_render_populated_skeleton_includes_all_sections() -> None:
    skel = CodebaseSkeleton(
        languages=("python", "javascript"),
        framework="fastapi",
        test_command="pytest -x",
        file_count_total=1000,
        source_file_count=800,
        head_sha="abc1234567890",
        subsystems=(
            SubsystemHint(
                path="augmentum/auth", file_count=12,
                has_routes=True, top_docstring="Multi-tenant authentication.",
            ),
        ),
        routes=(
            RouteHint(method="GET", path="/api/health",
                      file="augmentum/proxy/health_routes.py", line=5),
        ),
        settings_files=("config.py",),
        candidate_pillars_files=("augmentum/auth/__init__.py",),
        discovery_notes=("first run on this workspace",),
    )
    out = skel.render_for_prompt()
    assert "python, javascript" in out
    assert "fastapi" in out
    assert "pytest -x" in out
    assert "1,000" in out  # total file count
    assert "abc1234567" in out
    assert "augmentum/auth" in out
    assert "Multi-tenant" in out
    assert "GET    /api/health" in out
    assert "config.py" in out
    assert "first run" in out


# ---------------------------------------------------------------------------
# Mocked-CM build_skeleton — happy path
# ---------------------------------------------------------------------------


@dataclass
class _MockCM:
    """Minimal ContainerManager stand-in. Matches against substrings in
    the joined command and returns the first hit's value (callable or
    string)."""

    responses: list[tuple[Any, Any]] = field(default_factory=list)
    seen: list[str] = field(default_factory=list)

    async def run_command(
        self, workspace_id: str, cmd: list[str], timeout: float = 30.0,
        *, idle_timeout: float | None = None,
        progress_path: str | None = None,
        on_chunk: Any | None = None,
    ) -> str:
        joined = " ".join(cmd)
        self.seen.append(joined)
        for predicate, value in self.responses:
            if callable(predicate) and predicate(joined):
                return value(joined) if callable(value) else value
        return ""


@pytest.mark.asyncio
async def test_build_skeleton_happy_path_populates_all_dimensions() -> None:
    """Simulate a small Python+FastAPI workspace and confirm the
    skeleton captures languages, framework, test command, and a
    subsystem with a route."""
    def respond(joined: str) -> str:
        # Language detection — count .py files
        if "find" in joined and "*.py" in joined and "wc -l" in joined:
            return "42"
        if "find" in joined and "wc -l" in joined and "-type f -not -path" in joined:
            return "100"  # total file count
        # Framework detection
        if "grep" in joined and "from fastapi" in joined:
            return "/workspace/main.py"
        # Test command — pyproject.toml present
        if "test -f" in joined and "pyproject.toml" in joined:
            return "found"
        # Git HEAD
        if "git rev-parse HEAD" in joined:
            return "abc1234567890def"
        # Subsystem dirs
        if "find" in joined and "-mindepth 2 -maxdepth 3 -type d" in joined:
            return "/workspace/augmentum/auth\n/workspace/augmentum/proxy"
        # Per-subsystem file count
        if "find" in joined and "/auth" in joined and "wc -l" in joined:
            return "12"
        if "find" in joined and "/proxy" in joined and "wc -l" in joined:
            return "20"
        # Per-subsystem routes check
        if "_routes.py" in joined and "head -1" in joined and "/proxy" in joined:
            return "/workspace/augmentum/proxy/foo_routes.py"
        # __init__.py read
        if "head -25" in joined and "/auth/__init__.py" in joined:
            return '"""Multi-tenant authentication."""\n'
        # Route files
        if "find" in joined and "_routes.py" in joined and "head -60" in joined:
            return "/workspace/augmentum/proxy/foo_routes.py"
        # Route grep
        if "grep -n -E" in joined and "foo_routes.py" in joined:
            return (
                "5:@router.get('/api/health')\n"
                "10:@router.post('/api/items')"
            )
        # Settings files
        if "find" in joined and "config.py" in joined and "head -3" in joined:
            return "/workspace/config.py"
        if "find" in joined and "pyproject.toml" in joined and "head -3" in joined:
            return "/workspace/pyproject.toml"
        return ""

    cm = _MockCM(responses=[(lambda _c: True, respond)])
    skel = await build_skeleton(cm=cm, workspace_id="ws", root="/workspace")

    assert "python" in skel.languages
    assert skel.framework == "fastapi"
    assert skel.test_command == "pytest"
    assert skel.has_git
    assert skel.head_sha.startswith("abc1234567890")
    assert skel.source_file_count > 0
    assert skel.file_count_total > 0
    # Subsystems sorted by file_count desc; /proxy (20) > /auth (12)
    assert len(skel.subsystems) >= 1
    paths = [s.path for s in skel.subsystems]
    assert "augmentum/proxy" in paths or "augmentum/auth" in paths
    # Routes
    assert any(r.method == "GET" and r.path == "/api/health" for r in skel.routes)
    assert any(r.method == "POST" and r.path == "/api/items" for r in skel.routes)


@pytest.mark.asyncio
async def test_build_skeleton_degraded_returns_partial_skeleton() -> None:
    """When every command returns empty, we should still get a
    well-formed (mostly-empty) skeleton + discovery notes calling out
    the no-language case."""
    cm = _MockCM(responses=[(lambda _c: True, lambda _c: "")])
    skel = await build_skeleton(cm=cm, workspace_id="ws", root="/workspace")
    assert skel.languages == ()
    assert skel.routes == ()
    assert skel.subsystems == ()
    assert any("no source files" in n for n in skel.discovery_notes)


@pytest.mark.asyncio
async def test_build_skeleton_notes_framework_without_routes() -> None:
    """FastAPI detected but no route files found is a useful signal —
    surface it so the comprehender knows to look harder."""
    def respond(joined: str) -> str:
        # No source files but framework signature exists
        if "grep" in joined and "from fastapi" in joined:
            return "/workspace/main.py"
        # Some Python files (so language detection works)
        if "find" in joined and "*.py" in joined and "wc -l" in joined:
            return "5"
        if "find" in joined and "wc -l" in joined and "-type f" in joined:
            return "10"
        return ""

    cm = _MockCM(responses=[(lambda _c: True, respond)])
    skel = await build_skeleton(cm=cm, workspace_id="ws", root="/workspace")
    assert skel.framework == "fastapi"
    assert skel.routes == ()
    assert any("no routes found" in n for n in skel.discovery_notes)


# ---------------------------------------------------------------------------
# CodebaseSkeleton dataclass shape
# ---------------------------------------------------------------------------


def test_skeleton_is_frozen() -> None:
    s = CodebaseSkeleton()
    try:
        s.framework = "x"   # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("CodebaseSkeleton should be frozen")
