"""Wiring inspection tests — middleware chain, decorator chain, constant lookup.

Synthetic on-disk projects verify the AST walkers return the right
shapes for the FP patterns the auth audit surfaced (ASGI middleware
order, decorator presence on handlers, feature flag values).
"""

from __future__ import annotations

from pathlib import Path

from augmentum.bug_finder.wiring import (
    decorators_on,
    extract_middleware_chain,
    get_constant,
    render_middleware_brief,
    runs_first_to_last,
    trace_origin,
)


def _mkproject(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Middleware chain
# ---------------------------------------------------------------------------


def test_middleware_chain_orders_by_registration(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "server.py": (
            "def create():\n"
            "    app = FastAPI()\n"
            "    app.add_middleware(SecurityHeaders)\n"
            "    app.add_middleware(AuthMiddleware)\n"
            "    app.add_middleware(FabricPeerMiddleware)\n"
            "    return app\n"
        ),
    })
    entries = extract_middleware_chain(root)
    assert [e.name for e in entries] == [
        "SecurityHeaders", "AuthMiddleware", "FabricPeerMiddleware",
    ]
    assert [e.order for e in entries] == [0, 1, 2]
    # ASGI: last registered runs first
    flow = runs_first_to_last(entries)
    assert [e.name for e in flow] == [
        "FabricPeerMiddleware", "AuthMiddleware", "SecurityHeaders",
    ]


def test_middleware_chain_captures_kwargs(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "wire.py": (
            "def install(app):\n"
            "    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True)\n"
        ),
    })
    entries = extract_middleware_chain(root)
    assert len(entries) == 1
    assert entries[0].name == "CORSMiddleware"
    assert "allow_origins" in entries[0].kwargs_repr
    assert "['*']" in entries[0].kwargs_repr["allow_origins"]
    assert entries[0].kwargs_repr["allow_credentials"] == "True"


def test_middleware_chain_spans_files(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "a.py": "def f(app):\n    app.add_middleware(A)\n",
        "b.py": "def g(app):\n    app.add_middleware(B)\n",
    })
    entries = extract_middleware_chain(root)
    assert {e.name for e in entries} == {"A", "B"}
    # Sort is by (file, line) — deterministic
    assert entries[0].file < entries[1].file or entries[0].line < entries[1].line


def test_middleware_chain_skips_unrelated_methods(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "noise.py": (
            "def setup(app):\n"
            "    app.add_event_handler('startup', _on_start)\n"
            "    app.add_route('/x', _handler)\n"
        ),
    })
    assert extract_middleware_chain(root) == []


def test_middleware_chain_ignores_syntax_errors(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "broken.py": "def f(:\n    pass\n",
        "ok.py": "def g(app):\n    app.add_middleware(OK)\n",
    })
    entries = extract_middleware_chain(root)
    assert [e.name for e in entries] == ["OK"]


def test_render_middleware_brief_includes_flow(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "s.py": (
            "def f(app):\n"
            "    app.add_middleware(Auth)\n"
            "    app.add_middleware(Fabric)\n"
        ),
    })
    entries = extract_middleware_chain(root)
    brief = render_middleware_brief(entries)
    assert "Auth" in brief
    assert "Fabric" in brief
    # First-to-last: Fabric runs first
    assert brief.index("Fabric -> Auth") > 0


def test_render_middleware_brief_empty(tmp_path: Path) -> None:
    assert render_middleware_brief([]) == ""


# ---------------------------------------------------------------------------
# Decorator chain
# ---------------------------------------------------------------------------


def test_decorators_on_returns_full_chain(tmp_path: Path) -> None:
    src = (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "\n"
        "@router.post('/login')\n"
        "@require_auth\n"
        "@rate_limited(rpm=10)\n"
        "async def login(req):\n"
        "    return {'ok': True}\n"
    )
    root = _mkproject(tmp_path, {"auth_routes.py": src})
    # Line 7 = async def login
    decos = decorators_on(root, file="auth_routes.py", line=7)
    names = [d.name for d in decos]
    assert names == ["router.post", "require_auth", "rate_limited"]
    # First decorator captured its path arg
    assert decos[0].args_repr == ["'/login'"]
    # Third has rpm kwarg
    assert decos[2].kwargs_repr == {"rpm": "10"}


def test_decorators_on_matches_decorator_line(tmp_path: Path) -> None:
    """A finding flagged at a decorator line still resolves to the function."""
    src = (
        "@require_auth\n"          # line 1
        "@rate_limited(rpm=5)\n"   # line 2
        "def handler():\n"         # line 3
        "    pass\n"               # line 4
    )
    root = _mkproject(tmp_path, {"r.py": src})
    decos = decorators_on(root, file="r.py", line=2)
    assert [d.name for d in decos] == ["require_auth", "rate_limited"]


def test_decorators_on_empty_when_no_decorators(tmp_path: Path) -> None:
    src = "def plain():\n    return 1\n"
    root = _mkproject(tmp_path, {"p.py": src})
    assert decorators_on(root, file="p.py", line=1) == []


def test_decorators_on_unknown_file_returns_empty(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {"x.py": "def f(): pass\n"})
    assert decorators_on(root, file="missing.py", line=1) == []


def test_decorators_on_picks_innermost_nested(tmp_path: Path) -> None:
    src = (
        "@outer\n"
        "def parent():\n"
        "    @inner\n"
        "    def child():\n"
        "        return 1\n"
        "    return child\n"
    )
    root = _mkproject(tmp_path, {"n.py": src})
    # line 4 = inside child
    decos = decorators_on(root, file="n.py", line=4)
    assert [d.name for d in decos] == ["inner"]


# ---------------------------------------------------------------------------
# Constant lookup
# ---------------------------------------------------------------------------


def test_get_constant_literal_string(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "config.py": "API_VERSION = 'v2'\nDEBUG = True\n",
    })
    binding = get_constant(root, "API_VERSION")
    assert binding is not None
    assert binding.confident is True
    assert binding.value_repr == "'v2'"
    assert binding.file == "config.py"


def test_get_constant_literal_bool(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {"c.py": "ALLOW_INSECURE = False\n"})
    binding = get_constant(root, "ALLOW_INSECURE")
    assert binding is not None
    assert binding.value_repr == "False"
    assert binding.confident is True


def test_get_constant_literal_container(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "c.py": "CORS_ORIGINS = ['https://a.com', 'https://b.com']\n",
    })
    binding = get_constant(root, "CORS_ORIGINS")
    assert binding is not None
    assert binding.confident is True
    assert "https://a.com" in binding.value_repr


def test_get_constant_runtime_not_confident(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "c.py": "import os\nAPI_KEY = os.environ['API_KEY']\n",
    })
    binding = get_constant(root, "API_KEY")
    assert binding is not None
    assert binding.confident is False
    assert "os.environ" in binding.value_repr


def test_get_constant_annotated_assign(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "c.py": "MAX_RETRIES: int = 5\n",
    })
    binding = get_constant(root, "MAX_RETRIES")
    assert binding is not None
    assert binding.value_repr == "5"
    assert binding.confident is True


def test_get_constant_missing_returns_none(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {"c.py": "X = 1\n"})
    assert get_constant(root, "NOT_THERE") is None


def test_get_constant_first_match_across_files(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "a.py": "SHARED = 'from_a'\n",
        "b.py": "SHARED = 'from_b'\n",
    })
    binding = get_constant(root, "SHARED")
    assert binding is not None
    # rglob ordering isn't stable across OS but the value must be one of the two
    assert binding.value_repr in ("'from_a'", "'from_b'")


# ---------------------------------------------------------------------------
# Skip / discovery behavior
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# trace_origin
# ---------------------------------------------------------------------------


def test_trace_origin_parameter(tmp_path: Path) -> None:
    src = (
        "def handle(request, token: str):\n"      # line 1
        "    if not token:\n"                      # line 2
        "        raise ValueError\n"               # line 3
        "    return token\n"                       # line 4
    )
    root = _mkproject(tmp_path, {"h.py": src})
    trace = trace_origin(root, file="h.py", line=4, var="token")
    assert trace is not None
    assert trace.origin_kind == "parameter"
    assert trace.confident is True
    assert "str" in trace.origin_expr
    assert "handle" in trace.note


def test_trace_origin_local_assignment(tmp_path: Path) -> None:
    src = (
        "def get_user(scope):\n"                     # 1
        "    user = scope.get('user')\n"             # 2
        "    if user is None:\n"                     # 3
        "        return None\n"                      # 4
        "    return user\n"                          # 5
    )
    root = _mkproject(tmp_path, {"g.py": src})
    trace = trace_origin(root, file="g.py", line=5, var="user")
    assert trace is not None
    assert trace.origin_kind == "assignment"
    assert trace.origin_expr == "scope.get('user')"
    assert trace.origin_line == 2


def test_trace_origin_picks_latest_local(tmp_path: Path) -> None:
    src = (
        "def f():\n"            # 1
        "    x = 1\n"           # 2
        "    x = 2\n"           # 3
        "    return x\n"        # 4
    )
    root = _mkproject(tmp_path, {"f.py": src})
    trace = trace_origin(root, file="f.py", line=4, var="x")
    assert trace is not None
    assert trace.origin_line == 3
    assert trace.origin_expr == "2"


def test_trace_origin_module_level(tmp_path: Path) -> None:
    src = (
        "API_VERSION = 'v2'\n"           # 1
        "\n"                              # 2
        "def use():\n"                    # 3
        "    return API_VERSION\n"        # 4
    )
    root = _mkproject(tmp_path, {"m.py": src})
    trace = trace_origin(root, file="m.py", line=4, var="API_VERSION")
    assert trace is not None
    assert trace.origin_kind == "assignment"
    assert trace.origin_expr == "'v2'"
    assert trace.note == "latest module assignment"


def test_trace_origin_for_target(tmp_path: Path) -> None:
    src = (
        "def f(rows):\n"                 # 1
        "    for row in rows:\n"         # 2
        "        process(row)\n"          # 3
    )
    root = _mkproject(tmp_path, {"l.py": src})
    trace = trace_origin(root, file="l.py", line=3, var="row")
    assert trace is not None
    assert trace.origin_kind == "for_target"
    assert "iter: rows" in trace.origin_expr


def test_trace_origin_with_target(tmp_path: Path) -> None:
    src = (
        "def f():\n"                            # 1
        "    with open('x') as fh:\n"            # 2
        "        return fh.read()\n"            # 3
    )
    root = _mkproject(tmp_path, {"w.py": src})
    trace = trace_origin(root, file="w.py", line=3, var="fh")
    assert trace is not None
    assert trace.origin_kind == "with_target"
    assert "open" in trace.origin_expr


def test_trace_origin_import(tmp_path: Path) -> None:
    src = (
        "from pathlib import Path\n"     # 1
        "\n"                              # 2
        "def f():\n"                      # 3
        "    return Path('.')\n"          # 4
    )
    root = _mkproject(tmp_path, {"i.py": src})
    trace = trace_origin(root, file="i.py", line=4, var="Path")
    assert trace is not None
    assert trace.origin_kind == "import"
    assert "pathlib" in trace.origin_expr


def test_trace_origin_import_aliased(tmp_path: Path) -> None:
    src = (
        "import json as J\n"   # 1
        "\n"                    # 2
        "def f():\n"            # 3
        "    return J.dumps({})\n"  # 4
    )
    root = _mkproject(tmp_path, {"a.py": src})
    trace = trace_origin(root, file="a.py", line=4, var="J")
    assert trace is not None
    assert trace.origin_kind == "import"


def test_trace_origin_returns_none_for_unbound(tmp_path: Path) -> None:
    src = (
        "def f():\n"
        "    return whatever_undefined\n"
    )
    root = _mkproject(tmp_path, {"u.py": src})
    assert trace_origin(root, file="u.py", line=2, var="whatever_undefined") is None


def test_trace_origin_one_hop_only(tmp_path: Path) -> None:
    """trace_origin reports the IMMEDIATE binding, not the transitive source."""
    src = (
        "def get_user():\n"            # 1
        "    raw = fetch()\n"          # 2
        "    user = parse(raw)\n"      # 3
        "    cleaned = strip(user)\n"  # 4
        "    return cleaned\n"         # 5
    )
    root = _mkproject(tmp_path, {"u.py": src})
    trace = trace_origin(root, file="u.py", line=5, var="cleaned")
    assert trace is not None
    # We report 'strip(user)' — one hop. We do NOT chase back to user → parse → raw → fetch.
    assert trace.origin_expr == "strip(user)"
    assert trace.origin_line == 4


def test_trace_origin_nested_function_picks_innermost(tmp_path: Path) -> None:
    src = (
        "def outer():\n"             # 1
        "    x = 1\n"                 # 2
        "    def inner():\n"          # 3
        "        x = 2\n"             # 4
        "        return x\n"          # 5
        "    return inner()\n"        # 6
    )
    root = _mkproject(tmp_path, {"n.py": src})
    # Line 5 is inside inner — should report the line-4 binding
    trace = trace_origin(root, file="n.py", line=5, var="x")
    assert trace is not None
    assert trace.origin_line == 4
    assert trace.origin_expr == "2"


def test_trace_origin_skip_post_line_bindings(tmp_path: Path) -> None:
    """A later assignment after ``line`` must not be returned."""
    src = (
        "def f():\n"           # 1
        "    x = 1\n"           # 2
        "    y = x\n"           # 3 — use site for x
        "    x = 99\n"          # 4 — would shadow but is BELOW use line
        "    return y\n"        # 5
    )
    root = _mkproject(tmp_path, {"s.py": src})
    trace = trace_origin(root, file="s.py", line=3, var="x")
    assert trace is not None
    assert trace.origin_line == 2
    assert trace.origin_expr == "1"


def test_trace_origin_unpacked_tuple_target(tmp_path: Path) -> None:
    src = (
        "def f():\n"             # 1
        "    a, b = (1, 2)\n"     # 2
        "    return a + b\n"     # 3
    )
    root = _mkproject(tmp_path, {"t.py": src})
    trace = trace_origin(root, file="t.py", line=3, var="b")
    assert trace is not None
    assert trace.origin_line == 2


def test_skips_excluded_dirs(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": "def f(app):\n    app.add_middleware(Real)\n",
        ".venv/lib/site-packages/foo.py": (
            "def f(app):\n    app.add_middleware(VendorJunk)\n"
        ),
        "node_modules/pkg/mod.py": (
            "def f(app):\n    app.add_middleware(NodeJunk)\n"
        ),
    })
    entries = extract_middleware_chain(root)
    assert [e.name for e in entries] == ["Real"]
