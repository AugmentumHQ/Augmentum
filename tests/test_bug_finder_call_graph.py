"""Call-graph builder tests.

Builds graphs from synthetic on-disk Python projects (real ast parses)
to verify the caller/callee queries work as advertised. Also checks
JSON round-trip + the prompt-renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augmentum.bug_finder.call_graph import (
    CallGraph,
    build_from_directory,
    load_graph,
    render_call_graph_facts,
    save_graph,
)


# ---------------------------------------------------------------------------
# Helpers — write synthetic projects to a tmp dir
# ---------------------------------------------------------------------------


def _mkproject(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create `files` (relative path → source) under `tmp_path`."""
    for rel, src in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_caller_callee(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": (
            "def handler(req):\n"
            "    return run_query(req.name)\n"
            "\n"
            "def run_query(name):\n"
            "    pass\n"
        ),
    })
    g = build_from_directory(root)
    assert "app.handler" in g.nodes
    assert "app.run_query" in g.nodes
    assert g.callees_of("app.handler") == ["run_query"]
    assert "app.handler" in g.callers_of("run_query")


def test_class_methods_are_qualified(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "store.py": (
            "class NoteStore:\n"
            "    def get(self, note_id):\n"
            "        return self.db.execute('SELECT * FROM notes')\n"
            "    def all(self):\n"
            "        return [self.get(i) for i in range(10)]\n"
        ),
    })
    g = build_from_directory(root)
    assert "store.NoteStore.get" in g.nodes
    assert "store.NoteStore.all" in g.nodes
    # all() calls self.get → bare name "get" is recorded
    callees = g.callees_of("store.NoteStore.all")
    assert "self.get" in callees or "get" in callees


def test_async_functions_captured(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "worker.py": (
            "async def main():\n"
            "    await do_work()\n"
            "\n"
            "async def do_work():\n"
            "    pass\n"
        ),
    })
    g = build_from_directory(root)
    assert "worker.main" in g.nodes
    assert "worker.do_work" in g.nodes
    assert "do_work" in g.callees_of("worker.main")


def test_dangerous_sinks_traced_to_callers(tmp_path: Path) -> None:
    """The bug-finder's bread and butter use case: 'who calls eval/exec?'"""
    root = _mkproject(tmp_path, {
        "evil.py": (
            "def run_user_expr(expr):\n"
            "    return eval(expr)\n"
            "\n"
            "def safe_compute(a, b):\n"
            "    return a + b\n"
        ),
        "api.py": (
            "from evil import run_user_expr\n"
            "\n"
            "def handler(body):\n"
            "    return run_user_expr(body)\n"
        ),
    })
    g = build_from_directory(root)
    # eval is bare-matched; only run_user_expr calls it
    assert "evil.run_user_expr" in g.callers_of("eval")
    # who calls run_user_expr? Only api.handler
    assert "api.handler" in g.callers_of("run_user_expr")
    # safe_compute is not connected to either
    assert "evil.safe_compute" not in g.callers_of("eval")


def test_module_level_calls_use_module_as_caller(tmp_path: Path) -> None:
    """Top-level `eval(...)` calls — caller is the module itself."""
    root = _mkproject(tmp_path, {
        "init.py": "value = eval('1 + 1')\n",
    })
    g = build_from_directory(root)
    # The module-level caller is the module name (no function scope)
    assert "init" in g.callers_of("eval")


def test_skip_dirs_excluded(tmp_path: Path) -> None:
    """`.venv`, `__pycache__`, `node_modules` etc. must not be walked."""
    root = _mkproject(tmp_path, {
        "app.py": "def real(): pass\n",
        ".venv/lib/site-packages/junk.py": "def in_venv_should_not_count(): eval('x')\n",
        "node_modules/dep/index.py": "def in_node_modules(): eval('x')\n",
        "__pycache__/cached.py": "def cached(): eval('x')\n",
    })
    g = build_from_directory(root)
    assert "app.real" in g.nodes
    assert not any("in_venv" in n for n in g.nodes)
    assert not any("in_node_modules" in n for n in g.nodes)
    assert not any("cached" in n for n in g.nodes)


def test_syntax_error_files_skipped(tmp_path: Path) -> None:
    """Test fixtures often have intentional syntax errors. Skip, don't fail."""
    root = _mkproject(tmp_path, {
        "good.py": "def f(): return 1\n",
        "bad.py": "def oops( missing colon\n  pass\n",
    })
    g = build_from_directory(root)
    assert "good.f" in g.nodes
    # bad.py was skipped — no `bad.oops` node
    assert not any("oops" in n for n in g.nodes)


def test_call_sites_record_line_numbers(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "x.py": (
            "def callee(): pass\n"
            "\n"
            "def caller():\n"
            "    # line 4\n"
            "    callee()  # this is line 5\n"
        ),
    })
    g = build_from_directory(root)
    sites = g.call_sites_for_target("callee")
    assert any(s.line == 5 and s.file == "x.py" for s in sites)


def test_json_round_trip(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "app.py": (
            "def handler(): run()\n"
            "def run(): pass\n"
        ),
    })
    g1 = build_from_directory(root)
    saved = tmp_path / "_graph.json"
    save_graph(g1, saved)
    # JSON is valid
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert "nodes" in payload
    g2 = load_graph(saved)
    assert g2.nodes == g1.nodes
    assert g2.callers_of("run") == g1.callers_of("run")
    assert len(g2.call_sites) == len(g1.call_sites)


def test_max_files_caps_walk(tmp_path: Path) -> None:
    """A misconfigured root shouldn't walk forever."""
    files = {f"file_{i}.py": "def f(): pass\n" for i in range(30)}
    root = _mkproject(tmp_path, files)
    g = build_from_directory(root, max_files=10)
    # Only 10 files scanned. Each scanned file contributes 2 nodes
    # (the module + the `f` function inside it), so ≤ 20 file_-prefixed
    # nodes total.
    file_nodes = [n for n in g.nodes if n.startswith("file_")]
    assert len(file_nodes) <= 20
    function_nodes = [n for n in file_nodes if n.endswith(".f")]
    assert len(function_nodes) <= 10


def test_render_facts_groups_by_target(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "auth.py": (
            "def login_handler(body):\n"
            "    return eval(body['expr'])\n"
        ),
        "admin.py": (
            "def admin_eval(body):\n"
            "    eval(body)\n"
        ),
        "safe.py": "def harmless(): pass\n",
    })
    g = build_from_directory(root)
    out = render_call_graph_facts(g, target_names=["eval"])
    assert "eval" in out
    assert "auth.login_handler" in out
    assert "admin.admin_eval" in out
    # safe.py shouldn't show up under `eval`
    assert "safe.harmless" not in out
    # The brief should include file:line info
    assert "auth.py:" in out


def test_render_facts_empty_when_no_hits(tmp_path: Path) -> None:
    root = _mkproject(tmp_path, {
        "safe.py": "def harmless(): pass\n",
    })
    g = build_from_directory(root)
    out = render_call_graph_facts(g, target_names=["pickle.loads", "eval"])
    assert out == ""


def test_render_facts_truncates_long_lists(tmp_path: Path) -> None:
    files = {
        f"caller_{i}.py": (
            f"def f_{i}():\n"
            "    eval('x')\n"
        )
        for i in range(15)
    }
    root = _mkproject(tmp_path, files)
    g = build_from_directory(root)
    out = render_call_graph_facts(g, target_names=["eval"], max_callers_per_target=5)
    assert "and 10 more" in out


def test_callers_of_handles_qualified_target_names(tmp_path: Path) -> None:
    """`pickle.loads` and bare `loads` should both find the same callers
    (bare-name matching, by design — alias resolution is out of scope)."""
    root = _mkproject(tmp_path, {
        "a.py": (
            "import pickle\n"
            "def f(data):\n"
            "    return pickle.loads(data)\n"
        ),
    })
    g = build_from_directory(root)
    via_qualified = g.callers_of("pickle.loads")
    via_bare = g.callers_of("loads")
    assert via_qualified == via_bare
    assert "a.f" in via_qualified


# ---------------------------------------------------------------------------
# Dynamic-dispatch detection (item #4 — AppSecSanta Type-B coverage)
# ---------------------------------------------------------------------------


def test_dynamic_dispatch_detects_include_router(tmp_path: Path) -> None:
    """FastAPI / Starlette ``include_router`` composition is the
    canonical "follow the route" blind spot. Capture it explicitly."""
    root = _mkproject(tmp_path, {
        "server.py": (
            "from fastapi import FastAPI\n"
            "from .auth import auth_router\n"
            "from .media import media_router\n"
            "\n"
            "def create_app():\n"
            "    app = FastAPI()\n"
            "    app.include_router(auth_router)\n"
            "    app.include_router(media_router, prefix='/media')\n"
            "    return app\n"
        ),
    })
    g = build_from_directory(root)
    kinds = {d.kind for d in g.dynamic_dispatch_sites}
    assert "include_router" in kinds
    # Both router compositions captured
    include_sites = [
        d for d in g.dynamic_dispatch_sites
        if d.kind == "include_router"
    ]
    assert len(include_sites) == 2
    # Args carry through so the LLM can tell which router
    args_seen = {d.args_repr[0] for d in include_sites if d.args_repr}
    assert args_seen == {"auth_router", "media_router"}


def test_dynamic_dispatch_detects_register_blueprint(tmp_path: Path) -> None:
    """Flask Blueprint composition uses ``register_blueprint`` instead
    of ``include_router``. Same semantic — must be detected."""
    root = _mkproject(tmp_path, {
        "app.py": (
            "from flask import Flask\n"
            "from .auth import bp_auth\n"
            "\n"
            "def create_app():\n"
            "    app = Flask(__name__)\n"
            "    app.register_blueprint(bp_auth, url_prefix='/auth')\n"
            "    return app\n"
        ),
    })
    g = build_from_directory(root)
    kinds = {d.kind for d in g.dynamic_dispatch_sites}
    assert "include_router" in kinds  # normalized to the same bucket


def test_dynamic_dispatch_detects_add_route(tmp_path: Path) -> None:
    """``add_route`` / ``add_url_rule`` / ``add_api_route`` — direct
    route registration. The bare name maps them all to the ``add_route``
    bucket."""
    root = _mkproject(tmp_path, {
        "wire.py": (
            "def setup(app):\n"
            "    app.add_route('/legacy', legacy_handler)\n"
            "    app.add_api_route('/v2', v2_handler, methods=['GET'])\n"
        ),
    })
    g = build_from_directory(root)
    add_route_sites = [
        d for d in g.dynamic_dispatch_sites if d.kind == "add_route"
    ]
    assert len(add_route_sites) == 2


def test_dynamic_dispatch_detects_importlib(tmp_path: Path) -> None:
    """``importlib.import_module(name_var)`` is the canonical dynamic
    import. Captured so the LLM doesn't conclude an arbitrarily-named
    module is unreachable from static analysis."""
    root = _mkproject(tmp_path, {
        "loader.py": (
            "import importlib\n"
            "def load(name):\n"
            "    return importlib.import_module(name)\n"
        ),
    })
    g = build_from_directory(root)
    kinds = {d.kind for d in g.dynamic_dispatch_sites}
    assert "importlib" in kinds


def test_dynamic_dispatch_importlib_requires_qualifier(tmp_path: Path) -> None:
    """A bare ``import_module(...)`` call NOT qualified by importlib
    must not be misclassified — it could be any method named
    ``import_module`` on some unrelated object. Be conservative."""
    root = _mkproject(tmp_path, {
        "x.py": (
            "def f(loader, name):\n"
            "    # 'loader.import_module' — not importlib\n"
            "    return loader.import_module(name)\n"
        ),
    })
    g = build_from_directory(root)
    importlib_sites = [
        d for d in g.dynamic_dispatch_sites if d.kind == "importlib"
    ]
    # Should NOT misclassify — loader.import_module isn't importlib
    assert importlib_sites == []


def test_dynamic_dispatch_in_file_query(tmp_path: Path) -> None:
    """The query helper filters by file substring — used to ask
    'what dynamic dispatch happens in this file?' before concluding
    a chunk is fully understood."""
    root = _mkproject(tmp_path, {
        "a/server.py": (
            "def setup(app):\n"
            "    app.include_router(r)\n"
        ),
        "b/loader.py": (
            "import importlib\n"
            "def go(n):\n"
            "    return importlib.import_module(n)\n"
        ),
    })
    g = build_from_directory(root)
    a_sites = g.dynamic_dispatch_in_file("server.py")
    b_sites = g.dynamic_dispatch_in_file("loader.py")
    assert len(a_sites) == 1
    assert a_sites[0].kind == "include_router"
    assert len(b_sites) == 1
    assert b_sites[0].kind == "importlib"
    # Empty filter returns all
    assert len(g.dynamic_dispatch_in_file()) == 2


def test_dynamic_dispatch_survives_json_round_trip(tmp_path: Path) -> None:
    """Persistence contract — when the call graph is saved to disk
    and reloaded, dispatch sites must survive intact. Otherwise the
    cached substrate loses the dynamic-dispatch metadata between runs."""
    root = _mkproject(tmp_path, {
        "s.py": (
            "import importlib\n"
            "def go(name):\n"
            "    return importlib.import_module(name)\n"
        ),
    })
    g = build_from_directory(root)
    save_path = tmp_path / "graph.json"
    save_graph(g, save_path)
    g2 = load_graph(save_path)
    assert len(g2.dynamic_dispatch_sites) == len(g.dynamic_dispatch_sites)
    orig = g.dynamic_dispatch_sites[0]
    rt = g2.dynamic_dispatch_sites[0]
    assert (rt.kind, rt.file, rt.line, rt.raw_call) == (
        orig.kind, orig.file, orig.line, orig.raw_call,
    )
