"""Call-graph pre-computation for bug_finder.

A workspace-wide function-level call graph built up-front and surfaced
to detectors as `query_callers(name)` / `query_callees(name)`. The
problem this fixes: the current detector can only see one chunk at a
time, so any cross-procedural reasoning ("this unsafe sink is reached
from a public route") requires N file reads per finding. With a pre-
computed graph, the question is one lookup.

Python-only for Phase 1. We use Python's stdlib `ast` module — no
third-party dependency, no container build changes. Multi-language
support via tree-sitter is a follow-up; CodeMind already loads the
parser on the frontend, but server-side multi-language analysis is a
separate piece of work.

The graph is workspace-local — built per-run, stored at
`<workspace>/.augmentum/call_graph.json`. Re-built every run rather
than incrementally maintained because (a) bug-finder is set-and-
forget — runtime budget dominates and the build is fast, and (b)
incremental graphs require change detection that adds more code than
it saves.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallSite:
    """One observed call to `target` from `caller`."""

    caller: str          # qualified name: "module.Class.method"
    target: str          # bare name as seen in source: "foo" or "obj.foo"
    file: str            # relative to workspace root
    line: int            # 1-based source line


@dataclass(frozen=True)
class DynamicDispatch:
    """One site where the static call graph cannot tell you what
    actually gets dispatched.

    Captures the three patterns LLM bug-finders systematically miss
    when they reason "follow the imports / follow the route":

    * ``include_router`` — FastAPI / Starlette / Flask Blueprint
      composition. The graph sees the call; it does NOT know which
      module-defined router is being mounted.
    * ``add_route`` / ``add_url_rule`` / ``add_route_handler`` —
      same shape via a different API.
    * ``importlib.import_module(...)`` — dynamic import. The graph
      can't resolve the target module from a string at static time.

    This is the *AppSecSanta Type-B* failure class — covered as far
    as the static AST can go; the rest needs runtime introspection.
    """

    kind: str            # "include_router" | "add_route" | "importlib"
    file: str
    line: int
    caller: str          # qualified caller name
    raw_call: str        # the dotted call as seen in source
    args_repr: tuple[str, ...] = ()


@dataclass
class CallGraph:
    """Function-level call graph for one workspace.

    `nodes` = the set of definitions found. Keys are qualified names
    like ``module.Class.method`` (path-derived).

    `edges` (forward) = caller → set of bare target names.
    `reverse` = bare target name → set of qualified callers.

    The asymmetry is deliberate: callers are qualified (we know
    *where* the call comes from); targets are bare names (we don't
    resolve through imports/aliases). This is enough for the
    "anything calls this risky function" query and avoids the
    intractable parts of static analysis. False positives (a name
    matches but isn't actually the same function) are surfaced for
    the model to weigh, not asserted as truth.
    """

    nodes: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=dict)
    reverse: dict[str, set[str]] = field(default_factory=dict)
    call_sites: list[CallSite] = field(default_factory=list)
    dynamic_dispatch_sites: list[DynamicDispatch] = field(default_factory=list)
    workspace_root: str = ""

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(v) for v in self.edges.values())

    def callers_of(self, target: str) -> list[str]:
        """Qualified names that reference `target` as a call.

        `target` is matched against the bare name — if you ask for
        `eval`, you get every place `eval(...)` appears regardless of
        which `eval` it is. The caller weighs which match is real.
        """
        bare = target.rsplit(".", 1)[-1]
        return sorted(self.reverse.get(bare, set()))

    def callees_of(self, qualified_caller: str) -> list[str]:
        """Bare names called from `qualified_caller`."""
        return sorted(self.edges.get(qualified_caller, set()))

    def call_sites_for_target(self, target: str) -> list[CallSite]:
        """Every observed call to `target`. Useful for "show me with line numbers"."""
        bare = target.rsplit(".", 1)[-1]
        return [c for c in self.call_sites if c.target.rsplit(".", 1)[-1] == bare]

    def dynamic_dispatch_in_file(
        self, file_substring: str = "",
    ) -> list[DynamicDispatch]:
        """Return dynamic-dispatch sites filtered by ``file_substring``
        (empty = all). Use this BEFORE concluding that "this code
        doesn't reach X" — dynamic dispatch is the blind spot of
        any static call graph and the most common source of
        false-negative reachability claims."""
        if not file_substring:
            return list(self.dynamic_dispatch_sites)
        needle = file_substring.lower()
        return [
            d for d in self.dynamic_dispatch_sites
            if needle in d.file.lower()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": self.workspace_root,
            "nodes": sorted(self.nodes),
            "edges": {k: sorted(v) for k, v in self.edges.items()},
            "reverse": {k: sorted(v) for k, v in self.reverse.items()},
            "call_sites": [
                {"caller": c.caller, "target": c.target,
                 "file": c.file, "line": c.line}
                for c in self.call_sites
            ],
            "dynamic_dispatch_sites": [
                {
                    "kind": d.kind, "file": d.file, "line": d.line,
                    "caller": d.caller, "raw_call": d.raw_call,
                    "args_repr": list(d.args_repr),
                }
                for d in self.dynamic_dispatch_sites
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CallGraph:
        edges = {k: set(v) for k, v in (d.get("edges") or {}).items()}
        reverse = {k: set(v) for k, v in (d.get("reverse") or {}).items()}
        call_sites = [
            CallSite(
                caller=str(c.get("caller") or ""),
                target=str(c.get("target") or ""),
                file=str(c.get("file") or ""),
                line=int(c.get("line") or 0),
            )
            for c in (d.get("call_sites") or [])
        ]
        dispatch_sites = [
            DynamicDispatch(
                kind=str(x.get("kind") or ""),
                file=str(x.get("file") or ""),
                line=int(x.get("line") or 0),
                caller=str(x.get("caller") or ""),
                raw_call=str(x.get("raw_call") or ""),
                args_repr=tuple(str(a) for a in (x.get("args_repr") or ())),
            )
            for x in (d.get("dynamic_dispatch_sites") or [])
        ]
        return cls(
            nodes=set(d.get("nodes") or []),
            edges=edges,
            reverse=reverse,
            call_sites=call_sites,
            dynamic_dispatch_sites=dispatch_sites,
            workspace_root=str(d.get("workspace_root") or ""),
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _module_name_from_path(file: Path, root: Path) -> str:
    """Derive a dotted module name for `file` relative to `root`."""
    try:
        rel = file.relative_to(root).with_suffix("")
    except ValueError:
        return file.stem
    parts = [p for p in rel.parts if p and p != "__init__"]
    return ".".join(parts) if parts else file.stem


class _CallCollector(ast.NodeVisitor):
    """Walks a module's AST, building qualified names + call sites."""

    def __init__(self, module: str, file: str, graph: CallGraph) -> None:
        self._module = module
        self._file = file
        self._graph = graph
        # Stack of enclosing scopes (class + function names).
        self._scope: list[str] = []

    def _qualified(self, name: str) -> str:
        return ".".join([self._module] + self._scope + [name])

    def _push_def(self, name: str) -> None:
        qname = self._qualified(name)
        self._graph.nodes.add(qname)
        self._scope.append(name)

    def _pop_def(self) -> None:
        self._scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._push_def(node.name)
        self.generic_visit(node)
        self._pop_def()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._push_def(node.name)
        self.generic_visit(node)
        self._pop_def()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._push_def(node.name)
        self.generic_visit(node)
        self._pop_def()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        target = _call_name(node.func)
        if not target:
            self.generic_visit(node)
            return
        # We need a containing function for the qualified caller.
        if not self._scope:
            caller = self._module  # module-level call
        else:
            caller = ".".join([self._module] + self._scope)
        bare = target.rsplit(".", 1)[-1]
        self._graph.edges.setdefault(caller, set()).add(target)
        self._graph.reverse.setdefault(bare, set()).add(caller)
        line = getattr(node, "lineno", 0) or 0
        self._graph.call_sites.append(CallSite(
            caller=caller,
            target=target,
            file=self._file,
            line=line,
        ))
        # Detect dynamic-dispatch patterns the static graph can't
        # otherwise represent. The detection is intentionally based
        # on the BARE name (last segment of the dotted target) so it
        # catches both ``app.include_router(...)`` and ``self.include_router(...)``.
        dispatch_kind = _classify_dynamic_dispatch(target, bare)
        if dispatch_kind:
            self._graph.dynamic_dispatch_sites.append(DynamicDispatch(
                kind=dispatch_kind,
                file=self._file,
                line=line,
                caller=caller,
                raw_call=target,
                args_repr=tuple(
                    _summarize_arg(a) for a in (node.args or ())
                ),
            ))
        self.generic_visit(node)


def _call_name(func: ast.expr) -> str:
    """Render a call target as a dotted string. Handles Name, Attribute,
    and Subscript chains."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    if isinstance(func, ast.Subscript):
        # e.g. handlers["x"]() — give up on the subscript piece
        parent = _call_name(func.value)
        return parent
    return ""


# Bare-name patterns that indicate dynamic dispatch. The static graph
# captures these as ordinary calls; the dynamic_dispatch_sites list
# tags them so the planner/lead/detector can ask "where does control
# flow go that the call graph alone won't reveal?".
_DYNAMIC_DISPATCH_BY_BARE: dict[str, str] = {
    # FastAPI / Starlette / Flask-Blueprint composition
    "include_router": "include_router",
    "register_blueprint": "include_router",
    "mount": "include_router",
    # Direct route addition
    "add_route": "add_route",
    "add_url_rule": "add_route",
    "add_api_route": "add_route",
    "add_websocket_route": "add_route",
    # Dynamic imports
    "import_module": "importlib",
}


def _classify_dynamic_dispatch(target: str, bare: str) -> str:
    """Return the dispatch kind for a target call, or ``""``.

    ``target`` is the dotted-path form (``app.include_router``);
    ``bare`` is the last segment (``include_router``). The bare name
    drives the match because most apps use ``self.x``, ``app.x``, or
    ``router.x`` interchangeably and we want to catch all of them.
    """
    kind = _DYNAMIC_DISPATCH_BY_BARE.get(bare, "")
    if not kind:
        return ""
    # Guard: ``import_module`` is uniquely shipped by ``importlib``.
    # Require the qualifier when the bare name is too generic alone.
    if kind == "importlib" and "importlib" not in target:
        return ""
    return kind


def _summarize_arg(arg: ast.expr) -> str:
    """Best-effort one-line representation of a call argument.

    Used for ``DynamicDispatch.args_repr`` so a future reader can tell
    ``include_router(auth_router)`` apart from ``include_router(media_router)``.
    Returns ``"<expr>"`` when the node is too complex to summarize.
    """
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Attribute):
        return _call_name(arg) or "<attr>"
    if isinstance(arg, ast.Constant):
        if isinstance(arg.value, str):
            return repr(arg.value)
        return repr(arg.value)
    if isinstance(arg, ast.Call):
        return (_call_name(arg.func) or "<call>") + "(...)"
    return "<expr>"


def _iter_py_files(root: Path) -> Iterator[Path]:
    """Walk `root` for .py files, skipping common irrelevant dirs."""
    skip_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".tox", "node_modules",
        "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".augmentum",
        # Skip worktrees (sibling checkouts of the same repo) so we don't
        # double-count or surface stale code from research branches.
        ".claude",
    }
    for path in root.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        yield path


def build_from_directory(
    root: Path,
    *,
    max_files: int = 5000,
) -> CallGraph:
    """Build a call graph by parsing every .py file under `root`.

    `max_files` caps the walk so a misconfigured target directory
    can't accidentally scan an entire home folder. Each file gets
    parsed once; syntax errors are skipped with a warning (the user's
    test files often contain intentional syntax errors for testing).
    """
    if not root.is_dir():
        raise FileNotFoundError(f"call graph root not found: {root}")
    graph = CallGraph(workspace_root=str(root))
    seen = 0
    for path in _iter_py_files(root):
        if seen >= max_files:
            log.warning(
                "bug_finder_call_graph_truncated",
                root=str(root), max_files=max_files,
            )
            break
        seen += 1
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            # Skip files we can't parse. Test fixtures commonly include
            # intentionally broken syntax for parser testing.
            continue
        module = _module_name_from_path(path, root)
        rel_file = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        graph.nodes.add(module)
        _CallCollector(module, rel_file, graph).visit(tree)
    log.info(
        "bug_finder_call_graph_built",
        root=str(root), files=seen,
        nodes=graph.node_count, edges=graph.edge_count,
    )
    return graph


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_graph(graph: CallGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph.to_dict(), indent=None, separators=(",", ":")),
        encoding="utf-8",
    )


def load_graph(path: Path) -> CallGraph:
    return CallGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Prompt-friendly renderer (planner / detector context block)
# ---------------------------------------------------------------------------


def render_call_graph_facts(
    graph: CallGraph,
    *,
    target_names: list[str],
    max_callers_per_target: int = 8,
) -> str:
    """Produce a compact "who calls what" brief for the detector.

    Used as a context-injection block: when the planner identifies a
    risky function (e.g. `pickle.loads`, `eval`, `subprocess.Popen`),
    we render the call sites so the detector knows whether they're
    reachable from public entry points or buried in dead/test code.

    Returns "" when the graph has nothing for any target — caller
    passes through without modifying the system prompt."""
    if not graph.nodes or not target_names:
        return ""
    lines: list[str] = []
    lines.append("## Call-site context for risky names")
    lines.append("")
    lines.append(
        "Bare-name matches from the workspace call graph. Use as a "
        "reachability prior: a sink called from a route handler is much "
        "more interesting than one called from dead/test code only. "
        "Does NOT resolve aliases — same name in two modules collapses.",
    )
    lines.append("")
    rendered_anything = False
    for target in target_names:
        callers = graph.callers_of(target)
        if not callers:
            continue
        rendered_anything = True
        lines.append(f"### {target}")
        for caller in callers[:max_callers_per_target]:
            site = next(
                (c for c in graph.call_sites
                 if c.caller == caller and c.target.rsplit(".", 1)[-1] == target.rsplit(".", 1)[-1]),
                None,
            )
            if site:
                lines.append(f"  - {caller}  ({site.file}:{site.line})")
            else:
                lines.append(f"  - {caller}")
        if len(callers) > max_callers_per_target:
            lines.append(
                f"  - … and {len(callers) - max_callers_per_target} more",
            )
        lines.append("")
    return "\n".join(lines) if rendered_anything else ""
