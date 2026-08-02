"""Wiring inspection — middleware chain, decorator chain, static constants.

The detector and investigator both miss things a single chunk read can't
see: that ``AuthMiddleware`` is always installed before route handlers,
that a function carries ``@require_auth``, that a feature flag is set to
``True`` in production. Three pure-AST lookups close those gaps:

* ``extract_middleware_chain(root)`` — every ``app.add_middleware(...)``
  registration in source order. ASGI execution order is the reverse
  (last registered = outermost = runs first).
* ``decorators_on(root, file=..., line=...)`` — the decorator list on
  the function spanning a given line.
* ``get_constant(root, name)`` — first module-level ``name = <literal>``
  assignment. ``confident=False`` when the RHS depends on runtime/env.

Pure stdlib ``ast`` — no Joern, no third-party. Cheap enough to call on
every interesting finding. Surfaced to LLM detectors via
``agent_tools.py`` wrappers.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".tox", "node_modules",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".augmentum",
    # Skip worktrees (sibling checkouts of the same repo) so we don't
    # double-count or surface stale code from research branches.
    ".claude",
}


def _iter_py_files(root: Path) -> Iterator[Path]:
    for p in root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        yield p


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 — unparse can raise for malformed nodes
        return f"<{type(node).__name__}>"


# ---------------------------------------------------------------------------
# Middleware chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiddlewareEntry:
    """One ASGI middleware registration site.

    ``order`` is the global source-order index across the workspace
    (file path, then line). Lower order = registered earlier.

    ASGI execution semantics: middleware wraps onion-style. The LAST
    registered middleware wraps the OUTERMOST, so it runs FIRST on the
    incoming request. Use ``runs_first_to_last(entries)`` to get the
    request-flow order directly.
    """

    name: str                 # class name as written in source
    file: str                 # relative to workspace root
    line: int                 # source line of the add_middleware call
    order: int                # 0-based registration order
    kwargs_repr: dict[str, str] = field(default_factory=dict)


def extract_middleware_chain(root: Path) -> list[MiddlewareEntry]:
    """Find every ``<app>.add_middleware(<Cls>, **kwargs)`` call under ``root``.

    Returns entries sorted by (file, line) — registration order. Each
    entry's ``order`` matches its position in this list. Use
    ``runs_first_to_last`` to get the request-flow ordering.

    Doesn't resolve imports — the ``name`` is the class identifier as
    written at the call site. Aliased imports surface as the alias.
    """
    raw: list[tuple[str, str, int, dict[str, str]]] = []
    for path in _iter_py_files(root):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue
        rel = _rel(path, root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "add_middleware":
                continue
            if not node.args:
                continue
            cls_name = _safe_unparse(node.args[0])
            kwargs: dict[str, str] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                kwargs[kw.arg] = _safe_unparse(kw.value)
            raw.append((cls_name, rel, getattr(node, "lineno", 0) or 0, kwargs))
    raw.sort(key=lambda t: (t[1], t[2]))
    return [
        MiddlewareEntry(name=n, file=f, line=ln, order=i, kwargs_repr=kw)
        for i, (n, f, ln, kw) in enumerate(raw)
    ]


def runs_first_to_last(entries: list[MiddlewareEntry]) -> list[MiddlewareEntry]:
    """Return ``entries`` in request-flow order (outermost first).

    ASGI: last registered wraps the others, so it runs first on the
    incoming request. Reversing the registration list yields the order
    a request actually traverses.
    """
    return list(reversed(entries))


# ---------------------------------------------------------------------------
# Decorator chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoratorInfo:
    """One decorator applied to a function definition."""

    name: str                 # dotted name as written, e.g. "router.get" or "require_auth"
    file: str
    line: int                 # decorator's own source line
    args_repr: list[str] = field(default_factory=list)
    kwargs_repr: dict[str, str] = field(default_factory=dict)


def decorators_on(
    root: Path, *, file: str, line: int,
) -> list[DecoratorInfo]:
    """Return decorators on the function whose source range contains ``line``.

    Matches any line from the first decorator down to the function's
    end. Returns the decorators in SOURCE ORDER (topmost first), which
    is the order a reader encounters them — Python applies them
    bottom-up.
    """
    target = (root / file).resolve()
    if not target.exists():
        return []
    try:
        src = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    chosen: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.decorator_list:
            start = min(getattr(d, "lineno", node.lineno) for d in node.decorator_list)
        else:
            start = node.lineno
        end = getattr(node, "end_lineno", None) or node.lineno
        if start <= line <= end:
            # Prefer the innermost-defined function (later start = nested closer to target)
            if chosen is None or start > (
                min(getattr(d, "lineno", chosen.lineno) for d in chosen.decorator_list)
                if chosen.decorator_list else chosen.lineno
            ):
                chosen = node
    if chosen is None:
        return []

    out: list[DecoratorInfo] = []
    for dec in chosen.decorator_list:
        if isinstance(dec, ast.Call):
            name = _safe_unparse(dec.func)
            args = [_safe_unparse(a) for a in dec.args]
            kwargs = {kw.arg: _safe_unparse(kw.value) for kw in dec.keywords if kw.arg}
            dline = getattr(dec, "lineno", 0) or 0
        else:
            name = _safe_unparse(dec)
            args = []
            kwargs = {}
            dline = getattr(dec, "lineno", 0) or 0
        out.append(DecoratorInfo(
            name=name, file=file, line=dline,
            args_repr=args, kwargs_repr=kwargs,
        ))
    return out


# ---------------------------------------------------------------------------
# Static constant lookup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantBinding:
    """A module-level constant assignment recovered statically."""

    name: str
    value_repr: str           # repr of the value when confident, else unparsed source
    file: str
    line: int
    confident: bool           # True when RHS is a pure literal


def get_constant(root: Path, name: str) -> ConstantBinding | None:
    """Find the first module-level ``name = <expr>`` (or ``name: T = <expr>``).

    Returns ``confident=True`` only when the RHS is a literal — bare
    constants, or containers (tuple/list/dict/set) whose every leaf is a
    constant. Any name reference, function call, or attribute access on
    the RHS yields ``confident=False`` so the caller knows the value
    depends on runtime / env / imports.

    Bare-name match across all files. If the same name binds in two
    modules, only the first encountered is returned.
    """
    for path in _iter_py_files(root):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = _rel(path, root)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == name:
                        return _binding(node.value, name, rel, node.lineno)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == name and node.value is not None:
                    return _binding(node.value, name, rel, node.lineno)
    return None


def _binding(
    value: ast.expr, name: str, file: str, line: int,
) -> ConstantBinding:
    if isinstance(value, ast.Constant):
        return ConstantBinding(
            name=name, value_repr=repr(value.value),
            file=file, line=line, confident=True,
        )
    if isinstance(value, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        if _all_literal(value):
            return ConstantBinding(
                name=name, value_repr=_safe_unparse(value),
                file=file, line=line, confident=True,
            )
    return ConstantBinding(
        name=name, value_repr=_safe_unparse(value),
        file=file, line=line, confident=False,
    )


def _all_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_all_literal(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if k is None or not _all_literal(k) or not _all_literal(v):
                return False
        return True
    return False


# ---------------------------------------------------------------------------
# One-hop origin trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OriginTrace:
    """Where the value of ``var`` used at ``file:line`` came from.

    One-hop only — we report the immediate binding (parameter,
    assignment, import). We do NOT follow through ``if/else``, across
    function calls, or through ``await`` / yield. The caller asks
    further questions if needed.

    ``confident=False`` means the trace landed on something the AST
    can't resolve statically (a subscript, a complex expression) and
    the LLM should treat ``origin_expr`` as a hint, not ground truth.
    """

    var: str
    file: str                 # use-site file (same as origin_file when local)
    line: int                 # use-site line
    origin_kind: str          # 'parameter' | 'assignment' | 'ann_assignment' | 'import' | 'for_target' | 'with_target' | 'global_assignment' | 'unknown'
    origin_file: str
    origin_line: int
    origin_expr: str          # rendered RHS, signature, or import statement
    confident: bool
    note: str = ""            # human-readable scope hint


def trace_origin(
    root: Path, *, file: str, line: int, var: str,
) -> OriginTrace | None:
    """Return where ``var`` was last bound before ``line`` in ``file``.

    Resolution order:
      1. Function parameter (innermost enclosing function)
      2. ``for`` / ``with`` target in the enclosing block
      3. Latest local assignment above ``line``
      4. Module-level assignment
      5. Import statement

    Returns ``None`` only when no static binding is found anywhere —
    that itself is a useful signal (the value is set by an external
    framework / closure / runtime injection).
    """
    target = (root / file).resolve()
    if not target.exists():
        return None
    try:
        src = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    enclosing = _enclosing_function(tree, line)
    if enclosing is not None:
        param_trace = _param_origin(enclosing, var, file, line)
        if param_trace is not None:
            return param_trace
        local = _latest_local_binding(enclosing, var, line)
        if local is not None:
            return _binding_to_origin(local, var, file, line, "local")

    module_binding = _latest_module_binding(tree, var, line)
    if module_binding is not None:
        return _binding_to_origin(module_binding, var, file, line, "module")

    imp = _find_import_for(tree, var)
    if imp is not None:
        return OriginTrace(
            var=var, file=file, line=line,
            origin_kind="import",
            origin_file=file,
            origin_line=getattr(imp, "lineno", 0) or 0,
            origin_expr=_safe_unparse(imp),
            confident=True,
            note="imported at module level",
        )
    return None


def _enclosing_function(
    tree: ast.Module, line: int,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the INNERMOST function whose body contains ``line``."""
    chosen: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None) or node.lineno
        if start <= line <= end:
            if chosen is None or start > chosen.lineno:
                chosen = node
    return chosen


def _param_origin(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    var: str, file: str, line: int,
) -> OriginTrace | None:
    args = func.args
    all_args = [
        *args.posonlyargs, *args.args, *args.kwonlyargs,
    ]
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    for arg in all_args:
        if arg.arg != var:
            continue
        annotation = ""
        if arg.annotation is not None:
            annotation = f": {_safe_unparse(arg.annotation)}"
        return OriginTrace(
            var=var, file=file, line=line,
            origin_kind="parameter",
            origin_file=file,
            origin_line=getattr(arg, "lineno", func.lineno) or func.lineno,
            origin_expr=f"{var}{annotation}",
            confident=True,
            note=f"parameter of {func.name}",
        )
    return None


def _latest_local_binding(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    var: str, line: int,
) -> ast.AST | None:
    """Latest binding of ``var`` inside ``func`` whose line is < ``line``.

    Considers Assign, AnnAssign, AugAssign, For targets, With targets,
    and named expressions (walrus). One-hop — does not recurse into
    nested function bodies.
    """
    candidates: list[ast.AST] = []
    for node in _iter_statements(func.body):
        node_line = getattr(node, "lineno", 0) or 0
        if node_line >= line:
            continue
        if _binds_name(node, var):
            candidates.append(node)
    if not candidates:
        return None
    candidates.sort(key=lambda n: getattr(n, "lineno", 0) or 0)
    return candidates[-1]


def _iter_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield every statement reachable from ``body`` without descending
    into nested function/class definitions (those are their own scopes)."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                yield from _iter_statements([child])
            elif isinstance(child, list):
                # Some nodes (If, For, While, Try) have ``body`` / ``orelse``
                # list children we want to recurse into.
                for sub in child:
                    if isinstance(sub, ast.stmt):
                        yield from _iter_statements([sub])


def _binds_name(node: ast.AST, var: str) -> bool:
    """Does ``node`` produce a binding for ``var``?"""
    if isinstance(node, ast.Assign):
        return any(_target_names_contain(t, var) for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.target.id == var:
            return True
    if isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name) and node.target.id == var:
            return True
    if isinstance(node, ast.For):
        return _target_names_contain(node.target, var)
    if isinstance(node, ast.AsyncFor):
        return _target_names_contain(node.target, var)
    if isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
        for item in node.items:
            if item.optional_vars is not None and _target_names_contain(item.optional_vars, var):
                return True
    return False


def _target_names_contain(target: ast.expr, var: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == var
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_names_contain(e, var) for e in target.elts)
    if isinstance(target, ast.Starred):
        return _target_names_contain(target.value, var)
    return False


def _latest_module_binding(
    tree: ast.Module, var: str, line: int,
) -> ast.AST | None:
    candidates: list[ast.AST] = []
    for node in tree.body:
        node_line = getattr(node, "lineno", 0) or 0
        if node_line >= line:
            continue
        if _binds_name(node, var):
            candidates.append(node)
    if not candidates:
        return None
    candidates.sort(key=lambda n: getattr(n, "lineno", 0) or 0)
    return candidates[-1]


def _find_import_for(tree: ast.Module, var: str) -> ast.AST | None:
    """Return the import statement (if any) that brings ``var`` into scope."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound == var:
                    return node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound == var:
                    return node
    return None


def _binding_to_origin(
    node: ast.AST, var: str, file: str, line: int, scope: str,
) -> OriginTrace:
    line_no = getattr(node, "lineno", 0) or 0
    if isinstance(node, ast.Assign):
        rhs = _safe_unparse(node.value)
        return OriginTrace(
            var=var, file=file, line=line,
            origin_kind="assignment",
            origin_file=file, origin_line=line_no,
            origin_expr=rhs,
            confident=isinstance(node.value, (ast.Constant, ast.Name, ast.Attribute, ast.Call)),
            note=f"latest {scope} assignment",
        )
    if isinstance(node, ast.AnnAssign):
        rhs = _safe_unparse(node.value) if node.value is not None else "<no value>"
        return OriginTrace(
            var=var, file=file, line=line,
            origin_kind="ann_assignment",
            origin_file=file, origin_line=line_no,
            origin_expr=rhs,
            confident=node.value is not None,
            note=f"latest {scope} annotated assignment",
        )
    if isinstance(node, ast.AugAssign):
        rhs = f"{var} {_op_symbol(node.op)}= {_safe_unparse(node.value)}"
        return OriginTrace(
            var=var, file=file, line=line,
            origin_kind="assignment",
            origin_file=file, origin_line=line_no,
            origin_expr=rhs,
            confident=False,
            note=f"latest {scope} augmented assignment (prior binding required)",
        )
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return OriginTrace(
            var=var, file=file, line=line,
            origin_kind="for_target",
            origin_file=file, origin_line=line_no,
            origin_expr=f"iter: {_safe_unparse(node.iter)}",
            confident=True,
            note=f"{scope} for-loop target",
        )
    if isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None and _target_names_contain(item.optional_vars, var):
                return OriginTrace(
                    var=var, file=file, line=line,
                    origin_kind="with_target",
                    origin_file=file, origin_line=line_no,
                    origin_expr=f"context: {_safe_unparse(item.context_expr)}",
                    confident=True,
                    note=f"{scope} with-statement target",
                )
    return OriginTrace(
        var=var, file=file, line=line,
        origin_kind="unknown",
        origin_file=file, origin_line=line_no,
        origin_expr=_safe_unparse(node),
        confident=False,
        note=f"{scope} binding with unrecognized shape",
    )


_OP_SYMBOLS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>",
    ast.BitOr: "|", ast.BitXor: "^", ast.BitAnd: "&", ast.FloorDiv: "//",
    ast.MatMult: "@",
}


def _op_symbol(op: ast.operator) -> str:
    return _OP_SYMBOLS.get(type(op), "?")


# ---------------------------------------------------------------------------
# Prompt-friendly renderer
# ---------------------------------------------------------------------------


def render_middleware_brief(entries: list[MiddlewareEntry]) -> str:
    """Render middleware chain as a compact detector brief.

    Returns the empty string when there are no entries so the caller
    can drop the section cleanly.
    """
    if not entries:
        return ""
    lines = ["## ASGI middleware chain", ""]
    lines.append(
        "Registered order (top = registered first). On incoming "
        "requests, the LAST registered middleware runs FIRST — request "
        "flow is the reverse of this list.",
    )
    lines.append("")
    for entry in entries:
        kwargs = ""
        if entry.kwargs_repr:
            rendered = ", ".join(f"{k}={v}" for k, v in entry.kwargs_repr.items())
            kwargs = f"  ({rendered})"
        lines.append(f"  {entry.order}. {entry.name}{kwargs}  [{entry.file}:{entry.line}]")
    lines.append("")
    first_to_last = runs_first_to_last(entries)
    lines.append("Request flow (first to last): " + " -> ".join(e.name for e in first_to_last))
    return "\n".join(lines)
